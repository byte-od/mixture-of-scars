"""Unified Ecosystem `VQ4-Scar` Module (`Native PyTorch` + `Physical WebGPU WGSL 1.0`).

Replaces legacy scalar Gaussian quantization across the entire System One / MoS ecosystem
with a unified binary operator:
  1. `LUT256` (`256 x vec4<f16>` = `2,048 B` per matrix, `4 KB` in WGSL `var<workgroup>`):
     Entry `0` hard-pinned to exact `(0, 0, 0, 0)` + sparse axis spikes + `D4` root shells +
     `alpha^2`-energy-weighted GPU Lloyd-Max (`K-Means`).
  2. Per-channel activation equilibration (`d_eq_f16`, `2 * K` bytes) + orthonormal Block-64
     Rademacher-Hadamard (`FWHT-64`) applied ONLY ONCE per input vector `x` (`O(K)` cost,
     zero Hadamard operations inside the `O(N*K)` matrix-vector loop!).
  3. `2-Pass Residual Scar Healing` with `INT4`-Compressed `ASVD` Scars (`3.76x` smaller than FP16):
     - `A_scar` (`[rank, in_dim]`): Block-64 INT4 (`[-8..+7]` nibbles + FP16 block scales)
     - `B_scar` (`[out_dim, rank]`): Row-scaled INT4 (`[-8..+7]` nibbles + FP16 row scales, fused
       directly inside `mos_fused_vq4_int4scar_gemv` in WGSL 1.0!).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


WGSL_SHADER_PATH = Path(__file__).resolve().parent / "webgpu" / "mos_scache_kernels.wgsl"


def rademacher_signs_u32_exact(k: int, device: torch.device) -> torch.Tensor:
    """Bit-identical Rademacher signs `{-1.0, +1.0}` matching `vq4_rademacher_sign` in WGSL 1.0."""
    idx = torch.arange(k, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


@torch.no_grad()
def fwht_block64_orthonormal(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Orthonormal 64-point Fast Walsh-Hadamard Transform with Rademacher pre/post-signs."""
    orig_shape = x.shape
    assert orig_shape[-1] % 64 == 0
    y = x if inverse else (x * signs)
    y = y.reshape(-1, 64).clone()
    h = 1
    while h < 64:
        y = y.view(-1, 64 // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    out = (y.view(-1, 64) * 0.125).view(orig_shape)
    return (out * signs) if inverse else out


@torch.no_grad()
def pack_block64_int4(w_f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs `[rows, cols]` (`cols % 64 == 0`) into WebGPU `Block-64 INT4` (`uint8` nibbles + `FP16` scales)."""
    rows, cols = w_f32.shape
    assert cols % 64 == 0
    blks = w_f32.reshape(rows, cols // 64, 64)
    sc = torch.maximum((-blks.amin(dim=-1)) / 8.0, blks.amax(dim=-1) / 7.0).clamp(min=1e-7).to(torch.float16)
    q = torch.clamp(torch.round(blks / sc.float().unsqueeze(-1)), -8.0, 7.0).to(torch.int16)
    wq = (q.float() * sc.float().unsqueeze(-1)).reshape(rows, cols)

    u4 = (q + 8).to(torch.uint8).reshape(rows, cols // 2, 2)
    packed_u8 = (u4[:, :, 0] | (u4[:, :, 1] << 4)).contiguous()
    return packed_u8, sc.contiguous(), wq


@torch.no_grad()
def dequant_block64_int4(packed_u8: torch.Tensor, scales_f16: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch dequantization of `pack_block64_int4`."""
    rows, half_cols = packed_u8.shape
    cols = half_cols * 2
    low = (packed_u8 & 0x0F).to(torch.int16) - 8
    high = ((packed_u8 >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=-1).reshape(rows, cols // 64, 64).float()
    return (q * scales_f16.float().unsqueeze(-1)).reshape(rows, cols)


@torch.no_grad()
def pack_rowwise_int4(b_f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packs `B_scar [out_dim, rank]` (`rank % 8 == 0`) with 1 FP16 scale per row + INT4 nibbles."""
    out_dim, rank = b_f32.shape
    assert rank > 0 and rank % 8 == 0
    sc = torch.maximum((-b_f32.amin(dim=-1)) / 8.0, b_f32.amax(dim=-1) / 7.0).clamp(min=1e-7).to(torch.float16)
    q = torch.clamp(torch.round(b_f32 / sc.float().unsqueeze(-1)), -8.0, 7.0).to(torch.int16)
    bq = q.float() * sc.float().unsqueeze(-1)

    u4 = (q + 8).to(torch.uint8).reshape(out_dim, rank // 2, 2)
    packed_u8 = (u4[:, :, 0] | (u4[:, :, 1] << 4)).contiguous()
    return packed_u8, sc.contiguous(), bq


@torch.no_grad()
def dequant_rowwise_int4(packed_u8: torch.Tensor, row_scales_f16: torch.Tensor) -> torch.Tensor:
    out_dim, half_rank = packed_u8.shape
    low = (packed_u8 & 0x0F).to(torch.int16) - 8
    high = ((packed_u8 >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=-1).reshape(out_dim, half_rank * 2).float()
    return q * row_scales_f16.float().unsqueeze(-1)


@torch.no_grad()
def _seed_d4_zero_lattice(z_sub: torch.Tensor) -> torch.Tensor:
    dev = z_sub.device
    pts: List[List[float]] = [[0.0, 0.0, 0.0, 0.0]]  # Hard-pinned exact zero at index 0!
    for r in (1.20, 2.45):
        for ax in range(4):
            for sgn in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0, 0.0]
                v[ax] = sgn * r
                pts.append(v)
    for r in (0.72, 1.38, 2.12):
        for i in range(4):
            for j in range(i + 1, 4):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        v = [0.0, 0.0, 0.0, 0.0]
                        v[i] = si * r
                        v[j] = sj * r
                        pts.append(v)
    for r in (0.55, 1.12):
        for mask in range(16):
            pts.append([r if ((mask >> c) & 1) else -r for c in range(4)])

    c_fixed = torch.tensor(pts, dtype=torch.float32, device=dev)
    rem = 256 - c_fixed.shape[0]
    stride = max(1, z_sub.shape[0] // rem)
    c_data = z_sub[::stride][:rem].clone()
    if c_data.shape[0] < rem:
        c_data = torch.cat([c_data, z_sub[: rem - c_data.shape[0]].clone()], dim=0)
    return torch.cat([c_fixed, c_data], dim=0)


@torch.no_grad()
def _nearest_lut256(z_f32: torch.Tensor, lut_f32: torch.Tensor, chunk: int = 1048576) -> torch.Tensor:
    use_half = z_f32.device.type in ("mps", "cuda")
    M4 = z_f32.shape[0]
    out_idx = torch.empty(M4, dtype=torch.int64, device=z_f32.device)
    if use_half:
        c_h = lut_f32.to(torch.float16)
        c_n2 = lut_f32.square().sum(dim=1).unsqueeze(0).to(torch.float16)
        for s in range(0, M4, chunk):
            e = min(s + chunk, M4)
            dots = z_f32[s:e].to(torch.float16) @ c_h.T
            out_idx[s:e] = (c_n2 - 2.0 * dots).argmin(dim=1)
    else:
        c_n2 = lut_f32.square().sum(dim=1).unsqueeze(0)
        for s in range(0, M4, chunk):
            e = min(s + chunk, M4)
            dots = z_f32[s:e] @ lut_f32.T
            out_idx[s:e] = (c_n2 - 2.0 * dots).argmin(dim=1)
    return out_idx


@torch.no_grad()
def _extract_compressed_int4_asvd(
    w_target_f32: torch.Tensor,
    col_rms: torch.Tensor,
    rank: int,
    alpha: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts activation-weighted rank-`r` scar and packs `(A_scar, B_scar)` in INT4."""
    N, K = w_target_f32.shape
    assert rank > 0 and rank % 8 == 0
    dev = w_target_f32.device
    s = (col_rms / col_rms.median().clamp(min=1e-7)).pow(alpha).clamp(0.05, 64.0)
    w_act = w_target_f32 * s.unsqueeze(0)

    q = min(rank + 12, min(N, K))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261116 + N + K + rank)
    omega = torch.randn(K, q, generator=gen, dtype=torch.float32).to(dev)

    y = w_act @ omega
    q_mat = torch.linalg.qr(y.cpu(), mode="reduced").Q.to(dev)
    for _ in range(2):
        z = w_act.T @ q_mat
        qz = torch.linalg.qr(z.cpu(), mode="reduced").Q.to(dev)
        y = w_act @ qz
        q_mat = torch.linalg.qr(y.cpu(), mode="reduced").Q.to(dev)

    b_small = (q_mat.T @ w_act).cpu()
    u_b, s_val, vh = torch.linalg.svd(b_small, full_matrices=False)
    u_r = q_mat.to(dev) @ u_b[:, :rank].to(dev)
    sv_r = s_val[:rank].sqrt().to(dev)
    vh_r = vh[:rank, :].to(dev)

    b_raw = (u_r * sv_r.unsqueeze(0)).contiguous()  # [out_dim, rank]
    a_raw = ((vh_r * sv_r.unsqueeze(1)) / s.unsqueeze(0)).contiguous()  # [rank, in_dim]

    a_packed, a_scales, a_q = pack_block64_int4(a_raw)
    b_packed, b_scales, b_q = pack_rowwise_int4(b_raw)
    scar_hat = b_q @ a_q
    return a_packed, a_scales, b_packed, b_scales, scar_hat


@dataclass
class VQ4ScarPackedMatrix:
    """Bit-exact unified container for BOTH Native PyTorch (`CPU/MPS/CUDA`) and WebGPU (`WGSL 1.0`)."""

    out_dim: int
    in_dim: int
    rank: int
    vq_codes_u8: torch.Tensor              # [out_dim, in_dim // 4] uint8 (1 byte = 4 weights)
    vq_scales_f16: torch.Tensor            # [out_dim, in_dim // 64] float16
    lut256_f16: torch.Tensor               # [256, 4] float16 (lut256_f16[0] == [0,0,0,0])
    d_eq_f16: torch.Tensor                 # [in_dim] float16
    a_scar_packed_u8: Optional[torch.Tensor] = None  # [rank, in_dim // 2] uint8
    a_scar_scales_f16: Optional[torch.Tensor] = None # [rank, in_dim // 64] float16
    b_scar_packed_u8: Optional[torch.Tensor] = None  # [out_dim, rank // 2] uint8
    b_scar_scales_f16: Optional[torch.Tensor] = None # [out_dim] float16
    bias_f16: Optional[torch.Tensor] = None          # [out_dim] float16

    @property
    def physical_bytes(self) -> int:
        total = (
            self.vq_codes_u8.numel()
            + self.vq_scales_f16.numel() * 2
            + self.lut256_f16.numel() * 2
            + self.d_eq_f16.numel() * 2
        )
        if self.rank > 0 and self.a_scar_packed_u8 is not None and self.b_scar_packed_u8 is not None:
            assert self.a_scar_scales_f16 is not None and self.b_scar_scales_f16 is not None
            total += (
                self.a_scar_packed_u8.numel()
                + self.a_scar_scales_f16.numel() * 2
                + self.b_scar_packed_u8.numel()
                + self.b_scar_scales_f16.numel() * 2
            )
        if self.bias_f16 is not None:
            total += self.bias_f16.numel() * 2
        return int(total)

    def to_device(self, device: torch.device) -> "VQ4ScarPackedMatrix":
        return VQ4ScarPackedMatrix(
            out_dim=self.out_dim,
            in_dim=self.in_dim,
            rank=self.rank,
            vq_codes_u8=self.vq_codes_u8.to(device),
            vq_scales_f16=self.vq_scales_f16.to(device),
            lut256_f16=self.lut256_f16.to(device),
            d_eq_f16=self.d_eq_f16.to(device),
            a_scar_packed_u8=self.a_scar_packed_u8.to(device) if self.a_scar_packed_u8 is not None else None,
            a_scar_scales_f16=self.a_scar_scales_f16.to(device) if self.a_scar_scales_f16 is not None else None,
            b_scar_packed_u8=self.b_scar_packed_u8.to(device) if self.b_scar_packed_u8 is not None else None,
            b_scar_scales_f16=self.b_scar_scales_f16.to(device) if self.b_scar_scales_f16 is not None else None,
            bias_f16=self.bias_f16.to(device) if self.bias_f16 is not None else None,
        )

    @torch.no_grad()
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the exact `VQ4-Scar` operator natively in PyTorch (`CPU / Metal MPS / CUDA`).

        Identical algebra to WGSL Kernels 16 & 17: transforms ONLY `x -> x_rot` via `FWHT-64`,
        then evaluates `<w_rot_vq4, x_rot> + B_scar @ (A_scar @ x) + bias`!
        """
        orig_dtype = x.dtype
        orig_prefix = x.shape[:-1]
        x2d = x.reshape(-1, self.in_dim).float()
        dev = x2d.device

        # 1. Input equilibration + orthonormal FWHT-64 (O(T * K) — zero Hadamard on weight rows!)
        signs = rademacher_signs_u32_exact(self.in_dim, dev)
        x_eq = x2d / self.d_eq_f16.to(dev).float().unsqueeze(0)
        x_rot = fwht_block64_orthonormal(x_eq, signs, inverse=False)  # [T, in_dim]

        # 2. Decode VQ4 rotated weights from LUT256 and multiply against x_rot
        lut_f32 = self.lut256_f16.to(dev).float()
        codes_idx = self.vq_codes_u8.to(dev).long()  # [out_dim, in_dim // 4]
        w_rot_blk = lut_f32[codes_idx].reshape(self.out_dim, self.in_dim // 64, 64)
        w_rot_q = (w_rot_blk * self.vq_scales_f16.to(dev).float().unsqueeze(-1)).reshape(self.out_dim, self.in_dim)
        y = x_rot @ w_rot_q.T

        # 3. Compressed INT4 ASVD Scar branch: u = x2d @ A_scar^T, y += u @ B_scar^T
        if self.rank > 0 and self.a_scar_packed_u8 is not None and self.b_scar_packed_u8 is not None:
            assert self.a_scar_scales_f16 is not None and self.b_scar_scales_f16 is not None
            a_q = dequant_block64_int4(self.a_scar_packed_u8.to(dev), self.a_scar_scales_f16.to(dev))
            b_q = dequant_rowwise_int4(self.b_scar_packed_u8.to(dev), self.b_scar_scales_f16.to(dev))
            u = x2d @ a_q.T
            y = y + (u @ b_q.T)

        if self.bias_f16 is not None:
            y = y + self.bias_f16.to(dev).float().unsqueeze(0)

        return y.reshape(*orig_prefix, self.out_dim).to(orig_dtype)

    def pack_webgpu_u32_shard(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """Packs all components into a single aligned `uint32` buffer + offset table for WGSL 1.0."""
        chunks: List[np.ndarray] = []
        offsets: Dict[str, int] = {}
        cur_u32 = 0

        def _append_bytes(name: str, raw: bytes) -> None:
            nonlocal cur_u32
            if len(raw) % 4 != 0:
                raw = raw + b"\x00" * (4 - (len(raw) % 4))
            arr = np.frombuffer(raw, dtype=np.uint32).copy()
            offsets[name] = cur_u32
            chunks.append(arr)
            cur_u32 += int(arr.size)

        _append_bytes("d_eq_u32_off", self.d_eq_f16.detach().cpu().contiguous().numpy().tobytes())
        _append_bytes("vq_codes_u32_off", self.vq_codes_u8.detach().cpu().contiguous().numpy().tobytes())
        _append_bytes("vq_scales_u32_off", self.vq_scales_f16.detach().cpu().contiguous().numpy().tobytes())
        _append_bytes("lut256_u32_off", self.lut256_f16.detach().cpu().contiguous().numpy().tobytes())

        if self.rank > 0 and self.a_scar_packed_u8 is not None and self.b_scar_packed_u8 is not None:
            assert self.a_scar_scales_f16 is not None and self.b_scar_scales_f16 is not None
            _append_bytes("ascar_packed_u32_off", self.a_scar_packed_u8.detach().cpu().contiguous().numpy().tobytes())
            _append_bytes("ascar_scales_u32_off", self.a_scar_scales_f16.detach().cpu().contiguous().numpy().tobytes())
            # `bscar_u32_off` concatenates packed INT4 [out_dim, rank/8] + aligned FP16 row scales [out_dim]
            b_pack_raw = self.b_scar_packed_u8.detach().cpu().contiguous().numpy().tobytes()
            b_sc_raw = self.b_scar_scales_f16.detach().cpu().contiguous().numpy().tobytes()
            _append_bytes("bscar_u32_off", b_pack_raw + b_sc_raw)
        else:
            offsets["ascar_packed_u32_off"] = 0
            offsets["ascar_scales_u32_off"] = 0
            offsets["bscar_u32_off"] = 0

        if self.bias_f16 is not None:
            _append_bytes("bias_u32_off", self.bias_f16.detach().cpu().contiguous().numpy().tobytes())
            offsets["has_bias"] = 1
        else:
            offsets["bias_u32_off"] = 0
            offsets["has_bias"] = 0

        return np.concatenate(chunks, axis=0), offsets


@torch.no_grad()
def _fit_vq4_pass(
    w_res_f32: torch.Tensor,
    d_f32: torch.Tensor,
    signs: torch.Tensor,
    lut_init: Optional[torch.Tensor] = None,
    kmeans_iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fits/applies `VQ4 LUT256` (`2.25 bpw`) and returns `(codes_u8, scales_f16, lut256_f16, wq_res_f32)`."""
    N, K = w_res_f32.shape
    dev = w_res_f32.device
    w_eq = w_res_f32 * d_f32.unsqueeze(0)
    w_rot = fwht_block64_orthonormal(w_eq, signs, inverse=False)
    w_blk = w_rot.reshape(-1, 64)
    n_blocks = w_blk.shape[0]

    alpha = w_blk.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-7, max=65000.0).to(torch.float16).float()
    z_all = (w_blk / alpha).reshape(-1, 4)

    if lut_init is None:
        step = max(1, z_all.shape[0] // 65536)
        z_sub = z_all[::step][:65536].contiguous()
        alpha_4 = alpha.repeat_interleave(16, dim=0).squeeze(1)
        w_sub = alpha_4[::step][: z_sub.shape[0]].square().clamp(min=1e-12)
        w_sub = (w_sub / w_sub.mean().clamp(min=1e-12)).contiguous()

        C = _seed_d4_zero_lattice(z_sub)
        wz_sub = z_sub * w_sub.unsqueeze(1)
        for _ in range(kmeans_iters):
            idx = _nearest_lut256(z_sub, C)
            sums = torch.zeros_like(C)
            cnts = torch.zeros(256, dtype=torch.float32, device=dev)
            sums.index_add_(0, idx, wz_sub)
            cnts.index_add_(0, idx, w_sub)
            mask = cnts > 1e-4
            mask[0] = False
            C[mask] = sums[mask] / cnts[mask].unsqueeze(1)
            C[0].zero_()
        lut_f16 = C.to(torch.float16)
        lut_f16[0].zero_()
        C = lut_f16.float()
    else:
        lut_f16 = lut_init.to(torch.float16)
        lut_f16[0].zero_()
        C = lut_f16.float()

    idx_all = _nearest_lut256(z_all, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha_f16 = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
    )
    z_all2 = (w_blk / alpha_f16.float()).reshape(-1, 4)
    idx_all = _nearest_lut256(z_all2, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha_f16 = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
    )

    codes_u8 = idx_all.to(torch.uint8).reshape(N, K // 4).contiguous()
    scales_f16 = alpha_f16.reshape(N, K // 64).contiguous()
    wq_rot = (c_blk * alpha_f16.float()).reshape(N, K)
    wq_res = fwht_block64_orthonormal(wq_rot, signs, inverse=True) / d_f32.unsqueeze(0)
    return codes_u8, scales_f16, lut_f16.contiguous(), wq_res


@torch.no_grad()
def quantize_matrix_vq4_scar_2pass(
    weight: torch.Tensor,
    col_rms: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rank: int = 64,
    eq_beta: float = 0.65,
    two_pass: bool = True,
    kmeans_iters: int = 8,
) -> VQ4ScarPackedMatrix:
    """Quantizes `[out_dim, in_dim]` into a unified `VQ4ScarPackedMatrix` (Native + WebGPU ready)."""
    w_f32 = weight.detach().float()
    N, K = w_f32.shape
    assert K % 64 == 0, f"in_dim={K} must be divisible by 64"
    dev = w_f32.device

    if col_rms is None:
        c_rms = w_f32.square().mean(dim=0).sqrt().clamp(min=1e-6)
        d_f16 = torch.ones(K, dtype=torch.float16, device=dev)
    else:
        c_rms = col_rms.detach().to(dev, dtype=torch.float32).clamp(min=1e-6)
        d_f16 = (c_rms / c_rms.median().clamp(min=1e-7)).pow(eq_beta).clamp(0.125, 32.0).to(torch.float16)
    d_f32 = d_f16.float()
    signs = rademacher_signs_u32_exact(K, dev)
    bias_f16 = bias.detach().to(dev, dtype=torch.float16).contiguous() if bias is not None else None

    if rank <= 0:
        codes_u8, scales_f16, lut_f16, _ = _fit_vq4_pass(
            w_f32, d_f32, signs, lut_init=None, kmeans_iters=kmeans_iters
        )
        return VQ4ScarPackedMatrix(
            out_dim=N,
            in_dim=K,
            rank=0,
            vq_codes_u8=codes_u8,
            vq_scales_f16=scales_f16,
            lut256_f16=lut_f16,
            d_eq_f16=d_f16.contiguous(),
            bias_f16=bias_f16,
        )

    assert rank % 8 == 0, f"rank={rank} must be divisible by 8"
    a_p1, a_s1, b_p1, b_s1, scar_1 = _extract_compressed_int4_asvd(w_f32, c_rms, rank=rank)
    codes_1, scales_1, lut_f16, wq_vq1 = _fit_vq4_pass(
        w_f32 - scar_1, d_f32, signs, lut_init=None, kmeans_iters=kmeans_iters
    )

    if not two_pass:
        return VQ4ScarPackedMatrix(
            out_dim=N,
            in_dim=K,
            rank=rank,
            vq_codes_u8=codes_1,
            vq_scales_f16=scales_1,
            lut256_f16=lut_f16,
            d_eq_f16=d_f16.contiguous(),
            a_scar_packed_u8=a_p1,
            a_scar_scales_f16=a_s1,
            b_scar_packed_u8=b_p1,
            b_scar_scales_f16=b_s1,
            bias_f16=bias_f16,
        )

    # Pass 2: Residual Scar Healing directly on `W - wq_vq1`!
    a_p2, a_s2, b_p2, b_s2, scar_2 = _extract_compressed_int4_asvd(w_f32 - wq_vq1, c_rms, rank=rank)
    codes_2, scales_2, lut_f16, _ = _fit_vq4_pass(
        w_f32 - scar_2, d_f32, signs, lut_init=lut_f16, kmeans_iters=kmeans_iters
    )
    return VQ4ScarPackedMatrix(
        out_dim=N,
        in_dim=K,
        rank=rank,
        vq_codes_u8=codes_2,
        vq_scales_f16=scales_2,
        lut256_f16=lut_f16,
        d_eq_f16=d_f16.contiguous(),
        a_scar_packed_u8=a_p2,
        a_scar_scales_f16=a_s2,
        b_scar_packed_u8=b_p2,
        b_scar_scales_f16=b_s2,
        bias_f16=bias_f16,
    )


class VQ4ScarLinear(nn.Module):
    """Drop-in Native PyTorch `nn.Linear` replacement executing unified `VQ4-Scar`."""

    def __init__(self, packed: VQ4ScarPackedMatrix) -> None:
        super().__init__()
        self.out_features = packed.out_dim
        self.in_features = packed.in_dim
        self.rank = packed.rank
        self.register_buffer("vq_codes_u8", packed.vq_codes_u8)
        self.register_buffer("vq_scales_f16", packed.vq_scales_f16)
        self.register_buffer("lut256_f16", packed.lut256_f16)
        self.register_buffer("d_eq_f16", packed.d_eq_f16)
        self.register_buffer("a_scar_packed_u8", packed.a_scar_packed_u8)
        self.register_buffer("a_scar_scales_f16", packed.a_scar_scales_f16)
        self.register_buffer("b_scar_packed_u8", packed.b_scar_packed_u8)
        self.register_buffer("b_scar_scales_f16", packed.b_scar_scales_f16)
        self.register_buffer("bias_f16", packed.bias_f16)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        col_rms: Optional[torch.Tensor] = None,
        rank: int = 64,
        eq_beta: float = 0.65,
        two_pass: bool = True,
    ) -> "VQ4ScarLinear":
        packed = quantize_matrix_vq4_scar_2pass(
            linear.weight,
            col_rms=col_rms,
            bias=linear.bias,
            rank=rank,
            eq_beta=eq_beta,
            two_pass=two_pass,
        )
        return cls(packed)

    def as_packed(self) -> VQ4ScarPackedMatrix:
        return VQ4ScarPackedMatrix(
            out_dim=self.out_features,
            in_dim=self.in_features,
            rank=self.rank,
            vq_codes_u8=self.vq_codes_u8,
            vq_scales_f16=self.vq_scales_f16,
            lut256_f16=self.lut256_f16,
            d_eq_f16=self.d_eq_f16,
            a_scar_packed_u8=self.a_scar_packed_u8,
            a_scar_scales_f16=self.a_scar_scales_f16,
            b_scar_packed_u8=self.b_scar_packed_u8,
            b_scar_scales_f16=self.b_scar_scales_f16,
            bias_f16=self.bias_f16,
        )

    @property
    def physical_bytes(self) -> int:
        return self.as_packed().physical_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.as_packed().forward_native(x)


class WebGPUVQ4ScarExecutor:
    """Physical WebGPU (`wgpu` WGSL 1.0) executor for `VQ4ScarPackedMatrix` parity & speed tests."""

    def __init__(self, wgsl_path: Optional[Path] = None) -> None:
        import wgpu

        self.wgpu = wgpu
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if self.adapter is None:
            raise RuntimeError("No physical WebGPU adapter found.")
        info = self.adapter.info
        self.adapter_name = str(info.get("device") or info.get("description") or "WebGPU Adapter")
        self.device = self.adapter.request_device_sync(required_limits={})

        shader_src = (wgsl_path or WGSL_SHADER_PATH).read_text(encoding="utf-8")
        self.shader_module = self.device.create_shader_module(code=shader_src)
        self.pipelines = {
            ep: self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": self.shader_module, "entry_point": ep},
            )
            for ep in (
                "vq4_equilibrate_fwht64",
                "mos_fused_int4_bscar_gemv",
                "mos_fused_vq4_int4scar_gemv",
            )
        }

    def _params_buf(
        self,
        p0: int = 0,
        p1: int = 0,
        p2: int = 0,
        p3: int = 0,
        p4: int = 0,
        p5: int = 0,
        p6: int = 0,
        p7: int = 0,
    ) -> Any:
        raw = struct.pack("<8I8f", p0, p1, p2, p3, p4, p5, p6, p7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return self.device.create_buffer_with_data(data=raw, usage=self.wgpu.BufferUsage.UNIFORM)

    def _bind_group(self, ep: str, bindings: Dict[int, Any]) -> Any:
        layout = self.pipelines[ep].get_bind_group_layout(0)
        entries = [
            {"binding": b_idx, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
            for b_idx, buf in sorted(bindings.items())
        ]
        return self.device.create_bind_group(layout=layout, entries=entries)

    def execute_gemv(self, packed: VQ4ScarPackedMatrix, x_batch_f32: np.ndarray) -> np.ndarray:
        """Runs physical WGSL 1.0 `vq4_equilibrate_fwht64` + `A_scar` + `mos_fused_vq4_int4scar_gemv`."""
        U = self.wgpu.BufferUsage
        shard_u32, offs = packed.pack_webgpu_u32_shard()
        assert shard_u32.nbytes <= 128 * 1024 * 1024, "Single shard exceeds WebGPU 128 MiB binding limit!"

        shard_buf = self.device.create_buffer_with_data(data=shard_u32.tobytes(), usage=U.STORAGE)
        x_in_buf = self.device.create_buffer(size=packed.in_dim * 4, usage=U.STORAGE | U.COPY_DST)
        x_rot_buf = self.device.create_buffer(size=packed.in_dim * 4, usage=U.STORAGE)
        u_buf = self.device.create_buffer(size=max(4, packed.rank * 4), usage=U.STORAGE)
        dummy_f32 = self.device.create_buffer(size=4, usage=U.STORAGE)
        y_out_buf = self.device.create_buffer(size=packed.out_dim * 4, usage=U.STORAGE | U.COPY_SRC)

        u_fwht = self._params_buf(p0=packed.in_dim, p1=offs["d_eq_u32_off"])
        bg_fwht = self._bind_group(
            "vq4_equilibrate_fwht64",
            {0: u_fwht, 1: shard_buf, 3: x_in_buf, 6: x_rot_buf},
        )

        bg_ascar = None
        if packed.rank > 0:
            u_ascar = self._params_buf(
                p0=packed.rank,
                p1=packed.in_dim,
                p2=0,
                p3=offs["ascar_packed_u32_off"],
                p4=offs["ascar_scales_u32_off"],
                p5=0,
                p6=0,
                p7=0,
            )
            bg_ascar = self._bind_group(
                "mos_fused_int4_bscar_gemv",
                {0: u_ascar, 1: shard_buf, 3: x_in_buf, 4: dummy_f32, 6: u_buf},
            )

        p7_bias = ((1 << 31) | offs["bias_u32_off"]) if offs["has_bias"] else 0
        u_vq4 = self._params_buf(
            p0=packed.out_dim,
            p1=packed.in_dim,
            p2=packed.rank,
            p3=offs["vq_codes_u32_off"],
            p4=offs["vq_scales_u32_off"],
            p5=offs["lut256_u32_off"],
            p6=offs["bscar_u32_off"],
            p7=p7_bias,
        )
        bg_vq4 = self._bind_group(
            "mos_fused_vq4_int4scar_gemv",
            {0: u_vq4, 1: shard_buf, 3: x_rot_buf, 4: u_buf, 6: y_out_buf},
        )

        x2d = np.ascontiguousarray(x_batch_f32.reshape(-1, packed.in_dim), dtype=np.float32)
        outs: List[np.ndarray] = []
        for t in range(x2d.shape[0]):
            self.device.queue.write_buffer(x_in_buf, 0, x2d[t].tobytes())
            enc = self.device.create_command_encoder()

            c1 = enc.begin_compute_pass()
            c1.set_pipeline(self.pipelines["vq4_equilibrate_fwht64"])
            c1.set_bind_group(0, bg_fwht)
            c1.dispatch_workgroups(packed.in_dim >> 6, 1, 1)
            c1.end()

            if bg_ascar is not None:
                c2 = enc.begin_compute_pass()
                c2.set_pipeline(self.pipelines["mos_fused_int4_bscar_gemv"])
                c2.set_bind_group(0, bg_ascar)
                c2.dispatch_workgroups((packed.rank + 7) >> 3, 1, 1)
                c2.end()

            c3 = enc.begin_compute_pass()
            c3.set_pipeline(self.pipelines["mos_fused_vq4_int4scar_gemv"])
            c3.set_bind_group(0, bg_vq4)
            c3.dispatch_workgroups((packed.out_dim + 7) >> 3, 1, 1)
            c3.end()

            self.device.queue.submit([enc.finish()])
            raw_y = self.device.queue.read_buffer(y_out_buf)
            outs.append(np.frombuffer(raw_y, dtype=np.float32).copy())

        return np.stack(outs, axis=0).reshape(*x_batch_f32.shape[:-1], packed.out_dim)
