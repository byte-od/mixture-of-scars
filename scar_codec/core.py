"""Domain-Independent Physical & Hardware ScarCodec (`scar_codec` v5.0.0).

Extracts the five pure mathematical operators of Mixture of Scars (MoS) and SCache O(1)
into a standalone, neural-network-free signal, tensor, storage, and telemetry library:
  1. Multiplierless Orthonormal `FWHT-64` + Channel Equilibration (`d_eq_f16`).
  2. `LUT256` (`2,048 B` SRAM table) with hard-pinned exact zero `(0,0,0,0)` + `D4` lattice.
  3. `2-Pass INT4 Residual Scar Healing` + Sparse Point-Shock Capture with `O(1)` Random Block Read.
  4. `OrthogonalSubspaceProjector` (`P_perp`) for zero-collateral interference cancellation (`RFI`).
  5. `SurpriseGatedAnomalySCache` (`34,816 B` = `34.8 KB` constant-space black-box & blackout recorder).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

__version__ = "5.0.0"


def rademacher_signs_u32_exact(k: int, device: torch.device) -> torch.Tensor:
    """Bit-identical Rademacher signs `{-1.0, +1.0}` matching `vq4_rademacher_sign` in WGSL 1.0."""
    idx = torch.arange(k, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


def fwht_block64_orthonormal(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Orthonormal 64-point Fast Walsh-Hadamard Transform with Rademacher pre/post-signs (autograd-safe)."""
    orig_shape = x.shape
    assert orig_shape[-1] % 64 == 0
    y = x if inverse else (x * signs)
    y = y.reshape(-1, 64)
    h = 1
    while h < 64:
        y = y.reshape(-1, 64 // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2)
        h *= 2
    out = (y.reshape(-1, 64) * 0.125).reshape(orig_shape)
    return (out * signs) if inverse else out


@torch.no_grad()
def pack_block64_int4(w_f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    rows, half_cols = packed_u8.shape
    cols = half_cols * 2
    low = (packed_u8 & 0x0F).to(torch.int16) - 8
    high = ((packed_u8 >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=-1).reshape(rows, cols // 64, 64).float()
    return (q * scales_f16.float().unsqueeze(-1)).reshape(rows, cols)


@torch.no_grad()
def pack_rowwise_int4(b_f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    pts: List[List[float]] = [[0.0, 0.0, 0.0, 0.0]]
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
    m4 = z_f32.shape[0]
    out_idx = torch.empty(m4, dtype=torch.int64, device=z_f32.device)
    if use_half:
        c_h = lut_f32.to(torch.float16)
        c_n2 = lut_f32.square().sum(dim=1).unsqueeze(0).to(torch.float16)
        for s in range(0, m4, chunk):
            e = min(s + chunk, m4)
            dots = z_f32[s:e].to(torch.float16) @ c_h.T
            out_idx[s:e] = (c_n2 - 2.0 * dots).argmin(dim=1)
    else:
        c_n2 = lut_f32.square().sum(dim=1).unsqueeze(0)
        for s in range(0, m4, chunk):
            e = min(s + chunk, m4)
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
    n_out, k_in = w_target_f32.shape
    assert rank > 0 and rank % 8 == 0
    dev = w_target_f32.device
    s = (col_rms / col_rms.median().clamp(min=1e-7)).pow(alpha).clamp(0.05, 64.0)
    w_act = w_target_f32 * s.unsqueeze(0)

    q = min(rank + 12, min(n_out, k_in))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261116 + n_out + k_in + rank)
    omega = torch.randn(k_in, q, generator=gen, dtype=torch.float32).to(dev)

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

    b_raw = (u_r * sv_r.unsqueeze(0)).contiguous()
    a_raw = ((vh_r * sv_r.unsqueeze(1)) / s.unsqueeze(0)).contiguous()
    a_packed, a_scales, a_q = pack_block64_int4(a_raw)
    b_packed, b_scales, b_q = pack_rowwise_int4(b_raw)
    return a_packed, a_scales, b_packed, b_scales, (b_q @ a_q)


def unique_tensor_storage_bytes(tensors: List[torch.Tensor]) -> int:
    """Computes exact physical memory footprint across unique `untyped_storage().data_ptr()` allocations."""
    seen: Dict[Tuple[str, int], int] = {}
    for t in tensors:
        if t is None:
            continue
        st = t.untyped_storage()
        key = (str(t.device), int(st.data_ptr()))
        seen[key] = max(seen.get(key, 0), int(st.nbytes()))
    return sum(seen.values())


@dataclass
class PhysicalVQ4ScarPackage:
    """Self-contained compressed physical field/tensor supporting `O(1)` random block read."""

    orig_rows: int
    orig_cols: int
    padded_cols: int
    rank: int
    codes_u32: torch.Tensor          # [N, K_pad // 16] uint32 (1 u32 = 16 values = 4 x vec4)
    scales_f16: torch.Tensor         # [N, K_pad // 64] float16 block scales
    lut256_f16: torch.Tensor         # [256, 4] float16 (2,048 B SRAM codebook; lut[0] == (0,0,0,0))
    d_eq_f16: torch.Tensor           # [K_pad] float16 per-channel equilibration scales
    a_scar_u8: torch.Tensor          # [rank, K_pad // 2] uint8 Block-64 INT4
    a_scales_f16: torch.Tensor       # [rank, K_pad // 64] float16
    b_scar_u8: torch.Tensor          # [N, rank // 2] uint8 rowwise INT4
    b_row_scales_f16: torch.Tensor   # [N] float16
    spike_flat_idx_i32: torch.Tensor # [M_spikes] int32 sorted flat indices in [0, N * K_pad)
    spike_residual_f16: torch.Tensor # [M_spikes] float16 residual shocks

    def all_tensors(self) -> List[torch.Tensor]:
        return [
            self.codes_u32,
            self.scales_f16,
            self.lut256_f16,
            self.d_eq_f16,
            self.a_scar_u8,
            self.a_scales_f16,
            self.b_scar_u8,
            self.b_row_scales_f16,
            self.spike_flat_idx_i32,
            self.spike_residual_f16,
        ]

    def physical_bytes(self) -> int:
        return unique_tensor_storage_bytes(self.all_tensors())

    def compression_ratio_vs_fp32(self) -> float:
        raw_fp32_bytes = self.orig_rows * self.orig_cols * 4
        return raw_fp32_bytes / max(1, self.physical_bytes())


class PhysicalVQ4ScarCodec:
    """Domain-independent `VQ4-Scar` + `2-Pass INT4 Scar` + `O(1)` Random-Access Block Codec."""

    def __init__(
        self,
        rank: int = 16,
        equilibration_beta: float = 0.5,
        kmeans_iters: int = 8,
        max_point_spikes: int = 512,
        spike_sigma_threshold: float = 4.5,
    ) -> None:
        if rank < 0 or (rank > 0 and rank % 8 != 0):
            raise ValueError(f"rank must be 0 or a multiple of 8, got {rank}")
        self.rank = rank
        self.equilibration_beta = equilibration_beta
        self.kmeans_iters = kmeans_iters
        self.max_point_spikes = max_point_spikes
        self.spike_sigma_threshold = spike_sigma_threshold

    @torch.no_grad()
    def encode(self, field_2d: torch.Tensor) -> PhysicalVQ4ScarPackage:
        """Compresses a 2D physical field/signal matrix `[N, K]` into a `PhysicalVQ4ScarPackage`."""
        if field_2d.ndim != 2:
            raise ValueError(f"Expected 2D tensor [N, K], got shape {tuple(field_2d.shape)}")
        dev = field_2d.device
        x_orig = field_2d.detach().float()
        n_rows, n_cols = x_orig.shape
        pad_cols = ((n_cols + 63) // 64) * 64
        if pad_cols > n_cols:
            x = torch.zeros((n_rows, pad_cols), dtype=torch.float32, device=dev)
            x[:, :n_cols] = x_orig
        else:
            x = x_orig.clone()

        # Exact vacuum row mask (`0` extra storage bytes: zero rows get `scales_f16=0` and `b_row_scales_f16=0`)
        orig_zero_row = (x.abs().amax(dim=1) <= 1e-12)

        # Step 0: Pre-FWHT robust point-shock peeling so extreme HDR / cosmic-ray glints do not blow up 64-block scales
        x_base = x.clone()
        pre_spike_idx = torch.zeros((0,), dtype=torch.int64, device=dev)
        if self.max_point_spikes > 0:
            blk_view = x_base.reshape(n_rows * (pad_cols // 64), 64)
            blk_med = blk_view.median(dim=1, keepdim=True).values
            blk_dev = (blk_view - blk_med).reshape(n_rows, pad_cols)
            if pad_cols > n_cols:
                blk_dev[:, n_cols:] = 0.0
            active_devs = blk_dev[~orig_zero_row].abs().reshape(-1)
            if active_devs.numel() > 0:
                mad_std = (1.4826 * active_devs.median()).clamp(min=1e-6)
                cand_pre = blk_dev.reshape(-1).abs() > (self.spike_sigma_threshold * mad_std)
                if bool(cand_pre.any().item()):
                    c_idx = torch.nonzero(cand_pre, as_tuple=False).squeeze(-1)
                    if c_idx.numel() > self.max_point_spikes:
                        _, top_r = torch.topk(blk_dev.reshape(-1)[c_idx].abs(), k=self.max_point_spikes)
                        c_idx = c_idx[top_r]
                    pre_spike_idx = c_idx
                    med_expanded = blk_med.expand_as(blk_view).reshape(-1)
                    x_base.reshape(-1)[pre_spike_idx] = med_expanded[pre_spike_idx]
                    x_base[orig_zero_row] = 0.0

        # Channel equilibration (`d_eq_f16`): isolates high-excursion channels before FWHT-64
        col_rms = x_base.square().mean(dim=0).sqrt().clamp(min=1e-6)
        med_rms = col_rms.median().clamp(min=1e-6)
        if self.equilibration_beta > 0.0:
            d_eq_f16 = ((col_rms / med_rms).pow(self.equilibration_beta)).clamp(0.05, 32.0).to(torch.float16)
        else:
            d_eq_f16 = torch.ones(pad_cols, dtype=torch.float16, device=dev)
        d_f32 = d_eq_f16.float()

        # Pass 1: Low-rank INT4 scar extraction (captures oblique shock fronts / multipath clusters)
        eff_rank = self.rank if (self.rank > 0 and min(n_rows, pad_cols) >= self.rank) else 0
        if eff_rank > 0:
            a_u8, a_sc, b_u8, b_sc, scar_p1 = _extract_compressed_int4_asvd(
                x_base, torch.ones_like(col_rms), rank=eff_rank, alpha=0.5
            )
            b_sc = torch.where(orig_zero_row, torch.zeros_like(b_sc), b_sc)
            scar_p1 = torch.where(orig_zero_row.unsqueeze(1), torch.zeros_like(scar_p1), scar_p1)
            res_for_vq = x_base - scar_p1
        else:
            a_u8 = torch.zeros((0, pad_cols // 2), dtype=torch.uint8, device=dev)
            a_sc = torch.zeros((0, pad_cols // 64), dtype=torch.float16, device=dev)
            b_u8 = torch.zeros((n_rows, 0), dtype=torch.uint8, device=dev)
            b_sc = torch.zeros((n_rows,), dtype=torch.float16, device=dev)
            res_for_vq = x_base

        # Orthonormal FWHT-64 in equilibrated domain (zero multipliers in hardware)
        signs = rademacher_signs_u32_exact(pad_cols, dev)
        x_rot = fwht_block64_orthonormal(res_for_vq / d_f32.unsqueeze(0), signs, inverse=False)
        n_blocks_per_row = pad_cols // 64
        blocks = x_rot.reshape(n_rows * n_blocks_per_row, 64)
        blk_rms = blocks.square().mean(dim=1, keepdim=True).sqrt()
        # Preserve exact zero vacuum blocks (`blk_rms == 0 -> scale = 0, code = 0 -> exact 0.0`)!
        zero_blk_mask = (blocks.abs().amax(dim=1, keepdim=True) <= 1e-12)
        scales_f16 = torch.where(zero_blk_mask, torch.zeros_like(blk_rms), blk_rms.clamp(min=1e-7)).to(torch.float16)
        sc_safe = scales_f16.float().clamp(min=1e-7)
        z_4d = torch.where(zero_blk_mask, torch.zeros_like(blocks), blocks / sc_safe).reshape(-1, 4)
        w_energy = (scales_f16.float().square()).repeat_interleave(16, dim=0).squeeze(-1)

        # Fit `LUT256` (`lut[0] = (0,0,0,0)` hard-pinned + D4 root shells + energy-weighted Lloyd-Max)
        active_mask = w_energy > 0
        z_seed = z_4d[active_mask] if int(active_mask.sum().item()) >= 256 else z_4d
        lut = _seed_d4_zero_lattice(z_seed)
        lut[0].zero_()
        for _ in range(self.kmeans_iters):
            idx = _nearest_lut256(z_4d, lut)
            lut[0].zero_()
            w_z = z_4d * w_energy.unsqueeze(1)
            num = torch.zeros((256, 4), dtype=torch.float32, device=dev)
            den = torch.zeros(256, dtype=torch.float32, device=dev)
            num.index_add_(0, idx, w_z)
            den.index_add_(0, idx, w_energy)
            valid = den > 1e-10
            valid[0] = False
            lut[valid] = num[valid] / den[valid].unsqueeze(1)
            lut[0].zero_()

        lut256_f16 = lut.to(torch.float16).contiguous()
        lut_q = lut256_f16.float()
        codes_u8 = _nearest_lut256(z_4d, lut_q).to(torch.uint8)
        # Force exact zero blocks strictly to code `0` (`(0,0,0,0)`)
        zero_vec4_mask = zero_blk_mask.repeat_interleave(16, dim=0).squeeze(-1)
        codes_u8 = torch.where(zero_vec4_mask, torch.zeros_like(codes_u8), codes_u8)

        vq_blocks = lut_q[codes_u8.long()].reshape(n_rows * n_blocks_per_row, 64) * scales_f16.float()
        x_vq = fwht_block64_orthonormal(vq_blocks.reshape(n_rows, pad_cols), signs, inverse=True) * d_f32.unsqueeze(0)
        # Exact zero guarantee on vacuum blocks before scar addition
        zero_elem_mask = zero_blk_mask.reshape(n_rows, n_blocks_per_row, 1).expand(n_rows, n_blocks_per_row, 64).reshape(n_rows, pad_cols)
        x_vq = torch.where(zero_elem_mask, torch.zeros_like(x_vq), x_vq)

        # Pass 2: Re-solve compressed INT4 scar directly against peeled VQ4 residual `(X_base - X_vq)`!
        if eff_rank > 0:
            a_u8, a_sc, b_u8, b_sc, _ = _extract_compressed_int4_asvd(
                x_base - x_vq, torch.ones_like(col_rms), rank=eff_rank, alpha=0.5
            )
            b_sc = torch.where(orig_zero_row, torch.zeros_like(b_sc), b_sc)
            scar_p2 = dequant_rowwise_int4(b_u8, b_sc) @ dequant_block64_int4(a_u8, a_sc)
            x_approx = x_vq + scar_p2
        else:
            x_approx = x_vq

        # Sparse point-shock capture against true original `x` (refines peeled spikes + any remaining high-sigma shocks)
        rem_err = x - x_approx
        if pad_cols > n_cols:
            rem_err[:, n_cols:] = 0.0
        rem_err[orig_zero_row] = 0.0
        flat_err = rem_err.reshape(-1)
        err_std = x_base[~orig_zero_row].square().mean().sqrt().clamp(min=1e-8) if bool((~orig_zero_row).any().item()) else torch.tensor(1.0, device=dev)
        cand_mask = flat_err.abs() > (self.spike_sigma_threshold * 0.25 * err_std)
        if pre_spike_idx.numel() > 0:
            cand_mask[pre_spike_idx] = True
        if self.max_point_spikes > 0 and bool(cand_mask.any().item()):
            cand_idx = torch.nonzero(cand_mask, as_tuple=False).squeeze(-1)
            if cand_idx.numel() > self.max_point_spikes:
                _, top_rel = torch.topk(flat_err[cand_idx].abs(), k=self.max_point_spikes)
                cand_idx = cand_idx[top_rel]
            cand_idx, _ = torch.sort(cand_idx)
            spike_idx_i32 = cand_idx.to(torch.int32).contiguous()
            spike_val_f16 = flat_err[cand_idx].to(torch.float16).contiguous()
        else:
            spike_idx_i32 = torch.zeros((0,), dtype=torch.int32, device=dev)
            spike_val_f16 = torch.zeros((0,), dtype=torch.float16, device=dev)

        # Pack 4 x uint8 codes into 1 uint32 (`1 u32 = 16 floats`) matching WGSL layout
        c4 = codes_u8.reshape(n_rows, pad_cols // 16, 4).to(torch.int64)
        codes_u32 = (c4[:, :, 0] | (c4[:, :, 1] << 8) | (c4[:, :, 2] << 16) | (c4[:, :, 3] << 24)).to(torch.uint32).contiguous()

        return PhysicalVQ4ScarPackage(
            orig_rows=n_rows,
            orig_cols=n_cols,
            padded_cols=pad_cols,
            rank=eff_rank,
            codes_u32=codes_u32,
            scales_f16=scales_f16.reshape(n_rows, n_blocks_per_row).contiguous(),
            lut256_f16=lut256_f16,
            d_eq_f16=d_eq_f16.contiguous(),
            a_scar_u8=a_u8.contiguous(),
            a_scales_f16=a_sc.contiguous(),
            b_scar_u8=b_u8.contiguous(),
            b_row_scales_f16=b_sc.contiguous(),
            spike_flat_idx_i32=spike_idx_i32,
            spike_residual_f16=spike_val_f16,
        )

    @staticmethod
    @torch.no_grad()
    def decode_full(pkg: PhysicalVQ4ScarPackage) -> torch.Tensor:
        """Reconstructs the full `[orig_rows, orig_cols]` tensor from a `PhysicalVQ4ScarPackage`."""
        dev = pkg.codes_u32.device
        n_rows, pad_cols = pkg.orig_rows, pkg.padded_cols
        signs = rademacher_signs_u32_exact(pad_cols, dev)

        c_i64 = pkg.codes_u32.to(torch.int64)
        c0 = c_i64 & 0xFF
        c1 = (c_i64 >> 8) & 0xFF
        c2 = (c_i64 >> 16) & 0xFF
        c3 = (c_i64 >> 24) & 0xFF
        codes_flat = torch.stack((c0, c1, c2, c3), dim=-1).reshape(-1)

        lut_f32 = pkg.lut256_f16.float()
        vq_blks = lut_f32[codes_flat].reshape(n_rows, pad_cols // 64, 64) * pkg.scales_f16.float().unsqueeze(-1)
        zero_blk = (pkg.scales_f16 == 0).unsqueeze(-1)
        x_vq = fwht_block64_orthonormal(vq_blks.reshape(n_rows, pad_cols), signs, inverse=True) * pkg.d_eq_f16.float().unsqueeze(0)
        x_vq = torch.where(zero_blk.expand_as(vq_blks).reshape(n_rows, pad_cols), torch.zeros_like(x_vq), x_vq)

        if pkg.rank > 0:
            a_f32 = dequant_block64_int4(pkg.a_scar_u8, pkg.a_scales_f16)
            b_f32 = dequant_rowwise_int4(pkg.b_scar_u8, pkg.b_row_scales_f16)
            out = x_vq + (b_f32 @ a_f32)
        else:
            out = x_vq

        if pkg.spike_flat_idx_i32.numel() > 0:
            flat_out = out.reshape(-1)
            flat_out.index_add_(0, pkg.spike_flat_idx_i32.long(), pkg.spike_residual_f16.float())

        return out[:, : pkg.orig_cols]

    @staticmethod
    @torch.no_grad()
    def decode_block_o1(pkg: PhysicalVQ4ScarPackage, row_idx: int, block_idx: int) -> torch.Tensor:
        """True `O(1)` random-access read of a single 64-element block `[row_idx, block_idx*64 : (block_idx+1)*64]`.

        Reads ONLY the `16 B` packed codes + `2 B` scale + single-block scar slice (`0` neighbor blocks touched!).
        """
        dev = pkg.codes_u32.device
        start_col = block_idx * 64
        end_col = start_col + 64
        if not (0 <= row_idx < pkg.orig_rows and 0 <= start_col < pkg.padded_cols):
            raise IndexError(f"Block ({row_idx}, {block_idx}) out of bounds")

        scale_val = float(pkg.scales_f16[row_idx, block_idx].item())
        if scale_val == 0.0:
            blk_vq = torch.zeros(64, dtype=torch.float32, device=dev)
        else:
            u32_quad = pkg.codes_u32[row_idx, block_idx * 4 : (block_idx + 1) * 4].to(torch.int64)
            c16 = torch.stack(
                (u32_quad & 0xFF, (u32_quad >> 8) & 0xFF, (u32_quad >> 16) & 0xFF, (u32_quad >> 24) & 0xFF),
                dim=-1,
            ).reshape(16)
            vec64_rot = pkg.lut256_f16.float()[c16].reshape(64) * scale_val
            # Local Rademacher signs for `[start_col : end_col]` Only (`O(1)` — 64 elements!)
            idx64 = torch.arange(start_col, end_col, device=dev, dtype=torch.int64)
            h = (idx64 * 2654435761) & 0xFFFFFFFF
            h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
            signs64 = torch.where((h & 1) == 0, 1.0, -1.0)
            blk_vq = fwht_block64_orthonormal(vec64_rot, signs64, inverse=True) * pkg.d_eq_f16[start_col:end_col].float()

        if pkg.rank > 0:
            b_row_u8 = pkg.b_scar_u8[row_idx : row_idx + 1]
            b_row_sc = pkg.b_row_scales_f16[row_idx : row_idx + 1]
            b_row = dequant_rowwise_int4(b_row_u8, b_row_sc).squeeze(0)  # [rank]

            a_blk_u8 = pkg.a_scar_u8[:, block_idx * 32 : (block_idx + 1) * 32]
            a_blk_sc = pkg.a_scales_f16[:, block_idx : block_idx + 1]
            a_blk = dequant_block64_int4(a_blk_u8, a_blk_sc)  # [rank, 64]
            out64 = blk_vq + (b_row @ a_blk)
        else:
            out64 = blk_vq

        if pkg.spike_flat_idx_i32.numel() > 0:
            flat_lo = row_idx * pkg.padded_cols + start_col
            flat_hi = flat_lo + 64
            s_i64 = pkg.spike_flat_idx_i32.long()
            lo_pos = int(torch.searchsorted(s_i64, flat_lo, right=False).item())
            hi_pos = int(torch.searchsorted(s_i64, flat_hi, right=False).item())
            if hi_pos > lo_pos:
                local_offsets = s_i64[lo_pos:hi_pos] - flat_lo
                out64.index_add_(0, local_offsets, pkg.spike_residual_f16[lo_pos:hi_pos].float())

        valid_len = min(64, max(0, pkg.orig_cols - start_col))
        return out64[:valid_len]


class OrthogonalSubspaceProjector:
    """`P_perp = I - U U^T` structured interference canceller (`SKA / VLBI RFI` & `Adaptive Optics`).

    Guarantees exact `0.0` collateral distortion on signal components orthogonal to `span(U)`.
    """

    def __init__(self, basis_u: torch.Tensor) -> None:
        # `basis_u`: [C, k] orthonormal columns (`U^T U = I_k`)
        q, _ = torch.linalg.qr(basis_u.detach().float(), mode="reduced")
        self.basis_u = q

    @classmethod
    @torch.no_grad()
    def fit_from_interference_calibration(
        cls,
        calib_matrix: torch.Tensor,
        subspace_rank: int = 4,
    ) -> "OrthogonalSubspaceProjector":
        """Extracts top-`k` structured interference directions from `[C, T_cal]` array snapshots."""
        u, _, _ = torch.linalg.svd(calib_matrix.detach().float(), full_matrices=False)
        return cls(u[:, :subspace_rank])

    @torch.no_grad()
    def project_clean(self, array_signal: torch.Tensor) -> torch.Tensor:
        """Applies `P_perp X = X - U (U^T X)` across `[C, T]` or `[..., C]` tensors."""
        x = array_signal.float()
        u = self.basis_u.to(x.device)
        if x.shape[0] == u.shape[0] and x.ndim == 2:
            return x - u @ (u.T @ x)
        return x - (x @ u) @ u.T


class CircularRingBufferOT:
    """Baseline fixed-capacity circular recorder (`FDR / Telemetry Ring`) subject to `O(T)` overwrite."""

    def __init__(self, channels: int = 64, budget_bytes: int = 34816, device: Optional[torch.device] = None) -> None:
        dev = device or torch.device("cpu")
        self.channels = channels
        # Each stored frame needs `channels * 2` bytes (FP16) + `8` bytes timestamp (int64)
        bytes_per_entry = channels * 2 + 8
        self.capacity = max(1, budget_bytes // bytes_per_entry)
        self.frames_f16 = torch.zeros((self.capacity, channels), dtype=torch.float16, device=dev)
        self.timestamps_i64 = torch.full((self.capacity,), -1, dtype=torch.int64, device=dev)
        self.write_ptr = 0
        self.total_ingested = 0

    @torch.no_grad()
    def ingest(self, frame_1d: torch.Tensor, timestamp: int) -> None:
        slot = self.write_ptr % self.capacity
        self.frames_f16[slot].copy_(frame_1d.to(torch.float16))
        self.timestamps_i64[slot] = int(timestamp)
        self.write_ptr += 1
        self.total_ingested += 1

    def physical_bytes(self) -> int:
        return unique_tensor_storage_bytes([self.frames_f16, self.timestamps_i64])


class SurpriseGatedAnomalySCache:
    """`34.8 KB` (`34,816 Bytes` exact) Constant-Space `SCache O(1)` Black-Box & Satellite Recorder.

    Physical layout (`C = 64`, `S = 48` anomaly slots, `R = 39` local ring frames = `34,816 B` exact):
      - `w_scar_f32`       : `[64, 64]` float32 Widrow-Hoff Delta associative fast-weights (`16,384 B`)
      - `slot_keys_f16`    : `[48, 64]` float16 normalized shock key vectors              (` 6,144 B`)
      - `slot_vals_f16`    : `[48, 64]` float16 shock deviation snapshots                 (` 6,144 B`)
      - `slot_ts_i64`      : `[48]`     int64 mission frame timestamps                    (`   384 B`)
      - `slot_surprise_f32`: `[48]`     float32 peak surprise scores                      (`   192 B`)
      - `ema_mean_f32`     : `[64]`     float32 nominal manifold tracker                  (`   256 B`)
      - `ema_var_f32`      : `[64]`     float32 nominal variance tracker                  (`   256 B`)
      - `recent_ring_f16`  : `[39, 64]` float16 rolling short-term ring                   (` 4,992 B`)
      - `state_meta_i64`   : `[8]`      int64 counters/pointers                           (`    64 B`)
      ------------------------------------------------------------------------------------------------
      TOTAL PHYSICAL STORAGE                                                              = `34,816 B` (`34.816 KB`)
    """

    def __init__(
        self,
        channels: int = 64,
        num_slots: int = 48,
        ring_len: int = 39,
        surprise_threshold: float = 4.2,
        ema_decay: float = 0.99,
        delta_lr: float = 0.35,
        device: Optional[torch.device] = None,
    ) -> None:
        dev = device or torch.device("cpu")
        self.channels = channels
        self.num_slots = num_slots
        self.ring_len = ring_len
        self.surprise_threshold = surprise_threshold
        self.ema_decay = ema_decay
        self.delta_lr = delta_lr

        self.w_scar_f32 = torch.zeros((channels, channels), dtype=torch.float32, device=dev)
        self.slot_keys_f16 = torch.zeros((num_slots, channels), dtype=torch.float16, device=dev)
        self.slot_vals_f16 = torch.zeros((num_slots, channels), dtype=torch.float16, device=dev)
        self.slot_ts_i64 = torch.full((num_slots,), -1, dtype=torch.int64, device=dev)
        self.slot_surprise_f32 = torch.zeros((num_slots,), dtype=torch.float32, device=dev)
        self.ema_mean_f32 = torch.zeros((channels,), dtype=torch.float32, device=dev)
        self.ema_var_f32 = torch.ones((channels,), dtype=torch.float32, device=dev)
        self.recent_ring_f16 = torch.zeros((ring_len, channels), dtype=torch.float16, device=dev)
        self.state_meta_i64 = torch.zeros((8,), dtype=torch.int64, device=dev)

    def all_tensors(self) -> List[torch.Tensor]:
        return [
            self.w_scar_f32,
            self.slot_keys_f16,
            self.slot_vals_f16,
            self.slot_ts_i64,
            self.slot_surprise_f32,
            self.ema_mean_f32,
            self.ema_var_f32,
            self.recent_ring_f16,
            self.state_meta_i64,
        ]

    def physical_bytes(self) -> int:
        return unique_tensor_storage_bytes(self.all_tensors())

    @torch.no_grad()
    def ingest(self, frame_1d: torch.Tensor, timestamp: int) -> float:
        """Ingests one sensor frame in strictly `O(1)` memory (`+0 B` allocation), returning surprise score."""
        x = frame_1d.detach().float()
        step = int(self.state_meta_i64[0].item())

        # Always update rolling short-term ring (`O(1)` wrap-around)
        ring_slot = step % self.ring_len
        self.recent_ring_f16[ring_slot].copy_(x.to(torch.float16))
        self.state_meta_i64[0] = step + 1

        # Warmup nominal statistics on initial routine frames
        if step < 32:
            alpha = 1.0 / (step + 1)
            diff = x - self.ema_mean_f32
            self.ema_mean_f32.add_(diff * alpha)
            self.ema_var_f32.mul_(1.0 - alpha).add_(diff.square() * alpha)
            return 0.0

        diff = x - self.ema_mean_f32
        std = self.ema_var_f32.clamp(min=1e-6).sqrt()
        # RMS z-score across channels (`~1.0` for nominal Gaussian telemetry, `>> 4.2` on physical shocks)
        surprise = float((diff / std).square().mean().sqrt().item())

        if surprise <= self.surprise_threshold:
            # Routine nominal frame: update baseline EMA ONLY; zero overwrite of anomaly slots!
            d = self.ema_decay
            self.ema_mean_f32.mul_(d).add_(x * (1.0 - d))
            self.ema_var_f32.mul_(d).add_(diff.square() * (1.0 - d))
            return surprise

        # High-surprise physical anomaly detected!
        k_hat = diff / diff.norm().clamp(min=1e-8)
        # Widrow-Hoff Delta associative fast-weight update: W <- W + eta * (diff - W k_hat) k_hat^T
        pred = self.w_scar_f32 @ k_hat
        self.w_scar_f32.add_(torch.outer(diff - pred, k_hat) * self.delta_lr)

        # Insert into lowest-surprise (or empty) anomaly slot if stronger than minimum stored shock
        min_val, min_idx_t = torch.min(self.slot_surprise_f32, dim=0)
        slot_idx = int(min_idx_t.item())
        if surprise > float(min_val.item()):
            self.slot_keys_f16[slot_idx].copy_(k_hat.to(torch.float16))
            self.slot_vals_f16[slot_idx].copy_(diff.to(torch.float16))
            self.slot_ts_i64[slot_idx] = int(timestamp)
            self.slot_surprise_f32[slot_idx] = surprise
            self.state_meta_i64[1] += 1

        return surprise

    @torch.no_grad()
    def query_associative(self, query_direction: torch.Tensor) -> torch.Tensor:
        """Recalls anomaly deviation vector from `W_scar` via associative unit-key probe."""
        q_hat = query_direction.float() / query_direction.float().norm().clamp(min=1e-8)
        return self.w_scar_f32 @ q_hat

    @torch.no_grad()
    def recall_from_partial_cue(
        self,
        partial_frame: torch.Tensor,
        observed_mask: torch.Tensor,
        input_is_deviation: bool = False,
    ) -> torch.Tensor:
        """Reconstructs a full `C`-channel frame (or deviation) from a subset of observed channels (`+0 B` alloc).

        Args:
            partial_frame: `[C]` tensor containing sensor readings (or shock deviations if `input_is_deviation=True`).
            observed_mask: `[C]` boolean mask (`True` for known/intact sensors, `False` for missing/failed sensors).
            input_is_deviation: If `True`, treats `partial_frame` as already baseline-centered (`x - ema_mean`)
                and returns the recalled `C`-channel shock deviation directly.
        """
        mask = observed_mask.to(dtype=torch.bool, device=self.w_scar_f32.device)
        pf = partial_frame.to(dtype=torch.float32, device=self.w_scar_f32.device)
        dev_obs = pf if input_is_deviation else (pf - self.ema_mean_f32)
        cue = torch.where(mask, dev_obs, torch.zeros_like(dev_obs))
        recalled_dev = self.query_associative(cue)
        return recalled_dev if input_is_deviation else (self.ema_mean_f32 + recalled_dev)

    def recorded_timestamps(self) -> List[int]:
        valid = self.slot_ts_i64[self.slot_ts_i64 >= 0].tolist()
        return sorted(int(t) for t in valid)
