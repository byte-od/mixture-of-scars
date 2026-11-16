"""Pure Scar-Native 3D Point-Cloud Rectified Flow Generator (`PointScarFlow` v1.0.0).

Synthesizes continuous, non-catalog 3D point-cloud manifolds (`N = 1,024 x [x, y, z, salience]`)
directly from `Qwen2.5-3B` `Layer 23` `P_perp_lang`-projected semantic vectors (`z_23 in R^2048`):

  1. **100% Tabula-Rasa `ScarNativeLinear` (`0` `torch.nn.Linear` dense layers):**
     Every projection uses multiplier-free `FWHT-64` + channel equilibration `d_eq` +
     canonical `256 x vec4<f16>` `D4` lattice codebook (`LUT256[0] == (0,0,0,0)`) via `VQ4-STE`
     + compressed `INT4-STE` structural scars `(A_scar, B_scar)`. Packs to binary with
     **exact `0.000000` post-training quantization shock**.
  2. **Deterministic Spherical Optimal-Transport Rectified Flow (`0` `Math.random()`):**
     Starts from a canonical Fibonacci sphere `X_0 in S^2` and integrates straight-line Brenier
     Optimal-Transport trajectories `dX/dtau = v_theta(X_tau, tau | z_23)` in `4` Euler steps.
  3. **Bidirectional Induced-Set Part-Point Cross-Attention (`O(N * M)` linear complexity):**
     Synthesizes `M = 16` latent 3D structural part tokens from `z_23` and exchanges global/local
     geometry with `N = 1,024` points via exact `head_dim = 32` Flash Cross-Attention on GPU.
  4. **Dual Native + Physical WebGPU WGSL 1.0 Execution (`cos > 0.99999` verified parity).**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_native.model import NativeVQ4ScarPackedMatrix, RMSNormF32, ScarNativeLinear

POINT_SCAR_WGSL_PATH = Path(__file__).resolve().parent / "point_scar_flow.wgsl"


@dataclass
class PointScarFlowConfig:
    cond_in_dim: int = 2048
    cond_dim: int = 512
    d_model: int = 256
    d_ff: int = 512
    num_layers: int = 4
    num_heads: int = 8
    head_dim: int = 32
    num_parts: int = 16
    num_points: int = 1024
    scar_rank: int = 16
    cond_scar_rank: int = 24
    head_scar_rank: int = 8
    rms_norm_eps: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_canonical_fibonacci_sphere(
    num_points: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Deterministic golden-angle Fibonacci sphere `X_0 [num_points, 3]` (`0` randomness)."""
    dev = torch.device(device)
    idx = torch.arange(num_points, dtype=torch.float32, device=dev)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (2.0 * (idx + 0.5)) / float(num_points)
    rad = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = idx * phi
    x = rad * torch.cos(theta)
    z = rad * torch.sin(theta)
    return torch.stack([x, y, z], dim=-1).contiguous()


def build_sinusoidal_time_embed_64(tau: torch.Tensor) -> torch.Tensor:
    """Matches `psf_time_embed_64_f32` in `point_scar_flow.wgsl` bit-for-bit (`tau`: `[B]`)."""
    dev = tau.device
    k = torch.arange(32, dtype=torch.float32, device=dev)
    freq = torch.exp(-math.log(1000.0) * (k / 31.0)) * 32.0
    ang = tau.float().unsqueeze(-1) * freq.unsqueeze(0)  # [B, 32]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def build_point_stem_features_64(
    x_tau: torch.Tensor,
    x_0: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Matches `psf_build_point_features_f32` in `point_scar_flow.wgsl` (`[B, N, 64]`)."""
    bsz, n_pts, _ = x_tau.shape
    xf = x_tau.float()
    x0 = x_0.float().unsqueeze(0).expand(bsz, n_pts, 3)

    pos = xf[..., :3]
    disp = pos - x0
    rad = torch.sqrt(torch.clamp(pos.square().sum(dim=-1, keepdim=True), min=1e-12))
    t_col = tau.float().view(bsz, 1, 1).expand(bsz, n_pts, 1)
    one_minus_t = 1.0 - t_col

    feats: List[torch.Tensor] = [xf, x0, disp, rad, t_col, one_minus_t]  # 4 + 3 + 3 + 1 + 1 + 1 = 13

    for f_idx in range(4):
        omega = math.pi * float(1 << f_idx)
        for ax in range(3):
            c = pos[..., ax : ax + 1] * omega
            feats.append(torch.sin(c))
            feats.append(torch.cos(c))

    for f_idx in range(4):
        omega = math.pi * float(1 << f_idx)
        for ax in range(3):
            c0 = x0[..., ax : ax + 1] * omega
            feats.append(torch.sin(c0))
            feats.append(torch.cos(c0))

    xy = x0[..., 0:1] * x0[..., 1:2]
    yz = x0[..., 1:2] * x0[..., 2:3]
    zx = x0[..., 2:3] * x0[..., 0:1]
    feats.extend([xy, yz, zx])

    out = torch.cat(feats, dim=-1)
    assert out.shape[-1] == 64
    return out


def _cross_attention_hd32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Multi-head cross attention (`q: [B, N_q, D]`, `k,v: [B, N_kv, D]`)."""
    bsz, n_q, d_model = q.shape
    n_kv = k.shape[1]
    qh = q.view(bsz, n_q, num_heads, head_dim).transpose(1, 2)
    kh = k.view(bsz, n_kv, num_heads, head_dim).transpose(1, 2)
    vh = v.view(bsz, n_kv, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).contiguous().view(bsz, n_q, d_model)


class PointScarFlowBlock(nn.Module):
    """Bidirectional Part-Point Induced-Set Block using 100% `ScarNativeLinear` operators."""

    def __init__(self, cfg: PointScarFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        ff = cfg.d_ff
        rk = cfg.scar_rank

        self.ada_mod = ScarNativeLinear(cfg.cond_dim, 6 * d, scar_rank=rk)
        with torch.no_grad():
            self.ada_mod.w_master.mul_(0.05)

        # Stage A: Part tokens read global point-cloud state (`Part <- Point`) + Part SwiGLU
        self.part_ln1 = RMSNormF32(d, cfg.rms_norm_eps)
        self.pt_kv_ln = RMSNormF32(d, cfg.rms_norm_eps)
        self.part_q = ScarNativeLinear(d, d, scar_rank=rk)
        self.part_k_from_pt = ScarNativeLinear(d, d, scar_rank=rk)
        self.part_v_from_pt = ScarNativeLinear(d, d, scar_rank=rk)
        self.part_o = ScarNativeLinear(d, d, scar_rank=rk)

        self.part_ln2 = RMSNormF32(d, cfg.rms_norm_eps)
        self.part_gate = ScarNativeLinear(d, ff, scar_rank=rk)
        self.part_up = ScarNativeLinear(d, ff, scar_rank=rk)
        self.part_down = ScarNativeLinear(ff, d, scar_rank=rk)

        # Stage B: Points attend to semantic 3D part tokens (`Point <- Part`)
        self.pt_ln1 = RMSNormF32(d, cfg.rms_norm_eps)
        self.part_kv_ln = RMSNormF32(d, cfg.rms_norm_eps)
        self.pt_q = ScarNativeLinear(d, d, scar_rank=rk)
        self.pt_k_from_part = ScarNativeLinear(d, d, scar_rank=rk)
        self.pt_v_from_part = ScarNativeLinear(d, d, scar_rank=rk)
        self.pt_o = ScarNativeLinear(d, d, scar_rank=rk)

        # Stage C: Per-point AdaLN ScarNative SwiGLU
        self.pt_ln2 = RMSNormF32(d, cfg.rms_norm_eps)
        self.pt_gate = ScarNativeLinear(d, ff, scar_rank=rk)
        self.pt_up = ScarNativeLinear(d, ff, scar_rank=rk)
        self.pt_down = ScarNativeLinear(ff, d, scar_rank=rk)

    def forward(
        self,
        h_pts: torch.Tensor,
        parts: torch.Tensor,
        c_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg.d_model
        nh = self.cfg.num_heads
        hd = self.cfg.head_dim

        mod = self.ada_mod(F.silu(c_global))  # [B, 6 * D]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6, dim=-1)

        # Stage A: `Part <- Point` + Part SwiGLU
        p_n1 = self.part_ln1(parts)
        pt_kv = self.pt_kv_ln(h_pts)
        p_attn = _cross_attention_hd32(
            self.part_q(p_n1),
            self.part_k_from_pt(pt_kv),
            self.part_v_from_pt(pt_kv),
            nh,
            hd,
        )
        parts = parts + self.part_o(p_attn)
        p_n2 = self.part_ln2(parts)
        parts = parts + self.part_down(F.silu(self.part_gate(p_n2)) * self.part_up(p_n2))

        # Stage B: `Point <- Part` with AdaLN modulation
        pt_n1 = self.pt_ln1(h_pts) * (1.0 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        part_kv = self.part_kv_ln(parts)
        pt_attn = _cross_attention_hd32(
            self.pt_q(pt_n1),
            self.pt_k_from_part(part_kv),
            self.pt_v_from_part(part_kv),
            nh,
            hd,
        )
        h_pts = h_pts + (1.0 + gate_a.unsqueeze(1)) * self.pt_o(pt_attn)

        # Stage C: Point SwiGLU with AdaLN modulation
        pt_n2 = self.pt_ln2(h_pts) * (1.0 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
        pt_mlp = self.pt_down(F.silu(self.pt_gate(pt_n2)) * self.pt_up(pt_n2))
        h_pts = h_pts + (1.0 + gate_m.unsqueeze(1)) * pt_mlp

        assert h_pts.shape[-1] == d
        return h_pts, parts


class PointScarFlow(nn.Module):
    """Tabula-Rasa `ScarNative` Continuous 3D Point-Cloud Generator (`0` `nn.Linear` dense layers)."""

    def __init__(self, cfg: Optional[PointScarFlowConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or PointScarFlowConfig()
        c = self.cfg
        if c.d_model != c.num_heads * c.head_dim:
            raise ValueError("d_model must equal num_heads * head_dim")
        if c.head_dim != 32:
            raise ValueError("head_dim must be 32 to match `psf_flash_cross_attn_hd32` WGSL kernel")

        x0 = build_canonical_fibonacci_sphere(c.num_points, "cpu")
        self.register_buffer("x0_sphere", x0, persistent=True)

        # Global conditioning encoder (`z_23 [2048]` + `t_emb [64]` -> `c_global [512]` + `parts [M, D]`)
        self.cond_z = ScarNativeLinear(c.cond_in_dim, c.cond_dim, scar_rank=c.cond_scar_rank)
        self.cond_t = ScarNativeLinear(64, c.cond_dim, scar_rank=8)
        self.cond_norm = RMSNormF32(c.cond_dim, c.rms_norm_eps)
        self.cond_mlp = ScarNativeLinear(c.cond_dim, c.cond_dim, scar_rank=c.cond_scar_rank)
        self.part_seed = ScarNativeLinear(c.cond_dim, c.num_parts * c.d_model, scar_rank=c.cond_scar_rank)

        # Point stem (`64 -> d_model`)
        self.point_stem = ScarNativeLinear(64, c.d_model, scar_rank=c.scar_rank)

        # Induced-set Point-Scar Flow blocks
        self.blocks = nn.ModuleList([PointScarFlowBlock(c) for _ in range(c.num_layers)])

        # Velocity head (`d_model -> 4`: `[vx, vy, vz, vs]`)
        self.final_norm = RMSNormF32(c.d_model, c.rms_norm_eps)
        self.vel_head = ScarNativeLinear(c.d_model, 4, scar_rank=c.head_scar_rank)
        with torch.no_grad():
            self.vel_head.w_master.mul_(0.05)

    def assert_zero_dense_linear(self) -> None:
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                raise AssertionError(f"Forbidden `torch.nn.Linear` found at `{name}`!")

    def total_commitment_loss(self) -> torch.Tensor:
        losses = [m.commitment_loss() for m in self.modules() if isinstance(m, ScarNativeLinear)]
        if not losses:
            return torch.zeros((), device=self.x0_sphere.device)
        return torch.stack(losses).mean()

    def pack_all_vq4_scar(self, drop_master: bool = False) -> int:
        total_bytes = 0
        for mod in self.modules():
            if isinstance(mod, ScarNativeLinear):
                pkg = mod.pack_vq4_scar(drop_master=drop_master)
                total_bytes += pkg.physical_bytes()
        return total_bytes

    def initial_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Deterministic initial state `X_0 [B, N, 4]` (`[x0, y0, z0, 0.5]`, `0` randomness)."""
        dev = device or self.x0_sphere.device
        x0 = self.x0_sphere.to(dev).unsqueeze(0).expand(batch_size, self.cfg.num_points, 3)
        s0 = torch.full((batch_size, self.cfg.num_points, 1), 0.5, dtype=torch.float32, device=dev)
        return torch.cat([x0, s0], dim=-1).contiguous()

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        z_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts velocity `v_theta(X_tau, tau | z_cond)` in `[B, N, 4]`."""
        bsz = x_tau.shape[0]
        c = self.cfg
        t_emb = build_sinusoidal_time_embed_64(tau)
        h_cond = F.silu(self.cond_z(z_cond.float()) + self.cond_t(t_emb))
        c_global = self.cond_mlp(self.cond_norm(h_cond))
        parts = self.part_seed(F.silu(c_global)).view(bsz, c.num_parts, c.d_model)

        feats64 = build_point_stem_features_64(x_tau, self.x0_sphere, tau)
        h_pts = self.point_stem(feats64)

        for blk in self.blocks:
            h_pts, parts = blk(h_pts, parts, c_global)

        return self.vel_head(self.final_norm(h_pts))

    @torch.no_grad()
    def sample_euler(
        self,
        z_cond: torch.Tensor,
        num_steps: int = 4,
        guidance_scale: float = 1.75,
    ) -> torch.Tensor:
        """Deterministic `num_steps` Rectified Flow Euler solver with Classifier-Free Guidance."""
        if z_cond.ndim == 1:
            z_in = z_cond.unsqueeze(0)
            squeeze_out = True
        else:
            z_in = z_cond
            squeeze_out = False

        bsz = z_in.shape[0]
        dev = z_in.device
        x_cur = self.initial_state(bsz, dev)
        dt = 1.0 / float(max(1, num_steps))
        use_cfg = abs(guidance_scale - 1.0) > 1e-5
        z_null = torch.zeros_like(z_in) if use_cfg else None

        for step in range(num_steps):
            t_val = float(step) * dt
            tau = torch.full((bsz,), t_val, dtype=torch.float32, device=dev)
            v_cond = self.forward(x_cur, tau, z_in)
            if use_cfg and z_null is not None:
                v_uncond = self.forward(x_cur, tau, z_null)
                v_eff = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_eff = v_cond
            x_cur = x_cur + dt * v_eff

        pos_clamped = torch.clamp(x_cur[..., :3], -1.25, 1.25)
        sal_clamped = torch.clamp(x_cur[..., 3:4], 0.0, 1.0)
        out = torch.cat([pos_clamped, sal_clamped], dim=-1)
        return out.squeeze(0) if squeeze_out else out


class _BinaryShardPacker:
    def __init__(self) -> None:
        self.buf = bytearray()

    def append_aligned(self, raw: bytes) -> int:
        pad = (-len(self.buf)) & 3
        if pad:
            self.buf.extend(b"\x00" * pad)
        u32_off = len(self.buf) // 4
        self.buf.extend(raw)
        tail = (-len(self.buf)) & 3
        if tail:
            self.buf.extend(b"\x00" * tail)
        return u32_off


def export_pointscar_webgpu_bundle(
    model: PointScarFlow,
    out_dir: str | Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Exports `PointScarFlow` into a zero-shock `VQ4-Scar` WebGPU bundle (`pointscar_v1_shard.bin`)."""
    model.assert_zero_dense_linear()
    model.eval()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    packer = _BinaryShardPacker()
    cfg = model.cfg

    def _pack_linear(lin: ScarNativeLinear) -> Dict[str, Any]:
        pkg = lin.pack_vq4_scar(drop_master=False)
        vq_off = packer.append_aligned(pkg.codes_u32.detach().cpu().contiguous().numpy().tobytes())
        sc_off = packer.append_aligned(pkg.scales_f16.detach().cpu().contiguous().numpy().tobytes())
        lut_off = packer.append_aligned(pkg.lut256_f16.detach().cpu().contiguous().numpy().tobytes())
        deq_off = packer.append_aligned(pkg.d_eq_f16.detach().cpu().contiguous().numpy().tobytes())
        spec: Dict[str, Any] = {
            "out_dim": lin.out_features,
            "in_dim": lin.in_features,
            "rank": lin.scar_rank,
            "vq_codes_u32_off": vq_off,
            "vq_scales_u32_off": sc_off,
            "lut256_u32_off": lut_off,
            "d_eq_u32_off": deq_off,
        }
        if lin.scar_rank > 0:
            ap_off = packer.append_aligned(pkg.a_scar_u8.detach().cpu().contiguous().numpy().tobytes())
            as_off = packer.append_aligned(pkg.a_scales_f16.detach().cpu().contiguous().numpy().tobytes())
            b_comb = (
                pkg.b_scar_u8.detach().cpu().contiguous().numpy().tobytes()
                + pkg.b_row_scales_f16.detach().cpu().contiguous().numpy().tobytes()
            )
            bs_off = packer.append_aligned(b_comb)
            spec["ascar_packed_u32_off"] = ap_off
            spec["ascar_scales_u32_off"] = as_off
            spec["bscar_u32_off"] = bs_off
        return spec

    def _pack_norm(norm: RMSNormF32) -> int:
        return packer.append_aligned(norm.weight.detach().cpu().to(torch.float16).contiguous().numpy().tobytes())

    x0_f32 = model.x0_sphere.detach().cpu().float().contiguous().numpy()
    x0_off = packer.append_aligned(x0_f32.tobytes())

    cond_spec = {
        "cond_z": _pack_linear(model.cond_z),
        "cond_t": _pack_linear(model.cond_t),
        "cond_norm_u32_off": _pack_norm(model.cond_norm),
        "cond_mlp": _pack_linear(model.cond_mlp),
        "part_seed": _pack_linear(model.part_seed),
    }
    stem_spec = _pack_linear(model.point_stem)

    blocks_spec: List[Dict[str, Any]] = []
    for l_idx, blk in enumerate(model.blocks):
        assert isinstance(blk, PointScarFlowBlock)
        blocks_spec.append(
            {
                "layer_idx": l_idx,
                "ada_mod": _pack_linear(blk.ada_mod),
                "part_ln1_u32_off": _pack_norm(blk.part_ln1),
                "pt_kv_ln_u32_off": _pack_norm(blk.pt_kv_ln),
                "part_q": _pack_linear(blk.part_q),
                "part_k_from_pt": _pack_linear(blk.part_k_from_pt),
                "part_v_from_pt": _pack_linear(blk.part_v_from_pt),
                "part_o": _pack_linear(blk.part_o),
                "part_ln2_u32_off": _pack_norm(blk.part_ln2),
                "part_gate": _pack_linear(blk.part_gate),
                "part_up": _pack_linear(blk.part_up),
                "part_down": _pack_linear(blk.part_down),
                "pt_ln1_u32_off": _pack_norm(blk.pt_ln1),
                "part_kv_ln_u32_off": _pack_norm(blk.part_kv_ln),
                "pt_q": _pack_linear(blk.pt_q),
                "pt_k_from_part": _pack_linear(blk.pt_k_from_part),
                "pt_v_from_part": _pack_linear(blk.pt_v_from_part),
                "pt_o": _pack_linear(blk.pt_o),
                "pt_ln2_u32_off": _pack_norm(blk.pt_ln2),
                "pt_gate": _pack_linear(blk.pt_gate),
                "pt_up": _pack_linear(blk.pt_up),
                "pt_down": _pack_linear(blk.pt_down),
            }
        )

    final_norm_off = _pack_norm(model.final_norm)
    vel_head_spec = _pack_linear(model.vel_head)

    shard_file = "pointscar_v1_shard.bin"
    (out_path / shard_file).write_bytes(bytes(packer.buf))
    shutil.copyfile(POINT_SCAR_WGSL_PATH, out_path / "point_scar_flow.wgsl")

    manifest: Dict[str, Any] = {
        "format": "point_scar_flow_webgpu_v1",
        "version": "1.0.0",
        "architecture": "ScarNative_VQ4_OT_RectifiedFlow",
        "config": cfg.to_dict(),
        "shard_file": shard_file,
        "shard_bytes": len(packer.buf),
        "wgsl_shader": "point_scar_flow.wgsl",
        "x0_f32_u32_off": x0_off,
        "cond": cond_spec,
        "point_stem": stem_spec,
        "blocks": blocks_spec,
        "final_norm_u32_off": final_norm_off,
        "vel_head": vel_head_spec,
    }
    if extra_metadata:
        manifest["training_metadata"] = extra_metadata

    (out_path / "pointscar_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_pointscar_from_webgpu_bundle(
    bundle_dir: str | Path,
    device: torch.device | str = "cpu",
) -> PointScarFlow:
    """Reconstructs a zero-master packed `PointScarFlow` directly from `pointscar_v1_shard.bin`."""
    b_path = Path(bundle_dir)
    manifest = json.loads((b_path / "pointscar_manifest.json").read_text(encoding="utf-8"))
    raw = (b_path / manifest["shard_file"]).read_bytes()
    dev = torch.device(device)
    cfg = PointScarFlowConfig(**manifest["config"])
    model = PointScarFlow(cfg)

    def _slice(u32_off: int, byte_len: int) -> bytes:
        s = int(u32_off) * 4
        return raw[s : s + byte_len]

    def _load_norm(norm: RMSNormF32, u32_off: int) -> None:
        dim = int(norm.weight.numel())
        arr = np.frombuffer(_slice(u32_off, dim * 2), dtype=np.float16).astype(np.float32)
        with torch.no_grad():
            norm.weight.copy_(torch.from_numpy(arr))

    def _load_linear(lin: ScarNativeLinear, spec: Dict[str, Any]) -> None:
        out_d = int(spec["out_dim"])
        in_d = int(spec["in_dim"])
        rk = int(spec["rank"])
        codes = torch.from_numpy(
            np.frombuffer(_slice(spec["vq_codes_u32_off"], out_d * (in_d // 16) * 4), dtype=np.uint32)
            .reshape(out_d, in_d // 16)
            .copy()
        ).to(dev)
        scales = torch.from_numpy(
            np.frombuffer(_slice(spec["vq_scales_u32_off"], out_d * (in_d // 64) * 2), dtype=np.float16)
            .reshape(out_d, in_d // 64)
            .copy()
        ).to(dev)
        lut = torch.from_numpy(
            np.frombuffer(_slice(spec["lut256_u32_off"], 256 * 4 * 2), dtype=np.float16)
            .reshape(256, 4)
            .copy()
        ).to(dev)
        deq = torch.from_numpy(
            np.frombuffer(_slice(spec["d_eq_u32_off"], in_d * 2), dtype=np.float16)
            .reshape(in_d)
            .copy()
        ).to(dev)

        if rk > 0:
            a_u8 = torch.from_numpy(
                np.frombuffer(_slice(spec["ascar_packed_u32_off"], rk * (in_d // 2)), dtype=np.uint8)
                .reshape(rk, in_d // 2)
                .copy()
            ).to(dev)
            a_sc = torch.from_numpy(
                np.frombuffer(_slice(spec["ascar_scales_u32_off"], rk * (in_d // 64) * 2), dtype=np.float16)
                .reshape(rk, in_d // 64)
                .copy()
            ).to(dev)
            b_bytes = out_d * (rk // 2)
            b_comb = _slice(spec["bscar_u32_off"], b_bytes + out_d * 2)
            b_u8 = torch.from_numpy(
                np.frombuffer(b_comb[:b_bytes], dtype=np.uint8).reshape(out_d, rk // 2).copy()
            ).to(dev)
            b_sc = torch.from_numpy(
                np.frombuffer(b_comb[b_bytes:], dtype=np.float16).reshape(out_d).copy()
            ).to(dev)
        else:
            a_u8 = torch.zeros((0, in_d // 2), dtype=torch.uint8, device=dev)
            a_sc = torch.zeros((0, in_d // 64), dtype=torch.float16, device=dev)
            b_u8 = torch.zeros((out_d, 0), dtype=torch.uint8, device=dev)
            b_sc = torch.zeros((out_d,), dtype=torch.float16, device=dev)

        lin.packed = NativeVQ4ScarPackedMatrix(
            out_features=out_d,
            in_features=in_d,
            rank=rk,
            codes_u32=codes,
            scales_f16=scales,
            lut256_f16=lut,
            d_eq_f16=deq,
            a_scar_u8=a_u8,
            a_scales_f16=a_sc,
            b_scar_u8=b_u8,
            b_row_scales_f16=b_sc,
            bias_f16=None,
        )
        lin.w_master = None  # type: ignore[assignment]
        lin.log_d_eq = None  # type: ignore[assignment]
        lin.a_scar_master = None
        lin.b_scar_master = None

    x0_arr = np.frombuffer(_slice(manifest["x0_f32_u32_off"], cfg.num_points * 3 * 4), dtype=np.float32).reshape(
        cfg.num_points, 3
    )
    with torch.no_grad():
        model.x0_sphere.copy_(torch.from_numpy(x0_arr.copy()))

    cond = manifest["cond"]
    _load_linear(model.cond_z, cond["cond_z"])
    _load_linear(model.cond_t, cond["cond_t"])
    _load_norm(model.cond_norm, cond["cond_norm_u32_off"])
    _load_linear(model.cond_mlp, cond["cond_mlp"])
    _load_linear(model.part_seed, cond["part_seed"])
    _load_linear(model.point_stem, manifest["point_stem"])

    for l_idx, blk in enumerate(model.blocks):
        assert isinstance(blk, PointScarFlowBlock)
        bs = manifest["blocks"][l_idx]
        _load_linear(blk.ada_mod, bs["ada_mod"])
        _load_norm(blk.part_ln1, bs["part_ln1_u32_off"])
        _load_norm(blk.pt_kv_ln, bs["pt_kv_ln_u32_off"])
        _load_linear(blk.part_q, bs["part_q"])
        _load_linear(blk.part_k_from_pt, bs["part_k_from_pt"])
        _load_linear(blk.part_v_from_pt, bs["part_v_from_pt"])
        _load_linear(blk.part_o, bs["part_o"])
        _load_norm(blk.part_ln2, bs["part_ln2_u32_off"])
        _load_linear(blk.part_gate, bs["part_gate"])
        _load_linear(blk.part_up, bs["part_up"])
        _load_linear(blk.part_down, bs["part_down"])
        _load_norm(blk.pt_ln1, bs["pt_ln1_u32_off"])
        _load_norm(blk.part_kv_ln, bs["part_kv_ln_u32_off"])
        _load_linear(blk.pt_q, bs["pt_q"])
        _load_linear(blk.pt_k_from_part, bs["pt_k_from_part"])
        _load_linear(blk.pt_v_from_part, bs["pt_v_from_part"])
        _load_linear(blk.pt_o, bs["pt_o"])
        _load_norm(blk.pt_ln2, bs["pt_ln2_u32_off"])
        _load_linear(blk.pt_gate, bs["pt_gate"])
        _load_linear(blk.pt_up, bs["pt_up"])
        _load_linear(blk.pt_down, bs["pt_down"])

    _load_norm(model.final_norm, manifest["final_norm_u32_off"])
    _load_linear(model.vel_head, manifest["vel_head"])
    model.to(dev)
    model.eval()
    return model


class PointScarFlowWebGPUEngine:
    """Physical WebGPU (`wgpu` Metal / Vulkan / DX12) Executor for `PointScarFlow` (`v1.0.0`)."""

    def __init__(self, bundle_dir: str | Path) -> None:
        import wgpu

        self.wgpu = wgpu
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "pointscar_manifest.json").read_text(encoding="utf-8"))
        self.cfg = PointScarFlowConfig(**self.manifest["config"])

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter available.")
        self.device = adapter.request_device_sync(required_limits={})
        info = adapter.info
        self.adapter_name = str(info.get("device") or info.get("description") or "WebGPU Adapter")

        wgsl_code = (self.bundle_dir / self.manifest["wgsl_shader"]).read_text(encoding="utf-8")
        self.shader_mod = self.device.create_shader_module(code=wgsl_code)
        self.pipelines: Dict[str, Any] = {}
        for ep in (
            "psf_build_point_features_f32",
            "psf_time_embed_64_f32",
            "psf_vq4_batched_equilibrate_fwht64",
            "psf_batched_int4_ascar_gemm",
            "psf_batched_vq4_int4scar_gemm",
            "psf_rmsnorm_adaln_f32",
            "psf_part_from_point_attn_hd32",
            "psf_point_from_part_attn_hd32",
            "psf_add_silu_f32",
            "psf_swiglu_mul_f32",
            "psf_gated_residual_add_f32",
            "psf_euler_step_cfg_f32",
        ):
            self.pipelines[ep] = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": self.shader_mod, "entry_point": ep},
            )
        self._compiled_schedules: Dict[Tuple[int, int], List[Tuple[Any, Any, int, int, int]]] = {}

        U = wgpu.BufferUsage
        shard_bytes = (self.bundle_dir / self.manifest["shard_file"]).read_bytes()
        self.shard_buf = self.device.create_buffer_with_data(data=shard_bytes, usage=U.STORAGE)

        c = self.cfg
        n = c.num_points
        m = c.num_parts
        d = c.d_model
        ff = c.d_ff

        x0_off_bytes = int(self.manifest["x0_f32_u32_off"]) * 4
        x0_bytes = shard_bytes[x0_off_bytes : x0_off_bytes + n * 3 * 4]
        self.x0_buf = self.device.create_buffer_with_data(data=x0_bytes, usage=U.STORAGE)

        x0_np = np.frombuffer(x0_bytes, dtype=np.float32).reshape(n, 3)
        init_x = np.concatenate([x0_np, np.full((n, 1), 0.5, dtype=np.float32)], axis=-1).astype(np.float32)
        self._init_x_bytes = init_x.tobytes()

        self.dummy_u32_buf = self.device.create_buffer(size=16, usage=U.STORAGE)
        self.dummy_f32_buf = self.device.create_buffer(size=16, usage=U.STORAGE)

        self.z_cond_buf = self.device.create_buffer(size=c.cond_in_dim * 4, usage=U.STORAGE | U.COPY_DST)
        self.z_null_buf = self.device.create_buffer_with_data(
            data=np.zeros((c.cond_in_dim,), dtype=np.float32).tobytes(),
            usage=U.STORAGE,
        )
        self.t_emb_buf = self.device.create_buffer(size=64 * 4, usage=U.STORAGE)
        self.cond_a_buf = self.device.create_buffer(size=c.cond_dim * 4, usage=U.STORAGE)
        self.cond_b_buf = self.device.create_buffer(size=c.cond_dim * 4, usage=U.STORAGE)
        self.c_global_buf = self.device.create_buffer(size=c.cond_dim * 4, usage=U.STORAGE)
        self.c_silu_buf = self.device.create_buffer(size=c.cond_dim * 4, usage=U.STORAGE)
        self.mod_buf = self.device.create_buffer(size=6 * d * 4, usage=U.STORAGE)

        self.x_cur_buf = self.device.create_buffer(size=n * 4 * 4, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC)
        self.feat64_buf = self.device.create_buffer(size=n * 64 * 4, usage=U.STORAGE)
        self.h_pts_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.parts_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)

        max_batch_in = max(c.cond_in_dim, n * max(d, ff, 64))
        max_rank = max(c.scar_rank, c.cond_scar_rank, c.head_scar_rank, 8)
        self.vq4_rot_buf = self.device.create_buffer(size=max_batch_in * 4, usage=U.STORAGE)
        self.ascar_u_buf = self.device.create_buffer(size=n * max_rank * 4, usage=U.STORAGE)

        self.part_n_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.pt_n_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.part_q_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.pt_k_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.pt_v_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.part_attn_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.part_proj_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.part_gate_buf = self.device.create_buffer(size=m * ff * 4, usage=U.STORAGE)
        self.part_up_buf = self.device.create_buffer(size=m * ff * 4, usage=U.STORAGE)
        self.part_act_buf = self.device.create_buffer(size=m * ff * 4, usage=U.STORAGE)

        self.pt_q_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.part_k_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.part_v_buf = self.device.create_buffer(size=m * d * 4, usage=U.STORAGE)
        self.pt_attn_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.pt_proj_buf = self.device.create_buffer(size=n * d * 4, usage=U.STORAGE)
        self.pt_gate_buf = self.device.create_buffer(size=n * ff * 4, usage=U.STORAGE)
        self.pt_up_buf = self.device.create_buffer(size=n * ff * 4, usage=U.STORAGE)
        self.pt_act_buf = self.device.create_buffer(size=n * ff * 4, usage=U.STORAGE)

        self.v_cond_buf = self.device.create_buffer(size=n * 4 * 4, usage=U.STORAGE)
        self.v_uncond_buf = self.device.create_buffer(size=n * 4 * 4, usage=U.STORAGE)
        self.read_buf = self.device.create_buffer(size=n * 4 * 4, usage=U.MAP_READ | U.COPY_DST)

    def _make_params_buf(
        self,
        p0: int = 0,
        p1: int = 0,
        p2: int = 0,
        p3: int = 0,
        p4: int = 0,
        p5: int = 0,
        p6: int = 0,
        p7: int = 0,
        f0: float = 0.0,
        f1: float = 0.0,
        f2: float = 0.0,
        f3: float = 0.0,
    ) -> Any:
        raw = struct.pack("<8I4f", p0, p1, p2, p3, p4, p5, p6, p7, f0, f1, f2, f3)
        return self.device.create_buffer_with_data(data=raw, usage=self.wgpu.BufferUsage.UNIFORM)

    def _compile_op(
        self,
        ops: List[Tuple[Any, Any, int, int, int]],
        ep: str,
        params_buf: Any,
        bindings: Dict[int, Any],
        gx: int,
        gy: int = 1,
        gz: int = 1,
    ) -> None:
        pipe = self.pipelines[ep]
        bgl = pipe.get_bind_group_layout(0)
        entries = [{"binding": 0, "resource": {"buffer": params_buf, "offset": 0, "size": 48}}]
        for b_idx, buf in bindings.items():
            entries.append({"binding": b_idx, "resource": {"buffer": buf, "offset": 0, "size": buf.size}})
        bg = self.device.create_bind_group(layout=bgl, entries=entries)
        ops.append((pipe, bg, gx, gy, gz))

    def _compile_scar_linear(
        self,
        ops: List[Tuple[Any, Any, int, int, int]],
        spec: Dict[str, Any],
        x_in_buf: Any,
        y_out_buf: Any,
        batch_size: int,
    ) -> None:
        out_dim = int(spec["out_dim"])
        in_dim = int(spec["in_dim"])
        rank = int(spec["rank"])

        p_rot = self._make_params_buf(p0=in_dim, p1=int(spec["d_eq_u32_off"]), p2=batch_size)
        self._compile_op(
            ops,
            "psf_vq4_batched_equilibrate_fwht64",
            p_rot,
            {1: self.shard_buf, 2: x_in_buf, 5: self.vq4_rot_buf},
            in_dim >> 6,
            batch_size,
        )

        if rank > 0:
            p_asc = self._make_params_buf(
                p0=rank,
                p1=in_dim,
                p2=batch_size,
                p3=int(spec["ascar_packed_u32_off"]),
                p4=int(spec["ascar_scales_u32_off"]),
            )
            self._compile_op(
                ops,
                "psf_batched_int4_ascar_gemm",
                p_asc,
                {1: self.shard_buf, 2: x_in_buf, 5: self.ascar_u_buf},
                (batch_size * rank + 63) >> 6,
                1,
            )

        p_vq = self._make_params_buf(
            p0=out_dim,
            p1=in_dim,
            p2=rank,
            p3=int(spec["vq_codes_u32_off"]),
            p4=int(spec["vq_scales_u32_off"]),
            p5=int(spec["lut256_u32_off"]),
            p6=int(spec.get("bscar_u32_off", 0)),
            p7=batch_size,
        )
        self._compile_op(
            ops,
            "psf_batched_vq4_int4scar_gemm",
            p_vq,
            {
                1: self.shard_buf,
                2: self.vq4_rot_buf,
                3: self.ascar_u_buf if rank > 0 else self.dummy_f32_buf,
                5: y_out_buf,
            },
            (out_dim + 63) >> 6,
            (batch_size + 7) >> 3,
        )

    def _compile_velocity_pass(
        self,
        ops: List[Tuple[Any, Any, int, int, int]],
        z_buf: Any,
        tau_val: float,
        v_out_buf: Any,
    ) -> None:
        c = self.cfg
        n = c.num_points
        m = c.num_parts
        d = c.d_model
        ff = c.d_ff
        eps = c.rms_norm_eps
        cond = self.manifest["cond"]

        # 1. Time embed + Conditioning encoder
        p_t = self._make_params_buf(f0=tau_val)
        self._compile_op(ops, "psf_time_embed_64_f32", p_t, {5: self.t_emb_buf}, 1)

        self._compile_scar_linear(ops, cond["cond_z"], z_buf, self.cond_a_buf, 1)
        self._compile_scar_linear(ops, cond["cond_t"], self.t_emb_buf, self.cond_b_buf, 1)

        p_csilu = self._make_params_buf(p0=c.cond_dim, p1=1)
        self._compile_op(
            ops,
            "psf_add_silu_f32",
            p_csilu,
            {2: self.cond_a_buf, 3: self.cond_b_buf, 5: self.c_silu_buf},
            (c.cond_dim + 255) >> 8,
        )

        p_cnorm = self._make_params_buf(
            p0=c.cond_dim,
            p1=int(cond["cond_norm_u32_off"]),
            p2=1,
            p3=0,
            f0=eps,
        )
        self._compile_op(
            ops,
            "psf_rmsnorm_adaln_f32",
            p_cnorm,
            {1: self.shard_buf, 2: self.c_silu_buf, 3: self.dummy_f32_buf, 5: self.cond_a_buf},
            1,
        )
        self._compile_scar_linear(ops, cond["cond_mlp"], self.cond_a_buf, self.c_global_buf, 1)

        p_gsilu = self._make_params_buf(p0=c.cond_dim, p1=0)
        self._compile_op(
            ops,
            "psf_add_silu_f32",
            p_gsilu,
            {2: self.c_global_buf, 3: self.dummy_f32_buf, 5: self.c_silu_buf},
            (c.cond_dim + 255) >> 8,
        )
        self._compile_scar_linear(ops, cond["part_seed"], self.c_silu_buf, self.parts_buf, 1)

        # 2. Point stem (`[N, 64] -> [N, D]`)
        p_feat = self._make_params_buf(p0=n, f0=tau_val)
        self._compile_op(
            ops,
            "psf_build_point_features_f32",
            p_feat,
            {2: self.x_cur_buf, 3: self.x0_buf, 5: self.feat64_buf},
            (n + 63) >> 6,
        )
        self._compile_scar_linear(ops, self.manifest["point_stem"], self.feat64_buf, self.h_pts_buf, n)

        inv_sqrt_hd = 1.0 / math.sqrt(float(c.head_dim))

        # 3. Induced-set blocks
        for blk in self.manifest["blocks"]:
            self._compile_scar_linear(ops, blk["ada_mod"], self.c_silu_buf, self.mod_buf, 1)

            # Stage A: `Part <- Point`
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["part_ln1_u32_off"]), p2=m, p3=0, f0=eps),
                {1: self.shard_buf, 2: self.parts_buf, 3: self.dummy_f32_buf, 5: self.part_n_buf},
                m,
            )
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["pt_kv_ln_u32_off"]), p2=n, p3=0, f0=eps),
                {1: self.shard_buf, 2: self.h_pts_buf, 3: self.dummy_f32_buf, 5: self.pt_n_buf},
                n,
            )
            self._compile_scar_linear(ops, blk["part_q"], self.part_n_buf, self.part_q_buf, m)
            self._compile_scar_linear(ops, blk["part_k_from_pt"], self.pt_n_buf, self.pt_k_buf, n)
            self._compile_scar_linear(ops, blk["part_v_from_pt"], self.pt_n_buf, self.pt_v_buf, n)

            self._compile_op(
                ops,
                "psf_part_from_point_attn_hd32",
                self._make_params_buf(p0=m, p1=n, p2=c.num_heads, p3=c.head_dim, f0=inv_sqrt_hd),
                {2: self.part_q_buf, 3: self.pt_k_buf, 4: self.pt_v_buf, 5: self.part_attn_buf},
                m,
                c.num_heads,
            )
            self._compile_scar_linear(ops, blk["part_o"], self.part_attn_buf, self.part_proj_buf, m)
            self._compile_op(
                ops,
                "psf_gated_residual_add_f32",
                self._make_params_buf(p0=m * d, p1=d, p2=0, p3=0),
                {2: self.part_proj_buf, 3: self.dummy_f32_buf, 5: self.parts_buf},
                (m * d + 255) >> 8,
            )

            # Part SwiGLU
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["part_ln2_u32_off"]), p2=m, p3=0, f0=eps),
                {1: self.shard_buf, 2: self.parts_buf, 3: self.dummy_f32_buf, 5: self.part_n_buf},
                m,
            )
            self._compile_scar_linear(ops, blk["part_gate"], self.part_n_buf, self.part_gate_buf, m)
            self._compile_scar_linear(ops, blk["part_up"], self.part_n_buf, self.part_up_buf, m)
            self._compile_op(
                ops,
                "psf_swiglu_mul_f32",
                self._make_params_buf(p0=m * ff),
                {2: self.part_gate_buf, 3: self.part_up_buf, 5: self.part_act_buf},
                (m * ff + 255) >> 8,
            )
            self._compile_scar_linear(ops, blk["part_down"], self.part_act_buf, self.part_proj_buf, m)
            self._compile_op(
                ops,
                "psf_gated_residual_add_f32",
                self._make_params_buf(p0=m * d, p1=d, p2=0, p3=0),
                {2: self.part_proj_buf, 3: self.dummy_f32_buf, 5: self.parts_buf},
                (m * d + 255) >> 8,
            )

            # Stage B: `Point <- Part` (AdaLN slot 0/1, gate slot 2)
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["pt_ln1_u32_off"]), p2=n, p3=1, p4=0, f0=eps),
                {1: self.shard_buf, 2: self.h_pts_buf, 3: self.mod_buf, 5: self.pt_n_buf},
                n,
            )
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["part_kv_ln_u32_off"]), p2=m, p3=0, f0=eps),
                {1: self.shard_buf, 2: self.parts_buf, 3: self.dummy_f32_buf, 5: self.part_n_buf},
                m,
            )
            self._compile_scar_linear(ops, blk["pt_q"], self.pt_n_buf, self.pt_q_buf, n)
            self._compile_scar_linear(ops, blk["pt_k_from_part"], self.part_n_buf, self.part_k_buf, m)
            self._compile_scar_linear(ops, blk["pt_v_from_part"], self.part_n_buf, self.part_v_buf, m)

            self._compile_op(
                ops,
                "psf_point_from_part_attn_hd32",
                self._make_params_buf(p0=n, p1=m, p2=c.num_heads, p3=c.head_dim, f0=inv_sqrt_hd),
                {2: self.pt_q_buf, 3: self.part_k_buf, 4: self.part_v_buf, 5: self.pt_attn_buf},
                (n + 63) >> 6,
                c.num_heads,
            )
            self._compile_scar_linear(ops, blk["pt_o"], self.pt_attn_buf, self.pt_proj_buf, n)
            self._compile_op(
                ops,
                "psf_gated_residual_add_f32",
                self._make_params_buf(p0=n * d, p1=d, p2=1, p3=2),
                {2: self.pt_proj_buf, 3: self.mod_buf, 5: self.h_pts_buf},
                (n * d + 255) >> 8,
            )

            # Stage C: Point SwiGLU (AdaLN slot 3/4, gate slot 5)
            self._compile_op(
                ops,
                "psf_rmsnorm_adaln_f32",
                self._make_params_buf(p0=d, p1=int(blk["pt_ln2_u32_off"]), p2=n, p3=1, p4=3, f0=eps),
                {1: self.shard_buf, 2: self.h_pts_buf, 3: self.mod_buf, 5: self.pt_n_buf},
                n,
            )
            self._compile_scar_linear(ops, blk["pt_gate"], self.pt_n_buf, self.pt_gate_buf, n)
            self._compile_scar_linear(ops, blk["pt_up"], self.pt_n_buf, self.pt_up_buf, n)
            self._compile_op(
                ops,
                "psf_swiglu_mul_f32",
                self._make_params_buf(p0=n * ff),
                {2: self.pt_gate_buf, 3: self.pt_up_buf, 5: self.pt_act_buf},
                (n * ff + 255) >> 8,
            )
            self._compile_scar_linear(ops, blk["pt_down"], self.pt_act_buf, self.pt_proj_buf, n)
            self._compile_op(
                ops,
                "psf_gated_residual_add_f32",
                self._make_params_buf(p0=n * d, p1=d, p2=1, p3=5),
                {2: self.pt_proj_buf, 3: self.mod_buf, 5: self.h_pts_buf},
                (n * d + 255) >> 8,
            )

        # 4. Final Norm + Velocity Head (`[N, D] -> [N, 4]`)
        self._compile_op(
            ops,
            "psf_rmsnorm_adaln_f32",
            self._make_params_buf(p0=d, p1=int(self.manifest["final_norm_u32_off"]), p2=n, p3=0, f0=eps),
            {1: self.shard_buf, 2: self.h_pts_buf, 3: self.dummy_f32_buf, 5: self.pt_n_buf},
            n,
        )
        self._compile_scar_linear(ops, self.manifest["vel_head"], self.pt_n_buf, v_out_buf, n)

    def _get_schedule(self, num_steps: int, guidance_scale: float) -> List[Tuple[Any, Any, int, int, int]]:
        key = (int(num_steps), int(round(guidance_scale * 10000)))
        cached = self._compiled_schedules.get(key)
        if cached is not None:
            return cached

        ops: List[Tuple[Any, Any, int, int, int]] = []
        n4 = self.cfg.num_points * 4
        dt = 1.0 / float(max(1, num_steps))
        use_cfg = abs(guidance_scale - 1.0) > 1e-5

        for step in range(num_steps):
            tau_val = float(step) * dt
            self._compile_velocity_pass(ops, self.z_cond_buf, tau_val, self.v_cond_buf)
            if use_cfg:
                self._compile_velocity_pass(ops, self.z_null_buf, tau_val, self.v_uncond_buf)
            is_final = 1 if (step == num_steps - 1) else 0
            p_euler = self._make_params_buf(
                p0=n4,
                p1=1 if use_cfg else 0,
                p2=is_final,
                f0=dt,
                f1=guidance_scale,
            )
            self._compile_op(
                ops,
                "psf_euler_step_cfg_f32",
                p_euler,
                {
                    2: self.v_cond_buf,
                    3: self.v_uncond_buf if use_cfg else self.dummy_f32_buf,
                    5: self.x_cur_buf,
                },
                (n4 + 255) >> 8,
            )
        self._compiled_schedules[key] = ops
        return ops

    def sample_euler(
        self,
        z_cond_np: np.ndarray,
        num_steps: int = 4,
        guidance_scale: float = 1.75,
    ) -> np.ndarray:
        """Runs deterministic `num_steps` Rectified Flow ODE integration 100% on physical WebGPU."""
        z_f32 = np.asarray(z_cond_np, dtype=np.float32).reshape(self.cfg.cond_in_dim)
        self.device.queue.write_buffer(self.z_cond_buf, 0, z_f32.tobytes())
        self.device.queue.write_buffer(self.x_cur_buf, 0, self._init_x_bytes)

        ops = self._get_schedule(num_steps, guidance_scale)
        n4 = self.cfg.num_points * 4
        enc = self.device.create_command_encoder()
        cpass = enc.begin_compute_pass()
        for pipe, bg, gx, gy, gz in ops:
            cpass.set_pipeline(pipe)
            cpass.set_bind_group(0, bg)
            cpass.dispatch_workgroups(gx, gy, gz)
        cpass.end()
        enc.copy_buffer_to_buffer(self.x_cur_buf, 0, self.read_buf, 0, n4 * 4)
        self.device.queue.submit([enc.finish()])

        self.read_buf.map_sync(self.wgpu.MapMode.READ)
        raw = bytes(self.read_buf.read_mapped())
        self.read_buf.unmap()
        return np.frombuffer(raw, dtype=np.float32).reshape(self.cfg.num_points, 4).copy()
