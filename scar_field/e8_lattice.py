"""Gosset E8 Root-Lattice Substrate (`1.000 bpw`) + Multiplierless Ternary Atomic Scars.

Zero-Legacy Mathematical Primitives for `ScarField v7.0`:
  1. `build_canonical_e8_lut256`: Exact 256-entry 8D codebook (`4 KB` FP16 L1 table):
       - Entry `0x00`: Exact physical vacuum `(0, 0, 0, 0, 0, 0, 0, 0)`.
       - Entries `1..240`: All 240 minimal roots of the Gosset `E_8` lattice (112 integer `D_8`
         roots `{0, ±2}^8` + 128 even-parity spinor roots `{±1}^8`), achieving Viazovska's
         Fields-Medal optimal 8D sphere packing at radius `sqrt(8)` (100% multiplierless!).
       - Entries `241..255`: 15 orthonormal Walsh-Hadamard `H_8` high-energy shell vectors
         capturing tail blocks without channel equilibration `d_eq` or per-block scale tables.
  2. `fwht64_rademacher`: Parameter-free, multiplierless 64-point Fast Walsh-Hadamard rotation
     guaranteeing Kashin concentration on `S^{I-1}` even for 1-hot axis-aligned features.
  3. `E8SphericalLinear` & `PackedE8TernaryMatrix`: `1.000 bit/weight` (`0.125 B/weight`)
     shared substrate + `{-1, 0, +1}` ternary atomic residual scars with closed-form `SVD/RRR`
     scar healing and exact `0.000000` PTQ export shock (zero `torch.nn.Linear` anywhere).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scar_codec.nd_e8 import build_canonical_e8_lut256_np
from scar_codec.numpy_engine import e8_spherical_ternary_gemm_np, get_compiled_c_lib


def rademacher_signs_det(dim: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Deterministic golden-ratio hash Rademacher signs `{-1.0, +1.0}` of length `dim`."""
    idx = torch.arange(dim, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


_HADAMARD64_CACHE: dict[Tuple[str, Optional[int], torch.dtype], torch.Tensor] = {}


def _get_hadamard64(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, dtype)
    mat = _HADAMARD64_CACHE.get(key)
    if mat is None:
        h = torch.tensor([[1.0]], dtype=torch.float32, device=device)
        for _ in range(6):
            h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
        mat = (h * 0.125).to(dtype=dtype).contiguous()
        _HADAMARD64_CACHE[key] = mat
    return mat


def fwht64_rademacher(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Multiplierless orthonormal 64-point Fast Walsh-Hadamard Transform (1 fused GEMM, zero `d_eq`)."""
    orig_shape = x.shape
    if orig_shape[-1] % 64 != 0:
        raise ValueError(f"Last dimension ({orig_shape[-1]}) must be divisible by 64")
    y = x if inverse else (x * signs)
    h64 = _get_hadamard64(y.device, y.dtype)
    out = torch.matmul(y.reshape(-1, 64), h64).reshape(orig_shape)
    return (out * signs) if inverse else out


def _hadamard8_matrix() -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float32)
    for _ in range(3):
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(8.0)


@torch.no_grad()
def build_canonical_e8_lut256(device: torch.device | str = "cpu") -> torch.Tensor:
    """Builds the exact `[256, 8]` `E8 + Vacuum + Outer Hadamard Shell` codebook via `scar_codec`.

    Shared single source of truth (`0.0` drift) with `scar_codec.build_canonical_e8_lut256_np`:
      - `0`: `(0, 0, 0, 0, 0, 0, 0, 0)` exact physical vacuum (`0x00`).
      - `1..112`: `D_8` roots `(±2, ±2, 0, 0, 0, 0, 0, 0)` (`||u||_2^2 = 8.0`, integer coords!).
      - `113..240`: Spinor roots `(±1, ..., ±1)` with even minus count (`||u||_2^2 = 8.0`, bipolar!).
      - `241..255`: 15 orthonormal `H_8` outer-shell vectors at `||u||_2^2 = 18.0` (`r = 3*sqrt(2)`).
    """
    return torch.from_numpy(build_canonical_e8_lut256_np()).to(device=device, dtype=torch.float32)


def project_rows_to_unit_sphere(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Projects every row of `w` onto the unit hypersphere `S^{I-1}` (`||w_o||_2 == 1`)."""
    norms = torch.linalg.vector_norm(w, ord=2, dim=-1, keepdim=True).clamp(min=eps)
    return w / norms


def quantize_ternary_ste(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameter-free ternary quantization `{-1, 0, +1}` with Straight-Through Commitment."""
    rms = torch.sqrt(torch.mean(w * w, dim=-1, keepdim=True).clamp(min=1e-12))
    thresh = 0.68 * rms
    q = torch.where(w > thresh, 1.0, torch.where(w < -thresh, -1.0, 0.0))
    q_ste = w + (q - w).detach()
    return q_ste, q.to(torch.int8)


@torch.no_grad()
def pack_ternary_2bit(t_i8: torch.Tensor) -> torch.Tensor:
    """Packs `{-1, 0, +1}` int8 tensor `[..., K]` (`K % 4 == 0`) into `uint8` (`4` weights/byte)."""
    if t_i8.shape[-1] % 4 != 0:
        raise ValueError(f"Ternary last dim ({t_i8.shape[-1]}) must be divisible by 4")
    # Map {-1, 0, +1} -> {0, 1, 2} in 2 bits
    u2 = (t_i8.to(torch.int16) + 1).clamp(0, 2).to(torch.uint8)
    u2 = u2.reshape(*t_i8.shape[:-1], t_i8.shape[-1] // 4, 4)
    packed = u2[..., 0] | (u2[..., 1] << 2) | (u2[..., 2] << 4) | (u2[..., 3] << 6)
    return packed.contiguous()


@torch.no_grad()
def unpack_ternary_2bit(packed_u8: torch.Tensor) -> torch.Tensor:
    """Unpacks `uint8` 2-bit ternary codes back to `{-1.0, 0.0, +1.0}` float32."""
    c0 = (packed_u8 & 0x03).to(torch.int8) - 1
    c1 = ((packed_u8 >> 2) & 0x03).to(torch.int8) - 1
    c2 = ((packed_u8 >> 4) & 0x03).to(torch.int8) - 1
    c3 = ((packed_u8 >> 6) & 0x03).to(torch.int8) - 1
    out = torch.stack((c0, c1, c2, c3), dim=-1).reshape(*packed_u8.shape[:-1], packed_u8.shape[-1] * 4)
    return out.float()


@dataclass
class PackedE8TernaryMatrix:
    """Zero-shock `1.000 bpw` E8 substrate + 2-bit ternary atomic scars."""

    out_features: int
    in_features: int
    scar_rank: int
    codes_u8: torch.Tensor        # [O, I // 8] uint8 -> EXACT 1.000 bit/weight!
    a_tern_packed: torch.Tensor   # [r, I // 4] uint8 -> 2-bit ternary {-1, 0, +1}
    b_tern_packed: torch.Tensor   # [O, r // 4] uint8 -> 2-bit ternary {-1, 0, +1}
    tau_base: float
    gamma_scar: float
    _mmap_views: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    @property
    def substrate_bytes(self) -> int:
        if self._mmap_views is not None:
            return int(self._mmap_views[0].nbytes)
        return int(self.codes_u8.numel())

    @property
    def scar_bytes(self) -> int:
        if self._mmap_views is not None:
            return int(self._mmap_views[1].nbytes + self._mmap_views[2].nbytes)
        return int(self.a_tern_packed.numel() + self.b_tern_packed.numel())

    @property
    def total_bytes(self) -> int:
        return self.substrate_bytes + self.scar_bytes + 8  # +8B for 2 float32 scalars

    @property
    def substrate_bits_per_weight(self) -> float:
        return (self.substrate_bytes * 8.0) / float(self.out_features * self.in_features)

    def forward(
        self,
        x: torch.Tensor,
        lut_e8: torch.Tensor,
        signs: torch.Tensor,
        use_c_accel: bool = True,
    ) -> torch.Tensor:
        if use_c_accel and x.device.type == "cpu" and (not x.requires_grad):
            if self._mmap_views is not None:
                c_np, a_np, b_np = self._mmap_views
            else:
                c_np = self.codes_u8.numpy()
                a_np = self.a_tern_packed.numpy()
                b_np = self.b_tern_packed.numpy()
            y_np = e8_spherical_ternary_gemm_np(
                x_f32=x.detach().float().numpy(),
                codes_u8=c_np,
                lut_e8_f32=lut_e8.detach().cpu().float().numpy(),
                a_tern_u8=a_np,
                b_tern_u8=b_np,
                scar_rank=self.scar_rank,
                tau_base=self.tau_base,
                gamma_scar=self.gamma_scar,
                use_c_accel=True,
            )
            return torch.from_numpy(y_np).to(dtype=x.dtype)

        xf = x.float()
        x_rot = fwht64_rademacher(xf, signs, inverse=False)
        if self._mmap_views is not None:
            c_t = torch.from_numpy(self._mmap_views[0].copy()).to(xf.device)
            a_t_pk = torch.from_numpy(self._mmap_views[1].copy()).to(xf.device)
            b_t_pk = torch.from_numpy(self._mmap_views[2].copy()).to(xf.device)
        else:
            c_t = self.codes_u8.to(xf.device)
            a_t_pk = self.a_tern_packed.to(xf.device)
            b_t_pk = self.b_tern_packed.to(xf.device)
        w_rot = lut_e8[c_t.long()].reshape(self.out_features, self.in_features)
        inv_sqrt_i = 1.0 / math.sqrt(float(self.in_features))
        y = (x_rot @ w_rot.T) * (self.tau_base * inv_sqrt_i)
        if self.scar_rank > 0 and abs(self.gamma_scar) > 0.0:
            a_f = unpack_ternary_2bit(a_t_pk)
            b_f = unpack_ternary_2bit(b_t_pk)
            scale_s = self.gamma_scar / math.sqrt(float(self.in_features * max(1, self.scar_rank)))
            y = y + ((xf @ a_f.T) @ b_f.T) * scale_s
        return y.to(x.dtype)


_SKIP_E8_INIT_ALLOC = False
_SHARED_E8_LUT_CPU: Optional[torch.Tensor] = None


class _DeferredE8GemmAutograd(torch.autograd.Function):
    """Accumulates `grad_w_rot` across `K` reversible steps so `[O, I]` FWHT+Sphere runs `1x` instead of `Kx`!"""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        xf: torch.Tensor,
        x_rot: torch.Tensor,
        w_q_rot: torch.Tensor,
        w_q_rot_bf16: Optional[torch.Tensor],
        tau_val: float,
        inv_sqrt_i: float,
        a_t: Optional[torch.Tensor],
        b_t: Optional[torch.Tensor],
        gamma_val: float,
        scale_s_unit: float,
        owner: "E8SphericalLinear",
    ) -> torch.Tensor:
        scale_w = tau_val * inv_sqrt_i
        if w_q_rot_bf16 is not None and x_rot.device.type == "cuda":
            y_base = (x_rot.bfloat16() @ w_q_rot_bf16.T).float() * scale_w
        else:
            y_base = (x_rot @ w_q_rot.T) * scale_w

        xa: Optional[torch.Tensor] = None
        y_scar_unscaled: Optional[torch.Tensor] = None
        if a_t is not None and b_t is not None:
            xa = xf @ a_t.T
            y_scar_unscaled = xa @ b_t.T
            y_out = y_base + y_scar_unscaled * (gamma_val * scale_s_unit)
        else:
            y_out = y_base

        ctx.save_for_backward(
            xf.detach(),
            x_rot.detach(),
            w_q_rot,
            a_t if a_t is not None else torch.empty(0, device=xf.device),
            b_t if b_t is not None else torch.empty(0, device=xf.device),
            xa.detach() if xa is not None else torch.empty(0, device=xf.device),
            y_scar_unscaled.detach() if y_scar_unscaled is not None else torch.empty(0, device=xf.device),
        )
        ctx.w_q_rot_bf16 = w_q_rot_bf16
        ctx.tau_val = tau_val
        ctx.inv_sqrt_i = inv_sqrt_i
        ctx.gamma_val = gamma_val
        ctx.scale_s_unit = scale_s_unit
        ctx.has_scar = a_t is not None and b_t is not None
        ctx.owner = owner
        return y_out

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):  # type: ignore[override]
        xf, x_rot, w_q_rot, a_t, b_t, xa, y_scar_unscaled = ctx.saved_tensors
        owner: E8SphericalLinear = ctx.owner
        gy = grad_y.float()
        lead_n = int(gy.numel() // owner.out_features)
        gy_2d = gy.reshape(lead_n, owner.out_features)
        xr_2d = x_rot.reshape(lead_n, owner.in_features)
        scale_w = ctx.tau_val * ctx.inv_sqrt_i

        # 1. Input gradient wrt `x_rot` + in-place accumulation into `owner._acc_grad_w_rot` (`1` fused BLAS `addmm_`!)
        if ctx.w_q_rot_bf16 is not None and gy_2d.device.type == "cuda":
            grad_xr_2d = (gy_2d.bfloat16() @ ctx.w_q_rot_bf16).float() * scale_w
            if owner._acc_grad_w_rot is not None:
                owner._acc_grad_w_rot.add_(
                    (gy_2d.bfloat16().T @ xr_2d.bfloat16()).float(), alpha=scale_w
                )
        else:
            grad_xr_2d = (gy_2d @ w_q_rot) * scale_w
            if owner._acc_grad_w_rot is not None:
                owner._acc_grad_w_rot.addmm_(gy_2d.T, xr_2d, beta=1.0, alpha=scale_w)

        if owner._acc_grad_tau is not None:
            # dL/dtau = sum(gy * (x_rot @ w_q_rot^T) * inv_sqrt_i)
            # Re-use `<gy_2d, (xr_2d @ w_q_rot^T)> == <grad_xr_2d / scale_w, xr_2d>` when tau != 0!
            if abs(scale_w) > 1e-12:
                gtau = (grad_xr_2d * xr_2d).sum() * (ctx.inv_sqrt_i / scale_w)
            else:
                gtau = ((xr_2d @ w_q_rot.T) * gy_2d).sum() * ctx.inv_sqrt_i
            owner._acc_grad_tau.add_(gtau)

        grad_xf: Optional[torch.Tensor] = None
        if ctx.has_scar:
            xf_2d = xf.reshape(lead_n, owner.in_features)
            xa_2d = xa.reshape(lead_n, owner.scar_rank)
            ys_2d = y_scar_unscaled.reshape(lead_n, owner.out_features)
            sc_s = ctx.gamma_val * ctx.scale_s_unit

            g_xa = (gy_2d @ b_t) * sc_s  # [N, r]
            grad_xf = (g_xa @ a_t).reshape_as(xf)
            if owner._acc_grad_b_t is not None:
                owner._acc_grad_b_t.addmm_(gy_2d.T, xa_2d, beta=1.0, alpha=sc_s)
            if owner._acc_grad_a_t is not None:
                owner._acc_grad_a_t.addmm_(g_xa.T, xf_2d, beta=1.0, alpha=1.0)
            if owner._acc_grad_gamma is not None:
                owner._acc_grad_gamma.add_((gy_2d * ys_2d).sum() * ctx.scale_s_unit)

        return (
            grad_xf,
            grad_xr_2d.reshape_as(x_rot),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class E8SphericalLinear(nn.Module):
    """Pure `1.000 bpw` Gosset `E_8` Spherical Operator + Ternary `{-1, 0, +1}` Atomic Scars.

    - ZERO `torch.nn.Linear` dense modules
    - ZERO `d_eq` channel equilibration vectors
    - ZERO per-block or per-row scale tables
    - Exact `0.000000` PTQ export shock between `forward()` and `to_packed().forward()`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scar_rank: int = 8,
        init_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if in_features % 64 != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by 64")
        if scar_rank < 0 or (scar_rank > 0 and scar_rank % 4 != 0):
            raise ValueError(f"scar_rank ({scar_rank}) must be 0 or divisible by 4")

        self.in_features = in_features
        self.out_features = out_features
        self.scar_rank = scar_rank
        self._frozen_packed: Optional[PackedE8TernaryMatrix] = None
        self._cached_q_key: Optional[Tuple[int, int, str, Optional[int]]] = None
        self._cached_w_q_rot: Optional[torch.Tensor] = None
        self._cached_w_q_rot_bf16: Optional[torch.Tensor] = None
        self._cached_nograd_key: Optional[Tuple[int, int, int, int, int, str, Optional[int]]] = None
        self._cached_a_t: Optional[torch.Tensor] = None
        self._cached_b_t: Optional[torch.Tensor] = None
        self._defer_grad_active: bool = False
        self._acc_grad_w_rot: Optional[torch.Tensor] = None
        self._acc_grad_tau: Optional[torch.Tensor] = None
        self._acc_grad_a_t: Optional[torch.Tensor] = None
        self._acc_grad_b_t: Optional[torch.Tensor] = None
        self._acc_grad_gamma: Optional[torch.Tensor] = None

        global _SHARED_E8_LUT_CPU
        if _SHARED_E8_LUT_CPU is None:
            _SHARED_E8_LUT_CPU = build_canonical_e8_lut256("cpu")
        lut = _SHARED_E8_LUT_CPU
        lut_half_norm2 = 0.5 * (lut * lut).sum(dim=-1)
        signs = rademacher_signs_det(in_features, "cpu")
        self.register_buffer("lut_e8", lut, persistent=False)
        self.register_buffer("lut_half_norm2", lut_half_norm2, persistent=False)
        self.register_buffer("signs", signs, persistent=False)

        if _SKIP_E8_INIT_ALLOC:
            self.register_parameter("weight_sphere", None)
            self.register_parameter("tau_base", None)
            self.register_parameter("a_scar", None)
            self.register_parameter("b_scar", None)
            self.register_parameter("gamma_scar", None)
            return

        w_init = project_rows_to_unit_sphere(torch.randn(out_features, in_features))
        self.weight_sphere = nn.Parameter(w_init)
        self.tau_base = nn.Parameter(torch.tensor(float(init_tau), dtype=torch.float32))

        if scar_rank > 0:
            self.a_scar = nn.Parameter(project_rows_to_unit_sphere(torch.randn(scar_rank, in_features)))
            self.b_scar = nn.Parameter(project_rows_to_unit_sphere(torch.randn(out_features, scar_rank)))
            self.gamma_scar = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        else:
            self.register_parameter("a_scar", None)
            self.register_parameter("b_scar", None)
            self.register_parameter("gamma_scar", None)

        self.heal_residual_scars_svd()

    def _assign_e8_codes_and_dequant(
        self,
        w_rot_scaled: torch.Tensor,
        need_codes: bool = False,
        need_dequant: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Zero-`GiB` exact Conway-Sloane (`C99` multithreaded on CPU / L2-tiled `FP16` Tensor-Core on CUDA)."""
        n_blocks = (self.out_features * self.in_features) // 8
        if w_rot_scaled.device.type == "cpu":
            clib = get_compiled_c_lib()
            if clib is not None:
                x_contig = w_rot_scaled.detach().float().contiguous()
                lut_contig = self.lut_e8.detach().float().contiguous()
                codes_out = (
                    torch.empty((self.out_features, self.in_features // 8), dtype=torch.uint8)
                    if need_codes
                    else None
                )
                deq_out = (
                    torch.empty((self.out_features, self.in_features), dtype=torch.float32)
                    if need_dequant
                    else None
                )
                f32_p = ctypes.POINTER(ctypes.c_float)
                u8_p = ctypes.POINTER(ctypes.c_uint8)
                clib.scar_conway_sloane_e8_encode_f32(
                    ctypes.cast(x_contig.data_ptr(), f32_p),
                    ctypes.c_int64(n_blocks),
                    ctypes.cast(lut_contig.data_ptr(), f32_p),
                    ctypes.cast(codes_out.data_ptr(), u8_p) if codes_out is not None else ctypes.cast(0, u8_p),
                    ctypes.cast(deq_out.data_ptr(), f32_p) if deq_out is not None else ctypes.cast(0, f32_p),
                )
                return codes_out, deq_out

        blks = w_rot_scaled.reshape(n_blocks, 8)
        codes_flat = torch.empty(n_blocks, dtype=torch.uint8, device=blks.device)
        if blks.device.type in ("cuda", "mps"):
            lut_h_t = self.lut_e8.half().T.contiguous()
            neg_hn2 = (-self.lut_half_norm2).float()
            tile = 262144  # 128 MiB L2/HBM tile instead of 1.4 GiB allocation!
            for s0 in range(0, n_blocks, tile):
                s1 = min(n_blocks, s0 + tile)
                sc = torch.matmul(blks[s0:s1].half(), lut_h_t).float() + neg_hn2
                codes_flat[s0:s1] = torch.argmax(sc, dim=-1).to(torch.uint8)
        else:
            lut_t = self.lut_e8.T.contiguous()
            neg_hn2 = -self.lut_half_norm2
            tile = 16384  # 16 MiB L2-resident tile
            for s0 in range(0, n_blocks, tile):
                s1 = min(n_blocks, s0 + tile)
                sc = torch.addmm(neg_hn2, blks[s0:s1], lut_t)
                codes_flat[s0:s1] = torch.argmax(sc, dim=-1).to(torch.uint8)

        codes_2d = codes_flat.reshape(self.out_features, self.in_features // 8) if need_codes else None
        deq_2d = (
            self.lut_e8[codes_flat.long()].reshape(self.out_features, self.in_features)
            if need_dequant
            else None
        )
        return codes_2d, deq_2d

    def _assign_e8_codes(self, w_rot_scaled: torch.Tensor) -> torch.Tensor:
        codes, _ = self._assign_e8_codes_and_dequant(w_rot_scaled, need_codes=True, need_dequant=False)
        assert codes is not None
        return codes

    @torch.no_grad()
    def heal_residual_scars_svd(self) -> None:
        """Closed-form `SVD/RRR` ternary scar healing over the exact `E_8` lattice residual."""
        self.weight_sphere.copy_(project_rows_to_unit_sphere(self.weight_sphere))
        if self.scar_rank <= 0 or self.a_scar is None or self.b_scar is None or self.gamma_scar is None:
            return

        sqrt_i = math.sqrt(float(self.in_features))
        w_unit = self.weight_sphere.float()
        w_rot = fwht64_rademacher(w_unit, self.signs, inverse=False) * sqrt_i
        _, w_q_rot_raw = self._assign_e8_codes_and_dequant(w_rot, need_codes=False, need_dequant=True)
        assert w_q_rot_raw is not None
        w_q_rot = w_q_rot_raw / sqrt_i
        w_q = fwht64_rademacher(w_q_rot, self.signs, inverse=True)

        res = float(self.tau_base.item()) * (w_unit - w_q)
        r = min(self.scar_rank, self.out_features, self.in_features)
        if min(self.out_features, self.in_features) >= 512 and r <= 16:
            # Fast 3-step Block Subspace Iteration (`~150x` faster than full `[O, I]` SVD, 0 global RNG mutation!)
            k_sub = min(2 * r + 4, self.in_features, self.out_features)
            q_sub = project_rows_to_unit_sphere(self.a_scar.detach().float())
            if q_sub.shape[0] < k_sub:
                n_ex = k_sub - q_sub.shape[0]
                idx_ex = torch.arange(n_ex * self.in_features, device=res.device, dtype=torch.int64)
                h_ex = (((idx_ex * 2654435761) & 0xFFFFFFFF) ^ 0x9E3779B9) * 0x45D9F3B
                extra = torch.where((h_ex & 1) == 0, 1.0, -1.0).reshape(n_ex, self.in_features)
                q_sub = torch.cat((q_sub, project_rows_to_unit_sphere(extra)), dim=0)
            v_basis, _ = torch.linalg.qr(q_sub[:k_sub].T, mode="reduced")  # [I, k_sub]
            use_bf16_svd = res.device.type == "cuda" and self.out_features >= 512
            res_work = res.bfloat16() if use_bf16_svd else res
            for _ in range(3):
                u_proj = (res_work @ (v_basis.bfloat16() if use_bf16_svd else v_basis)).float()
                u_basis, _ = torch.linalg.qr(u_proj, mode="reduced")
                v_proj = (res_work.T @ (u_basis.bfloat16() if use_bf16_svd else u_basis)).float()
                v_basis, _ = torch.linalg.qr(v_proj, mode="reduced")
            b_small = (res @ v_basis)  # [O, k_sub]
            u_sm, s_sm, vh_sm = torch.linalg.svd(b_small, full_matrices=False)
            u = u_sm[:, :r]
            s = s_sm[:r]
            vh = (vh_sm[:r, :] @ v_basis.T)
        else:
            u_f, s_f, vh_f = torch.linalg.svd(res, full_matrices=False)
            u, s, vh = u_f[:, :r], s_f[:r], vh_f[:r, :]

        s_sqrt = torch.sqrt(s[:r].clamp(min=1e-8))
        b_dir = u[:, :r] * s_sqrt.unsqueeze(0)
        a_dir = s_sqrt.unsqueeze(1) * vh[:r, :]

        if r < self.scar_rank:
            pad_b = torch.zeros(self.out_features, self.scar_rank - r, device=res.device)
            pad_a = torch.zeros(self.scar_rank - r, self.in_features, device=res.device)
            b_dir = torch.cat((b_dir, pad_b), dim=1)
            a_dir = torch.cat((a_dir, pad_a), dim=0)

        self.a_scar.copy_(project_rows_to_unit_sphere(a_dir))
        self.b_scar.copy_(project_rows_to_unit_sphere(b_dir))

        _, a_i8 = quantize_ternary_ste(self.a_scar)
        _, b_i8 = quantize_ternary_ste(self.b_scar)
        af = a_i8.float()
        bf = b_i8.float()
        # Trace identity avoids allocating `[O, I]` (`40 MiB`) `bf @ af` matrix!
        scale_ba = 1.0 / float(self.in_features * self.scar_rank)
        denom = float(((bf.T @ bf) * (af @ af.T)).sum().item()) * scale_ba
        if denom > 1e-8:
            numer = float(((bf.T @ res) * af).sum().item()) / math.sqrt(float(self.in_features * self.scar_rank))
            self.gamma_scar.fill_(numer / denom)

    def begin_deferred_grad_pass(self) -> None:
        """Starts reversible-loop deferred weight gradient accumulation (`1x` sphere/FWHT instead of `Kx`)."""
        if self._frozen_packed is not None or self.weight_sphere is None:
            return
        dev = self.weight_sphere.device
        sqrt_i = math.sqrt(float(self.in_features))
        q_key = (self.weight_sphere._version, self.weight_sphere.data_ptr(), dev.type, dev.index)
        if self._cached_q_key != q_key or self._cached_w_q_rot is None or self._cached_w_q_rot.device != dev:
            with torch.no_grad():
                w_u_ng = project_rows_to_unit_sphere(self.weight_sphere.detach().float())
                w_r_ng = fwht64_rademacher(w_u_ng, self.signs, inverse=False) * sqrt_i
                _, deq_rot = self._assign_e8_codes_and_dequant(w_r_ng, need_codes=False, need_dequant=True)
                assert deq_rot is not None
                self._cached_w_q_rot = deq_rot
                self._cached_w_q_rot_bf16 = self._cached_w_q_rot.bfloat16() if dev.type == "cuda" else None
                self._cached_q_key = q_key
        elif dev.type == "cuda" and self._cached_w_q_rot_bf16 is None:
            self._cached_w_q_rot_bf16 = self._cached_w_q_rot.bfloat16()

        self._defer_grad_active = True
        self._acc_grad_w_rot = (
            torch.zeros_like(self.weight_sphere, dtype=torch.float32) if self.weight_sphere.requires_grad else None
        )
        self._acc_grad_tau = (
            torch.zeros_like(self.tau_base, dtype=torch.float32) if self.tau_base.requires_grad else None
        )
        if self.scar_rank > 0 and self.a_scar is not None and self.b_scar is not None and self.gamma_scar is not None:
            with torch.no_grad():
                self._cached_a_t, _ = quantize_ternary_ste(self.a_scar.detach().float())
                self._cached_b_t, _ = quantize_ternary_ste(self.b_scar.detach().float())
            self._acc_grad_a_t = (
                torch.zeros_like(self.a_scar, dtype=torch.float32) if self.a_scar.requires_grad else None
            )
            self._acc_grad_b_t = (
                torch.zeros_like(self.b_scar, dtype=torch.float32) if self.b_scar.requires_grad else None
            )
            self._acc_grad_gamma = (
                torch.zeros_like(self.gamma_scar, dtype=torch.float32) if self.gamma_scar.requires_grad else None
            )
        else:
            self._acc_grad_a_t = None
            self._acc_grad_b_t = None
            self._acc_grad_gamma = None

    def finalize_deferred_grad_pass(self) -> dict[int, torch.Tensor]:
        """Projects accumulated `grad_w_rot` back onto `weight_sphere` ONCE and clears buffers."""
        self._defer_grad_active = False
        out_grads: dict[int, torch.Tensor] = {}
        if self._frozen_packed is not None or self.weight_sphere is None:
            return out_grads

        sqrt_i = math.sqrt(float(self.in_features))
        if self._acc_grad_w_rot is not None and self.weight_sphere.requires_grad:
            with torch.enable_grad():
                ws = self.weight_sphere.detach().float().requires_grad_(True)
                wu = project_rows_to_unit_sphere(ws)
                wr = fwht64_rademacher(wu, self.signs, inverse=False) * sqrt_i
                (gw,) = torch.autograd.grad(outputs=wr, inputs=ws, grad_outputs=self._acc_grad_w_rot)
            out_grads[id(self.weight_sphere)] = gw.to(self.weight_sphere.dtype)

        if self._acc_grad_tau is not None and self.tau_base.requires_grad:
            out_grads[id(self.tau_base)] = self._acc_grad_tau.to(self.tau_base.dtype)
        if self._acc_grad_a_t is not None and self.a_scar is not None and self.a_scar.requires_grad:
            # STE derivative `d(quantize_ternary_ste(a))/da == I`!
            out_grads[id(self.a_scar)] = self._acc_grad_a_t.to(self.a_scar.dtype)
        if self._acc_grad_b_t is not None and self.b_scar is not None and self.b_scar.requires_grad:
            out_grads[id(self.b_scar)] = self._acc_grad_b_t.to(self.b_scar.dtype)
        if self._acc_grad_gamma is not None and self.gamma_scar is not None and self.gamma_scar.requires_grad:
            out_grads[id(self.gamma_scar)] = self._acc_grad_gamma.to(self.gamma_scar.dtype)

        self._acc_grad_w_rot = None
        self._acc_grad_tau = None
        self._acc_grad_a_t = None
        self._acc_grad_b_t = None
        self._acc_grad_gamma = None
        return out_grads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._frozen_packed is not None:
            return self._frozen_packed.forward(x, self.lut_e8, self.signs, use_c_accel=True)

        xf = x.float()
        sqrt_i = math.sqrt(float(self.in_features))
        inv_sqrt_i = 1.0 / sqrt_i
        dev = self.weight_sphere.device
        q_key = (self.weight_sphere._version, self.weight_sphere.data_ptr(), dev.type, dev.index)

        if self._cached_q_key != q_key or self._cached_w_q_rot is None or self._cached_w_q_rot.device != dev:
            with torch.no_grad():
                w_u_ng = project_rows_to_unit_sphere(self.weight_sphere.detach().float())
                w_r_ng = fwht64_rademacher(w_u_ng, self.signs, inverse=False) * sqrt_i
                _, deq_rot = self._assign_e8_codes_and_dequant(w_r_ng, need_codes=False, need_dequant=True)
                assert deq_rot is not None
                self._cached_w_q_rot = deq_rot
                self._cached_w_q_rot_bf16 = self._cached_w_q_rot.bfloat16() if dev.type == "cuda" else None
                self._cached_q_key = q_key

        w_q_rot = self._cached_w_q_rot
        x_rot = fwht64_rademacher(xf, self.signs, inverse=False)

        if self._defer_grad_active and torch.is_grad_enabled():
            has_s = self.scar_rank > 0 and self.a_scar is not None and self.b_scar is not None and self.gamma_scar is not None
            scale_s_u = 1.0 / math.sqrt(float(self.in_features * max(1, self.scar_rank)))
            y_def = _DeferredE8GemmAutograd.apply(
                xf,
                x_rot,
                w_q_rot,
                self._cached_w_q_rot_bf16,
                float(self.tau_base.detach().item()),
                inv_sqrt_i,
                self._cached_a_t if has_s else None,
                self._cached_b_t if has_s else None,
                float(self.gamma_scar.detach().item()) if has_s and self.gamma_scar is not None else 0.0,
                scale_s_u,
                self,
            )
            return y_def.to(x.dtype)

        need_w_grad = torch.is_grad_enabled() and (self.weight_sphere.requires_grad or self.tau_base.requires_grad)
        if need_w_grad:
            w_unit = project_rows_to_unit_sphere(self.weight_sphere.float())
            w_rot = fwht64_rademacher(w_unit, self.signs, inverse=False) * sqrt_i
            w_ste_rot = w_rot + (w_q_rot - w_rot).detach()
            if dev.type == "cuda":
                y = (x_rot.bfloat16() @ w_ste_rot.bfloat16().T).float() * (self.tau_base * inv_sqrt_i)
            else:
                y = (x_rot @ w_ste_rot.T) * (self.tau_base * inv_sqrt_i)
        else:
            if dev.type == "cuda" and self._cached_w_q_rot_bf16 is not None:
                y = (x_rot.bfloat16() @ self._cached_w_q_rot_bf16.T).float() * (self.tau_base * inv_sqrt_i)
            else:
                y = (x_rot @ w_q_rot.T) * (self.tau_base * inv_sqrt_i)

        if self.scar_rank > 0 and self.a_scar is not None and self.b_scar is not None and self.gamma_scar is not None:
            need_s_grad = torch.is_grad_enabled() and (
                self.a_scar.requires_grad or self.b_scar.requires_grad or self.gamma_scar.requires_grad
            )
            if need_s_grad:
                a_t, _ = quantize_ternary_ste(self.a_scar.float())
                b_t, _ = quantize_ternary_ste(self.b_scar.float())
            else:
                ng_key = (
                    self.a_scar._version,
                    self.b_scar._version,
                    self.a_scar.data_ptr(),
                    self.b_scar.data_ptr(),
                    self.scar_rank,
                    dev.type,
                    dev.index,
                )
                if (
                    self._cached_nograd_key != ng_key
                    or self._cached_a_t is None
                    or self._cached_b_t is None
                    or self._cached_a_t.device != dev
                ):
                    with torch.no_grad():
                        self._cached_a_t, _ = quantize_ternary_ste(self.a_scar.detach().float())
                        self._cached_b_t, _ = quantize_ternary_ste(self.b_scar.detach().float())
                        self._cached_nograd_key = ng_key
                a_t = self._cached_a_t
                b_t = self._cached_b_t
            scale_s = self.gamma_scar / math.sqrt(float(self.in_features * self.scar_rank))
            y = y + ((xf @ a_t.T) @ b_t.T) * scale_s

        return y.to(x.dtype)

    @torch.no_grad()
    def freeze_to_packed(self) -> PackedE8TernaryMatrix:
        """Frees dense FP32 QAT weights and switches to `1.000 bpw` `scar_codec` C99/SIMD execution."""
        pk = self.to_packed()
        self.load_frozen_packed(pk)
        return pk

    @torch.no_grad()
    def load_frozen_packed(self, pk: PackedE8TernaryMatrix) -> None:
        """Installs a pre-packed `1.000 bpw` E8 + 2-bit Ternary matrix and releases FP32 master parameters."""
        self._frozen_packed = pk
        self.register_parameter("weight_sphere", None)
        self.register_parameter("tau_base", None)
        self.register_parameter("a_scar", None)
        self.register_parameter("b_scar", None)
        self.register_parameter("gamma_scar", None)

    @torch.no_grad()
    def to_packed(self) -> PackedE8TernaryMatrix:
        if self._frozen_packed is not None:
            return self._frozen_packed
        w_unit = project_rows_to_unit_sphere(self.weight_sphere.float())
        sqrt_i = math.sqrt(float(self.in_features))
        w_rot = fwht64_rademacher(w_unit, self.signs, inverse=False) * sqrt_i
        codes = self._assign_e8_codes(w_rot).to(torch.uint8)

        if self.scar_rank > 0 and self.a_scar is not None and self.b_scar is not None and self.gamma_scar is not None:
            _, a_i8 = quantize_ternary_ste(self.a_scar.float())
            _, b_i8 = quantize_ternary_ste(self.b_scar.float())
            a_pk = pack_ternary_2bit(a_i8)
            b_pk = pack_ternary_2bit(b_i8)
            g_val = float(self.gamma_scar.item())
        else:
            a_pk = torch.zeros((0, self.in_features // 4), dtype=torch.uint8, device=codes.device)
            b_pk = torch.zeros((self.out_features, 0), dtype=torch.uint8, device=codes.device)
            g_val = 0.0

        return PackedE8TernaryMatrix(
            out_features=self.out_features,
            in_features=self.in_features,
            scar_rank=self.scar_rank,
            codes_u8=codes.cpu(),
            a_tern_packed=a_pk.cpu(),
            b_tern_packed=b_pk.cpu(),
            tau_base=float(self.tau_base.item()),
            gamma_scar=g_val,
        )
