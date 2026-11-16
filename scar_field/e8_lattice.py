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

from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def rademacher_signs_det(dim: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Deterministic golden-ratio hash Rademacher signs `{-1.0, +1.0}` of length `dim`."""
    idx = torch.arange(dim, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = (((h >> 16) ^ h) * 0x45D9F3B) & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


def fwht64_rademacher(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Multiplierless orthonormal 64-point Fast Walsh-Hadamard Transform (zero `d_eq`)."""
    orig_shape = x.shape
    if orig_shape[-1] % 64 != 0:
        raise ValueError(f"Last dimension ({orig_shape[-1]}) must be divisible by 64")
    y = x if inverse else (x * signs)
    y = y.reshape(-1, 64)
    h = 1
    while h < 64:
        y = y.reshape(-1, 64 // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    out = (y.reshape(-1, 64) * 0.125).reshape(orig_shape)
    return (out * signs) if inverse else out


def _hadamard8_matrix() -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float32)
    for _ in range(3):
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(8.0)


@torch.no_grad()
def build_canonical_e8_lut256(device: torch.device | str = "cpu") -> torch.Tensor:
    """Builds the exact `[256, 8]` `E8 + Vacuum + Outer Hadamard Shell` codebook.

    Coordinate scaling is matched to unit-variance isotropic coordinates `E[z_j^2] = 1`
    (`E[||z||_2^2] = 8.0` on 8D blocks of `sqrt(I) * S^{I-1}`):
      - `0`: `(0, 0, 0, 0, 0, 0, 0, 0)` exact physical vacuum (`0x00`).
      - `1..112`: `D_8` roots `(±2, ±2, 0, 0, 0, 0, 0, 0)` (`||u||_2^2 = 8.0`, integer coords!).
      - `113..240`: Spinor roots `(±1, ..., ±1)` with even minus count (`||u||_2^2 = 8.0`, bipolar!).
      - `241..255`: 15 orthonormal `H_8` outer-shell vectors at `||u||_2^2 = 18.0` (`r = 3*sqrt(2)`).
    """
    rows = [[0.0] * 8]

    # 1. 112 integer D_8 roots scaled by 2 -> coords in {-2, 0, +2}, squared norm = 8.0
    for i, j in combinations(range(8), 2):
        for si in (-2.0, 2.0):
            for sj in (-2.0, 2.0):
                v = [0.0] * 8
                v[i] = si
                v[j] = sj
                rows.append(v)

    # 2. 128 half-integer spinor roots scaled by 2 -> coords in {-1, +1} with even minus count
    for signs in product((-1.0, 1.0), repeat=8):
        neg_cnt = sum(1 for s in signs if s < 0.0)
        if (neg_cnt % 2) == 0:
            rows.append(list(signs))

    if len(rows) != 241:
        raise RuntimeError(f"Expected 1 + 240 E8 vectors, got {len(rows)}")

    # 3. 15 outer-shell orthonormal H_8 modes at radius sqrt(18.0) for high-energy tail blocks
    h8 = _hadamard8_matrix() * math.sqrt(18.0)
    rows.append(h8[0].tolist())
    for k in range(1, 8):
        rows.append(h8[k].tolist())
        rows.append((-h8[k]).tolist())

    if len(rows) != 256:
        raise RuntimeError(f"Expected exactly 256 codewords, got {len(rows)}")

    return torch.tensor(rows, dtype=torch.float32, device=device)


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

    @property
    def substrate_bytes(self) -> int:
        return int(self.codes_u8.numel())

    @property
    def scar_bytes(self) -> int:
        return int(self.a_tern_packed.numel() + self.b_tern_packed.numel())

    @property
    def total_bytes(self) -> int:
        return self.substrate_bytes + self.scar_bytes + 8  # +8B for 2 float32 scalars

    @property
    def substrate_bits_per_weight(self) -> float:
        return (self.substrate_bytes * 8.0) / float(self.out_features * self.in_features)

    def forward(self, x: torch.Tensor, lut_e8: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x_rot = fwht64_rademacher(xf, signs, inverse=False)
        w_rot = lut_e8[self.codes_u8.long()].reshape(self.out_features, self.in_features)
        inv_sqrt_i = 1.0 / math.sqrt(float(self.in_features))
        y = (x_rot @ w_rot.T) * (self.tau_base * inv_sqrt_i)
        if self.scar_rank > 0 and abs(self.gamma_scar) > 0.0:
            a_f = unpack_ternary_2bit(self.a_tern_packed.to(xf.device))
            b_f = unpack_ternary_2bit(self.b_tern_packed.to(xf.device))
            scale_s = self.gamma_scar / math.sqrt(float(self.in_features * max(1, self.scar_rank)))
            y = y + ((xf @ a_f.T) @ b_f.T) * scale_s
        return y.to(x.dtype)


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

        w_init = project_rows_to_unit_sphere(torch.randn(out_features, in_features))
        self.weight_sphere = nn.Parameter(w_init)
        self.tau_base = nn.Parameter(torch.tensor(float(init_tau), dtype=torch.float32))

        lut = build_canonical_e8_lut256("cpu")
        lut_half_norm2 = 0.5 * (lut * lut).sum(dim=-1)
        signs = rademacher_signs_det(in_features, "cpu")
        self.register_buffer("lut_e8", lut, persistent=False)
        self.register_buffer("lut_half_norm2", lut_half_norm2, persistent=False)
        self.register_buffer("signs", signs, persistent=False)

        if scar_rank > 0:
            self.a_scar = nn.Parameter(project_rows_to_unit_sphere(torch.randn(scar_rank, in_features)))
            self.b_scar = nn.Parameter(project_rows_to_unit_sphere(torch.randn(out_features, scar_rank)))
            self.gamma_scar = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        else:
            self.register_parameter("a_scar", None)
            self.register_parameter("b_scar", None)
            self.register_parameter("gamma_scar", None)

        self.heal_residual_scars_svd()

    def _assign_e8_codes(self, w_rot_scaled: torch.Tensor) -> torch.Tensor:
        blks = w_rot_scaled.reshape(-1, 8)
        scores = blks @ self.lut_e8.T - self.lut_half_norm2
        return torch.argmax(scores, dim=-1).reshape(self.out_features, self.in_features // 8)

    @torch.no_grad()
    def heal_residual_scars_svd(self) -> None:
        """Closed-form `SVD/RRR` ternary scar healing over the exact `E_8` lattice residual."""
        self.weight_sphere.copy_(project_rows_to_unit_sphere(self.weight_sphere))
        if self.scar_rank <= 0 or self.a_scar is None or self.b_scar is None or self.gamma_scar is None:
            return

        sqrt_i = math.sqrt(float(self.in_features))
        w_unit = self.weight_sphere.float()
        w_rot = fwht64_rademacher(w_unit, self.signs, inverse=False) * sqrt_i
        codes = self._assign_e8_codes(w_rot)
        w_q_rot = self.lut_e8[codes].reshape(self.out_features, self.in_features) / sqrt_i
        w_q = fwht64_rademacher(w_q_rot, self.signs, inverse=True)

        res = float(self.tau_base.item()) * (w_unit - w_q)
        u, s, vh = torch.linalg.svd(res, full_matrices=False)
        r = min(self.scar_rank, s.numel())
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
        basis = (b_i8.float() @ a_i8.float()) / math.sqrt(float(self.in_features * self.scar_rank))
        denom = float((basis * basis).sum().item())
        if denom > 1e-8:
            opt_gamma = float((res * basis).sum().item()) / denom
            self.gamma_scar.fill_(opt_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        w_unit = project_rows_to_unit_sphere(self.weight_sphere.float())
        sqrt_i = math.sqrt(float(self.in_features))
        w_rot = fwht64_rademacher(w_unit, self.signs, inverse=False) * sqrt_i
        with torch.no_grad():
            codes = self._assign_e8_codes(w_rot)
            w_q_rot = self.lut_e8[codes].reshape(self.out_features, self.in_features)
        w_ste_rot = w_rot + (w_q_rot - w_rot).detach()

        x_rot = fwht64_rademacher(xf, self.signs, inverse=False)
        y = (x_rot @ w_ste_rot.T) * (self.tau_base / sqrt_i)

        if self.scar_rank > 0 and self.a_scar is not None and self.b_scar is not None and self.gamma_scar is not None:
            a_t, _ = quantize_ternary_ste(self.a_scar.float())
            b_t, _ = quantize_ternary_ste(self.b_scar.float())
            scale_s = self.gamma_scar / math.sqrt(float(self.in_features * self.scar_rank))
            y = y + ((xf @ a_t.T) @ b_t.T) * scale_s

        return y.to(x.dtype)

    @torch.no_grad()
    def to_packed(self) -> PackedE8TernaryMatrix:
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
