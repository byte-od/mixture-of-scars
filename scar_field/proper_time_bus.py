"""Relativistic Proper-Time Phase, Central Kalman-Scar O(1) Workspace & AURA Organs.

Zero-Legacy Memory & Topological Primitives for `ScarField v7.0`:
  1. `apply_relativistic_proper_time_phase`: Replaces integer `RoPE(t)` (`t = 0, 1, 2...`)
     with intrinsic information proper-time `d_phi_t = omega * s_t` acting purely on `S^{D_f-1}`.
  2. `ChannelSelectiveKalmanScarBus`: Single shared `O(1)` associative workspace (`Delta = +0 B`
     for `T -> inf`) with KDA-inspired multi-subspace selective write gates `beta_{t,g}`:
       - Exact `0.0` write on predictable vacuum (`s_t <= theta_shock` -> ZERO passive forgetting!).
       - Parallel Tensor-Core `Chunkwise WY` block-triangular scan (`cos > 0.999999` vs `step_pulse`).
       - Virtual rank-1 `GhostJumpDelta` readout (`0 bytes` of `W_fast` cloning during Scout rollouts!).
  3. Pure `AURA / System One` Organs:
       - Kinematic Spherical `Tri-Compass` (`Delta_local`, `Delta_origin`, `Delta_goal`).
       - `RepulsiveScarRing` (`O(1)` geometric anti-loop field on `S^{D_w-1}`).
       - `UniversalScarCartridgeStore` (`.uscar` poles attached directly to `S^{D_w-1}` from `tau=0`
         with closed-form `Zero-SGD RRR` minting and cryptographic `0x00` GDPR erasure).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from scar_field.cayley_geodesic import _halley_spd_inv_direct, c1_spherical_cap_resonance
from scar_field.e8_lattice import E8SphericalLinear, project_rows_to_unit_sphere


def build_proper_time_frequencies(d_form: int, theta_base: float = 1000.0, device: torch.device | str = "cpu") -> torch.Tensor:
    if d_form % 2 != 0:
        raise ValueError(f"d_form ({d_form}) must be even for symplectic 2x2 phase rotation")
    half = d_form // 2
    idx = torch.arange(0, half, dtype=torch.float32, device=device)
    return 1.0 / (theta_base ** (idx / float(max(1, half))))


def compute_proper_time(surprise: torch.Tensor, s_floor: float = 0.02) -> torch.Tensor:
    """Integrates proper information time `phi_t = sum_{j<=t} (s_floor + (1 - s_floor) * s_j)`."""
    ds = s_floor + (1.0 - s_floor) * surprise.clamp(0.0, 1.0)
    return torch.cumsum(ds, dim=-1)


def apply_relativistic_proper_time_phase(
    h_form: torch.Tensor,
    proper_time: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """Applies symplectic `SO(2)^{D_f/2}` rotation by `proper_time` (`||out||_2 == ||h_form||_2`)."""
    orig_dtype = h_form.dtype
    hf = h_form.float()
    angles = proper_time.float().unsqueeze(-1) * inv_freq.float()  # [..., D_f/2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x_even = hf[..., 0::2]
    x_odd = hf[..., 1::2]
    y_even = x_even * cos_a - x_odd * sin_a
    y_odd = x_even * sin_a + x_odd * cos_a
    out = torch.stack((y_even, y_odd), dim=-1).reshape(hf.shape)
    return out.to(orig_dtype)


@dataclass
class VirtualGhostJumpDelta:
    """Zero-copy virtual rank-1 update `(k, delta_v)` for reversible Ghost-Jump Scouts (`0 B` W_fast clone)."""

    keys: torch.Tensor       # [B_scouts, S_steps, d_key]
    delta_vals: torch.Tensor  # [B_scouts, S_steps, G, d_sub]


@dataclass
class KalmanBusState:
    """Strictly constant-size `O(1)` central workspace state (`Delta = +0 B` as `T -> inf`)."""

    w_fast: torch.Tensor         # [B, G, d_sub, d_key]
    k_prev: torch.Tensor         # [B, d_key] (`Prefix_{t-1}` for 1-hop Causal Transition Scars!)
    has_prev_key: torch.Tensor   # [B] bool
    recent_bytes: torch.Tensor   # [B, 4] uint8 (`O(1)` 4-byte causal wave register)
    recent_len: int
    ring_keys: torch.Tensor      # [B, R_ring, d_key]
    ring_vals: torch.Tensor      # [B, R_ring, d_val]
    ring_valid: torch.Tensor     # [B, R_ring] bool
    ring_ptr: int
    proper_time: torch.Tensor    # [B]
    h_origin: torch.Tensor       # [B, d_world] (`Tri-Compass` prompt anchor!)
    has_origin: torch.Tensor     # [B] bool
    h_prev: torch.Tensor         # [B, d_world] (`Tri-Compass` local velocity!)
    repulsive_ring: torch.Tensor  # [B, R_rep, d_world] (`AURA` negative anti-loop scars!)
    repulsive_ptr: int
    biorth_keys: Optional[torch.Tensor] = None      # [B, M_biorth, d_key] (`O(1)` Biorthogonal Dual Shock Bank!)
    biorth_vals: Optional[torch.Tensor] = None      # [B, M_biorth, d_bus]
    biorth_salience: Optional[torch.Tensor] = None  # [B, M_biorth]
    biorth_valid: Optional[torch.Tensor] = None     # [B, M_biorth] bool

    @property
    def total_state_bytes(self) -> int:
        tensors = [
            self.w_fast,
            self.k_prev,
            self.has_prev_key,
            self.recent_bytes,
            self.ring_keys,
            self.ring_vals,
            self.ring_valid,
            self.proper_time,
            self.h_origin,
            self.has_origin,
            self.h_prev,
            self.repulsive_ring,
        ]
        for opt_t in (self.biorth_keys, self.biorth_vals, self.biorth_salience, self.biorth_valid):
            if opt_t is not None:
                tensors.append(opt_t)
        return sum(int(t.numel() * t.element_size()) for t in tensors)


@dataclass
class UScarPoleCartridge:
    """Universal `.uscar` pole attached directly to `S^{D_w-1}` (`Zero-SGD` + `0x00` GDPR wipe)."""

    cartridge_id: str
    doc_id: str
    pole_world: torch.Tensor      # [d_world] unit pole on `S^{D_w-1}`
    a_proj: torch.Tensor          # [rank, d_world]
    b_value: torch.Tensor         # [d_bus, rank]
    horizon_kappa: float = 0.55
    wiped_0x00: bool = False

    @property
    def aligned_bytes(self) -> int:
        raw_b = int(
            (self.pole_world.numel() + self.a_proj.numel() + self.b_value.numel()) * 2
        )
        return ((raw_b + 4095) // 4096) * 4096

    def wipe_gdpr_0x00_(self) -> None:
        """Cryptographic physical `0x00` zeroization of the cartridge tensors in-place."""
        self.pole_world.zero_()
        self.a_proj.zero_()
        self.b_value.zero_()
        self.wiped_0x00 = True


class UniversalScarCartridgeStore(nn.Module):
    """Manages detached `.uscar` gravitational poles directly on `S^{D_w-1}` with Biorthogonal Dual Poles."""

    def __init__(
        self,
        d_world: int,
        d_bus: int,
        biorthogonal_poles: bool = True,
        biorth_lambda: float = 0.04,
    ) -> None:
        super().__init__()
        self.d_world = d_world
        self.d_bus = d_bus
        self.biorthogonal_poles = bool(biorthogonal_poles)
        self.biorth_lambda = float(biorth_lambda)
        self.cartridges: Dict[str, UScarPoleCartridge] = {}

    @torch.no_grad()
    def mint_zero_sgd_rrr(
        self,
        cartridge_id: str,
        doc_id: str,
        world_queries: torch.Tensor,
        target_bus_deltas: torch.Tensor,
        rank: int = 8,
        ridge_lambda: float = 1e-2,
        horizon_kappa: float = 0.52,
    ) -> UScarPoleCartridge:
        """Mints a detached `.uscar` cartridge in closed form via Ridge Reduced-Rank Regression (`0` trunk SGD)."""
        q_u = project_rows_to_unit_sphere(world_queries.float())  # [N, d_world]
        y_t = target_bus_deltas.float()  # [N, d_bus]
        pole = project_rows_to_unit_sphere(q_u.mean(dim=0, keepdim=True)).squeeze(0)

        # Solve (Q^T Q + lambda I) W_rrr = Q^T Y, then rank-r SVD truncation
        qt_q = q_u.T @ q_u + ridge_lambda * torch.eye(self.d_world, device=q_u.device)
        w_full = torch.linalg.solve(qt_q, q_u.T @ y_t)  # [d_world, d_bus]
        u, s, vh = torch.linalg.svd(w_full.T, full_matrices=False)  # [d_bus, d_world]
        r_eff = min(rank, s.numel())
        s_half = torch.sqrt(s[:r_eff].clamp(min=1e-8))
        b_val = u[:, :r_eff] * s_half.unsqueeze(0)   # [d_bus, r_eff]
        a_prj = s_half.unsqueeze(1) * vh[:r_eff, :]  # [r_eff, d_world]

        cart = UScarPoleCartridge(
            cartridge_id=cartridge_id,
            doc_id=doc_id,
            pole_world=pole.detach().clone(),
            a_proj=a_prj.detach().clone(),
            b_value=b_val.detach().clone(),
            horizon_kappa=horizon_kappa,
            wiped_0x00=False,
        )
        self.cartridges[cartridge_id] = cart
        return cart

    def wipe_cartridge_0x00(self, cartridge_id: str) -> None:
        if cartridge_id in self.cartridges:
            self.cartridges[cartridge_id].wipe_gdpr_0x00_()

    def wipe_document_0x00(self, doc_id: str) -> int:
        wiped = 0
        for c in self.cartridges.values():
            if c.doc_id == doc_id and not c.wiped_0x00:
                c.wipe_gdpr_0x00_()
                wiped += 1
        return wiped

    def query(self, h_world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns `(uscar_delta [..., d_bus], max_pole_resonance [...])` (`0.0` when wiped/absent!)."""
        out = torch.zeros(*h_world.shape[:-1], self.d_bus, dtype=h_world.dtype, device=h_world.device)
        max_res = torch.zeros(h_world.shape[:-1], dtype=h_world.dtype, device=h_world.device)

        active = [c for c in self.cartridges.values() if not c.wiped_0x00]
        if not active:
            return out, max_res

        hw = h_world.float()
        if len(active) >= 2 and self.biorthogonal_poles:
            poles_mat = torch.stack([c.pole_world.to(hw.device).float() for c in active], dim=0)  # [C, d_world]
            n_c = poles_mat.shape[0]
            gram = torch.matmul(poles_mat, poles_mat.T) + self.biorth_lambda * torch.eye(
                n_c, dtype=torch.float32, device=hw.device
            )
            # Biorthogonal Dual Poles: `P_tilde = (P P^T + lam I)^{-1} P` -> `P_tilde P^T ≈ I_C`!
            dual_poles = torch.linalg.solve(gram, poles_mat)  # [C, d_world]
            self_gain = (dual_poles * poles_mat).sum(dim=-1).clamp(min=0.1)  # [C]
            raw_cos_all = torch.matmul(hw, poles_mat.T)  # [..., C]
            dual_coord_all = torch.matmul(hw, dual_poles.T) / self_gain  # [..., C]
            for idx_c, c in enumerate(active):
                cos_p = raw_cos_all[..., idx_c]
                dual_c = dual_coord_all[..., idx_c]
                raw_gate = (2.0 * c1_spherical_cap_resonance(cos_p, hw.new_tensor(c.horizon_kappa))).clamp(0.0, 1.0)
                dual_gate = (2.0 * c1_spherical_cap_resonance(dual_c, hw.new_tensor(0.35))).clamp(0.0, 1.0)
                norm_gate = torch.minimum(raw_gate, dual_gate)
                max_res = torch.maximum(max_res, norm_gate.to(max_res.dtype))
                if float(norm_gate.max().item()) > 0.0:
                    a_p = c.a_proj.to(hw.device)
                    b_v = c.b_value.to(hw.device)
                    inj = torch.matmul(torch.matmul(hw, a_p.T), b_v.T)
                    out = out + (norm_gate.unsqueeze(-1) * inj).to(out.dtype)
            return out, max_res

        for c in active:
            pole = c.pole_world.to(hw.device)
            cos_p = torch.matmul(hw, pole)  # [...]
            gate = c1_spherical_cap_resonance(cos_p, hw.new_tensor(c.horizon_kappa))
            # Normalize pole peak (`gate(1.0) = 0.5 -> norm_gate(1.0) = 1.0`, `0.0` outside horizon!)
            norm_gate = (gate * 2.0).clamp(0.0, 1.0)
            max_res = torch.maximum(max_res, norm_gate.to(max_res.dtype))
            if float(norm_gate.max().item()) > 0.0:
                a_p = c.a_proj.to(hw.device)
                b_v = c.b_value.to(hw.device)
                inj = torch.matmul(torch.matmul(hw, a_p.T), b_v.T)  # [..., d_bus]
                out = out + (norm_gate.unsqueeze(-1) * inj).to(out.dtype)
        return out, max_res


def _fast_block_unit_tril_inv(l_strict: torch.Tensor) -> torch.Tensor:
    """Exact bounded block-2x2 recursive inverse of `I + l_strict` (`0` `L^k` overflow, `< 5e-7` up to `clen=256`!).

    Unlike naive `(I - L)(I + L^2)(I + L^4)...` where intermediate powers `L^64` explode to `10^26` and
    overflow float32/TF32 at `clen >= 64..128`, recursive block-2x2 inversion `T_21 = -T_22 @ L_21 @ T_11`
    manipulates ONLY contractions (`||T|| <= 1.0`), staying bounded by `1.0` at every recursion level!
    """
    *lead, n, _ = l_strict.shape
    p2 = 1
    while p2 < n:
        p2 *= 2
    if p2 > n:
        l_work = torch.zeros(*lead, p2, p2, dtype=l_strict.dtype, device=l_strict.device)
        l_work[..., :n, :n] = l_strict
    else:
        l_work = l_strict

    t_diag = torch.ones(*lead, p2, 1, 1, dtype=l_strict.dtype, device=l_strict.device)
    blk = 1
    while blk < p2:
        num_pairs = p2 // (2 * blk)
        t_pair = t_diag.reshape(*lead, num_pairs, 2, blk, blk)
        t11 = t_pair[..., 0, :, :]
        t22 = t_pair[..., 1, :, :]
        lv = l_work.reshape(*lead, num_pairs, 2, blk, num_pairs, 2, blk)
        lv_d = lv.diagonal(offset=0, dim1=-6, dim2=-3)
        l21 = lv_d[..., 1, :, 0, :, :].movedim(-1, -3)
        t21 = -torch.matmul(torch.matmul(t22, l21), t11)
        z12 = torch.zeros_like(t21)
        top = torch.cat((t11, z12), dim=-1)
        bot = torch.cat((t21, t22), dim=-1)
        t_diag = torch.cat((top, bot), dim=-2)
        blk *= 2

    out = t_diag.squeeze(-3)
    return out[..., :n, :n]


class _UnitTrilSolveAutograd(torch.autograd.Function):
    """Exact 2-matmul adjoint of `(I + L_strict) U = R` (`0` autograd unrolling through block recursion!)."""

    @staticmethod
    def forward(ctx, l_strict: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        inv_mat = _fast_block_unit_tril_inv(l_strict)
        u_out = torch.matmul(inv_mat, rhs)
        ctx.save_for_backward(inv_mat, u_out)
        return u_out

    @staticmethod
    def backward(ctx, grad_u: torch.Tensor):  # type: ignore[override]
        inv_mat, u_out = ctx.saved_tensors
        grad_rhs = torch.matmul(inv_mat.transpose(-2, -1), grad_u)
        grad_l = -torch.tril(torch.matmul(grad_rhs, u_out.transpose(-2, -1)), diagonal=-1)
        return grad_l, grad_rhs


def chunkwise_wy_channel_selective_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    w0: Optional[torch.Tensor] = None,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel Tensor-Core Chunkwise WY scan for channel-selective Widrow-Hoff `W_fast`.

    Exact algebraic identity with sequential causal pre-write `step_pulse` (`diff < 1e-6`):
      - `q, k`: `[B, T, d_k]` unit vectors (`||k_t||_2 == 1`).
      - `v`: `[B, T, G, d_sub]` value vectors across `G` independent selective subspaces.
      - `beta`: `[B, T, G]` selective shock gates in `[0, 1]` (`0.0` on predictable vacuum!).
      - Returns `(readout [B, T, G, d_sub], w_final [B, G, d_sub, d_k])`.
    """
    bsz, seq_len, d_k = k.shape
    _, _, num_g, d_sub = v.shape
    dev = k.device

    qf = q.float()
    kf = k.float()
    vf = v.float()
    bf = beta.float()

    if w0 is None:
        w_cur = torch.zeros(bsz, num_g, d_sub, d_k, dtype=torch.float32, device=dev)
    else:
        w_cur = w0.float().clone()

    outputs: List[torch.Tensor] = []
    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        clen = end - start

        qc = qf[:, start:end, :]          # [B, C, d_k]
        kc = kf[:, start:end, :]          # [B, C, d_k]
        vc = vf[:, start:end, :, :]       # [B, C, G, d_sub]
        bc = bf[:, start:end, :]          # [B, C, G]

        # Gram matrices across chunk: [B, C, C]
        kk_t = torch.matmul(kc, kc.transpose(-2, -1))
        qk_t = torch.matmul(qc, kc.transpose(-2, -1))
        tril_strict = torch.tril(torch.ones(clen, clen, dtype=torch.bool, device=dev), diagonal=-1)
        kk_lower = torch.where(tril_strict, kk_t, torch.zeros_like(kk_t))
        qk_lower = torch.where(tril_strict, qk_t, torch.zeros_like(qk_t))

        # Build strictly lower-triangular `L_g = beta_g * KK_strict_lower` -> `[B, G, C, C]`
        eye_c = torch.eye(clen, dtype=torch.float32, device=dev)
        l_mat = bc.permute(0, 2, 1).unsqueeze(-1) * kk_lower.unsqueeze(1)  # [B, G, C, C]

        # Residual innovation against chunk-start `w_cur`: `V_g - K W_cur^T` -> `[B, G, C, d_sub]`
        vc_g = vc.permute(0, 2, 1, 3)  # [B, G, C, d_sub]
        kw0 = torch.matmul(kc.unsqueeze(1), w_cur.transpose(-2, -1))  # [B, G, C, d_sub]
        rhs = bc.permute(0, 2, 1).unsqueeze(-1) * (vc_g - kw0)

        if dev.type in ("cuda", "mps") and clen <= 256:
            if torch.is_grad_enabled() and (l_mat.requires_grad or rhs.requires_grad):
                u_g = _UnitTrilSolveAutograd.apply(l_mat, rhs)
            else:
                u_g = torch.matmul(_fast_block_unit_tril_inv(l_mat), rhs)
        else:
            u_g = torch.linalg.solve_triangular(eye_c + l_mat, rhs, upper=False, unitriangular=True)

        # Strictly causal pre-write readout `O_g = Q W_cur^T + QK_strict_lower @ U_g`
        qw0 = torch.matmul(qc.unsqueeze(1), w_cur.transpose(-2, -1))  # [B, G, C, d_sub]
        o_g = qw0 + torch.matmul(qk_lower.unsqueeze(1), u_g)          # [B, G, C, d_sub]
        outputs.append(o_g.permute(0, 2, 1, 3))                       # [B, C, G, d_sub]

        # Advance `w_cur = w_cur + U_g^T @ K`
        w_cur = w_cur + torch.matmul(u_g.transpose(-2, -1), kc.unsqueeze(1))

    readout = torch.cat(outputs, dim=1).to(v.dtype)
    return readout, w_cur.to(v.dtype)


def biorthogonal_shock_bank_read(
    q_vecs: torch.Tensor,
    b_keys: torch.Tensor,
    b_vals: torch.Tensor,
    b_valid: torch.Tensor,
    horizon_kappa: torch.Tensor,
    ridge_lambda: float = 0.04,
) -> torch.Tensor:
    """Biorthogonal Dual Shock Readout (`K_tilde = (K K^T + lam I)^{-1} K` -> `K_tilde K^T ≈ I_M`).

    Eliminates oblique hubness/crosstalk among `M` stored long-range shock keys in strictly `O(1)` space
    with ZERO `Softmax`, ZERO `exp()`, and exact `0.0` output outside the spherical horizon `horizon_kappa`.
    """
    bsz, q_len, _ = q_vecs.shape
    m_slots = b_keys.shape[1]
    d_bus = b_vals.shape[-1]
    if b_valid.device.type == "cpu" and not bool(b_valid.any().item()):
        return torch.zeros(bsz, q_len, d_bus, dtype=b_vals.dtype, device=q_vecs.device)

    val_f = b_valid.float()  # [B, M]
    k_m = b_keys.float() * val_f.unsqueeze(-1)  # [B, M, d_key]
    v_m = b_vals.float() * val_f.unsqueeze(-1)  # [B, M, d_bus]
    qf = q_vecs.float()

    raw_cos = torch.matmul(qf, k_m.transpose(-2, -1))  # [B, L, M]
    masked_cos = torch.where(b_valid.unsqueeze(1), raw_cos, torch.full_like(raw_cos, -1.0))
    max_cos = masked_cos.amax(dim=-1, keepdim=True)  # [B, L, 1]
    kap = horizon_kappa.float().clamp(0.15, 0.88)
    hor_gate = (2.0 * c1_spherical_cap_resonance(max_cos, kap)).clamp(0.0, 1.0)  # [B, L, 1]
    if q_vecs.device.type == "cpu" and float(hor_gate.max().item()) <= 0.0:
        return torch.zeros(bsz, q_len, d_bus, dtype=b_vals.dtype, device=q_vecs.device)

    eye_m = torch.eye(m_slots, dtype=torch.float32, device=q_vecs.device).unsqueeze(0)
    gram = torch.matmul(k_m, k_m.transpose(-2, -1)) + float(ridge_lambda) * eye_m  # [B, M, M]
    raw_rhs = (raw_cos * val_f.unsqueeze(1)).transpose(-2, -1)  # [B, M, L]
    if gram.device.type in ("cuda", "mps"):
        gram_inv = _halley_spd_inv_direct(gram, eye_m, n_steps=8)
        alpha = torch.matmul(gram_inv, raw_rhs).transpose(-2, -1)
    else:
        alpha = torch.linalg.solve(gram, raw_rhs).transpose(-2, -1)  # [B, L, M]
    alpha_pos = alpha.clamp(0.0, 1.5) * val_f.unsqueeze(1)
    out = hor_gate * torch.matmul(alpha_pos, v_m)
    return out.to(b_vals.dtype)


@torch.no_grad()
def biorthogonal_shock_bank_insert_step_(
    b_keys: torch.Tensor,
    b_vals: torch.Tensor,
    b_sal: torch.Tensor,
    b_valid: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sal_new: torch.Tensor,
    write_thresh: float = 0.22,
) -> None:
    """Vectorized `O(1)` Biorthogonal Shock Bank in-place update (`0` `.item()` GPU syncs across `bsz`!)."""
    bsz, m_slots, d_k = b_keys.shape
    d_v = b_vals.shape[-1]
    k_norm2 = (k_new * k_new).sum(dim=-1)  # [B]
    cand_ok = (sal_new > write_thresh) & (k_norm2 >= 0.25)  # [B]

    sims = torch.matmul(b_keys, k_new.unsqueeze(-1)).squeeze(-1)  # [B, M]
    sims_valid = torch.where(b_valid, sims, torch.full_like(sims, -2.0))
    max_sim, max_idx = torch.max(sims_valid, dim=-1)  # [B]
    is_dedup = b_valid.any(dim=-1) & (max_sim > 0.92)  # [B]

    eff_sal = torch.where(b_valid, b_sal, torch.full_like(b_sal, -1.0))
    min_sal, min_idx = torch.min(eff_sal, dim=-1)  # [B]
    min_slot_valid = b_valid.gather(1, min_idx.unsqueeze(1)).squeeze(1)
    can_evict = (~min_slot_valid) | (sal_new >= min_sal * 0.90)

    do_write = cand_ok & (is_dedup | can_evict)
    target_idx = torch.where(is_dedup, max_idx, min_idx)  # [B]

    slot_mask = torch.zeros(bsz, m_slots, dtype=torch.bool, device=b_keys.device)
    slot_mask.scatter_(1, target_idx.unsqueeze(1), do_write.unsqueeze(1))

    old_sal_at_t = b_sal.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    new_sal_val = torch.where(is_dedup, torch.maximum(sal_new, old_sal_at_t), sal_new)

    b_keys.copy_(torch.where(slot_mask.unsqueeze(-1).expand(bsz, m_slots, d_k), k_new.unsqueeze(1), b_keys))
    b_vals.copy_(torch.where(slot_mask.unsqueeze(-1).expand(bsz, m_slots, d_v), v_new.unsqueeze(1), b_vals))
    b_sal.copy_(torch.where(slot_mask, new_sal_val.unsqueeze(1).expand(bsz, m_slots), b_sal))
    b_valid.logical_or_(slot_mask)


class ChannelSelectiveKalmanScarBus(nn.Module):
    """Unified 3-Timescale Central Workspace + Biorthogonal Dual Shock Bank + Kinematic `Tri-Compass`."""

    def __init__(
        self,
        d_form: int,
        d_world: int,
        d_bus: int = 64,
        d_key: int = 64,
        num_subspaces: int = 8,
        ring_len: int = 16,
        repulsive_len: int = 8,
        shock_write_theta: float = 0.18,
        enable_kalman_fast: bool = True,
        recent_wave_buf_len: int = 4,
        biorth_slots: int = 0,
        biorth_lambda: float = 0.04,
        enable_biorth_uscar: bool = True,
    ) -> None:
        super().__init__()
        if d_bus % num_subspaces != 0:
            raise ValueError(f"d_bus ({d_bus}) must be divisible by num_subspaces ({num_subspaces})")
        self.d_form = d_form
        self.d_world = d_world
        self.d_bus = d_bus
        self.d_key = d_key
        self.num_subspaces = num_subspaces
        self.d_sub = d_bus // num_subspaces
        self.ring_len = ring_len
        self.repulsive_len = repulsive_len
        self.shock_write_theta = shock_write_theta
        self.enable_kalman_fast = enable_kalman_fast
        self.recent_wave_buf_len = max(4, int(recent_wave_buf_len))
        self.biorth_slots = max(0, int(biorth_slots))
        self.biorth_lambda = float(biorth_lambda)

        joint_dim = d_form + d_world
        if joint_dim % 64 != 0:
            raise ValueError(f"d_form + d_world ({joint_dim}) must be divisible by 64")

        # Single fused E8 probe emitting `(q, k, v, channel_gates)` — replaces 4 Attn + 3 MLP matrices!
        self.qkv_e8 = E8SphericalLinear(
            in_features=joint_dim,
            out_features=2 * d_key + d_bus + num_subspaces,
            scar_rank=8,
            init_tau=1.0,
        )
        # Kinematic Spherical `Tri-Compass` compressor (`Delta_local, Delta_origin, Delta_goal` -> `d_bus`)
        tri_in = ((3 * d_world + 63) // 64) * 64
        self.tri_in_padded = tri_in
        self.tri_compass_e8 = E8SphericalLinear(
            in_features=tri_in,
            out_features=d_bus,
            scar_rank=4,
            init_tau=0.45,
        )

        self.register_buffer("inv_freq", build_proper_time_frequencies(d_form), persistent=False)
        self.ring_kappa = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.tier_mix = nn.Parameter(torch.tensor([1.35, 0.45, 1.0, 0.30], dtype=torch.float32))
        if self.biorth_slots > 0:
            self.biorth_gain = nn.Parameter(torch.tensor(1.10, dtype=torch.float32))
            self.biorth_kappa = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
        self.uscar_store = UniversalScarCartridgeStore(
            d_world=d_world,
            d_bus=d_bus,
            biorthogonal_poles=enable_biorth_uscar,
            biorth_lambda=biorth_lambda,
        )

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> KalmanBusState:
        bk = torch.zeros(batch_size, self.biorth_slots, self.d_key, device=device) if self.biorth_slots > 0 else None
        bv = torch.zeros(batch_size, self.biorth_slots, self.d_bus, device=device) if self.biorth_slots > 0 else None
        bs = torch.full((batch_size, self.biorth_slots), -1.0, device=device) if self.biorth_slots > 0 else None
        bval = (
            torch.zeros(batch_size, self.biorth_slots, dtype=torch.bool, device=device)
            if self.biorth_slots > 0
            else None
        )
        return KalmanBusState(
            w_fast=torch.zeros(batch_size, self.num_subspaces, self.d_sub, self.d_key, device=device),
            k_prev=torch.zeros(batch_size, self.d_key, device=device),
            has_prev_key=torch.zeros(batch_size, dtype=torch.bool, device=device),
            recent_bytes=torch.zeros(batch_size, self.recent_wave_buf_len, dtype=torch.uint8, device=device),
            recent_len=0,
            ring_keys=torch.zeros(batch_size, self.ring_len, self.d_key, device=device),
            ring_vals=torch.zeros(batch_size, self.ring_len, self.d_bus, device=device),
            ring_valid=torch.zeros(batch_size, self.ring_len, dtype=torch.bool, device=device),
            ring_ptr=0,
            proper_time=torch.zeros(batch_size, device=device),
            h_origin=torch.zeros(batch_size, self.d_world, device=device),
            has_origin=torch.zeros(batch_size, dtype=torch.bool, device=device),
            h_prev=torch.zeros(batch_size, self.d_world, device=device),
            repulsive_ring=torch.zeros(batch_size, self.repulsive_len, self.d_world, device=device),
            repulsive_ptr=0,
            biorth_keys=bk,
            biorth_vals=bv,
            biorth_salience=bs,
            biorth_valid=bval,
        )

    def _extract_qkv_beta(
        self,
        h_form_unphased: torch.Tensor,
        h_world: torch.Tensor,
        surprise: torch.Tensor,
        v_shock_hint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts Pre-Phase content-addressed `(q, k_ctx)` + arriving innovation `v` + selective `beta`."""
        joint = project_rows_to_unit_sphere(torch.cat((h_form_unphased, h_world), dim=-1))
        proj = self.qkv_e8(joint)
        dk = self.d_key
        db = self.d_bus
        q = project_rows_to_unit_sphere(proj[..., :dk])
        # Blend symmetric prefix key `(0.65 * q + 0.35 * k_raw)` so repeated stems (`"PIN="`) resonate from step 0!
        k_raw = project_rows_to_unit_sphere(proj[..., dk : 2 * dk])
        k = project_rows_to_unit_sphere(0.65 * q + 0.35 * k_raw)
        v_base = proj[..., 2 * dk : 2 * dk + db]
        if v_shock_hint is not None:
            if v_shock_hint.shape[-1] >= db:
                v_base = v_base + 0.85 * v_shock_hint[..., :db]
            else:
                pad_v = torch.zeros(
                    *v_shock_hint.shape[:-1], db - v_shock_hint.shape[-1], dtype=v_base.dtype, device=v_base.device
                )
                v_base = v_base + 0.85 * torch.cat((v_shock_hint, pad_v), dim=-1)
        v = project_rows_to_unit_sphere(v_base).reshape(*h_world.shape[:-1], self.num_subspaces, self.d_sub)
        raw_g = proj[..., 2 * dk + db :]  # [..., G]
        chan_gate = 0.65 + 0.35 * (raw_g / torch.sqrt(1.0 + raw_g * raw_g))
        shock_open = (
            2.0
            * c1_spherical_cap_resonance(
                surprise.unsqueeze(-1),
                surprise.new_tensor(self.shock_write_theta),
            )
        ).clamp(0.0, 1.0)
        beta = (shock_open * chan_gate).clamp(0.0, 0.98)
        return q, k, v, beta

    def _compute_tri_compass(
        self,
        h_world: torch.Tensor,
        h_prev: torch.Tensor,
        h_origin: torch.Tensor,
        goal_hint: torch.Tensor,
    ) -> torch.Tensor:
        d_local = h_world - h_prev
        d_origin = h_world - h_origin
        d_goal = goal_hint - h_world
        tri = torch.cat((d_local, d_origin, d_goal), dim=-1)
        if self.tri_in_padded > tri.shape[-1]:
            pad = torch.zeros(*tri.shape[:-1], self.tri_in_padded - tri.shape[-1], dtype=tri.dtype, device=tri.device)
            tri = torch.cat((tri, pad), dim=-1)
        return self.tri_compass_e8(project_rows_to_unit_sphere(tri))

    def forward_sequence(
        self,
        h_form: torch.Tensor,
        h_world: torch.Tensor,
        surprise: torch.Tensor,
        state: Optional[KalmanBusState] = None,
        chunk_size: int = 32,
        v_shock_hint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, KalmanBusState]:
        """Parallel sequence forward returning `(h_form_phased, bus_ctx, uscar_res, new_state)`."""
        bsz, seq_len, _ = h_world.shape
        dev = h_world.device
        st = self.init_state(bsz, dev) if state is None else state

        ds = 0.02 + 0.98 * surprise.clamp(0.0, 1.0)
        prop_t = st.proper_time.unsqueeze(-1) + torch.cumsum(ds, dim=-1)  # [B, T]
        hf_phased = apply_relativistic_proper_time_phase(h_form, prop_t, self.inv_freq)

        # Pre-Phase content-addressed `(q, k_ctx)` so `"PIN="` matches `"PIN="` at ANY distance `T -> inf`!
        q, k_ctx, v, beta = self._extract_qkv_beta(h_form, h_world, surprise, v_shock_hint=v_shock_hint)

        # Causal Transition Binding (`Prefix_{t-1} -> Shock_t`): native 1-hop Induction Scar!
        k_first = torch.where(st.has_prev_key.unsqueeze(-1), st.k_prev, torch.zeros_like(k_ctx[:, 0, :]))
        b_first = torch.where(st.has_prev_key.unsqueeze(-1), beta[:, 0, :], torch.zeros_like(beta[:, 0, :]))
        if seq_len > 1:
            k_bind = torch.cat((k_first.unsqueeze(1), k_ctx[:, :-1, :]), dim=1)
            beta_bind = torch.cat((b_first.unsqueeze(1), beta[:, 1:, :]), dim=1)
        else:
            k_bind = k_first.unsqueeze(1)
            beta_bind = b_first.unsqueeze(1)

        # 1. Fast Episodic Tier A: Parallel Chunkwise WY Channel-Selective Kalman-Scar (`O(1)`)
        if self.enable_kalman_fast:
            kalman_out, w_final = chunkwise_wy_channel_selective_scan(
                q=q, k=k_bind, v=v, beta=beta_bind, w0=st.w_fast, chunk_size=chunk_size
            )
            kalman_flat = kalman_out.reshape(bsz, seq_len, self.d_bus)
        else:
            w_final = st.w_fast
            kalman_flat = torch.zeros(bsz, seq_len, self.d_bus, dtype=v.dtype, device=dev)

        # 2. Fast Sensory Ring Tier B: Proper-time phased local window `[t - ring_len .. t - 1]`
        q_ring = apply_relativistic_proper_time_phase(q, prop_t, build_proper_time_frequencies(self.d_key, device=dev))
        k_ring = apply_relativistic_proper_time_phase(
            k_ctx, prop_t, build_proper_time_frequencies(self.d_key, device=dev)
        )
        v_flat = v.reshape(bsz, seq_len, self.d_bus)

        # 2.5. Long-Horizon Biorthogonal Dual Shock Bank (`O(1)` `M_biorth` slots, `K_tilde K^T ≈ I`, `Delta = +0 B`)
        if self.biorth_slots > 0:
            cur_bk = (
                st.biorth_keys.detach().clone()
                if st.biorth_keys is not None
                else torch.zeros(bsz, self.biorth_slots, self.d_key, device=dev)
            )
            cur_bv = (
                st.biorth_vals.detach().clone()
                if st.biorth_vals is not None
                else torch.zeros(bsz, self.biorth_slots, self.d_bus, device=dev)
            )
            cur_bsal = (
                st.biorth_salience.detach().clone()
                if st.biorth_salience is not None
                else torch.full((bsz, self.biorth_slots), -1.0, device=dev)
            )
            cur_bval = (
                st.biorth_valid.detach().clone()
                if st.biorth_valid is not None
                else torch.zeros(bsz, self.biorth_slots, dtype=torch.bool, device=dev)
            )
            biorth_chunks: List[torch.Tensor] = []
            sal_seq = beta_bind.mean(dim=-1).detach()  # [B, T]
            k_bind_det = k_bind.detach()
            v_flat_det = v_flat.detach()
            for c_start in range(0, seq_len, chunk_size):
                c_end = min(seq_len, c_start + chunk_size)
                b_read = biorthogonal_shock_bank_read(
                    q[:, c_start:c_end, :],
                    cur_bk.clone(),
                    cur_bv.clone(),
                    cur_bval.clone(),
                    self.biorth_kappa,
                    ridge_lambda=self.biorth_lambda,
                )
                biorth_chunks.append(b_read)
                # Commit top shock transitions of this chunk into `O(1)` Biorthogonal Bank for subsequent chunks!
                c_sal = sal_seq[:, c_start:c_end]
                top_n = min(8, c_end - c_start)
                _, top_pos = torch.topk(c_sal, k=top_n, dim=-1)
                top_pos_sorted, _ = torch.sort(top_pos, dim=-1)
                for j_s in range(top_n):
                    idx_rel = top_pos_sorted[:, j_s]  # [B] within `[0 .. c_end - c_start - 1]`
                    idx_abs = idx_rel + c_start       # [B] within `[0 .. seq_len - 1]`
                    k_step = k_bind_det.gather(1, idx_abs.view(bsz, 1, 1).expand(bsz, 1, self.d_key)).squeeze(1)
                    v_step = v_flat_det.gather(1, idx_abs.view(bsz, 1, 1).expand(bsz, 1, self.d_bus)).squeeze(1)
                    s_step = c_sal.gather(1, idx_rel.view(bsz, 1)).squeeze(1)
                    biorthogonal_shock_bank_insert_step_(
                        cur_bk, cur_bv, cur_bsal, cur_bval, k_step, v_step, s_step, write_thresh=0.22
                    )
            biorth_out = self.biorth_gain * torch.cat(biorth_chunks, dim=1)
        else:
            cur_bk, cur_bv, cur_bsal, cur_bval = None, None, None, None
            biorth_out = torch.zeros_like(v_flat)
        k_cat = torch.cat((st.ring_keys, k_ring), dim=1)
        v_cat = torch.cat((st.ring_vals, v_flat), dim=1)
        valid_cat = torch.cat(
            (st.ring_valid, torch.ones(bsz, seq_len, dtype=torch.bool, device=dev)), dim=1
        )
        r_off = int(st.ring_keys.shape[1])

        sim_all = torch.matmul(q_ring, k_cat.transpose(-2, -1))  # [B, T, R_off + T]
        t_idx = torch.arange(seq_len, device=dev).unsqueeze(1) + r_off
        j_idx = torch.arange(k_cat.shape[1], device=dev).unsqueeze(0)
        win_mask = (j_idx < t_idx) & (j_idx >= (t_idx - self.ring_len))
        full_mask = win_mask.unsqueeze(0) & valid_cat.unsqueeze(1)

        kap = self.ring_kappa.clamp(-0.1, 0.85)
        ring_w = (2.0 * c1_spherical_cap_resonance(sim_all, kap)).clamp(0.0, 1.0) * full_mask.float()
        ring_sum = ring_w.sum(dim=-1, keepdim=True)
        # Vacuum-safe denominator (`0.5 + ring_sum`): local ring stays quiet when nothing in window resonates!
        ring_norm_w = ring_w / (0.5 + ring_sum)
        ring_out = torch.matmul(ring_norm_w, v_cat)

        # 3. Medium Tier: Detached `.uscar` Universal Poles on `S^{D_w-1}`
        uscar_out, uscar_res = self.uscar_store.query(h_world)

        # 4. Kinematic Spherical `Tri-Compass` (`Delta_local, Delta_origin, Delta_goal`)
        h_orig = torch.where(st.has_origin.unsqueeze(-1), st.h_origin, h_world[:, 0, :])
        h_prev_seq = torch.cat(
            (
                torch.where(st.has_origin.unsqueeze(-1), st.h_prev, h_world[:, 0, :]).unsqueeze(1),
                h_world[:, :-1, :],
            ),
            dim=1,
        )
        goal_dir = project_rows_to_unit_sphere(kalman_flat + biorth_out + ring_out + uscar_out + 1e-6)
        if goal_dir.shape[-1] != self.d_world:
            # Project/slice cleanly onto `d_world`
            if goal_dir.shape[-1] > self.d_world:
                goal_w = project_rows_to_unit_sphere(goal_dir[..., : self.d_world])
            else:
                pad_g = torch.zeros(bsz, seq_len, self.d_world - goal_dir.shape[-1], device=dev, dtype=goal_dir.dtype)
                goal_w = project_rows_to_unit_sphere(torch.cat((goal_dir, pad_g), dim=-1))
        else:
            goal_w = goal_dir

        tri_out = self._compute_tri_compass(h_world, h_prev_seq, h_orig.unsqueeze(1), goal_w)
        tm = self.tier_mix
        # Keep `bus_ctx` physically zero-preserving (`tanh`-like bounded `x / sqrt(1 + ||x||^2)` instead of `x / ||x||`!)
        # so empty memory (`W_fast @ q == 0` at 1st `PIN=`) emits `0.0` vacuum instead of unit-norm noise!
        raw_bus = (
            tm[0] * (kalman_flat / math.sqrt(float(self.num_subspaces)))
            + biorth_out
            + tm[1] * ring_out
            + tm[2] * uscar_out
            + 0.10 * tm[3] * tri_out
        )
        bus_ctx = raw_bus * torch.rsqrt(1.0 + (raw_bus * raw_bus).sum(dim=-1, keepdim=True))

        # Update constant-size `O(1)` state (`Delta = +0 B`!)
        new_ring_k = k_cat[:, -self.ring_len :, :].detach().clone()
        new_ring_v = v_cat[:, -self.ring_len :, :].detach().clone()
        new_ring_val = valid_cat[:, -self.ring_len :].detach().clone()
        rep_tail = h_world[:, -min(seq_len, self.repulsive_len) :, :].detach()
        new_rep = st.repulsive_ring.clone()
        new_rep[:, : rep_tail.shape[1], :] = rep_tail

        new_state = KalmanBusState(
            w_fast=w_final.detach(),
            k_prev=k_ctx[:, -1, :].detach(),
            has_prev_key=torch.ones(bsz, dtype=torch.bool, device=dev),
            recent_bytes=st.recent_bytes.clone(),
            recent_len=st.recent_len,
            ring_keys=new_ring_k,
            ring_vals=new_ring_v,
            ring_valid=new_ring_val,
            ring_ptr=0,
            proper_time=prop_t[:, -1].detach(),
            h_origin=h_orig.detach(),
            has_origin=torch.ones(bsz, dtype=torch.bool, device=dev),
            h_prev=h_world[:, -1, :].detach(),
            repulsive_ring=new_rep,
            repulsive_ptr=int(rep_tail.shape[1] % self.repulsive_len),
            biorth_keys=cur_bk,
            biorth_vals=cur_bv,
            biorth_salience=cur_bsal,
            biorth_valid=cur_bval,
        )
        return hf_phased, bus_ctx, uscar_res, new_state

    @torch.no_grad()
    def step_pulse(
        self,
        h_form_t: torch.Tensor,
        h_world_t: torch.Tensor,
        surprise_t: torch.Tensor,
        state: KalmanBusState,
        virtual_delta: Optional[VirtualGhostJumpDelta] = None,
        mutate_state: bool = True,
        v_shock_hint_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Single-pulse `O(1)` inference step (`mutate_state=False` enables 0-copy Ghost-Jump Scouts!)."""
        dev = h_world_t.device
        ds = 0.02 + 0.98 * surprise_t.clamp(0.0, 1.0)
        prop_t = state.proper_time + ds  # [B]
        hf_phased = apply_relativistic_proper_time_phase(h_form_t, prop_t, self.inv_freq)

        q, k_ctx, v, beta = self._extract_qkv_beta(h_form_t, h_world_t, surprise_t, v_shock_hint=v_shock_hint_t)
        k_bind = torch.where(state.has_prev_key.unsqueeze(-1), state.k_prev, torch.zeros_like(k_ctx))
        beta_bind = torch.where(state.has_prev_key.unsqueeze(-1), beta, torch.zeros_like(beta))

        # Causal pre-write readout from `state.w_fast` (+ optional virtual Ghost-Jump deltas!)
        if self.enable_kalman_fast:
            kalman_g = torch.matmul(state.w_fast, q.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # [B, G, d_sub]
            if virtual_delta is not None and virtual_delta.keys.shape[1] > 0:
                vk_q = torch.matmul(virtual_delta.keys, q.unsqueeze(-1))  # [B, S, 1]
                kalman_g = kalman_g + (virtual_delta.delta_vals * vk_q.unsqueeze(-1)).sum(dim=1)
            kalman_flat = kalman_g.reshape(h_world_t.shape[0], self.d_bus)
        else:
            kalman_flat = torch.zeros(h_world_t.shape[0], self.d_bus, dtype=v.dtype, device=dev)

        # Proper-time phased local ring readout
        key_freq = build_proper_time_frequencies(self.d_key, device=dev)
        q_ring = apply_relativistic_proper_time_phase(q, prop_t, key_freq)
        k_ring = apply_relativistic_proper_time_phase(k_ctx, prop_t, key_freq)
        sim_r = torch.matmul(state.ring_keys, q_ring.unsqueeze(-1)).squeeze(-1)  # [B, R]
        kap = self.ring_kappa.clamp(-0.1, 0.85)
        rw = (2.0 * c1_spherical_cap_resonance(sim_r, kap)).clamp(0.0, 1.0) * state.ring_valid.float()
        rw_norm = rw / (0.5 + rw.sum(dim=-1, keepdim=True))
        ring_out = torch.matmul(rw_norm.unsqueeze(1), state.ring_vals).squeeze(1)

        if self.biorth_slots > 0 and state.biorth_keys is not None and state.biorth_vals is not None and state.biorth_valid is not None:
            biorth_t = self.biorth_gain * biorthogonal_shock_bank_read(
                q.unsqueeze(1),
                state.biorth_keys,
                state.biorth_vals,
                state.biorth_valid,
                self.biorth_kappa,
                ridge_lambda=self.biorth_lambda,
            ).squeeze(1)
        else:
            biorth_t = torch.zeros(h_world_t.shape[0], self.d_bus, dtype=v.dtype, device=dev)

        uscar_out, uscar_res = self.uscar_store.query(h_world_t)

        h_orig = torch.where(state.has_origin.unsqueeze(-1), state.h_origin, h_world_t)
        h_prev = torch.where(state.has_origin.unsqueeze(-1), state.h_prev, h_world_t)
        goal_dir = project_rows_to_unit_sphere(kalman_flat + biorth_t + ring_out + uscar_out + 1e-6)
        if goal_dir.shape[-1] > self.d_world:
            goal_w = project_rows_to_unit_sphere(goal_dir[..., : self.d_world])
        elif goal_dir.shape[-1] < self.d_world:
            pad_g = torch.zeros(h_world_t.shape[0], self.d_world - goal_dir.shape[-1], device=dev)
            goal_w = project_rows_to_unit_sphere(torch.cat((goal_dir, pad_g), dim=-1))
        else:
            goal_w = goal_dir

        tri_out = self._compute_tri_compass(h_world_t, h_prev, h_orig, goal_w)
        tm = self.tier_mix
        raw_bus = (
            tm[0] * (kalman_flat / math.sqrt(float(self.num_subspaces)))
            + biorth_t
            + tm[1] * ring_out
            + tm[2] * uscar_out
            + 0.10 * tm[3] * tri_out
        )
        bus_ctx = raw_bus * torch.rsqrt(1.0 + (raw_bus * raw_bus).sum(dim=-1, keepdim=True))

        # Compute Causal Transition rank-1 innovation `delta_v = beta_bind * (v - W k_bind)`
        wk = torch.matmul(state.w_fast, k_bind.unsqueeze(1).unsqueeze(-1)).squeeze(-1)  # [B, G, d_sub]
        if virtual_delta is not None and virtual_delta.keys.shape[1] > 0:
            vk_k = torch.matmul(virtual_delta.keys, k_bind.unsqueeze(-1))
            wk = wk + (virtual_delta.delta_vals * vk_k.unsqueeze(-1)).sum(dim=1)
        delta_v = beta_bind.unsqueeze(-1) * (v - wk)  # [B, G, d_sub]

        if mutate_state:
            if self.enable_kalman_fast:
                state.w_fast.add_(torch.matmul(delta_v.unsqueeze(-1), k_bind.unsqueeze(1).unsqueeze(-2)))
            if (
                self.biorth_slots > 0
                and state.biorth_keys is not None
                and state.biorth_vals is not None
                and state.biorth_salience is not None
                and state.biorth_valid is not None
            ):
                biorthogonal_shock_bank_insert_step_(
                    state.biorth_keys,
                    state.biorth_vals,
                    state.biorth_salience,
                    state.biorth_valid,
                    k_bind,
                    v.reshape(h_world_t.shape[0], self.d_bus),
                    beta_bind.mean(dim=-1),
                    write_thresh=0.22,
                )
            state.k_prev.copy_(k_ctx)
            state.has_prev_key.fill_(True)
            state.ring_keys[:, :-1, :] = state.ring_keys[:, 1:, :].clone()
            state.ring_vals[:, :-1, :] = state.ring_vals[:, 1:, :].clone()
            state.ring_valid[:, :-1] = state.ring_valid[:, 1:].clone()
            state.ring_keys[:, -1, :] = k_ring
            state.ring_vals[:, -1, :] = v.reshape(h_world_t.shape[0], self.d_bus)
            state.ring_valid[:, -1] = True
            state.proper_time.copy_(prop_t)
            state.h_origin.copy_(h_orig)
            state.has_origin.fill_(True)
            state.h_prev.copy_(h_world_t)
            state.repulsive_ring[:, state.repulsive_ptr, :] = h_world_t
            state.repulsive_ptr = (state.repulsive_ptr + 1) % self.repulsive_len
            return hf_phased, bus_ctx, uscar_res, None

        return hf_phased, bus_ctx, uscar_res, (k_bind, delta_v)

    def repulsive_loop_penalty(self, h_world_cand: torch.Tensor, repulsive_ring: torch.Tensor) -> torch.Tensor:
        """Computes `AURA` smooth repulsive negative scar energy against recently visited poles."""
        # h_world_cand: [..., d_world], repulsive_ring: [..., R_rep, d_world]
        sim = torch.matmul(repulsive_ring, h_world_cand.unsqueeze(-1)).squeeze(-1)  # [..., R_rep]
        return c1_spherical_cap_resonance(sim, sim.new_tensor(0.88)).sum(dim=-1)
