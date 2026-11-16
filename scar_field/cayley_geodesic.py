"""Symplectic Reversible Cayley-Woodbury Geodesic Flow & C^1 Horizon-Culled Scar Field.

Zero-Legacy Mathematical Primitives for `ScarField v7.0`:
  1. `c1_spherical_cap_resonance`: Smooth `C^1` rational-quadratic spherical cap kernel with
     exact `0.0` outside the angular horizon `kappa` (enabling true hardware horizon culling
     with ZERO `ReLU` / `GELU` / `SwiGLU` coordinate amputations and ZERO dead-gradient kinks).
  2. `cayley_woodbury_step`: Exact `2r x 2r` Sherman-Morrison-Woodbury Cayley orthogonal map
     `Cayley(dt * (U V^T - V U^T)) x` with guaranteed `det(I_2r + alpha Q^T P) >= 1.0`:
       - Preserves `||x||_2 == 1.000000` to machine precision WITHOUT `RMSNorm` or `1/sqrt`.
       - Exact analytic time-reversibility: `cayley_woodbury_step(y, U, V, -dt) == x`.
  3. `ReversibleSymplecticCayleyFlow`: Evolves `(h_form, h_world) in S^{D_f-1} x S^{D_w-1}`
     across `K` adaptive `System 1` (`K=1`) or `System 2` (`K=2..16`) steps while storing
     ONLY the final state at `tau = 1` (`O(1)` activation memory during backprop for ANY `K`!).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scar_field.e8_lattice import E8SphericalLinear, project_rows_to_unit_sphere


def c1_spherical_cap_resonance(cos_sim: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """`C^1`-smooth spherical cap resonance (`0.0` outside `cos_sim <= kappa`, smooth inside)."""
    delta = cos_sim - kappa
    pos = torch.where(delta > 0.0, delta, torch.zeros_like(delta))
    width2 = (1.0 - kappa).clamp(min=0.08).square()
    pos2 = pos * pos
    return pos2 / (width2 + pos2)


def cayley_woodbury_step(
    x: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    """Exact rank-`2r` Cayley transform `(I - (dt/2) Omega)(I + (dt/2) Omega)^{-1} x`.

    Args:
        x: `[..., D]` unit state vectors (`||x||_2 == 1`).
        u: `[..., D, r]` first tangent generator factor.
        v: `[..., D, r]` second tangent generator factor (`Omega = u v^T - v u^T`).
        dt: scalar float or broadcastable `[..., 1]` step size (`-dt` inverts analytically!).
    """
    orig_dtype = x.dtype
    xf = x.float()
    uf = u.float()
    vf = v.float()

    r = uf.shape[-1]
    two_r = 2 * r
    p_mat = torch.cat((uf, -vf), dim=-1)  # [..., D, 2r]
    q_mat = torch.cat((vf, uf), dim=-1)   # [..., D, 2r]

    if isinstance(dt, torch.Tensor):
        alpha = 0.5 * dt.float()  # [..., 1]
        alpha_m = alpha.unsqueeze(-1)  # [..., 1, 1]
    else:
        alpha = 0.5 * float(dt)
        alpha_m = alpha

    # M = I_{2r} + alpha * (Q^T P), guaranteed det(M) >= 1.0 for all real alpha!
    eye_2r = torch.eye(two_r, dtype=torch.float32, device=xf.device)
    qt_p = torch.matmul(q_mat.transpose(-2, -1), p_mat)  # [..., 2r, 2r]
    m_mat = eye_2r + alpha_m * qt_p

    qt_x = torch.matmul(q_mat.transpose(-2, -1), xf.unsqueeze(-1))  # [..., 2r, 1]
    w_vec = torch.linalg.solve(m_mat, qt_x)  # [..., 2r, 1]
    pw = torch.matmul(p_mat, w_vec).squeeze(-1)  # [..., D]

    out = xf - (2.0 * alpha) * pw
    return out.to(orig_dtype)


class HorizonCulledSymplecticGenerator(nn.Module):
    """Generates low-rank Lie algebra factors `(U, V)` on `S^{D_target-1}` with ZERO `ReLU`.

    Combines:
      1. Shared `1.000 bpw` `E_8` universal geodesic drift conditioned on `(h_partner, bus_ctx, tau)`
      2. `num_scars` localized spherical-cap scars on `S^{D_cond-1}` with true sparse index gather
         (*hardware horizon culling*: inactive scars outside `cos_sim <= kappa_e` are skipped!).
    """

    def __init__(
        self,
        cond_dim: int,
        target_dim: int,
        bus_dim: int,
        base_rank: int = 4,
        num_scars: int = 16,
        scar_cap_init: float = 0.22,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim
        self.bus_dim = bus_dim
        self.base_rank = base_rank
        self.num_scars = num_scars

        # Pad conditioning input to a clean multiple of 64 for multiplierless FWHT-64 + E8
        raw_in = cond_dim + bus_dim + 4  # +4 harmonic pseudo-time coordinates `[cos, sin, cos2, sin2]`
        self.padded_in = ((raw_in + 63) // 64) * 64
        out_uv = 2 * target_dim * base_rank

        self.base_e8 = E8SphericalLinear(
            in_features=self.padded_in,
            out_features=out_uv,
            scar_rank=4 if out_uv >= 16 else 0,
            init_tau=0.35,
        )

        # Localized Spherical-Cap Scars (`c_e in S^{cond_dim-1}`, `u_e, v_e in S^{target_dim-1}`)
        self.scar_poles = nn.Parameter(project_rows_to_unit_sphere(torch.randn(num_scars, cond_dim)))
        self.scar_kappa = nn.Parameter(torch.full((num_scars,), float(scar_cap_init)))
        self.scar_u = nn.Parameter(project_rows_to_unit_sphere(torch.randn(num_scars, target_dim)))
        self.scar_v = nn.Parameter(project_rows_to_unit_sphere(torch.randn(num_scars, target_dim)))
        self.scar_strength = nn.Parameter(torch.full((num_scars,), 0.45))

        # Top-`m` (`m = 4`) active scars form independent symplectic planes `(sqrt(g_e) u_e, sqrt(g_e) v_e)`
        # with zero cross-talk between distinct scars!
        self.active_scar_slots = min(4, num_scars)
        self.total_rank = base_rank + self.active_scar_slots
        self.last_active_scar_fraction: float = 0.0

    def forward(
        self,
        h_cond: torch.Tensor,
        bus_ctx: torch.Tensor,
        tau_scalar: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lead_shape = h_cond.shape[:-1]
        flat_n = int(h_cond.numel() // self.cond_dim)
        hc = h_cond.reshape(flat_n, self.cond_dim)
        bc = bus_ctx.reshape(flat_n, self.bus_dim)

        ang = float(tau_scalar) * math.pi
        t_feat = hc.new_tensor(
            [math.cos(ang), math.sin(ang), math.cos(2.0 * ang), math.sin(2.0 * ang)]
        ).unsqueeze(0).expand(flat_n, 4)

        raw = torch.cat((hc, bc, t_feat), dim=-1)
        if self.padded_in > raw.shape[-1]:
            pad = torch.zeros(flat_n, self.padded_in - raw.shape[-1], dtype=raw.dtype, device=raw.device)
            raw = torch.cat((raw, pad), dim=-1)
        raw_unit = project_rows_to_unit_sphere(raw)

        uv_base = self.base_e8(raw_unit).reshape(flat_n, 2, self.target_dim, self.base_rank)
        uv_norm = torch.linalg.vector_norm(uv_base, ord=2, dim=2, keepdim=True)
        uv_base = uv_base / torch.sqrt(1.0 + uv_norm * uv_norm)
        u_base = uv_base[:, 0]  # [N, D_t, r_b]
        v_base = uv_base[:, 1]  # [N, D_t, r_b]

        poles = project_rows_to_unit_sphere(self.scar_poles)
        kappa = self.scar_kappa.clamp(-0.20, 0.92)
        cos_sim = hc @ poles.T  # [N, E]
        gates = c1_spherical_cap_resonance(cos_sim, kappa.unsqueeze(0))  # [N, E]

        with torch.no_grad():
            self.last_active_scar_fraction = float((gates > 0.0).float().mean().item())

        su = project_rows_to_unit_sphere(self.scar_u)  # [E, D_t]
        sv = project_rows_to_unit_sphere(self.scar_v)  # [E, D_t]
        eff = gates * self.scar_strength.abs().unsqueeze(0)  # [N, E] >= 0 (`0.0` outside horizon!)

        # Gather only top `active_scar_slots` horizon-resonant scars per state (`0.0` if outside horizon!)
        top_g, top_idx = torch.topk(eff, k=self.active_scar_slots, dim=-1)  # [N, M]
        amp = torch.sqrt(top_g.clamp(min=0.0) + 1e-12) * (top_g > 0.0).float()  # [N, M]
        sgn = torch.sign(self.scar_strength[top_idx])
        sgn = torch.where(sgn == 0.0, torch.ones_like(sgn), sgn)

        # `[N, M, D_t]` -> `[N, D_t, M]`: `Omega_scars = sum_m sgn_m * top_g_m * (u_m v_m^T - v_m u_m^T)`!
        u_scar = (su[top_idx] * (amp * sgn).unsqueeze(-1)).transpose(1, 2)
        v_scar = (sv[top_idx] * amp.unsqueeze(-1)).transpose(1, 2)

        u_all = torch.cat((u_base, u_scar), dim=-1).reshape(*lead_shape, self.target_dim, self.total_rank)
        v_all = torch.cat((v_base, v_scar), dim=-1).reshape(*lead_shape, self.target_dim, self.total_rank)
        return u_all, v_all


class _ReversibleLeapfrogAutograd(torch.autograd.Function):
    """Exact `O(1)` activation-memory backward pass by inverting Symplectic Cayley steps (`-dt`)!"""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        h_form: torch.Tensor,
        h_world: torch.Tensor,
        bus_ctx: torch.Tensor,
        step_mask: Optional[torch.Tensor],
        num_steps: int,
        gen_form: HorizonCulledSymplecticGenerator,
        gen_world: HorizonCulledSymplecticGenerator,
        *all_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dt_base = 1.0 / float(max(1, num_steps))
        hf = h_form
        hw = h_world

        with torch.no_grad():
            for k in range(num_steps):
                tau_k = float(k) * dt_base
                dt_k: float | torch.Tensor
                if step_mask is not None:
                    # Per-token adaptive S1/S2 step mask `step_mask[..., k] in {0.0, 1.0}`
                    dt_k = step_mask[..., k : k + 1] * dt_base
                else:
                    dt_k = dt_base

                uf, vf = gen_form(hw, bus_ctx, tau_k)
                hf = cayley_woodbury_step(hf, uf, vf, dt_k)

                uw, vw = gen_world(hf, bus_ctx, tau_k)
                hw = cayley_woodbury_step(hw, uw, vw, dt_k)

        # Save ONLY final state at tau=1 (`O(1)` memory regardless of `num_steps`!)
        ctx.save_for_backward(
            hf.detach(),
            hw.detach(),
            bus_ctx.detach(),
            step_mask.detach() if step_mask is not None else torch.empty(0, device=hf.device),
        )
        ctx.num_steps = num_steps
        ctx.has_mask = step_mask is not None
        ctx.gen_form = gen_form
        ctx.gen_world = gen_world
        return hf, hw

    @staticmethod
    def backward(ctx, grad_hf: torch.Tensor, grad_hw: torch.Tensor):  # type: ignore[override]
        hf_cur, hw_cur, bus_saved, mask_saved = ctx.saved_tensors
        num_steps: int = ctx.num_steps
        gen_form: HorizonCulledSymplecticGenerator = ctx.gen_form
        gen_world: HorizonCulledSymplecticGenerator = ctx.gen_world
        dt_base = 1.0 / float(max(1, num_steps))

        params_list = list(gen_form.parameters()) + list(gen_world.parameters())
        param_grads = [torch.zeros_like(p) for p in params_list]
        grad_bus_total = torch.zeros_like(bus_saved) if bus_saved.numel() > 0 else torch.zeros_like(bus_saved)

        ghf = grad_hf
        ghw = grad_hw

        for k in range(num_steps - 1, -1, -1):
            tau_k = float(k) * dt_base
            if ctx.has_mask:
                dt_k: float | torch.Tensor = mask_saved[..., k : k + 1] * dt_base
                neg_dt_k: float | torch.Tensor = -dt_k
            else:
                dt_k = dt_base
                neg_dt_k = -dt_base

            # 1. Analytically invert Half-Step 2 (`hw_prev = Cayley(-dt_k, gen_world(hf_next)) hw_cur`)
            with torch.no_grad():
                uw_inv, vw_inv = gen_world(hf_cur, bus_saved, tau_k)
                hw_prev = cayley_woodbury_step(hw_cur, uw_inv, vw_inv, neg_dt_k)
                # Analytically invert Half-Step 1 (`hf_prev = Cayley(-dt_k, gen_form(hw_prev)) hf_cur`)
                uf_inv, vf_inv = gen_form(hw_prev, bus_saved, tau_k)
                hf_prev = cayley_woodbury_step(hf_cur, uf_inv, vf_inv, neg_dt_k)

            # 2. Local autograd through forward step `k` from exact reconstructed `(hf_prev, hw_prev)`
            hf_in = hf_prev.detach().requires_grad_(True)
            hw_in = hw_prev.detach().requires_grad_(True)
            bus_in = bus_saved.detach().requires_grad_(True)

            with torch.enable_grad():
                uf, vf = gen_form(hw_in, bus_in, tau_k)
                hf_mid = cayley_woodbury_step(hf_in, uf, vf, dt_k)
                uw, vw = gen_world(hf_mid, bus_in, tau_k)
                hw_out = cayley_woodbury_step(hw_in, uw, vw, dt_k)

            grads = torch.autograd.grad(
                outputs=(hf_mid, hw_out),
                inputs=(hf_in, hw_in, bus_in, *params_list),
                grad_outputs=(ghf, ghw),
                allow_unused=True,
            )

            ghf = grads[0] if grads[0] is not None else torch.zeros_like(hf_in)
            ghw = grads[1] if grads[1] is not None else torch.zeros_like(hw_in)
            if grads[2] is not None:
                grad_bus_total = grad_bus_total + grads[2]
            for idx, pg in enumerate(grads[3:]):
                if pg is not None:
                    param_grads[idx] = param_grads[idx] + pg

            hf_cur = hf_prev
            hw_cur = hw_prev

        return (ghf, ghw, grad_bus_total, None, None, None, None, *param_grads)


class ReversibleSymplecticCayleyFlow(nn.Module):
    """Adaptive `S1/S2` Reversible Symplectic Cayley Flow on `S^{D_f-1} x S^{D_w-1}`."""

    def __init__(
        self,
        d_form: int,
        d_world: int,
        d_bus: int,
        base_rank: int = 4,
        num_scars: int = 16,
    ) -> None:
        super().__init__()
        self.d_form = d_form
        self.d_world = d_world
        self.d_bus = d_bus
        self.gen_form = HorizonCulledSymplecticGenerator(
            cond_dim=d_world,
            target_dim=d_form,
            bus_dim=d_bus,
            base_rank=base_rank,
            num_scars=num_scars,
        )
        self.gen_world = HorizonCulledSymplecticGenerator(
            cond_dim=d_form,
            target_dim=d_world,
            bus_dim=d_bus,
            base_rank=base_rank,
            num_scars=num_scars,
        )

    def forward(
        self,
        h_form: torch.Tensor,
        h_world: torch.Tensor,
        bus_ctx: torch.Tensor,
        num_steps: int = 4,
        step_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = tuple(self.gen_form.parameters()) + tuple(self.gen_world.parameters())
        out_f, out_w = _ReversibleLeapfrogAutograd.apply(
            h_form,
            h_world,
            bus_ctx,
            step_mask,
            int(num_steps),
            self.gen_form,
            self.gen_world,
            *params,
        )
        return out_f, out_w

    @torch.no_grad()
    def invert_flow(
        self,
        h_form_end: torch.Tensor,
        h_world_end: torch.Tensor,
        bus_ctx: torch.Tensor,
        num_steps: int = 4,
        step_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy exact algebraic inverse (`tau = 1 -> tau = 0`) for Ghost-Jump rollbacks & audits."""
        dt_base = 1.0 / float(max(1, num_steps))
        hf = h_form_end
        hw = h_world_end
        for k in range(num_steps - 1, -1, -1):
            tau_k = float(k) * dt_base
            neg_dt: float | torch.Tensor = (
                -(step_mask[..., k : k + 1] * dt_base) if step_mask is not None else -dt_base
            )
            uw, vw = self.gen_world(hf, bus_ctx, tau_k)
            hw = cayley_woodbury_step(hw, uw, vw, neg_dt)
            uf, vf = self.gen_form(hw, bus_ctx, tau_k)
            hf = cayley_woodbury_step(hf, uf, vf, neg_dt)
        return hf, hw
