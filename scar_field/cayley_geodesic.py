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


def _halley_spd_inv_direct(g_mat: torch.Tensor, eye_n: torch.Tensor, n_steps: int = 7) -> torch.Tensor:
    """Zero-`cuSOLVER` cubic Halley inverse (`3^n_steps` convergence) for strictly SPD `g_mat`."""
    l1_norm = g_mat.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
    g_inv = (1.85 / l1_norm) * eye_n
    for _ in range(n_steps):
        r_m = eye_n - torch.matmul(g_mat, g_inv)
        g_inv = torch.matmul(g_inv, eye_n + torch.matmul(r_m, eye_n + r_m))
    return g_inv


def _halley_spd_matrix_inv_cuda(m_mat: torch.Tensor, eye_2r: torch.Tensor) -> torch.Tensor:
    """Zero-`cuSOLVER` Tensor-Core `bmm` matrix inverse of `M = I_{2r} + alpha Q^T P` (`det(M) >= 1`).

    Uses 6 cubic Halley iterations (`3^6 = 729`th-order convergence) on the strictly SPD normal
    matrix `G = M^T M` followed by 2 quadratic Schulz polish steps directly on `M`.
    """
    mt = m_mat.transpose(-2, -1)
    g_mat = torch.matmul(mt, m_mat)  # [..., 2r, 2r] SPD with lambda_min > 0
    g_inv = _halley_spd_inv_direct(g_mat, eye_2r, n_steps=6)
    m_inv = torch.matmul(g_inv, mt)
    m_inv = torch.matmul(m_inv, 2.0 * eye_2r - torch.matmul(m_mat, m_inv))
    m_inv = torch.matmul(m_inv, 2.0 * eye_2r - torch.matmul(m_mat, m_inv))
    return m_inv


class _FastCayleySolveAutograd(torch.autograd.Function):
    """Solves `M w = b` with ZERO `cuSOLVER` launches on CUDA/MPS and ZERO solves in backward (`2` tiny `bmm`s)!"""

    @staticmethod
    def forward(ctx, m_mat: torch.Tensor, qt_x: torch.Tensor, eye_2r: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if m_mat.device.type in ("cuda", "mps"):
            m_inv = _halley_spd_matrix_inv_cuda(m_mat, eye_2r)
        else:
            m_inv = torch.linalg.inv(m_mat)
        w_vec = torch.matmul(m_inv, qt_x)
        ctx.save_for_backward(m_inv, w_vec)
        return w_vec

    @staticmethod
    def backward(ctx, grad_w: torch.Tensor):  # type: ignore[override]
        m_inv, w_vec = ctx.saved_tensors
        grad_qt_x = torch.matmul(m_inv.transpose(-2, -1), grad_w)
        grad_m = -torch.matmul(grad_qt_x, w_vec.transpose(-2, -1))
        return grad_m, grad_qt_x, None


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
    if torch.is_grad_enabled() and (m_mat.requires_grad or qt_x.requires_grad):
        w_vec = _FastCayleySolveAutograd.apply(m_mat, qt_x, eye_2r)
    elif m_mat.device.type in ("cuda", "mps"):
        w_vec = torch.matmul(_halley_spd_matrix_inv_cuda(m_mat, eye_2r), qt_x)
    else:
        w_vec = torch.linalg.solve(m_mat, qt_x)
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

        if hc.device.type == "cpu":
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
    """Exact `O(1)` activation-memory backward pass with:
      1. `System 1` Scout `Gate G0` Sparse Token Compaction (skips `dt_k == 0` tokens on `k >= 0` with `0.0` drift!),
      2. Deferred `1x` `E8SphericalLinear` weight-gradient projection (instead of `Kx` sphere/FWHT passes!).
    """

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
        lead_shape = h_form.shape[:-1]
        d_f = h_form.shape[-1]
        d_w = h_world.shape[-1]
        d_b = bus_ctx.shape[-1]
        n_tot = int(h_form.numel() // d_f)

        hf = h_form.reshape(n_tot, d_f).clone()
        hw = h_world.reshape(n_tot, d_w).clone()
        bc = bus_ctx.reshape(n_tot, d_b)
        sm = step_mask.reshape(n_tot, num_steps) if step_mask is not None else None
        # Pre-compute active token index lists with at most ONE host sync per flow (0 syncs inside K-loop!)
        active_indices_per_k: Optional[list[Optional[torch.Tensor]]] = None
        active_counts_per_k: Optional[list[int]] = None
        if sm is not None:
            sm_cpu = sm.detach().cpu()
            active_indices_per_k = []
            active_counts_per_k = []
            for k in range(num_steps):
                idx_cpu = torch.nonzero(sm_cpu[:, k].abs() > 0.0, as_tuple=False).squeeze(-1)
                cnt = int(idx_cpu.numel())
                active_counts_per_k.append(cnt)
                if 0 < cnt < n_tot:
                    active_indices_per_k.append(idx_cpu.to(device=hf.device, non_blocking=True))
                else:
                    active_indices_per_k.append(None)

        with torch.no_grad():
            for k in range(num_steps):
                tau_k = float(k) * dt_base
                if sm is not None and active_counts_per_k is not None and active_indices_per_k is not None:
                    n_act = active_counts_per_k[k]
                    if n_act == 0:
                        continue
                    idx_k = active_indices_per_k[k]
                    if idx_k is not None:
                        dt_sub = sm[idx_k, k : k + 1] * dt_base
                        hf_sub = hf[idx_k]
                        hw_sub = hw[idx_k]
                        bc_sub = bc[idx_k]
                        uf, vf = gen_form(hw_sub, bc_sub, tau_k)
                        hf_sub = cayley_woodbury_step(hf_sub, uf, vf, dt_sub)
                        uw, vw = gen_world(hf_sub, bc_sub, tau_k)
                        hw_sub = cayley_woodbury_step(hw_sub, uw, vw, dt_sub)
                        hf[idx_k] = hf_sub
                        hw[idx_k] = hw_sub
                        continue
                    dt_k: float | torch.Tensor = sm[:, k : k + 1] * dt_base
                else:
                    dt_k = dt_base

                uf, vf = gen_form(hw, bc, tau_k)
                hf = cayley_woodbury_step(hf, uf, vf, dt_k)
                uw, vw = gen_world(hf, bc, tau_k)
                hw = cayley_woodbury_step(hw, uw, vw, dt_k)

        hf_out = hf.reshape(*lead_shape, d_f)
        hw_out = hw.reshape(*lead_shape, d_w)

        # Save ONLY final state at tau=1 (`O(1)` memory regardless of `num_steps`!)
        ctx.save_for_backward(
            hf_out.detach(),
            hw_out.detach(),
            bus_ctx.detach(),
            step_mask.detach() if step_mask is not None else torch.empty(0, device=hf_out.device),
        )
        ctx.num_steps = num_steps
        ctx.has_mask = step_mask is not None
        ctx.active_indices_per_k = active_indices_per_k
        ctx.active_counts_per_k = active_counts_per_k
        ctx.gen_form = gen_form
        ctx.gen_world = gen_world
        return hf_out, hw_out

    @staticmethod
    def backward(ctx, grad_hf: torch.Tensor, grad_hw: torch.Tensor):  # type: ignore[override]
        hf_saved, hw_saved, bus_saved, mask_saved = ctx.saved_tensors
        num_steps: int = ctx.num_steps
        gen_form: HorizonCulledSymplecticGenerator = ctx.gen_form
        gen_world: HorizonCulledSymplecticGenerator = ctx.gen_world
        dt_base = 1.0 / float(max(1, num_steps))

        lead_shape = hf_saved.shape[:-1]
        d_f = hf_saved.shape[-1]
        d_w = hw_saved.shape[-1]
        d_b = bus_saved.shape[-1]
        n_tot = int(hf_saved.numel() // d_f)

        hf_cur = hf_saved.reshape(n_tot, d_f).clone()
        hw_cur = hw_saved.reshape(n_tot, d_w).clone()
        bc_flat = bus_saved.reshape(n_tot, d_b)
        sm_flat = mask_saved.reshape(n_tot, num_steps) if ctx.has_mask else None

        ghf = grad_hf.reshape(n_tot, d_f).clone()
        ghw = grad_hw.reshape(n_tot, d_w).clone()
        grad_bus_flat = torch.zeros_like(bc_flat)

        params_form = list(gen_form.parameters())
        params_world = list(gen_world.parameters())
        params_list = params_form + params_world
        e8_mods = [m for m in (list(gen_form.modules()) + list(gen_world.modules())) if isinstance(m, E8SphericalLinear)]
        e8_param_ids = {id(p) for m in e8_mods for p in m.parameters()}
        direct_form_idx = [i for i, p in enumerate(params_form) if id(p) not in e8_param_ids and p.requires_grad]
        direct_form_params = [params_form[i] for i in direct_form_idx]
        n_form = len(params_form)
        direct_world_idx = [n_form + i for i, p in enumerate(params_world) if id(p) not in e8_param_ids and p.requires_grad]
        direct_world_params = [params_world[i - n_form] for i in direct_world_idx]
        param_grads: list[Optional[torch.Tensor]] = [
            torch.zeros_like(p) if p.requires_grad else None for p in params_list
        ]

        active_indices_per_k = ctx.active_indices_per_k
        active_counts_per_k = ctx.active_counts_per_k

        for m in e8_mods:
            m.begin_deferred_grad_pass()

        try:
            for k in range(num_steps - 1, -1, -1):
                tau_k = float(k) * dt_base
                if sm_flat is not None and active_counts_per_k is not None and active_indices_per_k is not None:
                    n_act = active_counts_per_k[k]
                    if n_act == 0:
                        continue
                    idx_k = active_indices_per_k[k]
                    use_sparse = idx_k is not None
                else:
                    idx_k = None
                    use_sparse = False

                if use_sparse and idx_k is not None:
                    dt_sub: float | torch.Tensor = sm_flat[idx_k, k : k + 1] * dt_base  # type: ignore[index]
                    neg_dt_sub: float | torch.Tensor = -dt_sub
                    hf_c = hf_cur[idx_k]
                    hw_c = hw_cur[idx_k]
                    bc_c = bc_flat[idx_k]
                    ghf_c = ghf[idx_k]
                    ghw_c = ghw[idx_k]
                else:
                    if sm_flat is not None:
                        dt_sub = sm_flat[:, k : k + 1] * dt_base
                        neg_dt_sub = -dt_sub
                    else:
                        dt_sub = dt_base
                        neg_dt_sub = -dt_base
                    hf_c = hf_cur
                    hw_c = hw_cur
                    bc_c = bc_flat
                    ghf_c = ghf
                    ghw_c = ghw

                # Half-step B1: Single `gen_world(hf_mid, bc)` eval both inverts `hw` AND backprops `hw_out`!
                hf_mid_in = hf_c.detach().requires_grad_(True)
                bc_w_in = bc_c.detach().requires_grad_(True)
                with torch.enable_grad():
                    uw, vw = gen_world(hf_mid_in, bc_w_in, tau_k)
                with torch.no_grad():
                    hw_prev_c = cayley_woodbury_step(hw_c, uw.detach(), vw.detach(), neg_dt_sub)
                hw_in = hw_prev_c.detach().requires_grad_(True)
                with torch.enable_grad():
                    hw_out = cayley_woodbury_step(hw_in, uw, vw, dt_sub)

                grads_w = torch.autograd.grad(
                    outputs=hw_out,
                    inputs=(hw_in, hf_mid_in, bc_w_in, *direct_world_params),
                    grad_outputs=ghw_c,
                    allow_unused=True,
                )
                ghw_from_w = grads_w[0] if grads_w[0] is not None else torch.zeros_like(hw_in)
                ghf_mid_tot = ghf_c + grads_w[1] if grads_w[1] is not None else ghf_c
                gbc_w = grads_w[2]

                # Half-step B2: Single `gen_form(hw_prev, bc)` eval both inverts `hf` AND backprops `hf_mid`!
                hw_prev_in = hw_prev_c.detach().requires_grad_(True)
                bc_f_in = bc_c.detach().requires_grad_(True)
                with torch.enable_grad():
                    uf, vf = gen_form(hw_prev_in, bc_f_in, tau_k)
                with torch.no_grad():
                    hf_prev_c = cayley_woodbury_step(hf_c, uf.detach(), vf.detach(), neg_dt_sub)
                hf_in = hf_prev_c.detach().requires_grad_(True)
                with torch.enable_grad():
                    hf_mid = cayley_woodbury_step(hf_in, uf, vf, dt_sub)

                grads_f = torch.autograd.grad(
                    outputs=hf_mid,
                    inputs=(hf_in, hw_prev_in, bc_f_in, *direct_form_params),
                    grad_outputs=ghf_mid_tot,
                    allow_unused=True,
                )
                new_ghf = grads_f[0] if grads_f[0] is not None else torch.zeros_like(hf_in)
                new_ghw = ghw_from_w + grads_f[1] if grads_f[1] is not None else ghw_from_w
                gbc_f = grads_f[2]
                if gbc_w is not None and gbc_f is not None:
                    new_gbc: Optional[torch.Tensor] = gbc_w + gbc_f
                else:
                    new_gbc = gbc_w if gbc_w is not None else gbc_f

                if use_sparse and idx_k is not None:
                    ghf[idx_k] = new_ghf
                    ghw[idx_k] = new_ghw
                    if new_gbc is not None:
                        grad_bus_flat[idx_k] = grad_bus_flat[idx_k] + new_gbc
                    hf_cur[idx_k] = hf_prev_c
                    hw_cur[idx_k] = hw_prev_c
                else:
                    ghf = new_ghf
                    ghw = new_ghw
                    if new_gbc is not None:
                        grad_bus_flat = grad_bus_flat + new_gbc
                    hf_cur = hf_prev_c
                    hw_cur = hw_prev_c

                for loc_i, pg in enumerate(grads_w[3:]):
                    if pg is not None:
                        p_idx = direct_world_idx[loc_i]
                        cur_pg = param_grads[p_idx]
                        param_grads[p_idx] = pg if cur_pg is None else (cur_pg + pg)
                for loc_i, pg in enumerate(grads_f[3:]):
                    if pg is not None:
                        p_idx = direct_form_idx[loc_i]
                        cur_pg = param_grads[p_idx]
                        param_grads[p_idx] = pg if cur_pg is None else (cur_pg + pg)
        finally:
            e8_final_grads: dict[int, torch.Tensor] = {}
            for m in e8_mods:
                e8_final_grads.update(m.finalize_deferred_grad_pass())

        for i, p in enumerate(params_list):
            pid = id(p)
            if pid in e8_final_grads:
                param_grads[i] = e8_final_grads[pid]

        return (
            ghf.reshape(*lead_shape, d_f),
            ghw.reshape(*lead_shape, d_w),
            grad_bus_flat.reshape(*lead_shape, d_b),
            None,
            None,
            None,
            None,
            *param_grads,
        )


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
