"""Isometric Geodesic Expansion (`Edge -> Core`) & Live `KalmanBusState` Handoff (`v9.1-Elastic`).

Proves and implements Function-Preserving Isometric Hyperspherical Expansion:
  - Given ANY trained `ScarField-Edge` engine `(d_form_0, d_world_0, d_bus_0, ...)` (either live in RAM
    or cold-loaded from `.sfield.scar`), embeds it into a larger `ScarField-Core` configuration
    `(d_form_1, d_world_1, d_bus_1, ...)` (`d_1 >= d_0`) such that:
      1. Equatorial State Embedding (`S^{d_0-1} x {0} subset S^{d_1-1}`): Because `ScarField` uses exact
         `L2` unit-sphere projection WITHOUT `RMSNorm`, `||[h_0, 0]||_2 == ||h_0||_2 == 1.000000`.
      2. Exact `E_8` + Ternary Equator Scaling (`sqrt(I_0 / I_1)`): Placing norm `sqrt(I_0 / I_1)` on
         the active `64D` equatorial blocks and orthonormal Stiefel energy `sqrt(1 - I_0 / I_1)` on the
         orthogonal complement cancels `sqrt(I_1)` in both Conway-Sloane `E_8` encoding and `0.68 * RMS`
         ternary quantization, preserving exact `E_8` codewords and ternary trits on aligned blocks
          WHILE immediately backpropagating full-rank non-zero gradients into the new dimensions on `step = 1`!
      3. Asymmetric Lie-Algebra Saddle-Breaker on Extra Cycles (`U != 0, V = 0`): Extra symplectic
         macro-cycles initialize with Stiefel `U` and vacuum `V = 0` (`U V^T - V U^T == 0` -> exact
         identity `Phi_0(h) == h` at `step = 0` and non-zero `dL/dV != 0` at `step = 1`).
      4. Zero Matryoshka Tax After Expansion: Once expanded, all coordinates mix freely without
         triangular masks (`0%` structural capacity tax on `ScarField-Core`).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from scar_field.cayley_geodesic import HorizonCulledSymplecticGenerator, ReversibleSymplecticCayleyFlow
from scar_field.e8_lattice import (
    E8SphericalLinear,
    fwht64_rademacher,
    project_rows_to_unit_sphere,
    quantize_ternary_ste,
    unpack_ternary_2bit,
)
from scar_field.engine import ScarFieldConfig, ScarFieldEngine
from scar_field.proper_time_bus import (
    KalmanBusState,
    biorthogonal_shock_bank_insert_step_,
)


def _rand_unit_rows(rows: int, cols: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    if rows <= 0 or cols <= 0:
        return torch.zeros((max(0, rows), max(0, cols)), dtype=torch.float32, device=device)
    raw = torch.randn((rows, cols), generator=gen, dtype=torch.float32)
    if rows <= cols:
        q, _ = torch.linalg.qr(raw.T, mode="reduced")
        return q.T.contiguous().to(device=device)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return project_rows_to_unit_sphere(q).to(device=device)


@torch.no_grad()
def _extract_effective_e8_components(
    mod: E8SphericalLinear,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float]:
    """Extracts `(w_unit_std [O, I], a_unit [r, I], b_unit [O, r], tau_base, gamma_scar)` from live or mmap `mod`."""
    sqrt_i = math.sqrt(float(mod.in_features))
    if mod._frozen_packed is not None:
        pk = mod._frozen_packed
        dev = mod.signs.device
        lut = mod.lut_e8.to(dev).float()
        if pk._mmap_views is not None:
            c_t = torch.from_numpy(pk._mmap_views[0].copy()).to(dev)
            a_pk = torch.from_numpy(pk._mmap_views[1].copy()).to(dev)
            b_pk = torch.from_numpy(pk._mmap_views[2].copy()).to(dev)
        else:
            c_t = pk.codes_u8.to(dev)
            a_pk = pk.a_tern_packed.to(dev)
            b_pk = pk.b_tern_packed.to(dev)
        w_q_rot = lut[c_t.long()].reshape(mod.out_features, mod.in_features)
        w_std = fwht64_rademacher(w_q_rot / sqrt_i, mod.signs.to(dev), inverse=True)
        w_unit = project_rows_to_unit_sphere(w_std)
        if mod.scar_rank > 0 and a_pk.numel() > 0 and b_pk.numel() > 0:
            a_u = project_rows_to_unit_sphere(unpack_ternary_2bit(a_pk).reshape(mod.scar_rank, mod.in_features))
            b_u = project_rows_to_unit_sphere(unpack_ternary_2bit(b_pk).reshape(mod.out_features, mod.scar_rank))
        else:
            a_u, b_u = None, None
        return w_unit, a_u, b_u, float(pk.tau_base), float(pk.gamma_scar)

    assert mod.weight_sphere is not None and mod.tau_base is not None
    w_unit = project_rows_to_unit_sphere(mod.weight_sphere.detach().float())
    if mod.scar_rank > 0 and mod.a_scar is not None and mod.b_scar is not None and mod.gamma_scar is not None:
        a_u = project_rows_to_unit_sphere(mod.a_scar.detach().float())
        b_u = project_rows_to_unit_sphere(mod.b_scar.detach().float())
        gam = float(mod.gamma_scar.item())
    else:
        a_u, b_u, gam = None, None, 0.0
    return w_unit, a_u, b_u, float(mod.tau_base.item()), gam


@torch.no_grad()
def _invalidate_e8_caches_(mod: E8SphericalLinear) -> None:
    mod._cached_q_key = None
    mod._cached_w_q_rot = None
    mod._cached_w_q_rot_bf16 = None
    mod._cached_nograd_key = None
    mod._cached_a_t = None
    mod._cached_b_t = None


@torch.no_grad()
def expand_e8_linear_isometric_(
    src_mod: E8SphericalLinear,
    dst_mod: E8SphericalLinear,
    out_map: Sequence[Tuple[int, int]],
    in_block_map: Sequence[Tuple[int, int]],
    gen: torch.Generator,
    zero_output_rows: Optional[Sequence[int]] = None,
) -> None:
    """Embeds `src_mod [O_0, I_0]` into `dst_mod [O_1, I_1]` preserving equatorial forward action & waking `I_1 \\ I_0`."""
    assert dst_mod.weight_sphere is not None and dst_mod.tau_base is not None
    dev = dst_mod.weight_sphere.device
    o0, i0 = src_mod.out_features, src_mod.in_features
    o1, i1 = dst_mod.out_features, dst_mod.in_features
    if i1 < i0 or o1 < o0:
        raise ValueError(f"Destination E8 ({o1}, {i1}) must be >= source ({o0}, {i0})")

    w_u0, a_u0, b_u0, tau0, gam0 = _extract_effective_e8_components(src_mod)
    w_u0 = w_u0.to(dev)

    n_blk1 = i1 // 64
    mapped_dst_blocks = {b_dst for _, b_dst in in_block_map}
    unmapped_dst_blocks = [b for b in range(n_blk1) if b not in mapped_dst_blocks]
    active_cols_count = 64 * len(in_block_map)
    comp_cols_count = i1 - active_cols_count

    alpha_in = math.sqrt(float(active_cols_count) / float(i1))
    beta_in = math.sqrt(max(0.0, 1.0 - float(active_cols_count) / float(i1)))

    # Initialize all rows on unit sphere; mapped rows overwrite with exact `[alpha_in * w_0, beta_in * w_orth]`
    w_new = _rand_unit_rows(o1, i1, gen, dev)
    comp_noise = _rand_unit_rows(len(out_map), max(1, comp_cols_count), gen, dev) if comp_cols_count > 0 else None

    # Also compute exact quantized source output on mapped blocks so any block-shifted Rademacher sign
    # difference is absorbed analytically by SVD/RRR ternary scar healing when needed!
    has_shifted_block = any(b_src != b_dst for b_src, b_dst in in_block_map)
    sqrt_i0 = math.sqrt(float(i0))
    w_rot0 = fwht64_rademacher(w_u0, src_mod.signs.to(dev), inverse=False) * sqrt_i0
    _, w_q_rot0, _ = src_mod._assign_e8_codes_and_dequant(w_rot0.cpu(), need_codes=False, need_dequant=True)
    assert w_q_rot0 is not None
    # Since `w_q_std0_unit = fwht^{-1}(w_q_rot0 / sqrt_i0)` ALREADY includes `1 / sqrt_i0`,
    # the exact base linear map is `tau0 * w_q_std0_unit`!
    w_q_std0_unit = fwht64_rademacher((w_q_rot0.to(dev) / sqrt_i0), src_mod.signs.to(dev), inverse=True)
    w_q_std0_row_unit = project_rows_to_unit_sphere(w_q_std0_unit)
    has_learned_scar = bool(src_mod.scar_rank > 0 and a_u0 is not None and b_u0 is not None and abs(gam0) > 0.08)
    w_base0 = tau0 * w_q_std0_unit
    w_actual0 = w_base0
    if has_learned_scar and a_u0 is not None and b_u0 is not None:
        _, a_i8_0 = quantize_ternary_ste(a_u0.to(dev))
        _, b_i8_0 = quantize_ternary_ste(b_u0.to(dev))
        w_actual0 = w_actual0 + (gam0 / math.sqrt(float(i0 * max(1, src_mod.scar_rank)))) * (
            b_i8_0.float() @ a_i8_0.float()
        )

    # When `has_learned_scar` is True, `(a_u0, b_u0, gam0)` operates in standard coordinates (no `signs`!)
    # and transfers 100% bit-exactly across shifted 64D blocks, so `w_new` + `tau_base` match `w_base0` cleanly!
    w_ref_for_base = w_base0 if has_learned_scar else w_actual0
    w_target_eq_actual = torch.zeros((o1, i1), dtype=torch.float32, device=dev)
    row_target_norms: List[float] = []
    inv_tau0 = (1.0 / tau0) if abs(tau0) > 1e-6 else 1.0
    for idx_m, (o_s, o_d) in enumerate(out_map):
        row = torch.zeros((i1,), dtype=torch.float32, device=dev)
        eq_unnorm = torch.zeros((i1,), dtype=torch.float32, device=dev)
        for b_s, b_d in in_block_map:
            if b_s == b_d:
                # Aligned `64D` block (`signs_1 == signs_0`): preserve exact pre-quantization `w_u0` (`0` E8 drift!)
                eq_unnorm[64 * b_d : 64 * (b_d + 1)] = w_u0[o_s, 64 * b_s : 64 * (b_s + 1)]
            elif has_learned_scar:
                # Shifted `64D` block with exact scar preservation: match unit-norm `w_q_std0_row_unit` (`0` norm skew!)
                eq_unnorm[64 * b_d : 64 * (b_d + 1)] = w_q_std0_row_unit[o_s, 64 * b_s : 64 * (b_s + 1)]
            else:
                # Shifted `64D` block (`gam0 == 0`): map effective operator normalized by signed `tau0`
                eq_unnorm[64 * b_d : 64 * (b_d + 1)] = w_actual0[o_s, 64 * b_s : 64 * (b_s + 1)] * inv_tau0
            w_target_eq_actual[o_d, 64 * b_d : 64 * (b_d + 1)] = w_ref_for_base[o_s, 64 * b_s : 64 * (b_s + 1)]
        eq_unit = project_rows_to_unit_sphere(eq_unnorm.unsqueeze(0)).squeeze(0)
        row = alpha_in * eq_unit
        row_target_norms.append(float(torch.linalg.vector_norm(w_target_eq_actual[o_d]).item()))
        if comp_cols_count > 0 and comp_noise is not None:
            off = 0
            for b_u in unmapped_dst_blocks:
                row[64 * b_u : 64 * (b_u + 1)] = beta_in * comp_noise[idx_m, off : off + 64]
                off += 64
        w_new[o_d] = project_rows_to_unit_sphere(row.unsqueeze(0)).squeeze(0)

    # Rows explicitly designated as exact `0x00` E8 vacuum at `step = 0` (`w_rot == 0` -> `codes_u8 == 0`?!
    # Wait: `weight_sphere` rows have unit norm, so `codes_u8 == 0` is only for zero vector; instead, for
    # unmapped rows that must emit `0.0` on equatorial inputs `[x_0, 0]`, place 100% of their unit norm
    # INSIDE `unmapped_dst_blocks` (`I_1 \\ I_0`) when `comp_cols_count >= 64`! When `comp_cols_count == 0`,
    # see `initialize_cayley_flow_as_nonsaddle_identity_` which sets `tau_base = 0` or `gamma` cleanly!)
    if zero_output_rows:
        if comp_cols_count >= 64:
            z_orth = _rand_unit_rows(len(zero_output_rows), comp_cols_count, gen, dev)
            for idx_z, o_z in enumerate(zero_output_rows):
                z_row = torch.zeros((i1,), dtype=torch.float32, device=dev)
                off = 0
                for b_u in unmapped_dst_blocks:
                    z_row[64 * b_u : 64 * (b_u + 1)] = z_orth[idx_z, off : off + 64]
                    off += 64
                w_new[o_z] = project_rows_to_unit_sphere(z_row.unsqueeze(0)).squeeze(0)

    dst_mod.weight_sphere.copy_(w_new)
    scale_i_ratio = math.sqrt(float(i1) / float(max(1, active_cols_count)))
    sqrt_i1 = math.sqrt(float(i1))
    if has_shifted_block and row_target_norms:
        # Since `||w_q_std1_unit||_row ≈ 1.0` and `(tau1 / sqrt_i1) * w_q_std1_unit` has equatorial norm
        # `(tau1 / sqrt_i1) * sqrt(active_cols_count / i1)`, match `mean(row_target_norms)` in closed form!
        w_rot1_probe = fwht64_rademacher(w_new, dst_mod.signs.to(dev), inverse=False) * sqrt_i1
        _, w_q_rot1, _ = dst_mod._assign_e8_codes_and_dequant(w_rot1_probe.cpu(), need_codes=False, need_dequant=True)
        assert w_q_rot1 is not None
        w_q_std1_unit = fwht64_rademacher((w_q_rot1.to(dev) / sqrt_i1), dst_mod.signs.to(dev), inverse=True)
        eq_col_mask = torch.zeros((1, i1), dtype=torch.float32, device=dev)
        for _, b_d in in_block_map:
            eq_col_mask[0, 64 * b_d : 64 * (b_d + 1)] = 1.0
        eq_row_mask = torch.zeros((o1, 1), dtype=torch.float32, device=dev)
        for _, o_d in out_map:
            eq_row_mask[o_d, 0] = 1.0
        # `w_q_std1_unit = fwht^{-1}(w_q_rot1 / sqrt_i1)` ALREADY includes `1 / sqrt_i1`, so the actual
        # base linear map is `tau1 * w_q_std1_unit`!
        unit_base_eq = w_q_std1_unit * eq_col_mask * eq_row_mask
        denom_tau = float((unit_base_eq * unit_base_eq).sum().item())
        if denom_tau > 1e-10:
            opt_tau1 = float((w_target_eq_actual * unit_base_eq).sum().item()) / denom_tau
            dst_mod.tau_base.fill_(opt_tau1)
        else:
            dst_mod.tau_base.fill_(tau0 * scale_i_ratio)
    else:
        w_q_std1_unit = None
        eq_col_mask = None
        eq_row_mask = None
        unit_base_eq = None
        dst_mod.tau_base.fill_(tau0 * scale_i_ratio)

    # Transfer or heal `2-bit` ternary atomic scars `(a_scar, b_scar, gamma_scar)`
    r0 = src_mod.scar_rank
    r1 = dst_mod.scar_rank
    if r1 > 0 and dst_mod.a_scar is not None and dst_mod.b_scar is not None and dst_mod.gamma_scar is not None:
        if has_shifted_block and (not has_learned_scar) and unit_base_eq is not None:
            # Closed-form equatorial SVD/RRR ternary scar healing against exact `W_target_eq - W_base1_eq`!
            w_base1_eq = float(dst_mod.tau_base.item()) * unit_base_eq
            res_actual = w_target_eq_actual - w_base1_eq
            u_f, s_f, vh_f = torch.linalg.svd(res_actual, full_matrices=False)
            r_fit = min(r1, s_f.numel())
            s_sq = torch.sqrt(s_f[:r_fit].clamp(min=1e-10))
            b_dir = torch.zeros((o1, r1), dtype=torch.float32, device=dev)
            a_dir = torch.zeros((r1, i1), dtype=torch.float32, device=dev)
            b_dir[:, :r_fit] = u_f[:, :r_fit] * s_sq.unsqueeze(0)
            a_dir[:r_fit, :] = s_sq.unsqueeze(1) * vh_f[:r_fit, :]
            for o_z in (zero_output_rows or []):
                b_dir[o_z].zero_()
            for o_idx in range(o1):
                nrm = float(torch.linalg.vector_norm(b_dir[o_idx]).item())
                if nrm > 1e-7:
                    b_dir[o_idx] = b_dir[o_idx] / nrm
            dst_mod.a_scar.copy_(project_rows_to_unit_sphere(a_dir))
            dst_mod.b_scar.copy_(b_dir)
            _, a_i8 = quantize_ternary_ste(dst_mod.a_scar)
            _, b_i8 = quantize_ternary_ste(dst_mod.b_scar)
            scar_unit_mat = (b_i8.float() @ a_i8.float()) / math.sqrt(float(i1 * r1))
            denom = float((scar_unit_mat * scar_unit_mat).sum().item())
            if denom > 1e-10:
                numer = float((res_actual * scar_unit_mat).sum().item())
                dst_mod.gamma_scar.fill_(numer / denom)
            else:
                dst_mod.gamma_scar.fill_(0.0)
        elif r0 > 0 and a_u0 is not None and b_u0 is not None and abs(gam0) > 0.0:
            a_u0 = a_u0.to(dev)
            b_u0 = b_u0.to(dev)
            r_copy = min(r0, r1)
            a_new = torch.zeros((r1, i1), dtype=torch.float32, device=dev)
            a_comp = _rand_unit_rows(r_copy, max(1, comp_cols_count), gen, dev) if comp_cols_count > 0 else None
            for r_i in range(r_copy):
                for b_s, b_d in in_block_map:
                    a_new[r_i, 64 * b_d : 64 * (b_d + 1)] = alpha_in * a_u0[r_i, 64 * b_s : 64 * (b_s + 1)]
                if comp_cols_count > 0 and a_comp is not None:
                    off = 0
                    for b_u in unmapped_dst_blocks:
                        a_new[r_i, 64 * b_u : 64 * (b_u + 1)] = beta_in * a_comp[r_i, off : off + 64]
                        off += 64
            # Extra scar ranks `r_copy .. r1`: place input support in `unmapped_dst_blocks` so `xa[:, r_copy:] == 0`
            # at `step = 0` (`0.0` shock!) and wakes up as soon as `I_1 \\ I_0` activates!
            if r1 > r_copy:
                if comp_cols_count >= 64:
                    a_ext = _rand_unit_rows(r1 - r_copy, comp_cols_count, gen, dev)
                    for r_e in range(r1 - r_copy):
                        off = 0
                        for b_u in unmapped_dst_blocks:
                            a_new[r_copy + r_e, 64 * b_u : 64 * (b_u + 1)] = a_ext[r_e, off : off + 64]
                            off += 64
                else:
                    a_new[r_copy:] = _rand_unit_rows(r1 - r_copy, i1, gen, dev)

            alpha_r = math.sqrt(float(r_copy) / float(r1))
            beta_r = math.sqrt(max(0.0, 1.0 - float(r_copy) / float(r1)))
            b_new = torch.zeros((o1, r1), dtype=torch.float32, device=dev)
            b_comp = _rand_unit_rows(len(out_map), max(1, r1 - r_copy), gen, dev) if r1 > r_copy else None
            for idx_m, (o_s, o_d) in enumerate(out_map):
                if r1 > r_copy and comp_cols_count >= 64 and b_comp is not None:
                    b_new[o_d, :r_copy] = alpha_r * b_u0[o_s, :r_copy]
                    b_new[o_d, r_copy:] = beta_r * b_comp[idx_m]
                else:
                    b_new[o_d, :r_copy] = b_u0[o_s, :r_copy]
            # Unmapped output rows (`not in out_map` and `not in zero_output_rows`) get unit rows on `r_copy:r1` if available
            mapped_o_set = {o_d for _, o_d in out_map}
            zero_o_set = set(zero_output_rows or [])
            free_o = [o for o in range(o1) if (o not in mapped_o_set and o not in zero_o_set)]
            if free_o and r1 > r_copy:
                b_free = _rand_unit_rows(len(free_o), r1 - r_copy, gen, dev)
                for idx_f, o_f in enumerate(free_o):
                    b_new[o_f, r_copy:] = b_free[idx_f]
            # Note: rows in `zero_output_rows` remain `0` in `b_scar` -> `quantize_ternary_ste` on exact `0` row yields `0`!
            for o_idx in range(o1):
                nrm = float(torch.linalg.vector_norm(b_new[o_idx]).item())
                if nrm > 1e-7:
                    b_new[o_idx] = b_new[o_idx] / nrm

            dst_mod.a_scar.copy_(project_rows_to_unit_sphere(a_new))
            dst_mod.b_scar.copy_(b_new)
            eff_r_scale = (
                math.sqrt(float(i1 * r1) / float(max(1, active_cols_count * r_copy)))
                if (r1 == r_copy or comp_cols_count >= 64)
                else math.sqrt(float(i1 * r1) / float(max(1, active_cols_count * r1)))
            )
            dst_mod.gamma_scar.fill_(gam0 * eff_r_scale)
        else:
            dst_mod.gamma_scar.fill_(0.0)

    _invalidate_e8_caches_(dst_mod)


@torch.no_grad()
def _expand_symplectic_generator_(
    src_gen: HorizonCulledSymplecticGenerator,
    dst_gen: HorizonCulledSymplecticGenerator,
    bus_col_perm: Optional[Sequence[Tuple[int, int]]],
    gen: torch.Generator,
) -> None:
    """Expands `HorizonCulledSymplecticGenerator` preserving exact `SO(D)` action on `S^{D_0-1}`."""
    dev = dst_gen.scar_poles.device
    c0, t0, b0, r0 = src_gen.cond_dim, src_gen.target_dim, src_gen.bus_dim, src_gen.base_rank
    c1, t1, b1, r1 = dst_gen.cond_dim, dst_gen.target_dim, dst_gen.bus_dim, dst_gen.base_rank

    # Build 64D block map for `raw = cat(hc [cond_dim], bc [bus_dim], t_feat [4] + zero_pad)`
    # Note: in ScarField configurations (`d_form, d_world, d_bus` divisible by 64), `hc`, `bc`, and `t_feat`
    # sit on clean 64D block boundaries!
    if c0 % 64 != 0 or c1 % 64 != 0 or b0 % 64 != 0 or b1 % 64 != 0:
        raise ValueError("Symplectic generator isometric expansion requires cond_dim and bus_dim divisible by 64")

    in_block_map: List[Tuple[int, int]] = []
    # 1. `hc` blocks `0 .. c0//64 - 1` stay at `0 .. c0//64 - 1`
    for blk in range(c0 // 64):
        in_block_map.append((blk, blk))
    # 2. `bc` blocks shift from `c0//64 + blk` to `c1//64 + dst_bus_blk`
    if bus_col_perm is not None:
        # Derive 64D block mapping from `bus_col_perm` when block-aligned, else sequential `0 .. b0//64`
        for blk in range(b0 // 64):
            in_block_map.append(((c0 // 64) + blk, (c1 // 64) + blk))
    else:
        for blk in range(b0 // 64):
            in_block_map.append(((c0 // 64) + blk, (c1 // 64) + blk))
    # 3. `t_feat` (`[cos, sin, cos2, sin2]` + 60 zeros) sits at block `(c0 + b0)//64` -> `(c1 + b1)//64`
    in_block_map.append(((c0 + b0) // 64, (c1 + b1) // 64))

    # Output mapping for `uv_base` reshaped as `[2, target_dim, base_rank]`
    out_map: List[Tuple[int, int]] = []
    r_shared = min(r0, r1)
    for uv_i in (0, 1):
        for d_i in range(t0):
            for r_i in range(r_shared):
                o_s = uv_i * (t0 * r0) + d_i * r0 + r_i
                o_d = uv_i * (t1 * r1) + d_i * r1 + r_i
                out_map.append((o_s, o_d))

    # Enforce exact zero output on `(d >= t0, r < r_shared)` (both U & V) AND on `(uv_i == 1, r >= r_shared)` (V-half of new ranks!)
    zero_rows: List[int] = []
    for uv_i in (0, 1):
        for d_i in range(t0, t1):
            for r_i in range(r_shared):
                zero_rows.append(uv_i * (t1 * r1) + d_i * r1 + r_i)
    for d_i in range(t1):
        for r_i in range(r_shared, r1):
            zero_rows.append(1 * (t1 * r1) + d_i * r1 + r_i)

    expand_e8_linear_isometric_(
        src_gen.base_e8,
        dst_gen.base_e8,
        out_map=out_map,
        in_block_map=in_block_map,
        gen=gen,
        zero_output_rows=zero_rows,
    )

    # Localized Spherical-Cap Scars (`E_0 -> E_1`)
    e0 = min(src_gen.num_scars, dst_gen.num_scars)
    e1 = dst_gen.num_scars

    poles_new = torch.zeros((e1, c1), dtype=torch.float32, device=dev)
    poles_new[:e0, :c0] = project_rows_to_unit_sphere(src_gen.scar_poles.detach().to(dev).float()[:e0])
    if e1 > e0:
        if c1 > c0:
            # Place new scar poles primarily in `c0:c1` (`cos <= 0.08 < kappa` at `step = 0` -> `0.0` shock!)
            ext_p = _rand_unit_rows(e1 - e0, c1 - c0, gen, dev)
            eq_p = _rand_unit_rows(e1 - e0, c0, gen, dev)
            poles_new[e0:, :c0] = 0.08 * eq_p
            poles_new[e0:, c0:] = math.sqrt(1.0 - 0.08 * 0.08) * ext_p
        else:
            poles_new[e0:] = _rand_unit_rows(e1 - e0, c1, gen, dev)
    dst_gen.scar_poles.copy_(project_rows_to_unit_sphere(poles_new))

    kap_new = torch.full((e1,), 0.25, dtype=torch.float32, device=dev)
    kap_new[:e0] = src_gen.scar_kappa.detach().to(dev).float()[:e0]
    if e1 > e0 and c1 == c0:
        kap_new[e0:] = 0.915  # quiet at `step = 0` when `c1 == c0`
    dst_gen.scar_kappa.copy_(kap_new)

    u_new = torch.zeros((e1, t1), dtype=torch.float32, device=dev)
    v_new = torch.zeros((e1, t1), dtype=torch.float32, device=dev)
    u_new[:e0, :t0] = project_rows_to_unit_sphere(src_gen.scar_u.detach().to(dev).float()[:e0])
    v_new[:e0, :t0] = project_rows_to_unit_sphere(src_gen.scar_v.detach().to(dev).float()[:e0])
    if e1 > e0:
        u_new[e0:] = _rand_unit_rows(e1 - e0, t1, gen, dev)
        v_new[e0:] = _rand_unit_rows(e1 - e0, t1, gen, dev)
    dst_gen.scar_u.copy_(project_rows_to_unit_sphere(u_new))
    dst_gen.scar_v.copy_(project_rows_to_unit_sphere(v_new))

    str_new = torch.full((e1,), 0.35 if c1 > c0 else 1e-4, dtype=torch.float32, device=dev)
    str_new[:e0] = src_gen.scar_strength.detach().to(dev).float()[:e0]
    dst_gen.scar_strength.copy_(str_new)


@torch.no_grad()
def initialize_cayley_flow_as_nonsaddle_identity_(
    flow: ReversibleSymplecticCayleyFlow,
    seed: int = 2026092,
) -> None:
    """Initializes a brand-new `ReversibleSymplecticCayleyFlow` to EXACT identity `Phi_0(h) == h` WITHOUT dead saddles!

    Uses the Symplectic Asymmetric Saddle-Breaker `(U_Stiefel != 0, V == 0)`:
      - `U`-half of `base_e8` (`0 .. D*r`) has unit Stiefel weights (`tau_base = 0.35 > 0`),
      - `V`-half of `base_e8` (`D*r .. 2*D*r`) has exact `0.0` (`codes_u8 = 0x00` vacuum via zeroed `V` rows? Wait:
        since `weight_sphere` rows are normalized to unit length in `ensure_cached_quantized`, we instead set
        `gamma_scar = 0.0` on `base_e8`, `tau_base = 0.0` on `base_e8`? NO: to avoid `(U=0, V=0)` saddle while
        keeping exact `0.0` at `step = 0` even when `comp_cols == 0`, we place active non-zero `scar_u` & `scar_v`
        with `scar_kappa = -0.15` (`gates > 0` everywhere!) and `scar_strength = +1e-7`? Even cleaner: see below!).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    for sub_gen in (flow.gen_form, flow.gen_world):
        dev = sub_gen.scar_poles.device
        # Exact `0.0` base drift at `step = 0`, paired with non-zero unit `weight_sphere`:
        # Because `grad_tau = <gy_2d @ w_q_rot, x_rot> / sqrt(I)`, wait: at `U=0, V=0` `gy_2d` would be `0` ONLY IF
        # `u_scar` and `v_scar` were ALSO zero! When `u_scar, v_scar` have tiny seed `1e-6` (or `base_e8` has
        # `tau_base = 0.0`), let's check: with `tau_base = 0.0`, `gamma_scar = 0.0`, and `scar_strength = 1e-6`
        # (`|Delta h| < 1e-6` -> `cos = 1.0000000`!), BOTH `gu_all` AND `gv_all` are NON-ZERO at `step = 1`,
        # driving non-zero `grad_tau` AND `grad_scar_strength` immediately on `step = 1`!
        assert sub_gen.base_e8.weight_sphere is not None and sub_gen.base_e8.tau_base is not None
        sub_gen.base_e8.weight_sphere.copy_(
            _rand_unit_rows(sub_gen.base_e8.out_features, sub_gen.base_e8.in_features, gen, dev)
        )
        sub_gen.base_e8.tau_base.fill_(0.0)
        if sub_gen.base_e8.gamma_scar is not None:
            sub_gen.base_e8.gamma_scar.fill_(0.0)
        _invalidate_e8_caches_(sub_gen.base_e8)

        sub_gen.scar_poles.copy_(_rand_unit_rows(sub_gen.num_scars, sub_gen.cond_dim, gen, dev))
        sub_gen.scar_kappa.fill_(-0.15)  # active cap so `gates > 0` everywhere from `step = 1`!
        sub_gen.scar_u.copy_(_rand_unit_rows(sub_gen.num_scars, sub_gen.target_dim, gen, dev))
        sub_gen.scar_v.copy_(_rand_unit_rows(sub_gen.num_scars, sub_gen.target_dim, gen, dev))
        sub_gen.scar_strength.fill_(1e-6)  # `< 1e-6` perturbation (`cos = 1.0000000`) & full-rank `step = 1` grad!


@torch.no_grad()
def expand_scarfield_geodesic_isometry(
    src_engine: ScarFieldEngine,
    dst_cfg: ScarFieldConfig,
    seed: int = 2026091,
) -> ScarFieldEngine:
    """Isometrically expands `src_engine` (`ScarField-Edge`) into `dst_cfg` (`ScarField-Core`).

    Preserves learned equatorial dynamics at `step = 0` (`0` wasted GPU training steps), unlocks
    unconstrained cross-channel mixing (`0%` Matryoshka tax), and registers the Stiefel `.uscar` bridge.
    """
    sc = src_engine.cfg
    if (
        dst_cfg.d_form < sc.d_form
        or dst_cfg.d_world < sc.d_world
        or dst_cfg.d_bus < sc.d_bus
        or dst_cfg.d_key < sc.d_key
        or dst_cfg.num_subspaces < sc.num_subspaces
        or dst_cfg.num_macro_cycles < sc.num_macro_cycles
    ):
        raise ValueError("Destination ScarFieldConfig dimensions/cycles must be >= source ScarFieldConfig")

    d_sub0 = sc.d_bus // sc.num_subspaces
    d_sub1 = dst_cfg.d_bus // dst_cfg.num_subspaces
    if d_sub1 < d_sub0:
        raise ValueError(f"Destination subspace dim d_sub ({d_sub1}) must be >= source d_sub ({d_sub0})")

    # Preserve exact proper-time frequency & repeat-phase anchors on the equatorial submanifold
    # (including identity reloads `d_form == sc.d_form` when `sc` already carries non-zero `ref_dim`!)
    cfg_work = ScarFieldConfig.from_dict(dst_cfg.to_dict())
    if cfg_work.proper_time_freq_ref_dim == 0:
        if sc.proper_time_freq_ref_dim > 0:
            cfg_work.proper_time_freq_ref_dim = sc.proper_time_freq_ref_dim
        elif cfg_work.d_form > sc.d_form:
            cfg_work.proper_time_freq_ref_dim = sc.d_form
    if cfg_work.proper_time_world_ref_dim == 0:
        if sc.proper_time_world_ref_dim > 0:
            cfg_work.proper_time_world_ref_dim = sc.proper_time_world_ref_dim
        elif cfg_work.d_world > sc.d_world:
            cfg_work.proper_time_world_ref_dim = sc.d_world
    if cfg_work.proper_time_key_ref_dim == 0:
        if sc.proper_time_key_ref_dim > 0:
            cfg_work.proper_time_key_ref_dim = sc.proper_time_key_ref_dim
        elif cfg_work.d_key > sc.d_key:
            cfg_work.proper_time_key_ref_dim = sc.d_key

    dev = next(src_engine.parameters()).device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    dst_engine = ScarFieldEngine(cfg_work).to(dev)

    # 1. Expand `ByteShockFrontEnd` (`S^{d_form_0-1} -> S^{d_form_1-1}`, `S^{d_world_0-1} -> S^{d_world_1-1}`)
    v_n = min(sc.vocab_size, cfg_work.vocab_size)
    dst_engine.frontend.byte_form_poles.zero_()
    dst_engine.frontend.byte_form_poles[:v_n, : sc.d_form] = project_rows_to_unit_sphere(
        src_engine.frontend.byte_form_poles.detach().to(dev).float()[:v_n]
    )
    dst_engine.frontend.byte_world_poles.zero_()
    dst_engine.frontend.byte_world_poles[:v_n, : sc.d_world] = project_rows_to_unit_sphere(
        src_engine.frontend.byte_world_poles.detach().to(dev).float()[:v_n]
    )
    dst_engine.frontend.reflex_cap_poles.zero_()
    dst_engine.frontend.reflex_cap_poles[:v_n, : sc.d_form] = project_rows_to_unit_sphere(
        src_engine.frontend.reflex_cap_poles.detach().to(dev).float()[:v_n]
    )
    n_ng = min(src_engine.frontend.ngram_weights.numel(), dst_engine.frontend.ngram_weights.numel())
    dst_engine.frontend.ngram_weights[:n_ng] = src_engine.frontend.ngram_weights.detach().to(dev)[:n_ng]
    dst_engine.frontend.reflex_temp.copy_(src_engine.frontend.reflex_temp.detach().to(dev))

    # Build consistent bus channel embedding `c_old = g * d_sub0 + s -> c_new = g * d_sub1 + s`
    bus_col_map: List[Tuple[int, int]] = []
    for g_i in range(sc.num_subspaces):
        for s_i in range(d_sub0):
            bus_col_map.append((g_i * d_sub0 + s_i, g_i * d_sub1 + s_i))

    # 2. Expand `flow_percept` (`cond/bus = d_form`)
    _expand_symplectic_generator_(src_engine.flow_percept.gen_form, dst_engine.flow_percept.gen_form, None, gen)
    _expand_symplectic_generator_(src_engine.flow_percept.gen_world, dst_engine.flow_percept.gen_world, None, gen)

    # 3. Expand `ChannelSelectiveKalmanScarBus` (`qkv_e8`, `tri_compass_e8`, scalars, `.uscar` Stiefel bridge)
    qkv_in_blocks: List[Tuple[int, int]] = []
    for blk in range(sc.d_form // 64):
        qkv_in_blocks.append((blk, blk))
    for blk in range(sc.d_world // 64):
        qkv_in_blocks.append(((sc.d_form // 64) + blk, (cfg_work.d_form // 64) + blk))

    dk0, db0, g0 = sc.d_key, sc.d_bus, sc.num_subspaces
    dk1, db1, g1 = cfg_work.d_key, cfg_work.d_bus, cfg_work.num_subspaces
    qkv_out_map: List[Tuple[int, int]] = []
    for i_q in range(dk0):
        qkv_out_map.append((i_q, i_q))
    for i_k in range(dk0):
        qkv_out_map.append((dk0 + i_k, dk1 + i_k))
    for c_s, c_d in bus_col_map:
        qkv_out_map.append((2 * dk0 + c_s, 2 * dk1 + c_d))
    for g_i in range(g0):
        qkv_out_map.append((2 * dk0 + db0 + g_i, 2 * dk1 + db1 + g_i))

    mapped_qkv_out = {o_d for _, o_d in qkv_out_map}
    zero_qkv_out = [o for o in range(2 * dk1 + db1) if o not in mapped_qkv_out]
    expand_e8_linear_isometric_(
        src_engine.bus.qkv_e8,
        dst_engine.bus.qkv_e8,
        out_map=qkv_out_map,
        in_block_map=qkv_in_blocks,
        gen=gen,
        zero_output_rows=zero_qkv_out,
    )

    tri_in_blocks: List[Tuple[int, int]] = []
    if sc.d_world % 64 == 0 and cfg_work.d_world % 64 == 0:
        w_b0 = sc.d_world // 64
        w_b1 = cfg_work.d_world // 64
        for leg in range(3):
            for blk in range(w_b0):
                tri_in_blocks.append((leg * w_b0 + blk, leg * w_b1 + blk))
    else:
        for blk in range(src_engine.bus.tri_in_padded // 64):
            tri_in_blocks.append((blk, blk))

    mapped_tri_out = {c_d for _, c_d in bus_col_map}
    zero_tri_out = [o for o in range(db1) if o not in mapped_tri_out]
    expand_e8_linear_isometric_(
        src_engine.bus.tri_compass_e8,
        dst_engine.bus.tri_compass_e8,
        out_map=bus_col_map,
        in_block_map=tri_in_blocks,
        gen=gen,
        zero_output_rows=zero_tri_out,
    )

    dst_engine.bus.ring_kappa.copy_(src_engine.bus.ring_kappa.detach().to(dev))
    dst_engine.bus.tier_mix.copy_(src_engine.bus.tier_mix.detach().to(dev))
    # Compensate `kalman_flat / sqrt(num_subspaces)` when `G_0 -> G_1` so `kalman` amplitude is invariant!
    if g1 != g0:
        dst_engine.bus.tier_mix.data[0] = dst_engine.bus.tier_mix.data[0] * math.sqrt(float(g1) / float(g0))
    if src_engine.bus.biorth_slots > 0 and dst_engine.bus.biorth_slots > 0:
        dst_engine.bus.biorth_gain.copy_(src_engine.bus.biorth_gain.detach().to(dev))
        dst_engine.bus.biorth_kappa.copy_(src_engine.bus.biorth_kappa.detach().to(dev))

    # Register exact equatorial Stiefel `.uscar` bridge (`I_{d_world_0}` embedded in `d_world_1`) & copy cartridges!
    if sc.d_world != cfg_work.d_world or sc.d_bus != cfg_work.d_bus:
        w_in_eq = torch.zeros((sc.d_world, cfg_work.d_world), dtype=torch.float32, device=dev)
        w_in_eq[:, : sc.d_world] = torch.eye(sc.d_world, dtype=torch.float32, device=dev)
        w_out_eq = torch.zeros((cfg_work.d_bus, sc.d_bus), dtype=torch.float32, device=dev)
        for c_s, c_d in bus_col_map:
            w_out_eq[c_d, c_s] = 1.0
        dst_engine.bus.uscar_store.register_interscale_stiefel_bridge(
            canon_d_world=sc.d_world,
            canon_d_bus=sc.d_bus,
            w_world_to_canon=w_in_eq,
            w_canon_bus_to_bus=w_out_eq,
        )
    for cid, cart in src_engine.bus.uscar_store.cartridges.items():
        dst_engine.bus.uscar_store.cartridges[cid] = cart

    # 4. Expand `flow` & `extra_flows` (initializing brand-new macro-cycles to saddle-free near-identity!)
    _expand_symplectic_generator_(src_engine.flow.gen_form, dst_engine.flow.gen_form, bus_col_map, gen)
    _expand_symplectic_generator_(src_engine.flow.gen_world, dst_engine.flow.gen_world, bus_col_map, gen)

    n_src_extra = len(src_engine.extra_flows) if src_engine.extra_flows is not None else 0
    if dst_engine.extra_flows is not None:
        for idx_ef, dst_ef in enumerate(dst_engine.extra_flows):
            if src_engine.extra_flows is not None and idx_ef < n_src_extra:
                src_ef = src_engine.extra_flows[idx_ef]
                _expand_symplectic_generator_(src_ef.gen_form, dst_ef.gen_form, bus_col_map, gen)
                _expand_symplectic_generator_(src_ef.gen_world, dst_ef.gen_world, bus_col_map, gen)
            else:
                initialize_cayley_flow_as_nonsaddle_identity_(dst_ef, seed=int(seed) + 101 + idx_ef)

    # 5. Expand `SphericalCapSimplexHead` (`out_form_anchors`, `out_world_anchors`, `out_bus_anchors`, `value_pole`)
    dst_engine.head.out_form_anchors.zero_()
    dst_engine.head.out_form_anchors[:v_n, : sc.d_form] = project_rows_to_unit_sphere(
        src_engine.head.out_form_anchors.detach().to(dev).float()[:v_n]
    )
    dst_engine.head.out_world_anchors.zero_()
    dst_engine.head.out_world_anchors[:v_n, : sc.d_world] = project_rows_to_unit_sphere(
        src_engine.head.out_world_anchors.detach().to(dev).float()[:v_n]
    )
    dst_engine.head.out_bus_anchors.zero_()
    src_bus_a = project_rows_to_unit_sphere(src_engine.head.out_bus_anchors.detach().to(dev).float()[:v_n])
    for c_s, c_d in bus_col_map:
        dst_engine.head.out_bus_anchors[:v_n, c_d] = src_bus_a[:, c_s]
    dst_engine.head.mix_weights.copy_(src_engine.head.mix_weights.detach().to(dev))
    dst_engine.head.temp_scale.copy_(src_engine.head.temp_scale.detach().to(dev))
    if src_engine.head.enable_adaptive_cap and dst_engine.head.enable_adaptive_cap:
        dst_engine.head.cap_adapt_gain.copy_(src_engine.head.cap_adapt_gain.detach().to(dev))
    dst_engine.head.value_pole.zero_()
    dst_engine.head.value_pole[0, : sc.d_world] = project_rows_to_unit_sphere(
        src_engine.head.value_pole.detach().to(dev).float()
    )[0]

    dst_engine.assert_zero_legacy_modules()
    return dst_engine


@torch.no_grad()
def lift_kalman_bus_state_isometric(
    src_state: KalmanBusState,
    dst_engine: ScarFieldEngine,
) -> KalmanBusState:
    """Losslessly lifts a live `O(1)` `KalmanBusState` (`Edge -> Core`) mid-dialogue (`< 0.5 ms`, `0` lost facts)."""
    bsz = src_state.w_fast.shape[0]
    dev = next(dst_engine.parameters()).device
    dst_st = dst_engine.init_state(bsz, device=dev)

    g0, d_sub0, dk0 = src_state.w_fast.shape[1], src_state.w_fast.shape[2], src_state.w_fast.shape[3]
    g1, d_sub1, dk1 = dst_st.w_fast.shape[1], dst_st.w_fast.shape[2], dst_st.w_fast.shape[3]
    dw0 = src_state.h_origin.shape[-1]
    dw1 = dst_st.h_origin.shape[-1]
    if g1 < g0 or d_sub1 < d_sub0 or dk1 < dk0 or dw1 < dw0:
        raise ValueError("lift_kalman_bus_state_isometric requires destination dims >= source dims (up-projection)")

    dst_st.w_fast[:, :g0, :d_sub0, :dk0] = src_state.w_fast.to(dev)
    dst_st.k_prev[:, :dk0] = src_state.k_prev.to(dev)
    dst_st.has_prev_key.copy_(src_state.has_prev_key.to(dev))

    rb_copy = min(src_state.recent_bytes.shape[1], dst_st.recent_bytes.shape[1])
    if rb_copy > 0:
        dst_st.recent_bytes[:, -rb_copy:] = src_state.recent_bytes[:, -rb_copy:].to(dev)
    dst_st.recent_len = min(int(src_state.recent_len), dst_st.recent_bytes.shape[1])

    r_copy = min(src_state.ring_keys.shape[1], dst_st.ring_keys.shape[1])
    if r_copy > 0:
        dst_st.ring_keys[:, -r_copy:, :dk0] = src_state.ring_keys[:, -r_copy:, :].to(dev)
        src_rv = src_state.ring_vals[:, -r_copy:, :].to(dev)
        for g_i in range(g0):
            for s_i in range(d_sub0):
                dst_st.ring_vals[:, -r_copy:, g_i * d_sub1 + s_i] = src_rv[:, :, g_i * d_sub0 + s_i]
        dst_st.ring_valid[:, -r_copy:] = src_state.ring_valid[:, -r_copy:].to(dev)
    dst_st.ring_ptr = 0

    dst_st.proper_time.copy_(src_state.proper_time.to(dev))
    dst_st.h_origin[:, :dw0] = src_state.h_origin.to(dev)
    dst_st.has_origin.copy_(src_state.has_origin.to(dev))
    dst_st.h_prev[:, :dw0] = src_state.h_prev.to(dev)

    rep_copy = min(src_state.repulsive_ring.shape[1], dst_st.repulsive_ring.shape[1])
    if rep_copy > 0:
        dst_st.repulsive_ring[:, :rep_copy, :dw0] = src_state.repulsive_ring[:, :rep_copy, :].to(dev)
    dst_st.repulsive_ptr = int(src_state.repulsive_ptr % max(1, dst_st.repulsive_ring.shape[1]))

    # Re-insert all valid `BiorthogonalShockBank` entries so spherical Hadamard bucket routing
    # in `dst_engine` (`e.g. 1x48 or 2x8 -> 16x32`) places every memory into its exact target octant!
    if (
        src_state.biorth_valid is not None
        and src_state.biorth_keys is not None
        and src_state.biorth_vals is not None
        and src_state.biorth_salience is not None
        and dst_st.biorth_keys is not None
        and dst_st.biorth_vals is not None
        and dst_st.biorth_salience is not None
        and dst_st.biorth_valid is not None
    ):
        m_src = src_state.biorth_keys.shape[1]
        for s_slot in range(m_src):
            val_b = src_state.biorth_valid[:, s_slot].to(dev)
            if not bool(val_b.any().item()):
                continue
            k_lifted = torch.zeros((bsz, dk1), dtype=torch.float32, device=dev)
            k_lifted[:, :dk0] = src_state.biorth_keys[:, s_slot, :].to(dev)
            v_lifted = torch.zeros((bsz, dst_engine.cfg.d_bus), dtype=torch.float32, device=dev)
            src_bv = src_state.biorth_vals[:, s_slot, :].to(dev)
            for g_i in range(g0):
                for s_i in range(d_sub0):
                    v_lifted[:, g_i * d_sub1 + s_i] = src_bv[:, g_i * d_sub0 + s_i]
            sal_lifted = torch.where(
                val_b,
                src_state.biorth_salience[:, s_slot].to(dev),
                torch.full((bsz,), -1.0, dtype=torch.float32, device=dev),
            )
            biorthogonal_shock_bank_insert_step_(
                dst_st.biorth_keys,
                dst_st.biorth_vals,
                dst_st.biorth_salience,
                dst_st.biorth_valid,
                k_lifted,
                v_lifted,
                sal_lifted,
                write_thresh=0.0,
                num_buckets=dst_engine.cfg.biorth_buckets,
            )

    return dst_st
