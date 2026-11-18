"""ScarField v8.0 — Geodesic Macro-Scout Fleet & Shared-Crystal E8 Swarm.

Implements the three canonical tiers of the `ScarField v8.0` Macro-Scout Geodesic Fleet
(`docs/SCARFIELD_V7_5_FUSED_AND_V8_MACRO_SCOUT_FLEET.md`, Zero-Facade / Zero-Legacy):

  1. **Level 1 — Concurrent Multi-Stream GPU SM Saturation**:
       - Exploits `ScarField`'s ultra-compact `~1.35 GiB` training VRAM (`9.53 MiB` `.sfield.scar`,
         reversible `O(1)` Cayley activations, `O(1)` `KalmanBusState`, single-buffer `RiemannianMuon`)
         to dispatch `M = 4..8` simultaneous Macro-Scouts across independent `torch.cuda.Stream`s
         in zero-host-sync lockstep waves (`Phase A: S1 Scout -> Phase B: Fwd -> Phase C: Bwd+Muon`).

  2. **Level 2 — Specialist Macro-Scouts + Biorthogonal Tangent-Space Geodesic Merge**:
       - Eliminates sequential multi-task gradient interference (`Prose`, `Code-AST`, `Long-Horizon
         Biorthogonal Passkey`, `Stateful O(1) Chat & 0x00 Safety`) by evolving `M` domain/horizon
         specialists for `H` local steps, mapping their chords to `T_{w_0} S^{I-1}` via exact
         Riemannian `Log_{w_0}(w_m)`, removing negative-cosine conflicts (`<Delta_m, Delta_n> < 0`),
         solving the `[M x M]` Biorthogonal Dual Gram system `((1 - lam) U U^T + lam I) c = alpha`
         so unique specialist directions suffer zero `1/M` dilution while redundant directions
         deduplicate automatically, retracting via exact `Exp_{w_0}`, and running `1x` fleet-wide
         `heal_all_e8_scars` per `M * H` steps.

  3. **Level 3 — Shared-Crystal Gosset `E_8` MoS Swarm (`SharedCrystalMacroSwarm`)**:
       - Shares `1` read-only `1.000 bpw` Gosset `E_8` substrate crystal by pointer across `S = 8..16`
         ultralight workers (`0 B` of `[O, I]` `weight_sphere` per worker, `~31x` fewer mutable
         parameters, skipping `[O, I]` weight-gradient GEMMs) that evolve ONLY ternary atomic scars
         `(a_scar, b_scar, gamma_scar)` + cap poles, then distills winning scar innovations directly
         back into the master `1.000 bpw` `E_8` crystal (`Delta = +0 B` `.sfield.scar` footprint!).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import scar_field.e8_lattice as _e8_mod
from scar_field.e8_lattice import E8SphericalLinear, project_rows_to_unit_sphere, quantize_ternary_ste
from scar_field.engine import ScarFieldEngine
from scar_field.riemannian_muon import RiemannianMuon, heal_all_e8_scars


@dataclass(frozen=True)
class MacroScoutSpec:
    """Configuration of a specialist Macro-Scout inside `MacroScoutGeodesicFleet`."""

    name: str
    weight: float = 1.0
    lr_scale: float = 1.0
    owns_biorth_organ: bool = False


@dataclass(frozen=True)
class MacroBatchSpec:
    """Zero-sync batch payload returned by a Macro-Scout sampler for one local step."""

    byte_ids: torch.Tensor
    paired_translation_ids: Optional[torch.Tensor] = None
    out_of_cone_mask: Optional[torch.Tensor] = None
    target_loss_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class GeodesicMergeTelemetry:
    """Audit telemetry emitted at each `H`-step Biorthogonal Geodesic Fleet merge."""

    num_scouts: int
    local_steps_per_scout: int
    total_fleet_steps: int
    conflict_pairs_projected: int
    mean_offdiag_gram_cosine: float
    mean_tangent_chord_rad: float
    scout_mean_cap_losses: Dict[str, float]
    biorth_gain_after_merge: float


@dataclass(frozen=True)
class SharedCrystalSwarmTelemetry:
    """Audit telemetry emitted by `SharedCrystalMacroSwarm.distill_into_master_crystal`."""

    num_swarm_workers: int
    mutable_params_per_worker: int
    master_total_params: int
    param_reduction_ratio_x: float
    conflict_pairs_projected: int
    crystal_layers_distilled: int
    mean_scar_innovation_fro: float


def riemannian_log_chord_2d(w0: torch.Tensor, wm: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Exact row-wise Riemannian logarithm `Log_{u_0}(u_m) in T_{u_0} S^{I-1}` (`<delta, u_0> == 0`)."""
    u0 = project_rows_to_unit_sphere(w0.float(), eps=eps)
    um = project_rows_to_unit_sphere(wm.float(), eps=eps)
    cos_t = (um * u0).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    v_perp = um - cos_t * u0
    # Re-project once for numerical ULP orthogonality `<v_perp, u0> ~ 1e-8`
    v_perp = v_perp - (v_perp * u0).sum(dim=-1, keepdim=True) * u0
    sin_t = torch.linalg.vector_norm(v_perp, ord=2, dim=-1, keepdim=True)
    theta = torch.atan2(sin_t, cos_t)
    scale = torch.where(sin_t > 1e-4, theta / sin_t.clamp(min=1e-12), 1.0 + (theta * theta) / 6.0)
    return v_perp * scale


def riemannian_exp_retract_2d(w0: torch.Tensor, delta: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Exact row-wise Riemannian exponential map `Exp_{u_0}(delta)` onto `S^{I-1}` (`||w_new||_2 == 1`)."""
    u0 = project_rows_to_unit_sphere(w0.float(), eps=eps)
    # Enforce exact tangent orthogonality before geodesic rotation
    d_tan = delta.float() - (delta.float() * u0).sum(dim=-1, keepdim=True) * u0
    theta = torch.linalg.vector_norm(d_tan, ord=2, dim=-1, keepdim=True)
    sinc_t = torch.where(theta > 1e-4, torch.sin(theta) / theta.clamp(min=1e-12), 1.0 - (theta * theta) / 6.0)
    w_exp = torch.cos(theta) * u0 + sinc_t * d_tan
    return project_rows_to_unit_sphere(w_exp, eps=eps).to(w0.dtype)


@torch.no_grad()
def biorthogonal_conflict_free_tangent_merge(
    deltas: Sequence[torch.Tensor],
    w0_sphere_2d: Optional[torch.Tensor] = None,
    prior_weights: Optional[Sequence[float]] = None,
    ridge_lambda: float = 0.12,
    activity_eps: float = 1e-8,
) -> Tuple[torch.Tensor, int, float]:
    """Merges `M` specialist tangent displacements with zero destructive interference & zero `1/M` skill dilution.

    1. Filters inactive scouts (`||Delta_m||_F <= activity_eps`) so idle organs never dilute active specialists.
    2. Projects out negative pairwise inner products `<Delta_m~, Delta_n>_F < 0` on `T_{w_0} S^{I-1}`.
    3. Solves the SPD Biorthogonal Dual Gram system `((1 - lam) U U^T + lam I) c = alpha`:
         - Redundant colinear scouts (`G_ab = 1`) receive `c_a ≈ 1/K` (zero `K`-fold overshooting!).
         - Orthogonal unique specialists (`G_ab = delta_ab`) receive `c_a = 1.0` (zero `1/M` skill dilution!).
    """
    m_tot = len(deltas)
    if m_tot == 0:
        raise ValueError("deltas must be non-empty")
    ref = deltas[0]
    if m_tot == 1:
        return ref.float().clone(), 0, 0.0

    flat_orig = torch.stack([d.detach().float().reshape(-1) for d in deltas], dim=0)  # [M, N]
    norms = torch.linalg.vector_norm(flat_orig, ord=2, dim=-1)  # [M]
    max_nrm = float(norms.max().item())
    if max_nrm <= activity_eps:
        return torch.zeros_like(ref, dtype=torch.float32), 0, 0.0

    # Keep scouts with non-trivial activity (`>= 2%` of max active norm)
    act_mask = norms > max(activity_eps, 0.02 * max_nrm)
    act_indices = torch.nonzero(act_mask, as_tuple=False).squeeze(-1).tolist()
    if not act_indices:
        return torch.zeros_like(ref, dtype=torch.float32), 0, 0.0
    if len(act_indices) == 1:
        idx0 = act_indices[0]
        pw0 = float(prior_weights[idx0]) if prior_weights is not None else 1.0
        return deltas[idx0].detach().float() * pw0, 0, 0.0

    orig_act = flat_orig[act_indices]  # [M_a, N]
    surg = orig_act.clone()
    orig_norm_sq = (orig_act * orig_act).sum(dim=-1).clamp(min=1e-12)
    m_act = orig_act.shape[0]
    conflicts = 0

    # Symmetric cyclic conflict projection (`PCGrad` in tangent space)
    for i in range(m_act):
        vi = surg[i]
        for offset in range(1, m_act):
            j = (i + offset) % m_act
            dot_ij = float(torch.dot(vi, orig_act[j]).item())
            if dot_ij < 0.0:
                vi = vi - (dot_ij / orig_norm_sq[j]) * orig_act[j]
                conflicts += 1
        surg[i] = vi

    if w0_sphere_2d is not None:
        u0 = project_rows_to_unit_sphere(w0_sphere_2d.detach().float())
        surg_3d = surg.reshape(m_act, *w0_sphere_2d.shape)
        surg_3d = surg_3d - (surg_3d * u0.unsqueeze(0)).sum(dim=-1, keepdim=True) * u0.unsqueeze(0)
        surg = surg_3d.reshape(m_act, -1)

    surg_norms = torch.linalg.vector_norm(surg, ord=2, dim=-1, keepdim=True).clamp(min=1e-12)
    u_hat = surg / surg_norms  # [M_a, N]
    gram = torch.matmul(u_hat, u_hat.T)  # [M_a, M_a] SPD!
    lam = float(min(0.95, max(1e-3, ridge_lambda)))
    eye_a = torch.eye(m_act, dtype=torch.float32, device=ref.device)
    gram_reg = (1.0 - lam) * gram + lam * eye_a

    if prior_weights is not None:
        pw_t = torch.tensor([float(prior_weights[k]) for k in act_indices], dtype=torch.float32, device=ref.device)
        pw_mean = pw_t.mean().clamp(min=1e-6)
        rhs = pw_t / pw_mean
    else:
        rhs = torch.ones(m_act, dtype=torch.float32, device=ref.device)

    c_dual = torch.linalg.solve(gram_reg, rhs).clamp(min=0.0, max=1.5)  # [M_a]
    merged_flat = torch.matmul(c_dual.unsqueeze(0), surg).squeeze(0)

    off_diag = gram - torch.diag(torch.diag(gram))
    mean_off = float(off_diag.sum().item() / max(1, m_act * (m_act - 1)))
    return merged_flat.reshape_as(ref), conflicts, mean_off


@torch.no_grad()
def clone_scarfield_engine_fast(master: ScarFieldEngine, device: torch.device) -> ScarFieldEngine:
    """Clones `master` onto `device` in `< 5 ms` WITHOUT redundant `[O, I]` `randn` + `SVD` init!"""
    _e8_mod._SKIP_E8_INIT_ALLOC = True
    try:
        worker = ScarFieldEngine(copy.deepcopy(master.cfg))
    finally:
        _e8_mod._SKIP_E8_INIT_ALLOC = False

    worker = worker.to(device)
    master_mods = dict(master.named_modules())
    for name, mod in worker.named_modules():
        if isinstance(mod, E8SphericalLinear):
            m_src = master_mods[name]
            assert isinstance(m_src, E8SphericalLinear)
            mod.lut_e8 = mod.lut_e8.to(device)
            mod.lut_half_norm2 = mod.lut_half_norm2.to(device)
            mod.signs = mod.signs.to(device)
            if m_src.weight_sphere is not None and m_src.tau_base is not None:
                mod.weight_sphere = nn.Parameter(m_src.weight_sphere.detach().to(device).clone())
                mod.tau_base = nn.Parameter(m_src.tau_base.detach().to(device).clone())
            if (
                mod.scar_rank > 0
                and m_src.a_scar is not None
                and m_src.b_scar is not None
                and m_src.gamma_scar is not None
            ):
                mod.a_scar = nn.Parameter(m_src.a_scar.detach().to(device).clone())
                mod.b_scar = nn.Parameter(m_src.b_scar.detach().to(device).clone())
                mod.gamma_scar = nn.Parameter(m_src.gamma_scar.detach().to(device).clone())
            mod.ensure_cached_quantized()

    m_params = dict(master.named_parameters())
    for pname, p_w in worker.named_parameters():
        if pname in m_params and not any(
            pname.endswith(suf) for suf in (".weight_sphere", ".tau_base", ".a_scar", ".b_scar", ".gamma_scar")
        ):
            p_w.copy_(m_params[pname].detach().to(device))

    worker.assert_zero_legacy_modules()
    return worker


class MacroScoutGeodesicFleet:
    """Level 1 + Level 2: Concurrent Multi-Stream Specialist Macro-Scouts + Biorthogonal Geodesic Merge."""

    def __init__(
        self,
        master: ScarFieldEngine,
        scout_specs: Sequence[MacroScoutSpec],
        base_lr: float = 1.0e-3,
        scalar_lr: float = 1.4e-3,
        local_momentum: float = 0.88,
        outer_lr: float = 0.85,
        outer_momentum: float = 0.55,
        merge_ridge_lambda: float = 0.14,
        local_mom_retain: float = 0.30,
        biorth_gain_floor: float = 0.32,
        enable_cuda_streams: bool = True,
    ) -> None:
        if len(scout_specs) < 1:
            raise ValueError("scout_specs must contain at least 1 MacroScoutSpec")
        self.master = master
        self.device = master.frontend.byte_form_poles.device
        if self.device.type in ("cuda", "mps"):
            _e8_mod._get_hadamard64(self.device, torch.float32)
        self.specs: List[MacroScoutSpec] = list(scout_specs)
        self.num_scouts = len(self.specs)
        self.base_lr = float(base_lr)
        self.scalar_lr = float(scalar_lr)
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.merge_ridge_lambda = float(merge_ridge_lambda)
        self.local_mom_retain = float(local_mom_retain)
        self.biorth_gain_floor = float(biorth_gain_floor)
        self.use_cuda_streams = bool(enable_cuda_streams and self.device.type == "cuda" and self.num_scouts > 1)

        heal_all_e8_scars(self.master)
        self.scouts: List[ScarFieldEngine] = []
        self.optimizers: List[RiemannianMuon] = []
        self.streams: List[Optional[torch.cuda.Stream]] = []
        for sp in self.specs:
            w_eng = clone_scarfield_engine_fast(self.master, self.device)
            w_eng.set_async_stream_mode(self.use_cuda_streams)
            opt = RiemannianMuon(
                w_eng.parameters(),
                lr=self.base_lr * sp.lr_scale,
                scalar_lr=self.scalar_lr * sp.lr_scale,
                momentum=local_momentum,
                ns_steps=5,
            )
            self.scouts.append(w_eng)
            self.optimizers.append(opt)
            self.streams.append(torch.cuda.Stream(device=self.device) if self.use_cuda_streams else None)

        self._outer_vel: Dict[str, torch.Tensor] = {}

    def set_learning_rate(self, lr: float, scalar_lr: Optional[float] = None) -> None:
        self.base_lr = float(lr)
        if scalar_lr is not None:
            self.scalar_lr = float(scalar_lr)
        for sp, opt in zip(self.specs, self.optimizers):
            for pg in opt.param_groups:
                pg["lr"] = self.base_lr * sp.lr_scale
                pg["scalar_lr"] = self.scalar_lr * sp.lr_scale

    @torch.no_grad()
    def broadcast_master_to_scouts(self) -> None:
        """Copies healed master parameters (+ pre-quantized E8 caches!) to all `M` scouts on-device."""
        for m_mod in self.master.modules():
            if isinstance(m_mod, E8SphericalLinear):
                m_mod.ensure_cached_quantized()

        m_params = dict(self.master.named_parameters())
        m_mods = dict(self.master.named_modules())
        for w_eng, opt in zip(self.scouts, self.optimizers):
            for pname, p_w in w_eng.named_parameters():
                p_m = m_params[pname]
                p_w.copy_(p_m.detach())
                st = opt.state.get(p_w)
                if st is not None and "momentum_buffer" in st:
                    buf: torch.Tensor = st["momentum_buffer"]
                    buf.mul_(self.local_mom_retain)
                    if p_w.ndim == 2 and p_w.shape[1] >= 4:
                        wu = project_rows_to_unit_sphere(p_w.detach().float())
                        dot_bw = (buf.float() * wu).sum(dim=-1, keepdim=True)
                        buf.copy_((buf.float() - dot_bw * wu).to(buf.dtype))
            # Share freshly healed `_cached_w_q_rot` for step 0 without re-running `M` E8 quantizations!
            for mod_name, w_mod in w_eng.named_modules():
                if isinstance(w_mod, E8SphericalLinear):
                    src_mod = m_mods[mod_name]
                    assert isinstance(src_mod, E8SphericalLinear)
                    if src_mod._cached_w_q_rot is not None and w_mod.weight_sphere is not None:
                        w_mod._cached_w_q_rot = src_mod._cached_w_q_rot.clone()
                        w_mod._cached_w_q_rot_bf16 = (
                            src_mod._cached_w_q_rot_bf16.clone() if src_mod._cached_w_q_rot_bf16 is not None else None
                        )
                        w_mod._cached_q_key = (
                            w_mod.weight_sphere._version,
                            w_mod.weight_sphere.data_ptr(),
                            self.device.type,
                            self.device.index,
                        )

    def run_concurrent_round(
        self,
        batch_samplers: Sequence[Callable[[ScarFieldEngine, int], MacroBatchSpec]],
        local_steps: int = 20,
        grad_clip: float = 1.0,
        merge_after: bool = True,
    ) -> GeodesicMergeTelemetry:
        """Executes `local_steps` concurrent Multi-Stream steps across all `M` scouts, then merges."""
        if len(batch_samplers) != self.num_scouts:
            raise ValueError(f"Expected {self.num_scouts} batch_samplers, got {len(batch_samplers)}")
        h_steps = max(1, int(local_steps))

        if self.use_cuda_streams:
            def_stream = torch.cuda.current_stream(self.device)
            for s in self.streams:
                assert s is not None
                s.wait_stream(def_stream)

        loss_acc_tensors: List[torch.Tensor] = [
            torch.zeros((), dtype=torch.float32, device=self.device) for _ in range(self.num_scouts)
        ]

        for h in range(h_steps):
            # Phase A: Zero-sync batch generation / System 1 Scout mining across all M streams
            batches: List[MacroBatchSpec] = []
            for idx, (w_eng, opt, sampler, strm) in enumerate(
                zip(self.scouts, self.optimizers, batch_samplers, self.streams)
            ):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        opt.zero_grad(set_to_none=True)
                        batches.append(sampler(w_eng, h))
                else:
                    opt.zero_grad(set_to_none=True)
                    batches.append(sampler(w_eng, h))

            # Phase B: Lockstep asynchronous forward + Tsallis/Ghost-Jump/Vacuum loss across all M streams
            loss_dicts: List[Dict[str, torch.Tensor]] = []
            for idx, (w_eng, bspec, strm) in enumerate(zip(self.scouts, batches, self.streams)):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        ld = w_eng.compute_loss(
                            bspec.byte_ids,
                            paired_translation_ids=bspec.paired_translation_ids,
                            out_of_cone_mask=bspec.out_of_cone_mask,
                            target_loss_mask=bspec.target_loss_mask,
                        )
                        loss_acc_tensors[idx] = loss_acc_tensors[idx] + ld["loss_cap"]
                        loss_dicts.append(ld)
                else:
                    ld = w_eng.compute_loss(
                        bspec.byte_ids,
                        paired_translation_ids=bspec.paired_translation_ids,
                        out_of_cone_mask=bspec.out_of_cone_mask,
                        target_loss_mask=bspec.target_loss_mask,
                    )
                    loss_acc_tensors[idx] = loss_acc_tensors[idx] + ld["loss_cap"]
                    loss_dicts.append(ld)

            # Phase C: Lockstep zero-autograd symplectic backward + foreach grad clip + Triton RiemannianMuon!
            for w_eng, opt, ld, strm in zip(self.scouts, self.optimizers, loss_dicts, self.streams):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        ld["loss"].backward()
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                w_eng.parameters(), grad_clip, foreach=(self.device.type == "cuda")
                            )
                        opt.step()
                else:
                    ld["loss"].backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            w_eng.parameters(), grad_clip, foreach=(self.device.type == "cuda")
                        )
                    opt.step()

        if self.use_cuda_streams:
            def_stream = torch.cuda.current_stream(self.device)
            for s in self.streams:
                assert s is not None
                def_stream.wait_stream(s)

        mean_losses_cpu = (torch.stack(loss_acc_tensors) / float(h_steps)).detach().cpu().tolist()
        loss_by_scout = {
            sp.name: round(float(val), 4) for sp, val in zip(self.specs, mean_losses_cpu)
        }

        if merge_after:
            return self.merge_scouts_into_master(
                local_steps=h_steps, scout_mean_cap_losses=loss_by_scout
            )

        bg_val = (
            float(self.master.bus.biorth_gain.item())
            if hasattr(self.master.bus, "biorth_gain")
            else 0.0
        )
        return GeodesicMergeTelemetry(
            num_scouts=self.num_scouts,
            local_steps_per_scout=h_steps,
            total_fleet_steps=self.num_scouts * h_steps,
            conflict_pairs_projected=0,
            mean_offdiag_gram_cosine=0.0,
            mean_tangent_chord_rad=0.0,
            scout_mean_cap_losses=loss_by_scout,
            biorth_gain_after_merge=round(bg_val, 4),
        )

    @torch.no_grad()
    def merge_scouts_into_master(
        self,
        local_steps: int = 1,
        scout_mean_cap_losses: Optional[Dict[str, float]] = None,
    ) -> GeodesicMergeTelemetry:
        """Performs Biorthogonal Conflict-Free Geodesic Merge on `S^{I-1}` + `1x` fleet-wide `E_8` heal."""
        priors = [sp.weight for sp in self.specs]
        biorth_owners = [idx for idx, sp in enumerate(self.specs) if sp.owns_biorth_organ]

        m_params = dict(self.master.named_parameters())
        s_param_maps = [dict(w.named_parameters()) for w in self.scouts]

        total_conflicts = 0
        gram_cos_list: List[float] = []
        chord_rads: List[float] = []

        for pname, p0 in m_params.items():
            is_biorth_param = "biorth_gain" in pname or "biorth_kappa" in pname
            if is_biorth_param and biorth_owners:
                active_ids = biorth_owners
                sub_priors = [priors[i] for i in active_ids]
            else:
                active_ids = list(range(self.num_scouts))
                sub_priors = priors

            if p0.ndim == 2 and p0.shape[1] >= 4:
                deltas_2d = [riemannian_log_chord_2d(p0, s_param_maps[i][pname]) for i in active_ids]
                merged_d, conf_cnt, off_g = biorthogonal_conflict_free_tangent_merge(
                    deltas_2d,
                    w0_sphere_2d=p0,
                    prior_weights=sub_priors,
                    ridge_lambda=self.merge_ridge_lambda,
                )
                total_conflicts += conf_cnt
                gram_cos_list.append(off_g)
                chord_rads.append(float(torch.linalg.vector_norm(merged_d, ord=2, dim=-1).mean().item()))

                u0 = project_rows_to_unit_sphere(p0.float())
                vel = self._outer_vel.get(pname)
                if vel is None or vel.shape != p0.shape or vel.device != p0.device:
                    vel = merged_d.clone()
                else:
                    # Parallel transport outer velocity onto `T_{u_0} S^{I-1}`
                    v_tr = vel - (vel * u0).sum(dim=-1, keepdim=True) * u0
                    vel = self.outer_momentum * v_tr + merged_d
                self._outer_vel[pname] = vel

                step_tan = self.outer_lr * (merged_d + self.outer_momentum * vel)
                p0.copy_(riemannian_exp_retract_2d(p0, step_tan))
            else:
                deltas_1d = [(s_param_maps[i][pname].detach().float() - p0.detach().float()) for i in active_ids]
                if p0.numel() == 1:
                    # Activity-weighted scalar merge (prevents idle scouts from damping 1-element scalars!)
                    mags = torch.stack([d.abs().reshape(()) for d in deltas_1d])
                    max_m = float(mags.max().item())
                    if max_m > 1e-9:
                        w_s = torch.tensor(sub_priors, dtype=torch.float32, device=p0.device) * (mags + 0.05 * max_m)
                        w_s = w_s / w_s.sum().clamp(min=1e-12)
                        merged_1d = sum(w_s[k] * deltas_1d[k] for k in range(len(deltas_1d)))
                    else:
                        merged_1d = torch.zeros_like(p0, dtype=torch.float32)
                else:
                    merged_1d, conf_1d, _ = biorthogonal_conflict_free_tangent_merge(
                        deltas_1d,
                        w0_sphere_2d=None,
                        prior_weights=sub_priors,
                        ridge_lambda=self.merge_ridge_lambda,
                    )
                    total_conflicts += conf_1d

                vel1 = self._outer_vel.get(pname)
                if vel1 is None or vel1.shape != p0.shape or vel1.device != p0.device:
                    vel1 = merged_1d.clone()
                else:
                    vel1 = self.outer_momentum * vel1 + merged_1d
                self._outer_vel[pname] = vel1
                p0.add_((merged_1d + self.outer_momentum * vel1).to(p0.dtype), alpha=self.outer_lr)

                if "biorth_gain" in pname and self.biorth_gain_floor > 0.0:
                    p0.clamp_(min=self.biorth_gain_floor)

        # Single amortized `SVD/RRR` E8 lattice scar healing for the entire `M * H` fleet step round!
        heal_all_e8_scars(self.master)
        self.broadcast_master_to_scouts()

        bg_after = (
            float(self.master.bus.biorth_gain.item())
            if hasattr(self.master.bus, "biorth_gain")
            else 0.0
        )
        return GeodesicMergeTelemetry(
            num_scouts=self.num_scouts,
            local_steps_per_scout=int(local_steps),
            total_fleet_steps=self.num_scouts * int(local_steps),
            conflict_pairs_projected=total_conflicts,
            mean_offdiag_gram_cosine=round(float(sum(gram_cos_list) / max(1, len(gram_cos_list))), 5),
            mean_tangent_chord_rad=round(float(sum(chord_rads) / max(1, len(chord_rads))), 5),
            scout_mean_cap_losses=scout_mean_cap_losses or {},
            biorth_gain_after_merge=round(bg_after, 4),
        )


class SharedCrystalMacroSwarm:
    """Level 3: `1` Read-Only `1.000 bpw` Gosset `E_8` Crystal Shared Across `S` Ultralight Scar Workers."""

    def __init__(
        self,
        master: ScarFieldEngine,
        num_workers: int = 8,
        scar_lr: float = 1.6e-3,
        scalar_lr: float = 1.0e-3,
        distill_eta: float = 0.70,
        merge_ridge_lambda: float = 0.14,
        biorth_gain_floor: float = 0.32,
        enable_cuda_streams: bool = True,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.master = master
        self.device = master.frontend.byte_form_poles.device
        if self.device.type in ("cuda", "mps"):
            _e8_mod._get_hadamard64(self.device, torch.float32)
        self.num_workers = int(num_workers)
        self.scar_lr = float(scar_lr)
        self.scalar_lr = float(scalar_lr)
        self.distill_eta = float(distill_eta)
        self.merge_ridge_lambda = float(merge_ridge_lambda)
        self.biorth_gain_floor = float(biorth_gain_floor)
        self.use_cuda_streams = bool(enable_cuda_streams and self.device.type == "cuda" and self.num_workers > 1)

        heal_all_e8_scars(self.master)
        for m_mod in self.master.modules():
            if isinstance(m_mod, E8SphericalLinear):
                m_mod.ensure_cached_quantized()

        self.workers: List[ScarFieldEngine] = []
        self.optimizers: List[RiemannianMuon] = []
        self.streams: List[Optional[torch.cuda.Stream]] = []

        m_params = dict(self.master.named_parameters())
        for _ in range(self.num_workers):
            _e8_mod._SKIP_E8_INIT_ALLOC = True
            try:
                w = ScarFieldEngine(copy.deepcopy(self.master.cfg))
            finally:
                _e8_mod._SKIP_E8_INIT_ALLOC = False
            w = w.to(self.device)
            w.bind_shared_e8_crystal_from(self.master)
            with torch.no_grad():
                for pname, pw in w.named_parameters():
                    if pname in m_params and not any(pname.endswith(s) for s in (".a_scar", ".b_scar", ".gamma_scar")):
                        pw.copy_(m_params[pname].detach())
            w.set_async_stream_mode(self.use_cuda_streams)
            w.assert_zero_legacy_modules()

            opt = RiemannianMuon(
                w.parameters(),
                lr=self.scar_lr,
                scalar_lr=self.scalar_lr,
                momentum=0.86,
                ns_steps=5,
            )
            self.workers.append(w)
            self.optimizers.append(opt)
            self.streams.append(torch.cuda.Stream(device=self.device) if self.use_cuda_streams else None)

    @property
    def master_total_params(self) -> int:
        return sum(p.numel() for p in self.master.parameters())

    @property
    def mutable_params_per_worker(self) -> int:
        return sum(p.numel() for p in self.workers[0].parameters())

    @torch.no_grad()
    def sync_workers_from_master(self) -> None:
        for m_mod in self.master.modules():
            if isinstance(m_mod, E8SphericalLinear):
                m_mod.ensure_cached_quantized()
        m_params = dict(self.master.named_parameters())
        for w, opt in zip(self.workers, self.optimizers):
            for pname, pw in w.named_parameters():
                if pname in m_params:
                    pw.copy_(m_params[pname].detach())
                st = opt.state.get(pw)
                if st is not None and "momentum_buffer" in st:
                    st["momentum_buffer"].mul_(0.25)

    def run_swarm_round(
        self,
        batch_samplers: Sequence[Callable[[ScarFieldEngine, int], MacroBatchSpec]],
        local_steps: int = 12,
        grad_clip: float = 1.0,
    ) -> SharedCrystalSwarmTelemetry:
        """Evolves `S` ultralight scar workers on the shared `E_8` crystal, then distills into Master."""
        if len(batch_samplers) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} batch_samplers, got {len(batch_samplers)}")
        h_steps = max(1, int(local_steps))

        if self.use_cuda_streams:
            def_stream = torch.cuda.current_stream(self.device)
            for s in self.streams:
                assert s is not None
                s.wait_stream(def_stream)

        for h in range(h_steps):
            batches: List[MacroBatchSpec] = []
            for w, opt, sampler, strm in zip(self.workers, self.optimizers, batch_samplers, self.streams):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        opt.zero_grad(set_to_none=True)
                        batches.append(sampler(w, h))
                else:
                    opt.zero_grad(set_to_none=True)
                    batches.append(sampler(w, h))

            loss_dicts: List[Dict[str, torch.Tensor]] = []
            for w, bspec, strm in zip(self.workers, batches, self.streams):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        loss_dicts.append(
                            w.compute_loss(
                                bspec.byte_ids,
                                paired_translation_ids=bspec.paired_translation_ids,
                                out_of_cone_mask=bspec.out_of_cone_mask,
                                target_loss_mask=bspec.target_loss_mask,
                            )
                        )
                else:
                    loss_dicts.append(
                        w.compute_loss(
                            bspec.byte_ids,
                            paired_translation_ids=bspec.paired_translation_ids,
                            out_of_cone_mask=bspec.out_of_cone_mask,
                            target_loss_mask=bspec.target_loss_mask,
                        )
                    )

            for w, opt, ld, strm in zip(self.workers, self.optimizers, loss_dicts, self.streams):
                if strm is not None:
                    with torch.cuda.stream(strm):
                        ld["loss"].backward()
                        if grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                w.parameters(), grad_clip, foreach=(self.device.type == "cuda")
                            )
                        opt.step()
                else:
                    ld["loss"].backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            w.parameters(), grad_clip, foreach=(self.device.type == "cuda")
                        )
                    opt.step()

        if self.use_cuda_streams:
            def_stream = torch.cuda.current_stream(self.device)
            for s in self.streams:
                assert s is not None
                def_stream.wait_stream(s)

        return self.distill_into_master_crystal()

    @torch.no_grad()
    def distill_into_master_crystal(self) -> SharedCrystalSwarmTelemetry:
        """Distills `S` workers' ternary scars + pole updates directly into Master's `1.000 bpw` `E_8` crystal."""
        total_conflicts = 0
        layers_distilled = 0
        scar_innov_fros: List[float] = []

        # 1. Distill low-rank ternary scar innovations `Delta S_s = (gamma_s B_s A_s - gamma_0 B_0 A_0) / sqrt(I*r)`
        #    into Master's spherical weight substrate (`Delta = +0 B` model size!)
        m_mods = dict(self.master.named_modules())
        w_mod_maps = [dict(w.named_modules()) for w in self.workers]

        for mname, m_mod in m_mods.items():
            if not isinstance(m_mod, E8SphericalLinear):
                continue
            if (
                m_mod.scar_rank <= 0
                or m_mod.weight_sphere is None
                or m_mod.tau_base is None
                or m_mod.a_scar is None
                or m_mod.b_scar is None
                or m_mod.gamma_scar is None
            ):
                continue

            scale_u = 1.0 / math.sqrt(float(m_mod.in_features * m_mod.scar_rank))
            a0_q, _ = quantize_ternary_ste(m_mod.a_scar.detach().float())
            b0_q, _ = quantize_ternary_ste(m_mod.b_scar.detach().float())
            s0_mat = (m_mod.gamma_scar.detach().float() * scale_u) * (b0_q @ a0_q)  # [O, I]

            innovs: List[torch.Tensor] = []
            for wm in w_mod_maps:
                w_mod = wm[mname]
                assert isinstance(w_mod, E8SphericalLinear)
                assert w_mod.a_scar is not None and w_mod.b_scar is not None and w_mod.gamma_scar is not None
                as_q, _ = quantize_ternary_ste(w_mod.a_scar.detach().float())
                bs_q, _ = quantize_ternary_ste(w_mod.b_scar.detach().float())
                ss_mat = (w_mod.gamma_scar.detach().float() * scale_u) * (bs_q @ as_q)
                innovs.append(ss_mat - s0_mat)

            merged_innov, conf_s, _ = biorthogonal_conflict_free_tangent_merge(
                innovs, w0_sphere_2d=None, ridge_lambda=self.merge_ridge_lambda
            )
            total_conflicts += conf_s
            fro_val = float(torch.linalg.matrix_norm(merged_innov, ord="fro").item())
            scar_innov_fros.append(fro_val)
            if fro_val > 1e-8:
                tau_f = m_mod.tau_base.detach().float()
                w_u0 = project_rows_to_unit_sphere(m_mod.weight_sphere.detach().float())
                w_target = tau_f * w_u0 + self.distill_eta * merged_innov
                m_mod.weight_sphere.copy_(project_rows_to_unit_sphere(w_target).to(m_mod.weight_sphere.dtype))
                layers_distilled += 1

        # 2. Merge non-E8 spherical poles & 1D/scalar parameters via Biorthogonal Tangent Surgery
        m_params = dict(self.master.named_parameters())
        w_params = [dict(w.named_parameters()) for w in self.workers]
        for pname, p0 in m_params.items():
            if any(pname.endswith(s) for s in (".weight_sphere", ".tau_base", ".a_scar", ".b_scar", ".gamma_scar")):
                continue
            if pname not in w_params[0]:
                continue
            if p0.ndim == 2 and p0.shape[1] >= 4:
                d_poles = [riemannian_log_chord_2d(p0, wp[pname]) for wp in w_params]
                merged_p, conf_p, _ = biorthogonal_conflict_free_tangent_merge(
                    d_poles, w0_sphere_2d=p0, ridge_lambda=self.merge_ridge_lambda
                )
                total_conflicts += conf_p
                p0.copy_(riemannian_exp_retract_2d(p0, self.distill_eta * merged_p))
            else:
                d_1d = [wp[pname].detach().float() - p0.detach().float() for wp in w_params]
                if p0.numel() == 1:
                    mags = torch.stack([d.abs().reshape(()) for d in d_1d])
                    max_m = float(mags.max().item())
                    if max_m > 1e-9:
                        ws = mags + 0.05 * max_m
                        ws = ws / ws.sum().clamp(min=1e-12)
                        merged_1 = sum(ws[k] * d_1d[k] for k in range(len(d_1d)))
                        p0.add_((self.distill_eta * merged_1).to(p0.dtype))
                else:
                    merged_1, conf_1, _ = biorthogonal_conflict_free_tangent_merge(
                        d_1d, w0_sphere_2d=None, ridge_lambda=self.merge_ridge_lambda
                    )
                    total_conflicts += conf_1
                    p0.add_((self.distill_eta * merged_1).to(p0.dtype))
                if "biorth_gain" in pname and self.biorth_gain_floor > 0.0:
                    p0.clamp_(min=self.biorth_gain_floor)

        # 3. Re-crystallize Master's `1.000 bpw` Gosset `E_8` lattice + optimal residual scars & refresh workers!
        heal_all_e8_scars(self.master)
        self.sync_workers_from_master()

        m_tot = self.master_total_params
        w_mut = self.mutable_params_per_worker
        return SharedCrystalSwarmTelemetry(
            num_swarm_workers=self.num_workers,
            mutable_params_per_worker=w_mut,
            master_total_params=m_tot,
            param_reduction_ratio_x=round(float(m_tot) / float(max(1, w_mut)), 2),
            conflict_pairs_projected=total_conflicts,
            crystal_layers_distilled=layers_distilled,
            mean_scar_innovation_fro=round(float(sum(scar_innov_fros) / max(1, len(scar_innov_fros))), 5),
        )
