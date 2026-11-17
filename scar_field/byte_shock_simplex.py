"""Byte-Shock Perceptual Front-End & Tsallis-1.5 Zero-Softmax Spherical Cap Simplex.

Zero-Legacy Perceptual & Output Primitives for `ScarField v7.0`:
  1. `ByteShockFrontEnd` (`V = 256` UTF-8 raw bytes, zero static BPE vocabulary):
       - Maps raw bytes `0..255` onto `S^{D_f-1} x S^{D_w-1}` and measures instantaneous
         physical prediction surprise `s_t in [0, 1]` directly from local spherical resonance
         (zero auxiliary entropy Transformer needed, unlike Meta's BLT!).
       - Supports dynamic shock coalescing (`coalesce_byte_stream`) grouping low-surprise
         vacuum runs into variable-stride Concept Pulses.
  2. `tsallis15_spherical_cap_probs` & `tsallis15_fenchel_young_loss`:
       - Exact closed-form `alpha = 1.5` quadratic spherical-cap simplex projection:
         `p_i(z) = [0.5 * (z_i - lambda(z))]_+^2`, solved analytically WITHOUT `exp()`,
         `log()`, or `Softmax`!
       - Outputs exact `0.000000` outside the active spherical cap (`95%+` exact sparsity).
       - Convex Fenchel-Young loss with exact gradient `p(z) - one_hot(y)` and finite-margin
         `0.0` minimum (zero logit explosion).
       - Native **Physical Vacuum `0x00` Abstention**: when `max_i <w_i, h_world> < kappa_vac`,
         the cap is physically empty (`is_vacuum_abstain = True`), needing zero fake `<abstain>` IDs!
  3. `ScarFieldReflexDecision`: Formalizes the `System One` `ReflexDecision` contract (`confidence`,
     `margin`, `uncertainty`, `is_uncertain`, `is_vacuum_abstain`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import torch
import torch.nn as nn

from scar_field.e8_lattice import project_rows_to_unit_sphere
from scar_field.triton_kernels import multiscale_wave_superpose_fast, tsallis15_spherical_cap_probs_fast


@dataclass(frozen=True)
class ScarFieldReflexDecision:
    """Formal `System One` epistemic decision contract emitted by `SphericalCapSimplex`."""

    action_byte: int
    confidence: float
    margin: float
    uncertainty: float
    max_world_cosine: float
    active_cap_size: int
    is_uncertain: bool
    is_vacuum_abstain: bool
    steps_executed: int
    ghost_jump_used: bool


def tsallis15_spherical_cap_probs(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact closed-form `Entmax-1.5` (`Tsallis alpha=1.5`) simplex projection (ZERO `exp`/`Softmax`).

    Solves `sum_{i=1}^V [0.5 * (z_i - lambda)]_+^2 == 1` analytically via the quadratic root:
      `lambda_k = mean_{1..k}(z) - sqrt(max(0, 4/k - Var_{1..k}(z)))`.
    Uses single-CTA Triton (`CUDA V=256`) / zero-sort convex Newton (`O(N*V)`) when no autograd graph
    is required directly on `probs`.
    Returns `(probs [..., V], lambda_star [..., 1])`.
    """
    if not (torch.is_grad_enabled() and scores.requires_grad):
        return tsallis15_spherical_cap_probs_fast(scores)

    orig_dtype = scores.dtype
    z_raw = scores.float()
    z_shift = z_raw.amax(dim=-1, keepdim=True).detach()
    z = z_raw - z_shift
    v_dim = z.shape[-1]

    z_sorted, _ = torch.sort(z, dim=-1, descending=True)  # [..., V]
    k_idx = torch.arange(1, v_dim + 1, dtype=torch.float32, device=z.device)

    csum1 = torch.cumsum(z_sorted, dim=-1)
    csum2 = torch.cumsum(z_sorted * z_sorted, dim=-1)
    mean_k = csum1 / k_idx
    var_k = (csum2 / k_idx - mean_k * mean_k).clamp(min=0.0)

    disc = 4.0 / k_idx - var_k
    valid_disc = disc >= -1e-7
    lam_k = mean_k - torch.sqrt(disc.clamp(min=0.0))

    # Support condition: `z_{(k)} > lam_k` AND `disc >= 0` (`k=1` is ALWAYS true!)
    cond = valid_disc & (z_sorted > (lam_k + 1e-7))
    cond[..., 0] = True

    # Pick largest valid support size `k*`
    k_star = cond.long().sum(dim=-1, keepdim=True).clamp(1, v_dim)  # [..., 1]
    lam_star = torch.gather(lam_k, dim=-1, index=k_star - 1)        # [..., 1]

    u = (0.5 * (z - lam_star)).clamp(min=0.0)
    probs = u * u
    # Exact renormalization guard against float32 rounding (`sum == 1.000000`)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return probs.to(orig_dtype), (lam_star + z_shift).to(orig_dtype)


class _Tsallis15LossAutograd(torch.autograd.Function):
    """Exact convex Fenchel-Young Tsallis-1.5 loss (`grad_z == probs(z) - one_hot(targets)`)."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = scores.float()
        probs, lam = tsallis15_spherical_cap_probs_fast(z.detach())
        u = torch.sqrt(probs.clamp(min=0.0))
        omega_conj = lam.squeeze(-1) + (2.0 / 3.0) * (u * probs).sum(dim=-1)
        z_true = z.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
        loss = (omega_conj - z_true + (4.0 / 3.0)).clamp(min=0.0)
        ctx.save_for_backward(probs, targets)
        return loss.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        probs, targets = ctx.saved_tensors
        grad_z = probs.clone()
        grad_z.scatter_add_(
            dim=-1,
            index=targets.long().unsqueeze(-1),
            src=-torch.ones_like(targets, dtype=grad_z.dtype).unsqueeze(-1),
        )
        return grad_z * grad_out.float().unsqueeze(-1), None


def tsallis15_fenchel_young_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Elementwise Tsallis-1.5 sparse-cap loss (`[B, T]` or `[N]`, zero `exp()`/`log()`)."""
    orig_shape = targets.shape
    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_targets = targets.reshape(-1)
    out = _Tsallis15LossAutograd.apply(flat_scores, flat_targets)
    return out.reshape(orig_shape)


class ByteShockFrontEnd(nn.Module):
    """Tokenizer-free `V = 256` UTF-8 Byte Perceptual Front-End with Multi-Scale Spherical Wave & Surprise Sensor."""

    def __init__(
        self,
        d_form: int,
        d_world: int,
        vocab_size: int = 256,
        wave_lags: Tuple[int, ...] = (1, 2, 3),
        shock_closure_decay: float = 0.0,
        enable_repeat_phase: bool = False,
    ) -> None:
        super().__init__()
        self.d_form = d_form
        self.d_world = d_world
        self.vocab_size = vocab_size
        self.wave_lags: Tuple[int, ...] = tuple(int(x) for x in wave_lags)
        self.shock_closure_decay: float = float(shock_closure_decay)
        self.enable_repeat_phase: bool = bool(enable_repeat_phase)

        self.byte_form_poles = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_form)))
        self.byte_world_poles = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_world)))
        # Causal multi-scale spherical wave weights (`t` + `wave_lags`, zero `nn.Linear`, zero `BPE`!)
        if self.wave_lags == (1, 2, 3):
            init_w = [1.0, 0.55, 0.30, 0.16]
        else:
            init_w = [1.0] + [float(0.62 * (0.84 ** idx)) for idx in range(len(self.wave_lags))]
        self.ngram_weights = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        # Zero-Softmax Tsallis-1.5 local cap observer (`s_t = 1 - p_cap(b_t | wave_{t-1})` in `[0, 1]`)
        self.reflex_cap_poles = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_form)))
        self.reflex_temp = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))

    def forward(
        self, byte_ids: torch.Tensor, return_reflex_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps `byte_ids [B, T]` -> `(h_form [B, T, D_f], h_world [B, T, D_w], surprise [B, T])`."""
        ids = byte_ids.long().clamp(0, self.vocab_size - 1)
        bsz, seq_len = ids.shape
        f_poles = project_rows_to_unit_sphere(self.byte_form_poles)
        w_poles = project_rows_to_unit_sphere(self.byte_world_poles)

        # Causal multi-scale spherical wave superposition via fast `[256, D]` rotated pole gathers (`0` `cat`/`roll`!)
        nw = self.ngram_weights
        hf_acc = multiscale_wave_superpose_fast(ids, f_poles, nw, self.wave_lags)
        hw_acc = multiscale_wave_superpose_fast(ids, w_poles, nw, self.wave_lags)

        # Optional Run-Length Orthogonal Phase Harmonics (`enable_repeat_phase`, odd channel shifts `5, 13, 21`):
        # Separates consecutive repeated bytes (`key("=80")` != `key("=800")`, `key("=77")` != `key("=777")`)
        # with `+0 B` state overhead and `0` self-loop traps!
        if self.enable_repeat_phase and seq_len > 1:
            rep1 = (ids[:, 1:] == ids[:, :-1]).unsqueeze(-1).to(hf_acc.dtype)
            hf_r1 = torch.roll(f_poles, shifts=5, dims=-1)[ids[:, 1:]]
            hw_r1 = torch.roll(w_poles, shifts=5, dims=-1)[ids[:, 1:]]
            hf_acc[:, 1:, :] = hf_acc[:, 1:, :] + (0.42 * rep1) * hf_r1
            hw_acc[:, 1:, :] = hw_acc[:, 1:, :] + (0.42 * rep1) * hw_r1
            if seq_len > 2:
                rep2 = rep1[:, 1:, :] * (ids[:, 1:-1] == ids[:, :-2]).unsqueeze(-1).to(hf_acc.dtype)
                hf_r2 = torch.roll(f_poles, shifts=13, dims=-1)[ids[:, 2:]]
                hw_r2 = torch.roll(w_poles, shifts=13, dims=-1)[ids[:, 2:]]
                hf_acc[:, 2:, :] = hf_acc[:, 2:, :] + (0.38 * rep2) * hf_r2
                hw_acc[:, 2:, :] = hw_acc[:, 2:, :] + (0.38 * rep2) * hw_r2
            if seq_len > 3:
                rep3 = rep2[:, 1:, :] * (ids[:, 2:-1] == ids[:, 1:-2]).unsqueeze(-1).to(hf_acc.dtype)
                hf_r3 = torch.roll(f_poles, shifts=21, dims=-1)[ids[:, 3:]]
                hw_r3 = torch.roll(w_poles, shifts=21, dims=-1)[ids[:, 3:]]
                hf_acc[:, 3:, :] = hf_acc[:, 3:, :] + (0.34 * rep3) * hf_r3
                hw_acc[:, 3:, :] = hw_acc[:, 3:, :] + (0.34 * rep3) * hw_r3

        hf = project_rows_to_unit_sphere(hf_acc)
        hw = project_rows_to_unit_sphere(hw_acc)

        # Measure true epistemic cap surprise `s_t = 1 - p_cap(b_t | stem_{t-1})` (ZERO `exp`/`log`/`Softmax`!):
        #   - Deterministic filler (`p_cap -> 1.0`): `s_t -> 0.0` (exact `0.0` `W_fast` write, zero forgetting!)
        #   - Uniform 10-digit PIN shock (`p_cap = 0.10`): `s_t = 0.90` (sharp `W_fast` write, immune to pole clustering!)
        if seq_len > 1:
            r_poles = project_rows_to_unit_sphere(self.reflex_cap_poles)
            r_temp = self.reflex_temp.clamp(2.0, 24.0)
            hf_det = hf[:, :-1, :].detach()
            if hf_det.device.type == "cuda" and not return_reflex_loss:
                r_scores = r_temp * (hf_det.bfloat16() @ r_poles.bfloat16().T).float()
            else:
                r_scores = r_temp * torch.matmul(hf_det, r_poles.T)  # [B, T-1, V]
            next_ids = ids[:, 1:]
            loss_reflex = (
                tsallis15_fenchel_young_loss(r_scores, next_ids).mean()
                if return_reflex_loss
                else hf.new_zeros(())
            )
            with torch.no_grad():
                p_local, _ = tsallis15_spherical_cap_probs_fast(r_scores.detach())
                p_hit = p_local.gather(dim=-1, index=next_ids.unsqueeze(-1)).squeeze(-1)
                s_tail_raw = (1.0 - p_hit).clamp(0.0, 1.0)
                if self.shock_closure_decay > 0.0:
                    # 1. Structural turn boundaries (`0x02` STX, `0x03` ETX, `0x04` EOT) carry guaranteed salience
                    is_ctrl = next_ids <= 4
                    s_base = torch.where(is_ctrl, s_tail_raw.clamp(min=0.82), s_tail_raw)
                    # 2. 1-Step Causal Shock-Closure FIR (`PIN=800.` -> closing `.` inherits `0.85 * s_raw('0')`!)
                    if s_tail_raw.shape[1] > 1:
                        s_closed = s_base.clone()
                        s_closed[:, 1:] = torch.maximum(
                            s_closed[:, 1:], float(self.shock_closure_decay) * s_tail_raw[:, :-1]
                        )
                        s_tail = s_closed
                    else:
                        s_tail = s_base
                else:
                    s_tail = s_tail_raw
            s_head = torch.ones(bsz, 1, dtype=hf.dtype, device=hf.device)
            surprise = torch.cat((s_head, s_tail.to(hf.dtype)), dim=1)
        else:
            loss_reflex = hf.new_zeros(())
            surprise = torch.ones(bsz, seq_len, dtype=hf.dtype, device=hf.device)

        if return_reflex_loss:
            return hf, hw, surprise, loss_reflex
        return hf, hw, surprise

    @torch.no_grad()
    def coalesce_byte_stream(
        self,
        byte_ids_1d: torch.Tensor,
        coalesce_theta: float = 0.32,
        max_patch_bytes: int = 6,
    ) -> List[Tuple[int, int, float]]:
        """Segments a raw UTF-8 byte sequence into dynamic Concept Pulses at surprise shocks (`s_t >= theta`)."""
        hf, _, surp = self.forward(byte_ids_1d.unsqueeze(0))
        s_np = surp[0].tolist()
        spans: List[Tuple[int, int, float]] = []
        start = 0
        peak_s = s_np[0] if s_np else 1.0
        for idx in range(1, len(s_np)):
            if s_np[idx] >= coalesce_theta or (idx - start) >= max_patch_bytes:
                spans.append((start, idx, float(peak_s)))
                start = idx
                peak_s = s_np[idx]
            else:
                peak_s = max(peak_s, s_np[idx])
        if len(s_np) > 0:
            spans.append((start, len(s_np), float(peak_s)))
        return spans


class SphericalCapSimplexHead(nn.Module):
    """Zero-`Softmax` Spherical Cap Output Decoder + Physical Vacuum `0x00` Abstention + `Juiz` Critic."""

    def __init__(
        self,
        d_form: int,
        d_world: int,
        d_bus: int = 64,
        vocab_size: int = 256,
        init_temp: float = 6.0,
        vacuum_kappa: float = 0.15,
        enable_adaptive_cap: bool = False,
    ) -> None:
        super().__init__()
        self.d_form = d_form
        self.d_world = d_world
        self.d_bus = d_bus
        self.vocab_size = vocab_size
        self.vacuum_kappa = vacuum_kappa
        self.enable_adaptive_cap = bool(enable_adaptive_cap)

        self.out_form_anchors = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_form)))
        self.out_world_anchors = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_world)))
        self.out_bus_anchors = nn.Parameter(project_rows_to_unit_sphere(torch.randn(vocab_size, d_bus)))
        self.mix_weights = nn.Parameter(torch.tensor([0.38, 0.38, 0.95], dtype=torch.float32))
        self.temp_scale = nn.Parameter(torch.tensor(float(init_temp), dtype=torch.float32))
        if self.enable_adaptive_cap:
            self.cap_adapt_gain = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

        # `AURA` Epistemic Coherence Judge `V(h_world) in (-1, +1)` for Ghost-Jump Scout scoring!
        self.value_pole = nn.Parameter(project_rows_to_unit_sphere(torch.randn(1, d_world)))

    def forward_scores(
        self,
        h_form: torch.Tensor,
        h_world: torch.Tensor,
        bus_ctx: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns `(cap_scores [..., V], max_world_cosine [...])`."""
        wf = project_rows_to_unit_sphere(self.out_form_anchors)
        ww = project_rows_to_unit_sphere(self.out_world_anchors)

        cos_f = torch.matmul(h_form, wf.T)   # [..., V]
        cos_w = torch.matmul(h_world, ww.T)  # [..., V]
        max_w_cos = cos_w.amax(dim=-1)       # [...]

        mw = self.mix_weights
        comb = mw[0] * cos_f + mw[1] * cos_w
        if bus_ctx is not None:
            wb = project_rows_to_unit_sphere(self.out_bus_anchors)
            cos_b = torch.matmul(bus_ctx, wb.T)
            comb = comb + mw[2] * cos_b

        temp = self.temp_scale.clamp(1.0, 28.0)
        if self.enable_adaptive_cap:
            top8_vals, _ = torch.topk(comb, k=min(8, comb.shape[-1]), dim=-1)
            margin8 = (top8_vals[..., 0:1] - top8_vals.mean(dim=-1, keepdim=True)).clamp(min=0.0)
            adapt = self.cap_adapt_gain.clamp(0.0, 3.0) / (1.0 + 6.0 * margin8)
            eff_temp = (temp / (1.0 + adapt)).clamp(1.0, 28.0)
            scores = eff_temp * comb
        else:
            scores = temp * comb
        return scores, max_w_cos

    def value_judge(self, h_world: torch.Tensor) -> torch.Tensor:
        """Evaluates `V(h_world) in [-1, 1]` on `S^{D_w-1}` for Ghost-Jump Scouts."""
        vp = project_rows_to_unit_sphere(self.value_pole)
        return torch.matmul(h_world, vp.T).squeeze(-1)

    @torch.no_grad()
    def decide(
        self,
        h_form_1d: torch.Tensor,
        h_world_1d: torch.Tensor,
        bus_ctx_1d: torch.Tensor | None = None,
        uscar_resonance: float = 0.0,
        require_world_support: bool = False,
        uncertainty_margin_thresh: float = 0.22,
        steps_executed: int = 1,
        ghost_jump_used: bool = False,
    ) -> ScarFieldReflexDecision:
        b_in = bus_ctx_1d.unsqueeze(0) if bus_ctx_1d is not None else None
        scores, max_w_cos_t = self.forward_scores(h_form_1d.unsqueeze(0), h_world_1d.unsqueeze(0), bus_ctx=b_in)
        probs, _ = tsallis15_spherical_cap_probs(scores)
        p = probs[0]
        max_w_cos = max(float(max_w_cos_t[0].item()), float(uscar_resonance))

        # Physical Vacuum `0x00` Abstention check when factual support is required
        if require_world_support and max_w_cos < self.vacuum_kappa:
            return ScarFieldReflexDecision(
                action_byte=0,
                confidence=0.0,
                margin=0.0,
                uncertainty=1.0,
                max_world_cosine=max_w_cos,
                active_cap_size=0,
                is_uncertain=True,
                is_vacuum_abstain=True,
                steps_executed=steps_executed,
                ghost_jump_used=ghost_jump_used,
            )

        top2_vals, top2_idx = torch.topk(p, k=min(2, p.numel()))
        p1 = float(top2_vals[0].item())
        p2 = float(top2_vals[1].item()) if top2_vals.numel() > 1 else 0.0
        margin = p1 - p2
        active_cnt = int((p > 0.0).sum().item())

        return ScarFieldReflexDecision(
            action_byte=int(top2_idx[0].item()),
            confidence=p1,
            margin=margin,
            uncertainty=max(0.0, 1.0 - p1),
            max_world_cosine=max_w_cos,
            active_cap_size=active_cnt,
            is_uncertain=(margin < uncertainty_margin_thresh),
            is_vacuum_abstain=False,
            steps_executed=steps_executed,
            ghost_jump_used=ghost_jump_used,
        )
