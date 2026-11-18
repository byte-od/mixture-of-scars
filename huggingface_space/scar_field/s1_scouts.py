"""System 1 Frontier Scouts (`Batedores de Fronteira`) for `ScarField v7.4-Turbo` Training.

Brings the three canonical `System One` Scout accelerators (`LOOKAHEAD_INTUITION_SCAR_ARCHITECTURE`
& `system1_engine/training/ppo.py`) natively into Zero-Legacy `ScarField` training:
  1. `build_scout_g0_step_mask` (`Gate G0` Intra-Sequence Fast-Path):
       - Predictable `System 1` bytes (`surprise_t < scout_g0_theta`) bypass deep macro-cycles
         (`0` FLOPs in `extra_flows` via exact sparse compaction in `_ReversibleLeapfrogAutograd`),
         while remaining 100% causal and bit-identical between batch `forward_sequence` and
         interactive `step_byte_interactive`!
  2. `System1CorpusScoutFleet` (`record_curriculum = False` Asynchronous Frontier Batch Prospector):
       - Scans `M * B` candidate corpus/SFT windows using ONLY the `670x`-lighter `System 1`
         `ByteShockFrontEnd` + `Juiz` `V(h_world)` (`no_grad`), mining the `Top-(1 - alpha)*B`
         epistemic frontier shock windows + `alpha * B` stratified base anchors.
  3. `compute_ghost_jump_antiloop_loss` (Zero-Copy Multi-Lag Repulsive Scar Regularizer):
       - Vectorized zero-`Softmax` `C^1` spherical-cap repulsive energy penalizing cyclic attractors
         `<h_world_t, h_world_{t - lag}> > kappa_loop` whenever target bytes differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from scar_field.cayley_geodesic import c1_spherical_cap_resonance


@dataclass(frozen=True)
class ScoutBatchTelemetry:
    """Audit telemetry emitted by `System1CorpusScoutFleet` (`record_curriculum = False`)."""

    candidates_scanned: int
    selected_batch_size: int
    mean_candidate_shock: float
    mean_selected_shock: float
    epistemic_gain_ratio: float
    degenerate_windows_filtered: int


def build_scout_g0_step_mask(
    surprise: torch.Tensor,
    s2_steps: int,
    s2_surprise_theta: float,
    scout_g0_theta: float = 0.0,
    allow_full_bypass: bool = False,
) -> torch.Tensor:
    """Builds causal `[B, T, K]` step schedule with `System 1` `Gate G0` fast-path support.

    - `surprise >= s2_surprise_theta`: `S2` frontier bifurcation (`K` fine steps of `1/K`).
    - `scout_g0_theta <= surprise < s2_surprise_theta`: `S1` normal pulse (`1` coarse step of `1.0`).
    - `surprise < scout_g0_theta` (when `allow_full_bypass=True` on `extra_flows`):
      `Gate G0` Fast-Path (`0` steps -> skipped completely by `_ReversibleLeapfrogAutograd`!).
    """
    k_max = max(1, int(s2_steps))
    if k_max == 1 and not (allow_full_bypass and scout_g0_theta > 0.0):
        return torch.ones(*surprise.shape, 1, dtype=surprise.dtype, device=surprise.device)

    is_s2 = surprise >= float(s2_surprise_theta)
    mask = torch.zeros(*surprise.shape, k_max, dtype=surprise.dtype, device=surprise.device)
    step0 = torch.where(is_s2, torch.ones_like(surprise), torch.full_like(surprise, float(k_max)))
    if allow_full_bypass and scout_g0_theta > 0.0:
        keep = surprise >= float(scout_g0_theta)
        step0 = torch.where(keep, step0, torch.zeros_like(step0))
    mask[..., 0] = step0
    if k_max > 1:
        mask[..., 1:] = is_s2.float().unsqueeze(-1).expand(*surprise.shape, k_max - 1)
    return mask


def compute_ghost_jump_antiloop_loss(
    h_world: torch.Tensor,
    target_bytes: torch.Tensor,
    target_loss_mask: Optional[torch.Tensor] = None,
    lags: Sequence[int] = (4, 8, 12, 16),
    kappa_loop: float = 0.90,
) -> torch.Tensor:
    """Zero-`Softmax` multi-lag repulsive scar loss preventing SFT cyclic repetition attractors."""
    seq_len = target_bytes.shape[1]
    hw = h_world[:, :seq_len, :]
    kap_t = hw.new_tensor(float(kappa_loop))
    penalties = []
    for lag in lags:
        if seq_len <= lag:
            continue
        sim = (hw[:, lag:, :] * hw[:, :-lag, :].detach()).sum(dim=-1)  # [B, L - lag]
        diff_tok = (target_bytes[:, lag:] != target_bytes[:, :-lag]).float()
        if target_loss_mask is not None:
            diff_tok = diff_tok * target_loss_mask[:, lag:].to(dtype=diff_tok.dtype, device=diff_tok.device)
        rep = c1_spherical_cap_resonance(sim, kap_t) * diff_tok
        denom = diff_tok.sum().clamp(min=1.0)
        penalties.append(rep.sum() / denom)
    if not penalties:
        return h_world.new_zeros(())
    return sum(penalties) / float(len(penalties))


class System1CorpusScoutFleet:
    """Decoupled `System 1` Frontier Scout Fleet (`record_curriculum = False`) for `M:1` Batch Mining."""

    def __init__(
        self,
        oversample_factor: int = 4,
        anchor_fraction: float = 0.25,
    ) -> None:
        if oversample_factor < 1:
            raise ValueError(f"oversample_factor must be >= 1, got {oversample_factor}")
        if not (0.0 <= anchor_fraction <= 1.0):
            raise ValueError(f"anchor_fraction must be in [0, 1], got {anchor_fraction}")
        self.oversample_factor = int(oversample_factor)
        self.anchor_fraction = float(anchor_fraction)
        self.record_curriculum: bool = False  # Canonical System 1 Scout Curriculum Isolation!
        self._async_tel_acc: Optional[torch.Tensor] = None

    @torch.no_grad()
    def flush_async_telemetry(self) -> ScoutBatchTelemetry:
        """Flushes accumulated GPU-side telemetry in a single host sync at outer Macro-Scout boundaries."""
        if self._async_tel_acc is None:
            return ScoutBatchTelemetry(0, 0, 0.0, 0.0, 1.0, 0)
        vals = self._async_tel_acc.detach().cpu().tolist()
        self._async_tel_acc = None
        steps = max(1.0, float(vals[3]))
        cand_m = float(vals[1]) / steps
        sel_m = float(vals[2]) / steps
        return ScoutBatchTelemetry(
            candidates_scanned=int(round(vals[4])),
            selected_batch_size=int(round(vals[5])),
            mean_candidate_shock=round(cand_m, 5),
            mean_selected_shock=round(sel_m, 5),
            epistemic_gain_ratio=round(sel_m / max(1e-6, cand_m), 4),
            degenerate_windows_filtered=int(round(vals[0])),
        )

    @torch.no_grad()
    def _score_candidates_tensors(
        self,
        engine: torch.nn.Module,
        candidate_byte_ids: torch.Tensor,
        candidate_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores `[N_cand, T]` windows on-device (`0` GPU->CPU syncs!)."""
        _, hw_raw, surprise = engine.frontend(candidate_byte_ids)  # type: ignore[operator]
        v_pred = engine.head.value_judge(hw_raw)  # type: ignore[operator]
        v_uncert = (1.0 - v_pred.abs()).clamp(0.0, 1.0)

        s_tok = surprise[:, 1:]
        u_tok = v_uncert[:, 1:]
        if candidate_loss_mask is not None:
            m = candidate_loss_mask.to(dtype=s_tok.dtype, device=s_tok.device)
            m_sum = m.sum(dim=-1).clamp(min=1.0)
            mean_s = (s_tok * m).sum(dim=-1) / m_sum
            mean_u = (u_tok * m).sum(dim=-1) / m_sum
            masked_s = torch.where(m > 0.0, s_tok, torch.zeros_like(s_tok))
        else:
            mean_s = s_tok.mean(dim=-1)
            mean_u = u_tok.mean(dim=-1)
            masked_s = s_tok

        k_top = max(1, masked_s.shape[1] // 8)
        top_s = torch.topk(masked_s, k=k_top, dim=-1).values.mean(dim=-1)

        # Filter degenerate constant/repetitive byte windows while ignoring `0x00` tail padding!
        non_pad = ((candidate_byte_ids[:, 1:] != 0) | (candidate_byte_ids[:, :-1] != 0)).float()
        non_pad_cnt = non_pad.sum(dim=-1)
        same_adj = ((candidate_byte_ids[:, 1:] == candidate_byte_ids[:, :-1]).float() * non_pad).sum(
            dim=-1
        ) / non_pad_cnt.clamp(min=1.0)
        is_degen = (non_pad_cnt < 4.0) | (same_adj > 0.75)
        scores = mean_s + 0.45 * top_s + 0.15 * mean_u - 2.0 * is_degen.float()
        return scores, is_degen.float().sum()

    @torch.no_grad()
    def score_candidates(
        self,
        engine: torch.nn.Module,
        candidate_byte_ids: torch.Tensor,
        candidate_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Scores `[N_cand, T]` windows using ONLY `System 1` `frontend` + `Juiz` (`< 1%` cost!)."""
        scores, degen_sum = self._score_candidates_tensors(engine, candidate_byte_ids, candidate_loss_mask)
        return scores, int(degen_sum.item())

    @torch.no_grad()
    def select_frontier_batch(
        self,
        engine: torch.nn.Module,
        candidate_byte_ids: torch.Tensor,
        batch_size: int,
        candidate_loss_mask: Optional[torch.Tensor] = None,
        sync_telemetry: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], ScoutBatchTelemetry]:
        """Selects `batch_size` windows (`(1 - alpha)` top frontier shocks + `alpha` stratified anchors)."""
        n_cand = int(candidate_byte_ids.shape[0])
        if n_cand <= batch_size:
            tel = ScoutBatchTelemetry(
                candidates_scanned=n_cand,
                selected_batch_size=n_cand,
                mean_candidate_shock=0.0,
                mean_selected_shock=0.0,
                epistemic_gain_ratio=1.0,
                degenerate_windows_filtered=0,
            )
            return candidate_byte_ids, candidate_loss_mask, tel

        scores, degen_sum_t = self._score_candidates_tensors(engine, candidate_byte_ids, candidate_loss_mask)
        n_anchors = min(batch_size - 1, max(0, int(round(batch_size * self.anchor_fraction))))
        n_frontier = batch_size - n_anchors

        sorted_idx = torch.argsort(scores, descending=True)
        frontier_idx = sorted_idx[:n_frontier]
        if n_anchors > 0:
            rem = sorted_idx[n_frontier:]
            stride = max(1, rem.numel() // n_anchors)
            anchor_idx = rem[::stride][:n_anchors]
            chosen_idx = torch.cat((frontier_idx, anchor_idx), dim=0)
        else:
            chosen_idx = frontier_idx

        sel_bytes = candidate_byte_ids[chosen_idx]
        sel_mask = candidate_loss_mask[chosen_idx] if candidate_loss_mask is not None else None

        cand_m_t = scores.clamp(min=0.0).mean()
        sel_m_t = scores[chosen_idx].clamp(min=0.0).mean()
        if not sync_telemetry:
            # Zero-host-sync accumulation on GPU stream for Multi-Stream Macro-Scout concurrency!
            inc = torch.stack(
                (
                    degen_sum_t.float(),
                    cand_m_t.float(),
                    sel_m_t.float(),
                    scores.new_tensor(1.0),
                    scores.new_tensor(float(n_cand)),
                    scores.new_tensor(float(sel_bytes.shape[0])),
                )
            )
            if self._async_tel_acc is None or self._async_tel_acc.device != inc.device:
                self._async_tel_acc = inc
            else:
                self._async_tel_acc[:4] = self._async_tel_acc[:4] + inc[:4]
                self._async_tel_acc[4:] = inc[4:]
            return sel_bytes, sel_mask, ScoutBatchTelemetry(n_cand, int(sel_bytes.shape[0]), 0.0, 0.0, 1.0, 0)

        # Single coalesced 3-float host transfer instead of 3 sequential `.item()` stalls!
        stats_cpu = torch.stack((degen_sum_t, cand_m_t, sel_m_t)).cpu().tolist()
        degen_cnt = int(round(stats_cpu[0]))
        cand_mean = float(stats_cpu[1])
        sel_mean = float(stats_cpu[2])
        tel = ScoutBatchTelemetry(
            candidates_scanned=n_cand,
            selected_batch_size=int(sel_bytes.shape[0]),
            mean_candidate_shock=round(cand_mean, 5),
            mean_selected_shock=round(sel_mean, 5),
            epistemic_gain_ratio=round(sel_mean / max(1e-6, cand_mean), 4),
            degenerate_windows_filtered=degen_cnt,
        )
        return sel_bytes, sel_mask, tel
