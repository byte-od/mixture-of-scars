"""ScarField v7.0 — Pure Post-Transformer Geodesic Scar Intelligence Engine.

Unifies all five Zero-Legacy mathematical pillars and pure `AURA / System One` organs:
  1. `ByteShockFrontEnd` (`V = 256` UTF-8 raw bytes, zero BPE, native surprise sensor `s_t`).
  2. Bipartite Product Manifold `S_form^{D_f-1} x S_world^{D_w-1}` (`||h||_2 == 1.0` from `t = 0`).
  3. `ChannelSelectiveKalmanScarBus` (`O(1)` central workspace, `Delta = +0 B`, relativistic
     proper-time `d_phi = omega * s_t`, parallel `Chunkwise WY`, Kinematic `Tri-Compass`,
     `RepulsiveScarRing`, and detached `.uscar` poles with `0x00` GDPR erasure).
  4. `ReversibleSymplecticCayleyFlow` (`1.000 bpw` Gosset `E_8` substrate + `C^1` horizon-culled
     scars, zero `ReLU/SwiGLU`, zero `RMSNorm`, exact `O(1)` activation memory during backprop,
     adaptive `System 1` `K=1` <-> `System 2` `K=s2_steps` execution).
  5. `SphericalCapSimplexHead` (Tsallis-1.5 exact zero-`Softmax` cap decoder, native `0x00`
     physical vacuum abstention, and Zero-Copy Latent Ghost-Jump Scouts + `Juiz` `V(h_world)`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from scar_codec.binary_format import ScarFieldMmapContainer, save_scarfield_v7_container

from scar_field.byte_shock_simplex import (
    ByteShockFrontEnd,
    ScarFieldReflexDecision,
    SphericalCapSimplexHead,
    tsallis15_fenchel_young_loss,
    tsallis15_spherical_cap_probs,
)
from scar_field.cayley_geodesic import ReversibleSymplecticCayleyFlow
from scar_field.e8_lattice import E8SphericalLinear, PackedE8TernaryMatrix, project_rows_to_unit_sphere
from scar_field.proper_time_bus import (
    ChannelSelectiveKalmanScarBus,
    KalmanBusState,
    VirtualGhostJumpDelta,
)
from scar_field.s1_scouts import (
    ScoutBatchTelemetry,
    System1CorpusScoutFleet,
    build_scout_g0_step_mask,
    compute_ghost_jump_antiloop_loss,
)

SCARFIELD_USER_BYTE: int = 0x02       # ASCII STX (Start of User Prompt)
SCARFIELD_ASSISTANT_BYTE: int = 0x03  # ASCII ETX (Start of Assistant Reply)
SCARFIELD_EOT_BYTE: int = 0x04        # ASCII EOT (Deterministic End of Reply; 0x00 stays Physical Vacuum!)


@dataclass
class ScarFieldChatReply:
    """Structured multi-turn conversational output emitted by `ScarFieldEngine.generate_reply`."""

    text: str
    bytes_generated: int
    abstained_0x00: bool
    mean_confidence: float
    s2_deliberation_fraction: float
    state_bytes: int
    state: KalmanBusState


def format_scarfield_chat_turn(user_text: str, assistant_text: Optional[str] = None) -> bytes:
    """Formats a raw UTF-8 conversational turn (`0` static BPE vocabulary needed!)."""
    clean_u = user_text.replace("\x00", "").replace("\x02", "").replace("\x03", "").replace("\x04", "").strip()
    prefix = b"\x02User: " + clean_u.encode("utf-8") + b"\n\x03Assistant: "
    if assistant_text is None:
        return prefix
    clean_a = assistant_text.replace("\x00", "").replace("\x02", "").replace("\x03", "").replace("\x04", "").strip()
    return prefix + clean_a.encode("utf-8") + b"\x04\n"


def encode_sft_dialogue_bytes_and_mask(turns: list[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes multi-turn `[(user, assistant), ...]` into `(byte_ids [L], target_mask [L-1])`.

    `target_mask[t] == 1.0` strictly when predicting `byte_ids[t + 1]` belonging to the Assistant reply
    or its terminating `0x04` EOT byte (`0.0` on User prompt bytes!).
    """
    raw_bytes = bytearray()
    is_target_pos: list[float] = []
    for u_txt, a_txt in turns:
        prefix = format_scarfield_chat_turn(u_txt, assistant_text=None)
        clean_a = a_txt.replace("\x00", "").replace("\x02", "").replace("\x03", "").replace("\x04", "").strip()
        reply = clean_a.encode("utf-8") + b"\x04"
        suffix = b"\n"

        raw_bytes.extend(prefix)
        is_target_pos.extend([0.0] * len(prefix))

        raw_bytes.extend(reply)
        is_target_pos.extend([1.0] * len(reply))

        raw_bytes.extend(suffix)
        is_target_pos.extend([0.0] * len(suffix))

    arr_u8 = np.asarray(raw_bytes, dtype=np.uint8)
    # `target_mask[t]` masks loss on target `byte_ids[t + 1]`!
    mask_f32 = np.asarray(is_target_pos[1:], dtype=np.float32)
    return arr_u8, mask_f32


@dataclass
class ScarFieldConfig:
    vocab_size: int = 256
    d_form: int = 64
    d_world: int = 64
    d_bus: int = 64
    d_key: int = 64
    num_subspaces: int = 8
    ring_len: int = 16
    repulsive_len: int = 8
    base_rank: int = 4
    num_scars: int = 16
    s1_steps: int = 1
    s2_steps: int = 4
    s2_surprise_theta: float = 0.45
    shock_write_theta: float = 0.50
    vacuum_kappa: float = 0.18
    uncertainty_margin_thresh: float = 0.24
    ghost_jump_candidates: int = 3
    ghost_jump_gamma: float = 0.35
    wy_chunk_size: int = 32
    enable_kalman_fast: bool = True
    wave_lags: Tuple[int, ...] = (1, 2, 3)
    num_macro_cycles: int = 1
    biorth_slots: int = 0
    biorth_lambda: float = 0.04
    enable_biorth_uscar: bool = True
    enable_adaptive_cap: bool = False
    scout_g0_theta: float = 0.0
    antiloop_weight: float = 0.0
    shock_closure_decay: float = 0.0
    enable_repeat_phase: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.wave_lags, list):
            object.__setattr__(self, "wave_lags", tuple(int(x) for x in self.wave_lags))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScarFieldConfig":
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        if "wave_lags" in filtered and isinstance(filtered["wave_lags"], list):
            filtered["wave_lags"] = tuple(int(x) for x in filtered["wave_lags"])
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScarFieldEngine(nn.Module):
    """Complete Post-Transformer `ScarField v7.0 / v7.5` Engine (Zero `nn.Linear`, `ReLU`, `Softmax`, `RoPE(t)`)."""

    def __init__(self, cfg: Optional[ScarFieldConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ScarFieldConfig()
        c = self.cfg
        if c.shock_closure_decay > 0.0 or c.enable_repeat_phase:
            eff_max_lag = max(max(c.wave_lags) if c.wave_lags else 0, 3 if c.enable_repeat_phase else 0)
            extra_lb = 2 if c.shock_closure_decay > 0.0 else 1
            max_wave_buf = max(6, eff_max_lag + extra_lb)
        else:
            max_wave_buf = max(4, (max(c.wave_lags) + 1) if c.wave_lags else 4)
        self._recent_buf_cap = max_wave_buf

        self.frontend = ByteShockFrontEnd(
            d_form=c.d_form,
            d_world=c.d_world,
            vocab_size=c.vocab_size,
            wave_lags=c.wave_lags,
            shock_closure_decay=c.shock_closure_decay,
            enable_repeat_phase=c.enable_repeat_phase,
        )
        # Stage 1: Perceptual Cayley Geodesic Flow (refines raw byte wave into sharp semantic queries)
        self.flow_percept = ReversibleSymplecticCayleyFlow(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_form,
            base_rank=c.base_rank,
            num_scars=max(8, c.num_scars // 2),
        )
        # Central Single O(1) Workspace (`Delta = +0 B` shared across entire engine!)
        self.bus = ChannelSelectiveKalmanScarBus(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_bus,
            d_key=c.d_key,
            num_subspaces=c.num_subspaces,
            ring_len=c.ring_len,
            repulsive_len=c.repulsive_len,
            shock_write_theta=c.shock_write_theta,
            enable_kalman_fast=c.enable_kalman_fast,
            recent_wave_buf_len=max_wave_buf,
            biorth_slots=c.biorth_slots,
            biorth_lambda=c.biorth_lambda,
            enable_biorth_uscar=c.enable_biorth_uscar,
        )
        # Stage 2: Reasoning/Synthesis Cayley Geodesic Flow (`S1` K=1 <-> `S2` K=s2_steps)
        self.flow = ReversibleSymplecticCayleyFlow(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_bus,
            base_rank=c.base_rank,
            num_scars=c.num_scars,
        )
        # Optional Deep Macro-Cycles (`num_macro_cycles >= 2` for `v7.2-Deep`, `0` overhead when `== 1`!)
        if c.num_macro_cycles > 1:
            self.extra_flows = nn.ModuleList(
                [
                    ReversibleSymplecticCayleyFlow(
                        d_form=c.d_form,
                        d_world=c.d_world,
                        d_bus=c.d_bus,
                        base_rank=c.base_rank,
                        num_scars=c.num_scars,
                    )
                    for _ in range(c.num_macro_cycles - 1)
                ]
            )
        else:
            self.extra_flows = None
        self.head = SphericalCapSimplexHead(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_bus,
            vocab_size=c.vocab_size,
            vacuum_kappa=c.vacuum_kappa,
            enable_adaptive_cap=c.enable_adaptive_cap,
        )
        self.assert_zero_legacy_modules()

    def assert_zero_legacy_modules(self) -> None:
        """Forensic architectural integrity guard: rejects any pre-scar legacy PyTorch layer."""
        forbidden = (
            nn.Linear,
            nn.LayerNorm,
            nn.RMSNorm,
            nn.ReLU,
            nn.GELU,
            nn.SiLU,
            nn.Softmax,
            nn.LogSoftmax,
            nn.CrossEntropyLoss,
        )
        for name, mod in self.named_modules():
            if isinstance(mod, forbidden):
                raise AssertionError(f"Forbidden pre-scar legacy module `{type(mod).__name__}` at `{name}`!")

    def init_state(self, batch_size: int = 1, device: torch.device | str = "cpu") -> KalmanBusState:
        return self.bus.init_state(batch_size=batch_size, device=device)

    def _build_adaptive_step_mask(self, surprise: torch.Tensor) -> torch.Tensor:
        """Builds bit-exact `S1 (1-step)` vs `S2 (K-step)` step schedule mask `[B, T, K]`."""
        k_max = self.cfg.s2_steps
        if k_max <= 1:
            return torch.ones(*surprise.shape, 1, dtype=surprise.dtype, device=surprise.device)
        is_s2 = surprise >= self.cfg.s2_surprise_theta  # [B, T]
        mask = torch.zeros(*surprise.shape, k_max, dtype=surprise.dtype, device=surprise.device)
        # S2 tokens take `K` fine steps of `1/K`; S1 tokens take `1` single step of `K * (1/K) == 1.0`!
        mask[..., 0] = torch.where(is_s2, torch.ones_like(surprise), torch.full_like(surprise, float(k_max)))
        mask[..., 1:] = is_s2.float().unsqueeze(-1).expand(*surprise.shape, k_max - 1)
        return mask

    def forward_sequence(
        self,
        byte_ids: torch.Tensor,
        state: Optional[KalmanBusState] = None,
        adaptive_depth: bool = True,
    ) -> Dict[str, Any]:
        if state is not None and state.recent_len > 0:
            hist = state.recent_bytes[:, -state.recent_len :].long()
            full_ids = torch.cat((hist, byte_ids.long()), dim=1)
            hf_all, hw_all, surp_all, loss_reflex = self.frontend(full_ids, return_reflex_loss=True)  # type: ignore[misc]
            seq_len = byte_ids.shape[1]
            hf_raw = hf_all[:, -seq_len:]
            hw_raw = hw_all[:, -seq_len:]
            surprise = surp_all[:, -seq_len:]
        else:
            full_ids = byte_ids.long()
            hf_raw, hw_raw, surprise, loss_reflex = self.frontend(full_ids, return_reflex_loss=True)  # type: ignore[misc]

        hf0, hw0 = self.flow_percept(hf_raw, hw_raw, hf_raw, num_steps=2)
        v_sym = project_rows_to_unit_sphere(self.head.out_bus_anchors)[byte_ids.long().clamp(0, self.cfg.vocab_size - 1)]
        hf_phased, bus_ctx, uscar_res, next_state = self.bus.forward_sequence(
            hf0, hw0, surprise, state=state, chunk_size=self.cfg.wy_chunk_size, v_shock_hint=v_sym
        )
        tail_n = min(self._recent_buf_cap, full_ids.shape[1])
        next_state.recent_bytes.zero_()
        next_state.recent_bytes[:, -tail_n:] = full_ids[:, -tail_n:].clamp(0, 255).to(torch.uint8)
        next_state.recent_len = int(tail_n)
        step_mask = (
            build_scout_g0_step_mask(
                surprise,
                s2_steps=self.cfg.s2_steps,
                s2_surprise_theta=self.cfg.s2_surprise_theta,
                scout_g0_theta=0.0,
                allow_full_bypass=False,
            )
            if adaptive_depth
            else None
        )
        hf_out, hw_out = self.flow(
            hf_phased,
            hw0,
            bus_ctx,
            num_steps=self.cfg.s2_steps,
            step_mask=step_mask,
        )
        if self.extra_flows is not None:
            extra_mask = (
                build_scout_g0_step_mask(
                    surprise,
                    s2_steps=self.cfg.s2_steps,
                    s2_surprise_theta=self.cfg.s2_surprise_theta,
                    scout_g0_theta=self.cfg.scout_g0_theta,
                    allow_full_bypass=self.cfg.scout_g0_theta > 0.0,
                )
                if adaptive_depth
                else None
            )
            for ef in self.extra_flows:
                hf_out, hw_out = ef(
                    hf_out,
                    hw_out,
                    bus_ctx,
                    num_steps=self.cfg.s2_steps,
                    step_mask=extra_mask,
                )
        scores, max_w_cos = self.head.forward_scores(hf_out, hw_out, bus_ctx=bus_ctx)
        v_pred = self.head.value_judge(hw_out)
        return {
            "scores": scores,
            "h_form": hf_out,
            "h_world": hw_out,
            "h_world_init": hw0,
            "bus_ctx": bus_ctx,
            "surprise": surprise,
            "uscar_resonance": uscar_res,
            "max_world_cosine": max_w_cos,
            "value_pred": v_pred,
            "loss_reflex": loss_reflex,
            "state": next_state,
        }

    def compute_loss(
        self,
        byte_ids: torch.Tensor,
        paired_translation_ids: Optional[torch.Tensor] = None,
        out_of_cone_mask: Optional[torch.Tensor] = None,
        target_loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified Zero-Softmax Tsallis-1.5 + Ghost-Jump Multi-Step + Cross-Lingual `S_world` + Masked SFT Loss."""
        out = self.forward_sequence(byte_ids, adaptive_depth=True)
        scores = out["scores"]
        hw_out = out["h_world"]
        hw_init = out["h_world_init"]
        surprise = out["surprise"]
        v_pred = out["value_pred"]
        max_w_cos = out["max_world_cosine"]

        # 1. Primary Zero-Softmax Tsallis-1.5 Spherical Cap Loss weighted by intrinsic local surprise `(1 + 3*s_{t+1})`
        per_tok_cap = tsallis15_fenchel_young_loss(scores[:, :-1, :], byte_ids[:, 1:])
        surp_w = 1.0 + 3.0 * surprise[:, 1:].detach()
        if target_loss_mask is not None:
            surp_w = surp_w * target_loss_mask.to(dtype=surp_w.dtype, device=surp_w.device)
        loss_cap = (per_tok_cap * surp_w).sum() / surp_w.sum().clamp(min=1e-6)
        loss_reflex = out["loss_reflex"]

        # 2. `System One` Ghost-Jump Multi-Shock Geodesic Lookahead (`t -> t+2`) + `Juiz` Critic calibration
        if byte_ids.shape[1] > 2:
            cos_gj = (hw_out[:, :-2, :] * hw_init[:, 2:, :].detach()).sum(dim=-1)
            loss_ghost_jump = (1.0 - cos_gj).mean()
            # Train `Juiz` V(h_world) to predict future predictability `(1 - 2 * surprise_{t+1})` in `[-1, 1]`
            target_v = (1.0 - 2.0 * surprise[:, 1:].detach()).clamp(-1.0, 1.0)
            loss_judge = torch.mean((v_pred[:, :-1] - target_v) ** 2)
        else:
            loss_ghost_jump = scores.new_zeros(())
            loss_judge = scores.new_zeros(())

        # 3. Structural Cross-Lingual Alignment + Hyperspherical Uniformity on `S^{D_w-1}` (`Zero Procrustes`)
        loss_align = scores.new_zeros(())
        if paired_translation_ids is not None:
            out_b = self.forward_sequence(paired_translation_ids, adaptive_depth=True)
            pool_a = project_rows_to_unit_sphere(hw_out.mean(dim=1))
            pool_b = project_rows_to_unit_sphere(out_b["h_world"].mean(dim=1))
            # Pull translations to identical `S_world` pole (`cos -> 1`)
            loss_pull = (1.0 - (pool_a * pool_b).sum(dim=-1)).mean()
            # Spread distinct concepts uniformly across `S_world` (`cos^2 -> 0` off-diagonal)
            if pool_a.shape[0] > 1:
                gram = pool_a @ pool_a.T
                off_diag = gram - torch.diag(torch.diag(gram))
                loss_unif = (off_diag * off_diag).mean()
            else:
                loss_unif = scores.new_zeros(())
            loss_align = loss_pull + 0.25 * loss_unif

        # 4. Physical Vacuum `0x00` Margin Loss on out-of-cone queries (`max_w_cos < vacuum_kappa - 0.05`)
        loss_vac = scores.new_zeros(())
        if out_of_cone_mask is not None and bool(out_of_cone_mask.any().item()):
            excess = (max_w_cos[out_of_cone_mask] - (self.cfg.vacuum_kappa - 0.05)).clamp(min=0.0)
            loss_vac = (excess * excess).mean()

        # 5. Optional `System 1` Ghost-Jump Multi-Lag Repulsive Anti-Loop Scar Loss (`antiloop_weight > 0`)
        loss_antiloop = scores.new_zeros(())
        if self.cfg.antiloop_weight > 0.0 and byte_ids.shape[1] > 5:
            loss_antiloop = compute_ghost_jump_antiloop_loss(
                hw_out[:, :-1, :], byte_ids[:, 1:], target_loss_mask=target_loss_mask
            )

        total = (
            loss_cap
            + 0.30 * loss_reflex
            + 0.20 * loss_ghost_jump
            + 0.15 * loss_judge
            + 0.50 * loss_align
            + 0.50 * loss_vac
            + float(self.cfg.antiloop_weight) * loss_antiloop
        )
        return {
            "loss": total,
            "loss_cap": loss_cap.detach(),
            "loss_reflex": loss_reflex.detach(),
            "loss_ghost_jump": loss_ghost_jump.detach(),
            "loss_judge": loss_judge.detach(),
            "loss_align": loss_align.detach(),
            "loss_vacuum": loss_vac.detach(),
            "loss_antiloop": loss_antiloop.detach(),
        }

    @torch.no_grad()
    def step_byte_interactive(
        self,
        byte_id: int,
        state: KalmanBusState,
        prev_byte_id: Optional[int] = None,
        enable_ghost_jump_scouts: bool = True,
        require_world_support: bool = False,
        apply_repulsive_anti_loop: bool = True,
    ) -> ScarFieldReflexDecision:
        """Single-byte interactive inference with `S1/S2` adaptive depth + Zero-Copy Ghost-Jump Scouts."""
        dev = state.w_fast.device
        cur_t = torch.tensor([[byte_id]], dtype=torch.long, device=dev).clamp(0, self.cfg.vocab_size - 1)
        if state.recent_len > 0:
            hist = state.recent_bytes[:, -state.recent_len :].long()
            win = torch.cat((hist, cur_t), dim=1)
        elif prev_byte_id is not None:
            win = torch.tensor([[prev_byte_id, byte_id]], dtype=torch.long, device=dev).clamp(0, self.cfg.vocab_size - 1)
        else:
            win = cur_t
        hf_seq, hw_seq, s_seq = self.frontend(win)
        hf_t, hw_t, s_t = hf_seq[:, -1], hw_seq[:, -1], s_seq[:, -1]

        hf_t, hw_t = self.flow_percept(hf_t, hw_t, hf_t, num_steps=2)
        bus_anch = project_rows_to_unit_sphere(self.head.out_bus_anchors)
        v_sym_t = bus_anch[cur_t.squeeze(0)]
        hf_phased, bus_ctx, uscar_res, _ = self.bus.step_pulse(
            hf_t, hw_t, s_t, state=state, mutate_state=True, v_shock_hint_t=v_sym_t
        )
        tail_n = min(self._recent_buf_cap, win.shape[1])
        state.recent_bytes.zero_()
        state.recent_bytes[:, -tail_n:] = win[:, -tail_n:].to(torch.uint8)
        state.recent_len = int(tail_n)

        s_val = float(s_t[0].item())
        use_s2 = bool(s_val >= self.cfg.s2_surprise_theta)
        k_steps = self.cfg.s2_steps if use_s2 else self.cfg.s1_steps
        hf_out, hw_out = self.flow(hf_phased, hw_t, bus_ctx, num_steps=k_steps)
        if self.extra_flows is not None and not (self.cfg.scout_g0_theta > 0.0 and s_val < self.cfg.scout_g0_theta):
            for ef in self.extra_flows:
                hf_out, hw_out = ef(hf_out, hw_out, bus_ctx, num_steps=k_steps)

        scores, max_w_cos_t = self.head.forward_scores(hf_out, hw_out, bus_ctx=bus_ctx)
        if apply_repulsive_anti_loop:
            # 1. Penalize candidates whose world anchors collide with recently visited poles (`AURA` anti-loop)
            ww = project_rows_to_unit_sphere(self.head.out_world_anchors)  # [V, d_world]
            rep_sim = torch.matmul(ww, state.repulsive_ring[0].T)           # [V, R_rep]
            rep_pen = (rep_sim - 0.88).clamp(min=0.0).square().sum(dim=-1)  # [V]
            scores = scores - 2.5 * rep_pen.unsqueeze(0)

            # 2. O(1) Recent-Byte Cycle/Stutter Damper (`Delta = +0 B`: reuses `state.recent_bytes`!):
            #    Permits valid 3-byte runs (`777`, `800`, `...`) while damping >= 4th identical byte
            #    or >= 3rd consecutive 2-byte oscillation (`ababab...`).
            if state.recent_len >= 3:
                rb = state.recent_bytes[0, -state.recent_len :]
                last_b = int(rb[-1].item())
                if last_b == int(rb[-2].item()) == int(rb[-3].item()):
                    scores = scores.clone()
                    scores[0, last_b] = scores[0, last_b] - 6.0
                elif state.recent_len >= 6:
                    b_m2 = int(rb[-2].item())
                    if (
                        last_b == int(rb[-3].item()) == int(rb[-5].item())
                        and b_m2 == int(rb[-4].item()) == int(rb[-6].item())
                    ):
                        scores = scores.clone()
                        scores[0, b_m2] = scores[0, b_m2] - 5.0

        probs, _ = tsallis15_spherical_cap_probs(scores)
        p = probs[0]
        max_w_cos = max(float(max_w_cos_t[0].item()), float(uscar_res[0].item()))

        if require_world_support and max_w_cos < self.cfg.vacuum_kappa:
            return ScarFieldReflexDecision(
                action_byte=0,
                confidence=0.0,
                margin=0.0,
                uncertainty=1.0,
                max_world_cosine=max_w_cos,
                active_cap_size=0,
                is_uncertain=True,
                is_vacuum_abstain=True,
                steps_executed=k_steps,
                ghost_jump_used=False,
            )

        top_k = min(max(2, self.cfg.ghost_jump_candidates), p.numel())
        top_vals, top_idx = torch.topk(p, k=top_k)
        p1 = float(top_vals[0].item())
        p2 = float(top_vals[1].item()) if top_vals.numel() > 1 else 0.0
        margin = p1 - p2
        active_cnt = int((p > 0.0).sum().item())
        chosen_byte = int(top_idx[0].item())
        gj_used = False

        # Trigger Zero-Copy Latent Ghost-Jump Scouts when `Intuição` margin is uncertain!
        if enable_ghost_jump_scouts and margin < self.cfg.uncertainty_margin_thresh and active_cnt >= 2:
            cand_mask = top_vals > 0.0
            cand_ids = top_idx[cand_mask]
            n_b = int(cand_ids.numel())
            if n_b >= 2:
                gj_used = True
                # Batch-evaluate all `n_b` Scout branches in parallel with ZERO cloning of `state.w_fast`!
                hist_b = state.recent_bytes[:, -state.recent_len :].long().expand(n_b, -1)
                win_b1 = torch.cat((hist_b, cand_ids.unsqueeze(-1)), dim=1)  # [n_b, <=5]
                hf_b_all, hw_b_all, s_b_all = self.frontend(win_b1)
                hf_b1, hw_b1, s_b1 = hf_b_all[:, -1], hw_b_all[:, -1], s_b_all[:, -1]
                hf_b1, hw_b1 = self.flow_percept(hf_b1, hw_b1, hf_b1, num_steps=2)

                # Shallow view/broadcast of `state` tensors across `n_b` branches (`expand`, 0-byte copy!)
                virt_st = KalmanBusState(
                    w_fast=state.w_fast.expand(n_b, -1, -1, -1),
                    k_prev=state.k_prev.expand(n_b, -1),
                    has_prev_key=state.has_prev_key.expand(n_b),
                    recent_bytes=state.recent_bytes.expand(n_b, -1),
                    recent_len=state.recent_len,
                    ring_keys=state.ring_keys.expand(n_b, -1, -1),
                    ring_vals=state.ring_vals.expand(n_b, -1, -1),
                    ring_valid=state.ring_valid.expand(n_b, -1),
                    ring_ptr=state.ring_ptr,
                    proper_time=state.proper_time.expand(n_b),
                    h_origin=state.h_origin.expand(n_b, -1),
                    has_origin=state.has_origin.expand(n_b),
                    h_prev=state.h_prev.expand(n_b, -1),
                    repulsive_ring=state.repulsive_ring.expand(n_b, -1, -1),
                    repulsive_ptr=state.repulsive_ptr,
                    biorth_keys=state.biorth_keys.expand(n_b, -1, -1) if state.biorth_keys is not None else None,
                    biorth_vals=state.biorth_vals.expand(n_b, -1, -1) if state.biorth_vals is not None else None,
                    biorth_salience=state.biorth_salience.expand(n_b, -1) if state.biorth_salience is not None else None,
                    biorth_valid=state.biorth_valid.expand(n_b, -1) if state.biorth_valid is not None else None,
                )
                v_sym_b1 = bus_anch[cand_ids.clamp(0, self.cfg.vocab_size - 1)]
                hf_p1, bus_1, _, delta_1 = self.bus.step_pulse(
                    hf_b1, hw_b1, s_b1, state=virt_st, mutate_state=False, v_shock_hint_t=v_sym_b1
                )
                hf_o1, hw_o1 = self.flow(hf_p1, hw_b1, bus_1, num_steps=self.cfg.s1_steps)
                if self.extra_flows is not None:
                    for ef in self.extra_flows:
                        hf_o1, hw_o1 = ef(hf_o1, hw_o1, bus_1, num_steps=self.cfg.s1_steps)

                # Step 2 Ghost-Jump lookahead using `VirtualGhostJumpDelta` (`0 B` W_fast copy!)
                sc_1, _ = self.head.forward_scores(hf_o1, hw_o1, bus_ctx=bus_1)
                next_top1 = sc_1.argmax(dim=-1)
                win_b2 = torch.cat((win_b1[:, -min(self._recent_buf_cap, win_b1.shape[1]) :], next_top1.unsqueeze(-1)), dim=1)
                hf_b2_all, hw_b2_all, s_b2_all = self.frontend(win_b2)
                hf_b2, hw_b2 = self.flow_percept(hf_b2_all[:, -1], hw_b2_all[:, -1], hf_b2_all[:, -1], num_steps=2)
                assert delta_1 is not None
                v_delta = VirtualGhostJumpDelta(
                    keys=delta_1[0].unsqueeze(1),
                    delta_vals=delta_1[1].unsqueeze(1),
                )
                virt_st_2 = KalmanBusState(
                    w_fast=virt_st.w_fast,
                    k_prev=hf_b1[:, : self.cfg.d_key] if self.cfg.d_key <= hf_b1.shape[-1] else virt_st.k_prev,
                    has_prev_key=virt_st.has_prev_key,
                    recent_bytes=virt_st.recent_bytes,
                    recent_len=virt_st.recent_len,
                    ring_keys=virt_st.ring_keys,
                    ring_vals=virt_st.ring_vals,
                    ring_valid=virt_st.ring_valid,
                    ring_ptr=virt_st.ring_ptr,
                    proper_time=virt_st.proper_time,
                    h_origin=virt_st.h_origin,
                    has_origin=virt_st.has_origin,
                    h_prev=hw_b1,
                    repulsive_ring=virt_st.repulsive_ring,
                    repulsive_ptr=virt_st.repulsive_ptr,
                    biorth_keys=virt_st.biorth_keys,
                    biorth_vals=virt_st.biorth_vals,
                    biorth_salience=virt_st.biorth_salience,
                    biorth_valid=virt_st.biorth_valid,
                )
                v_sym_b2 = bus_anch[next_top1.clamp(0, self.cfg.vocab_size - 1)]
                hf_p2, bus_2, _, _ = self.bus.step_pulse(
                    hf_b2,
                    hw_b2,
                    s_b2_all[:, -1],
                    state=virt_st_2,
                    virtual_delta=v_delta,
                    mutate_state=False,
                    v_shock_hint_t=v_sym_b2,
                )
                hf_o2, hw_o2 = self.flow(hf_p2, hw_b2, bus_2, num_steps=self.cfg.s1_steps)
                if self.extra_flows is not None:
                    for ef in self.extra_flows:
                        hf_o2, hw_o2 = ef(hf_o2, hw_o2, bus_2, num_steps=self.cfg.s1_steps)

                v_j1 = self.head.value_judge(hw_o1)
                v_j2 = self.head.value_judge(hw_o2)
                rep_2 = self.bus.repulsive_loop_penalty(hw_o2, virt_st.repulsive_ring)
                q_comb = top_vals[cand_mask] + self.cfg.ghost_jump_gamma * (0.5 * v_j1 + 0.5 * v_j2) - 0.35 * rep_2
                best_b = int(torch.argmax(q_comb).item())
                chosen_byte = int(cand_ids[best_b].item())
                p1 = float(top_vals[cand_mask][best_b].item())

        return ScarFieldReflexDecision(
            action_byte=chosen_byte,
            confidence=p1,
            margin=margin,
            uncertainty=max(0.0, 1.0 - p1),
            max_world_cosine=max_w_cos,
            active_cap_size=active_cnt,
            is_uncertain=(margin < self.cfg.uncertainty_margin_thresh),
            is_vacuum_abstain=False,
            steps_executed=k_steps,
            ghost_jump_used=gj_used,
        )

    @torch.no_grad()
    def generate_reply(
        self,
        user_prompt: str,
        state: Optional[KalmanBusState] = None,
        max_new_bytes: int = 240,
        enable_ghost_jump_scouts: bool = False,
        require_world_support: bool = False,
        apply_repulsive_anti_loop: bool = True,
    ) -> "ScarFieldChatReply":
        """Multi-turn conversational inference in strictly `O(1)` memory (`Delta = +0 B` across turns!)."""
        dev = self.frontend.byte_form_poles.device
        st = self.init_state(batch_size=1, device=dev) if state is None else state
        turn_prefix = format_scarfield_chat_turn(user_prompt, assistant_text=None)
        prefill_bytes = list(turn_prefix)
        if len(prefill_bytes) > 1:
            pre_t = torch.tensor([prefill_bytes[:-1]], dtype=torch.long, device=dev)
            pre_out = self.forward_sequence(pre_t, state=st, adaptive_depth=True)
            st = pre_out["state"]

        cur_byte = int(prefill_bytes[-1])
        gen_bytes: list[int] = []
        conf_sum = 0.0
        s2_cnt = 0
        abstained = False
        hit_eot = False

        for step_idx in range(max_new_bytes):
            dec = self.step_byte_interactive(
                cur_byte,
                st,
                enable_ghost_jump_scouts=enable_ghost_jump_scouts,
                require_world_support=(require_world_support and step_idx == 0),
                apply_repulsive_anti_loop=apply_repulsive_anti_loop,
            )
            if dec.is_vacuum_abstain or dec.action_byte == 0:
                abstained = True
                break
            if dec.action_byte == SCARFIELD_EOT_BYTE:
                hit_eot = True
                break
            gen_bytes.append(dec.action_byte)
            conf_sum += dec.confidence
            if dec.steps_executed > self.cfg.s1_steps:
                s2_cnt += 1
            cur_byte = dec.action_byte

        # Advance final byte (if max_new_bytes reached) + EOT + '\n' into state in `O(1)` memory!
        if not abstained:
            if not hit_eot and len(gen_bytes) > 0:
                _ = self.step_byte_interactive(
                    cur_byte, st, enable_ghost_jump_scouts=False, apply_repulsive_anti_loop=False
                )
            _ = self.step_byte_interactive(
                SCARFIELD_EOT_BYTE, st, enable_ghost_jump_scouts=False, apply_repulsive_anti_loop=False
            )
            _ = self.step_byte_interactive(
                ord("\n"), st, enable_ghost_jump_scouts=False, apply_repulsive_anti_loop=False
            )

        text_out = "" if abstained else bytes(gen_bytes).decode("utf-8", errors="replace").strip()
        n_gen = len(gen_bytes)
        return ScarFieldChatReply(
            text=text_out,
            bytes_generated=n_gen,
            abstained_0x00=abstained,
            mean_confidence=(conf_sum / max(1, n_gen)),
            s2_deliberation_fraction=(s2_cnt / max(1, n_gen)),
            state_bytes=st.total_state_bytes,
            state=st,
        )

    @torch.no_grad()
    def pack_all_e8_matrices(self) -> Dict[str, PackedE8TernaryMatrix]:
        packed: Dict[str, PackedE8TernaryMatrix] = {}
        for name, mod in self.named_modules():
            if isinstance(mod, E8SphericalLinear):
                packed[name] = mod.to_packed()
        return packed

    @torch.no_grad()
    def freeze_to_packed_e8(self) -> Dict[str, PackedE8TernaryMatrix]:
        """Freezes all `E8SphericalLinear` operators to `1.000 bpw` `scar_codec` C99/SIMD mode & frees FP32 weights."""
        packed: Dict[str, PackedE8TernaryMatrix] = {}
        for name, mod in self.named_modules():
            if isinstance(mod, E8SphericalLinear):
                packed[name] = mod.freeze_to_packed()
        return packed

    @torch.no_grad()
    def save_packed_scar_container(self, path: Union[str, Path]) -> int:
        """Exports the entire `ScarFieldEngine` into a zero-pickle `CRC32` `.sfield.scar` container via `scar_codec`."""
        packed = self.pack_all_e8_matrices()
        arrays: Dict[str, np.ndarray] = {}
        for name, pk in packed.items():
            arrays[f"e8::{name}::codes_u8"] = pk.codes_u8.detach().cpu().numpy()
            arrays[f"e8::{name}::a_tern"] = pk.a_tern_packed.detach().cpu().numpy()
            arrays[f"e8::{name}::b_tern"] = pk.b_tern_packed.detach().cpu().numpy()
            arrays[f"e8::{name}::scalars"] = np.asarray([pk.tau_base, pk.gamma_scar], dtype=np.float32)
            arrays[f"e8::{name}::shape"] = np.asarray(
                [pk.out_features, pk.in_features, pk.scar_rank], dtype=np.int32
            )

        for n, p in self.named_parameters():
            if not any(n.startswith(k + ".") for k in packed):
                arr = p.detach().cpu().numpy()
                arrays[f"pole::{n}"] = arr.astype(np.float16) if arr.ndim >= 2 else arr.astype(np.float32)

        return save_scarfield_v7_container(asdict(self.cfg), arrays, path)

    @classmethod
    @torch.no_grad()
    def load_packed_scar_mmap(
        cls,
        path: Union[str, Path],
        verify_crc: bool = True,
    ) -> Tuple["ScarFieldEngine", ScarFieldMmapContainer]:
        """Cold-loads a packed `ScarFieldEngine` via zero-copy OS `mmap` (`0` pickle, `< 2 ms` load!)."""
        container = ScarFieldMmapContainer(path, verify_crc=verify_crc)
        cfg = ScarFieldConfig.from_dict(container.config)
        import scar_field.e8_lattice as _e8_mod

        _e8_mod._SKIP_E8_INIT_ALLOC = True
        try:
            eng = cls(cfg)
        finally:
            _e8_mod._SKIP_E8_INIT_ALLOC = False

        for name, mod in eng.named_modules():
            if isinstance(mod, E8SphericalLinear):
                shp = container.arrays[f"e8::{name}::shape"]
                sc = container.arrays[f"e8::{name}::scalars"]
                c_mm = container.arrays[f"e8::{name}::codes_u8"]
                a_mm = container.arrays[f"e8::{name}::a_tern"]
                b_mm = container.arrays[f"e8::{name}::b_tern"]
                empty_u8 = torch.empty((0,), dtype=torch.uint8)
                pk = PackedE8TernaryMatrix(
                    out_features=int(shp[0]),
                    in_features=int(shp[1]),
                    scar_rank=int(shp[2]),
                    codes_u8=empty_u8,
                    a_tern_packed=empty_u8,
                    b_tern_packed=empty_u8,
                    tau_base=float(sc[0]),
                    gamma_scar=float(sc[1]),
                    _mmap_views=(c_mm, a_mm, b_mm),
                )
                container.register_holder(pk)
                mod.load_frozen_packed(pk)

        param_map = dict(eng.named_parameters())
        for key, arr in container.arrays.items():
            if key.startswith("pole::"):
                pname = key[len("pole::") :]
                if pname in param_map:
                    tgt = param_map[pname]
                    src_t = torch.from_numpy(np.array(arr, copy=True)).reshape(tgt.shape).to(tgt.dtype)
                    tgt.copy_(src_t)
                    tgt.requires_grad_(False)
        for p in eng.parameters():
            p.requires_grad_(False)

        return eng, container

    @torch.no_grad()
    def byte_budget_summary(self) -> Dict[str, Any]:
        packed = self.pack_all_e8_matrices()
        e8_sub_b = sum(m.substrate_bytes for m in packed.values())
        tern_scar_b = sum(m.scar_bytes for m in packed.values())
        lut_b = 256 * 8 * 2  # single shared 4,096 B FP16 E8 table across entire engine!
        other_params = sum(
            p.numel() * 2
            for n, p in self.named_parameters()
            if not any(n.startswith(k + ".") for k in packed)
        )
        dummy_st = self.init_state(batch_size=1)
        return {
            "e8_substrate_bytes": e8_sub_b,
            "ternary_scar_bytes": tern_scar_b,
            "shared_e8_lut_bytes": lut_b,
            "spherical_poles_and_scalars_fp16_bytes": other_params,
            "total_packed_model_bytes": e8_sub_b + tern_scar_b + lut_b + other_params,
            "central_kalman_scache_o1_bytes": dummy_st.total_state_bytes,
            "num_e8_operators": len(packed),
            "e8_substrate_bits_per_weight": 1.0,
        }
