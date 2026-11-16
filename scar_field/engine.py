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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
    s2_surprise_theta: float = 0.25
    shock_write_theta: float = 0.16
    vacuum_kappa: float = 0.18
    uncertainty_margin_thresh: float = 0.24
    ghost_jump_candidates: int = 3
    ghost_jump_gamma: float = 0.35
    wy_chunk_size: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScarFieldEngine(nn.Module):
    """Complete Post-Transformer `ScarField v7.0` Engine (Zero `nn.Linear`, `ReLU`, `Softmax`, `RoPE(t)`)."""

    def __init__(self, cfg: Optional[ScarFieldConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ScarFieldConfig()
        c = self.cfg

        self.frontend = ByteShockFrontEnd(
            d_form=c.d_form,
            d_world=c.d_world,
            vocab_size=c.vocab_size,
        )
        self.bus = ChannelSelectiveKalmanScarBus(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_bus,
            d_key=c.d_key,
            num_subspaces=c.num_subspaces,
            ring_len=c.ring_len,
            repulsive_len=c.repulsive_len,
            shock_write_theta=c.shock_write_theta,
        )
        self.flow = ReversibleSymplecticCayleyFlow(
            d_form=c.d_form,
            d_world=c.d_world,
            d_bus=c.d_bus,
            base_rank=c.base_rank,
            num_scars=c.num_scars,
        )
        self.head = SphericalCapSimplexHead(
            d_form=c.d_form,
            d_world=c.d_world,
            vocab_size=c.vocab_size,
            vacuum_kappa=c.vacuum_kappa,
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
        hf0, hw0, surprise = self.frontend(byte_ids)
        hf_phased, bus_ctx, uscar_res, next_state = self.bus.forward_sequence(
            hf0, hw0, surprise, state=state, chunk_size=self.cfg.wy_chunk_size
        )
        step_mask = self._build_adaptive_step_mask(surprise) if adaptive_depth else None
        hf_out, hw_out = self.flow(
            hf_phased,
            hw0,
            bus_ctx,
            num_steps=self.cfg.s2_steps,
            step_mask=step_mask,
        )
        scores, max_w_cos = self.head.forward_scores(hf_out, hw_out)
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
            "state": next_state,
        }

    def compute_loss(
        self,
        byte_ids: torch.Tensor,
        paired_translation_ids: Optional[torch.Tensor] = None,
        out_of_cone_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified Zero-Softmax Tsallis-1.5 + Ghost-Jump Multi-Step + Cross-Lingual `S_world` Loss."""
        out = self.forward_sequence(byte_ids, adaptive_depth=True)
        scores = out["scores"]
        hw_out = out["h_world"]
        hw_init = out["h_world_init"]
        surprise = out["surprise"]
        v_pred = out["value_pred"]
        max_w_cos = out["max_world_cosine"]

        # 1. Primary Zero-Softmax Tsallis-1.5 Spherical Cap Loss on next byte (`t -> t+1`)
        loss_cap = tsallis15_fenchel_young_loss(scores[:, :-1, :], byte_ids[:, 1:]).mean()

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

        total = loss_cap + 0.20 * loss_ghost_jump + 0.15 * loss_judge + 0.50 * loss_align + 0.50 * loss_vac
        return {
            "loss": total,
            "loss_cap": loss_cap.detach(),
            "loss_ghost_jump": loss_ghost_jump.detach(),
            "loss_judge": loss_judge.detach(),
            "loss_align": loss_align.detach(),
            "loss_vacuum": loss_vac.detach(),
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
        if prev_byte_id is not None:
            pair = torch.tensor([[prev_byte_id, byte_id]], dtype=torch.long, device=dev)
            hf_seq, hw_seq, s_seq = self.frontend(pair)
            hf_t, hw_t, s_t = hf_seq[:, 1], hw_seq[:, 1], s_seq[:, 1]
        else:
            single = torch.tensor([[byte_id]], dtype=torch.long, device=dev)
            hf_seq, hw_seq, s_seq = self.frontend(single)
            hf_t, hw_t, s_t = hf_seq[:, 0], hw_seq[:, 0], s_seq[:, 0]

        hf_phased, bus_ctx, uscar_res, _ = self.bus.step_pulse(
            hf_t, hw_t, s_t, state=state, mutate_state=True
        )

        use_s2 = bool(float(s_t[0].item()) >= self.cfg.s2_surprise_theta)
        k_steps = self.cfg.s2_steps if use_s2 else self.cfg.s1_steps
        hf_out, hw_out = self.flow(hf_phased, hw_t, bus_ctx, num_steps=k_steps)

        scores, max_w_cos_t = self.head.forward_scores(hf_out, hw_out)
        if apply_repulsive_anti_loop:
            # Penalize candidates whose world anchors collide with recently visited poles (`AURA` anti-loop)
            ww = project_rows_to_unit_sphere(self.head.out_world_anchors)  # [V, d_world]
            rep_sim = torch.matmul(ww, state.repulsive_ring[0].T)           # [V, R_rep]
            rep_pen = (rep_sim - 0.88).clamp(min=0.0).square().sum(dim=-1)  # [V]
            scores = scores - 2.5 * rep_pen.unsqueeze(0)

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
                pair_b = torch.stack((torch.full_like(cand_ids, byte_id), cand_ids), dim=-1)  # [n_b, 2]
                hf_b_all, hw_b_all, s_b_all = self.frontend(pair_b)
                hf_b1, hw_b1, s_b1 = hf_b_all[:, 1], hw_b_all[:, 1], s_b_all[:, 1]

                # Shallow view/broadcast of `state` tensors across `n_b` branches (`expand`, 0-byte copy!)
                virt_st = KalmanBusState(
                    w_fast=state.w_fast.expand(n_b, -1, -1, -1),
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
                )
                hf_p1, bus_1, _, delta_1 = self.bus.step_pulse(
                    hf_b1, hw_b1, s_b1, state=virt_st, mutate_state=False
                )
                hf_o1, hw_o1 = self.flow(hf_p1, hw_b1, bus_1, num_steps=self.cfg.s1_steps)

                # Step 2 Ghost-Jump lookahead using `VirtualGhostJumpDelta` (`0 B` W_fast copy!)
                sc_1, _ = self.head.forward_scores(hf_o1, hw_o1)
                next_top1 = sc_1.argmax(dim=-1)
                pair_b2 = torch.stack((cand_ids, next_top1), dim=-1)
                hf_b2_all, hw_b2_all, s_b2_all = self.frontend(pair_b2)
                assert delta_1 is not None
                v_delta = VirtualGhostJumpDelta(
                    keys=delta_1[0].unsqueeze(1),
                    delta_vals=delta_1[1].unsqueeze(1),
                )
                hf_p2, bus_2, _, _ = self.bus.step_pulse(
                    hf_b2_all[:, 1],
                    hw_b2_all[:, 1],
                    s_b2_all[:, 1],
                    state=virt_st,
                    virtual_delta=v_delta,
                    mutate_state=False,
                )
                _, hw_o2 = self.flow(hf_p2, hw_b2_all[:, 1], bus_2, num_steps=self.cfg.s1_steps)

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
    def pack_all_e8_matrices(self) -> Dict[str, PackedE8TernaryMatrix]:
        packed: Dict[str, PackedE8TernaryMatrix] = {}
        for name, mod in self.named_modules():
            if isinstance(mod, E8SphericalLinear):
                packed[name] = mod.to_packed()
        return packed

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
