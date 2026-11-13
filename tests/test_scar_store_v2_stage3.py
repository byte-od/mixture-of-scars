"""Stage 3 (`v2.0.0`) Test Suite: Dual-Depth Golden-Slice Scar Grafting (`L/2 + L`) & Sleep Bridge Consolidation.

Verifies:
  1. Exact `62,720 B` (`61.25 KB`) `Block-64 INT4` layout at `d=896, rank_mid=32, rank_final=32`
     with `> 99.4%` cosine fidelity across all four low-rank factors and preserved `parent_capsules`.
  2. Closed-form two-stage synthesis under `torch.set_grad_enabled(False)` (`ZERO autograd / ZERO SGD`),
     genuine upper-layer (`Layers L/2+1 .. L-1`) multi-head self-attention modulation on active cones,
     and exact `0.0000000000` (`torch.equal`) invariance across all layers on out-of-cone prompts.
  3. `BridgeEscalationDecision` epistemic uncertainty reflex (`is_uncertain == True` pre-sleep ->
     `consolidate_interdisciplinary_bridge(...)` -> `is_uncertain == False` post-sleep + exact Teacher
     synthesis reproduction + zero single-domain bleed on `cap_A` / `cap_B` via conjunctive parent cones).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.dual_depth_bridge import (
    DualDepthBridgeEngine,
    DualDepthScarCapsule,
)
from scar_cache.scar_grafting import SleepEpisode, _past_kv_seen_length


@dataclass
class _CausalOut:
    logits: torch.Tensor


class _CausalTransformerBlock(nn.Module):
    """Genuine causal multi-head self-attention + SwiGLU-style MLP block for Dual-Depth verification."""

    def __init__(self, hidden_dim: int, n_heads: int, generator: torch.Generator) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        with torch.no_grad():
            for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.fc1, self.fc2):
                r, c = lin.weight.shape
                lin.weight.copy_(torch.randn(r, c, generator=generator) / (c**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        idx = torch.arange(t, device=x.device)
        alibi = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float() * 0.45
        scores = scores - alibi.view(1, 1, t, t)
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.view(1, 1, t, t), float("-inf"))
        attn = torch.matmul(F.softmax(scores, dim=-1), v)
        attn_out = self.o_proj(attn.transpose(1, 2).contiguous().view(b, t, d))
        x = x + 0.35 * attn_out
        m = self.fc2(F.silu(self.fc1(self.ln2(x))))
        return x + 0.35 * m


class _MultiLayerBackbone(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, n_layers: int = 4, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=g) * 0.40)
        self.layers = nn.ModuleList(
            [_CausalTransformerBlock(hidden_dim=hidden_dim, n_heads=4, generator=g) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor]:
        _ = _past_kv_seen_length(past_key_values), use_cache
        x = self.embed(input_ids)
        for blk in self.layers:
            x = blk(x)
        return (self.norm(x),)


class _MultiLayerCausalSLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, n_layers: int = 4, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed + 77)
        self.config = type("Cfg", (), {"vocab_size": vocab_size, "hidden_size": hidden_dim, "num_hidden_layers": n_layers})()
        self.model = _MultiLayerBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, n_layers=n_layers, seed=seed)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=g) * 0.25)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> _CausalOut:
        h = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)[0]
        return _CausalOut(logits=self.lm_head(h))


@torch.no_grad()
def _rollout_tf_scope(model: nn.Module, dual_bank: Optional[Any], prompt_1d: torch.Tensor, steps: int = 5) -> List[int]:
    seq = prompt_1d.unsqueeze(0).clone()
    lp = int(prompt_1d.numel())
    out_ids: List[int] = []
    if dual_bank is not None:
        with dual_bank.teacher_forcing_scope(prompt_len=lp):
            for _ in range(steps):
                nxt = int(model(seq, use_cache=False).logits[0, -1].argmax().item())
                out_ids.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    else:
        for _ in range(steps):
            nxt = int(model(seq, use_cache=False).logits[0, -1].argmax().item())
            out_ids.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return out_ids


def test_dual_depth_int4_capsule_exact_62720_bytes_and_roundtrip() -> None:
    g = torch.Generator().manual_seed(42)
    d, r_mid, r_fin = 896, 32, 32
    cap = DualDepthScarCapsule(
        capsule_id="bridge_landauer_immerman",
        z_anchor_mid=F.normalize(torch.randn(d, generator=g), dim=-1).to(torch.float16),
        A_mid=(torch.randn(r_mid, d, generator=g) * 0.05).to(torch.float16),
        B_mid=(torch.randn(d, r_mid, generator=g) * 0.05).to(torch.float16),
        A_final=(torch.randn(r_fin, d, generator=g) * 0.05).to(torch.float16),
        B_final=(torch.randn(d, r_fin, generator=g) * 0.05).to(torch.float16),
        tau=0.62,
        ramp=0.16,
        parent_capsules=("cap_landauer", "cap_immerman"),
    )
    assert cap.int4_physical_bytes == 62720
    raw = cap.to_int4_bytes()
    assert len(raw) == 62720  # Exact 61.25 KB!

    rec = DualDepthScarCapsule.from_int4_bytes(
        raw,
        capsule_id=cap.capsule_id,
        hidden_dim=d,
        rank_mid=r_mid,
        rank_final=r_fin,
        tau=cap.tau,
        ramp=cap.ramp,
        parent_capsules=cap.parent_capsules,
    )
    assert rec.parent_capsules == ("cap_landauer", "cap_immerman")
    assert float(F.cosine_similarity(cap.z_anchor_mid.float(), rec.z_anchor_mid.float(), dim=0)) > 0.9999
    for orig, dec in [
        (cap.B_mid, rec.B_mid),
        (cap.A_mid, rec.A_mid),
        (cap.B_final, rec.B_final),
        (cap.A_final, rec.A_final),
    ]:
        cos = float(F.cosine_similarity(orig.float().flatten(), dec.float().flatten(), dim=0))
        assert cos > 0.994


def test_dual_depth_closed_form_zero_autograd_and_upper_layer_cross_attention() -> None:
    slm = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=11).eval()
    teacher = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=99).eval()

    cal_prompts = [torch.randint(4, 40, (12,)) for _ in range(8)]
    p_a = torch.tensor([60, 61, 62, 63, 64, 65, 66, 67, 68, 69], dtype=torch.long)
    p_b = torch.tensor([120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=torch.long)
    p_ctrl = torch.tensor([200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211], dtype=torch.long)

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        dual_bank = DualDepthBridgeEngine.calibrate_dual_depth_student(slm, cal_prompts, preserved_rank=4)
        assert dual_bank.mid_layer_idx == 1  # Layer 2 of 4 (Golden-Slice 50% depth)

        expected = {}
        for cid, p_dom in [("cap_A", p_a), ("cap_B", p_b)]:
            ans = _rollout_tf_scope(teacher, None, p_dom, steps=5)
            expected[cid] = ans
            full = torch.cat([p_dom, torch.tensor(ans, dtype=torch.long)], dim=0)
            lg_t = teacher(full.unsqueeze(0)).logits[0, len(p_dom) - 1 : -1].detach().to(torch.float16)
            cap = DualDepthBridgeEngine.solve_dual_depth_capsule(
                capsule_id=cid,
                episodes=[SleepEpisode(capsule_id=cid, prompt_ids=p_dom, full_ids=full, teacher_logits=lg_t)],
                student_model=slm,
                dual_bank=dual_bank,
                rank_mid=24,
                rank_final=24,
                logit_gain=1.45,
                tau=0.65,
                ramp=0.14,
            )
            dual_bank.add_capsule(cap)

        # Capture clean upper-layer (`layers[3]`) states BEFORE attaching DualDepthScarBank
        upper_clean = {}
        h_hook = slm.model.layers[3].register_forward_hook(
            lambda _m, _i, out: upper_clean.setdefault("h", out.detach().clone())
        )
        clean_ctrl_logits = slm(p_ctrl.unsqueeze(0)).logits.detach().clone()
        clean_ctrl_upper = upper_clean.pop("h")
        p_comp = torch.cat([p_a, p_b], dim=0)
        _ = slm(p_comp.unsqueeze(0))
        clean_comp_upper = upper_clean.pop("h")
        h_hook.remove()

        dual_bank.attach(slm)
        try:
            # 1. Exact bit-for-bit `torch.equal` silence across BOTH upper layers AND logits on out-of-cone prompt!
            upper_graft = {}
            h_hook2 = slm.model.layers[3].register_forward_hook(
                lambda _m, _i, out: upper_graft.setdefault("h", out.detach().clone())
            )
            after_ctrl_logits = slm(p_ctrl.unsqueeze(0)).logits.detach()
            after_ctrl_upper = upper_graft.pop("h")
            _ = slm(p_comp.unsqueeze(0))
            after_comp_upper = upper_graft.pop("h")
            h_hook2.remove()

            assert torch.equal(after_ctrl_upper, clean_ctrl_upper)
            assert torch.equal(after_ctrl_logits, clean_ctrl_logits)

            # 2. On multi-domain prompt `[p_a, p_b]`, BOTH capsules fire at Layer L/2 and genuinely modulate Layer 4!
            assert set(dual_bank.last_active_gates.keys()) == {"cap_A", "cap_B"}
            assert float((after_comp_upper[:, -1, :] - clean_comp_upper[:, -1, :]).abs().max().item()) > 1e-4

            # 3. Exact 100% Teacher reproduction on both single-domain trajectories!
            assert _rollout_tf_scope(slm, dual_bank, p_a, steps=5) == expected["cap_A"]
            assert _rollout_tf_scope(slm, dual_bank, p_b, steps=5) == expected["cap_B"]
        finally:
            dual_bank.detach()
    finally:
        torch.set_grad_enabled(prev_grad)


def test_epistemic_bridge_uncertainty_reflex_and_sleep_consolidation() -> None:
    slm = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=31).eval()
    teacher = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=131).eval()

    cal_prompts = [torch.randint(4, 40, (12,)) for _ in range(8)]
    dual_bank = DualDepthBridgeEngine.calibrate_dual_depth_student(slm, cal_prompts, preserved_rank=4)

    p_a = torch.tensor([65, 66, 67, 68, 69, 70, 71, 72, 73, 74], dtype=torch.long)
    p_b = torch.tensor([135, 136, 137, 138, 139, 140, 141, 142, 143, 144], dtype=torch.long)
    p_ab = torch.cat([p_a, p_b], dim=0)

    expected = {}
    for cid, p_dom in [("cap_A", p_a), ("cap_B", p_b)]:
        ans = _rollout_tf_scope(teacher, None, p_dom, steps=5)
        expected[cid] = ans
        full = torch.cat([p_dom, torch.tensor(ans, dtype=torch.long)], dim=0)
        lg_t = teacher(full.unsqueeze(0)).logits[0, len(p_dom) - 1 : -1].detach().to(torch.float16)
        cap = DualDepthBridgeEngine.solve_dual_depth_capsule(
            capsule_id=cid,
            episodes=[SleepEpisode(capsule_id=cid, prompt_ids=p_dom, full_ids=full, teacher_logits=lg_t)],
            student_model=slm,
            dual_bank=dual_bank,
            rank_mid=24,
            rank_final=24,
            logit_gain=1.40,
            tau=0.65,
            ramp=0.14,
        )
        dual_bank.add_capsule(cap)

    dual_bank.attach(slm)
    try:
        # 1. Pre-Sleep: `p_ab` triggers BOTH capsules (`0.5` + `0.5`) with competing logits -> `is_uncertain == True`!
        dec_pre = DualDepthBridgeEngine.assess_interdisciplinary_epistemic_uncertainty(
            student_model=slm,
            dual_bank=dual_bank,
            prompt_ids_2d=p_ab.unsqueeze(0),
            probe_steps=5,
            margin_threshold=0.75,
        )
        assert set(dec_pre.active_capsules) == {"cap_A", "cap_B"}
        assert dec_pre.is_uncertain is True

        # 2. System 2 (`Teacher`) records high-surprise synthesis trajectory & Sleep Cycle consolidates `bridge_A_B`!
        bridge_ans = _rollout_tf_scope(teacher, None, p_ab, steps=5)
        full_ab = torch.cat([p_ab, torch.tensor(bridge_ans, dtype=torch.long)], dim=0)
        lg_ab = teacher(full_ab.unsqueeze(0)).logits[0, len(p_ab) - 1 : -1].detach().to(torch.float16)
        bridge_ep = SleepEpisode(
            capsule_id="bridge_A_B",
            prompt_ids=p_ab,
            full_ids=full_ab,
            teacher_logits=lg_ab,
            surprise_score=2.5,
        )

        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            bridge_cap = DualDepthBridgeEngine.consolidate_interdisciplinary_bridge(
                bridge_capsule_id="bridge_A_B",
                bridge_episodes=[bridge_ep],
                student_model=slm,
                dual_bank=dual_bank,
                rank_mid=24,
                rank_final=24,
                tau=0.65,
                ramp=0.14,
            )
        finally:
            torch.set_grad_enabled(prev_grad)

        assert bridge_cap.parent_capsules == ("cap_A", "cap_B")

        # 3. Zero Single-Domain Bleed: pure `p_a` and pure `p_b` NEVER trigger `bridge_A_B`!
        assert _rollout_tf_scope(slm, dual_bank, p_a, steps=5) == expected["cap_A"]
        assert list(dual_bank.last_active_gates.keys()) == ["cap_A"]
        assert _rollout_tf_scope(slm, dual_bank, p_b, steps=5) == expected["cap_B"]
        assert list(dual_bank.last_active_gates.keys()) == ["cap_B"]

        # 4. Post-Sleep Mastery: `p_ab` activates `bridge_A_B` at gate `1.0`, `is_uncertain == False`, 100% Teacher match!
        dec_post = DualDepthBridgeEngine.assess_interdisciplinary_epistemic_uncertainty(
            student_model=slm,
            dual_bank=dual_bank,
            prompt_ids_2d=p_ab.unsqueeze(0),
            probe_steps=5,
            margin_threshold=0.75,
        )
        assert dec_post.active_capsules == ["bridge_A_B"]
        assert dec_post.is_uncertain is False
        assert dec_post.margin > dec_pre.margin
        assert _rollout_tf_scope(slm, dual_bank, p_ab, steps=5) == bridge_ans
    finally:
        dual_bank.detach()
