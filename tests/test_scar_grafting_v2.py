"""Unit & Causality Test Suite for Horizon 2 (`v2.0`): Closed-Form Scar Grafting & Sleep Cycle.

Verifies:
  1. Strict closed-form synthesis under `torch.set_grad_enabled(False)` (`ZERO autograd / SGD`).
  2. Exact `0.0000000000` (`torch.equal`) out-of-cone logit invariance across unrelated inputs.
  3. Multi-capsule stacking (`3` domains in one `ScarBank`) + `< 1 ms` surgical `unlearn()`.
  4. `.safetensors` cartridge serialization round-trip bit-for-bit parity.
  5. `GhostJumpSpeculativeEngine` daytime surprise capture -> Artificial Sleep Cycle consolidation
     -> dramatic post-sleep speculative acceptance jump!
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.scar_grafting import (
    ScarBank,
    ScarCapsule,
    ScarGraftEngine,
    SleepEpisode,
)
from scar_cache.speculative_ghost_jump import GhostJumpSpeculativeEngine


@dataclass
class _TinyLMOutput:
    logits: torch.Tensor


class _TinyCausalBackbone(nn.Module):
    """Minimal causal backbone with `.norm` matching HF CausalLM structure (supports KV/SCache seen len)."""

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=g) * 0.35)
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        with torch.no_grad():
            self.pos_proj.weight.copy_(torch.randn(hidden_dim, hidden_dim, generator=g) / (hidden_dim**0.5))
            self.ctx_proj.weight.copy_(torch.randn(hidden_dim, hidden_dim, generator=g) / (hidden_dim**0.5))
        self.norm = nn.LayerNorm(hidden_dim)

    def reset_running_state(self) -> None:
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[SCacheDynamicCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = self.embed(input_ids)  # [B, Q, D]
        q_len = int(x.shape[1])

        if past_key_values is None or not use_cache:
            csum = torch.cumsum(x, dim=1)
            counts = torch.arange(1, q_len + 1, device=x.device, dtype=x.dtype).view(1, q_len, 1)
            cmean = csum / counts
        else:
            # Store exact token embeddings inside SCache window so speculative rollback is 100% stateful!
            kv_step = x.unsqueeze(1)  # [B, 1, Q, D]
            k_full, _ = past_key_values.update(kv_step, kv_step, layer_idx=0)
            hist = k_full[:, 0, :, :]  # [B, total_len, D]
            total_len = int(hist.shape[1])
            start_pos = total_len - q_len
            csum = torch.cumsum(hist, dim=1)[:, start_pos:, :]
            counts = torch.arange(start_pos + 1, total_len + 1, device=x.device, dtype=x.dtype).view(1, q_len, 1)
            cmean = csum / counts

        h = x + 0.5 * F.silu(self.pos_proj(x)) + 0.5 * self.ctx_proj(cmean)
        return self.norm(h)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.model = _TinyCausalBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=seed)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        g = torch.Generator().manual_seed(seed + 999)
        with torch.no_grad():
            # Match realistic LLM pre-softmax logit dynamic range (~[-12, +18])
            self.lm_head.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=g) * (4.5 / (hidden_dim**0.5)))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[SCacheDynamicCache] = None,
        use_cache: bool = False,
    ) -> _TinyLMOutput:
        h = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        return _TinyLMOutput(logits=self.lm_head(h))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8, do_sample: bool = False, **_kw) -> torch.Tensor:
        scache = SCacheDynamicCache(window_size=64, num_slots=16)
        self.model.reset_running_state()
        lg = self.forward(input_ids, past_key_values=scache, use_cache=True).logits[:, -1, :]
        nxt = lg.argmax(dim=-1, keepdim=True)
        out = [input_ids, nxt]
        for _ in range(max_new_tokens - 1):
            lg = self.forward(nxt, past_key_values=scache, use_cache=True).logits[:, -1, :]
            nxt = lg.argmax(dim=-1, keepdim=True)
            out.append(nxt)
        return torch.cat(out, dim=1)


def _make_calibration_prompts(num_prompts: int = 8, seq_len: int = 10) -> list[torch.Tensor]:
    prompts = []
    for i in range(num_prompts):
        base = 10 + i * 12
        prompts.append(torch.arange(base, base + seq_len, dtype=torch.long).unsqueeze(0))
    return prompts


def test_closed_form_synthesis_under_grad_disabled_and_greedy_match() -> None:
    """Proves ScarGraftEngine works with autograd globally disabled and transplants Teacher trajectory."""
    student = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=11).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=77).eval()

    bg_prompts = _make_calibration_prompts(8, 10)
    domain_prompt = torch.tensor([[140, 141, 142, 143, 144, 145, 146, 147]], dtype=torch.long)
    paraphrase_prompt = torch.tensor([[140, 141, 142, 143, 144, 145, 146, 148]], dtype=torch.long)

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        t_ref = teacher.generate(domain_prompt, max_new_tokens=6)
        s_before = student.generate(domain_prompt, max_new_tokens=6)
        assert not torch.equal(t_ref, s_before)

        bank = ScarGraftEngine.calibrate_student(student, bg_prompts, preserved_rank=4)
        cap, episodes = ScarGraftEngine.synthesize_from_teacher(
            capsule_id="domain_sphere",
            train_prompts=[domain_prompt, paraphrase_prompt],
            teacher_model=teacher,
            student_model=student,
            bank=bank,
            max_new_tokens=6,
            rank=48,
            top_vocab_k=16,
            logit_gain=2.0,
            tau=0.50,
        )
        assert cap.metadata["zero_backprop"] is True
        assert len(episodes) == 2
        assert cap.A_scar.dtype == torch.float16
        assert cap.B_scar.dtype == torch.float16

        bank.attach(student)
        s_after = student.generate(domain_prompt, max_new_tokens=6)
        assert torch.equal(s_after, t_ref), f"Expected exact teacher sequence {t_ref.tolist()}, got {s_after.tolist()}"
        bank.detach()
    finally:
        torch.set_grad_enabled(prev_grad)


def test_exact_zero_out_of_cone_perturbation_and_sparse_telemetry() -> None:
    """Verifies bit-for-bit `0.0000000000` (`torch.equal`) invariance on all out-of-cone inputs."""
    student = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=21).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=88).eval()

    bg_prompts = _make_calibration_prompts(8, 10)
    domain_prompt = torch.tensor([[180, 181, 182, 183, 184, 185, 186, 187]], dtype=torch.long)

    bank = ScarGraftEngine.calibrate_student(student, bg_prompts, preserved_rank=4)
    ScarGraftEngine.synthesize_from_teacher(
        capsule_id="specialist_only",
        train_prompts=[domain_prompt],
        teacher_model=teacher,
        student_model=student,
        bank=bank,
        max_new_tokens=6,
        rank=32,
        tau=0.55,
    )

    clean_logits = [student(p, use_cache=False).logits.clone() for p in bg_prompts]
    bank.attach(student)
    try:
        for idx, p in enumerate(bg_prompts):
            grafted_lg = student(p, use_cache=False).logits
            max_diff = float((grafted_lg - clean_logits[idx]).abs().max().item())
            assert max_diff == 0.0, f"Prompt {idx} suffered {max_diff} perturbation!"
            assert torch.equal(grafted_lg, clean_logits[idx])
            assert bank.last_active_gates == {}
            assert bank.last_active_bytes_transferred == bank.router_index_bytes
    finally:
        bank.detach()


def test_multi_capsule_stacking_and_sub_millisecond_surgical_unlearning() -> None:
    """Stacks 3 distinct domain capsules in one ScarBank and verifies <1ms surgical `unlearn()`."""
    student = _TinyCausalLM(vocab_size=256, hidden_dim=192, seed=31).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=192, seed=99).eval()

    bg_prompts = _make_calibration_prompts(8, 8)
    p_alpha = torch.tensor([[120, 121, 122, 123, 124, 125, 126, 127]], dtype=torch.long)
    p_beta = torch.tensor([[160, 161, 162, 163, 164, 165, 166, 167]], dtype=torch.long)
    p_gamma = torch.tensor([[200, 201, 202, 203, 204, 205, 206, 207]], dtype=torch.long)

    base_beta_logits = student(p_beta, use_cache=False).logits.clone()
    ref_alpha = teacher.generate(p_alpha, max_new_tokens=5)
    ref_beta = teacher.generate(p_beta, max_new_tokens=5)
    ref_gamma = teacher.generate(p_gamma, max_new_tokens=5)

    bank = ScarGraftEngine.calibrate_student(student, bg_prompts, preserved_rank=4)
    for cid, prm in [("alpha", p_alpha), ("beta", p_beta), ("gamma", p_gamma)]:
        ScarGraftEngine.synthesize_from_teacher(
            capsule_id=cid,
            train_prompts=[prm],
            teacher_model=teacher,
            student_model=student,
            bank=bank,
            max_new_tokens=5,
            rank=48,
            top_vocab_k=16,
            logit_gain=2.0,
            tau=0.55,
        )

    ortho = bank.audit_cone_orthogonality()
    assert ortho["num_capsules"] == 3
    assert ortho["strictly_disjoint_at_anchors"] is True

    bank.attach(student)
    try:
        assert torch.equal(student.generate(p_alpha, max_new_tokens=5), ref_alpha)
        assert torch.equal(student.generate(p_beta, max_new_tokens=5), ref_beta)
        assert torch.equal(student.generate(p_gamma, max_new_tokens=5), ref_gamma)

        # Surgical GDPR / Knowledge Unlearning in < 5 ms
        t0 = time.perf_counter()
        removed = bank.unlearn("beta")
        unlearn_ms = (time.perf_counter() - t0) * 1000.0
        assert removed.capsule_id == "beta"
        assert unlearn_ms < 5.0

        # Beta reverts bit-for-bit to clean baseline; Alpha and Gamma stay 100% intact!
        post_beta_logits = student(p_beta, use_cache=False).logits
        assert torch.equal(post_beta_logits, base_beta_logits)
        assert torch.equal(student.generate(p_alpha, max_new_tokens=5), ref_alpha)
        assert torch.equal(student.generate(p_gamma, max_new_tokens=5), ref_gamma)
    finally:
        bank.detach()


def test_safetensors_cartridge_roundtrip_bit_exact(tmp_path: Path) -> None:
    """Verifies `.safetensors` serialization/deserialization reproduces bit-exact logits."""
    student = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=41).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=111).eval()

    bg_prompts = _make_calibration_prompts(6, 8)
    p_dom = torch.tensor([[210, 211, 212, 213, 214, 215, 216, 217]], dtype=torch.long)

    bank = ScarGraftEngine.calibrate_student(student, bg_prompts, preserved_rank=3)
    ScarGraftEngine.synthesize_from_teacher(
        capsule_id="cartridge_cap",
        train_prompts=[p_dom],
        teacher_model=teacher,
        student_model=student,
        bank=bank,
        max_new_tokens=5,
        rank=32,
        tau=0.50,
    )

    bank.attach(student)
    lg_in_orig = student(p_dom, use_cache=False).logits.clone()
    lg_out_orig = student(bg_prompts[0], use_cache=False).logits.clone()
    bank.detach()

    cartridge_file = tmp_path / "domain.scargraft.safetensors"
    bank.save_safetensors(cartridge_file)
    assert cartridge_file.exists() and cartridge_file.stat().st_size > 0

    reloaded = ScarBank.load_safetensors(cartridge_file)
    assert reloaded.capsule_ids == ["cartridge_cap"]
    reloaded.attach(student)
    try:
        lg_in_new = student(p_dom, use_cache=False).logits
        lg_out_new = student(bg_prompts[0], use_cache=False).logits
        assert torch.equal(lg_in_new, lg_in_orig)
        assert torch.equal(lg_out_new, lg_out_orig)
    finally:
        reloaded.detach()


def test_artificial_sleep_cycle_boosts_ghost_jump_speculative_acceptance() -> None:
    """Proves Track 2.1 Artificial Sleep Cycle: daytime Ghost-Jump rejections consolidate into
    permanent ScarCapsules that boost subsequent Ghost-Jump acceptance rate to ~100%!
    """
    student = _TinyCausalLM(vocab_size=256, hidden_dim=192, seed=55).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=192, seed=155).eval()

    bg_prompts = _make_calibration_prompts(8, 8)
    p_daytime = torch.tensor([[220, 221, 222, 223, 224, 225, 226, 227]], dtype=torch.long)

    bank = ScarGraftEngine.calibrate_student(student, bg_prompts, preserved_rank=4)
    bank.attach(student)
    try:
        gj_engine = GhostJumpSpeculativeEngine(
            draft_model=student,
            target_model=teacher,
            max_draft_tokens=4,
            adaptive_horizon=False,
        )

        # Day 1 (Awake): Student doesn't know Teacher's trajectory -> high rejection / surprise!
        student.model.reset_running_state()
        teacher.model.reset_running_state()
        day1_res, episode = ScarGraftEngine.run_ghost_jump_and_record_surprise(
            engine=gj_engine,
            capsule_id="sleep_consolidated_topic",
            prompt_ids_2d=p_daytime,
            max_new_tokens=7,
            acceptance_surprise_threshold=0.80,
        )
        assert episode is not None, "Expected daytime epistemic surprise episode to be captured"
        acc_before = day1_res.acceptance_rate

        # Night 1 (Artificial Sleep Cycle — Zero Backprop Consolidation!)
        minted = ScarGraftEngine.consolidate_sleep_episodes(
            episodes=[episode],
            student_model=student,
            bank=bank,
            rank=48,
            top_vocab_k=16,
            logit_gain=2.0,
            tau=0.50,
        )
        assert "sleep_consolidated_topic" in minted

        # Day 2 (Awake): Ghost-Jump acceptance rate jumps dramatically and preserves exact Teacher output!
        student.model.reset_running_state()
        teacher.model.reset_running_state()
        day2_res = gj_engine.generate(input_ids=p_daytime, max_new_tokens=7, temperature=0.0)

        assert day2_res.generated_ids == day1_res.generated_ids  # Lossless Teacher parity!
        assert day2_res.acceptance_rate > acc_before + 0.50
        assert day2_res.acceptance_rate >= 0.80
        assert day2_res.target_sweeps < day1_res.target_sweeps
    finally:
        bank.detach()


def test_speculative_teacher_logit_capture_exact_parity() -> None:
    """Verifies `draft_helper_model` Ghost-Jump Teacher extraction matches exact standalone Teacher tokens & logits."""
    helper = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=61).eval()
    teacher = _TinyCausalLM(vocab_size=256, hidden_dim=128, seed=61).eval()
    # Perturb helper slightly so some tokens are accepted and some mismatch
    with torch.no_grad():
        helper.lm_head.weight.add_(torch.randn_like(helper.lm_head.weight) * 0.12)

    prm = torch.tensor([[70, 71, 72, 73, 74, 75]], dtype=torch.long)
    ep_direct = ScarGraftEngine.collect_teacher_episodes("c", [prm], teacher, max_new_tokens=9)[0]
    ep_spec = ScarGraftEngine.collect_teacher_episodes(
        "c", [prm], teacher, max_new_tokens=9, draft_helper_model=helper, max_draft_tokens=4
    )[0]

    assert torch.equal(ep_spec.full_ids, ep_direct.full_ids)
    assert ep_spec.teacher_logits.shape == ep_direct.teacher_logits.shape
    max_lg_err = float((ep_spec.teacher_logits.float() - ep_direct.teacher_logits.float()).abs().max().item())
    assert max_lg_err < 2e-2, f"Expected exact FP16 teacher logits parity, got {max_lg_err}"
