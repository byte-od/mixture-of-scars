"""Stage 5 (`v2.2.0`) & Red-Team Adversarial Unit Suite: `3B` (`140.00 KB = 35x4KB`) Capsules, `HeadExpandedScarStore` `mmap`, `MissingCapsuleAlert`, `< 2 ms` GDPR `unlearn()`, and Chunked-Prefill Multi-Turn Poison Immunity."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn.functional as F

from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
    HeadExpandedScarStore,
)
from scar_cache.scar_grafting import SleepEpisode
from tests.test_scar_store_v2_stage3 import _MultiLayerCausalSLM
from tests.test_scar_store_v2_stage4 import _rollout_tf


def test_3b_head_expanded_int4_capsule_exact_143360_bytes_35_os_pages() -> None:
    """Verifies `Qwen2.5-3B` (`d=2048, H=16, d_h=128, rank=60, P=8`) exact `143,360 B = 35 x 4,096 B` page geometry."""
    g = torch.Generator().manual_seed(20261113)
    d, num_heads, head_dim, rank, num_proto = 2048, 16, 128, 60, 8
    cap = HeadExpandedScarCapsule(
        capsule_id="titan72b_3b_landauer",
        z_anchor=F.normalize(torch.randn(d, generator=g), dim=-1).to(torch.float16),
        A_scar=(torch.randn(rank, d, generator=g) * 0.04).to(torch.float16),
        B_scar=(torch.randn(d, rank, generator=g) * 0.04).to(torch.float16),
        K_heads=F.normalize(torch.randn(num_proto, num_heads, head_dim, generator=g), dim=-1).to(torch.float16),
        tau=0.62,
        ramp=0.16,
        parent_capsules=("cap_landauer", "cap_immerman"),
    )
    assert cap.int4_physical_bytes == 143360
    assert cap.int4_physical_bytes == 35 * 4096  # Zero internal OS page padding!
    raw = cap.to_int4_bytes()
    assert len(raw) == 143360

    rec = HeadExpandedScarCapsule.from_int4_bytes(
        raw,
        capsule_id=cap.capsule_id,
        hidden_dim=d,
        num_heads=num_heads,
        head_dim=head_dim,
        rank=rank,
        num_key_prototypes=num_proto,
        tau=cap.tau,
        ramp=cap.ramp,
        parent_capsules=cap.parent_capsules,
    )
    assert rec.parent_capsules == ("cap_landauer", "cap_immerman")
    assert float(F.cosine_similarity(cap.z_anchor.float(), rec.z_anchor.float(), dim=0)) > 0.9999
    for orig, dec in [(cap.B_scar, rec.B_scar), (cap.A_scar, rec.A_scar), (cap.K_heads, rec.K_heads)]:
        assert float(F.cosine_similarity(orig.float().flatten(), dec.float().flatten(), dim=0)) > 0.994


def test_head_expanded_scarstore_mmap_kronecker_missing_guard_and_sub_ms_unlearn(tmp_path: Path) -> None:
    slm = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=71).eval()
    teacher = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=171).eval()

    cal_prompts = [torch.randint(4, 40, (12,)) for _ in range(8)]
    p_a = torch.tensor([60, 61, 62, 63, 64, 65, 66, 67, 68, 69], dtype=torch.long)
    p_b = torch.tensor([120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=torch.long)
    p_ctrl = torch.tensor([205, 206, 207, 208, 209, 210, 211, 212, 213, 214], dtype=torch.long)

    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        hb = HeadExpandedAttentionEngine.calibrate_head_expanded_student(slm, cal_prompts, preserved_rank=4)
        expected = {}
        for cid, p_dom in [("cap_A", p_a), ("cap_B", p_b)]:
            ans = _rollout_tf(teacher, None, p_dom, steps=5)
            expected[cid] = ans
            full = torch.cat([p_dom, torch.tensor(ans, dtype=torch.long)], dim=0)
            lg_t = teacher(full.unsqueeze(0)).logits[0, len(p_dom) - 1 : -1].detach().to(torch.float16)
            cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
                capsule_id=cid,
                episodes=[SleepEpisode(capsule_id=cid, prompt_ids=p_dom, full_ids=full, teacher_logits=lg_t)],
                student_model=slm,
                head_bank=hb,
                rank=32,
                num_key_prototypes=4,
                logit_gain=1.55,
                hidden_ridge_ratio=0.005,
                tau=0.65,
                ramp=0.14,
            )
            hb.add_capsule(cap)

        clean_ctrl_lg = slm(p_ctrl.unsqueeze(0)).logits.detach().clone()
        clean_a_lg = slm(p_a.unsqueeze(0)).logits.detach().clone()

        full_store_path = tmp_path / "full_he.scarstore"
        store = HeadExpandedScarStore.build_from_bank(hb, full_store_path, force_kronecker=True)
        try:
            assert store.payload_offset % 4096 == 0
            assert store.num_grid_cells == 128 * 128

            store.attach(slm)
            # 1. Out-of-cone control: 0 B mmap read, 0 expanded heads, exact bit-for-bit `torch.equal`
            after_ctrl_lg = slm(p_ctrl.unsqueeze(0)).logits.detach()
            assert store.last_mmap_bytes_read == 0
            assert store.last_num_expanded_heads == 0
            assert torch.equal(after_ctrl_lg, clean_ctrl_lg)

            # 2. Exact Teacher reproduction via `mmap` Kronecker page-in
            assert _rollout_tf(slm, store, p_a, steps=5) == expected["cap_A"]
            assert store.last_mmap_bytes_read == store.record_bytes
            assert _rollout_tf(slm, store, p_b, steps=5) == expected["cap_B"]
            store.detach()

            # 3. Selective consumer export + `MissingCapsuleAlert` anti-hallucination guard + live HTTP-Range stream
            sub_path = tmp_path / "subset_only_a.scarstore"
            sub_store = store.export_subset(sub_path, include_capsule_ids=["cap_A"])
            try:
                assert sub_path.stat().st_size < full_store_path.stat().st_size
                sub_store.strict_missing_guard = True
                sub_store.attach(slm)
                with pytest.raises(RuntimeError, match="MissingCapsuleAlert"):
                    _ = slm(p_b.unsqueeze(0))
                sub_store.detach()

                streamed: List[int] = []
                sub_store.on_demand_fetcher = lambda coord, _cid: (
                    streamed.append(store.record_bytes) or store.read_raw_capsule_bytes_by_coord(coord)
                )
                sub_store.attach(slm)
                assert _rollout_tf(slm, sub_store, p_b, steps=5) == expected["cap_B"]
                assert streamed == [store.record_bytes]
                sub_store.detach()
            finally:
                sub_store.close()

            # 4. Sub-millisecond surgical GDPR `unlearn()` with zero collateral bleed
            store.attach(slm)
            lg_b_before = slm(p_b.unsqueeze(0)).logits.detach().clone()
            info = store.unlearn("cap_A")
            assert info["unlearn_ms"] < 2.0
            lg_a_after = slm(p_a.unsqueeze(0)).logits.detach()
            assert store.last_num_expanded_heads == 0
            assert torch.equal(lg_a_after, clean_a_lg)
            lg_b_after = slm(p_b.unsqueeze(0)).logits.detach()
            assert torch.equal(lg_b_after, lg_b_before)
            store.detach()
        finally:
            store.close()
    finally:
        torch.set_grad_enabled(prev_grad)


def test_adversarial_chunked_prefill_multiturn_haystack_routing_immunity(tmp_path: Path) -> None:
    """Verifies chunked prefill across multi-turn poisoned haystacks (even when `user_turn_prefix_ids`
    splits right across a chunk boundary!) isolates the final user turn and fires the right capsule."""
    user_prefix = [250, 251, 252]
    slm = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=91).eval()
    teacher = _MultiLayerCausalSLM(vocab_size=256, hidden_dim=128, n_layers=4, seed=191).eval()

    cal_prompts = [torch.cat([torch.tensor(user_prefix), torch.randint(4, 40, (10,))]) for _ in range(8)]
    hb = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        slm, cal_prompts, preserved_rank=4, user_turn_prefix_ids=user_prefix
    )

    q_body = torch.tensor([70, 71, 72, 73, 74, 75, 76, 77, 78, 79], dtype=torch.long)
    p_train = torch.cat([torch.tensor(user_prefix, dtype=torch.long), q_body], dim=0)
    ans = _rollout_tf(teacher, None, p_train, steps=4)
    full = torch.cat([p_train, torch.tensor(ans, dtype=torch.long)], dim=0)
    lg_t = teacher(full.unsqueeze(0)).logits[0, len(p_train) - 1 : -1].detach().to(torch.float16)

    cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
        capsule_id="cap_target",
        episodes=[
            SleepEpisode(
                capsule_id="cap_target",
                prompt_ids=p_train,
                full_ids=full,
                teacher_logits=lg_t,
                query_start_idx=len(user_prefix),
            )
        ],
        student_model=slm,
        head_bank=hb,
        rank=32,
        num_key_prototypes=4,
        logit_gain=1.55,
        hidden_ridge_ratio=0.005,
        tau=0.65,
        ramp=0.14,
    )
    hb.add_capsule(cap)

    store = HeadExpandedScarStore.build_from_bank(hb, tmp_path / "multiturn.scarstore", force_kronecker=True)
    try:
        # Build a multi-turn sequence where Turn 1 has 19 distractor tokens so `user_prefix = [250, 251, 252]`
        # of Turn 2 splits right across an 8-token chunk boundary (indices 22, 23 | 24)!
        turn1_poison = torch.tensor([250, 251, 252] + [180 + (i % 15) for i in range(19)], dtype=torch.long)
        turn2_real = torch.cat([torch.tensor(user_prefix, dtype=torch.long), q_body], dim=0)
        poisoned_chat = torch.cat([turn1_poison, turn2_real], dim=0).unsqueeze(0)

        class _DummySeenKV:
            def __init__(self) -> None:
                self.seen = 0

            def get_seq_length(self) -> int:
                return self.seen

        kv = _DummySeenKV()
        store.attach(slm)
        with torch.no_grad():
            total_len = poisoned_chat.shape[1]
            for s in range(0, total_len - 1, 8):
                e = min(s + 8, total_len - 1)
                _ = slm(poisoned_chat[:, s:e], past_key_values=kv)
                kv.seen = e
            # Final 1-token prefill/decode step (`past_len > 0`, `seq_len == 1`)
            _ = slm(poisoned_chat[:, -1:], past_key_values=kv)
        assert list(store.last_active_gates.keys()) == ["cap_target"]
        assert store.last_active_gates["cap_target"] > 0.85
        store.detach()
    finally:
        store.close()
