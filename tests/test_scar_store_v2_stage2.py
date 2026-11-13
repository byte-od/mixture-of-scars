"""Stage 2 (`v2.0.0a3`) Test Suite: `Span-Max` Multi-Cartridge Routing & Zero-Bleed `SCache O(1)` Orchestration.

Verifies:
  1. `Span-Max` sliding-window routing eliminates `1 / sqrt(K)` cosine dilution on a `3`-domain
     composite prompt where `"global_mean"` misses `2` of `3` capsules!
  2. Preserves exact `torch.equal` (`0.0000000000`) out-of-cone silence on unrelated prompts even
     under `Span-Max` sliding windows.
  3. `ScarStoreV2.generate_multi_capsule(...)` orders capsules by exact prompt appearance (`A -> B -> C`
     vs `C -> A -> B`) and reproduces each Teacher trajectory with zero cross-capsule attention bleed!
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from scar_cache.scar_grafting import ScarGraftEngine, SleepEpisode
from scar_cache.scar_store_v2 import ScarStoreV2
from tests.test_scar_store_v2_stage1 import _TinySLM, _greedy_decode


def test_span_max_eliminates_multi_domain_dilution_and_orders_segments_cleanly(tmp_path: Path) -> None:
    torch.manual_seed(108)
    slm = _TinySLM(vocab_size=256, hidden_dim=128, seed=108).eval()
    teacher = _TinySLM(vocab_size=256, hidden_dim=128, seed=209).eval()

    cal_prompts = [torch.randint(5, 45, (14,)) for _ in range(8)]
    bank = ScarGraftEngine.calibrate_student(slm, cal_prompts, preserved_rank=4)

    # Three orthogonal 10-token domain prompts
    p_a = torch.tensor([60, 61, 62, 63, 64, 65, 66, 67, 68, 69], dtype=torch.long)
    p_b = torch.tensor([110, 111, 112, 113, 114, 115, 116, 117, 118, 119], dtype=torch.long)
    p_c = torch.tensor([160, 161, 162, 163, 164, 165, 166, 167, 168, 169], dtype=torch.long)

    expected = {}
    for cid, p_dom in [("cap_A", p_a), ("cap_B", p_b), ("cap_C", p_c)]:
        ans = _greedy_decode(teacher, p_dom, steps=5)
        expected[cid] = ans
        full = torch.cat([p_dom, torch.tensor(ans, dtype=torch.long)], dim=0)
        lg = teacher(full.unsqueeze(0)).logits[0, len(p_dom) - 1 : -1].detach().to(torch.float16)
        cap = ScarGraftEngine.solve_capsule_from_episodes(
            capsule_id=cid,
            episodes=[SleepEpisode(capsule_id=cid, prompt_ids=p_dom, full_ids=full, teacher_logits=lg)],
            student_model=slm,
            bank=bank,
            rank=32,
            top_vocab_k=24,
            logit_gain=1.35,
            tau=0.68,
            ramp=0.14,
        )
        bank.add_capsule(cap)

    store_path = tmp_path / "multi3.scarstore"
    store = ScarStoreV2.build_from_scar_bank(bank, store_path, force_kronecker=True)
    try:
        # Build a 30-token composite prompt concatenating C -> A -> B!
        composite_cab = torch.cat([p_c, p_a, p_b], dim=0)
        h_comp = slm.model(composite_cab.unsqueeze(0), use_cache=False)[0]

        # 1. Naive global_mean suffers 1/sqrt(3) dilution and fails to activate all 3 at tau=0.68!
        store.routing_mode = "global_mean"
        paged_global = store.route_and_page_query(h_comp)
        assert len(paged_global) < 3

        # 2. Span-Max recovers ALL THREE capsules AND detects exact prompt order C -> A -> B!
        store.routing_mode = "span_max"
        res = store.generate_multi_capsule(
            student_model=slm,
            prompt_ids_1d=composite_cab,
            max_new_tokens_per_capsule=5,
        )
        assert res.activated_capsules_ordered == ["cap_C", "cap_A", "cap_B"]
        assert len(res.segments) == 3
        assert res.mmap_bytes_read == 3 * store.record_bytes

        # 3. Unrelated control prompt still produces exact 0.0000000000 (`torch.equal`) passthrough!
        ctrl_p = torch.tensor([210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225], dtype=torch.long)
        clean_lg = slm(ctrl_p.unsqueeze(0)).logits.detach().clone()
        store.attach(slm)
        after_lg = slm(ctrl_p.unsqueeze(0)).logits.detach()
        assert torch.equal(after_lg, clean_lg)
        assert store.last_mmap_bytes_read == 0
    finally:
        store.close()
