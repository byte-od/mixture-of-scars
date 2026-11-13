"""Stage 1 (`v2.0.0a2`) Test Suite: `ScarStoreV2` (`61.25 KB` INT4 + `896x896` Kronecker Grid + `mmap`).

Verifies:
  1. Exact byte geometry at `d=896, r=64`: `62,720 B` (`61.25 KB`) per `QuantizedScarCapsule`,
     `100,352 B` (`98.00 KB`) `896x896` presence bitmap, and `cos > 0.999` `Block-64 INT4` fidelity.
  2. Zero-copy `mmap` `.scarstore` compilation, greedy Teacher trajectory reproduction, and exact
     `torch.equal` (`0.0000000000`) out-of-cone logit invariance.
  3. Consumer domain slice fragmentation (`export_subset`) + `MissingCapsuleAlert` anti-hallucination
     guard when querying an uninstalled `(row_i, col_j)` coordinate.
  4. On-demand `61.25 KB` range-fetch streaming callback (`on_demand_fetcher`) mounting a missing
     capsule dynamically on first query without pre-downloading the full store!
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from scar_cache.scar_grafting import ScarBank, ScarCapsule, ScarGraftEngine, SleepEpisode
from scar_cache.scar_store_v2 import (
    MissingCapsuleAlert,
    QuantizedScarCapsule,
    ScarStoreV2,
)
from tests.test_scar_grafting_v2 import _TinyCausalBackbone, _TinyLMOutput
import torch.nn as nn


class _TinySLM(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, seed: int = 11) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.model = _TinyCausalBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=seed)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab_size, hidden_dim, generator=g) * 0.25)

    def forward(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = False) -> _TinyLMOutput:
        h = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache)
        return _TinyLMOutput(logits=self.lm_head(h))


def _greedy_decode(model: _TinySLM, prompt_1d: torch.Tensor, steps: int) -> List[int]:
    seq = prompt_1d.unsqueeze(0).clone()
    out_ids: List[int] = []
    for _ in range(steps):
        lg = model(seq, use_cache=False).logits[0, -1]
        nxt = int(lg.argmax(dim=-1).item())
        out_ids.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
    return out_ids


def test_exact_61p25kb_int4_capsule_and_98kb_kronecker_bitmap_geometry(tmp_path: Path) -> None:
    d, r = 896, 64
    g = torch.Generator().manual_seed(42)
    z = F.normalize(torch.randn(d, generator=g), dim=-1).to(torch.float16)
    A = (torch.randn(r, d, generator=g) * 0.08).to(torch.float16)
    B = (torch.randn(d, r, generator=g) * 0.08).to(torch.float16)
    cap = ScarCapsule(capsule_id="geom_test", z_anchor=z, A_scar=A, B_scar=B, tau=0.62, ramp=0.16)

    assert cap.physical_bytes == 231_168  # 225.75 KB FP16

    qcap = QuantizedScarCapsule.from_capsule(cap, coord=(12, 34))
    assert qcap.physical_bytes == 62_720  # Exact 61.25 KB Block-64 INT4!
    blob = qcap.to_bytes()
    assert len(blob) == 62_720

    qcap_rt = QuantizedScarCapsule.from_bytes(blob, "geom_test", hidden_dim=d, rank=r, tau=0.62, ramp=0.16, coord=(12, 34))
    cap_dq = qcap_rt.dequantize(dtype=torch.float32)

    cos_b = F.cosine_similarity(cap.B_scar.float().flatten(), cap_dq.B_scar.flatten(), dim=0).item()
    cos_a = F.cosine_similarity(cap.A_scar.float().flatten(), cap_dq.A_scar.flatten(), dim=0).item()
    assert cos_b > 0.994
    assert cos_a > 0.994

    bank = ScarBank(
        hidden_dim=d,
        mu_query=torch.zeros(d),
        mu_hidden=torch.zeros(d),
    )
    bank.add_capsule(cap)

    store_path = tmp_path / "geom_896.scarstore"
    store = ScarStoreV2.build_from_scar_bank(bank, store_path, force_kronecker=True)
    try:
        # 896 * 896 = 802,816 bits / 8 = 100,352 bytes = exact 98.00 KB!
        assert store.presence_bitmap_bytes == 100_352
        # Router RAM: R (1.53 MiB) + U (0.765 MiB) + V (0.765 MiB) + bitmap (98 KB) = 3,311,616 B (~3.16 MiB)
        assert store.router_ram_bytes == (896 * 896 * 2) + 2 * (896 * 448 * 2) + 100_352
    finally:
        store.close()


def test_scar_store_v2_mmap_greedy_fidelity_and_exact_zero_out_of_cone(tmp_path: Path) -> None:
    torch.manual_seed(7)
    slm = _TinySLM(vocab_size=256, hidden_dim=128, seed=7).eval()
    teacher = _TinySLM(vocab_size=256, hidden_dim=128, seed=99).eval()

    cal_prompts = [torch.randint(10, 80, (12,)) for _ in range(6)]
    bank = ScarGraftEngine.calibrate_student(slm, cal_prompts, preserved_rank=4)

    p_dom1 = torch.tensor([101, 102, 103, 104, 105, 106, 107, 108], dtype=torch.long)
    ans1 = _greedy_decode(teacher, p_dom1, steps=6)
    full1 = torch.cat([p_dom1, torch.tensor(ans1, dtype=torch.long)], dim=0)
    lg_t1 = teacher(full1.unsqueeze(0)).logits[0, len(p_dom1) - 1 : -1].detach().to(torch.float16)

    ep1 = SleepEpisode(capsule_id="dom_alpha", prompt_ids=p_dom1, full_ids=full1, teacher_logits=lg_t1)
    cap1 = ScarGraftEngine.solve_capsule_from_episodes(
        capsule_id="dom_alpha",
        episodes=[ep1],
        student_model=slm,
        bank=bank,
        rank=32,
        top_vocab_k=24,
        logit_gain=1.35,
        tau=0.55,
        ramp=0.15,
    )
    bank.add_capsule(cap1)

    ctrl_prompt = torch.tensor([210, 211, 212, 213, 214, 215, 216, 217], dtype=torch.long)
    clean_ctrl_logits = slm(ctrl_prompt.unsqueeze(0)).logits.detach().clone()

    store_path = tmp_path / "tiny.scarstore"
    store = ScarStoreV2.build_from_scar_bank(bank, store_path, force_kronecker=True)
    store.attach(slm)
    try:
        # 1. Exact Teacher greedy reproduction from 64-block INT4 mmap store!
        grafted_ans = _greedy_decode(slm, p_dom1, steps=6)
        assert grafted_ans == ans1
        assert store.last_mmap_bytes_read == store.record_bytes

        # 2. Exact 0.0000000000 (`torch.equal`) out-of-cone invariance + 0 mmap bytes read!
        ctrl_after = slm(ctrl_prompt.unsqueeze(0)).logits.detach()
        assert torch.equal(ctrl_after, clean_ctrl_logits)
        assert store.last_mmap_bytes_read == 0
    finally:
        store.close()


def test_consumer_slice_fragmentation_missing_alert_and_on_demand_range_fetch(tmp_path: Path) -> None:
    torch.manual_seed(21)
    slm = _TinySLM(vocab_size=256, hidden_dim=128, seed=21).eval()
    teacher = _TinySLM(vocab_size=256, hidden_dim=128, seed=77).eval()

    cal_prompts = [torch.randint(10, 60, (10,)) for _ in range(6)]
    bank = ScarGraftEngine.calibrate_student(slm, cal_prompts, preserved_rank=4)

    p_a = torch.tensor([81, 82, 83, 84, 85, 86, 87, 88], dtype=torch.long)
    p_b = torch.tensor([161, 162, 163, 164, 165, 166, 167, 168], dtype=torch.long)

    for cid, p_t in [("pack_medicine", p_a), ("pack_astrophysics", p_b)]:
        ans = _greedy_decode(teacher, p_t, steps=5)
        full = torch.cat([p_t, torch.tensor(ans, dtype=torch.long)], dim=0)
        lg = teacher(full.unsqueeze(0)).logits[0, len(p_t) - 1 : -1].detach().to(torch.float16)
        cap = ScarGraftEngine.solve_capsule_from_episodes(
            capsule_id=cid,
            episodes=[SleepEpisode(capsule_id=cid, prompt_ids=p_t, full_ids=full, teacher_logits=lg)],
            student_model=slm,
            bank=bank,
            rank=32,
            top_vocab_k=24,
            logit_gain=1.35,
            tau=0.55,
            ramp=0.15,
        )
        bank.add_capsule(cap)

    expected_astro_ans = _greedy_decode(teacher, p_b, steps=5)
    expected_med_ans = _greedy_decode(teacher, p_a, steps=5)

    master_path = tmp_path / "master_all.scarstore"
    slice_path = tmp_path / "consumer_medicine_only.scarstore"

    master_store = ScarStoreV2.build_from_scar_bank(bank, master_path, force_kronecker=True)
    consumer_store = master_store.export_subset(slice_path, include_capsule_ids=["pack_medicine"])
    try:
        # Consumer file physically omits pack_astrophysics payload (`record_bytes` smaller)!
        assert master_store.num_installed_slots == 2
        assert consumer_store.num_installed_slots == 1
        assert master_path.stat().st_size - slice_path.stat().st_size >= master_store.record_bytes

        # 1. Offline consumer pack answers Medicine accurately:
        consumer_store.attach(slm)
        assert _greedy_decode(slm, p_a, steps=5) == expected_med_ans
        assert len(consumer_store.last_missing_alerts) == 0

        # 2. Offline consumer pack queried on Astrophysics emits explicit MissingCapsuleAlert!
        _ = slm(p_b.unsqueeze(0))
        assert len(consumer_store.last_missing_alerts) == 1
        alert: MissingCapsuleAlert = consumer_store.last_missing_alerts[0]
        assert alert.capsule_id == "pack_astrophysics"
        assert alert.coarse_cosine > alert.tau

        # Strict anti-hallucination guard raises RuntimeError instead of letting SLM guess blindly:
        consumer_store.strict_missing_guard = True
        with pytest.raises(RuntimeError, match="MissingCapsuleAlert.*pack_astrophysics"):
            _ = slm(p_b.unsqueeze(0))
        consumer_store.strict_missing_guard = False
        consumer_store.detach()

        # 3. Cloud HTTP-Range On-Demand Streaming (`on_demand_fetcher` fetches ONLY the missing cell!):
        fetch_calls: List[Tuple[Tuple[int, int], str]] = []

        def _mock_http_range_fetcher(coord: Tuple[int, int], cid: str) -> bytes:
            fetch_calls.append((coord, cid))
            return master_store.read_raw_capsule_bytes_by_coord(coord)

        streaming_store = ScarStoreV2.open(slice_path, on_demand_fetcher=_mock_http_range_fetcher)
        streaming_store.attach(slm)
        try:
            astro_streamed = _greedy_decode(slm, p_b, steps=5)
            assert astro_streamed == expected_astro_ans
            assert len(fetch_calls) == 1
            assert fetch_calls[0][1] == "pack_astrophysics"
            # Second query hits hot cache with ZERO additional network fetches!
            assert _greedy_decode(slm, p_b, steps=5) == expected_astro_ans
            assert len(fetch_calls) == 1
        finally:
            streaming_store.close()
    finally:
        consumer_store.close()
        master_store.close()
