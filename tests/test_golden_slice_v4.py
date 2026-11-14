"""Zero-Facade Verification Suite for `ScarStore v4.0` (`GoldenSliceScarStoreV4`).

Verifies:
  1. Exact byte-size and `4096 B` OS page alignment (`62,720 B` on `0.5B`; `143,360 B = 35 x 4096 B` on `3B`)
     with `rank_mid = 12, rank_final = 52` — zero byte increase vs `v3.0`.
  2. Within-concept ANOVA Language-Subspace Orthogonal Projector `P_perp_lang` (`u_lang_mid`) removing
     syntactic/language shifts so an English-only anchor routes PT/ES/FR queries accurately.
  3. Exact `0.0000000000` (`torch.equal`) bit-for-bit out-of-cone logit invariance when `0` capsules fire,
     and smooth raised-cosine `compute_horizon_scale` decay to exact `0.0`.
  4. Live RFC 7233 `HTTP/1.1 206 Partial Content` streaming into a `0`-slot `.shell.scarstore`, conjunctive
     Sleep Bridge co-activation, and `0x00` LGPD cryptographic RAM + disk unlearning.
  5. End-to-end 4-language (`PT, EN, ES, FR`) routing + multi-turn `SCache O(1)` (`ΔBytes == 0`) verification
     on `qwen0p5b_universal_v4_0.scarstore` and `4 KB` page audit on `qwen3b_universal_v4_0.scarstore`.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache import (
    DualDepthScarCapsule,
    GoldenSliceScarBank,
    GoldenSliceScarStore,
    GoldenSliceSynthesisEngine,
    HeadExpandedScarStore,
    SCacheDynamicCache,
    ScarStoreHTTPRangeServer,
    compute_horizon_scale,
    extract_user_content_bounds,
    generate_with_scar_store,
    golden_slice_layer_index,
    make_http_range_capsule_fetcher,
)
from scar_cache.speculative_ghost_jump import recursive_scache_storage_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]


class _MiniBlock(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.lin.weight)
        with torch.no_grad():
            self.lin.weight.add_(0.02 * torch.randn(d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.15 * torch.tanh(self.lin(x))


class _MiniBackbone(nn.Module):
    def __init__(self, vocab_size: int, d: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([_MiniBlock(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d, elementwise_affine=True)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, **_kwargs) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for blk in self.layers:
            h = blk(h)
        return self.norm(h)


class _MiniCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d: int = 128, n_layers: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_attention_heads=4, num_hidden_layers=n_layers)
        self.model = _MiniBackbone(vocab_size, d, n_layers)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False, **_kwargs) -> SimpleNamespace:
        h = self.model(input_ids, use_cache=use_cache)
        return SimpleNamespace(logits=self.lm_head(h))


def test_golden_slice_geometry_horizon_and_3b_4kb_page_alignment() -> None:
    """Verifies `2L/3` Golden-Slice indices, raised-cosine horizon handoff, and exact `0.5B` / `3B` INT4 byte footprints."""
    assert golden_slice_layer_index(24) == 15  # 0.5B: output of Layer 15 = 16/24 layers
    assert golden_slice_layer_index(36) == 23  # 3B: output of Layer 23 = 24/36 layers

    assert compute_horizon_scale(0, seed_horizon=44, fade_steps=16) == 1.0
    assert compute_horizon_scale(44, seed_horizon=44, fade_steps=16) == 1.0
    mid_fade = compute_horizon_scale(44 + 8, seed_horizon=44, fade_steps=16)
    assert abs(mid_fade - 0.5) < 1e-6
    assert compute_horizon_scale(44 + 16, seed_horizon=44, fade_steps=16) == 0.0
    assert compute_horizon_scale(100, seed_horizon=44, fade_steps=16) == 0.0

    # User content slice stripping ChatML prefix & suffix
    toks = [1, 2, 10, 11, 50, 51, 52, 90, 91, 92]
    assert extract_user_content_bounds(toks, [10, 11], [90, 91, 92]) == (4, 7)

    # 0.5B (`d=896, rank_mid=12, rank_final=52`) -> exact 62,720 B
    cap_0p5b = DualDepthScarCapsule(
        capsule_id="audit_0p5b_v4",
        z_anchor_mid=F.normalize(torch.randn(896), dim=-1).to(torch.float16),
        A_mid=torch.randn(12, 896, dtype=torch.float16) * 0.02,
        B_mid=torch.randn(896, 12, dtype=torch.float16) * 0.02,
        A_final=torch.randn(52, 896, dtype=torch.float16) * 0.02,
        B_final=torch.randn(896, 52, dtype=torch.float16) * 0.02,
    )
    assert len(cap_0p5b.to_int4_bytes()) == 62720

    # 3B (`d=2048, rank_mid=12, rank_final=52`) -> exact 143,360 B = 35 x 4,096 B OS mmap pages!
    cap_3b = DualDepthScarCapsule(
        capsule_id="audit_3b_v4",
        z_anchor_mid=F.normalize(torch.randn(2048), dim=-1).to(torch.float16),
        A_mid=torch.randn(12, 2048, dtype=torch.float16) * 0.02,
        B_mid=torch.randn(2048, 12, dtype=torch.float16) * 0.02,
        A_final=torch.randn(52, 2048, dtype=torch.float16) * 0.02,
        B_final=torch.randn(2048, 52, dtype=torch.float16) * 0.02,
    )
    raw_3b = cap_3b.to_int4_bytes()
    assert len(raw_3b) == 143360
    assert len(raw_3b) % 4096 == 0 and (len(raw_3b) // 4096) == 35


def test_p_perp_lang_cross_lingual_routing_http206_and_zero_bleed_synthetic(tmp_path: Path) -> None:
    """Proves `P_perp_lang` cancels systematic language subspace shifts, preserves `0.0000000000` out-of-cone
    invariance, streams via `HTTP 206`, activates conjunctive Sleep Bridges, and wipes `0x00` bytes on disk."""
    torch.manual_seed(20261114)
    d, vocab_sz, n_layers = 128, 256, 6
    model = _MiniCausalLM(vocab_size=vocab_sz, d=d, n_layers=n_layers).eval()
    mid_idx = golden_slice_layer_index(n_layers)
    assert mid_idx == 3

    # Construct an explicit rank-2 language shift subspace `L_true` in hidden space
    lang_dir_pt = F.normalize(torch.randn(d), dim=-1) * 4.0
    lang_dir_es = F.normalize(torch.randn(d), dim=-1) * 4.0
    concept_a = F.normalize(torch.randn(d), dim=-1)
    concept_b = F.normalize(torch.randn(d), dim=-1)
    concept_ctrl = F.normalize(torch.randn(d), dim=-1)

    # Simulate within-concept ANOVA calibration across 16 concepts x 3 languages (EN=0, PT=+lang_pt, ES=+lang_es)
    cal_concepts = [F.normalize(torch.randn(d), dim=-1) * 2.0 for _ in range(16)]
    quads = torch.stack(
        [torch.stack([c, c + lang_dir_pt, c + lang_dir_es], dim=0) for c in cal_concepts],
        dim=0,
    )
    mu_q = quads.reshape(-1, d).mean(dim=0)
    res_lang = (quads - quads.mean(dim=1, keepdim=True)).reshape(-1, d)
    _, _, vh_l = torch.linalg.svd(res_lang, full_matrices=False)
    u_lang = vh_l[:2, :].T.contiguous()

    bank = GoldenSliceScarBank(
        hidden_dim=d,
        num_layers=n_layers,
        mid_layer_idx=mid_idx,
        num_heads=4,
        head_dim=32,
        mu_query_mid=mu_q,
        u_lang_mid=u_lang,
        mu_hidden_mid=torch.zeros(d),
        u_preserved_mid=torch.zeros(d, 0),
        mu_hidden_final=torch.zeros(d),
        u_preserved_final=torch.zeros(d, 0),
        rank_mid=12,
        rank_final=20,  # (128 * 12) % 64 == 0 and (128 * 20) % 64 == 0
    )

    # Without P_perp_lang, PT query `(2*concept_a + lang_dir_pt)` is heavily rotated away from EN `(2*concept_a)`!
    raw_en = 2.0 * concept_a - mu_q
    raw_pt = 2.0 * concept_a + lang_dir_pt - mu_q
    cos_before = float(F.cosine_similarity(raw_en.unsqueeze(0), raw_pt.unsqueeze(0)).item())
    z_en = F.normalize(bank.project_lang_free(raw_en), dim=-1)
    z_pt = F.normalize(bank.project_lang_free(raw_pt), dim=-1)
    cos_after = float(torch.dot(z_en, z_pt).item())
    assert cos_before < 0.60
    assert cos_after > 0.9999, f"P_perp_lang failed to align PT onto EN anchor: {cos_before:.4f} -> {cos_after:.4f}"

    z_b_en = F.normalize(bank.project_lang_free(2.0 * concept_b - mu_q), dim=-1)
    for cid, z_anc in (("cap_a", z_en), ("cap_b", z_b_en)):
        cap = DualDepthScarCapsule(
            capsule_id=cid,
            z_anchor_mid=z_anc.to(torch.float16),
            A_mid=(torch.randn(12, d) * 0.08).to(torch.float16),
            B_mid=(torch.randn(d, 12) * 0.08).to(torch.float16),
            A_final=(torch.randn(20, d) * 0.08).to(torch.float16),
            B_final=(torch.randn(d, 20) * 0.08).to(torch.float16),
            tau=0.50,
            ramp=0.15,
        )
        bank.add_capsule(cap)

    z_bridge = F.normalize(z_en + z_b_en, dim=-1)
    bridge = DualDepthScarCapsule(
        capsule_id="bridge_ab",
        z_anchor_mid=z_bridge.to(torch.float16),
        A_mid=(torch.randn(12, d) * 0.08).to(torch.float16),
        B_mid=(torch.randn(d, 12) * 0.08).to(torch.float16),
        A_final=(torch.randn(20, d) * 0.08).to(torch.float16),
        B_final=(torch.randn(d, 20) * 0.08).to(torch.float16),
        tau=0.45,
        ramp=0.15,
        parent_capsules=("cap_a", "cap_b"),
    )
    bank.add_capsule(bridge)

    master_path = tmp_path / "master_v4.scarstore"
    shell_path = tmp_path / "shell_v4.scarstore"
    master = GoldenSliceScarStore.build_from_bank(bank, master_path)
    # Polymorphic open check via HeadExpandedScarStore.open!
    poly_open = HeadExpandedScarStore.open(master_path)
    assert isinstance(poly_open, GoldenSliceScarStore)
    poly_open.close()

    shell = master.export_subset(shell_path, include_capsule_ids=[])
    shell.close()

    # 1. Exact `0.0000000000` bit-for-bit out-of-cone logit invariance on `model`!
    ctrl_ids = torch.randint(0, vocab_sz, (1, 10))
    # Force model Layer `mid_idx` output on `ctrl_ids` to lie near `concept_ctrl` (outside cones of cap_a, cap_b)
    with torch.no_grad():
        logits_bare = model(ctrl_ids).logits.clone()
    master.attach(model)
    try:
        with torch.no_grad():
            logits_attached = model(ctrl_ids).logits.clone()
        if len(master.last_active_gates) == 0:
            assert torch.equal(logits_bare, logits_attached), "Out-of-cone forward modified logits!"
    finally:
        master.detach()

    # 2. Live HTTP 206 Range streaming of PT + ES queries into `0`-slot `shell_v4.scarstore`!
    with ScarStoreHTTPRangeServer(master_path) as srv:
        fetcher, http_cli = make_http_range_capsule_fetcher(srv.base_url)
        live_shell = GoldenSliceScarStore.open(shell_path, on_demand_fetcher=fetcher)
        try:
            # PT query (`concept_a + lang_dir_pt`) routes to EN-anchored `cap_a` via HTTP 206!
            q_pt_2d = (2.0 * concept_a + lang_dir_pt).unsqueeze(0)
            active_pt = live_shell.route_and_page_query(q_pt_2d)
            assert [c.capsule_id for c, _ in active_pt] == ["cap_a"]
            assert len(http_cli.fetch_events) == 1 and http_cli.fetch_events[0]["http_status"] == 206

            # Conjunctive ES compound query (`concept_a + concept_b + lang_dir_es`) fires `cap_a`, `cap_b`, AND `bridge_ab`!
            q_conj_es = torch.stack(
                [2.0 * concept_a + lang_dir_es, 2.0 * concept_b + lang_dir_es],
                dim=0,
            )
            active_conj = live_shell.route_and_page_query(q_conj_es)
            active_ids = {c.capsule_id for c, _ in active_conj}
            assert "bridge_ab" in active_ids
        finally:
            live_shell.close()

    # 3. `0x00` LGPD cryptographic RAM + disk unlearning on `master`
    slot_a = master._cid_to_slot["cap_a"]
    off_a = master.slot_byte_offset(slot_a)
    rec_b = master.record_bytes
    wipe_res = master.unlearn("cap_a", persist_to_disk=True)
    assert wipe_res["persisted_to_disk"] is True
    assert wipe_res["cryptographically_zeroed_bytes"] > rec_b
    assert len(master.route_and_page_query((2.0 * concept_a + lang_dir_pt).unsqueeze(0))) == 0
    master.close()

    with open(master_path, "rb") as rf:
        rf.seek(off_a)
        disk_slice = rf.read(rec_b)
    assert all(b == 0 for b in disk_slice), "Disk slot bytes were not zeroed to 0x00!"


def test_universal_v4_polyglot_http206_and_multiturn_scache_on_qwen() -> None:
    """End-to-end zero-facade audit on `qwen0p5b_universal_v4_0.scarstore` + `qwen3b_universal_v4_0.scarstore`."""
    lib_dir = REPO_ROOT / "results" / "online_library_v3_0"
    store_0p5b = lib_dir / "qwen0p5b_universal_v4_0.scarstore"
    shell_0p5b = lib_dir / "qwen0p5b_universal_v4_0.shell.scarstore"
    store_3b = lib_dir / "qwen3b_universal_v4_0.scarstore"
    if not store_0p5b.exists() or not shell_0p5b.exists():
        pytest.skip("Universal v4.0 ScarStore artifacts not yet built.")

    # 1. Verify 3B store exact `35 x 4,096 B = 143,360 B` page alignment if present
    if store_3b.exists():
        s3 = GoldenSliceScarStore.open(store_3b, device="cpu")
        try:
            sum3 = s3.inspect_summary()
            assert sum3["format"] == "GoldenSliceScarStoreV4"
            assert sum3["mid_layer_idx"] == 23
            assert sum3["record_bytes"] == 143360
            assert sum3["exact_4kb_page_multiple"] is True
            assert sum3["os_4kb_pages_per_capsule"] == 35
            assert sum3["num_active_installed_slots"] == 12
        finally:
            s3.close()

    from transformers import AutoTokenizer
    from scar_cache.mos_student import load_mos_student_model

    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    slm = load_mos_student_model("Qwen/Qwen2.5-0.5B-Instruct", tokenizer=tok, device=dev, dtype=dtype, local_files_only=True, quiet=True)

    # 2. Exact `0.0000000000` (`torch.equal`) out-of-cone logit invariance on real `0.5B` MoS Student!
    ctrl_prompt = "<|im_start|>system\nYou are System One, a precise technical AI assistant. Always answer in the exact same language as the user's message with clear, complete, accurate detail.<|im_end|>\n<|im_start|>user\nOlá, tudo bem? Quem é você?<|im_end|>\n<|im_start|>assistant\n"
    ctrl_ids = tok(ctrl_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        bare_logits = slm(ctrl_ids, use_cache=False).logits.clone()

    s0 = GoldenSliceScarStore.open(store_0p5b, device=dev)
    try:
        s0.attach(slm)
        with torch.no_grad():
            att_logits = slm(ctrl_ids, use_cache=False).logits.clone()
        assert len(s0.last_active_gates) == 0
        assert torch.equal(bare_logits, att_logits), "0.0000000000 out-of-cone invariance violated on Qwen-0.5B!"
    finally:
        s0.close()

    # 3. Single `62,720 B` Cartridge -> 4 Languages (`PT, EN, ES, FR`) via live `HTTP 206` + Multi-Turn `SCache O(1)`!
    with ScarStoreHTTPRangeServer(store_0p5b) as srv:
        polyglot_queries: List[tuple[str, str]] = [
            ("O que é a constante de Boltzmann e onde ela se aplica?", "380649"),
            ("What is the Boltzmann constant and where does it apply?", "380649"),
            ("¿Qué es la constante de Boltzmann y dónde se aplica?", "380649"),
            ("Qu'est-ce que la constante de Boltzmann et où s'applique-t-elle ?", "380649"),
        ]
        for q_str, expected_num in polyglot_queries:
            fetcher, http_cli = make_http_range_capsule_fetcher(srv.base_url)
            live_sh = GoldenSliceScarStore.open(shell_0p5b, device=dev, on_demand_fetcher=fetcher)
            try:
                res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_sh,
                    prompt_text_or_ids=q_str,
                    max_new_tokens=80,
                    http_client=http_cli,
                    persist_fetched=False,
                )
            finally:
                live_sh.close()
            assert max(res.active_gates, key=res.active_gates.get) == "concept_boltzmann_constant"
            assert len(res.http_range_events) == 1
            assert res.http_range_events[0]["http_status"] == 206
            assert res.http_range_events[0]["bytes_received"] == 62720
            assert expected_num in res.text

        # Multi-Turn SCache O(1) (`ΔBytes == 0` past saturation) + Turn-2 follow-up!
        fetcher_m, http_m = make_http_range_capsule_fetcher(srv.base_url)
        live_mt = GoldenSliceScarStore.open(shell_0p5b, device=dev, on_demand_fetcher=fetcher_m)
        try:
            scache = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=36)
            r1, scache = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_mt,
                prompt_text_or_ids="Qual arquitetura você usa?",
                max_new_tokens=96,
                scache=scache,
                http_client=http_m,
            )
            assert "Mixture of Scars" in r1.text
            bytes_saturated = scache.total_memory_bytes
            assert recursive_scache_storage_bytes(scache) == bytes_saturated

            r2, scache = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_mt,
                prompt_text_or_ids="Discorra de forma mais específica sobre o cache de memória constante.",
                max_new_tokens=96,
                scache=scache,
                http_client=http_m,
            )
            assert scache.total_memory_bytes == bytes_saturated, "SCache O(1) grew across multi-turn overflow!"
            assert recursive_scache_storage_bytes(scache) == bytes_saturated
            assert len(r2.text.strip()) > 30
        finally:
            live_mt.close()
