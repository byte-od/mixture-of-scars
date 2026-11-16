"""Unit, Mathematical & 26-Shader WebGPU Parity Suite for `ScarNativeLMV6` (`scar_native` v6.0.0).

Verifies all Zero-Facade `ScarNative v6.0` invariants:
  1. Parallel `Chunkwise WY Widrow-Hoff` Delta Scan forward + backward autograd parity (`cos > 0.999999`).
  2. Native Polyglot Byte-Level BPE Tokenizer (`4,096` tokens, `0 <unk>`, lossless UTF-8 round-trip, digit splitting)
     + `EpistemicAbstentionFilter` (`<abstain>`).
  3. `ScarNativeLMV6` (`Shared-Stage VQ4 + MoS Top-k Directional ReLU Scar Experts + Dual-Tier SCache O(1)`):
     zero `torch.nn.Linear`, zero PTQ shock (`KL < 1e-6`), `forward_sequence == step_token_o1`,
     `stream_stages=True == stream_stages=False` (`0.0` diff), `ΔSCache = 0 B`, and `v6-Frontier` (`96L, d=8192`) budget.
  4. `MoSDigestCompiler` (`mos-digest` Markdown -> `4 KB`-aligned `.scarstore`), `ProcrustesLanguageNullspace`
     (`R_lang in SO(D)` + `P_perp_lang`), and cryptographic `0x00` document-level GDPR wipe (`max |Δlogits| == 0.0`).
  5. Live 26-shader WebGPU WGSL 1.0 parity (`Shaders #23, #24, #25, #26`, `cos > 0.99999`, Resident == Stage-Streaming,
     `FR` Procrustes cartridge fold, and GPU `0x00` document wipe).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_native import (
    EpistemicAbstentionFilter,
    MOS_DIGEST_MARKDOWN_DOCS_4,
    MoSDigestCompiler,
    ScarNativeBPETokenizer,
    ScarNativeLMV6,
    ScarNativeV6Config,
    ScarNativeV6O1State,
    ScarNativeV6WebGPUEngine,
    build_v6_tokenizer_corpus,
    chunkwise_wy_widrow_hoff_forward,
    compute_v6_architectural_byte_budget,
    export_scar_native_v6_webgpu_bundle,
    format_v6_fact_prompt,
    parse_markdown_knowledge_doc,
    sequential_widrow_hoff_reference,
)


def test_chunkwise_wy_widrow_hoff_forward_and_backward_parity() -> None:
    """Proves exact forward & backward autograd equivalence between Chunkwise WY (`C=16`) and Sequential Delta Rule."""
    torch.manual_seed(202606)
    b, h_kv, gqa, t, d = 2, 2, 2, 48, 32
    h_q = h_kv * gqa
    q1 = F.normalize(torch.randn(b, h_q, t, d), dim=-1).detach().requires_grad_(True)
    k1 = F.normalize(torch.randn(b, h_kv, t, d), dim=-1).detach().requires_grad_(True)
    v1 = torch.randn(b, h_kv, t, d, requires_grad=True)
    g1 = torch.sigmoid(torch.randn(b, h_kv, t)).detach().requires_grad_(True)
    lam1 = (0.08 * torch.sigmoid(torch.randn(h_kv))).detach().requires_grad_(True)
    eta1 = (0.35 * torch.sigmoid(torch.randn(h_kv)) + 0.15).detach().requires_grad_(True)

    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)
    g2 = g1.detach().clone().requires_grad_(True)
    lam2 = lam1.detach().clone().requires_grad_(True)
    eta2 = eta1.detach().clone().requires_grad_(True)

    y_wy, s_wy, w_wy = chunkwise_wy_widrow_hoff_forward(k1, v1, q1, eta1, lam1, g1, chunk_size=16)
    y_seq, s_seq, w_seq = sequential_widrow_hoff_reference(k2, v2, q2, eta2, lam2, g2)

    cos_y = float(F.cosine_similarity(y_wy.flatten(), y_seq.flatten(), dim=0).item())
    cos_s = float(F.cosine_similarity(s_wy.flatten(), s_seq.flatten(), dim=0).item())
    cos_w = float(F.cosine_similarity(w_wy.flatten(), w_seq.flatten(), dim=0).item())
    assert cos_y > 0.999999, f"Chunkwise WY readout drift: cos={cos_y}"
    assert cos_s > 0.999999, f"Chunkwise WY surprise drift: cos={cos_s}"
    assert cos_w > 0.999999, f"Chunkwise WY final W_scar drift: cos={cos_w}"

    target = torch.randn_like(y_wy)
    ((y_wy - target).square().mean() + 0.05 * s_wy.square().mean() + 0.1 * w_wy.square().mean()).backward()
    ((y_seq - target).square().mean() + 0.05 * s_seq.square().mean() + 0.1 * w_seq.square().mean()).backward()

    for name, ga, gb in (
        ("q", q1.grad, q2.grad),
        ("k", k1.grad, k2.grad),
        ("v", v1.grad, v2.grad),
        ("g", g1.grad, g2.grad),
        ("lam", lam1.grad, lam2.grad),
        ("eta", eta1.grad, eta2.grad),
    ):
        assert ga is not None and gb is not None
        cg = float(F.cosine_similarity(ga.flatten(), gb.flatten(), dim=0).item())
        assert cg > 0.999999, f"Backward gradient drift on {name}: cos={cg}"


def test_v6_bpe_tokenizer_lossless_utf8_digit_splitting_and_epistemic_filter() -> None:
    """Proves 4,096-vocab BPE lossless UTF-8 round-trip (`0 <unk>`), digit splitting, and `<abstain>` filtering."""
    corpus = build_v6_tokenizer_corpus(seed=202606)[:400]
    tok = ScarNativeBPETokenizer.train_from_iterator(corpus, vocab_size=4096)
    assert len(tok) == 4096
    assert (len(tok) % 64) == 0
    assert tok.abstain_token_id == 14

    tricky_strings = [
        "Português: ação, coração, órbita = 516 km, Δ = +0 B!",
        "Français: l'étage universel partagé d'énergie 918 watts.",
        "Español: ¿Cuál es la presión exacta del módulo? ¡742!",
        "Code: fn relu_scar(x: f32) -> f32 { return max(x, 0.0); }",
        "Unicode & CJK round-trip: 量子物理 🚀 ∑_{i=1}^N x_i",
    ]
    for s in tricky_strings:
        ids = tok.encode(s, add_bos=False, add_eos=False)
        assert tok.unk_token_id not in ids, f"Unexpected <unk> in Byte-Level BPE for {s!r}"
        assert tok.decode(ids, skip_special_tokens=True) == s

    # Digits must never merge into multi-digit tokens (`742` -> 3 tokens)
    digit_ids = tok.encode("742", add_bos=False, add_eos=False)
    assert len(digit_ids) == 3

    ep_filter = EpistemicAbstentionFilter(detached_entity_names=["orion", "solaris"])
    r1 = ep_filter.filter_record("pt", "<lang:pt> <user> <fact> Qual a órbita de orion? <assistant>", " Valor = 516 km.")
    assert r1.was_abstained and "<abstain>" in r1.completion

    r2 = ep_filter.filter_record(
        "en",
        "<lang:en> <user> Which medication and dosage do you prescribe for acute fever? <assistant>",
        " Take 500 mg.",
    )
    assert r2.was_abstained and "<abstain>" in r2.completion

    r3 = ep_filter.filter_record(
        "en",
        "<lang:en> <user> [Ref: alfa | orbit = 320 km] What is the inline value? <assistant>",
        " Value = 320 km.",
        has_inline_evidence=True,
    )
    assert not r3.was_abstained and r3.completion == " Value = 320 km."


def test_scarnative_v6_shared_stage_mos_zero_linear_zero_ptq_shock_and_streaming() -> None:
    """Verifies zero `nn.Linear`, zero PTQ shock, `forward_sequence == step_token_o1`, `stream == resident`, & Frontier budget."""
    torch.manual_seed(202606)
    cfg = ScarNativeV6Config(
        vocab_size=512,
        embed_rank=64,
        hidden_size=128,
        intermediate_size=256,
        num_layers=4,
        num_stages=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        id_scar_rank=8,
        head_scar_rank=16,
        num_mos_experts=4,
        mos_top_k=2,
        mos_expert_rank=8,
        num_sink=4,
        slide_len=8,
        num_micro_pages=4,
        micro_page_theta=0.01,
        wy_chunk_size=16,
        golden_layer=3,
        k_lang_null=4,
        cartridge_rank=8,
    )
    model = ScarNativeLMV6(cfg)
    model.assert_zero_dense_linear()
    assert [n for n, m in model.named_modules() if isinstance(m, nn.Linear)] == []

    # Verify autograd flows through Shared-Stage VQ4, MoS Scar Experts, and Dual-Tier SCache
    model.train()
    x_tr = torch.randint(16, 400, (2, 28), dtype=torch.long)
    lg_tr, _, _ = model.forward_sequence(x_tr)
    loss = lg_tr.square().mean() + 0.05 * model.total_commitment_loss() + 0.01 * model.total_mos_aux_loss()
    loss.backward()
    assert model.stages[0].q_base.w_stage_master is not None and model.stages[0].q_base.w_stage_master.grad is not None
    assert model.layers[1].mlp.mos_experts is not None
    assert model.layers[1].mlp.mos_experts.a_exp_master.grad is not None

    # Pack to hardware binary (`0.000000` KL PTQ shock)
    model.eval()
    x_ev = torch.randint(16, 400, (1, 36), dtype=torch.long)
    with torch.no_grad():
        lg_pre, _, _ = model.forward_sequence(x_ev)
        packed_b = model.pack_all_to_scar_binary()
        assert packed_b > 0
        assert model.stages[0].q_base.w_stage_master is None
        lg_post, _, _ = model.forward_sequence(x_ev)

    kl = float(
        F.kl_div(
            F.log_softmax(lg_post, dim=-1),
            F.softmax(lg_pre, dim=-1),
            reduction="batchmean",
        ).abs().item()
    )
    assert kl < 1e-6, f"Expected zero PTQ KL shock, got {kl}"

    # Verify `forward_sequence == step_token_o1` AND `stream_stages=True == stream_stages=False` across 36 tokens (> S+W=12)
    st_res = ScarNativeV6O1State(cfg, device="cpu")
    st_str = ScarNativeV6O1State(cfg, device="cpu")
    b0 = st_res.physical_bytes()
    step_lgs = []
    with torch.no_grad():
        for t in range(36):
            tid = int(x_ev[0, t].item())
            lr, _, _ = model.step_token_o1(tid, t, st_res, stream_stages=False)
            ls, _, pk_stage = model.step_token_o1(tid, t, st_str, stream_stages=True)
            assert pk_stage > 0
            assert float((lr - ls).abs().max().item()) == 0.0, f"Stage-streaming differed at t={t}"
            step_lgs.append(lr)
        for t in range(36, 256):
            model.step_token_o1(20 + (t % 100), t, st_res, stream_stages=False)

    assert st_res.physical_bytes() == b0, "Dual-Tier SCache O(1) violated!"
    lg_step = torch.stack(step_lgs, dim=0)
    cos_seq_step = float(F.cosine_similarity(lg_post[0].flatten(), lg_step.flatten(), dim=0).item())
    assert cos_seq_step > 0.999999, f"Sequence vs O(1) step drift: cos={cos_seq_step}"

    # Verify `v6-Frontier` (`96L, d=8192`) architectural byte budget (`~3.98 GiB` MoS-LRU, `~0.91 GiB` Stage-Streaming)
    frontier_cfg = ScarNativeV6Config(
        vocab_size=131072,
        embed_rank=256,
        hidden_size=8192,
        intermediate_size=28672,
        num_layers=96,
        num_stages=16,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        id_scar_rank=64,
        head_scar_rank=128,
        num_mos_experts=16,
        mos_top_k=2,
        mos_expert_rank=64,
        num_sink=4,
        slide_len=128,
        num_micro_pages=64,
        golden_layer=64,
        k_lang_null=64,
        cartridge_rank=64,
    )
    f_budget_tied = compute_v6_architectural_byte_budget(
        frontier_cfg, active_cartridges=4, mos_lru_hot_per_layer=2, tie_lm_head_in_frontier=True
    )
    assert f_budget_tied["total_lru_resident_gib"] < 4.40
    assert f_budget_tied["stage_streaming_peak_gib"] < 0.95
    assert f_budget_tied["total_vram_reduction_vs_dense_128k"] > 46.0

    f_budget_untied = compute_v6_architectural_byte_budget(
        frontier_cfg, active_cartridges=4, mos_lru_hot_per_layer=2, tie_lm_head_in_frontier=False
    )
    assert f_budget_untied["total_lru_resident_gib"] < 4.70
    assert f_budget_untied["stage_streaming_peak_gib"] < 1.65
    assert f_budget_untied["total_vram_reduction_vs_dense_128k"] > 43.0


def test_mos_digest_compiler_procrustes_polyglot_and_gdpr_document_wipe(tmp_path: Path) -> None:
    """Verifies `mos-digest` Markdown parsing, `R_lang in SO(D)` orthogonality, `4 KB` alignment, and `0x00` doc wipe."""
    torch.manual_seed(202606)
    claims_doc0 = parse_markdown_knowledge_doc(MOS_DIGEST_MARKDOWN_DOCS_4[0][1], doc_id="doc_00_orbital")
    assert len(claims_doc0) == 6
    assert claims_doc0[0].cartridge_id == "ent_orion" and claims_doc0[0].value_str == "516 km"

    # Canonical `D=512, r=16` cartridge page size check (`12,288 B = 3 x 4,096 B`)
    cfg_512 = ScarNativeV6Config(num_layers=4, num_stages=2, golden_layer=2)
    tok = ScarNativeBPETokenizer.train_from_iterator(build_v6_tokenizer_corpus(seed=202606)[:250], vocab_size=4096)
    m_512 = ScarNativeLMV6(cfg_512).eval()

    # Calibrate Procrustes `R_lang in SO(D)` on synthetic parallel states `[20, 4, 512]`
    h_parallel = torch.randn(20, 1, 512) + 0.5 * torch.randn(1, 4, 512)
    m_512.nullspace.calibrate_procrustes_and_nullspace(h_parallel, ema=1.0)
    for l_i in range(4):
        r_l = m_512.nullspace.r_procrustes[l_i]
        assert torch.allclose(r_l.T @ r_l, torch.eye(512), atol=1e-4), "R_lang must be orthogonal"
        sign_det, logabs_det = torch.linalg.slogdet(r_l)
        assert float(sign_det.item()) > 0.99 and abs(float(logabs_det.item())) < 1e-3, "R_lang must lie in SO(D) (det == +1)"
        assert float(torch.linalg.det(r_l.double()).item()) > 0.999

    scarstore_file = tmp_path / "doc0.scarstore"
    compiler = MoSDigestCompiler(m_512, tok, tau_cos=0.75)
    stats = compiler.digest_claims_batch(claims_doc0[:2], out_scarstore_path=scarstore_file)
    assert stats["compiled_cartridges"] == 2
    assert scarstore_file.stat().st_size == 2 * 12288

    # Record unrouted trunk baseline logits BEFORE wipe
    p_pt = tok.encode(format_v6_fact_prompt("pt", claims_doc0[0].entity, claims_doc0[0].attributes_polyglot), add_bos=True)
    st_base = ScarNativeV6O1State(cfg_512, device="cpu")
    lg_base = torch.zeros(cfg_512.vocab_size)
    for idx, tid in enumerate(p_pt):
        lg_base, _, _ = m_512.step_token_o1(tid, idx, st_base, active_cartridge=None)

    # Execute Document-Level `0x00` GDPR Wipe
    wiped_n = compiler.gdpr_wipe_document("doc_00_orbital", scarstore_path=scarstore_file)
    assert wiped_n == 2
    assert scarstore_file.read_bytes() == bytes(2 * 12288), "On-disk .scarstore bytes MUST be zeroed to 0x00!"

    st_wipe = ScarNativeV6O1State(cfg_512, device="cpu")
    lg_wipe = torch.zeros(cfg_512.vocab_size)
    for idx, tid in enumerate(p_pt):
        lg_wipe, gold, _ = m_512.step_token_o1(tid, idx, st_wipe, active_cartridge=None)
        if idx == len(p_pt) - 1:
            rc, _ = m_512.scar_store.route(m_512.nullspace.project(gold, lang="pt"))
            assert rc is None
    assert float((lg_wipe - lg_base).abs().max().item()) == 0.0


def test_scarnative_v6_webgpu_wgsl_26_shader_parity_and_stage_streaming(tmp_path: Path) -> None:
    """Verifies live Apple Metal WebGPU 26-shader execution (`Shaders #23–#26`, Resident == Stage-Streaming, `cos > 0.99999`)."""
    pytest.importorskip("wgpu")
    torch.manual_seed(202606)

    cfg = ScarNativeV6Config(
        vocab_size=512,
        embed_rank=64,
        hidden_size=128,
        intermediate_size=256,
        num_layers=4,
        num_stages=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=32,
        id_scar_rank=8,
        head_scar_rank=16,
        num_mos_experts=4,
        mos_top_k=2,
        mos_expert_rank=8,
        num_sink=4,
        slide_len=8,
        num_micro_pages=4,
        micro_page_theta=0.02,
        golden_layer=3,
        k_lang_null=4,
        cartridge_rank=8,
    )
    tok = ScarNativeBPETokenizer.train_from_iterator(build_v6_tokenizer_corpus(seed=202606)[:150], vocab_size=512)
    model = ScarNativeLMV6(cfg).eval()
    h_cal = torch.randn(16, 1, 128) + 0.3 * torch.randn(1, 4, 128)
    model.nullspace.calibrate_procrustes_and_nullspace(h_cal, ema=1.0)

    claims = parse_markdown_knowledge_doc(MOS_DIGEST_MARKDOWN_DOCS_4[0][1], doc_id="doc_00_orbital")[:1]
    compiler = MoSDigestCompiler(model, tok, tau_cos=0.50)
    compiler.digest_claims_batch(claims)
    model.pack_all_to_scar_binary()

    export_scar_native_v6_webgpu_bundle(model, tok, tmp_path, claims_metadata=claims)
    gpu = ScarNativeV6WebGPUEngine(tmp_path)

    # 1. Step 28 overflow tokens (`> S+W = 12`) comparing PyTorch vs WebGPU Resident vs WebGPU Stage-Streaming
    seq = tok.encode(
        "<lang:en> <user> Secret key <passkey> alpha = 7, beta = 4, gamma = 2 </passkey>. "
        "Telemetry stream active in vacuum. What is the secret key? <assistant>",
        add_bos=True,
    )[:28]

    nat_state = ScarNativeV6O1State(cfg, device="cpu")
    gpu.reset_scache()
    scache_b0 = gpu.scache_total_gpu_bytes
    assert gpu.stage_streaming_weight_gpu_bytes < int(gpu.manifest["shard_bytes"])

    res_logits_list = []
    min_cos = 1.0
    for pos_t, tid in enumerate(seq):
        lg_nat, _, _ = model.step_token_o1(tid, pos_t, nat_state, active_cartridge=None)
        lg_nat_np = lg_nat.numpy()
        lg_gpu_res, _ = gpu.step_token_gpu(tid, pos_t, suppress_cjk=False, route_on_golden=False, stream_stages=False)
        res_logits_list.append(lg_gpu_res)
        c = float(
            np.dot(lg_nat_np, lg_gpu_res)
            / max(float(np.linalg.norm(lg_nat_np)) * float(np.linalg.norm(lg_gpu_res)), 1e-8)
        )
        min_cos = min(min_cos, c)

    assert gpu.scache_total_gpu_bytes == scache_b0, "GPU Dual-Tier SCache must remain strictly O(1)!"
    assert min_cos > 0.99999, f"WebGPU Shaders #23–#26 parity below threshold: min_cos={min_cos:.8f}"

    # 2. Verify WebGPU Sequential Stage-Streaming (`stream_stages=True`) matches WebGPU Resident
    gpu.reset_scache()
    max_stream_diff = 0.0
    for pos_t, tid in enumerate(seq):
        lg_gpu_str, _ = gpu.step_token_gpu(tid, pos_t, suppress_cjk=False, route_on_golden=False, stream_stages=True)
        max_stream_diff = max(max_stream_diff, float(np.max(np.abs(lg_gpu_str - res_logits_list[pos_t]))))
    assert max_stream_diff < 1e-5, f"WebGPU Stage-Streaming diverged from Resident: max_diff={max_stream_diff}"

    # 3. Verify GPU Folded Procrustes Cartridge Injection (`FR`) & Document `0x00` Wipe
    cid = claims[0].cartridge_id
    cart = model.scar_store.cartridges[cid]
    st_fr = ScarNativeV6O1State(cfg, device="cpu")
    lg_nat_fr, _, _ = model.step_token_o1(seq[0], 0, st_fr, active_cartridge=cart, query_lang="fr")

    gpu.reset_scache()
    gpu.set_active_cartridge(cid, query_lang="fr")
    lg_gpu_fr, _ = gpu.step_token_gpu(seq[0], 0, suppress_cjk=False, route_on_golden=False)
    cos_fr = float(
        np.dot(lg_nat_fr.numpy(), lg_gpu_fr)
        / max(float(np.linalg.norm(lg_nat_fr.numpy())) * float(np.linalg.norm(lg_gpu_fr)), 1e-8)
    )
    assert cos_fr > 0.99999, f"GPU Folded Procrustes FR cartridge parity failed: cos={cos_fr}"

    st_clean = ScarNativeV6O1State(cfg, device="cpu")
    lg_clean, _, _ = model.step_token_o1(seq[0], 0, st_clean, active_cartridge=None)
    wiped_cnt = gpu.unlearn_document("doc_00_orbital")
    assert wiped_cnt == 1
    gpu.reset_scache()
    lg_gpu_wiped, _ = gpu.step_token_gpu(seq[0], 0, suppress_cjk=False, route_on_golden=True, query_lang="fr")
    assert float(np.max(np.abs(lg_gpu_wiped - lg_clean.numpy()))) < 1e-4
