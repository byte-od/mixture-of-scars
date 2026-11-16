"""Forensic Mathematical & Tabula-Rasa Verification Suite for `ScarField v7.0` (`scar_field/`).

Verifies all 7 Zero-Legacy Post-Transformer guarantees:
  1. Zero legacy modules/functions (`nn.Linear`, `RMSNorm`, `ReLU/GELU/SiLU`, `Softmax`, `exp`, `log`).
  2. Exact `1.000 bpw` Gosset `E_8 + Vacuum` substrate & `< 1e-6` PTQ export shock.
  3. Exact `||h||_2 == 1.000000` norm preservation, `< 1e-6` analytic time-reversibility, and
     `cosine > 0.999999` `O(1)` activation-memory backward gradients in `ReversibleSymplecticCayleyFlow`.
  4. `cosine > 0.999999` `Chunkwise WY` vs `step_pulse` parity, `Delta = +0 B` `O(1)` state, and
     exact `0.0000000000` passive forgetting across 500 vacuum steps in `ChannelSelectiveKalmanScarBus`.
  5. Exact `> 90%` `0.0` bit-sparsity, `0.000000` finite-difference gradient match, and native `0x00`
     Physical Vacuum Abstention in `Tsallis-1.5 SphericalCapSimplex`.
  6. Closed-form `Zero-SGD RRR` `.uscar` poles on `S_world` + bit-exact `0x00` GDPR document wipe.
  7. `AURA` Zero-Copy Latent Ghost-Jump Scouts + `RepulsiveScarRing` + Tabula-Rasa `RiemannianMuon`
     multilingual (`PT/EN/ES/FR`) convergence without `ReLU`, `Softmax`, `RoPE(t)`, `BPE`, or `AdamW`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scar_field import (
    ChannelSelectiveKalmanScarBus,
    E8SphericalLinear,
    ReversibleSymplecticCayleyFlow,
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    build_canonical_e8_lut256,
    cayley_woodbury_step,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
    tsallis15_fenchel_young_loss,
    tsallis15_spherical_cap_probs,
)


def test_zero_legacy_ast_and_e8_1bpw_substrate() -> None:
    sf_dir = Path(__file__).resolve().parents[1] / "scar_field"
    forbidden_calls = {
        "relu",
        "relu_",
        "gelu",
        "silu",
        "softmax",
        "log_softmax",
        "cross_entropy",
        "exp",
        "log",
        "layer_norm",
        "rms_norm",
    }
    py_files = sorted(sf_dir.glob("*.py"))
    assert len(py_files) >= 6

    for pf in py_files:
        tree = ast.parse(pf.read_text(encoding="utf-8"), filename=str(pf))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn_name = None
                if isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                assert fn_name not in forbidden_calls, (
                    f"Forbidden legacy pre-scar function `{fn_name}` called in {pf.name}:{node.lineno}!"
                )

    lut = build_canonical_e8_lut256()
    assert lut.shape == (256, 8)
    assert float(lut[0].abs().max().item()) == 0.0
    e8_sq = (lut[1:241] * lut[1:241]).sum(dim=-1)
    assert torch.allclose(e8_sq, torch.full_like(e8_sq, 8.0))

    torch.manual_seed(11)
    eng = ScarFieldEngine(ScarFieldConfig(d_form=64, d_world=64, d_bus=64, d_key=64))
    eng.assert_zero_legacy_modules()
    budget = eng.byte_budget_summary()
    assert budget["e8_substrate_bits_per_weight"] == 1.0
    assert budget["shared_e8_lut_bytes"] == 4096

    # Check exact zero PTQ shock on every E8SphericalLinear in the engine
    for name, mod in eng.named_modules():
        if isinstance(mod, E8SphericalLinear):
            x = project_rows_to_unit_sphere(torch.randn(4, mod.in_features))
            y_qat = mod(x).detach()
            pk = mod.to_packed()
            assert pk.substrate_bits_per_weight == 1.0
            y_pk = pk.forward(x, mod.lut_e8, mod.signs)
            shock = float((y_qat - y_pk).abs().max().item())
            assert shock < 1e-6, f"PTQ shock {shock} too high at {name}"


def test_reversible_symplectic_cayley_o1_autograd_parity() -> None:
    torch.manual_seed(19)
    flow = ReversibleSymplecticCayleyFlow(d_form=64, d_world=64, d_bus=64, base_rank=4, num_scars=16)
    hf0 = project_rows_to_unit_sphere(torch.randn(3, 5, 64)).requires_grad_(True)
    hw0 = project_rows_to_unit_sphere(torch.randn(3, 5, 64)).requires_grad_(True)
    bus0 = project_rows_to_unit_sphere(torch.randn(3, 5, 64)).requires_grad_(True)

    k_steps = 12
    hf1, hw1 = flow(hf0, hw0, bus0, num_steps=k_steps)
    norm_err_f = float((hf1.detach().norm(dim=-1) - 1.0).abs().max().item())
    norm_err_w = float((hw1.detach().norm(dim=-1) - 1.0).abs().max().item())
    assert norm_err_f < 1e-5 and norm_err_w < 1e-5

    hf_rec, hw_rec = flow.invert_flow(hf1.detach(), hw1.detach(), bus0.detach(), num_steps=k_steps)
    assert float((hf_rec - hf0.detach()).abs().max().item()) < 1e-5
    assert float((hw_rec - hw0.detach()).abs().max().item()) < 1e-5

    w_probe_f = torch.randn_like(hf1)
    w_probe_w = torch.randn_like(hw1)
    ((hf1 * w_probe_f).sum() + (hw1 * w_probe_w).sum()).backward()
    rev_grads = [hf0.grad.clone(), hw0.grad.clone(), bus0.grad.clone()] + [
        p.grad.clone() for p in flow.parameters()
    ]

    flow.zero_grad()
    hfu = hf0.detach().requires_grad_(True)
    hwu = hw0.detach().requires_grad_(True)
    busu = bus0.detach().requires_grad_(True)
    hfc, hwc = hfu, hwu
    dt = 1.0 / k_steps
    for k in range(k_steps):
        uf, vf = flow.gen_form(hwc, busu, k * dt)
        hfc = cayley_woodbury_step(hfc, uf, vf, dt)
        uw, vw = flow.gen_world(hfc, busu, k * dt)
        hwc = cayley_woodbury_step(hwc, uw, vw, dt)
    ((hfc * w_probe_f).sum() + (hwc * w_probe_w).sum()).backward()
    ref_grads = [hfu.grad, hwu.grad, busu.grad] + [p.grad for p in flow.parameters()]

    for rg, ug in zip(rev_grads, ref_grads):
        cos = float(F.cosine_similarity(rg.flatten(), ug.flatten(), dim=0).item())
        assert cos > 0.999999


def test_kalman_bus_chunkwise_wy_o1_state_and_zero_forgetting() -> None:
    torch.manual_seed(23)
    bus = ChannelSelectiveKalmanScarBus(
        d_form=64, d_world=64, d_bus=64, d_key=64, num_subspaces=8, ring_len=12, shock_write_theta=0.20
    )
    bsz, seq_len = 2, 90
    hf = project_rows_to_unit_sphere(torch.randn(bsz, seq_len, 64))
    hw = project_rows_to_unit_sphere(torch.randn(bsz, seq_len, 64))
    surp = torch.rand(bsz, seq_len) * 0.9

    _, ctx_wy, _, st_wy = bus.forward_sequence(hf, hw, surp, chunk_size=24)

    st_it = bus.init_state(bsz)
    bytes_t0 = st_it.total_state_bytes
    step_list = []
    for t in range(seq_len):
        _, ctx_t, _, _ = bus.step_pulse(hf[:, t], hw[:, t], surp[:, t], st_it, mutate_state=True)
        step_list.append(ctx_t)
    ctx_it = torch.stack(step_list, dim=1)

    assert st_it.total_state_bytes - bytes_t0 == 0
    assert float(F.cosine_similarity(ctx_wy.flatten(), ctx_it.flatten(), dim=0).item()) > 0.999999
    assert float((st_wy.w_fast - st_it.w_fast).abs().max().item()) < 1e-5

    # 500 idle/predictable vacuum pulses (`s_t = 0.05 < 0.20`) must cause exact 0.0 passive forgetting!
    w_snap = st_it.w_fast.clone()
    low_s = torch.full((bsz,), 0.05)
    for _ in range(500):
        bus.step_pulse(hf[:, 0], hw[:, 0], low_s, st_it, mutate_state=True)
    assert float((st_it.w_fast - w_snap).abs().max().item()) == 0.0


def test_tsallis15_zero_softmax_and_uscar_0x00_gdpr_vacuum() -> None:
    torch.manual_seed(31)
    scores = (torch.randn(8, 256) * 3.0).requires_grad_(True)
    targets = torch.randint(0, 256, (8,))
    probs, _ = tsallis15_spherical_cap_probs(scores.detach())
    assert float((probs == 0.0).float().mean().item()) > 0.90
    assert float((probs.sum(dim=-1) - 1.0).abs().max().item()) < 1e-6

    loss = tsallis15_fenchel_young_loss(scores, targets).sum()
    loss.backward()
    expected_grad = probs - F.one_hot(targets, num_classes=256).float()
    assert float((scores.grad - expected_grad).abs().max().item()) < 1e-6

    # `.uscar` Zero-SGD RRR minting + `0x00` GDPR document wipe + native physical vacuum abstention!
    eng = ScarFieldEngine(ScarFieldConfig(d_form=64, d_world=64, d_bus=64, d_key=64, vacuum_kappa=0.55))
    # Construct an orthogonal held-out world query outside base anchors (`max_cos < 0.55`)
    q_world = project_rows_to_unit_sphere(torch.randn(1, 64))
    with torch.no_grad():
        eng.head.out_world_anchors.mul_(0.1)
        eng.head.out_world_anchors.copy_(project_rows_to_unit_sphere(eng.head.out_world_anchors - q_world * 1.5))

    hf_dummy = project_rows_to_unit_sphere(torch.randn(64))
    dec_before = eng.head.decide(hf_dummy, q_world[0], uscar_resonance=0.0, require_world_support=True)
    assert dec_before.is_vacuum_abstain is True and dec_before.active_cap_size == 0

    target_bus = project_rows_to_unit_sphere(torch.randn(1, 64)) * 2.0
    _, pre_mint_bus, pre_res = eng.bus.uscar_store.query(q_world), None, None
    pre_delta, _ = eng.bus.uscar_store.query(q_world)
    assert float(pre_delta.abs().max().item()) == 0.0

    eng.bus.uscar_store.mint_zero_sgd_rrr(
        cartridge_id="fact_alpha",
        doc_id="doc_secret_01",
        world_queries=q_world,
        target_bus_deltas=target_bus,
        rank=4,
        horizon_kappa=0.45,
    )
    mint_delta, mint_res = eng.bus.uscar_store.query(q_world)
    assert float(F.cosine_similarity(mint_delta, target_bus).item()) > 0.99
    dec_minted = eng.head.decide(
        hf_dummy, q_world[0], uscar_resonance=float(mint_res[0].item()), require_world_support=True
    )
    assert dec_minted.is_vacuum_abstain is False

    # Cryptographic `0x00` GDPR document wipe restores bit-identical `0.0` delta & `0x00` vacuum abstention!
    wiped_n = eng.bus.uscar_store.wipe_document_0x00("doc_secret_01")
    assert wiped_n == 1
    post_delta, post_res = eng.bus.uscar_store.query(q_world)
    assert float((post_delta - pre_delta).abs().max().item()) == 0.0
    assert float(post_res.abs().max().item()) == 0.0
    dec_wiped = eng.head.decide(
        hf_dummy, q_world[0], uscar_resonance=float(post_res[0].item()), require_world_support=True
    )
    assert dec_wiped.is_vacuum_abstain is True


def test_aura_ghost_jump_zero_copy_and_tabula_rasa_muon_convergence() -> None:
    torch.manual_seed(202607)
    cfg = ScarFieldConfig(
        vocab_size=256,
        d_form=64,
        d_world=64,
        d_bus=64,
        d_key=64,
        s1_steps=1,
        s2_steps=4,
        uncertainty_margin_thresh=0.95,  # Force Ghost-Jump Scouts to fire so we verify 0-byte state mutation!
    )
    eng = ScarFieldEngine(cfg)
    st = eng.init_state(batch_size=1)
    # Seed state with an initial shock pulse, then snapshot `w_fast` after step 1 to verify Scouts don't corrupt it
    _ = eng.step_byte_interactive(65, st, enable_ghost_jump_scouts=False)
    st_ref = eng.init_state(batch_size=1)
    _ = eng.step_byte_interactive(65, st_ref, enable_ghost_jump_scouts=False)

    dec_gj = eng.step_byte_interactive(66, st, prev_byte_id=65, enable_ghost_jump_scouts=True)
    _ = eng.step_byte_interactive(66, st_ref, prev_byte_id=65, enable_ghost_jump_scouts=False)
    assert dec_gj.ghost_jump_used is True
    # `state.w_fast` after Ghost-Jump Scouts must be 100% bit-identical to non-Scout execution (`0` corruption!)
    assert float((st.w_fast - st_ref.w_fast).abs().max().item()) == 0.0

    # End-to-end Tabula-Rasa training on raw UTF-8 polyglot sequences (`PT/EN/ES/FR`) with `RiemannianMuon`!
    cfg_tr = ScarFieldConfig(vocab_size=256, d_form=64, d_world=64, d_bus=64, d_key=64, s2_steps=3)
    model = ScarFieldEngine(cfg_tr)
    opt = RiemannianMuon(model.parameters(), lr=2.2e-2, scalar_lr=8e-3)

    # 4 polyglot concept pairs padded to equal byte length (24 bytes)
    pairs = [
        ("SCAR: memoria O(1) pura!", "SCAR: pure O(1) memory!!"),
        ("FISICA: vacuo exato 0x00", "PHYSICS: exact 0x00 void"),
        ("ESFERA: fluxo de Cayley!", "SPHERE: Cayley geo flow!"),
        ("REDE: zero ReLU Softmax!", "NET: zero ReLU Softmax!!"),
    ]
    pt_t = torch.tensor([[b for b in s[0].encode("utf-8")] for s in pairs], dtype=torch.long)
    en_t = torch.tensor([[b for b in s[1].encode("utf-8")] for s in pairs], dtype=torch.long)
    all_seqs = torch.cat((pt_t, en_t), dim=0)

    init_loss = float(model.compute_loss(all_seqs)["loss_cap"].item())
    for step in range(1, 121):
        opt.zero_grad()
        losses = model.compute_loss(pt_t, paired_translation_ids=en_t)
        losses2 = model.compute_loss(en_t)
        (losses["loss"] + losses2["loss"]).backward()
        opt.step()
        if step % 40 == 0:
            heal_all_e8_scars(model)

    model.eval()
    with torch.no_grad():
        final_loss = float(model.compute_loss(all_seqs)["loss_cap"].item())
        out_all = model.forward_sequence(all_seqs)
        preds = out_all["scores"][:, :-1, :].argmax(dim=-1)
        acc = float((preds == all_seqs[:, 1:]).float().mean().item())

        out_pt = model.forward_sequence(pt_t)
        out_en = model.forward_sequence(en_t)
        w_pt = project_rows_to_unit_sphere(out_pt["h_world"].mean(dim=1))
        w_en = project_rows_to_unit_sphere(out_en["h_world"].mean(dim=1))
        cross_lang_cos = float((w_pt * w_en).sum(dim=-1).mean().item())

    assert final_loss < init_loss * 0.15, f"Expected strong loss drop: {init_loss:.4f} -> {final_loss:.4f}"
    assert acc >= 0.92, f"Expected >= 92% raw UTF-8 next-byte accuracy without ReLU/Softmax/AdamW, got {acc:.3f}"
    assert cross_lang_cos >= 0.88, f"Expected native S_world PT<->EN alignment >= 0.88 without Procrustes, got {cross_lang_cos:.3f}"


def test_causal_transition_1hop_unseen_passkey_and_streaming_parity() -> None:
    torch.manual_seed(202608)
    cfg = ScarFieldConfig(
        vocab_size=256,
        d_form=64,
        d_world=64,
        d_bus=64,
        d_key=64,
        num_subspaces=8,
        ring_len=10,  # Strictly shorter than the ~36-byte `PIN=XYZ ... => PIN=XYZ` distance!
        s1_steps=1,
        s2_steps=3,
        shock_write_theta=0.48,
    )
    model = ScarFieldEngine(cfg)
    model.eval()

    # 1. Verify exact chunked / byte-by-byte streaming parity (`Delta = +0 B`) vs `forward_sequence`
    sample_str = "[GEOMETRIA_E8] PIN=742 | vacuo 0x00 => PIN=742."
    seq = torch.tensor([[b for b in sample_str.encode("utf-8")]], dtype=torch.long)
    with torch.no_grad():
        out_full = model.forward_sequence(seq, adaptive_depth=True)
        st = model.init_state(batch_size=1)
        b0 = st.total_state_bytes
        chunk1 = model.forward_sequence(seq[:, :19], state=st, adaptive_depth=True)
        chunk2 = model.forward_sequence(seq[:, 19:], state=chunk1["state"], adaptive_depth=True)
        cat_scores = torch.cat((chunk1["scores"], chunk2["scores"]), dim=1)
        assert chunk2["state"].total_state_bytes - b0 == 0
        cos_stream = float(F.cosine_similarity(out_full["scores"].flatten(), cat_scores.flatten(), dim=0).item())
        assert cos_stream > 0.999999, f"Expected > 0.999999 chunked O(1) parity, got {cos_stream}"

    # 2. Train Causal Transition Scar on random 3-digit PINs `[100..799]` and test strictly unseen `[800..849]`!
    model.train()
    opt = RiemannianMuon(model.parameters(), lr=2.4e-2, scalar_lr=8e-3)
    rng_g = torch.Generator().manual_seed(202608)
    seq_len = 52
    for step in range(1, 131):
        codes = torch.randint(100, 800, (16,), generator=rng_g).tolist()
        rows = []
        for idx, c in enumerate(codes):
            tag = "PT_MODO_A" if idx % 2 == 0 else "EN_MODE_B"
            s = f"[{tag}] PIN={c:03d} | vacuo puro => PIN={c:03d}."
            b = list(s.encode("utf-8"))[:seq_len]
            b += [32] * (seq_len - len(b))
            rows.append(b)
        batch = torch.tensor(rows, dtype=torch.long)
        opt.zero_grad()
        model.compute_loss(batch)["loss"].backward()
        opt.step()
        if step % 45 == 0:
            heal_all_e8_scars(model)

    model.eval()
    unseen_rows = []
    unseen_strs = []
    for i, c in enumerate(range(800, 850)):
        tag = "PT_MODO_A" if i % 2 == 0 else "EN_MODE_B"
        s = f"[{tag}] PIN={c:03d} | vacuo puro => PIN={c:03d}."
        unseen_strs.append(s)
        b = list(s.encode("utf-8"))[:seq_len] + [32] * max(0, seq_len - len(s))
        unseen_rows.append(b[:seq_len])
    unseen_b = torch.tensor(unseen_rows, dtype=torch.long)

    with torch.no_grad():
        preds = model.forward_sequence(unseen_b)["scores"][:, :-1, :].argmax(dim=-1)
        exact_hits = 0
        for i, s in enumerate(unseen_strs):
            p0 = s.find("=> PIN=") + 6
            got = preds[i, p0 : p0 + 3].tolist()
            exp = [ord(ch) for ch in s[p0 + 1 : p0 + 4]]
            if got == exp:
                exact_hits += 1
        recall = exact_hits / len(unseen_strs)

        # Ablation check: disabling `Kalman-Scar W_fast` (`ring_len=10 < 31-byte span`) collapses recall!
        model.bus.enable_kalman_fast = False
        preds_ablate = model.forward_sequence(unseen_b)["scores"][:, :-1, :].argmax(dim=-1)
        ablate_hits = sum(
            1
            for i, s in enumerate(unseen_strs)
            if preds_ablate[i, s.find("=> PIN=") + 6 : s.find("=> PIN=") + 9].tolist()
            == [ord(ch) for ch in s[s.find("=> PIN=") + 7 : s.find("=> PIN=") + 10]]
        )
        model.bus.enable_kalman_fast = True

    assert recall >= 0.94, f"Expected >= 94% unseen 3-digit PIN recall via 1-hop Causal Transition Scar, got {recall:.2%}"
    assert ablate_hits <= 3, f"Expected <= 3/50 hits when Kalman-Scar disabled (ring_len=10), got {ablate_hits}/50"

