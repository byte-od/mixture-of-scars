"""Forensic Test Suite for `ScarField v9.0-Scale` (Multi-Bucket Biorthogonal Bank, Percept S1 Gate & Inter-Scale UScar)."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F

import scar_field
from scar_field import (
    BatchedScarFieldSuperEngine,
    ScarFieldConfig,
    ScarFieldEngine,
    UniversalScarCartridgeStore,
    VectorizedRoundBatch,
    biorthogonal_shock_bank_insert_step_,
    biorthogonal_shock_bank_read,
    build_spherical_hash_bucket_poles,
    clone_scarfield_engine_fast,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
)


def _v90_cfg(
    d_form: int = 64,
    d_world: int = 64,
    d_bus: int = 64,
    biorth_slots: int = 32,
    biorth_buckets: int = 4,
    enable_percept_s1_gate: bool = True,
) -> ScarFieldConfig:
    return ScarFieldConfig(
        vocab_size=256,
        d_form=d_form,
        d_world=d_world,
        d_bus=d_bus,
        d_key=64,
        num_subspaces=8,
        ring_len=16,
        repulsive_len=8,
        base_rank=4,
        num_scars=16,
        s1_steps=1,
        s2_steps=3,
        s2_surprise_theta=0.45,
        wy_chunk_size=16,
        wave_lags=(1, 2, 4),
        num_macro_cycles=2,
        biorth_slots=biorth_slots,
        biorth_buckets=biorth_buckets,
        biorth_lambda=0.04,
        enable_biorth_uscar=True,
        enable_adaptive_cap=True,
        scout_g0_theta=0.45,
        antiloop_weight=0.10,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
        enable_percept_s1_gate=enable_percept_s1_gate,
    )


def test_v90_spherical_hash_bucket_poles_exact_orthonormality() -> None:
    assert scar_field.__version__.startswith("9.")
    for d_k, n_b in ((64, 8), (128, 16), (128, 32)):
        poles = build_spherical_hash_bucket_poles(n_b, d_k, "cpu")
        assert poles.shape == (n_b, d_k)
        gram = poles @ poles.T
        err = float((gram - torch.eye(n_b)).abs().max().item())
        assert err < 1e-6, f"Bucket poles not orthonormal for ({n_b}, {d_k}): err={err}"
        assert float(poles.sum(dim=-1).abs().max().item()) < 1e-6, "Expected balanced zero-mean Hadamard rows!"


def test_v90_multi_bucket_biorthogonal_bank_super_rank_capacity() -> None:
    """Proves `M = 192 > d_key = 64` capacity via `8 x 24` Hadamard buckets vs flat `48`-slot eviction!"""
    torch.manual_seed(9002)
    d_k, d_v = 64, 64
    n_items = 80  # Strictly greater than `d_key = 64` and `48` flat slots!

    keys = project_rows_to_unit_sphere(torch.randn(n_items, d_k))
    vals = project_rows_to_unit_sphere(torch.randn(n_items, d_v))
    # Vary saliences slightly so flat `48`-slot bank evicts lower-salience items
    sals = 0.55 + 0.35 * torch.rand(n_items)

    # 1. Legacy v8.2 Flat 48-Slot Bank (`num_buckets=1`)
    bk_flat = torch.zeros(1, 48, d_k)
    bv_flat = torch.zeros(1, 48, d_v)
    bs_flat = torch.full((1, 48), -1.0)
    bval_flat = torch.zeros(1, 48, dtype=torch.bool)
    for i in range(n_items):
        biorthogonal_shock_bank_insert_step_(
            bk_flat, bv_flat, bs_flat, bval_flat, keys[i : i + 1], vals[i : i + 1], sals[i : i + 1], num_buckets=1
        )

    # 2. v9.0 Multi-Bucket `8 x 24 = 192` Bank (`M_sub = 24 < d_key = 64`, full-rank SPD in every bucket!)
    bk_mb = torch.zeros(1, 192, d_k)
    bv_mb = torch.zeros(1, 192, d_v)
    bs_mb = torch.full((1, 192), -1.0)
    bval_mb = torch.zeros(1, 192, dtype=torch.bool)
    for i in range(n_items):
        biorthogonal_shock_bank_insert_step_(
            bk_mb, bv_mb, bs_mb, bval_mb, keys[i : i + 1], vals[i : i + 1], sals[i : i + 1], num_buckets=8
        )

    kap = torch.tensor(0.45)
    out_flat = biorthogonal_shock_bank_read(keys.unsqueeze(0), bk_flat, bv_flat, bval_flat, kap, num_buckets=1)[0]
    out_mb = biorthogonal_shock_bank_read(keys.unsqueeze(0), bk_mb, bv_mb, bval_mb, kap, num_buckets=8)[0]

    hits_flat = int((F.cosine_similarity(out_flat, vals, dim=-1) > 0.85).sum().item())
    hits_mb = int((F.cosine_similarity(out_mb, vals, dim=-1) > 0.85).sum().item())

    assert hits_flat <= 48
    assert hits_mb >= 76, f"Expected >= 76/80 super-rank recalls in 8x24 bucketed bank, got {hits_mb} (flat={hits_flat})"


def test_v90_streaming_and_batched_parity_with_buckets_and_percept_gate() -> None:
    """Verifies exact `forward_sequence` == `step_byte_interactive` == `BatchedScarFieldSuperEngine` parity."""
    torch.manual_seed(9003)
    cfg = _v90_cfg(biorth_slots=32, biorth_buckets=4, enable_percept_s1_gate=True)
    master = ScarFieldEngine(cfg)
    heal_all_e8_scars(master)
    master.eval()

    # 1a. Multi-chunk stateful `O(1)` continuation (`chunk0 [16] -> chunk1 [16] -> chunk2 [16]`) == full `48`-byte forward!
    seq = torch.randint(32, 122, (1, 48), dtype=torch.long)
    seq[0, 6:18] = ord(" ")  # Predictable vacuum run triggering `enable_percept_s1_gate` 1-step shortcut!

    with torch.no_grad():
        seq_out = master.forward_sequence(seq, state=None, adaptive_depth=True)
        c0 = master.forward_sequence(seq[:, :16], state=None, adaptive_depth=True)
        c1 = master.forward_sequence(seq[:, 16:32], state=c0["state"], adaptive_depth=True)
        c2 = master.forward_sequence(seq[:, 32:], state=c1["state"], adaptive_depth=True)
        cat_scores = torch.cat((c0["scores"], c1["scores"], c2["scores"]), dim=1)
        max_chunk_diff = float((seq_out["scores"] - cat_scores).abs().max().item())
        assert max_chunk_diff < 5e-5, f"Multi-chunk bucketed continuation drift: {max_chunk_diff}"

        # 1b. Byte-by-byte `enable_percept_s1_gate` parity (`wy_chunk_size=1` makes biorth bank update every step identically!)
        cfg_step = _v90_cfg(biorth_slots=32, biorth_buckets=4, enable_percept_s1_gate=True)
        cfg_step.wy_chunk_size = 1
        eng_step = ScarFieldEngine(cfg_step)
        eng_step.load_state_dict(master.state_dict())
        eng_step.eval()
        seq_short = seq[:, :20]
        seq_step_out = eng_step.forward_sequence(seq_short, state=None, adaptive_depth=True)
        st_iter = eng_step.init_state(1)
        step_scores = []
        for t in range(seq_short.shape[1]):
            b_id = int(seq_short[0, t].item())
            cur_t = torch.tensor([[b_id]], dtype=torch.long)
            win = torch.cat((st_iter.recent_bytes[:, -st_iter.recent_len :].long(), cur_t), dim=1) if st_iter.recent_len > 0 else cur_t
            hf_s, hw_s, s_s = eng_step.frontend(win)
            hf_t, hw_t, s_t = hf_s[:, -1], hw_s[:, -1], s_s[:, -1]
            use_s2 = bool(float(s_t[0].item()) >= cfg_step.s2_surprise_theta)
            hf_t, hw_t = eng_step.flow_percept(hf_t, hw_t, hf_t, num_steps=(2 if use_s2 else 1))
            v_sym_t = project_rows_to_unit_sphere(eng_step.head.out_bus_anchors)[cur_t.squeeze(0)]
            hf_ph, bus_c, _, _ = eng_step.bus.step_pulse(hf_t, hw_t, s_t, state=st_iter, mutate_state=True, v_shock_hint_t=v_sym_t)
            tn = min(eng_step._recent_buf_cap, win.shape[1])
            st_iter.recent_bytes.zero_()
            st_iter.recent_bytes[:, -tn:] = win[:, -tn:].to(torch.uint8)
            st_iter.recent_len = int(tn)
            k_st = cfg_step.s2_steps if use_s2 else cfg_step.s1_steps
            hf_o, hw_o = eng_step.flow(hf_ph, hw_t, bus_c, num_steps=k_st)
            if eng_step.extra_flows is not None and not (cfg_step.scout_g0_theta > 0.0 and float(s_t[0].item()) < cfg_step.scout_g0_theta):
                for ef in eng_step.extra_flows:
                    hf_o, hw_o = ef(hf_o, hw_o, bus_c, num_steps=k_st)
            sc_t, _ = eng_step.head.forward_scores(hf_o, hw_o, bus_ctx=bus_c)
            step_scores.append(sc_t)
        iter_scores = torch.stack(step_scores, dim=1)
        max_stream_diff = float((seq_step_out["scores"] - iter_scores).abs().max().item())
        assert max_stream_diff < 5e-5, f"Byte-by-byte vs sequence drift: {max_stream_diff}"

    # 2. Exact forward + backward parity against `BatchedScarFieldSuperEngine` (`M=2`)!
    master.train()
    m_cnt = 2
    models = [clone_scarfield_engine_fast(master, torch.device("cpu")) for _ in range(m_cnt)]
    bank = BatchedScarFieldSuperEngine(master, num_replicas=m_cnt, shared_crystal=False)
    with torch.no_grad():
        for pname, _, p_b in bank.master_to_bank_param_pairs(models[0]):
            assert p_b is not None
            p_b.copy_(torch.stack([dict(m.named_parameters())[pname].detach().clone() for m in models], dim=0))

    b_ids = torch.randint(32, 122, (m_cnt, 2, 36), dtype=torch.long)
    b_ids[0, :, 8:24] = ord("e")

    ref_tot = 0.0
    for i in range(m_cnt):
        models[i].zero_grad(set_to_none=True)
        di = models[i].compute_loss(b_ids[i])
        di["loss"].backward()
        ref_tot += float(di["loss"].item())

    bank.zero_grad(set_to_none=True)
    loss_b, _, _ = bank.compute_loss_batched(VectorizedRoundBatch(byte_ids=b_ids))
    loss_b.backward()
    assert abs(float(loss_b.item()) - ref_tot) < 1.5e-4

    for pname, _, p_b in bank.master_to_bank_param_pairs(models[0]):
        assert p_b is not None and p_b.grad is not None
        for i in range(m_cnt):
            g_ref = dict(models[i].named_parameters())[pname].grad
            assert g_ref is not None
            if float(torch.linalg.vector_norm(g_ref.float()).item()) > 1e-5 and g_ref.numel() >= 4:
                cos = float(F.cosine_similarity(g_ref.flatten().float(), p_b.grad[i].flatten().float(), dim=0).item())
                assert cos > 0.9999, f"Grad drift on `{pname}`[m={i}]: cos={cos}"


def test_v90_interscale_stiefel_uscar_transport_and_gdpr_wipe() -> None:
    """Verifies zero-recompilation `.uscar` cartridge transport from `d=64` to `d=128` + `0.0` GDPR wipe."""
    torch.manual_seed(9004)
    store_64 = UniversalScarCartridgeStore(d_world=64, d_bus=64)
    q_64 = project_rows_to_unit_sphere(torch.randn(1, 64))
    y_64 = torch.zeros(1, 64)
    y_64[0, 3] = 2.5
    cart = store_64.mint_zero_sgd_rrr("fact_alpha", "doc_sovereign", q_64, y_64, rank=4, horizon_kappa=0.45)

    # Mount unmodified `64D` cartridge into a `128D` scaled model!
    store_128 = UniversalScarCartridgeStore(d_world=128, d_bus=128)
    w_in, w_out = store_128.register_interscale_stiefel_bridge(64, 64, seed=42)
    assert float((w_in @ w_in.T - torch.eye(64)).abs().max().item()) < 1e-5
    store_128.cartridges["fact_alpha"] = cart

    # Lift query `q_64` onto `S^{127}` via `w_in^T`: must resonate at `1.0`!
    q_128_hit = project_rows_to_unit_sphere(q_64 @ w_in)
    out_hit, res_hit = store_128.query(q_128_hit)
    assert float(res_hit.item()) > 0.98
    assert float(out_hit.norm().item()) > 1.0

    # Orthogonal query in `128D` must yield exact `0.0`!
    q_128_orth = project_rows_to_unit_sphere(torch.randn(1, 128))
    q_128_orth = project_rows_to_unit_sphere(q_128_orth - (q_128_orth @ w_in.T) @ w_in)
    out_miss, res_miss = store_128.query(q_128_orth)
    assert float(res_miss.item()) == 0.0 and float(out_miss.abs().max().item()) == 0.0

    # Cryptographic `0x00` GDPR wipe -> exact `0.0` residual even on `q_128_hit`!
    store_128.wipe_document_0x00("doc_sovereign")
    out_wiped, res_wiped = store_128.query(q_128_hit)
    assert float(res_wiped.item()) == 0.0 and float(out_wiped.abs().max().item()) == 0.0


def test_v90_scaled_container_mmap_roundtrip(tmp_path: Path) -> None:
    """Verifies `.sfield.scar` export & C99/SIMD `mmap` cold-load preserves `biorth_buckets` & `enable_percept_s1_gate`."""
    torch.manual_seed(9005)
    cfg = _v90_cfg(d_form=128, d_world=128, d_bus=128, biorth_slots=64, biorth_buckets=8, enable_percept_s1_gate=True)
    eng = ScarFieldEngine(cfg)
    heal_all_e8_scars(eng)
    eng.eval()

    scar_p = tmp_path / "v90_scaled.sfield.scar"
    n_bytes = eng.save_packed_scar_container(scar_p)
    assert n_bytes > 0

    eng_mm, _ct = ScarFieldEngine.load_packed_scar_mmap(scar_p)
    eng_mm.eval()
    assert eng_mm.cfg.biorth_buckets == 8
    assert eng_mm.cfg.enable_percept_s1_gate is True

    x = torch.randint(32, 122, (2, 24), dtype=torch.long)
    with torch.no_grad():
        s_ref = eng.forward_sequence(x)["scores"]
        s_mm = eng_mm.forward_sequence(x)["scores"]
    cos = float(F.cosine_similarity(s_ref.flatten().float(), s_mm.flatten().float(), dim=0).item())
    assert cos > 0.99999, f"Expected exact mmap parity, got cos={cos}"
