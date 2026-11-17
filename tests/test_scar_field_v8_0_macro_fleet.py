"""Forensic Mathematical & Architectural Test Suite for `ScarField v8.0` Macro-Scout Geodesic Fleet."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import scar_field
from scar_field import (
    E8SphericalLinear,
    MacroBatchSpec,
    MacroScoutGeodesicFleet,
    MacroScoutSpec,
    ScarFieldConfig,
    ScarFieldEngine,
    SharedCrystalMacroSwarm,
    System1CorpusScoutFleet,
    biorthogonal_conflict_free_tangent_merge,
    biorthogonal_shock_bank_insert_step_,
    project_rows_to_unit_sphere,
    riemannian_exp_retract_2d,
    riemannian_log_chord_2d,
)


def _tiny_v8_cfg() -> ScarFieldConfig:
    return ScarFieldConfig(
        vocab_size=256,
        d_form=64,
        d_world=64,
        d_bus=64,
        d_key=64,
        num_subspaces=8,
        ring_len=16,
        repulsive_len=8,
        base_rank=4,
        num_scars=16,
        s1_steps=1,
        s2_steps=3,
        wy_chunk_size=16,
        wave_lags=(1, 2, 4),
        num_macro_cycles=2,
        biorth_slots=8,
        biorth_lambda=0.04,
        enable_biorth_uscar=True,
        enable_adaptive_cap=True,
        scout_g0_theta=0.38,
        antiloop_weight=0.10,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
    )


def test_version_and_riemannian_log_exp_exact_bijection() -> None:
    assert scar_field.__version__ == "8.0.0"
    torch.manual_seed(8001)
    w0 = project_rows_to_unit_sphere(torch.randn(32, 128))
    # Mix micro-steps (`~1e-4` rad) and wide geodesic chords (`~0.8` rad)
    pert = torch.randn(32, 128)
    pert[:16] *= 1e-4
    pert[16:] *= 0.45
    w1 = project_rows_to_unit_sphere(w0 + pert)

    delta = riemannian_log_chord_2d(w0, w1)
    # 1. Exact orthogonality to base point `w0`: `<delta, w0> == 0`
    ortho_err = float((delta * w0).sum(dim=-1).abs().max().item())
    assert ortho_err < 1e-6, f"Tangent orthogonality violated: {ortho_err}"

    # 2. Geodesic norm matches exact chord spherical angle `2 * arcsin(||w1 - w0|| / 2)` (0 FP32 acos cancellation!)
    chord_half = 0.5 * torch.linalg.vector_norm(w1 - w0, ord=2, dim=-1)
    ang_true = 2.0 * torch.asin(chord_half.clamp(max=1.0))
    ang_log = torch.linalg.vector_norm(delta, ord=2, dim=-1)
    assert float((ang_true - ang_log).abs().max().item()) < 1e-6

    # 3. Exact round-trip bijection `Exp_{w0}(Log_{w0}(w1)) == w1`
    w_rec = riemannian_exp_retract_2d(w0, delta)
    rec_err = float((w_rec - w1).abs().max().item())
    assert rec_err < 2e-6, f"Exp(Log(w1)) round-trip drift: {rec_err}"


def test_biorthogonal_conflict_free_tangent_merge_properties() -> None:
    torch.manual_seed(8002)
    w0 = project_rows_to_unit_sphere(torch.randn(16, 64))

    def _tan(v: torch.Tensor) -> torch.Tensor:
        return v - (v * w0).sum(dim=-1, keepdim=True) * w0

    e1 = _tan(torch.randn(16, 64))
    e1 = e1 / torch.linalg.matrix_norm(e1, ord="fro")
    e2 = _tan(torch.randn(16, 64))
    e2 = e2 - float((e2 * e1).sum().item()) * e1
    e2 = e2 / torch.linalg.matrix_norm(e2, ord="fro")

    # Case A: Destructive conflict (`d_a = e1 - 0.6 e2`, `d_b = e2 - 0.6 e1` -> `<d_a, d_b> < 0`)
    d_a = e1 - 0.60 * e2
    d_b = e2 - 0.60 * e1
    assert float((d_a * d_b).sum().item()) < -0.5
    merged_ab, conflicts, _ = biorthogonal_conflict_free_tangent_merge([d_a, d_b], w0_sphere_2d=w0)
    assert conflicts >= 2
    assert float((merged_ab * d_a).sum().item()) > 0.0
    assert float((merged_ab * d_b).sum().item()) > 0.0
    assert float((merged_ab * w0).sum(dim=-1).abs().max().item()) < 1e-6

    # Case B: 3 redundant colinear prose scouts (`e1`) + 1 orthogonal Code specialist (`e2`) + 1 idle scout (`0`)
    d_idle = torch.zeros_like(e1)
    merged_multi, _, _ = biorthogonal_conflict_free_tangent_merge(
        [0.10 * e1, 0.10 * e1, 0.10 * e1, 0.10 * e2, d_idle],
        w0_sphere_2d=w0,
        ridge_lambda=0.10,
    )
    proj_e1 = float((merged_multi * e1).sum().item())
    proj_e2 = float((merged_multi * e2).sum().item())
    # Neither `3x` prose explosion (`0.30`) nor `1/5` specialist dilution (`0.02`)!
    assert 0.09 <= proj_e1 <= 0.13, f"Expected deduplicated ~0.10 on e1, got {proj_e1}"
    assert 0.095 <= proj_e2 <= 0.105, f"Expected un-diluted 0.10 on unique specialist e2, got {proj_e2}"


def test_async_stream_mode_and_scout_telemetry_parity() -> None:
    torch.manual_seed(8003)
    cfg = _tiny_v8_cfg()
    eng_sync = ScarFieldEngine(cfg)
    eng_async = ScarFieldEngine(cfg)
    eng_async.load_state_dict(eng_sync.state_dict())
    eng_async.set_async_stream_mode(True)

    x = torch.randint(1, 200, (3, 40), dtype=torch.long)
    ooc = torch.zeros_like(x, dtype=torch.bool)
    ooc[0, -4:] = True

    ls_sync = eng_sync.compute_loss(x, out_of_cone_mask=ooc)
    ls_async = eng_async.compute_loss(x, out_of_cone_mask=ooc)
    assert abs(float(ls_sync["loss"].item() - ls_async["loss"].item())) < 1e-6

    ls_sync["loss"].backward()
    ls_async["loss"].backward()
    for (n1, p1), (n2, p2) in zip(eng_sync.named_parameters(), eng_async.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        g_cos = float(F.cosine_similarity(p1.grad.reshape(1, -1), p2.grad.reshape(1, -1)).item())
        if float(p1.grad.abs().max().item()) > 1e-7:
            assert g_cos > 0.99999, f"Grad mismatch on `{n1}`: cos={g_cos}"

    fleet_a = System1CorpusScoutFleet(oversample_factor=3, anchor_fraction=0.25)
    fleet_b = System1CorpusScoutFleet(oversample_factor=3, anchor_fraction=0.25)
    cand = torch.randint(1, 200, (12, 32), dtype=torch.long)
    sel_a, _, tel_a = fleet_a.select_frontier_batch(eng_sync, cand, batch_size=4, sync_telemetry=True)
    sel_b, _, _ = fleet_b.select_frontier_batch(eng_async, cand, batch_size=4, sync_telemetry=False)
    tel_b = fleet_b.flush_async_telemetry()
    assert torch.equal(sel_a, sel_b)
    assert abs(tel_a.epistemic_gain_ratio - tel_b.epistemic_gain_ratio) < 1e-4


def test_biorthogonal_shock_bank_monotonic_topm_and_live_grad() -> None:
    torch.manual_seed(8004)
    bsz, m_slots, dk, dv = 2, 4, 16, 16
    bk = torch.zeros(bsz, m_slots, dk)
    bv = torch.zeros(bsz, m_slots, dv)
    bsal = torch.full((bsz, m_slots), -1.0)
    bval = torch.zeros(bsz, m_slots, dtype=torch.bool)

    # Insert high-salience Passkey (`0.92`) at t=0
    k_pin = project_rows_to_unit_sphere(torch.randn(bsz, dk))
    v_pin = project_rows_to_unit_sphere(torch.randn(bsz, dv))
    biorthogonal_shock_bank_insert_step_(
        bk, bv, bsal, bval, k_pin, v_pin, torch.full((bsz,), 0.92), write_thresh=0.20
    )

    # Flood with 64 weaker shocks (`0.84` — previously `0.84 >= 0.92 * 0.90` caused priority-inversion eviction!)
    for step_i in range(64):
        k_d = project_rows_to_unit_sphere(torch.randn(bsz, dk))
        # Ensure non-dedup (`cos <= 0.92`)
        k_d = project_rows_to_unit_sphere(k_d - (k_d * k_pin).sum(dim=-1, keepdim=True) * k_pin)
        v_d = project_rows_to_unit_sphere(torch.randn(bsz, dv))
        sal_d = torch.full((bsz,), 0.84 - 0.0005 * (step_i % 10))
        biorthogonal_shock_bank_insert_step_(bk, bv, bsal, bval, k_d, v_d, sal_d, write_thresh=0.20)

    # Verify the `0.92` Passkey survived all 64 distractor chunks in every batch item!
    max_pin_cos = (bk @ k_pin.unsqueeze(-1)).squeeze(-1).amax(dim=-1)
    assert float(max_pin_cos.min().item()) > 0.99999, f"Passkey shock evicted: {max_pin_cos}"

    # Verify live differentiability + exact forward/streaming parity in `ChannelSelectiveKalmanScarBus`
    cfg = _tiny_v8_cfg()
    eng = ScarFieldEngine(cfg)
    seq = torch.randint(32, 122, (1, 64), dtype=torch.long)
    out_grad = eng.forward_sequence(seq, adaptive_depth=False)
    with torch.no_grad():
        out_nograd = eng.forward_sequence(seq, adaptive_depth=False)
    assert float((out_grad["scores"] - out_nograd["scores"]).abs().max().item()) < 1e-6

    # Check step-by-step `step_pulse` parity with `forward_sequence`
    st = eng.init_state(batch_size=1)
    step_scores = []
    with torch.no_grad():
        for t in range(seq.shape[1]):
            b_id = int(seq[0, t].item())
            hist = st.recent_bytes[:, -st.recent_len :].long() if st.recent_len > 0 else torch.empty(1, 0, dtype=torch.long)
            win = torch.cat((hist, torch.tensor([[b_id]])), dim=1)
            hf_s, hw_s, s_s = eng.frontend(win)
            hf_t, hw_t = eng.flow_percept(hf_s[:, -1], hw_s[:, -1], hf_s[:, -1], num_steps=2)
            v_sym = project_rows_to_unit_sphere(eng.head.out_bus_anchors)[torch.tensor([b_id])]
            hf_p, bus_t, _, _ = eng.bus.step_pulse(hf_t, hw_t, s_s[:, -1], state=st, mutate_state=True, v_shock_hint_t=v_sym)
            tail_n = min(eng._recent_buf_cap, win.shape[1])
            st.recent_bytes.zero_()
            st.recent_bytes[:, -tail_n:] = win[:, -tail_n:].to(torch.uint8)
            st.recent_len = int(tail_n)
            hf_o, hw_o = eng.flow(hf_p, hw_t, bus_t, num_steps=cfg.s2_steps)
            assert eng.extra_flows is not None
            for ef in eng.extra_flows:
                hf_o, hw_o = ef(hf_o, hw_o, bus_t, num_steps=cfg.s2_steps)
            sc_t, _ = eng.head.forward_scores(hf_o, hw_o, bus_ctx=bus_t)
            step_scores.append(sc_t)
    sc_cat = torch.cat(step_scores, dim=0).unsqueeze(0)
    cos_stream = float(F.cosine_similarity(out_nograd["scores"].reshape(1, -1), sc_cat.reshape(1, -1)).item())
    assert cos_stream > 0.9995, f"Chunked streaming parity degraded: cos={cos_stream}"

    # And with `wy_chunk_size = 1`, `forward_sequence` and `step_pulse` are bit-exact (`cos > 0.999999`)!
    eng.cfg.wy_chunk_size = 1
    with torch.no_grad():
        out_c1 = eng.forward_sequence(seq, adaptive_depth=False)
    cos_c1 = float(F.cosine_similarity(out_c1["scores"].reshape(1, -1), sc_cat.reshape(1, -1)).item())
    assert cos_c1 > 0.999999, f"Exact step parity degraded: cos={cos_c1}"


def test_macro_scout_geodesic_fleet_and_shared_crystal_swarm(tmp_path) -> None:
    torch.manual_seed(8005)
    cfg = _tiny_v8_cfg()
    master = ScarFieldEngine(cfg)
    size_before = master.save_packed_scar_container(tmp_path / "before.sfield.scar")

    b_prose = torch.tensor([[ord(c) for c in "Thermodynamics governs macroscopic entropy.\n"] * 2], dtype=torch.long)[:, :48]
    b_code = torch.tensor([[ord(c) for c in "def add(a: int, b: int) -> int:\n    return a + b\n"]], dtype=torch.long)[:, :48]
    b_pk = torch.tensor([[ord(c) for c in "[PASSKEY PIN=314] Entropy grows. [QUERY PIN=> 314.]"]], dtype=torch.long)[:, :48]

    with torch.no_grad():
        l0_prose = float(master.compute_loss(b_prose)["loss_cap"].item())
        l0_code = float(master.compute_loss(b_code)["loss_cap"].item())
        l0_pk = float(master.compute_loss(b_pk)["loss_cap"].item())

    specs = [
        MacroScoutSpec("prose", weight=1.0),
        MacroScoutSpec("code", weight=1.0),
        MacroScoutSpec("long_biorth", weight=1.0, owns_biorth_organ=True),
    ]
    fleet = MacroScoutGeodesicFleet(
        master,
        specs,
        base_lr=8.0e-3,
        scalar_lr=4.0e-3,
        outer_lr=0.90,
        outer_momentum=0.40,
        biorth_gain_floor=0.35,
    )

    samplers = [
        lambda _w, _h: MacroBatchSpec(b_prose),
        lambda _w, _h: MacroBatchSpec(b_code),
        lambda _w, _h: MacroBatchSpec(b_pk),
    ]
    for _ in range(3):
        tel = fleet.run_concurrent_round(samplers, local_steps=4, grad_clip=1.0)
    assert tel.biorth_gain_after_merge >= 0.35

    with torch.no_grad():
        l1_prose = float(master.compute_loss(b_prose)["loss_cap"].item())
        l1_code = float(master.compute_loss(b_code)["loss_cap"].item())
        l1_pk = float(master.compute_loss(b_pk)["loss_cap"].item())

    assert l1_prose < l0_prose, f"Prose loss did not improve: {l0_prose} -> {l1_prose}"
    assert l1_code < l0_code, f"Code loss did not improve: {l0_code} -> {l1_code}"
    assert l1_pk < l0_pk, f"Passkey loss did not improve: {l0_pk} -> {l1_pk}"

    # Verify Level-3 `SharedCrystalMacroSwarm`: `weight_sphere is None` on workers & `Delta = 0 B` `.sfield.scar`!
    swarm = SharedCrystalMacroSwarm(master, num_workers=3, scar_lr=5.0e-3, scalar_lr=2.0e-3, distill_eta=0.50)
    for w in swarm.workers:
        for mod in w.modules():
            if isinstance(mod, E8SphericalLinear):
                assert mod.weight_sphere is None and mod.tau_base is None
    assert swarm.mutable_params_per_worker < swarm.master_total_params

    sw_tel = swarm.run_swarm_round(samplers, local_steps=3, grad_clip=1.0)
    assert sw_tel.crystal_layers_distilled >= 1
    assert sw_tel.param_reduction_ratio_x > 2.5

    size_after = master.save_packed_scar_container(tmp_path / "after.sfield.scar")
    assert size_after == size_before, f"Container byte drift after v8.0 distillation: {size_before} != {size_after}"
