"""Forensic Verification Suite for `ScarField v7.4-Turbo` & `System 1` Scout-Guided Training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scar_field import (
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    System1CorpusScoutFleet,
    compute_ghost_jump_antiloop_loss,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
)
from scar_field.cayley_geodesic import _halley_spd_matrix_inv_cuda


def test_halley_spd_zero_cusolver_inverse_exactness() -> None:
    """Verifies `6`-step Cubic Halley SPD + `2`-step Schulz polish matches `linalg.inv` to `< 2e-6`."""
    torch.manual_seed(7401)
    u = torch.randn(32, 128, 8) * 0.25
    v = torch.randn(32, 128, 8) * 0.25
    p = torch.cat((u, -v), dim=-1)
    q = torch.cat((v, u), dim=-1)
    eye = torch.eye(16)
    m = eye + 0.35 * torch.matmul(q.transpose(-2, -1), p)

    inv_halley = _halley_spd_matrix_inv_cuda(m, eye)
    inv_ref = torch.linalg.inv(m)
    max_diff = float((inv_halley - inv_ref).abs().max().item())
    res_ident = float((torch.matmul(m, inv_halley) - eye).abs().max().item())
    assert max_diff < 2e-6
    assert res_ident < 2e-6


def test_scout_gate_g0_batch_vs_interactive_streaming_parity() -> None:
    """Verifies `Gate G0` sparse bypass on `extra_flows` preserves exact `cos > 0.999999` streaming parity."""
    torch.manual_seed(7402)
    cfg = ScarFieldConfig(
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
        s2_steps=4,
        s2_surprise_theta=0.45,
        num_macro_cycles=2,
        biorth_slots=16,
        scout_g0_theta=0.40,
        antiloop_weight=0.15,
    )
    eng = ScarFieldEngine(cfg)
    eng.eval()

    text = b"O batedor System 1 filtra o vacuo e foca nos choques PIN=914!"
    seq = torch.tensor([[b for b in text]], dtype=torch.long)

    with torch.no_grad():
        # 1. Exact chunk-aligned `O(1)` streaming parity (`wy_chunk_size=32`, `biorth_slots=16`, `scout_g0_theta=0.40`)
        out_batch = eng.forward_sequence(seq, adaptive_depth=True)
        st_chunk = eng.init_state(batch_size=1)
        b0 = st_chunk.total_state_bytes
        c1 = eng.forward_sequence(seq[:, :32], state=st_chunk, adaptive_depth=True)
        c2 = eng.forward_sequence(seq[:, 32:], state=c1["state"], adaptive_depth=True)
        cat_chunk = torch.cat((c1["scores"], c2["scores"]), dim=1)
        assert c2["state"].total_state_bytes - b0 == 0
        cos_chunk = float(F.cosine_similarity(out_batch["scores"].flatten(), cat_chunk.flatten(), dim=0).item())
        assert cos_chunk > 0.999999

        # 2. Exact single-byte (`1` byte at a time) streaming parity (`wy_chunk_size=1`)
        eng.cfg.wy_chunk_size = 1
        out_batch_1 = eng.forward_sequence(seq, adaptive_depth=True)
        st = eng.init_state(batch_size=1)
        step_scores = []
        for t in range(seq.shape[1]):
            cur_id = seq[0, t : t + 1].unsqueeze(0)
            out_t = eng.forward_sequence(cur_id, state=st, adaptive_depth=True)
            st = out_t["state"]
            step_scores.append(out_t["scores"])
        cat_scores = torch.cat(step_scores, dim=1)
        assert st.total_state_bytes - b0 == 0
        cos = float(F.cosine_similarity(out_batch_1["scores"].flatten(), cat_scores.flatten(), dim=0).item())
        assert cos > 0.999999


def test_system1_corpus_scout_fleet_and_antiloop_and_scalar_muon() -> None:
    """Verifies `System1CorpusScoutFleet`, `compute_ghost_jump_antiloop_loss`, and scalar `RiemannianMuon`."""
    torch.manual_seed(7403)
    cfg = ScarFieldConfig(
        d_form=64,
        d_world=64,
        d_bus=64,
        d_key=64,
        num_macro_cycles=2,
        biorth_slots=8,
        scout_g0_theta=0.35,
        antiloop_weight=0.20,
    )
    eng = ScarFieldEngine(cfg)
    fleet = System1CorpusScoutFleet(oversample_factor=4, anchor_fraction=0.25)
    assert fleet.record_curriculum is False

    # Train frontend briefly on a predictable sentence so scouts distinguish known vs novel windows
    known_row = list((b"memoria constante em espaco O(1) " * 3)[:64])
    f_opt = torch.optim.SGD(eng.frontend.parameters(), lr=0.3)
    known_t = torch.tensor([known_row] * 4, dtype=torch.long)
    for _ in range(12):
        f_opt.zero_grad()
        _, _, _, lf = eng.frontend(known_t, return_reflex_loss=True)
        lf.backward()
        f_opt.step()

    cands = []
    for i in range(16):
        if i < 4:
            cands.append([32] * 64)  # degenerate whitespace
        elif i < 12:
            cands.append(known_row)  # low-surprise familiar window
        else:
            cands.append([((b * 29 + i * 13) % 88) + 35 for b in known_row])  # high-surprise frontier!
    cand_t = torch.tensor(cands, dtype=torch.long)

    sel_b, _, tel = fleet.select_frontier_batch(eng, cand_t, batch_size=4)
    assert sel_b.shape == (4, 64)
    assert tel.degenerate_windows_filtered == 4
    assert tel.epistemic_gain_ratio > 1.25

    # Anti-loop repulsive loss is strictly positive on collapsed constant `h_world` when targets differ
    hw_loop = project_rows_to_unit_sphere(torch.ones(2, 24, 64))
    tgt_diff = torch.arange(24, dtype=torch.long).unsqueeze(0).expand(2, 24)
    l_loop = compute_ghost_jump_antiloop_loss(hw_loop, tgt_diff, lags=(4, 8))
    assert float(l_loop.item()) > 0.25

    # Scalar 1-element parameter in `RiemannianMuon` does NOT take a huge SignSGD jump on `1e-7` noise!
    p_scalar = torch.nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
    opt = RiemannianMuon([p_scalar], lr=1.5e-2, scalar_lr=5e-3)
    p_scalar.grad = torch.tensor(1e-7, dtype=torch.float32)
    opt.step()
    # Previously jumped by `-0.005` (`0.35 -> 0.345`); now moves `< 1e-6`!
    assert abs(float(p_scalar.item()) - 0.35) < 1e-5

    # End-to-end `heal_all_e8_scars` with fast subspace iteration
    healed = heal_all_e8_scars(eng)
    assert healed >= 6
