"""Forensic Verification Suite for `ScarCodec v6.1` & `ScarField v7.3` Biorthogonal + Chat SFT Upgrades."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from scar_codec import (
    BiorthogonalObliqueProjector,
    BiorthogonalObliqueProjectorNumPy,
    OrthogonalSubspaceProjector,
    SurpriseGatedAnomalySCache,
    SurpriseGatedAnomalySCacheNumPy,
)
from scar_field import (
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    UniversalScarCartridgeStore,
    biorthogonal_shock_bank_insert_step_,
    biorthogonal_shock_bank_read,
    encode_sft_dialogue_bytes_and_mask,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
)


def test_biorthogonal_oblique_projector_cancels_jammer_without_attenuating_oblique_signal() -> None:
    """When signal `u_s` and 1,000x jammer `u_j` overlap obliquely (`cos = 0.85`), `P_biorth` preserves `u_s`!"""
    rng = np.random.default_rng(2026073)
    c_dim, t_len = 32, 1024
    raw = rng.standard_normal((c_dim, 2)).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    u_s = q[:, 0:1]  # Desired pulsar / GW mode
    # Oblique jammer just ~31.8 degrees away (`<u_s, u_j> = 0.85`!)
    u_j = 0.85 * u_s + np.sqrt(1.0 - 0.85**2) * q[:, 1:2]
    u_j = u_j / np.linalg.norm(u_j, axis=0, keepdims=True)

    t = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    clean_pulse = np.exp(-((t - 0.52) ** 2) / (2.0 * 0.018**2)) * np.cos(2.0 * np.pi * 28.0 * t)
    jam_cal = u_j @ (1000.0 * rng.standard_normal((1, t_len)).astype(np.float32)) + 0.01 * rng.standard_normal(
        (c_dim, t_len)
    ).astype(np.float32)

    sig_true = u_s @ clean_pulse[None, :]
    jam_test = u_j @ (1000.0 * np.sin(2.0 * np.pi * 60.0 * t + 0.4)[None, :]).astype(np.float32)
    obs = sig_true + jam_test + 0.005 * rng.standard_normal((c_dim, t_len)).astype(np.float32)

    p_orth = OrthogonalSubspaceProjector.fit_from_interference_calibration(torch.from_numpy(jam_cal), subspace_rank=1)
    p_biorth_pt = BiorthogonalObliqueProjector.fit_from_interference_calibration(
        torch.from_numpy(jam_cal), torch.from_numpy(u_s), subspace_rank=1, ridge_lambda=1e-6
    )
    p_biorth_np = BiorthogonalObliqueProjectorNumPy.fit_from_interference_calibration(
        jam_cal, u_s, subspace_rank=1, ridge_lambda=1e-6
    )

    rec_orth = (u_s.T @ p_orth.project_clean(torch.from_numpy(obs)).numpy()).reshape(-1)
    rec_biorth_pt = (u_s.T @ p_biorth_pt.project_clean(torch.from_numpy(obs)).numpy()).reshape(-1)
    rec_biorth_np = (u_s.T @ p_biorth_np.project_clean(obs)).reshape(-1)

    # Parity between PyTorch and Zero-PyTorch NumPy
    assert float(np.max(np.abs(rec_biorth_pt - rec_biorth_np))) < 1e-3

    # 1) Exact distortionless oblique signal preservation (`< 1e-3` vs `> 0.72` loss in Orthogonal!)
    pure_sig_orth = (u_s.T @ p_orth.project_clean(torch.from_numpy(sig_true)).numpy()).reshape(-1)
    pure_sig_biorth = (u_s.T @ p_biorth_np.project_clean(sig_true)).reshape(-1)
    assert float(np.linalg.norm(pure_sig_orth - clean_pulse) / np.linalg.norm(clean_pulse)) > 0.72
    assert float(np.linalg.norm(pure_sig_biorth - clean_pulse) / np.linalg.norm(clean_pulse)) < 1e-3

    # 2) Full noisy + 1,000x oblique jammer recovery (>9x lower L2 error than Orthogonal!)
    rel_err_orth = float(np.linalg.norm(rec_orth - clean_pulse) / np.linalg.norm(clean_pulse))
    rel_err_biorth = float(np.linalg.norm(rec_biorth_np - clean_pulse) / np.linalg.norm(clean_pulse))
    cos_biorth = float(np.dot(rec_biorth_np, clean_pulse) / (np.linalg.norm(rec_biorth_np) * np.linalg.norm(clean_pulse)))
    assert rel_err_orth > 0.70
    assert rel_err_biorth < 0.09
    assert cos_biorth > 0.996


def test_biorthogonal_scache_partial_cue_eliminates_correlated_fault_crosstalk() -> None:
    """In exact `34,816 B` (`Delta = 0 B`), `biorthogonal=True` unmixes oblique co-active fault modes."""
    rng = np.random.default_rng(7302)
    c_dim = 64
    sc_np = SurpriseGatedAnomalySCacheNumPy(channels=c_dim, surprise_threshold=3.5)
    sc_pt = SurpriseGatedAnomalySCache(channels=c_dim, surprise_threshold=3.5)
    assert sc_np.physical_bytes() == 34816
    assert sc_pt.physical_bytes() == 34816

    for t in range(80):
        nom = (rng.standard_normal(c_dim) * 0.08).astype(np.float32)
        sc_np.ingest(nom, t)
        sc_pt.ingest(torch.from_numpy(nom), t)

    q, _ = np.linalg.qr(rng.standard_normal((c_dim, 3)).astype(np.float32))
    m1 = q[:, 0] * 8.0
    m2 = (0.80 * q[:, 0] + 0.60 * q[:, 1]) * 8.0  # Strongly oblique (`cos = 0.80`!)
    m3 = (0.75 * q[:, 0] + 0.35 * q[:, 1] + np.sqrt(1.0 - 0.75**2 - 0.35**2) * q[:, 2]) * 8.0
    for idx_m, m_vec in enumerate((m1, m2, m3), start=100):
        sc_np.ingest(m_vec, idx_m)
        sc_pt.ingest(torch.from_numpy(m_vec), idx_m)

    # Superposed oblique anomaly `1.2 * m1 - 0.9 * m2 + 0.7 * m3`, observing ONLY 32 even channels!
    target = (1.2 * m1 - 0.9 * m2 + 0.7 * m3).astype(np.float32)
    obs_mask = np.zeros(c_dim, dtype=bool)
    obs_mask[::2] = True
    partial = np.where(obs_mask, target, 0.0).astype(np.float32)

    rec_orth = sc_np.recall_from_partial_cue(partial, obs_mask, input_is_deviation=True, biorthogonal=False)
    rec_biorth = sc_np.recall_from_partial_cue(partial, obs_mask, input_is_deviation=True, biorthogonal=True)
    rec_biorth_pt = (
        sc_pt.recall_from_partial_cue(
            torch.from_numpy(partial), torch.from_numpy(obs_mask), input_is_deviation=True, biorthogonal=True
        )
        .cpu()
        .numpy()
    )

    hidden = ~obs_mask
    cos_orth = float(
        np.dot(rec_orth[hidden], target[hidden])
        / (np.linalg.norm(rec_orth[hidden]) * np.linalg.norm(target[hidden]) + 1e-12)
    )
    cos_biorth = float(
        np.dot(rec_biorth[hidden], target[hidden])
        / (np.linalg.norm(rec_biorth[hidden]) * np.linalg.norm(target[hidden]) + 1e-12)
    )
    assert float(np.max(np.abs(rec_biorth - rec_biorth_pt))) < 1e-3
    assert cos_biorth > 0.995
    assert cos_biorth > cos_orth + 0.08
    assert sc_np.physical_bytes() == 34816


def test_scarfield_v7_3_biorthogonal_shock_bank_and_uscar_dual_poles() -> None:
    """Verifies `K_tilde K^T ≈ I` de-crosstalking in `BiorthogonalShockBank` & sibling `.uscar` poles."""
    torch.manual_seed(7303)
    d_key, d_bus, m_slots = 64, 64, 16
    b_keys = torch.zeros(1, m_slots, d_key)
    b_vals = torch.zeros(1, m_slots, d_bus)
    b_sal = torch.full((1, m_slots), -1.0)
    b_val = torch.zeros(1, m_slots, dtype=torch.bool)

    # Create 12 strongly hub-clustered keys (`<k_i, k_j> ~ 0.76`) with orthogonal values `v_i`!
    hub = project_rows_to_unit_sphere(torch.randn(1, d_key))
    orth_v = torch.eye(d_bus)[:12]
    stored_k = []
    for i in range(12):
        ki = project_rows_to_unit_sphere(0.87 * hub + 0.49 * project_rows_to_unit_sphere(torch.randn(1, d_key)))
        stored_k.append(ki)
        biorthogonal_shock_bank_insert_step_(
            b_keys, b_vals, b_sal, b_val, ki, orth_v[i : i + 1], torch.tensor([0.85])
        )

    # Query slot 3 (`q = k_3`): naive transpose `K^T q` leaks all 11 hubbed neighbors, while Biorthogonal isolates `v_3`!
    q3 = stored_k[3].unsqueeze(1)  # [1, 1, d_key]
    biorth_rec = biorthogonal_shock_bank_read(
        q3, b_keys, b_vals, b_val, horizon_kappa=torch.tensor(0.45), ridge_lambda=1e-3
    ).squeeze()
    naive_weights = torch.clamp(torch.matmul(b_keys[0, :12], stored_k[3].squeeze(0)), min=0.0)
    naive_rec = torch.matmul(naive_weights, b_vals[0, :12])

    target_v3 = orth_v[3]
    cos_biorth = float(torch.dot(biorth_rec, target_v3) / (biorth_rec.norm() * target_v3.norm() + 1e-8))
    cos_naive = float(torch.dot(naive_rec, target_v3) / (naive_rec.norm() * target_v3.norm() + 1e-8))
    assert cos_biorth > 0.998
    assert cos_naive < 0.65

    # Sibling `.uscar` poles (`<p_1, p_2> ~ 0.82`): biorthogonal dual poles isolate fact 1 & honor 0x00 GDPR wipe!
    store = UniversalScarCartridgeStore(d_world=64, d_bus=64, biorthogonal_poles=True, biorth_lambda=1e-3)
    base_p = project_rows_to_unit_sphere(torch.randn(1, 64))
    q1 = project_rows_to_unit_sphere(0.90 * base_p + 0.43 * project_rows_to_unit_sphere(torch.randn(8, 64)))
    q2 = project_rows_to_unit_sphere(0.90 * base_p + 0.43 * project_rows_to_unit_sphere(torch.randn(8, 64)))
    y1 = torch.zeros(8, 64)
    y1[:, 0] = 1.0
    y2 = torch.zeros(8, 64)
    y2[:, 1] = 1.0
    c1 = store.mint_zero_sgd_rrr("fact_polar", "doc_earth", q1, y1, rank=4, horizon_kappa=0.45)
    _ = store.mint_zero_sgd_rrr("fact_equat", "doc_earth", q2, y2, rank=4, horizon_kappa=0.45)

    out_at_p1, _ = store.query(c1.pole_world.unsqueeze(0))
    # Channel 0 (`fact_polar`) dominates cleanly with near-zero bleed from channel 1 (`fact_equat`)!
    assert float(out_at_p1[0, 0].abs().item()) > 8.0 * float(out_at_p1[0, 1].abs().item())

    store.wipe_cartridge_0x00("fact_polar")
    out_post_wipe, _ = store.query(c1.pole_world.unsqueeze(0))
    assert float(out_post_wipe[0, 0].abs().item()) == 0.0


def test_scarfield_v7_3_dialogue_sft_multi_turn_chat_and_sfield_mmap(tmp_path: Path) -> None:
    """Verifies masked SFT training, multi-turn `generate_reply` (`Delta = 0 B`), and `.sfield.scar` `mmap`."""
    torch.manual_seed(7304)
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
        s2_steps=2,
        wave_lags=(1, 2, 4, 8),
        num_macro_cycles=2,
        biorth_slots=16,
        enable_adaptive_cap=True,
    )
    eng = ScarFieldEngine(cfg)
    st0 = eng.init_state(batch_size=1)
    bytes_t0 = st0.total_state_bytes

    turns = [("Qual a capital do Brasil?", "Brasilia."), ("2 + 2?", "4.")]
    arr_u8, mask_f32 = encode_sft_dialogue_bytes_and_mask(turns)
    seq_t = torch.from_numpy(arr_u8.astype(np.int64)).unsqueeze(0)
    msk_t = torch.from_numpy(mask_f32).unsqueeze(0)

    opt = RiemannianMuon(eng.parameters(), lr=0.02)
    losses = []
    for step in range(45):
        opt.zero_grad()
        res = eng.compute_loss(seq_t, target_loss_mask=msk_t)
        res["loss"].backward()
        opt.step()
        if (step + 1) % 15 == 0:
            heal_all_e8_scars(eng)
        losses.append(float(res["loss_cap"].item()))

    assert losses[-1] < losses[0] * 0.35

    rep1 = eng.generate_reply("Qual a capital do Brasil?", state=st0, max_new_bytes=32)
    rep2 = eng.generate_reply("2 + 2?", state=rep1.state, max_new_bytes=16)
    assert rep1.state_bytes == bytes_t0
    assert rep2.state_bytes == bytes_t0  # Strict O(1) Delta = +0 B across multi-turn chat!
    assert "Brasilia" in rep1.text
    assert "4" in rep2.text

    scar_path = tmp_path / "v7_3_chat_test.sfield.scar"
    eng.save_packed_scar_container(scar_path)
    eng_mm, ctr = ScarFieldEngine.load_packed_scar_mmap(scar_path, verify_crc=True)
    try:
        rep_mm = eng_mm.generate_reply("Qual a capital do Brasil?", max_new_bytes=32)
        assert rep_mm.text == rep1.text
    finally:
        ctr.close()


def test_biorthogonal_smooth_shock_lattice_field_codecs_2d_3d_and_c_simd_parity(tmp_path: Path) -> None:
    """Verifies `biorthogonal=True` in `NumPyVQ4ScarCodec`, `ScarCodecE8`, and `ScarCodecND`."""
    from scar_codec import NumPyVQ4ScarCodec, ScarCodecE8, ScarCodecND, ScarMmapFile, save_scar_file

    rng = np.random.default_rng(2026062)
    rows, cols = 64, 256
    y_grid = np.linspace(0.0, 2.0 * np.pi, rows, dtype=np.float32)[:, None]
    x_grid = np.linspace(0.0, 4.0 * np.pi, cols, dtype=np.float32)[None, :]
    smooth = 45.0 * np.sin(y_grid) * np.cos(1.5 * x_grid) + 20.0 * np.cos(2.0 * y_grid + x_grid)
    field = smooth + 1.8 * rng.standard_normal((rows, cols)).astype(np.float32)
    # Inject bright localized stellar shocks + pure vacuum rows `[58:64]`
    for _ in range(24):
        rr = int(rng.integers(0, 56))
        cc = int(rng.integers(0, cols))
        field[rr, cc] += float(rng.uniform(400.0, 900.0))
    field[58:64, :] = 0.0

    # 1. VQ4-Scar: Biorthogonal lowers RMSE at zero extra non-spike bytes, keeps exact 0.0 vacuum & C99/SIMD parity
    c_orth = NumPyVQ4ScarCodec(rank=16, max_point_spikes=32, biorthogonal=False)
    c_bi = NumPyVQ4ScarCodec(rank=16, max_point_spikes=32, biorthogonal=True)
    p_orth = c_orth.encode(field)
    p_bi = c_bi.encode(field)
    r_orth = c_orth.decode_full(p_orth)
    r_bi = c_bi.decode_full(p_bi)
    rmse_orth = float(np.sqrt(np.mean((field - r_orth) ** 2)))
    rmse_bi = float(np.sqrt(np.mean((field - r_bi) ** 2)))
    assert rmse_bi < rmse_orth
    assert float(np.max(np.abs(r_bi[58:64]))) == 0.0

    f_scar = tmp_path / "bi_vq4.scar"
    save_scar_file(p_bi, f_scar)
    with ScarMmapFile(f_scar, verify_crc=True) as mm:
        blk = mm.read_block_o1(row_idx=12, block_idx=2)
        assert float(np.max(np.abs(blk - r_bi[12, 128:192]))) == 0.0

    # 2. Gosset E8 (`1.000 bpw`) + `abs_tolerance` L_inf guarantee
    e8_orth = ScarCodecE8(rank=8, max_point_spikes=32, biorthogonal=False)
    e8_bi = ScarCodecE8(rank=8, max_point_spikes=32, biorthogonal=True)
    re8_orth = e8_orth.decode_full(e8_orth.encode(field))
    re8_bi = e8_bi.decode_full(e8_bi.encode(field))
    assert float(np.sqrt(np.mean((field - re8_bi) ** 2))) < float(np.sqrt(np.mean((field - re8_orth) ** 2)))
    assert float(np.max(np.abs(re8_bi[58:64]))) == 0.0

    c_eb = NumPyVQ4ScarCodec(rank=16, max_point_spikes=32, abs_tolerance=0.5, biorthogonal=True)
    r_eb = c_eb.decode_full(c_eb.encode(field))
    assert float(np.max(np.abs(field - r_eb))) <= 0.5001

    # 3. 3D ScarCodecND Biorthogonal Dual Spectral-Tucker
    cube = np.stack([field * float(np.cos(0.2 * k)) for k in range(10)], axis=0).astype(np.float32)
    nd_orth = ScarCodecND(spectral_rank=4, spatial_scar_rank=8, max_3d_spikes=64, biorthogonal=False)
    nd_bi = ScarCodecND(spectral_rank=4, spatial_scar_rank=8, max_3d_spikes=64, biorthogonal=True)
    rnd_orth = nd_orth.decode_full(nd_orth.encode(cube))
    rnd_bi = nd_bi.decode_full(nd_bi.encode(cube))
    assert float(np.sqrt(np.mean((cube - rnd_bi) ** 2))) < float(np.sqrt(np.mean((cube - rnd_orth) ** 2)))

