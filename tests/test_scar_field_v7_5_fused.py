"""Forensic Verification Suite for `ScarField v7.5` (Zero-Autograd Symplectic Adjoint, Triton Kernels & Anti-Stutter Closure)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from scar_field import (
    E8SphericalLinear,
    ReversibleSymplecticCayleyFlow,
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    build_scout_g0_step_mask,
    cayley_woodbury_step,
    conway_sloane_e8_cuda_fast,
    encode_sft_dialogue_bytes_and_mask,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
    tsallis15_spherical_cap_probs,
    tsallis15_spherical_cap_probs_fast,
)


def test_v7_5_conway_sloane_e8_and_tsallis15_exact_parity() -> None:
    """Verifies `0` code drift vs C99 `scar_conway_sloane_e8_encode_f32` and `< 2e-6` Tsallis-1.5 parity."""
    torch.manual_seed(7501)
    op = E8SphericalLinear(in_features=128, out_features=256, scar_rank=8)
    # Cover vacuum (near-zero), mid-shell E8 roots, and outer Hadamard shell (>3.5)
    x_test = torch.randn(256, 128) * 1.45
    x_test[:16].mul_(0.02)
    x_test[16:48].mul_(2.8)

    c_ref, deq_ref, _ = op._assign_e8_codes_and_dequant(x_test, need_codes=True, need_dequant=True)
    c_fast, deq_fast, deq_bf16 = conway_sloane_e8_cuda_fast(
        x_test, op.lut_e8, need_codes=True, need_dequant_f32=True, need_dequant_bf16=True
    )
    assert c_ref is not None and c_fast is not None
    assert deq_ref is not None and deq_fast is not None and deq_bf16 is not None
    assert int((c_ref != c_fast).sum().item()) == 0
    assert float((deq_ref - deq_fast).abs().max().item()) == 0.0
    assert deq_bf16.dtype == torch.bfloat16

    # Tsallis-1.5 zero-sort Newton + support polish vs exact sorted solver across 4 orders of magnitude
    for scale in (0.01, 0.25, 2.5, 25.0):
        z_raw = torch.randn(32, 256) * scale
        z_req = z_raw.clone().requires_grad_(True)
        p_sort, lam_sort = tsallis15_spherical_cap_probs(z_req)
        p_fast, lam_fast = tsallis15_spherical_cap_probs_fast(z_raw)
        assert float((p_sort.detach() - p_fast).abs().max().item()) < 2e-6
        assert float((lam_sort.detach() - lam_fast).abs().max().item()) < 2e-5
        assert float((p_fast.sum(dim=-1) - 1.0).abs().max().item()) < 1e-6


def test_v7_5_zero_autograd_symplectic_flow_masked_gradients() -> None:
    """Verifies `cos > 0.999999` across ALL 23 tensors of `ReversibleSymplecticCayleyFlow` with mixed `Gate G0/S1/S2` mask."""
    torch.manual_seed(7502)
    flow = ReversibleSymplecticCayleyFlow(d_form=64, d_world=64, d_bus=64, base_rank=4, num_scars=16)
    hf0 = project_rows_to_unit_sphere(torch.randn(3, 8, 64)).requires_grad_(True)
    hw0 = project_rows_to_unit_sphere(torch.randn(3, 8, 64)).requires_grad_(True)
    bus0 = project_rows_to_unit_sphere(torch.randn(3, 8, 64)).requires_grad_(True)

    # Mix Gate G0 (`0.10 < 0.35`), S1 (`0.40`), and S2 (`0.85 >= 0.60`) tokens
    surp = torch.tensor(
        [
            [0.10, 0.40, 0.85, 0.12, 0.72, 0.38, 0.05, 0.91],
            [0.80, 0.15, 0.42, 0.88, 0.11, 0.44, 0.65, 0.20],
            [0.39, 0.77, 0.08, 0.41, 0.93, 0.14, 0.50, 0.68],
        ],
        dtype=torch.float32,
    )
    k_steps = 4
    sm = build_scout_g0_step_mask(surp, s2_steps=k_steps, s2_surprise_theta=0.60, scout_g0_theta=0.35, allow_full_bypass=True)

    hf1, hw1 = flow(hf0, hw0, bus0, num_steps=k_steps, step_mask=sm)
    probe_f = torch.randn_like(hf1)
    probe_w = torch.randn_like(hw1)
    ((hf1 * probe_f).sum() + (hw1 * probe_w).sum()).backward()
    fused_grads = [hf0.grad.clone(), hw0.grad.clone(), bus0.grad.clone()] + [
        p.grad.clone() for p in flow.parameters()
    ]

    flow.zero_grad(set_to_none=True)
    hfu = hf0.detach().requires_grad_(True)
    hwu = hw0.detach().requires_grad_(True)
    busu = bus0.detach().requires_grad_(True)
    hfc, hwc = hfu, hwu
    dt_base = 1.0 / float(k_steps)
    for k in range(k_steps):
        dt_k = sm[..., k : k + 1] * dt_base
        uf, vf = flow.gen_form(hwc, busu, k * dt_base)
        hfc = cayley_woodbury_step(hfc, uf, vf, dt_k)
        uw, vw = flow.gen_world(hfc, busu, k * dt_base)
        hwc = cayley_woodbury_step(hwc, uw, vw, dt_k)
    ((hfc * probe_f).sum() + (hwc * probe_w).sum()).backward()
    ref_grads = [hfu.grad, hwu.grad, busu.grad] + [p.grad for p in flow.parameters()]

    for idx, (fg, rg) in enumerate(zip(fused_grads, ref_grads)):
        assert rg is not None
        if rg.numel() == 1:
            rel = abs(float(fg.item()) - float(rg.item())) / max(1e-6, abs(float(rg.item())))
            assert rel < 1e-4, f"Scalar param {idx} mismatch: {fg.item()} vs {rg.item()}"
        else:
            cos = float(F.cosine_similarity(fg.flatten(), rg.flatten(), dim=0).item())
            assert cos > 0.999999, f"Tensor {idx} cosine {cos} < 0.999999"


def test_v7_5_shock_closure_and_repeat_phase_streaming_parity() -> None:
    """Verifies `shock_closure_decay` + `enable_repeat_phase` streaming parity (`cos > 0.999999`, `Delta = +0 B`)."""
    torch.manual_seed(7503)
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
        wave_lags=(1, 2, 3, 4, 6, 8),
        num_macro_cycles=2,
        biorth_slots=16,
        scout_g0_theta=0.40,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
    )
    eng = ScarFieldEngine(cfg)
    eng.eval()

    # 1. Repeat-phase separates consecutive identical digits (`PIN=80` vs `PIN=800` vs `PIN=8000`)
    seq_rep = torch.tensor([[b for b in b"PIN=8000."]], dtype=torch.long)
    with torch.no_grad():
        hf_rep, _, surp_rep = eng.frontend(seq_rep)
        cos_0_vs_00 = float(F.cosine_similarity(hf_rep[0, 5], hf_rep[0, 6], dim=0).item())
        cos_00_vs_000 = float(F.cosine_similarity(hf_rep[0, 6], hf_rep[0, 7], dim=0).item())
        assert cos_0_vs_00 < 0.96, f"Expected distinct keys for '=80' vs '=800', got cos={cos_0_vs_00}"
        assert cos_00_vs_000 < 0.97, f"Expected distinct keys for '=800' vs '=8000', got cos={cos_00_vs_000}"
        # Closing '.' at index 8 inherits shock closure >= 0.85 * surprise('0')
        assert float(surp_rep[0, 8].item()) >= 0.84 * float(surp_rep[0, 7].item())

    # 2. Exact chunk-aligned AND 1-byte streaming parity (`Delta = +0 B`)!
    msg = b"\x02User: Grave PIN=800.\n\x03Assistant: Gravado PIN=800.\x04\n"
    seq = torch.tensor([[b for b in msg]], dtype=torch.long)
    with torch.no_grad():
        out_full = eng.forward_sequence(seq, adaptive_depth=True)
        st_c = eng.init_state(batch_size=1)
        b0 = st_c.total_state_bytes
        c1 = eng.forward_sequence(seq[:, :32], state=st_c, adaptive_depth=True)
        c2 = eng.forward_sequence(seq[:, 32:], state=c1["state"], adaptive_depth=True)
        cat_c = torch.cat((c1["scores"], c2["scores"]), dim=1)
        assert c2["state"].total_state_bytes - b0 == 0
        cos_chunk = float(F.cosine_similarity(out_full["scores"].flatten(), cat_c.flatten(), dim=0).item())
        assert cos_chunk > 0.999999, f"Chunk parity cosine {cos_chunk} < 0.999999"

        eng.cfg.wy_chunk_size = 1
        out_1b_ref = eng.forward_sequence(seq, adaptive_depth=True)
        st_1b = eng.init_state(batch_size=1)
        step_list = []
        for t in range(seq.shape[1]):
            o_t = eng.forward_sequence(seq[:, t : t + 1], state=st_1b, adaptive_depth=True)
            st_1b = o_t["state"]
            step_list.append(o_t["scores"])
        cat_1b = torch.cat(step_list, dim=1)
        assert st_1b.total_state_bytes - b0 == 0
        cos_1b = float(F.cosine_similarity(out_1b_ref["scores"].flatten(), cat_1b.flatten(), dim=0).item())
        assert cos_1b > 0.999999, f"Single-byte parity cosine {cos_1b} < 0.999999"


def test_v7_5_open_chat_repeated_digit_pin_closure_and_v7_4_mmap_compat() -> None:
    """Trains open `generate_reply` on unseen repeated-digit PINs (`800`, `777`) with `0` stuttering + checks `v7.4` mmap."""
    torch.manual_seed(7504)
    cfg = ScarFieldConfig(
        d_form=64,
        d_world=64,
        d_bus=64,
        d_key=64,
        num_subspaces=8,
        ring_len=12,
        repulsive_len=8,
        base_rank=4,
        num_scars=16,
        s1_steps=1,
        s2_steps=3,
        shock_write_theta=0.45,
        biorth_slots=12,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
    )
    model = ScarFieldEngine(cfg)
    opt = RiemannianMuon(model.parameters(), lr=2.4e-2, scalar_lr=8e-3)
    rng_g = torch.Generator().manual_seed(7504)

    seq_len = 88
    for step in range(1, 151):
        pins = torch.randint(100, 750, (16,), generator=rng_g).tolist()
        rows_b = []
        rows_m = []
        for pin in pins:
            turns = [
                (f"PIN={pin:03d} ok?", "OK."),
                ("Qual PIN?", f"PIN={pin:03d}."),
            ]
            arr_u8, mask_f32 = encode_sft_dialogue_bytes_and_mask(turns)
            pad_n = max(0, seq_len - len(arr_u8))
            b_row = list(arr_u8[:seq_len]) + [0] * pad_n
            m_row = list(mask_f32[: seq_len - 1]) + [0.0] * max(0, (seq_len - 1) - len(mask_f32))
            # Boost EOT & closing '.' salience in SFT mask
            for idx_t in range(len(m_row)):
                if m_row[idx_t] > 0.0 and b_row[idx_t + 1] in (4, ord(".")):
                    m_row[idx_t] = 2.5
            rows_b.append(b_row)
            rows_m.append(m_row)

        batch_b = torch.tensor(rows_b, dtype=torch.long)
        batch_m = torch.tensor(rows_m, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        model.compute_loss(batch_b, target_loss_mask=batch_m)["loss"].backward()
        opt.step()
        if step % 50 == 0:
            heal_all_e8_scars(model)

    model.eval()
    unseen_test_pins = [800, 777, 880, 900, 811, 955, 799, 808]
    hits = 0
    for pin in unseen_test_pins:
        r1 = model.generate_reply(f"PIN={pin:03d} ok?", max_new_bytes=24)
        r2 = model.generate_reply("Qual PIN?", state=r1.state, max_new_bytes=32)
        assert r2.state_bytes - r1.state_bytes == 0
        assert r2.bytes_generated <= 16, f"Stuttering detected on PIN={pin}: {r2.text!r} ({r2.bytes_generated} B)"
        if f"PIN={pin:03d}." in r2.text:
            hits += 1

    assert hits >= 7, f"Expected >= 7/8 crisp unseen repeated-digit PIN closures, got {hits}/8"

    # Backward compatibility check: `v7.4` `.sfield.scar` cold-loads cleanly under `v7.5`!
    v74_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "scarfield_v7_4_turbo_scouts"
        / "scarfield_v7_4_turbo_scouts.sfield.scar"
    )
    if v74_path.exists():
        eng_v74, mm_v74 = ScarFieldEngine.load_packed_scar_mmap(v74_path)
        assert eng_v74.cfg.shock_closure_decay == 0.0
        assert eng_v74.cfg.enable_repeat_phase is False
        mm_v74.close()
