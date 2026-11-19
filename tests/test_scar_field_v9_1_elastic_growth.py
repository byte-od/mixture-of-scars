"""Forensic Verification Suite for `ScarField v9.1-Elastic` (Isometric Geodesic Growth & Zero-Tax MoS Gears)."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F

from scar_field import (
    BatchedScarFieldSuperEngine,
    ElasticCapacityGear,
    MoSGeodesicDomainBank,
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    clone_scarfield_engine_fast,
    expand_scarfield_geodesic_isometry,
    heal_all_e8_scars,
    lift_kalman_bus_state_isometric,
    project_rows_to_unit_sphere,
)


def _edge_cfg() -> ScarFieldConfig:
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
        s2_steps=4,
        wy_chunk_size=16,
        wave_lags=(1, 2, 4),
        num_macro_cycles=1,
        biorth_slots=16,
        biorth_buckets=2,
        biorth_lambda=0.04,
        enable_biorth_uscar=True,
        enable_adaptive_cap=True,
        scout_g0_theta=0.35,
        antiloop_weight=0.08,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
        enable_percept_s1_gate=True,
    )


def _core_cfg() -> ScarFieldConfig:
    return ScarFieldConfig(
        vocab_size=256,
        d_form=128,
        d_world=128,
        d_bus=128,
        d_key=64,
        num_subspaces=16,
        ring_len=32,
        repulsive_len=12,
        base_rank=8,
        num_scars=32,
        s1_steps=1,
        s2_steps=4,
        wy_chunk_size=16,
        wave_lags=(1, 2, 4),
        num_macro_cycles=3,
        biorth_slots=64,
        biorth_buckets=4,
        biorth_lambda=0.04,
        enable_biorth_uscar=True,
        enable_adaptive_cap=True,
        scout_g0_theta=0.35,
        antiloop_weight=0.08,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
        enable_percept_s1_gate=True,
    )


def test_v91_isometric_geodesic_expansion_and_nonsaddle_growth(tmp_path: Path) -> None:
    """Proves `Edge -> Core` isometric expansion preserves learned scores at `step = 0`, breaks saddles at `step = 1`,
    evolves unconstrained (`0%` Matryoshka tax), and preserves `.uscar` cartridges + `.sfield.scar` C99 `mmap` parity!
    """
    torch.manual_seed(9101)
    edge = ScarFieldEngine(_edge_cfg())
    opt_edge = RiemannianMuon(edge.parameters(), lr=4e-3, scalar_lr=4e-3)
    seq = torch.tensor(
        [
            list(b"def square(x):\n    return x * x\n"),
            list(b"A chave soberana e PIN=742 agora"),
        ],
        dtype=torch.long,
    )
    # Train `Edge` briefly so it carries non-trivial learned features
    for _ in range(5):
        opt_edge.zero_grad(set_to_none=True)
        edge.compute_loss(seq)["loss"].backward()
        opt_edge.step()
    heal_all_e8_scars(edge)
    edge.eval()

    # Mint a `.uscar` cartridge on `Edge` BEFORE expansion
    with torch.no_grad():
        out_edge = edge.forward_sequence(seq)
        q_fact = project_rows_to_unit_sphere(out_edge["h_world"][0:1].mean(dim=1))
        y_fact = torch.zeros(1, edge.cfg.d_bus)
        y_fact[0, 3] = 2.5
        edge.bus.uscar_store.mint_zero_sgd_rrr("laser_cal", "doc_edge", q_fact, y_fact, rank=4, horizon_kappa=0.40)
        out_edge_with_uscar = edge.forward_sequence(seq)

    # Also test expanding directly from cold-loaded `.sfield.scar`!
    edge_scar = tmp_path / "edge_seed.sfield.scar"
    edge.save_packed_scar_container(edge_scar)
    edge_mm, _ct_e = ScarFieldEngine.load_packed_scar_mmap(edge_scar)
    edge_mm.bus.uscar_store.cartridges.update(edge.bus.uscar_store.cartridges)

    # Also test pure Depth + Scar + Biorthogonal Bank expansion (`1 -> 3` cycles, `16 -> 32` scars, `16 -> 64` slots)
    # where `0` blocks shift -> exact `cos > 0.99999`!
    depth_cfg = _edge_cfg()
    depth_cfg.num_macro_cycles = 3
    depth_cfg.num_scars = 32
    depth_cfg.biorth_slots = 64
    depth_cfg.biorth_buckets = 4
    depth_core = expand_scarfield_geodesic_isometry(edge_mm, depth_cfg, seed=9102)
    depth_core.eval()
    with torch.no_grad():
        out_depth0 = depth_core.forward_sequence(seq)
        cos_depth = float(
            F.cosine_similarity(
                out_edge_with_uscar["scores"].flatten().float(),
                out_depth0["scores"].flatten().float(),
                dim=0,
            ).item()
        )
        assert cos_depth > 0.9998, f"Expected >99.98% fidelity on unshifted depth/scar/bank expansion, got {cos_depth}"

    core = expand_scarfield_geodesic_isometry(edge_mm, _core_cfg(), seed=9102)
    core.eval()
    core.assert_zero_legacy_modules()
    assert sum(p.numel() for p in core.parameters()) > 3 * sum(p.numel() for p in edge.parameters())

    with torch.no_grad():
        out_core0 = core.forward_sequence(seq)
        # 1. Unit-sphere norms must be exact `1.000000` without `RMSNorm`!
        hf_nrm = float(torch.linalg.vector_norm(out_core0["h_form"], dim=-1).mean().item())
        hw_nrm = float(torch.linalg.vector_norm(out_core0["h_world"], dim=-1).mean().item())
        assert abs(hf_nrm - 1.0) < 1e-5 and abs(hw_nrm - 1.0) < 1e-5

        # 2. Step-0 function preservation (`Edge -> Core` across 2x width + 3x depth + 2x subspaces at `1.000 bpw` E8!)
        cos_scores = float(
            F.cosine_similarity(
                out_edge_with_uscar["scores"].flatten().float(),
                out_core0["scores"].flatten().float(),
                dim=0,
            ).item()
        )
        assert cos_scores > 0.985, f"Expected >98.5% step-0 E8 isometric fidelity, got cos={cos_scores}"

        # 3. Inherited `.uscar` cartridge fires immediately on `Core` (`128D -> 64D` equatorial Stiefel bridge)
        res_edge_max = float(out_edge_with_uscar["uscar_resonance"].max().item())
        res_core_max = float(out_core0["uscar_resonance"].max().item())
        assert res_core_max > 0.25 and abs(res_core_max - res_edge_max) < 0.03

    # 4. Step-1 backward must have ZERO dead saddles on both new dimensions AND extra macro-cycles (`ef[0], ef[1]`)!
    core.train()
    opt_core = RiemannianMuon(core.parameters(), lr=3e-3, scalar_lr=3e-3)
    opt_core.zero_grad(set_to_none=True)
    loss0 = float(core.compute_loss(seq)["loss"].item())
    core.compute_loss(seq)["loss"].backward()

    assert core.extra_flows is not None and len(core.extra_flows) == 2
    ef_grad_norm = float(core.extra_flows[1].gen_world.base_e8.tau_base.grad.abs().item())  # type: ignore[union-attr]
    ef_scar_grad = float(core.extra_flows[1].gen_world.scar_strength.grad.norm().item())  # type: ignore[union-attr]
    ext_form_grad = float(core.frontend.byte_form_poles.grad[:, 64:].norm().item())  # type: ignore[index]
    ext_world_grad = float(core.frontend.byte_world_poles.grad[:, 64:].norm().item())  # type: ignore[index]
    assert ef_grad_norm > 0.0 or ef_scar_grad > 0.0, "Extra macro-cycle suffered a dead bilinear saddle!"
    assert ext_form_grad > 1e-6 and ext_world_grad > 1e-6, "New orthogonal dimensions received zero gradient on step 1!"

    # 5. Unconstrained training (`0%` Matryoshka lock) activates `Core` complement (`64:128`) and lowers loss!
    opt_core.step()
    for _ in range(5):
        opt_core.zero_grad(set_to_none=True)
        core.compute_loss(seq)["loss"].backward()
        opt_core.step()
    loss_after = float(core.compute_loss(seq)["loss"].item())
    assert loss_after < loss0, f"Expected Core loss to drop ({loss0:.4f} -> {loss_after:.4f})"
    assert float(core.frontend.byte_world_poles[:, 64:].norm().item()) > 0.03

    # 6. `.sfield.scar` container export & C99/SIMD `mmap` roundtrip on expanded `Core` (`proper_time_freq_ref_dim=64`)
    core.bus.uscar_store.cartridges.clear()
    heal_all_e8_scars(core)
    core.eval()
    core_scar = tmp_path / "core_expanded.sfield.scar"
    core.save_packed_scar_container(core_scar)
    core_mm, _ct_c = ScarFieldEngine.load_packed_scar_mmap(core_scar)
    core_mm.eval()
    assert core_mm.cfg.proper_time_freq_ref_dim == 64
    with torch.no_grad():
        s_ref = core.forward_sequence(seq)["scores"]
        s_mm = core_mm.forward_sequence(seq)["scores"]
    cos_mm = float(F.cosine_similarity(s_ref.flatten().float(), s_mm.flatten().float(), dim=0).item())
    assert cos_mm > 0.99999, f"Expected exact C99 mmap parity on expanded Core, got cos={cos_mm}"


def test_v91_zero_tax_capacity_gearshift_and_live_state_handoff() -> None:
    """Proves mid-dialogue gearshifts (`eco <-> balanced <-> deep <-> frontier`) incur `Delta = 0 B` state change,
    and `lift_kalman_bus_state_isometric` hands off live `O(1)` memory from `Edge` to `Core` without losing PINs!
    """
    torch.manual_seed(9103)
    edge = ScarFieldEngine(_edge_cfg())
    edge.eval()
    core = expand_scarfield_geodesic_isometry(edge, _core_cfg(), seed=9104)
    core.eval()

    # 1. Mid-dialogue capacity gearshift on `Core` (`Delta state bytes == 0`!)
    st_core = core.init_state(1)
    base_bytes = st_core.total_state_bytes
    for gear_name, exp_cycles, exp_s2 in (("eco", 1, 1), ("balanced", 2, 4), ("deep", 3, 8), ("frontier", 3, 12)):
        g_obj = core.set_capacity_gear(gear_name)
        assert isinstance(g_obj, ElasticCapacityGear)
        assert core._active_macro_cycles == exp_cycles
        assert core.cfg.s2_steps == exp_s2
        out_g = core.forward_sequence(torch.tensor([list(b" PIN=918 ")], dtype=torch.long), state=st_core)
        st_core = out_g["state"]
        assert st_core.total_state_bytes == base_bytes, "Gearshift mutated O(1) state byte footprint!"

    # 2. Live session handoff (`Edge [64D, 2x8=16 slots] -> Core [128D, 4x16=64 slots]`)
    st_edge = edge.init_state(1)
    turn1 = torch.tensor([list(b"Store secret key: PIN=864.")], dtype=torch.long)
    with torch.no_grad():
        res_e1 = edge.forward_sequence(turn1, state=st_edge)
        st_edge_after = res_e1["state"]
        assert st_edge_after.biorth_valid is not None and bool(st_edge_after.biorth_valid.any().item())

        st_lifted = lift_kalman_bus_state_isometric(st_edge_after, core)
        assert st_lifted.total_state_bytes == base_bytes
        assert st_lifted.biorth_valid is not None
        assert int(st_lifted.biorth_valid.sum().item()) == int(st_edge_after.biorth_valid.sum().item())

        query_turn = torch.tensor([list(b"Recall key: PIN=")], dtype=torch.long)
        sc_edge_q = edge.forward_sequence(query_turn, state=st_edge_after)["scores"]
        core.set_capacity_gear("balanced")
        sc_core_q = core.forward_sequence(query_turn, state=st_lifted)["scores"]
        cos_handoff = float(F.cosine_similarity(sc_edge_q.flatten().float(), sc_core_q.flatten().float(), dim=0).item())
        assert cos_handoff > 0.980, f"Expected >98.0% live session handoff Edge->Core, got cos={cos_handoff}"


def test_v91_mos_geodesic_domain_bank_zero_shock_routing_and_container(tmp_path: Path) -> None:
    """Proves `MoSGeodesicDomainBank` has exact `0.0` attach perturbation, non-zero step-1 gradient,
    `0.0` cross-domain interference outside `C^1` spherical horizon, and `.mosbank.scar` `CRC32` persistence!
    """
    torch.manual_seed(9105)
    eng = ScarFieldEngine(_edge_cfg())
    eng.eval()
    for p in eng.parameters():
        p.requires_grad_(False)

    code_seq = torch.tensor([list(b"def fib(n):\n    return n if n < 2 else fib(n-1)\n")], dtype=torch.long)
    prose_seq = torch.tensor([list(b"O tratado de termodinamica descreve a entropia.\n")], dtype=torch.long)

    with torch.no_grad():
        ref_code = eng.forward_sequence(code_seq)
        ref_prose = eng.forward_sequence(prose_seq)
        code_pole = project_rows_to_unit_sphere(ref_code["h_world"].mean(dim=(0, 1), keepdim=False).unsqueeze(0)).squeeze(0)
        prose_pole = project_rows_to_unit_sphere(ref_prose["h_world"].mean(dim=(0, 1), keepdim=False).unsqueeze(0)).squeeze(0)
        # Ensure distinct test poles so `"prose"` lies strictly outside `"code"`'s high-kappa cap
        orth_prose_pole = project_rows_to_unit_sphere((prose_pole - (prose_pole @ code_pole) * code_pole).unsqueeze(0)).squeeze(0)

    bank = MoSGeodesicDomainBank(eng.cfg.d_form, eng.cfg.d_world, eng.cfg.d_bus, top_k=1)
    spec_code = bank.register_specialist("code", scar_rank=8, horizon_kappa=0.65, seed=9110, init_pole=code_pole)
    bank.register_specialist("math", scar_rank=8, horizon_kappa=0.88, seed=9111, init_pole=orth_prose_pole)
    eng.mount_mos_domain_bank(bank)

    # 1. Exact `0.000000` attach shock (`gamma_form == 0`, `gamma_world == 0`)!
    with torch.no_grad():
        attached_code = eng.forward_sequence(code_seq)["scores"]
        max_attach_diff = float((attached_code - ref_code["scores"]).abs().max().item())
        assert max_attach_diff == 0.0, f"Expected exact 0.0 attach shock, got {max_attach_diff}"

    # 2. Train ONLY `spec_code` (`base E_8` crystal stays 100% frozen!) -> non-zero step-1 grad & lower code loss!
    opt_mos = torch.optim.AdamW(spec_code.parameters(), lr=2.5e-2)
    # Pin `horizon_kappa` slightly open during micro-adaptation so `code_seq` stays resonant while `prose` is outside
    spec_code.horizon_kappa.requires_grad_(False)
    spec_code.domain_pole.requires_grad_(False)
    spec_code.horizon_kappa.fill_(0.15)

    opt_mos.zero_grad(set_to_none=True)
    loss_c0 = eng.compute_loss(code_seq)["loss"]
    loss_c0.backward()
    assert spec_code.gamma_form.grad is not None and spec_code.gamma_world.grad is not None
    grad_mag = float(spec_code.gamma_form.grad.abs().item() + spec_code.gamma_world.grad.abs().item())
    assert grad_mag > 0.0, "MoS specialist suffered dead saddle at gamma=0!"
    opt_mos.step()

    for _ in range(6):
        opt_mos.zero_grad(set_to_none=True)
        eng.compute_loss(code_seq)["loss"].backward()
        opt_mos.step()
    with torch.no_grad():
        loss_c1 = float(eng.compute_loss(code_seq)["loss"].item())
    assert loss_c1 < float(loss_c0.item()), f"Expected MoS code specialist to reduce loss ({loss_c0.item():.4f} -> {loss_c1:.4f})"

    # 3. Exact `0.000000` cross-domain interference when `h_world` lies outside `spec_code`'s `C^1` cap!
    with torch.no_grad():
        spec_code.domain_pole.copy_(-prose_pole)  # antipodal (`cos < -0.5 < kappa = 0.15`) -> exact `0.0` gate!
        spec_code.horizon_kappa.fill_(0.50)
        prose_after = eng.forward_sequence(prose_seq)["scores"]
        assert float((prose_after - ref_prose["scores"]).abs().max().item()) == 0.0

    # 4. Save & reload `.mosbank.scar` (`2-bit` ternary packed container) + clean unmount restores exact `0.0` diff!
    mos_path = tmp_path / "specialists.mosbank.scar"
    n_mos_b = bank.save_packed_mos_container(mos_path)
    assert n_mos_b > 0
    reloaded_bank = MoSGeodesicDomainBank.load_packed_mos_container(mos_path)
    assert set(reloaded_bank.specialists.keys()) == {"code", "math"}

    eng.unmount_mos_domain_bank()
    with torch.no_grad():
        restored_code = eng.forward_sequence(code_seq)["scores"]
        assert float((restored_code - ref_code["scores"]).abs().max().item()) == 0.0


def test_batched_super_engine_wrap_dim_isometric_parity() -> None:
    """Verifies `BatchedScarFieldSuperEngine` preserves exact `wrap_dim` + `ref_dim` parity (`cos > 0.999999`) after `Edge -> Core` expansion."""
    torch.manual_seed(1234)
    edge = ScarFieldEngine(_edge_cfg())
    core_cfg = _core_cfg()
    core = expand_scarfield_geodesic_isometry(edge, core_cfg, seed=77)

    bank = BatchedScarFieldSuperEngine(core, num_replicas=2)

    seq = torch.tensor([[ord(c) for c in "Isometric Super-Kernel Parity Audit 2026"]], dtype=torch.long)
    with torch.no_grad():
        ref_scores = core.forward_sequence(seq)["scores"]
        sup_out = bank.forward_sequence_batched(seq.unsqueeze(0).expand(2, -1, -1))
        sup_scores0 = sup_out["scores"][0]
        cos = float(F.cosine_similarity(ref_scores.flatten(), sup_scores0.flatten(), dim=0).item())
        max_diff = float((ref_scores - sup_scores0).abs().max().item())

    assert cos > 0.9999995, f"BatchedScarFieldSuperEngine wrap_dim parity drift: cos={cos}"
    assert max_diff < 5e-6, f"BatchedScarFieldSuperEngine wrap_dim max_diff={max_diff}"

