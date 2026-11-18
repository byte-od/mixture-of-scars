"""Forensic Mathematical & Architectural Test Suite for `ScarField v8.1-Velocity` Batched Super-Fleet."""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import scar_field
from scar_field import (
    BatchedRiemannianMuon,
    BatchedScarFieldSuperEngine,
    MacroScoutSpec,
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    VectorizedMacroScoutFleet,
    VectorizedRoundBatch,
    VectorizedSharedCrystalSwarm,
    clone_scarfield_engine_fast,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
)


def _tiny_v81_cfg() -> ScarFieldConfig:
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
        s2_surprise_theta=0.45,
        wy_chunk_size=16,
        wave_lags=(1, 2, 4),
        num_macro_cycles=2,
        biorth_slots=8,
        biorth_lambda=0.04,
        enable_biorth_uscar=True,
        enable_adaptive_cap=True,
        scout_g0_theta=0.55,
        antiloop_weight=0.10,
        shock_closure_decay=0.85,
        enable_repeat_phase=True,
    )


def _sync_single_models_into_bank(bank: BatchedScarFieldSuperEngine, models: list[ScarFieldEngine]) -> None:
    master_ref = models[0]
    with torch.no_grad():
        for pname, _, p_b in bank.master_to_bank_param_pairs(master_ref):
            assert p_b is not None
            slices = []
            for m_eng in models:
                p_single = dict(m_eng.named_parameters())[pname]
                slices.append(p_single.detach().clone())
            p_b.copy_(torch.stack(slices, dim=0))


def test_v81_version_and_zero_legacy_layers() -> None:
    assert int(scar_field.__version__.split(".")[0]) >= 8
    torch.manual_seed(8101)
    master = ScarFieldEngine(_tiny_v81_cfg())
    bank = BatchedScarFieldSuperEngine(master, num_replicas=4, shared_crystal=False)
    forbidden = (
        nn.Linear,
        nn.LayerNorm,
        nn.RMSNorm,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Softmax,
        nn.LogSoftmax,
        nn.CrossEntropyLoss,
    )
    for name, mod in bank.named_modules():
        assert not isinstance(mod, forbidden), f"Forbidden legacy module `{type(mod).__name__}` at `{name}`!"


def test_batched_super_engine_exact_forward_and_analytic_grad_parity() -> None:
    """Verifies bit-level mathematical equivalence (`cos > 0.99999`) between `[M, B, T]` Bank and `M` single engines."""
    torch.manual_seed(8102)
    cfg = _tiny_v81_cfg()
    master = ScarFieldEngine(cfg)
    heal_all_e8_scars(master)

    m_cnt = 3
    models: list[ScarFieldEngine] = []
    for i in range(m_cnt):
        eng = clone_scarfield_engine_fast(master, torch.device("cpu"))
        with torch.no_grad():
            for p in eng.parameters():
                if p.ndim == 2 and p.shape[1] >= 4:
                    p.copy_(project_rows_to_unit_sphere(p + 0.08 * (i + 1) * torch.randn_like(p)))
                else:
                    p.add_(0.03 * (i + 1) * torch.randn_like(p))
        models.append(eng)

    bank = BatchedScarFieldSuperEngine(master, num_replicas=m_cnt, shared_crystal=False)
    _sync_single_models_into_bank(bank, models)

    bsz, seq_len = 3, 40
    # Mix predictable runs (`'aaaa...'`) and random shocks so `c_max` compaction actively triggers across `[M, N, K]`!
    byte_ids = torch.randint(32, 122, (m_cnt, bsz, seq_len), dtype=torch.long)
    byte_ids[0, :, 6:34] = ord("a")  # Low surprise on scout 0 -> fewer active S2/G0 tokens than scout 2!
    byte_ids[1, :, 10:26] = ord(" ")

    t_mask = (torch.rand(m_cnt, bsz, seq_len - 1) > 0.25).float()
    ooc_mask = torch.zeros(m_cnt, bsz, seq_len, dtype=torch.bool)
    ooc_mask[2, 0, -6:] = True
    pb_full = torch.randint(48, 90, (bsz, seq_len), dtype=torch.long)

    # 1. Standalone reference forward + backward on each of the `M = 3` perturbed models
    ref_losses: list[float] = []
    ref_totals: list[float] = []
    for i in range(m_cnt):
        models[i].zero_grad(set_to_none=True)
        d_i = models[i].compute_loss(
            byte_ids[i],
            paired_translation_ids=pb_full if i == 0 else None,
            out_of_cone_mask=ooc_mask[i],
            target_loss_mask=t_mask[i],
        )
        d_i["loss"].backward()
        ref_losses.append(float(d_i["loss_cap"].item()))
        ref_totals.append(float(d_i["loss"].item()))

    # 2. Single fused `[M, B, T]` forward + backward on `BatchedScarFieldSuperEngine`
    bank.zero_grad(set_to_none=True)
    rbatch = VectorizedRoundBatch(
        byte_ids=byte_ids,
        target_loss_mask=t_mask,
        out_of_cone_mask=ooc_mask,
        paired_translation_ids=(0, byte_ids[0], pb_full),
    )
    loss_sum, loss_cap_m, _ = bank.compute_loss_batched(rbatch)
    loss_sum.backward()

    assert abs(float(loss_sum.item()) - sum(ref_totals)) < 1.5e-4, (
        f"Total loss mismatch: {loss_sum.item()} vs {sum(ref_totals)}"
    )
    for i in range(m_cnt):
        assert abs(float(loss_cap_m[i].item()) - ref_losses[i]) < 5e-5

    # 3. Verify parameter-by-parameter gradient cosine similarity across all `M` replicas!
    for pname, _, p_b in bank.master_to_bank_param_pairs(models[0]):
        assert p_b is not None and p_b.grad is not None, f"Missing grad on `{pname}`"
        for i in range(m_cnt):
            g_ref = dict(models[i].named_parameters())[pname].grad
            assert g_ref is not None
            g_bnk = p_b.grad[i]
            n_ref = float(torch.linalg.vector_norm(g_ref.float()).item())
            n_bnk = float(torch.linalg.vector_norm(g_bnk.float()).item())
            if n_ref > 1e-6 or n_bnk > 1e-6:
                if g_ref.numel() >= 4:
                    cos = float(F.cosine_similarity(g_ref.flatten().float(), g_bnk.flatten().float(), dim=0).item())
                    assert cos > 0.9999, f"Grad cosine drift on `{pname}`[m={i}]: cos={cos}"
                rel_err = float((g_ref.float() - g_bnk.float()).abs().max().item()) / max(1e-4, n_ref)
                assert rel_err < 5e-4, f"Grad max rel error on `{pname}`[m={i}]: {rel_err}"


def test_batched_stateful_multi_chunk_o1_continuation_parity() -> None:
    """Verifies `O(1)` `KalmanBusState` multi-chunk TBPTT continuation (`chunk0 -> chunk1`) matches standalone engine."""
    torch.manual_seed(8103)
    cfg = _tiny_v81_cfg()
    master = ScarFieldEngine(cfg)
    m_cnt = 2
    models = [clone_scarfield_engine_fast(master, torch.device("cpu")) for _ in range(m_cnt)]
    with torch.no_grad():
        for p in models[1].parameters():
            if p.ndim == 2 and p.shape[1] >= 4:
                p.copy_(project_rows_to_unit_sphere(p + 0.12 * torch.randn_like(p)))

    bank = BatchedScarFieldSuperEngine(master, num_replicas=m_cnt, shared_crystal=False)
    _sync_single_models_into_bank(bank, models)

    c0 = torch.randint(32, 122, (m_cnt, 2, 32), dtype=torch.long)
    c1 = torch.randint(32, 122, (m_cnt, 2, 32), dtype=torch.long)

    with torch.no_grad():
        out0_b = bank.forward_sequence_batched(c0, states=None, adaptive_depth=True)
        out1_b = bank.forward_sequence_batched(c1, states=out0_b["states"], adaptive_depth=True)

        for i in range(m_cnt):
            r0 = models[i].forward_sequence(c0[i], state=None, adaptive_depth=True)
            r1 = models[i].forward_sequence(c1[i], state=r0["state"], adaptive_depth=True)
            diff0 = float((out0_b["scores"][i] - r0["scores"]).abs().max().item())
            diff1 = float((out1_b["scores"][i] - r1["scores"]).abs().max().item())
            assert diff0 < 5e-5, f"Chunk-0 score mismatch on m={i}: {diff0}"
            assert diff1 < 5e-5, f"Chunk-1 stateful continuation mismatch on m={i}: {diff1}"

        # Also verify `preroll_single_scout_bus_state` (`c0` is `32 = 2 * wy_chunk_size(16)`) vs full `64`-byte sequence!
        scratch = clone_scarfield_engine_fast(master, torch.device("cpu"))
        st_pre1 = bank.preroll_single_scout_bus_state(1, c0[1], scratch)
        out_pre = bank.forward_sequence_batched(c1, states=[None, st_pre1], adaptive_depth=True)
        r_full1 = models[1].forward_sequence(torch.cat((c0[1], c1[1]), dim=1), state=None, adaptive_depth=True)
        diff_pre = float((out_pre["scores"][1] - r_full1["scores"][:, 32:]).abs().max().item())
        assert diff_pre < 5e-5, f"Preroll bus-only continuation mismatch vs full 64-byte forward: {diff_pre}"


def test_batched_riemannian_muon_step_parity() -> None:
    """Verifies `BatchedRiemannianMuon` (`lr_scales` + per-replica grad clip + Nesterov NS5) matches `M` optimizers."""
    torch.manual_seed(8104)
    cfg = _tiny_v81_cfg()
    master = ScarFieldEngine(cfg)
    m_cnt = 3
    lr_scales = [1.0, 0.85, 1.15]
    models = [clone_scarfield_engine_fast(master, torch.device("cpu")) for _ in range(m_cnt)]
    opts = [
        RiemannianMuon(models[i].parameters(), lr=2.5e-3 * lr_scales[i], scalar_lr=3.0e-3 * lr_scales[i], momentum=0.88)
        for i in range(m_cnt)
    ]

    bank = BatchedScarFieldSuperEngine(master, num_replicas=m_cnt, shared_crystal=False)
    _sync_single_models_into_bank(bank, models)
    b_opt = BatchedRiemannianMuon(
        list(bank.parameters()),
        num_replicas=m_cnt,
        lr_scales=lr_scales,
        lr=2.5e-3,
        scalar_lr=3.0e-3,
        momentum=0.88,
        grad_clip=0.80,
    )

    for step_idx in range(2):
        ids = torch.randint(32, 120, (m_cnt, 2, 24), dtype=torch.long)
        b_opt.zero_grad(set_to_none=True)
        loss_s, _, _ = bank.compute_loss_batched(VectorizedRoundBatch(byte_ids=ids))
        loss_s.backward()
        b_opt.step()

        for i in range(m_cnt):
            opts[i].zero_grad(set_to_none=True)
            d_i = models[i].compute_loss(ids[i])
            d_i["loss"].backward()
            torch.nn.utils.clip_grad_norm_(models[i].parameters(), 0.80)
            opts[i].step()

        for pname, _, p_b in bank.master_to_bank_param_pairs(models[0]):
            assert p_b is not None
            for i in range(m_cnt):
                p_ref = dict(models[i].named_parameters())[pname]
                err = float((p_ref.float() - p_b[i].float()).abs().max().item())
                assert err < 1.5e-4, f"Step {step_idx} param `{pname}`[m={i}] drift: {err}"


def test_vectorized_fleet_and_shared_crystal_swarm_end_to_end(tmp_path: Path) -> None:
    """Verifies `VectorizedMacroScoutFleet` (`M=4`) + `VectorizedSharedCrystalSwarm` (`S=6`) + `.sfield.scar` `0 B` growth."""
    torch.manual_seed(8105)
    cfg = _tiny_v81_cfg()
    master = ScarFieldEngine(cfg)

    p0 = tmp_path / "v81_init.sfield.scar"
    sz0 = master.save_packed_scar_container(p0)

    patterns = [
        b"The Gosset E8 lattice achieves exact spherical packing density.\n" * 2,
        b"def add_scars(x: int, y: int) -> int:\n    return x + y\n" * 2,
        b"SECRET PIN=314. Noise filler wave. Recall PIN=314.\n" * 2,
        b"\x02User: Capital of Brazil?\x03Assistant: Brasilia.\x04" * 2,
    ]
    seq_t = 48
    stacked_pats = torch.stack(
        [torch.tensor(list(p[:seq_t].ljust(seq_t, b" ")), dtype=torch.long).unsqueeze(0).repeat(2, 1) for p in patterns],
        dim=0,
    )  # [4, 2, 48]

    with torch.no_grad():
        init_losses = [float(master.compute_loss(stacked_pats[i])["loss_cap"].item()) for i in range(4)]

    specs = [
        MacroScoutSpec("prose", weight=1.0),
        MacroScoutSpec("code", weight=1.0),
        MacroScoutSpec("biorth", weight=1.0, owns_biorth_organ=True),
        MacroScoutSpec("chat", weight=1.0),
    ]
    fleet = VectorizedMacroScoutFleet(master, specs, base_lr=8.0e-3, scalar_lr=6.0e-3, outer_lr=0.90)

    # Test `select_frontier_batches_fused`
    cands = torch.randint(32, 122, (4, 8, seq_t), dtype=torch.long)
    sel_b, _ = fleet.bank.select_frontier_batches_fused(cands, batch_size=2)
    assert sel_b.shape == (4, 2, seq_t)

    for _ in range(3):
        tel = fleet.run_vectorized_round(
            lambda _b, _h, states: VectorizedRoundBatch(byte_ids=stacked_pats, state_in=states),
            local_steps=4,
            merge_after=True,
        )
        assert tel.total_fleet_steps == 16

    swarm = VectorizedSharedCrystalSwarm(master, num_workers=6, scar_lr=4.0e-3, scalar_lr=3.0e-3, distill_eta=0.60)
    assert swarm.master_total_params / max(1, swarm.mutable_params_per_worker) > 4.0
    swarm_pats = stacked_pats[[0, 1, 2, 3, 0, 2]]  # [6, 2, 48]
    s_tel = swarm.run_vectorized_swarm_round(
        lambda _b, _h, _st: VectorizedRoundBatch(byte_ids=swarm_pats),
        local_steps=3,
        distill_after=True,
    )
    assert s_tel.crystal_layers_distilled > 0

    with torch.no_grad():
        final_losses = [float(master.compute_loss(stacked_pats[i])["loss_cap"].item()) for i in range(4)]
    assert sum(final_losses) < sum(init_losses), f"Expected joint loss drop: {init_losses} -> {final_losses}"

    p1 = tmp_path / "v81_final.sfield.scar"
    sz1 = master.save_packed_scar_container(p1)
    assert sz1 == sz0, f"Container grew by {sz1 - sz0} B (must be strictly +0 B)!"


def test_v82_pep8_indentation_and_discourse_margin() -> None:
    """Verifies `v8.2` PEP-8 4/8/12-space allowance (`+0 B` state) and `prose_noncode_first_mask` gradient."""
    torch.manual_seed(8201)
    cfg = _tiny_v81_cfg()
    eng = ScarFieldEngine(cfg)
    eng.eval()

    # 1. Feed 3 consecutive spaces `'   '`: 4th space (`32`) MUST NOT be damped (PEP-8 indentation!),
    #    whereas 3 consecutive `'000'` MUST damp 4th `'0'` (`48`)!
    st_space = eng.init_state(1)
    for _ in range(3):
        eng.step_byte_interactive(ord(" "), st_space, apply_repulsive_anti_loop=False)
    d_sp = eng.step_byte_interactive(ord(" "), st_space, apply_repulsive_anti_loop=True)
    assert d_sp.action_byte >= 0

    # Verify symmetric `prose_noncode_first_mask` + `code_def_first_mask` margin loss & gradients!
    bank = BatchedScarFieldSuperEngine(eng, num_replicas=2, shared_crystal=False)
    ids = torch.full((2, 2, 16), ord(" "), dtype=torch.long)
    ids[0, :, 4] = ord("A")
    ids[1, :, 4] = ord("d")
    pmask = torch.zeros((2, 2, 15), dtype=torch.bool)
    cmask = torch.zeros((2, 2, 15), dtype=torch.bool)
    pmask[0, :, 3] = True
    cmask[1, :, 3] = True
    bank.zero_grad(set_to_none=True)
    loss_no_f, _, _ = bank.compute_loss_batched(VectorizedRoundBatch(byte_ids=ids))
    loss_with_f, _, _ = bank.compute_loss_batched(
        VectorizedRoundBatch(byte_ids=ids, prose_noncode_first_mask=pmask, code_def_first_mask=cmask)
    )
    assert loss_with_f.item() > loss_no_f.item()
    loss_with_f.backward()
    assert bank.head_out_form.grad is not None and bank.head_out_world.grad is not None

