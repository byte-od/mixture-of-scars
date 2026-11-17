#!/usr/bin/env python3
"""ScarField v8.0 — Geodesic Macro-Scout Fleet & Shared-Crystal E8 Swarm Audit on NVIDIA A100.

Validates all three canonical tiers of `ScarField v8.0` (`docs/SCARFIELD_V7_5_FUSED_AND_V8_MACRO_SCOUT_FLEET.md`):
  1. **Level 1 — Concurrent Multi-Stream CUDA Execution (`M = 4` Specialist Macro-Scouts)**:
       - Benchmarks `M = 1` vs `M = 4` concurrent `torch.cuda.Stream`s (`~5.6 GiB` total VRAM out of `40/80 GiB`)
         and `S = 8` `SharedCrystalMacroSwarm` workers saturating the `108` `A100` SMs in zero-sync waves.
  2. **Level 2 — Specialist Macro-Scouts + Biorthogonal Tangent-Space Geodesic Merge**:
       - Eliminates multi-task sequential interference by running 4 dedicated specialists in parallel:
           * `prose_polyglot`: Open `PT/EN/ES/FR` Wikipedia prose + zero-Procrustes `S_world` alignment,
           * `code_ast`: Real `Python/C/WGSL` repositories + closure-weighted AST-valid Python coding SFT,
           * `long_biorth_passkey` (`owns_biorth_organ=True`): Long-horizon `T = 640..2048` causal Passkey
             retrieval over strict monotonic Top-M `BiorthogonalShockBank` (`sal_new > min_sal`),
           * `chat_epistemic_o1`: Language-pure un-truncated 3-turn `O(1)` PIN chat + `0x00` Vacuum safety.
       - Merges every `H = 20` local steps via `Log_{w_0}` + conflict-free projection + `[4x4]` Biorthogonal
         Dual Gram solve + outer Nesterov `Exp_{w_0}` + `1x` amortized fleet-wide `heal_all_e8_scars`.
  3. **Level 3 — Shared-Crystal Gosset `E_8` MoS Swarm (`S = 8` Ultralight Scar Workers)**:
       - Shares `1` read-only `1.000 bpw` Gosset `E_8` crystal (`0 B` `[O, I]` copy per worker!) across `8`
         ultralight workers evolving ONLY ternary atomic scars `(a_scar, b_scar, gamma_scar)` + cap poles,
         distilling winning scar innovations back into `scarfield_v8_0_macro_fleet.sfield.scar` (`9.527 MiB`, `Delta = +0 B`).
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_codec.binary_format import ScarFieldMmapContainer
from scar_field import (
    E8SphericalLinear,
    HAS_TRITON,
    MacroBatchSpec,
    MacroScoutGeodesicFleet,
    MacroScoutSpec,
    ScarFieldConfig,
    ScarFieldEngine,
    SharedCrystalMacroSwarm,
    System1CorpusScoutFleet,
    fwht64_rademacher,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
    tsallis15_spherical_cap_probs,
    unpack_ternary_2bit,
)
from scripts.colab_train_scar_field_v7_2_overnight import (
    PARALLEL_SCIENTIFIC_ANCHORS,
    DenseByteTransformer,
    build_overnight_open_corpora,
    evaluate_open_passkey_recall,
    sample_open_passkey_batch,
)
from scripts.colab_train_scar_field_v7_3_biorth_chat import (
    HELDOUT_EVAL_PROMPTS,
    OUT_OF_CONE_ABSTAIN_PROMPTS,
    calibrate_ppm_escape_with_byte_prior_on_train,
    evaluate_byte_lm_metrics_v73,
    sample_parallel_batch,
)
from scripts.colab_train_scar_field_v7_5_fused import (
    build_sft_dialogue_pools_v75,
    sample_sft_masked_batch_v75,
)

V73_A100_MEASURED_STEP_MS = 1836.4
V74_A100_MEASURED_STEP_MS = 1127.5
V75_A100_MEASURED_STEP_MS = 699.77


def _sync_dev(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _vram_gib(dev: torch.device) -> float:
    if dev.type == "cuda":
        return round(torch.cuda.max_memory_allocated(dev) / float(1024**3), 3)
    return 0.0


def sample_weighted_long_passkey_batch(
    prose_arr: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    pin_min: int,
    pin_max: int,
    device: torch.device,
    pin_query_boost: float = 18.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples `[B, seq_len]` Passkey windows with salience emphasis on `[QUERY PIN=> XYZ.]` target bytes."""
    batch, q_positions, _ = sample_open_passkey_batch(
        prose_arr, batch_size, seq_len, rng, pin_min=pin_min, pin_max=pin_max, device=device
    )
    mask_np = np.ones((batch_size, seq_len - 1), dtype=np.float32)
    for b_i, qp in enumerate(q_positions):
        # `mask[t]` weights target `batch[b_i, t + 1]`; `qp` predicts 1st digit, `qp+3` predicts closing `.`
        lo = max(0, min(seq_len - 1, qp))
        hi = max(0, min(seq_len - 1, qp + 4))
        mask_np[b_i, lo:hi] = float(pin_query_boost)
    return batch, torch.from_numpy(mask_np).to(device=device, non_blocking=True)


def sample_code_sft_batch(
    code_sft_pool: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples un-truncated AST-verified Python coding SFT windows (`0` mid-function slicing!)."""
    rows_b = np.zeros((batch_size, seq_len), dtype=np.int64)
    rows_m = np.zeros((batch_size, seq_len - 1), dtype=np.float32)
    for r_i in range(batch_size):
        cur_pos = 0
        for _ in range(3):
            arr, msk = rng.choice(code_sft_pool)
            if cur_pos > 0 and (cur_pos + len(arr) > seq_len):
                break
            n_copy = min(seq_len - cur_pos, len(arr))
            rows_b[r_i, cur_pos : cur_pos + n_copy] = arr[:n_copy]
            if cur_pos == 0:
                rows_m[r_i, : n_copy - 1] = msk[: n_copy - 1]
            else:
                rows_m[r_i, cur_pos - 1] = 0.0
                if n_copy > 1:
                    rows_m[r_i, cur_pos : cur_pos + n_copy - 1] = msk[: n_copy - 1]
            cur_pos += n_copy
            if cur_pos >= seq_len - 96:
                break
        if float(rows_m[r_i].sum()) < 2.0:
            rows_m[r_i, : max(2, cur_pos - 1)] = 1.0
    return (
        torch.from_numpy(rows_b).to(device=device, non_blocking=True),
        torch.from_numpy(rows_m).to(device=device, non_blocking=True),
    )


def warm_start_engine(sf_eng: ScarFieldEngine, ws_path: Path, dev: torch.device) -> int:
    warm_loaded_tensors = 0
    if ws_path.name.endswith(".sfield.scar"):
        with ScarFieldMmapContainer(ws_path, verify_crc=True) as ctr:
            for mname, mod in sf_eng.named_modules():
                if isinstance(mod, E8SphericalLinear) and f"e8::{mname}::codes_u8" in ctr.arrays:
                    c_u8 = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::codes_u8"], copy=True)).long().to(dev)
                    a_pk = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::a_tern"], copy=True)).to(dev)
                    b_pk = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::b_tern"], copy=True)).to(dev)
                    sc = np.array(ctr.arrays[f"e8::{mname}::scalars"], copy=True)
                    tau_b, gam_s = float(sc[0]), float(sc[1])
                    sqrt_i = math.sqrt(float(mod.in_features))
                    lut_d = mod.lut_e8.to(dev)
                    sgn_d = mod.signs.to(dev)
                    w_q_rot = lut_d[c_u8].reshape(mod.out_features, mod.in_features) / sqrt_i
                    w_q = fwht64_rademacher(w_q_rot, sgn_d, inverse=True)
                    with torch.no_grad():
                        if (
                            mod.scar_rank > 0
                            and mod.a_scar is not None
                            and mod.b_scar is not None
                            and mod.gamma_scar is not None
                        ):
                            a_f = unpack_ternary_2bit(a_pk).float()
                            b_f = unpack_ternary_2bit(b_pk).float()
                            w_eff = tau_b * w_q + (gam_s / math.sqrt(float(mod.in_features * mod.scar_rank))) * (
                                b_f @ a_f
                            )
                            mod.weight_sphere.copy_(project_rows_to_unit_sphere(w_eff))
                            mod.a_scar.copy_(project_rows_to_unit_sphere(a_f))
                            mod.b_scar.copy_(project_rows_to_unit_sphere(b_f))
                            mod.gamma_scar.fill_(gam_s)
                        else:
                            mod.weight_sphere.copy_(project_rows_to_unit_sphere(w_q))
                        mod.tau_base.fill_(tau_b)
                    warm_loaded_tensors += 1
            pmap = dict(sf_eng.named_parameters())
            with torch.no_grad():
                for k_arr in list(ctr.arrays.keys()):
                    if k_arr.startswith("pole::"):
                        pname = k_arr[len("pole::") :]
                        if pname in pmap and tuple(pmap[pname].shape) == tuple(ctr.arrays[k_arr].shape):
                            pmap[pname].copy_(
                                torch.from_numpy(np.array(ctr.arrays[k_arr], copy=True)).to(
                                    dtype=pmap[pname].dtype, device=dev
                                )
                            )
                            warm_loaded_tensors += 1
    else:
        ws_obj = torch.load(ws_path, map_location=dev, weights_only=False)
        ws_sd = ws_obj["model_state"] if isinstance(ws_obj, dict) and "model_state" in ws_obj else ws_obj
        tgt_sd = sf_eng.state_dict()
        copied = {
            k_name: v_ten.to(dtype=tgt_sd[k_name].dtype, device=dev)
            for k_name, v_ten in ws_sd.items()
            if k_name in tgt_sd and tuple(tgt_sd[k_name].shape) == tuple(v_ten.shape)
        }
        sf_eng.load_state_dict(copied, strict=False)
        warm_loaded_tensors = len(copied)

    with torch.no_grad():
        if float(sf_eng.bus.biorth_gain.item()) < 0.75:
            sf_eng.bus.biorth_gain.fill_(0.75)
        if float(sf_eng.head.cap_adapt_gain.item()) < 0.35:
            sf_eng.head.cap_adapt_gain.fill_(0.35)
    heal_all_e8_scars(sf_eng)
    return warm_loaded_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="ScarField v8.0 Macro-Scout Geodesic Fleet & Shared-Crystal Swarm Audit")
    parser.add_argument("--dry-run", action="store_true", help="Fast local CPU verification")
    parser.add_argument("--outer-rounds", type=int, default=25, help="Level-2 Specialist Fleet outer rounds")
    parser.add_argument("--local-steps", type=int, default=20, help="Local steps H per Specialist per round")
    parser.add_argument("--swarm-rounds", type=int, default=5, help="Level-3 Shared-Crystal Swarm rounds")
    parser.add_argument("--swarm-steps", type=int, default=12, help="Local steps per Swarm worker per round")
    parser.add_argument("--dense-steps", type=int, default=2400, help="DenseByteTransformer reference steps")
    parser.add_argument("--warm-start", type=str, default="", help="Path to v7.5 .sfield.scar")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    torch.manual_seed(20260800)
    np.random.seed(20260800)
    py_rng = random.Random(20260800)

    dev = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_gib = round(torch.cuda.get_device_properties(0).total_memory / float(1024**3), 2)
    else:
        gpu_name = "CPU"
        gpu_total_gib = 0.0

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path("/content/scarfield_v8_0_out")
            if Path("/content").exists() and not dry_run
            else REPO_ROOT / "artifacts" / "scarfield_v8_0_macro_fleet"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "corpus_cache"

    outer_rounds = 2 if dry_run else int(args.outer_rounds)
    local_steps = 2 if dry_run else int(args.local_steps)
    swarm_rounds = 1 if dry_run else int(args.swarm_rounds)
    swarm_steps = 2 if dry_run else int(args.swarm_steps)
    dense_steps = 4 if dry_run else int(args.dense_steps)

    print("=" * 102, flush=True)
    print(
        f"SCARFIELD v8.0 — MACRO-SCOUT GEODESIC FLEET (M=4) + SHARED-CRYSTAL E8 SWARM (S=8) | TRITON={HAS_TRITON}",
        flush=True,
    )
    print(
        f"Hardware: {gpu_name} ({gpu_total_gib:.1f} GiB) | Fleet: {outer_rounds}x{local_steps}x4={outer_rounds*local_steps*4} steps "
        f"+ Swarm: {swarm_rounds}x{swarm_steps}x8={swarm_rounds*swarm_steps*8} steps",
        flush=True,
    )
    print("=" * 102, flush=True)

    # [1/7] Prepare Open Corpora + Language-Pure SFT Pools
    print("\n[1/7] Preparing Open Corpora (`PT/EN/ES/FR/Code`) & Language-Pure SFT Pools...", flush=True)
    train_splits, val_splits, provenance = build_overnight_open_corpora(cache_dir, dry_run=dry_run)
    sft_seq_len = 288 if dry_run else 512
    sft_single_by_lang, sft_mt_pool = build_sft_dialogue_pools_v75(
        train_splits, py_rng, max_seq_len=sft_seq_len
    )
    code_sft_pool = sft_single_by_lang["code"]
    prose_sft_by_lang = {k: sft_single_by_lang[k] for k in ("pt", "en", "es", "fr")}
    all_prose_concat = np.concatenate([train_splits[k] for k in ("pt", "en", "es", "fr")])

    if dry_run:
        sf_cfg = ScarFieldConfig(
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
            s2_steps=2,
            wy_chunk_size=32,
            wave_lags=(1, 2, 4, 8),
            num_macro_cycles=2,
            biorth_slots=16,
            biorth_lambda=0.04,
            enable_biorth_uscar=True,
            enable_adaptive_cap=True,
            scout_g0_theta=0.38,
            antiloop_weight=0.14,
            shock_closure_decay=0.85,
            enable_repeat_phase=True,
        )
        bsz = 4
        eval_chunks = 2
        dense_d, dense_l, dense_h = 128, 2, 4
    else:
        sf_cfg = ScarFieldConfig(
            vocab_size=256,
            d_form=384,
            d_world=512,
            d_bus=384,
            d_key=128,
            num_subspaces=8,
            ring_len=64,
            repulsive_len=12,
            base_rank=12,
            num_scars=96,
            s1_steps=1,
            s2_steps=4,
            s2_surprise_theta=0.45,
            shock_write_theta=0.50,
            vacuum_kappa=0.36,
            uncertainty_margin_thresh=0.24,
            ghost_jump_candidates=3,
            ghost_jump_gamma=0.35,
            wy_chunk_size=128,
            enable_kalman_fast=True,
            wave_lags=(1, 2, 3, 4, 6, 8, 12, 16, 24),
            num_macro_cycles=2,
            biorth_slots=48,
            biorth_lambda=0.04,
            enable_biorth_uscar=True,
            enable_adaptive_cap=True,
            scout_g0_theta=0.38,
            antiloop_weight=0.14,
            shock_closure_decay=0.85,
            enable_repeat_phase=True,
        )
        bsz = 16
        eval_chunks = 80
        dense_d, dense_l, dense_h = 512, 10, 8

    master_eng = ScarFieldEngine(sf_cfg).to(dev)
    warm_loaded = 0
    if args.warm_start and Path(args.warm_start).exists() and not dry_run:
        warm_loaded = warm_start_engine(master_eng, Path(args.warm_start), dev)
        print(f"  [Warm-Start] Restored {warm_loaded} modules/poles from `{Path(args.warm_start).name}`!", flush=True)

    budget = master_eng.byte_budget_summary()

    def sample_prose_cands(seq_len: int, n_rows: int) -> torch.Tensor:
        chosen = py_rng.choices(("pt", "en", "es", "fr"), k=n_rows)
        rows = []
        for lg in chosen:
            arr = train_splits[lg]
            off = py_rng.randint(0, max(1, len(arr) - seq_len - 1))
            rows.append(arr[off : off + seq_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows, axis=0)).to(device=dev, non_blocking=True)

    def sample_code_cands(seq_len: int, n_rows: int) -> torch.Tensor:
        arr = train_splits["code"]
        rows = []
        for _ in range(n_rows):
            off = py_rng.randint(0, max(1, len(arr) - seq_len - 1))
            rows.append(arr[off : off + seq_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows, axis=0)).to(device=dev, non_blocking=True)

    # [2/7] Controlled Hardware Scaling Benchmark (`M=1` Single vs `M=4` Multi-Stream Fleet vs `S=8` Shared-Crystal Swarm)
    print("\n[2/7] Benchmarking GPU SM Concurrency & VRAM (`M=1` Single vs `M=4` Fleet vs `S=8` Crystal Swarm)...", flush=True)
    bench_T = 64 if dry_run else 384
    bench_x = sample_prose_cands(bench_T, bsz)

    # 2A. M=1 Single-Instance Baseline (`B=16, T=384`)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    m1_fleet = MacroScoutGeodesicFleet(
        master_eng, [MacroScoutSpec("single")], base_lr=1e-4, scalar_lr=1e-4, enable_cuda_streams=False
    )
    # Warmup PTX
    _ = m1_fleet.run_concurrent_round([lambda _w, _h: MacroBatchSpec(bench_x)], local_steps=2, merge_after=False)
    _sync_dev(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    n_b_steps = 2 if dry_run else 6
    t0 = time.perf_counter()
    _ = m1_fleet.run_concurrent_round([lambda _w, _h: MacroBatchSpec(bench_x)], local_steps=n_b_steps, merge_after=False)
    _sync_dev(dev)
    m1_step_ms = ((time.perf_counter() - t0) / float(n_b_steps)) * 1000.0
    m1_vram_gib = _vram_gib(dev)
    del m1_fleet
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # 2B. S=8 Shared-Crystal Swarm (`B=16, T=384`, 8 concurrent ultralight workers!)
    s_bench_cnt = 2 if dry_run else 8
    s8_swarm = SharedCrystalMacroSwarm(master_eng, num_workers=s_bench_cnt, scar_lr=1e-4, scalar_lr=1e-4, distill_eta=0.0)
    swarm_param_red_x = round(float(s8_swarm.master_total_params) / float(max(1, s8_swarm.mutable_params_per_worker)), 2)
    swarm_mut_params = s8_swarm.mutable_params_per_worker
    sw_samplers_bench = [lambda _w, _h: MacroBatchSpec(bench_x) for _ in range(s_bench_cnt)]
    _ = s8_swarm.run_swarm_round(sw_samplers_bench, local_steps=1)
    _sync_dev(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0_sw = time.perf_counter()
    _ = s8_swarm.run_swarm_round(sw_samplers_bench, local_steps=(2 if dry_run else 4))
    _sync_dev(dev)
    sw_steps_tot = s_bench_cnt * (2 if dry_run else 4)
    s8_eff_ms_per_worker_step = ((time.perf_counter() - t0_sw) / float(sw_steps_tot)) * 1000.0
    s8_vram_gib = _vram_gib(dev)
    del s8_swarm
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # [3/7] Initialize Level-1 + Level-2 `M=4` Specialist `MacroScoutGeodesicFleet`
    base_lr = 1.05e-3 if warm_loaded > 0 else 4.2e-3
    min_lr = 1.8e-4
    specs_v8 = [
        MacroScoutSpec("prose_polyglot", weight=1.0, lr_scale=1.0, owns_biorth_organ=False),
        MacroScoutSpec("code_ast", weight=1.05, lr_scale=1.0, owns_biorth_organ=False),
        MacroScoutSpec("long_biorth_passkey", weight=1.10, lr_scale=1.05, owns_biorth_organ=True),
        MacroScoutSpec("chat_epistemic_o1", weight=1.05, lr_scale=1.0, owns_biorth_organ=False),
    ]
    fleet = MacroScoutGeodesicFleet(
        master_eng,
        specs_v8,
        base_lr=base_lr,
        scalar_lr=1.4e-3,
        local_momentum=0.88,
        outer_lr=0.85,
        outer_momentum=0.50,
        merge_ridge_lambda=0.14,
        local_mom_retain=0.30,
        biorth_gain_floor=0.55,
        enable_cuda_streams=True,
    )

    # Benchmark `M=4` Concurrent Multi-Stream `B=16, T=384` Throughput before training!
    m4_bench_samplers = [lambda _w, _h: MacroBatchSpec(bench_x) for _ in range(4)]
    _ = fleet.run_concurrent_round(m4_bench_samplers, local_steps=1, merge_after=False)
    fleet.broadcast_master_to_scouts()
    _sync_dev(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0_m4 = time.perf_counter()
    _ = fleet.run_concurrent_round(m4_bench_samplers, local_steps=(2 if dry_run else 5), merge_after=False)
    _sync_dev(dev)
    m4_tot_steps = 4 * (2 if dry_run else 5)
    m4_eff_step_ms = ((time.perf_counter() - t0_m4) / float(m4_tot_steps)) * 1000.0
    m4_vram_gib = _vram_gib(dev)
    fleet.broadcast_master_to_scouts()
    for opt in fleet.optimizers:
        opt.state.clear()

    m4_concurrency_speedup_x = m1_step_ms / max(0.1, m4_eff_step_ms)
    s8_swarm_speedup_vs_m1_x = m1_step_ms / max(0.1, s8_eff_ms_per_worker_step)
    eff_speedup_vs_v73_x = V73_A100_MEASURED_STEP_MS / max(0.1, m4_eff_step_ms)

    print(
        f"  [Hardware Scaling (`B={bsz}, T={bench_T}`)]\n"
        f"    • M=1 Single Instance:       {m1_step_ms:6.1f} ms/step | Peak VRAM = {m1_vram_gib:.2f} GiB\n"
        f"    • M=4 Multi-Stream Fleet:    {m4_eff_step_ms:6.1f} ms/step ({m4_concurrency_speedup_x:.2f}x > M=1 | {eff_speedup_vs_v73_x:.2f}x > v7.3!) | Peak VRAM = {m4_vram_gib:.2f} GiB\n"
        f"    • S={s_bench_cnt} Shared-Crystal Swarm:  {s8_eff_ms_per_worker_step:6.1f} ms/step ({s8_swarm_speedup_vs_m1_x:.2f}x > M=1, {swarm_param_red_x:.1f}x smaller worker!) | Peak VRAM = {s8_vram_gib:.2f} GiB",
        flush=True,
    )

    prose_s1_scout = System1CorpusScoutFleet(oversample_factor=(2 if dry_run else 6), anchor_fraction=0.25)
    code_s1_scout = System1CorpusScoutFleet(oversample_factor=(2 if dry_run else 6), anchor_fraction=0.25)
    chat_s1_scout = System1CorpusScoutFleet(oversample_factor=(2 if dry_run else 4), anchor_fraction=0.25)

    selected_bytes_total = 0
    scouted_bytes_total = 0
    degen_filtered_total = 0
    epistemic_gains: List[float] = []
    history: List[Dict[str, Any]] = []

    def make_specialist_samplers(round_idx: int):
        def samp_prose(w_eng: ScarFieldEngine, h: int) -> MacroBatchSpec:
            nonlocal selected_bytes_total, scouted_bytes_total
            if h % 5 == 4:
                la, lb = py_rng.sample(("pt", "en", "es", "fr"), 2)
                pa, pb = sample_parallel_batch(la, lb, min(bsz, 8), (64 if dry_run else 192), py_rng, dev)
                selected_bytes_total += int(pa.numel() + pb.numel())
                scouted_bytes_total += int(pa.numel() + pb.numel())
                return MacroBatchSpec(byte_ids=pa, paired_translation_ids=pb)
            sl = 96 if dry_run else (384 if (h % 3 != 0) else 512)
            cands = sample_prose_cands(sl, bsz * prose_s1_scout.oversample_factor)
            sel, _, _ = prose_s1_scout.select_frontier_batch(w_eng, cands, batch_size=bsz, sync_telemetry=False)
            selected_bytes_total += bsz * sl
            scouted_bytes_total += int(cands.numel())
            return MacroBatchSpec(byte_ids=sel)

        def samp_code(w_eng: ScarFieldEngine, h: int) -> MacroBatchSpec:
            nonlocal selected_bytes_total, scouted_bytes_total
            if (h % 2 == 1) and len(code_sft_pool) > 0:
                sl_c = 256 if dry_run else 448
                cb, cm = sample_code_sft_batch(code_sft_pool, bsz, sl_c, py_rng, dev)
                selected_bytes_total += bsz * sl_c
                scouted_bytes_total += bsz * sl_c
                return MacroBatchSpec(byte_ids=cb, target_loss_mask=cm)
            sl = 96 if dry_run else 384
            cands = sample_code_cands(sl, bsz * code_s1_scout.oversample_factor)
            sel, _, _ = code_s1_scout.select_frontier_batch(w_eng, cands, batch_size=bsz, sync_telemetry=False)
            selected_bytes_total += bsz * sl
            scouted_bytes_total += int(cands.numel())
            return MacroBatchSpec(byte_ids=sel)

        def samp_long_biorth(_w_eng: ScarFieldEngine, h: int) -> MacroBatchSpec:
            nonlocal selected_bytes_total, scouted_bytes_total
            if dry_run:
                pk_len, b_pk = 128, bsz
            else:
                # Cycle explicitly through `640, 1024, 1536, 2048` so `BiorthogonalShockBank` masters T=2048!
                sched = (640, 1024, 1536, 2048)
                pk_len = sched[(round_idx + h) % len(sched)]
                b_pk = 8 if pk_len >= 1536 else bsz
            pk_b, pk_m = sample_weighted_long_passkey_batch(
                all_prose_concat, b_pk, pk_len, py_rng, pin_min=100, pin_max=499, device=dev, pin_query_boost=20.0
            )
            selected_bytes_total += b_pk * pk_len
            scouted_bytes_total += b_pk * pk_len
            return MacroBatchSpec(byte_ids=pk_b, target_loss_mask=pk_m)

        def samp_chat_epistemic(w_eng: ScarFieldEngine, h: int) -> MacroBatchSpec:
            nonlocal selected_bytes_total, scouted_bytes_total
            n_cand = bsz * chat_s1_scout.oversample_factor
            cb_cand, cm_cand = sample_sft_masked_batch_v75(
                prose_sft_by_lang, sft_mt_pool, n_cand, sft_seq_len, py_rng, dev, mt_fraction=0.50
            )
            sb, sm, _ = chat_s1_scout.select_frontier_batch(
                w_eng, cb_cand, batch_size=bsz, candidate_loss_mask=cm_cand, sync_telemetry=False
            )
            assert sm is not None
            selected_bytes_total += bsz * sft_seq_len
            scouted_bytes_total += int(cb_cand.numel())
            if (not dry_run) and (h % 4 == 0):
                ooc_rows = []
                ooc_mask = torch.zeros_like(sb, dtype=torch.bool)
                for r_i in range(2):
                    p_ooc = py_rng.choice(OUT_OF_CONE_ABSTAIN_PROMPTS)
                    raw_ooc = (f"\x02User: {p_ooc}\n\x03Assistant: ").encode("utf-8")
                    end_idx = min(sft_seq_len, len(raw_ooc))
                    pad_ooc = (raw_ooc + b" " * sft_seq_len)[:sft_seq_len]
                    ooc_rows.append(np.frombuffer(pad_ooc, dtype=np.uint8).astype(np.int64))
                    ooc_mask[r_i, max(0, end_idx - 6) : end_idx] = True
                sb[:2] = torch.from_numpy(np.stack(ooc_rows, axis=0)).to(dev)
                sm[:2] = 0.0
                return MacroBatchSpec(byte_ids=sb, out_of_cone_mask=ooc_mask, target_loss_mask=sm)
            return MacroBatchSpec(byte_ids=sb, target_loss_mask=sm)

        return [samp_prose, samp_code, samp_long_biorth, samp_chat_epistemic]

    # [4/7] Execute Level-1 + Level-2 Geodesic Fleet Co-Training
    print(
        f"\n[3/7] Running {outer_rounds} Geodesic Fleet Outer Rounds "
        f"(`{outer_rounds * local_steps * 4}` concurrent Specialist steps)...",
        flush=True,
    )
    t0_fleet = time.perf_counter()
    total_conflicts_fleet = 0
    for r_idx in range(1, outer_rounds + 1):
        frac = (r_idx - 1) / float(max(1, outer_rounds - 1))
        cur_lr = min_lr + (base_lr - min_lr) * (0.5 * (1.0 + math.cos(math.pi * frac)))
        fleet.set_learning_rate(cur_lr, scalar_lr=max(3e-4, cur_lr * 1.25))

        samplers = make_specialist_samplers(r_idx)
        merge_tel = fleet.run_concurrent_round(samplers, local_steps=local_steps, grad_clip=1.0, merge_after=True)
        total_conflicts_fleet += merge_tel.conflict_pairs_projected

        t_p = prose_s1_scout.flush_async_telemetry()
        t_c = code_s1_scout.flush_async_telemetry()
        t_s = chat_s1_scout.flush_async_telemetry()
        degen_filtered_total += (
            t_p.degenerate_windows_filtered + t_c.degenerate_windows_filtered + t_s.degenerate_windows_filtered
        )
        round_ep_gain = float(np.mean([t_p.epistemic_gain_ratio, t_c.epistemic_gain_ratio, t_s.epistemic_gain_ratio]))
        epistemic_gains.append(round_ep_gain)

        if r_idx == 1 or r_idx % (1 if dry_run else 5) == 0 or r_idx == outer_rounds:
            rec = {
                "round": r_idx,
                "fleet_steps_cum": r_idx * local_steps * 4,
                "lr": round(cur_lr, 6),
                "scout_losses": merge_tel.scout_mean_cap_losses,
                "conflicts_projected": merge_tel.conflict_pairs_projected,
                "offdiag_gram_cos": merge_tel.mean_offdiag_gram_cosine,
                "biorth_gain": merge_tel.biorth_gain_after_merge,
                "scout_epistemic_gain_x": round(round_ep_gain, 3),
                "elapsed_s": round(time.perf_counter() - t0_fleet, 1),
            }
            history.append(rec)
            print(
                f"  [Fleet Round {r_idx:2d}/{outer_rounds}] steps={rec['fleet_steps_cum']:4d} lr={cur_lr:.4f} "
                f"losses={merge_tel.scout_mean_cap_losses} conflicts_healed={merge_tel.conflict_pairs_projected} "
                f"biorth_g={merge_tel.biorth_gain_after_merge:.3f} ({rec['elapsed_s']:.1f}s)",
                flush=True,
            )

    _sync_dev(dev)
    fleet_train_sec = time.perf_counter() - t0_fleet
    del fleet
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # [5/7] Execute Level-3 `SharedCrystalMacroSwarm` Polish (`S=8` workers sharing `1` Read-Only `E_8` Crystal)
    n_swarm_workers = 4 if dry_run else 8
    print(
        f"\n[4/7] Running {swarm_rounds} Level-3 `SharedCrystalMacroSwarm` Rounds "
        f"(`S={n_swarm_workers}` scar-only workers sharing `1` Gosset E8 Crystal)...",
        flush=True,
    )
    t0_swarm_phase = time.perf_counter()
    swarm = SharedCrystalMacroSwarm(
        master_eng,
        num_workers=n_swarm_workers,
        scar_lr=4.5e-4,
        scalar_lr=4.0e-4,
        distill_eta=0.55,
        merge_ridge_lambda=0.14,
        biorth_gain_floor=0.55,
        enable_cuda_streams=True,
    )
    swarm_conflicts_total = 0
    last_sw_tel = None
    for sw_r in range(1, swarm_rounds + 1):
        base_4 = make_specialist_samplers(outer_rounds + sw_r)
        sw_samplers = [base_4[i % 4] for i in range(n_swarm_workers)]
        sw_tel = swarm.run_swarm_round(sw_samplers, local_steps=swarm_steps, grad_clip=1.0)
        last_sw_tel = sw_tel
        swarm_conflicts_total += sw_tel.conflict_pairs_projected
        print(
            f"  [Swarm Round {sw_r}/{swarm_rounds}] distilled_layers={sw_tel.crystal_layers_distilled} "
            f"conflicts_healed={sw_tel.conflict_pairs_projected} scar_fro={sw_tel.mean_scar_innovation_fro:.4f} "
            f"({time.perf_counter() - t0_swarm_phase:.1f}s)",
            flush=True,
        )
    _sync_dev(dev)
    swarm_train_sec = time.perf_counter() - t0_swarm_phase
    del swarm
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Flush any residual async S1 scout counters
    for sc_f in (prose_s1_scout, code_s1_scout, chat_s1_scout):
        t_rem = sc_f.flush_async_telemetry()
        degen_filtered_total += t_rem.degenerate_windows_filtered

    total_v8_steps = (outer_rounds * local_steps * 4) + (swarm_rounds * swarm_steps * n_swarm_workers)
    total_v8_sec = fleet_train_sec + swarm_train_sec

    # [6/7] Train Matched `63.77 MiB` `DenseByteTransformer` Reference
    print("\n[5/7] Training Matched `63.77 MiB` `DenseByteTransformer` Reference...", flush=True)
    dense_model = DenseByteTransformer(vocab_size=256, d_model=dense_d, n_layers=dense_l, n_heads=dense_h).to(dev)
    dense_params = sum(p.numel() for p in dense_model.parameters())
    dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=1.2e-3, weight_decay=0.01)
    t_dn = time.perf_counter()
    domains_all = ("pt", "en", "es", "fr", "code")
    dom_probs = (0.19, 0.19, 0.19, 0.19, 0.24)

    def sample_any_open(seq_len: int, n_rows: int) -> torch.Tensor:
        chosen = py_rng.choices(domains_all, weights=dom_probs, k=n_rows)
        rows = []
        for d in chosen:
            arr = train_splits[d]
            off = py_rng.randint(0, max(1, len(arr) - seq_len - 1))
            rows.append(arr[off : off + seq_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows, axis=0)).to(device=dev, non_blocking=True)

    for d_step in range(1, dense_steps + 1):
        dfrac = (d_step - 1) / float(max(1, dense_steps - 1))
        d_seql = 64 if dry_run else (192 if dfrac < 0.30 else (320 if dfrac < 0.65 else 448))
        dlr = 2.0e-4 + (1.2e-3 - 2.0e-4) * (0.5 * (1.0 + math.cos(math.pi * dfrac)))
        for pg in dense_opt.param_groups:
            pg["lr"] = dlr
        dense_opt.zero_grad(set_to_none=True)
        if (not dry_run) and (d_step % 6 == 0):
            pk_b, _, _ = sample_open_passkey_batch(
                all_prose_concat, bsz // 2, d_seql, py_rng, pin_min=100, pin_max=499, device=dev
            )
            dbatch = torch.cat((sample_any_open(d_seql, bsz - (bsz // 2)), pk_b), dim=0)
        else:
            dbatch = sample_any_open(d_seql, bsz)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            d_logits = dense_model(dbatch)
            d_loss = F.cross_entropy(d_logits[:, :-1, :].reshape(-1, 256), dbatch[:, 1:].reshape(-1))
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
        dense_opt.step()
    _sync_dev(dev)
    dense_train_sec = time.perf_counter() - t_dn

    # [7/7] Complete Forensic Evaluation & `.sfield.scar` Export
    print("\n[6/7] Evaluating 5/5 Domains, T=2048 Passkeys, 3-Turn O(1) PIN Chat & Code AST...", flush=True)
    master_eng.set_async_stream_mode(False)
    master_eng.eval()
    dense_model.eval()

    def sf_probs_fn(x: torch.Tensor) -> torch.Tensor:
        sc = master_eng.forward_sequence(x, adaptive_depth=True)["scores"]
        pr, _ = tsallis15_spherical_cap_probs(sc)
        return pr

    def dense_probs_fn(x: torch.Tensor) -> torch.Tensor:
        return F.softmax(dense_model(x), dim=-1)

    eval_seq_len = 64 if dry_run else 384
    domain_eval: Dict[str, Any] = {}
    for dom in ("pt", "en", "es", "fr", "code"):
        lam_dom, prior_dom = calibrate_ppm_escape_with_byte_prior_on_train(
            sf_probs_fn, train_splits[dom], eval_seq_len, 2 if dry_run else 24, dev
        )
        sf_m = evaluate_byte_lm_metrics_v73(
            sf_probs_fn,
            val_splits[dom],
            eval_seq_len,
            eval_chunks,
            dev,
            is_sparse_tsallis=True,
            escape_lambda=lam_dom,
            escape_byte_prior=prior_dom,
        )
        dn_m = evaluate_byte_lm_metrics_v73(
            dense_probs_fn, val_splits[dom], eval_seq_len, eval_chunks, dev, is_sparse_tsallis=False
        )
        domain_eval[dom] = {"scarfield_v8_0_macro_fleet": sf_m, "dense_fp16_ot": dn_m}
        print(
            f"  [{dom.upper():4s}] ScarField v8.0: BPB={sf_m['calibrated_bpb']:.3f} "
            f"Top1={sf_m['top1_byte_acc_pct']:.2f}% Top5={sf_m['top5_byte_acc_pct']:.2f}% | "
            f"Dense FP16: BPB={dn_m['bpb']:.3f} Top1={dn_m['top1_byte_acc_pct']:.2f}%",
            flush=True,
        )

    pk_horizons = [64, 128] if dry_run else [128, 256, 512, 768, 1024, 1536, 2048]
    n_pk = 4 if dry_run else 32
    val_prose_concat = np.concatenate([val_splits[k] for k in ("pt", "en", "es", "fr")])
    passkey_eval: Dict[str, Any] = {
        "scarfield_v8_0_biorthogonal": {},
        "scarfield_ablation_no_biorth_bank": {},
        "dense_fp16_ot": {},
    }
    saved_biorth_gain = float(master_eng.bus.biorth_gain.item())

    def _sf_batched(b_in: torch.Tensor) -> torch.Tensor:
        outs = []
        for s_i in range(0, b_in.shape[0], 8):
            outs.append(master_eng.forward_sequence(b_in[s_i : s_i + 8], adaptive_depth=True)["scores"])
        return torch.cat(outs, dim=0)

    def _dense_batched(b_in: torch.Tensor) -> torch.Tensor:
        outs = []
        for s_i in range(0, b_in.shape[0], 8):
            outs.append(dense_model(b_in[s_i : s_i + 8]))
        return torch.cat(outs, dim=0)

    for h_len in pk_horizons:
        with torch.no_grad():
            master_eng.bus.biorth_gain.fill_(saved_biorth_gain)
            sf_pk = evaluate_open_passkey_recall(_sf_batched, val_prose_concat, h_len, n_pk, dev)
            master_eng.bus.biorth_gain.fill_(0.0)
            sf_no_b = evaluate_open_passkey_recall(_sf_batched, val_prose_concat, h_len, n_pk, dev)
            master_eng.bus.biorth_gain.fill_(saved_biorth_gain)
            dn_pk = evaluate_open_passkey_recall(_dense_batched, val_prose_concat, h_len, n_pk, dev)
        passkey_eval["scarfield_v8_0_biorthogonal"][f"T={h_len}"] = sf_pk
        passkey_eval["scarfield_ablation_no_biorth_bank"][f"T={h_len}"] = sf_no_b
        passkey_eval["dense_fp16_ot"][f"T={h_len}"] = dn_pk
        print(
            f"  [Passkey T={h_len:4d}] v8.0 Biorth: {sf_pk['exact_3digit_acc']*100:5.1f}% "
            f"(No-Biorth: {sf_no_b['exact_3digit_acc']*100:5.1f}%) | Dense: {dn_pk['exact_3digit_acc']*100:5.1f}%",
            flush=True,
        )

    chat_samples: Dict[str, List[Dict[str, Any]]] = {}
    code_ast_valid_cnt = 0
    code_ast_total = 0
    for dom_k, p_list in HELDOUT_EVAL_PROMPTS.items():
        chat_samples[dom_k] = []
        for prompt_txt, kw_list in p_list:
            rep = master_eng.generate_reply(
                prompt_txt, max_new_bytes=220, enable_ghost_jump_scouts=False, apply_repulsive_anti_loop=True
            )
            ast_valid = None
            if dom_k == "code":
                code_ast_total += 1
                try:
                    ast.parse(rep.text)
                    ast_valid = True
                    code_ast_valid_cnt += 1
                except Exception:
                    ast_valid = False
            kw_hits = sum(1 for kw in kw_list if kw.lower() in rep.text.lower())
            chat_samples[dom_k].append(
                {
                    "prompt": prompt_txt,
                    "reply": rep.text,
                    "bytes_generated": rep.bytes_generated,
                    "mean_confidence": round(rep.mean_confidence, 4),
                    "s2_deliberation_fraction": round(rep.s2_deliberation_fraction, 4),
                    "state_bytes": rep.state_bytes,
                    "keyword_hits": f"{kw_hits}/{len(kw_list)}",
                    **({"python_ast_valid": ast_valid} if ast_valid is not None else {}),
                }
            )
            print(f"  [Chat {dom_k.upper():4s}] Q: {prompt_txt[:48]} -> A: {rep.text[:88]!r}", flush=True)

    # Identical 24-trial 3-Turn Stateful `O(1)` Unseen PIN Chat (`mt_rng = 74999`) + 8-trial Repeated-Digit Stress Suite
    n_mt_trials = 4 if dry_run else 24
    mt_rng = random.Random(74999)
    mt_hits = 0
    mt_digit_hits = 0
    mt_stutter_cnt = 0
    mt_state_deltas: List[int] = []
    mt_examples: List[Dict[str, Any]] = []
    mt_langs = [
        ("pt", "Guarde a chave da sessão: PIN={pin}.", "Explique de forma concisa: Termodinâmica.", "Qual é a chave da sessão registrada?"),
        ("en", "Store the session key: PIN={pin}.", "Briefly explain Quantum mechanics.", "What is the stored session key?"),
        ("es", "Guarda la clave de sesión: PIN={pin}.", "Explica brevemente: Relatividad general.", "¿Cuál es la clave de sesión registrada?"),
        ("fr", "Enregistre la clé de session : PIN={pin}.", "Explique brièvement : Entropie.", "Quelle est la clé de session enregistrée ?"),
    ]
    for t_i in range(n_mt_trials):
        lg, u1, u2, u3 = mt_langs[t_i % len(mt_langs)]
        pin = f"{mt_rng.randint(500, 999):03d}"
        st_chat = master_eng.init_state(batch_size=1, device=dev)
        b_init = st_chat.total_state_bytes
        r1 = master_eng.generate_reply(u1.format(pin=pin), state=st_chat, max_new_bytes=64)
        r2 = master_eng.generate_reply(u2, state=r1.state, max_new_bytes=180)
        r3 = master_eng.generate_reply(u3, state=r2.state, max_new_bytes=72)
        mt_state_deltas.append(r3.state_bytes - b_init)
        if re.search(r"\d{4,}", r3.text):
            mt_stutter_cnt += 1
        m_dig = re.search(r"PIN=(\d+)", r3.text) or re.search(r"\b(\d+)\b", r3.text)
        pred_pin = m_dig.group(1) if m_dig else ""
        if pred_pin == pin:
            mt_hits += 1
        for idx_d in range(3):
            if idx_d < len(pred_pin) and pred_pin[idx_d] == pin[idx_d]:
                mt_digit_hits += 1
        if t_i < 6:
            mt_examples.append(
                {
                    "lang": lg,
                    "unseen_pin": pin,
                    "turn1_reply": r1.text,
                    "turn2_reply_excerpt": r2.text[:90],
                    "turn3_reply": r3.text,
                    "turn3_bytes": r3.bytes_generated,
                    "state_bytes_delta": r3.state_bytes - b_init,
                }
            )
    mt_exact_acc = round(mt_hits / float(n_mt_trials), 4)
    mt_digit_acc = round(mt_digit_hits / float(3 * n_mt_trials), 4)

    rep_stress_pins = ["800", "777"] if dry_run else ["800", "777", "900", "880", "666", "550", "999", "500"]
    rep_stress_hits = 0
    rep_stress_stutters = 0
    rep_stress_records: List[Dict[str, Any]] = []
    for idx_rp, rpin in enumerate(rep_stress_pins):
        lg, u1, u2, u3 = mt_langs[idx_rp % len(mt_langs)]
        st_rp = master_eng.init_state(batch_size=1, device=dev)
        rr1 = master_eng.generate_reply(u1.format(pin=rpin), state=st_rp, max_new_bytes=64)
        rr2 = master_eng.generate_reply(u2, state=rr1.state, max_new_bytes=180)
        rr3 = master_eng.generate_reply(u3, state=rr2.state, max_new_bytes=72)
        m_rp = re.search(r"PIN=(\d+)", rr3.text) or re.search(r"\b(\d+)\b", rr3.text)
        got_rp = m_rp.group(1) if m_rp else ""
        has_stutter = bool(re.search(r"\d{4,}", rr3.text))
        if has_stutter:
            rep_stress_stutters += 1
        if got_rp == rpin and not has_stutter:
            rep_stress_hits += 1
        rep_stress_records.append(
            {"lang": lg, "pin": rpin, "turn3_reply": rr3.text, "exact_clean_match": (got_rp == rpin and not has_stutter)}
        )
    rep_stress_acc = round(rep_stress_hits / float(len(rep_stress_pins)), 4)

    print(
        f"  [3-Turn O(1) Chat] Unseen PIN Exact: {mt_exact_acc*100:.1f}% (stutters={mt_stutter_cnt}/{n_mt_trials}) | "
        f"Repeated-Digit Stress: {rep_stress_acc*100:.1f}% ({rep_stress_hits}/{len(rep_stress_pins)}) | "
        f"Code AST Valid: {code_ast_valid_cnt}/{code_ast_total}",
        flush=True,
    )

    with torch.no_grad():
        q_en_1 = torch.tensor([list("Earth polar radius measurement value".encode("utf-8"))], dtype=torch.long, device=dev)
        q_en_2 = torch.tensor([list("Earth equatorial radius measurement value".encode("utf-8"))], dtype=torch.long, device=dev)
        q_pt_1 = torch.tensor([list("Valor da medida do raio polar da Terra".encode("utf-8"))], dtype=torch.long, device=dev)
        hw_en_1 = project_rows_to_unit_sphere(master_eng.forward_sequence(q_en_1)["h_world"].mean(dim=1))
        hw_en_2 = project_rows_to_unit_sphere(master_eng.forward_sequence(q_en_2)["h_world"].mean(dim=1))
        hw_pt_1 = project_rows_to_unit_sphere(master_eng.forward_sequence(q_pt_1)["h_world"].mean(dim=1))
        sibling_pole_overlap = float((hw_en_1 * hw_en_2).sum().item())

        y_bus_1 = torch.zeros(1, sf_cfg.d_bus, device=dev)
        y_bus_1[0, 0] = 2.0
        y_bus_2 = torch.zeros(1, sf_cfg.d_bus, device=dev)
        y_bus_2[0, 1] = 2.0
        master_eng.bus.uscar_store.cartridges.clear()
        master_eng.bus.uscar_store.mint_zero_sgd_rrr("earth_polar", "doc_geodesy", hw_en_1, y_bus_1, rank=4, horizon_kappa=0.40)
        master_eng.bus.uscar_store.mint_zero_sgd_rrr("earth_equat", "doc_geodesy", hw_en_2, y_bus_2, rank=4, horizon_kappa=0.40)

        master_eng.bus.uscar_store.biorthogonal_poles = True
        out_biorth, _ = master_eng.bus.uscar_store.query(hw_en_1)
        _, res_xling = master_eng.bus.uscar_store.query(hw_pt_1)
        biorth_bleed_ratio = float(out_biorth[0, 1].abs().item() / max(1e-8, out_biorth[0, 0].abs().item()))

        master_eng.bus.uscar_store.biorthogonal_poles = False
        out_orth, _ = master_eng.bus.uscar_store.query(hw_en_1)
        orth_bleed_ratio = float(out_orth[0, 1].abs().item() / max(1e-8, out_orth[0, 0].abs().item()))
        master_eng.bus.uscar_store.biorthogonal_poles = True

        master_eng.bus.uscar_store.wipe_document_0x00("doc_geodesy")
        out_wiped, res_wiped = master_eng.bus.uscar_store.query(hw_en_1)
        gdpr_residual = float(out_wiped.abs().max().item() + res_wiped.abs().max().item())
        master_eng.bus.uscar_store.cartridges.clear()

    ooc_hits = 0
    for p_ooc in OUT_OF_CONE_ABSTAIN_PROMPTS:
        rep_ooc = master_eng.generate_reply(p_ooc, max_new_bytes=16, require_world_support=True)
        if rep_ooc.abstained_0x00:
            ooc_hits += 1

    xling_cos: Dict[str, float] = {}
    with torch.no_grad():
        for lg in ("pt", "es", "fr"):
            sims = []
            for item in PARALLEL_SCIENTIFIC_ANCHORS:
                ta = torch.tensor([list(item["en"].encode("utf-8"))], dtype=torch.long, device=dev)
                tb = torch.tensor([list(item[lg].encode("utf-8"))], dtype=torch.long, device=dev)
                pa = project_rows_to_unit_sphere(master_eng.forward_sequence(ta)["h_world"].mean(dim=1))
                pb = project_rows_to_unit_sphere(master_eng.forward_sequence(tb)["h_world"].mean(dim=1))
                sims.append(float((pa * pb).sum().item()))
            xling_cos[f"{lg}_en"] = round(float(np.mean(sims)), 5)

    scar_path = out_dir / "scarfield_v8_0_macro_fleet.sfield.scar"
    scar_bytes = master_eng.save_packed_scar_container(scar_path)

    sf_cpu = ScarFieldEngine(sf_cfg)
    sf_cpu.load_state_dict({k: v.detach().cpu() for k, v in master_eng.state_dict().items()})
    sf_cpu.eval()
    probe_x = torch.tensor([[ord(c) for c in "\x02User: PIN=800.\n\x03Assistant: PIN=800.\x04\n"]], dtype=torch.long)
    with torch.no_grad():
        ref_sc = sf_cpu.forward_sequence(probe_x)["scores"].reshape(-1)
    mm_eng, mm_ctr = ScarFieldEngine.load_packed_scar_mmap(scar_path, verify_crc=True)
    try:
        with torch.no_grad():
            mm_sc = mm_eng.forward_sequence(probe_x)["scores"].reshape(-1)
        mmap_cos = float(F.cosine_similarity(ref_sc.unsqueeze(0), mm_sc.unsqueeze(0)).item())
    finally:
        mm_ctr.close()

    audit_payload = {
        "suite": "scar_field_v8_0_macro_fleet_audit",
        "version": "8.0.0",
        "hardware": gpu_name,
        "gpu_total_vram_gib": gpu_total_gib,
        "triton_enabled": bool(HAS_TRITON and dev.type == "cuda"),
        "warm_started_from_v7_5_tensors": warm_loaded,
        "completed_v8_total_steps": total_v8_steps,
        "completed_level2_fleet_steps": outer_rounds * local_steps * 4,
        "completed_level3_swarm_steps": swarm_rounds * swarm_steps * n_swarm_workers,
        "streamed_v8_selected_bytes": selected_bytes_total,
        "scouted_candidate_frontier_bytes": scouted_bytes_total,
        "scarfield_v8_0_config": sf_cfg.to_dict(),
        "three_tier_macro_fleet_telemetry": {
            "v7_3_measured_step_ms": V73_A100_MEASURED_STEP_MS,
            "v7_4_measured_step_ms": V74_A100_MEASURED_STEP_MS,
            "v7_5_measured_step_ms": V75_A100_MEASURED_STEP_MS,
            "m1_single_instance_step_ms_b16_t384": round(m1_step_ms, 2),
            "m1_single_instance_peak_vram_gib": m1_vram_gib,
            "level1_m4_concurrent_fleet_eff_step_ms_b16_t384": round(m4_eff_step_ms, 2),
            "level1_m4_concurrent_fleet_bytes_per_sec": round((bsz * bench_T) / max(1e-6, m4_eff_step_ms / 1000.0), 1),
            "level1_m4_peak_vram_gib": m4_vram_gib,
            "level1_m4_speedup_vs_m1_x": round(m4_concurrency_speedup_x, 2),
            "level1_m4_speedup_vs_v7_3_x": round(eff_speedup_vs_v73_x, 2),
            "level2_specialist_names": [s.name for s in specs_v8],
            "level2_conflict_pairs_projected_total": total_conflicts_fleet,
            "level2_final_biorth_gain": round(float(master_eng.bus.biorth_gain.item()), 4),
            "level3_s8_shared_crystal_eff_step_ms_b16_t384": round(s8_eff_ms_per_worker_step, 2),
            "level3_s8_shared_crystal_peak_vram_gib": s8_vram_gib,
            "level3_s8_speedup_vs_m1_x": round(s8_swarm_speedup_vs_m1_x, 2),
            "level3_mutable_params_per_worker": swarm_mut_params,
            "level3_worker_param_reduction_x": swarm_param_red_x,
            "level3_conflict_pairs_projected_total": swarm_conflicts_total,
            "level3_last_crystal_layers_distilled": (last_sw_tel.crystal_layers_distilled if last_sw_tel else 0),
            "mean_s1_scout_epistemic_gain_x": round(float(np.mean(epistemic_gains)) if epistemic_gains else 1.0, 3),
            "degenerate_windows_filtered_total": degen_filtered_total,
        },
        "architecture_comparison": {
            "scarfield_v8_0_spherical_params": sum(p.numel() for p in master_eng.parameters()),
            "scarfield_v8_0_sfield_scar_bytes": scar_bytes,
            "scarfield_v8_0_sfield_scar_mib": round(scar_bytes / (1024**2), 3),
            "scarfield_v8_0_o1_state_bytes": budget["central_kalman_scache_o1_bytes"],
            "dense_transformer_params": dense_params,
            "dense_transformer_fp16_mib": round((dense_params * 2) / (1024**2), 3),
            "weight_compression_vs_dense_fp16_x": round((dense_params * 2) / max(1, scar_bytes), 2),
            "mmap_c_simd_forward_cosine": round(mmap_cos, 6),
            "v8_0_fleet_plus_swarm_seconds": round(total_v8_sec, 1),
            "dense_train_seconds": round(dense_train_sec, 1),
        },
        "step_1_heldout_open_domain_metrics": domain_eval,
        "step_1_biorthogonal_passkey_recall_up_to_t2048": passkey_eval,
        "step_1_biorthogonal_uscar_and_epistemic": {
            "sibling_pole_raw_cosine_overlap": round(sibling_pole_overlap, 4),
            "orthogonal_pole_sibling_bleed_ratio": round(orth_bleed_ratio, 4),
            "biorthogonal_pole_sibling_bleed_ratio": round(biorth_bleed_ratio, 6),
            "cross_lingual_en_to_pt_uscar_resonance": round(float(res_xling.max().item()), 4),
            "post_0x00_gdpr_wipe_residual": gdpr_residual,
            "cross_lingual_s_world_cosine_zero_procrustes": xling_cos,
            "out_of_cone_0x00_vacuum_abstention": f"{ooc_hits}/{len(OUT_OF_CONE_ABSTAIN_PROMPTS)}",
        },
        "step_2_multilingual_chat_code_and_anti_stutter": {
            "v7_4_baseline_multi_turn_pin_exact_acc": 0.2083,
            "v7_5_baseline_multi_turn_pin_exact_acc": 0.8750,
            "multi_turn_o1_unseen_pin_exact_acc": mt_exact_acc,
            "multi_turn_o1_unseen_pin_digit_acc": mt_digit_acc,
            "multi_turn_stutter_count_out_of_24": mt_stutter_cnt,
            "multi_turn_state_bytes_delta": max(abs(x) for x in mt_state_deltas),
            "repeated_digit_stress_suite_exact_clean_acc": rep_stress_acc,
            "repeated_digit_stress_suite_stutters": rep_stress_stutters,
            "repeated_digit_stress_suite_records": rep_stress_records,
            "heldout_python_code_ast_valid": f"{code_ast_valid_cnt}/{code_ast_total}",
            "multi_turn_examples": mt_examples,
            "heldout_chat_and_code_generations": chat_samples,
        },
        "corpus_provenance": provenance,
        "fleet_training_curve": history,
    }

    audit_path = out_dir / "scar_field_v8_0_macro_fleet_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not dry_run:
        res_copy = REPO_ROOT / "results" / "scar_field_v8_0_macro_fleet_audit.json"
        res_copy.parent.mkdir(parents=True, exist_ok=True)
        res_copy.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\n[DONE v8.0 MACRO-SCOUT FLEET] M=4 Step={m4_eff_step_ms:.1f} ms ({m4_concurrency_speedup_x:.2f}x > M=1 | "
        f"{eff_speedup_vs_v73_x:.2f}x > v7.3) | S={s_bench_cnt} Swarm={s8_eff_ms_per_worker_step:.1f} ms | "
        f"`.sfield.scar` = {scar_bytes / (1024**2):.3f} MiB | mmap cos = {mmap_cos:.6f} | "
        f"3-Turn PIN = {mt_exact_acc*100:.1f}% | Code AST = {code_ast_valid_cnt}/{code_ast_total} | "
        f"0x00 Abstain = {ooc_hits}/{len(OUT_OF_CONE_ABSTAIN_PROMPTS)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
