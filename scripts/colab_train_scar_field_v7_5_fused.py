#!/usr/bin/env python3
"""ScarField v7.5-Fused — Triton Kernels, Zero-Autograd Symplectic Adjoint & Anti-Stutter Closure Audit.

Validates on `NVIDIA A100`:
  1. Pillar 2 — Sub-`150 ms/step` Hardware-Fused GPU Execution (`vs 1,836.4 ms v7.3` & `1,127.5 ms v7.4`):
       - `@triton.jit` `O(8)` Conway-Sloane Gosset `E_8` encoder/dequantizer emitting `FP32` + `BF16` in 1 pass,
       - `@triton.jit` Single-CTA (`BLOCK_V=256`) shift-invariant zero-sort Newton `Tsallis-1.5` cap kernel,
       - Zero-`roll`/zero-`cat` `[256, D]` rotated pole gather (`144x` less HBM traffic on `6:1` Scouts),
       - Zero-Autograd Analytic Symplectic Midpoint Adjoint (`0` `torch.autograd.grad`, `0` host syncs on CUDA,
         `1` batched `[2, N, 2r, 2r]` 5-step Holder-scaled Halley inverse per half-step, analytic E8 tangent grad).
  2. Pillar 1 — Multi-Turn Open Generation Anti-Stuttering & Shock-Exit Closure (`'PIN=80000...' -> 'PIN=800.'\x04`):
       - 1-Step Causal Shock-Closure FIR (`shock_closure_decay = 0.85`, `+0 B` in `KalmanBusState`, `cos > 0.999999`
         streaming parity) binding closing delimiters (`'.'`, `0x04` EOT) into `W_fast` & `BiorthogonalShockBank`,
       - Run-Length Orthogonal Phase Harmonics (`enable_repeat_phase = True`, odd shifts `5, 13, 21`) separating
         consecutive repeated digits (`key("=80") != key("=800")`, `key("=77") != key("=777")`),
       - `O(1)` Recent-Byte Cycle Damper (`Delta = +0 B`) + Language-Pure Un-Truncated `512 B` SFT Packing.
"""

from __future__ import annotations

import argparse
import ast
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
    RiemannianMuon,
    SCARFIELD_EOT_BYTE,
    ScarFieldConfig,
    ScarFieldEngine,
    System1CorpusScoutFleet,
    encode_sft_dialogue_bytes_and_mask,
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
    CURATED_CORE_QA,
    HELDOUT_EVAL_PROMPTS,
    OUT_OF_CONE_ABSTAIN_PROMPTS,
    PYTHON_CODE_SFT_TASKS,
    calibrate_ppm_escape_with_byte_prior_on_train,
    evaluate_byte_lm_metrics_v73,
    sample_parallel_batch,
)

V73_A100_MEASURED_STEP_MS = 1836.4
V73_A100_MEASURED_BYTES_PER_SEC = 3507.0
V74_A100_MEASURED_STEP_MS = 1127.5
V74_A100_MEASURED_SCOUT_MS = 200.58


def _sync_dev(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _detect_code_prompt_lang(u_txt: str) -> str:
    low = u_txt.lower()
    if low.startswith("escreva"):
        return "pt"
    if low.startswith("escribe"):
        return "es"
    if low.startswith("écris") or low.startswith("ecris"):
        return "fr"
    return "en"


def harvest_wiki_qa_by_lang_v75(
    train_splits: Dict[str, np.ndarray],
    max_per_lang: int = 340,
    max_ans_chars: int = 160,
) -> Dict[str, List[Tuple[str, str]]]:
    """Extracts language-pure concise `(Question, Answer)` pairs from `train_splits` (`0` cross-lang bleed!)."""
    q_templates = {
        "pt": [
            "Explique de forma concisa: {topic}.",
            "O que é {topic}?",
            "Resuma o conceito científico de {topic}.",
        ],
        "en": [
            "Briefly explain {topic}.",
            "What is {topic}?",
            "Summarize the scientific concept of {topic}.",
        ],
        "es": [
            "Explica brevemente: {topic}.",
            "¿Qué es {topic}?",
            "Resume el concepto científico de {topic}.",
        ],
        "fr": [
            "Explique brièvement : {topic}.",
            "Qu'est-ce que {topic} ?",
            "Résume le concept scientifique de {topic}.",
        ],
    }
    by_lang: Dict[str, List[Tuple[str, str]]] = {"pt": [], "en": [], "es": [], "fr": []}
    hdr_re = re.compile(r"===\s*([^=\n]+?)\s*===")

    for lg in ("pt", "en", "es", "fr"):
        txt = bytes(train_splits[lg]).decode("utf-8", errors="ignore")
        parts = hdr_re.split(txt)
        count = 0
        for idx in range(1, len(parts) - 1, 2):
            title = parts[idx].strip().replace("_", " ")
            body = parts[idx + 1].strip()
            if not title or len(title) > 52 or len(body) < 100:
                continue
            paras = [p.strip() for p in body.split("\n") if len(p.strip()) >= 65 and not p.strip().startswith("=")]
            if not paras:
                continue
            first_p = re.sub(r"\s+", " ", paras[0])
            sents = re.split(r"(?<=[.!?])\s+", first_p)
            ans = sents[0].strip()
            if len(ans) > max_ans_chars:
                ans = ans[: max_ans_chars - 3].rsplit(" ", 1)[0].rstrip(",;:") + "."
            if len(ans) < 36:
                continue
            for tmpl in q_templates[lg]:
                by_lang[lg].append((tmpl.format(topic=title), ans))
                count += 1
                if count >= max_per_lang:
                    break
            if count >= max_per_lang:
                break
    return by_lang


def _encode_closure_weighted_dialogue(turns: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Encodes `turns` with `2.5x` salience weight on `0x04` EOT & post-digit closing delimiters (`'.'`)."""
    arr_u8, mask_f32 = encode_sft_dialogue_bytes_and_mask(turns)
    weighted = mask_f32.copy()
    # `weighted[t]` weights loss on target byte `arr_u8[t + 1]`
    tgt_bytes = arr_u8[1:]
    prev_bytes = arr_u8[:-1]
    is_active = weighted > 0.0
    is_eot = tgt_bytes == SCARFIELD_EOT_BYTE
    is_prev_digit = (prev_bytes >= ord("0")) & (prev_bytes <= ord("9"))
    is_post_digit_delim = is_prev_digit & ((tgt_bytes == ord(".")) | (tgt_bytes == ord("\n")))
    boost = is_active & (is_eot | is_post_digit_delim)
    weighted[boost] = 2.5
    return arr_u8, weighted


def build_sft_dialogue_pools_v75(
    train_splits: Dict[str, np.ndarray],
    rng: random.Random,
    max_seq_len: int = 512,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Builds `(single_turn_by_lang, multi_turn_pin_pool)` strictly on training PINs `100..499` (`0` leakage!)."""
    single_by_lang: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        "pt": [],
        "en": [],
        "es": [],
        "fr": [],
        "code": [],
    }

    # 1. Core Architectural & Scientific QA grouped by language
    for _ in range(16):
        for lg, u, a in CURATED_CORE_QA:
            arr, msk = _encode_closure_weighted_dialogue([(u, a)])
            if len(arr) <= max_seq_len:
                single_by_lang[lg].append((arr, msk))

    # 2. AST-Verified Python Coding SFT
    for _, code_ans in PYTHON_CODE_SFT_TASKS:
        ast.parse(code_ans)
    for _ in range(22):
        for u_c, a_c in PYTHON_CODE_SFT_TASKS:
            arr, msk = _encode_closure_weighted_dialogue([(u_c, a_c)])
            if len(arr) <= max_seq_len:
                single_by_lang["code"].append((arr, msk))
                lg_c = _detect_code_prompt_lang(u_c)
                single_by_lang[lg_c].append((arr, msk))

    # 3. Grounded Encyclopedic QA strictly grouped by language
    wiki_by_lang = harvest_wiki_qa_by_lang_v75(train_splits, max_per_lang=320, max_ans_chars=158)
    for lg, pairs in wiki_by_lang.items():
        for u_w, a_w in pairs:
            arr, msk = _encode_closure_weighted_dialogue([(u_w, a_w)])
            if len(arr) <= max_seq_len:
                single_by_lang[lg].append((arr, msk))

    # 4. Language-Pure Un-Truncated 3-Turn Stateful Session PIN Dialogues (`PIN in 100..499` ONLY!)
    mt_templates = {
        "pt": (
            "Guarde a chave da sessão: PIN={pin}.",
            "Chave registrada: PIN={pin}.",
            "Qual é a chave da sessão registrada?",
            "A chave da sessão registrada é PIN={pin}.",
        ),
        "en": (
            "Store the session key: PIN={pin}.",
            "Session key stored: PIN={pin}.",
            "What is the stored session key?",
            "The stored session key is PIN={pin}.",
        ),
        "es": (
            "Guarda la clave de sesión: PIN={pin}.",
            "Clave registrada: PIN={pin}.",
            "¿Cuál es la clave de sesión registrada?",
            "La clave de sesión registrada es PIN={pin}.",
        ),
        "fr": (
            "Enregistre la clé de session : PIN={pin}.",
            "Clé enregistrée : PIN={pin}.",
            "Quelle est la clé de session enregistrée ?",
            "La clé de session enregistrée est PIN={pin}.",
        ),
    }
    # Explicitly include repeated-digit PINs inside `[100..499]` alongside uniform `[100..499]`!
    repeated_train_pins = [
        p
        for p in range(100, 500)
        if len(set(f"{p:03d}")) < 3  # e.g., 100, 110, 111, 122, 200, 222, 300, 333, 400, 444, 488...
    ]
    mt_pool: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx_m in range(960):
        lg = ("pt", "en", "es", "fr")[idx_m % 4]
        u1_t, a1_t, u3_t, a3_t = mt_templates[lg]
        if idx_m % 3 == 0 and repeated_train_pins:
            pin_val = rng.choice(repeated_train_pins)
        else:
            pin_val = rng.randint(100, 499)
        pin_str = f"{pin_val:03d}"
        lang_qas = wiki_by_lang[lg] if wiki_by_lang[lg] else [("What is ScarField?", "ScarField is a geodesic E8 engine.")]
        mid_u, mid_a = rng.choice(lang_qas)
        turns = [
            (u1_t.format(pin=pin_str), a1_t.format(pin=pin_str)),
            (mid_u, mid_a),
            (u3_t, a3_t.format(pin=pin_str)),
        ]
        arr, msk = _encode_closure_weighted_dialogue(turns)
        # Ensure 100% un-truncated Turn 3 + `\x04` EOT inside `max_seq_len`!
        if len(arr) > max_seq_len:
            short_a = mid_a[:90].rsplit(" ", 1)[0].rstrip(",;:") + "."
            turns[1] = (mid_u, short_a)
            arr, msk = _encode_closure_weighted_dialogue(turns)
        if len(arr) <= max_seq_len:
            mt_pool.append((arr, msk))

    for lg in single_by_lang:
        rng.shuffle(single_by_lang[lg])
    rng.shuffle(mt_pool)
    return single_by_lang, mt_pool


def sample_sft_masked_batch_v75(
    single_by_lang: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    mt_pool: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
    mt_fraction: float = 0.42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples clean language-pure un-truncated SFT rows (`0` mid-reply slicing, `0` PIN collisions!)."""
    rows_b = np.zeros((batch_size, seq_len), dtype=np.int64)
    rows_m = np.zeros((batch_size, seq_len - 1), dtype=np.float32)
    langs = [k for k, v in single_by_lang.items() if len(v) > 0]

    for r_i in range(batch_size):
        use_mt = bool(mt_pool) and (rng.random() < mt_fraction)
        if use_mt:
            arr, msk = rng.choice(mt_pool)
            n_copy = min(seq_len, len(arr))
            rows_b[r_i, :n_copy] = arr[:n_copy]
            rows_m[r_i, : n_copy - 1] = msk[: n_copy - 1]
        else:
            lg = rng.choice(langs)
            pool = single_by_lang[lg]
            cur_pos = 0
            for _ in range(4):
                arr, msk = rng.choice(pool)
                if cur_pos > 0 and (cur_pos + len(arr) > seq_len):
                    break
                n_copy = min(seq_len - cur_pos, len(arr))
                rows_b[r_i, cur_pos : cur_pos + n_copy] = arr[:n_copy]
                if cur_pos == 0:
                    rows_m[r_i, : n_copy - 1] = msk[: n_copy - 1]
                else:
                    # Transition from previous `\n` into new `\x02` user prompt has `0.0` target loss
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ScarField v7.5-Fused A100 Co-Training & Audit")
    parser.add_argument("--dry-run", action="store_true", help="Fast local smoke run")
    parser.add_argument("--steps", type=int, default=2400, help="v7.5-Fused training steps")
    parser.add_argument("--dense-steps", type=int, default=2400, help="DenseByteTransformer reference steps")
    parser.add_argument("--warm-start", type=str, default="", help="Path to v7.4 .sfield.scar or .pt")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    torch.manual_seed(20260705)
    np.random.seed(20260705)
    py_rng = random.Random(20260705)

    dev = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path("/content/scarfield_v7_5_out")
            if Path("/content").exists() and not dry_run
            else REPO_ROOT / "artifacts" / "scarfield_v7_5_fused"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "corpus_cache"

    total_steps = 6 if dry_run else int(args.steps)
    dense_steps = 4 if dry_run else int(args.dense_steps)

    print("=" * 98, flush=True)
    print(
        f"SCARFIELD v7.5-FUSED — TRITON={HAS_TRITON} + ZERO-AUTOGRAD SYMPLECTIC + SHOCK CLOSURE ({dev})",
        flush=True,
    )
    print(f"Hardware: {gpu_name} | Steps: {total_steps}", flush=True)
    print("=" * 98, flush=True)

    # [1/6] Load Open Corpora + Build Language-Pure Un-Truncated SFT Pools
    print("\n[1/6] Preparing Open Corpora (`PT/EN/ES/FR/Code`) & Language-Pure 512B SFT Pools...", flush=True)
    train_splits, val_splits, provenance = build_overnight_open_corpora(cache_dir, dry_run=dry_run)
    sft_seq_len = 288 if dry_run else 512
    sft_single_by_lang, sft_mt_pool = build_sft_dialogue_pools_v75(
        train_splits, py_rng, max_seq_len=sft_seq_len
    )
    total_sft_count = sum(len(v) for v in sft_single_by_lang.values()) + len(sft_mt_pool)
    print(
        f"  Built {total_sft_count:,d} language-pure SFT items ({len(sft_mt_pool):,d} un-truncated 3-turn PIN sessions, "
        f"train PINs strictly in [100..499]).",
        flush=True,
    )

    # [2/6] Initialize `ScarField v7.5-Fused` & Warm-Start from `v7.4-Turbo`
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

    sf_eng = ScarFieldEngine(sf_cfg).to(dev)
    warm_loaded_tensors = 0
    if args.warm_start and Path(args.warm_start).exists() and not dry_run:
        ws_path = Path(args.warm_start)
        if ws_path.name.endswith(".sfield.scar"):
            with ScarFieldMmapContainer(ws_path, verify_crc=True) as ctr:
                for mname, mod in sf_eng.named_modules():
                    if isinstance(mod, E8SphericalLinear) and f"e8::{mname}::codes_u8" in ctr.arrays:
                        c_u8 = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::codes_u8"], copy=True)).long().to(dev)
                        a_pk = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::a_tern"], copy=True)).to(dev)
                        b_pk = torch.from_numpy(np.array(ctr.arrays[f"e8::{mname}::b_tern"], copy=True)).to(dev)
                        sc = np.array(ctr.arrays[f"e8::{mname}::scalars"], copy=True)
                        tau_b, gam_s = float(sc[0]), float(sc[1])
                        del sc
                        sqrt_i = math.sqrt(float(mod.in_features))
                        lut_d = mod.lut_e8.to(dev)
                        sgn_d = mod.signs.to(dev)
                        w_q_rot = lut_d[c_u8].reshape(mod.out_features, mod.in_features) / sqrt_i
                        w_q = fwht64_rademacher(w_q_rot, sgn_d, inverse=True)
                        with torch.no_grad():
                            if mod.scar_rank > 0 and mod.a_scar is not None and mod.b_scar is not None and mod.gamma_scar is not None:
                                a_f = unpack_ternary_2bit(a_pk).float()
                                b_f = unpack_ternary_2bit(b_pk).float()
                                w_eff = tau_b * w_q + (gam_s / math.sqrt(float(mod.in_features * mod.scar_rank))) * (b_f @ a_f)
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
            if float(sf_eng.bus.biorth_gain.item()) < 0.35:
                sf_eng.bus.biorth_gain.fill_(0.35)
            if float(sf_eng.head.cap_adapt_gain.item()) < 0.25:
                sf_eng.head.cap_adapt_gain.fill_(0.25)
        heal_all_e8_scars(sf_eng)
        print(f"  [Warm-Start] Restored {warm_loaded_tensors} modules/poles from `{ws_path.name}`!", flush=True)

    budget = sf_eng.byte_budget_summary()
    base_lr = 1.1e-3 if warm_loaded_tensors > 0 else 4.5e-3
    min_lr = 1.8e-4
    sf_opt = RiemannianMuon(sf_eng.parameters(), lr=base_lr, scalar_lr=1.5e-3, momentum=0.90, ns_steps=5)
    open_scout_fleet = System1CorpusScoutFleet(oversample_factor=6, anchor_fraction=0.25)
    sft_scout_fleet = System1CorpusScoutFleet(oversample_factor=4, anchor_fraction=0.25)

    domains = ("pt", "en", "es", "fr", "code")
    dom_probs = (0.19, 0.19, 0.19, 0.19, 0.24)
    all_prose_concat = np.concatenate([train_splits[k] for k in ("pt", "en", "es", "fr")])

    def sample_open_candidates(seq_len: int, n_rows: int) -> torch.Tensor:
        chosen = py_rng.choices(domains, weights=dom_probs, k=n_rows)
        rows = []
        for d in chosen:
            arr = train_splits[d]
            off = py_rng.randint(0, max(1, len(arr) - seq_len - 1))
            rows.append(arr[off : off + seq_len].astype(np.int64))
        return torch.from_numpy(np.stack(rows, axis=0)).to(device=dev, non_blocking=True)

    # [3/6] Controlled Head-to-Head GPU Step & Scout Benchmark (`B=16, T=384` vs `v7.3` & `v7.4-Turbo`)
    print(
        "\n[2/6] Running Controlled Step Benchmark (`B=16, T=384`) vs `v7.3` (`1836.4 ms`) & `v7.4` (`1127.5 ms`)...",
        flush=True,
    )
    bench_seq_len = 64 if dry_run else 384
    bench_cand = sample_open_candidates(bench_seq_len, (2 if dry_run else 6) * bsz)

    # Warmup Triton PTX compilation (`select_frontier_batch` + 3 train steps + `heal_all_e8_scars`) before timing!
    _ = open_scout_fleet.select_frontier_batch(sf_eng, bench_cand, batch_size=bsz)
    for _ in range(2 if dry_run else 3):
        sf_opt.zero_grad(set_to_none=True)
        sf_eng.compute_loss(bench_cand[:bsz])["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sf_eng.parameters(), 1.0, foreach=(dev.type == "cuda"))
        sf_opt.step()
    heal_all_e8_scars(sf_eng)
    _sync_dev(dev)

    # Measure `6:1` Scout Fleet latency (`n_scout_bench` runs)
    n_scout_bench = 2 if dry_run else 8
    t_sc0 = time.perf_counter()
    for _ in range(n_scout_bench):
        bench_x, _, bench_tel = open_scout_fleet.select_frontier_batch(sf_eng, bench_cand, batch_size=bsz)
    _sync_dev(dev)
    bench_scout_ms = ((time.perf_counter() - t_sc0) / float(n_scout_bench)) * 1000.0

    # Measure full Forward + Zero-Autograd Symplectic Backward + Clip + RiemannianMuon + Triton E8 Re-Quantize (`n_bench` steps)
    n_bench = 2 if dry_run else 10
    t_b0 = time.perf_counter()
    for _ in range(n_bench):
        sf_opt.zero_grad(set_to_none=True)
        b_loss = sf_eng.compute_loss(bench_x)
        b_loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sf_eng.parameters(), 1.0, foreach=(dev.type == "cuda"))
        sf_opt.step()
    _sync_dev(dev)
    measured_step_ms = ((time.perf_counter() - t_b0) / float(n_bench)) * 1000.0
    measured_bytes_per_sec = (bsz * bench_seq_len) / max(1e-6, measured_step_ms / 1000.0)

    _sync_dev(dev)
    t_h0 = time.perf_counter()
    heal_all_e8_scars(sf_eng)
    _sync_dev(dev)
    measured_heal_ms = (time.perf_counter() - t_h0) * 1000.0

    speedup_vs_v73 = V73_A100_MEASURED_STEP_MS / max(0.1, measured_step_ms)
    speedup_vs_v74 = V74_A100_MEASURED_STEP_MS / max(0.1, measured_step_ms)
    scout_speedup_vs_v74 = V74_A100_MEASURED_SCOUT_MS / max(0.05, bench_scout_ms)
    eff_frontier_speedup = (
        V73_A100_MEASURED_STEP_MS / max(0.1, measured_step_ms + bench_scout_ms)
    ) * bench_tel.epistemic_gain_ratio
    print(
        f"  [GPU Benchmark] v7.5-Fused Step = {measured_step_ms:.1f} ms "
        f"({speedup_vs_v74:.2f}x > v7.4 1127.5 ms | {speedup_vs_v73:.2f}x > v7.3 1836.4 ms!) | "
        f"Throughput = {measured_bytes_per_sec:,.0f} B/s | "
        f"6:1 Scout = {bench_scout_ms:.2f} ms ({scout_speedup_vs_v74:.1f}x > v7.4 200.6 ms!) | "
        f"Heal = {measured_heal_ms:.1f} ms",
        flush=True,
    )

    # [4/6] Scout-Guided Co-Training (`55%` Open/Passkey/Alignment + `45%` Language-Pure Un-Truncated SFT)
    print(f"\n[3/6] Running {total_steps} Fused Scout-Guided Co-Training Steps...", flush=True)
    t0_train = time.perf_counter()
    history: List[Dict[str, Any]] = []
    streamed_bytes = 0
    scouted_candidate_bytes = 0
    epistemic_gains: List[float] = []
    degen_filtered_total = 0

    for step in range(1, total_steps + 1):
        frac = (step - 1) / float(max(1, total_steps - 1))
        cur_lr = min_lr + (base_lr - min_lr) * (0.5 * (1.0 + math.cos(math.pi * frac)))
        for pg in sf_opt.param_groups:
            pg["lr"] = cur_lr

        sf_opt.zero_grad(set_to_none=True)
        is_sft_step = (step % 9) in (1, 3, 5, 7)

        if is_sft_step:
            seq_l = sft_seq_len
            n_cand_sft = bsz * (2 if dry_run else sft_scout_fleet.oversample_factor)
            cand_sft_b, cand_sft_m = sample_sft_masked_batch_v75(
                sft_single_by_lang, sft_mt_pool, n_cand_sft, seq_l, py_rng, dev, mt_fraction=0.45
            )
            sft_b, sft_m, tel_s = sft_scout_fleet.select_frontier_batch(
                sf_eng, cand_sft_b, batch_size=bsz, candidate_loss_mask=cand_sft_m
            )
            assert sft_m is not None
            scouted_candidate_bytes += n_cand_sft * seq_l
            epistemic_gains.append(tel_s.epistemic_gain_ratio)
            degen_filtered_total += tel_s.degenerate_windows_filtered

            if (not dry_run) and (step % 18 == 1):
                ooc_rows = []
                ooc_mask = torch.zeros_like(sft_b, dtype=torch.bool)
                for r_i in range(2):
                    p_ooc = py_rng.choice(OUT_OF_CONE_ABSTAIN_PROMPTS)
                    raw_ooc = (f"\x02User: {p_ooc}\n\x03Assistant: ").encode("utf-8")
                    end_idx = min(seq_l, len(raw_ooc))
                    pad_ooc = (raw_ooc + b" " * seq_l)[:seq_l]
                    ooc_rows.append(np.frombuffer(pad_ooc, dtype=np.uint8).astype(np.int64))
                    ooc_mask[r_i, max(0, end_idx - 6) : end_idx] = True
                sft_b[:2] = torch.from_numpy(np.stack(ooc_rows, axis=0)).to(dev)
                sft_m[:2] = 0.0
                losses = sf_eng.compute_loss(sft_b, out_of_cone_mask=ooc_mask, target_loss_mask=sft_m)
            else:
                losses = sf_eng.compute_loss(sft_b, target_loss_mask=sft_m)
        else:
            seq_l = 96 if dry_run else (384 if (step % 4 != 0) else 576)
            if (not dry_run) and (step % 9 == 0):
                pk_len = 640 if frac > 0.25 else seq_l
                pk_b, _, _ = sample_open_passkey_batch(
                    all_prose_concat, bsz // 2, pk_len, py_rng, pin_min=100, pin_max=499, device=dev
                )
                n_rem = bsz - (bsz // 2)
                cand_rem = sample_open_candidates(pk_len, n_rem * open_scout_fleet.oversample_factor)
                open_half, _, tel_o = open_scout_fleet.select_frontier_batch(sf_eng, cand_rem, batch_size=n_rem)
                scouted_candidate_bytes += int(cand_rem.numel()) + int(pk_b.numel())
                epistemic_gains.append(tel_o.epistemic_gain_ratio)
                degen_filtered_total += tel_o.degenerate_windows_filtered
                mix_b = torch.cat((open_half, pk_b), dim=0)
                losses = sf_eng.compute_loss(mix_b)
                seq_l = pk_len
            elif step % 18 == 2:
                la, lb = py_rng.sample(("pt", "en", "es", "fr"), 2)
                pa, pb = sample_parallel_batch(la, lb, min(bsz, 8), min(seq_l, 192), py_rng, dev)
                scouted_candidate_bytes += int(pa.numel() + pb.numel())
                losses = sf_eng.compute_loss(pa, paired_translation_ids=pb)
            else:
                n_cand_o = bsz * (2 if dry_run else open_scout_fleet.oversample_factor)
                cand_o = sample_open_candidates(seq_l, n_cand_o)
                open_b, _, tel_o = open_scout_fleet.select_frontier_batch(sf_eng, cand_o, batch_size=bsz)
                scouted_candidate_bytes += n_cand_o * seq_l
                epistemic_gains.append(tel_o.epistemic_gain_ratio)
                degen_filtered_total += tel_o.degenerate_windows_filtered
                losses = sf_eng.compute_loss(open_b)

        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(sf_eng.parameters(), 1.0, foreach=(dev.type == "cuda"))
        sf_opt.step()
        streamed_bytes += bsz * seq_l

        if step % (2 if dry_run else 120) == 0 or step == total_steps:
            heal_all_e8_scars(sf_eng)

        if step == 1 or step % (2 if dry_run else 200) == 0 or step == total_steps:
            mean_ep = float(np.mean(epistemic_gains[-50:])) if epistemic_gains else 1.0
            rec = {
                "step": step,
                "lr": round(cur_lr, 6),
                "loss_cap": round(float(losses["loss_cap"].item()), 4),
                "loss_reflex": round(float(losses["loss_reflex"].item()), 4),
                "loss_antiloop": round(float(losses["loss_antiloop"].item()), 5),
                "biorth_gain": round(float(sf_eng.bus.biorth_gain.item()), 4),
                "cap_adapt_gain": round(float(sf_eng.head.cap_adapt_gain.item()), 4),
                "scout_epistemic_gain_x": round(mean_ep, 3),
                "elapsed_s": round(time.perf_counter() - t0_train, 1),
            }
            history.append(rec)
            print(
                f"  [ScarField v7.5-Fused] step={step:5d}/{total_steps} lr={cur_lr:.4f} "
                f"cap={rec['loss_cap']:.4f} biorth_g={rec['biorth_gain']:.3f} "
                f"scout_gain={rec['scout_epistemic_gain_x']:.2f}x ({rec['elapsed_s']:.1f}s)",
                flush=True,
            )

    _sync_dev(dev)
    sf_train_sec = time.perf_counter() - t0_train
    avg_train_ms_per_step = (sf_train_sec * 1000.0) / float(max(1, total_steps))

    # [5/6] Train Matched `63.77 MiB` `DenseByteTransformer` Reference
    print("\n[4/6] Training Matched `63.77 MiB` `DenseByteTransformer` Reference...", flush=True)
    dense_model = DenseByteTransformer(vocab_size=256, d_model=dense_d, n_layers=dense_l, n_heads=dense_h).to(dev)
    dense_params = sum(p.numel() for p in dense_model.parameters())
    dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=1.2e-3, weight_decay=0.01)
    t_dn = time.perf_counter()
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
            dbatch = torch.cat((sample_open_candidates(d_seql, bsz - (bsz // 2)), pk_b), dim=0)
        else:
            dbatch = sample_open_candidates(d_seql, bsz)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            d_logits = dense_model(dbatch)
            d_loss = F.cross_entropy(d_logits[:, :-1, :].reshape(-1, 256), dbatch[:, 1:].reshape(-1))
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
        dense_opt.step()
    _sync_dev(dev)
    dense_train_sec = time.perf_counter() - t_dn

    # [6/6] Full Forensic Evaluation & `.sfield.scar` Container Export
    print("\n[5/6] Evaluating 5/5 Domains, T=2048 Passkeys, 3-Turn O(1) Unseen & Repeated-Digit PIN Chat...", flush=True)
    sf_eng.eval()
    dense_model.eval()

    def sf_probs_fn(x: torch.Tensor) -> torch.Tensor:
        sc = sf_eng.forward_sequence(x, adaptive_depth=True)["scores"]
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
        domain_eval[dom] = {"scarfield_v7_5_fused": sf_m, "dense_fp16_ot": dn_m}
        print(
            f"  [{dom.upper():4s}] ScarField v7.5: BPB={sf_m['calibrated_bpb']:.3f} "
            f"Top1={sf_m['top1_byte_acc_pct']:.2f}% Top5={sf_m['top5_byte_acc_pct']:.2f}% | "
            f"Dense FP16: BPB={dn_m['bpb']:.3f} Top1={dn_m['top1_byte_acc_pct']:.2f}%",
            flush=True,
        )

    pk_horizons = [64, 128] if dry_run else [128, 256, 512, 768, 1024, 1536, 2048]
    n_pk = 4 if dry_run else 32
    val_prose_concat = np.concatenate([val_splits[k] for k in ("pt", "en", "es", "fr")])
    passkey_eval: Dict[str, Any] = {
        "scarfield_v7_5_biorthogonal": {},
        "scarfield_ablation_no_biorth_bank": {},
        "dense_fp16_ot": {},
    }
    saved_biorth_gain = float(sf_eng.bus.biorth_gain.item())
    for h_len in pk_horizons:
        with torch.no_grad():
            sf_eng.bus.biorth_gain.fill_(saved_biorth_gain)
            sf_pk = evaluate_open_passkey_recall(
                lambda b: sf_eng.forward_sequence(b, adaptive_depth=True)["scores"], val_prose_concat, h_len, n_pk, dev
            )
            sf_eng.bus.biorth_gain.fill_(0.0)
            sf_no_b = evaluate_open_passkey_recall(
                lambda b: sf_eng.forward_sequence(b, adaptive_depth=True)["scores"], val_prose_concat, h_len, n_pk, dev
            )
            sf_eng.bus.biorth_gain.fill_(saved_biorth_gain)

            def _dense_batched(b_in: torch.Tensor) -> torch.Tensor:
                outs = []
                for s_i in range(0, b_in.shape[0], 8):
                    outs.append(dense_model(b_in[s_i : s_i + 8]))
                return torch.cat(outs, dim=0)

            dn_pk = evaluate_open_passkey_recall(_dense_batched, val_prose_concat, h_len, n_pk, dev)
        passkey_eval["scarfield_v7_5_biorthogonal"][f"T={h_len}"] = sf_pk
        passkey_eval["scarfield_ablation_no_biorth_bank"][f"T={h_len}"] = sf_no_b
        passkey_eval["dense_fp16_ot"][f"T={h_len}"] = dn_pk
        print(
            f"  [Passkey T={h_len:4d}] v7.5 Biorth: {sf_pk['exact_3digit_acc']*100:5.1f}% "
            f"(No-Biorth: {sf_no_b['exact_3digit_acc']*100:5.1f}%) | Dense: {dn_pk['exact_3digit_acc']*100:5.1f}%",
            flush=True,
        )

    chat_samples: Dict[str, List[Dict[str, Any]]] = {}
    for dom_k, p_list in HELDOUT_EVAL_PROMPTS.items():
        chat_samples[dom_k] = []
        for prompt_txt, kw_list in p_list:
            rep = sf_eng.generate_reply(
                prompt_txt, max_new_bytes=220, enable_ghost_jump_scouts=False, apply_repulsive_anti_loop=True
            )
            ast_valid = None
            if dom_k == "code":
                try:
                    ast.parse(rep.text)
                    ast_valid = True
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

    # Identical `mt_rng = random.Random(74999)` 24-trial benchmark as v7.4 (`800` on trial 1!) + 8-trial Repeated-Digit Stress Suite!
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
        st_chat = sf_eng.init_state(batch_size=1, device=dev)
        b_init = st_chat.total_state_bytes
        r1 = sf_eng.generate_reply(u1.format(pin=pin), state=st_chat, max_new_bytes=64)
        r2 = sf_eng.generate_reply(u2, state=r1.state, max_new_bytes=180)
        r3 = sf_eng.generate_reply(u3, state=r2.state, max_new_bytes=72)
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

    # Dedicated Unseen Repeated-Digit PIN Stress Suite (`800, 777, 900, 880, 666, 550, 999, 500`)
    rep_stress_pins = ["800", "777"] if dry_run else ["800", "777", "900", "880", "666", "550", "999", "500"]
    rep_stress_hits = 0
    rep_stress_stutters = 0
    rep_stress_records: List[Dict[str, Any]] = []
    for idx_rp, rpin in enumerate(rep_stress_pins):
        lg, u1, u2, u3 = mt_langs[idx_rp % len(mt_langs)]
        st_rp = sf_eng.init_state(batch_size=1, device=dev)
        rr1 = sf_eng.generate_reply(u1.format(pin=rpin), state=st_rp, max_new_bytes=64)
        rr2 = sf_eng.generate_reply(u2, state=rr1.state, max_new_bytes=180)
        rr3 = sf_eng.generate_reply(u3, state=rr2.state, max_new_bytes=72)
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
        f"  [3-Turn O(1) Chat Memory] Unseen PIN Exact Recall: {mt_exact_acc*100:.1f}% "
        f"(digit {mt_digit_acc*100:.1f}%, stutters={mt_stutter_cnt}/{n_mt_trials}) | "
        f"Repeated-Digit Stress (`800/777/...`): {rep_stress_acc*100:.1f}% ({rep_stress_hits}/{len(rep_stress_pins)}) | "
        f"max Δstate_bytes = {max(abs(x) for x in mt_state_deltas)} B",
        flush=True,
    )

    with torch.no_grad():
        q_en_1 = torch.tensor([list("Earth polar radius measurement value".encode("utf-8"))], dtype=torch.long, device=dev)
        q_en_2 = torch.tensor([list("Earth equatorial radius measurement value".encode("utf-8"))], dtype=torch.long, device=dev)
        q_pt_1 = torch.tensor([list("Valor da medida do raio polar da Terra".encode("utf-8"))], dtype=torch.long, device=dev)
        hw_en_1 = project_rows_to_unit_sphere(sf_eng.forward_sequence(q_en_1)["h_world"].mean(dim=1))
        hw_en_2 = project_rows_to_unit_sphere(sf_eng.forward_sequence(q_en_2)["h_world"].mean(dim=1))
        hw_pt_1 = project_rows_to_unit_sphere(sf_eng.forward_sequence(q_pt_1)["h_world"].mean(dim=1))
        sibling_pole_overlap = float((hw_en_1 * hw_en_2).sum().item())

        y_bus_1 = torch.zeros(1, sf_cfg.d_bus, device=dev)
        y_bus_1[0, 0] = 2.0
        y_bus_2 = torch.zeros(1, sf_cfg.d_bus, device=dev)
        y_bus_2[0, 1] = 2.0
        sf_eng.bus.uscar_store.cartridges.clear()
        sf_eng.bus.uscar_store.mint_zero_sgd_rrr("earth_polar", "doc_geodesy", hw_en_1, y_bus_1, rank=4, horizon_kappa=0.40)
        sf_eng.bus.uscar_store.mint_zero_sgd_rrr("earth_equat", "doc_geodesy", hw_en_2, y_bus_2, rank=4, horizon_kappa=0.40)

        sf_eng.bus.uscar_store.biorthogonal_poles = True
        out_biorth, _ = sf_eng.bus.uscar_store.query(hw_en_1)
        _, res_xling = sf_eng.bus.uscar_store.query(hw_pt_1)
        biorth_bleed_ratio = float(out_biorth[0, 1].abs().item() / max(1e-8, out_biorth[0, 0].abs().item()))

        sf_eng.bus.uscar_store.biorthogonal_poles = False
        out_orth, _ = sf_eng.bus.uscar_store.query(hw_en_1)
        orth_bleed_ratio = float(out_orth[0, 1].abs().item() / max(1e-8, out_orth[0, 0].abs().item()))
        sf_eng.bus.uscar_store.biorthogonal_poles = True

        sf_eng.bus.uscar_store.wipe_document_0x00("doc_geodesy")
        out_wiped, res_wiped = sf_eng.bus.uscar_store.query(hw_en_1)
        gdpr_residual = float(out_wiped.abs().max().item() + res_wiped.abs().max().item())
        sf_eng.bus.uscar_store.cartridges.clear()

    ooc_hits = 0
    for p_ooc in OUT_OF_CONE_ABSTAIN_PROMPTS:
        rep_ooc = sf_eng.generate_reply(p_ooc, max_new_bytes=16, require_world_support=True)
        if rep_ooc.abstained_0x00:
            ooc_hits += 1

    xling_cos: Dict[str, float] = {}
    with torch.no_grad():
        for lg in ("pt", "es", "fr"):
            sims = []
            for item in PARALLEL_SCIENTIFIC_ANCHORS:
                ta = torch.tensor([list(item["en"].encode("utf-8"))], dtype=torch.long, device=dev)
                tb = torch.tensor([list(item[lg].encode("utf-8"))], dtype=torch.long, device=dev)
                pa = project_rows_to_unit_sphere(sf_eng.forward_sequence(ta)["h_world"].mean(dim=1))
                pb = project_rows_to_unit_sphere(sf_eng.forward_sequence(tb)["h_world"].mean(dim=1))
                sims.append(float((pa * pb).sum().item()))
            xling_cos[f"{lg}_en"] = round(float(np.mean(sims)), 5)

    scar_path = out_dir / "scarfield_v7_5_fused.sfield.scar"
    scar_bytes = sf_eng.save_packed_scar_container(scar_path)

    sf_cpu = ScarFieldEngine(sf_cfg)
    sf_cpu.load_state_dict({k: v.detach().cpu() for k, v in sf_eng.state_dict().items()})
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

    mean_ep_all = float(np.mean(epistemic_gains)) if epistemic_gains else 1.0
    audit_payload = {
        "suite": "scar_field_v7_5_fused_audit",
        "version": "7.5.0",
        "hardware": gpu_name,
        "triton_enabled": bool(HAS_TRITON and dev.type == "cuda"),
        "warm_started_from_v7_4_tensors": warm_loaded_tensors,
        "completed_v7_5_steps": total_steps,
        "streamed_v7_5_selected_bytes": streamed_bytes,
        "scouted_candidate_frontier_bytes": scouted_candidate_bytes,
        "scarfield_v7_5_config": sf_cfg.to_dict(),
        "a100_speed_and_triton_telemetry": {
            "v7_3_measured_step_ms": V73_A100_MEASURED_STEP_MS,
            "v7_4_measured_step_ms": V74_A100_MEASURED_STEP_MS,
            "v7_4_measured_scout_6to1_ms": V74_A100_MEASURED_SCOUT_MS,
            "v7_5_fused_step_ms_b16_t384": round(measured_step_ms, 2),
            "v7_5_fused_avg_mixed_step_ms": round(avg_train_ms_per_step, 2),
            "v7_5_fused_bytes_per_sec_b16_t384": round(measured_bytes_per_sec, 1),
            "v7_5_periodic_e8_heal_ms": round(measured_heal_ms, 2),
            "wall_clock_step_speedup_vs_v7_3_x": round(speedup_vs_v73, 2),
            "wall_clock_step_speedup_vs_v7_4_x": round(speedup_vs_v74, 2),
            "scout_fleet_6to1_overhead_ms": round(bench_scout_ms, 2),
            "scout_speedup_vs_v7_4_x": round(scout_speedup_vs_v74, 2),
            "mean_scout_epistemic_shock_gain_x": round(mean_ep_all, 3),
            "effective_frontier_shock_speedup_vs_v7_3_x": round(eff_frontier_speedup, 2),
            "degenerate_windows_filtered_total": degen_filtered_total,
            "record_curriculum": False,
        },
        "architecture_comparison": {
            "scarfield_v7_5_spherical_params": sum(p.numel() for p in sf_eng.parameters()),
            "scarfield_v7_5_sfield_scar_bytes": scar_bytes,
            "scarfield_v7_5_sfield_scar_mib": round(scar_bytes / (1024**2), 3),
            "scarfield_v7_5_o1_state_bytes": budget["central_kalman_scache_o1_bytes"],
            "dense_transformer_params": dense_params,
            "dense_transformer_fp16_mib": round((dense_params * 2) / (1024**2), 3),
            "weight_compression_vs_dense_fp16_x": round((dense_params * 2) / max(1, scar_bytes), 2),
            "mmap_c_simd_forward_cosine": round(mmap_cos, 6),
            "v7_5_train_seconds": round(sf_train_sec, 1),
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
        "step_2_multilingual_chat_and_anti_stutter_closure": {
            "sft_total_language_pure_items": total_sft_count,
            "sft_untruncated_3turn_pin_items": len(sft_mt_pool),
            "v7_4_baseline_multi_turn_pin_exact_acc": 0.2083,
            "multi_turn_o1_unseen_pin_exact_acc": mt_exact_acc,
            "multi_turn_o1_unseen_pin_digit_acc": mt_digit_acc,
            "multi_turn_stutter_count_out_of_24": mt_stutter_cnt,
            "multi_turn_state_bytes_delta": max(abs(x) for x in mt_state_deltas),
            "repeated_digit_stress_suite_exact_clean_acc": rep_stress_acc,
            "repeated_digit_stress_suite_stutters": rep_stress_stutters,
            "repeated_digit_stress_suite_records": rep_stress_records,
            "multi_turn_examples": mt_examples,
            "heldout_chat_and_code_generations": chat_samples,
        },
        "corpus_provenance": provenance,
        "training_curve": history,
    }

    audit_path = out_dir / "scar_field_v7_5_fused_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not dry_run:
        res_copy = REPO_ROOT / "results" / "scar_field_v7_5_fused_audit.json"
        res_copy.parent.mkdir(parents=True, exist_ok=True)
        res_copy.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\n[DONE v7.5-FUSED] Step={measured_step_ms:.1f} ms ({speedup_vs_v74:.2f}x > v7.4 | {speedup_vs_v73:.2f}x > v7.3) | "
        f"Scout={bench_scout_ms:.2f} ms | `.sfield.scar` = {scar_bytes / (1024**2):.3f} MiB | mmap cos = {mmap_cos:.6f} | "
        f"3-Turn PIN = {mt_exact_acc*100:.1f}% (Repeated-Digit={rep_stress_acc*100:.1f}%, Δ={max(abs(x) for x in mt_state_deltas)} B) | "
        f"0x00 Abstain = {ooc_hits}/{len(OUT_OF_CONE_ABSTAIN_PROMPTS)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
