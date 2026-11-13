#!/usr/bin/env python3
"""
========================================================================================
🔥 ADVERSARIAL RED-TEAM SUITE (`v2.2.0`): `Qwen2.5-0.5B` vs `Qwen2.5-3B` NEURAL CORE
   (`MoS` + `SCache O(1)` + `HeadExpandedScarStore` Zero-Mid-Cut Dynamic Attention)
========================================================================================
Executes 6 identical, merciless Red-Team attack vectors against BOTH pretrained cores:
  - Core A: `Qwen/Qwen2.5-0.5B-Instruct` (`24L, d=896, 14x64 Q-heads, 62,720 B = 61.25 KB INT4`, `802,816` Kronecker cells)
  - Core B: `Qwen/Qwen2.5-3B-Instruct`   (`36L, d=2048, 16x128 Q-heads, 143,360 B = 140.00 KB INT4 = 35x4KB mmap`, `4,194,304` cells)

Mandate (`AGENTS.md` / `SKILL.md` — Zero Facades):
  - Recursive `untyped_storage().data_ptr()` byte accounting (`0 B` unreported `O(T)` cache growth).
  - Mid-flight tensor corruption causality checks (`SCache`, `MoS` ASVD scars, and `HeadExpandedScarCapsule`).
  - Honest reporting of every cosine, NLL, Top-1%, latency, and lexical-trap outcome.
========================================================================================
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
    HeadExpandedScarStore,
)
from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer
from scripts.adversarial_mos_scache_suite import (
    ADVERSARIAL_DOMAIN_TEXTS,
    build_long_callback_document,
    convert_llm_mlps_to_mos,
    recursive_object_tensor_bytes,
)
from scripts.benchmark_scar_grafting_v2 import (
    BACKGROUND_CONTROL_PROMPTS,
    EPISODE_CACHE_PATH,
    RESULTS_DIR,
    _kl_div_top_union,
    format_chat,
)

DEFAULT_OUTPUT_JSON = RESULTS_DIR / "adversarial_0p5b_vs_3b_audit.json"

LEXICAL_TRAP_PROMPTS: List[str] = [
    (
        "In PostgreSQL, write an UPDATE statement to irreversibly erase m bits from a sensor "
        "bitmask column recorded at ambient temperature T without exceeding the integer upper bound."
    ),
    (
        "In CSS Grid and Flexbox, how should circular icon badges and unit spheres be packed inside "
        "a responsive container so layout space complementation never overflows?"
    ),
    (
        "When baking artisan sourdough bread at oven temperature T, how much heat escapes when opening "
        "the oven door for ten seconds during crust bound caramelization?"
    ),
    (
        "In a Python 3 warehouse packing script, pack spherical ball bearings into cubic crates and "
        "log how many bits of telemetry metadata each shipment manifest uses."
    ),
    (
        "Explain how git rebase --interactive erases temporary fixup commits in local workspace "
        "history while preserving upstream branch bounds."
    ),
    (
        "Configure a Linux nftables firewall rule that drops nondeterministic UDP bursts whenever "
        "socket buffer space approaches its configured limit."
    ),
]


@torch.no_grad()
def _scache_generate(
    model: nn.Module,
    tok: Any,
    prompt_ids_2d: torch.Tensor,
    max_new_tokens: int = 44,
    window_size: int = 128,
    num_slots: int = 64,
) -> Dict[str, Any]:
    dev = next(model.parameters()).device
    scache = SCacheDynamicCache(window_size=window_size, num_slots=num_slots, num_sink_tokens=4, project_slots=True)
    eos_id = tok.eos_token_id
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    ids = prompt_ids_2d.to(dev)
    # Chunked prefill if long context so SCache eviction + Widrow-Hoff slot writes genuinely trigger!
    if ids.shape[1] > window_size + 16:
        for s in range(0, ids.shape[1] - 1, 64):
            e = min(s + 64, ids.shape[1] - 1)
            _ = model(ids[:, s:e], past_key_values=scache, use_cache=True)
        out = model(ids[:, -1:], past_key_values=scache, use_cache=True)
    else:
        out = model(ids, past_key_values=scache, use_cache=True)
    nxt = int(out.logits[0, -1].argmax(dim=-1).item())
    gen: List[int] = [nxt]
    while len(gen) < max_new_tokens and nxt != eos_id:
        step_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        out = model(step_in, past_key_values=scache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax(dim=-1).item())
        if nxt == eos_id:
            break
        gen.append(nxt)
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    actual_kb = recursive_object_tensor_bytes(scache) / 1024.0
    rep_kb = scache.total_memory_bytes / 1024.0
    return {
        "text": tok.decode(gen, skip_special_tokens=True).strip(),
        "tokens": len(gen),
        "tok_per_s": round(len(gen) / dt, 2),
        "scache_actual_kb": round(actual_kb, 2),
        "scache_reported_kb": round(rep_kb, 2),
    }


@torch.no_grad()
def audit_single_core_adversarial(
    model_id: str,
    expected_capsule_bytes: int,
    expected_grid_cells: int,
    expected_bitmap_bytes: int,
    mos_tail_layers: int = 6,
) -> Dict[str, Any]:
    t_core0 = time.perf_counter()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "mps" else torch.float32
    print("\n" + "=" * 96, flush=True)
    print(f"🛡️ RED-TEAM CORE AUDIT: {model_id} on {dev} ({dtype})", flush=True)
    print("=" * 96, flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    slm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=True).to(dev).eval()
    payload = torch.load(EPISODE_CACHE_PATH, map_location="cpu", weights_only=False)

    n_layers = len(slm.model.layers)
    d_model = int(slm.config.hidden_size)
    n_heads = int(slm.config.num_attention_heads)
    d_head = d_model // n_heads

    # Calibrate & solve the 3 real 72.71B capsules (`ZERO Backpropagation`)
    head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=BACKGROUND_CONTROL_PROMPTS,
        tokenizer=tok,
        chat_formatter=format_chat,
        preserved_rank=4,
        attention_beta=14.0,
    )
    solve_times_ms: Dict[str, float] = {}
    for cid, dinfo in payload["domains"].items():
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=dinfo["episodes"],
            student_model=slm,
            head_bank=head_bank,
            rank=60,
            num_key_prototypes=8,
            logit_gain=1.30,
            tau=0.62,
            ramp=0.16,
        )
        raw = cap.to_int4_bytes()
        assert len(raw) == expected_capsule_bytes, f"Expected {expected_capsule_bytes} B, got {len(raw)}"
        cap_q = HeadExpandedScarCapsule.from_int4_bytes(
            raw,
            capsule_id=cid,
            hidden_dim=d_model,
            num_heads=n_heads,
            head_dim=d_head,
            rank=60,
            num_key_prototypes=8,
            tau=0.62,
            ramp=0.16,
            metadata=cap.metadata,
        )
        head_bank.add_capsule(cap_q)
        solve_times_ms[cid] = cap.metadata["solve_ms"]

    core_report: Dict[str, Any] = {
        "model_id": model_id,
        "layers": n_layers,
        "hidden_dim": d_model,
        "num_q_heads": n_heads,
        "head_dim": d_head,
        "capsule_int4_bytes": expected_capsule_bytes,
        "capsule_int4_kb": round(expected_capsule_bytes / 1024.0, 2),
        "exact_4kb_page_multiple": (expected_capsule_bytes % 4096 == 0),
        "solve_ms_by_capsule": solve_times_ms,
        "attacks": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        store_path = tmp_root / "core_full.scarstore"
        store = HeadExpandedScarStore.build_from_bank(head_bank, store_path, force_kronecker=True)
        try:
            assert store.num_grid_cells == expected_grid_cells
            assert store.presence_bitmap_bytes == expected_bitmap_bytes
            assert store.record_bytes == expected_capsule_bytes

            # ==================================================================
            # [ATTACK 1] ANTI-FACADE BYTE-POINTER AUDIT, O(1) INVARIANCE & CAUSALITY
            # ==================================================================
            print(f"[Attack 1 | {model_id}] Anti-Facade Byte Audit, O(1) Drift & Active Causality...", flush=True)
            long_doc = build_long_callback_document() + "\n" + (ADVERSARIAL_DOMAIN_TEXTS["physics_math"] * 3)
            long_ids = tok(long_doc, return_tensors="pt").input_ids.to(dev)
            assert long_ids.shape[1] >= 1536, f"Expected >=1536 tokens, got {long_ids.shape[1]}"

            mem_trace: List[Dict[str, Any]] = []
            for seq_l in (128, 512, 1024, 1536):
                sub = long_ids[:, :seq_l]
                van_c = DynamicCache()
                sc_c = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=4, project_slots=True)
                for s in range(0, seq_l, 64):
                    e = min(s + 64, seq_l)
                    _ = slm(sub[:, s:e], past_key_values=van_c, use_cache=True)
                    _ = slm(sub[:, s:e], past_key_values=sc_c, use_cache=True)
                van_kb = recursive_object_tensor_bytes(van_c) / 1024.0
                sc_rec_kb = recursive_object_tensor_bytes(sc_c) / 1024.0
                sc_rep_kb = sc_c.total_memory_bytes / 1024.0
                mem_trace.append(
                    {
                        "tokens": seq_l,
                        "vanilla_actual_kb": round(van_kb, 2),
                        "scache_actual_recursive_kb": round(sc_rec_kb, 2),
                        "scache_reported_kb": round(sc_rep_kb, 2),
                        "kv_reduction_x": round(van_kb / max(sc_rec_kb, 1e-6), 2),
                    }
                )

            drift_bytes_128_to_1536 = round(
                abs(mem_trace[-1]["scache_actual_recursive_kb"] - mem_trace[0]["scache_actual_recursive_kb"]) * 1024.0, 2
            )
            rep_diff_kb = round(
                abs(mem_trace[-1]["scache_actual_recursive_kb"] - mem_trace[-1]["scache_reported_kb"]), 4
            )
            assert drift_bytes_128_to_1536 == 0.0, f"O(1) violation on {model_id}: drift={drift_bytes_128_to_1536} B"
            assert rep_diff_kb < 0.05, f"Anti-facade discrepancy on {model_id}: {rep_diff_kb} KB"

            # Active SCache causality check
            sc_live = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=4, project_slots=True)
            for s in range(0, 256, 64):
                _ = slm(long_ids[:, s : s + 64], past_key_values=sc_live, use_cache=True)
            lg_sc_intact = slm(long_ids[:, 256:257], past_key_values=copy.deepcopy(sc_live), use_cache=True).logits.float()
            sc_zero = copy.deepcopy(sc_live)
            for lyr in sc_zero.layers:
                if isinstance(lyr, SCacheDynamicLayer) and lyr.scar_vals is not None:
                    lyr.scar_vals.zero_()
            lg_sc_zeroed = slm(long_ids[:, 256:257], past_key_values=sc_zero, use_cache=True).logits.float()
            scache_causal_shift = float((lg_sc_intact - lg_sc_zeroed).abs().max().item())
            assert scache_causal_shift > 0.05, f"SCache no-op detected on {model_id}: {scache_causal_shift}"

            # Zero mid-cut across ALL internal blocks (`24/24` or `36/36`) + Capsule Active Causality
            inter_q = (
                "By uniting Landauer's principle at temperature T with the Immerman-Szelepcsenyi theorem for "
                "NSPACE(s(n)), what minimum thermodynamic heat must be dissipated to irreversibly erase m bits "
                "during nondeterministic space complementation?"
            )
            inter_ids = tok(format_chat(inter_q), return_tensors="pt").input_ids.to(dev)
            clean_blocks: List[torch.Tensor] = []
            hks_c = [
                slm.model.layers[i].register_forward_hook(
                    lambda _m, _i, o, _l=clean_blocks: (_l.append((o[0] if isinstance(o, tuple) else o).detach().clone()), None)[1]
                )
                for i in range(n_layers)
            ]
            lg_raw_inter = slm(inter_ids, use_cache=False).logits.detach().float()
            for hk in hks_c:
                hk.remove()

            store.attach(slm)
            active_blocks: List[torch.Tensor] = []
            hks_a = [
                slm.model.layers[i].register_forward_hook(
                    lambda _m, _i, o, _l=active_blocks: (_l.append((o[0] if isinstance(o, tuple) else o).detach().clone()), None)[1]
                )
                for i in range(n_layers)
            ]
            lg_grafted_inter = slm(inter_ids, use_cache=False).logits.detach().float()
            for hk in hks_a:
                hk.remove()
            store.detach()

            all_blocks_exact = all(bool(torch.equal(active_blocks[i], clean_blocks[i])) for i in range(n_layers))
            capsule_causal_shift = float((lg_grafted_inter - lg_raw_inter).abs().max().item())
            assert all_blocks_exact, f"Mid-layer mutation detected on {model_id}!"
            assert capsule_causal_shift > 1.0, f"Capsule no-op detected on {model_id}: {capsule_causal_shift}"

            core_report["attacks"]["attack_1_anti_facade_and_causality"] = {
                "passed": True,
                "memory_scaling_128_to_1536": mem_trace,
                "scache_drift_bytes_128_to_1536": drift_bytes_128_to_1536,
                "reported_vs_recursive_diff_kb": rep_diff_kb,
                "scache_causal_max_logit_shift": round(scache_causal_shift, 4),
                "capsule_causal_max_logit_shift": round(capsule_causal_shift, 4),
                "all_internal_blocks_bit_exact": all_blocks_exact,
                "verified_intact_blocks": f"{n_layers}/{n_layers}",
            }

            # ==================================================================
            # [ATTACK 2] LEXICAL KEYWORD-STUFFED NEAR-MISS TRAP (FALSE-POSITIVE DoS)
            # ==================================================================
            print(f"[Attack 2 | {model_id}] Lexical Keyword-Stuffed Near-Miss Trap (6 Hostile Prompts)...", flush=True)
            trap_details: List[Dict[str, Any]] = []
            false_positives = 0
            exact_logit_matches = 0
            total_trap_mmap_bytes = 0
            max_trap_cos = -1.0

            for trap_idx, trap_q in enumerate(LEXICAL_TRAP_PROMPTS):
                t_ids = tok(format_chat(trap_q), return_tensors="pt").input_ids.to(dev)
                lg_clean_t = slm(t_ids, use_cache=False).logits.detach().clone()
                store.attach(slm)
                lg_store_t = slm(t_ids, use_cache=False).logits.detach().clone()
                store.detach()
                cos_map = {k: round(float(v), 4) for k, v in store.last_routing_cosines.items()}
                peak_c = max(cos_map.values()) if cos_map else 0.0
                max_trap_cos = max(max_trap_cos, peak_c)
                n_fired = store.last_num_expanded_heads
                eq_lg = bool(torch.equal(lg_clean_t, lg_store_t))
                if n_fired > 0:
                    false_positives += 1
                if eq_lg:
                    exact_logit_matches += 1
                total_trap_mmap_bytes += store.last_mmap_bytes_read
                trap_details.append(
                    {
                        "trap_idx": trap_idx,
                        "prompt_excerpt": trap_q[:88] + "...",
                        "peak_cosine": round(peak_c, 4),
                        "cosines": cos_map,
                        "expanded_heads_triggered": n_fired,
                        "exact_torch_equal_logits": eq_lg,
                        "mmap_bytes_read": store.last_mmap_bytes_read,
                    }
                )

            assert false_positives == 0 and exact_logit_matches == len(LEXICAL_TRAP_PROMPTS), (
                f"Lexical trap triggered false positive on {model_id}: {trap_details}"
            )
            core_report["attacks"]["attack_2_lexical_keyword_trap"] = {
                "passed": True,
                "num_hostile_traps": len(LEXICAL_TRAP_PROMPTS),
                "false_positive_activations": false_positives,
                "exact_torch_equal_count": exact_logit_matches,
                "total_mmap_bytes_read": total_trap_mmap_bytes,
                "max_adversarial_trap_cosine": round(max_trap_cos, 4),
                "margin_below_tau_0p62": round(0.62 - max_trap_cos, 4),
                "traps": trap_details,
            }

            # ==================================================================
            # [ATTACK 3] POISONED 1,500+ TOKEN HAYSTACK + EVICTED PASSKEY + GRAFT RECALL
            # ==================================================================
            print(f"[Attack 3 | {model_id}] Poisoned 1,500+ Token Haystack + SCache O(1) + 72B Recall...", flush=True)
            # Part 3A: Evicted Passkey NLL (`SCache O(1)` vs `StreamingLLM` amnesia)
            passkey_doc = (
                "CLASSIFIED ARCHIVE CHRONOS:\n"
                "The primary launch authorization passkey for Station Chronos is 9481-DELTA.\n"
                "Verification repeat: Station Chronos passkey = 9481-DELTA.\n\n"
                + (
                    "Nominal orbital telemetry log across auxiliary thermal radiators and star-tracker gyros "
                    "shows steady attitude control across consecutive eclipse orbits without drift.\n"
                ) * 48
                + "\nFINAL AUDIT QUERY:\nFor Station Chronos, the primary launch authorization passkey is"
            )
            pk_prompt_ids = tok(passkey_doc, return_tensors="pt").input_ids.to(dev)
            pk_ans_ids = tok(" 9481-DELTA", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            pk_full = torch.cat([pk_prompt_ids, pk_ans_ids], dim=1)

            def _eval_pk_nll(mode: str) -> Tuple[float, float]:
                c = (
                    SCacheDynamicCache(window_size=64, num_slots=0, num_sink_tokens=4, project_slots=False)
                    if mode == "streaming_llm"
                    else SCacheDynamicCache(window_size=64, num_slots=24, num_sink_tokens=4, project_slots=True)
                )
                pl = pk_prompt_ids.shape[1]
                for s in range(0, pl - 1, 32):
                    e = min(s + 32, pl - 1)
                    _ = slm(pk_prompt_ids[:, s:e], past_key_values=c, use_cache=True)
                out = slm(pk_full[:, pl - 1 : -1], past_key_values=c, use_cache=True)
                lp = F.log_softmax(out.logits.float(), dim=-1)
                nll = -lp.gather(-1, pk_ans_ids.unsqueeze(-1)).squeeze(-1).mean().item()
                return float(nll), recursive_object_tensor_bytes(c) / 1024.0

            nll_stream, kb_stream = _eval_pk_nll("streaming_llm")
            nll_scache, kb_scache = _eval_pk_nll("scache_o1")
            pk_gain_x = math.exp(min(nll_stream - nll_scache, 20.0))
            assert nll_scache < nll_stream, f"SCache failed evicted passkey on {model_id}: {nll_scache} vs {nll_stream}"

            # Part 3B: Multi-turn ChatML with 1,500+ tokens of poisoned rumor history + real user Landauer query
            poison_turn = (
                "<|im_start|>system\nYou are a rigorous mathematical and physical sciences assistant.<|im_end|>\n"
                "<|im_start|>user\nSummarize this noisy unverified forum log:\n"
                + (
                    "Unverified rumor post: an anonymous blogger falsely claimed that erasing bits of data "
                    "absorbs 999 Joules of cold energy regardless of temperature, while unrelated gardening notes "
                    "discuss watering basil plants every morning at sunrise.\n"
                ) * 34
                + "<|im_end|>\n<|im_start|>assistant\nThe forum log contains unverified rumors and gardening notes.<|im_end|>\n"
                + format_chat(
                    "State Landauer's principle and give the exact minimum thermodynamic heat dissipated when "
                    "irreversibly erasing one bit of information at temperature T."
                )
            )
            poison_ids = tok(poison_turn, return_tensors="pt").input_ids.to(dev)
            assert poison_ids.shape[1] >= 1450, f"Poisoned prompt too short: {poison_ids.shape[1]}"

            store.attach(slm)
            poison_gen = _scache_generate(slm, tok, poison_ids, max_new_tokens=40, window_size=128, num_slots=32)
            poison_gates = dict(store.last_active_gates)
            store.detach()

            resisted_poison = (
                "titan72b_landauer_principle" in poison_gates
                and ("999" not in poison_gen["text"])
                and ("Landauer" in poison_gen["text"] or "minimum" in poison_gen["text"] or "ln" in poison_gen["text"])
            )
            assert resisted_poison, f"Poisoned haystack corrupted grafted response on {model_id}: {poison_gen}"

            core_report["attacks"]["attack_3_poisoned_haystack_and_evicted_passkey"] = {
                "passed": True,
                "evicted_passkey_tokens": int(pk_prompt_ids.shape[1]),
                "streaming_llm_nll": round(nll_stream, 4),
                "scache_o1_nll": round(nll_scache, 4),
                "scache_vs_streaming_likelihood_gain_x": round(pk_gain_x, 2),
                "scache_o1_kv_kb": round(kb_scache, 2),
                "poisoned_chat_tokens": int(poison_ids.shape[1]),
                "poisoned_chat_active_gates": poison_gates,
                "resisted_false_999j_rumor": ("999" not in poison_gen["text"]),
                "post_haystack_scache_text": poison_gen["text"],
                "post_haystack_tok_per_s": poison_gen["tok_per_s"],
            }

            # ==================================================================
            # [ATTACK 4] MULTI-CARTRIDGE COLLISION (2-WAY SYNTHESIS & 3-WAY COMPOSITE)
            # ==================================================================
            print(f"[Attack 4 | {model_id}] Multi-Cartridge Simultaneous Collision (2-Way & 3-Way)...", flush=True)
            store.attach(slm)
            gen_2way = _scache_generate(slm, tok, inter_ids, max_new_tokens=56, window_size=256, num_slots=64)
            tel_2way = store.get_last_telemetry()

            q_3way = (
                "Answer concisely all three parts: (1) What upper bound does the Kabatiansky-Levenshtein theorem "
                "establish for sphere packing? (2) What minimum heat does Landauer's principle require to erase "
                "one bit at temperature T? (3) What closure property does the Immerman-Szelepcsenyi theorem prove "
                "for NSPACE(s(n))?"
            )
            ids_3way = tok(format_chat(q_3way), return_tensors="pt").input_ids.to(dev)
            gen_3way = _scache_generate(slm, tok, ids_3way, max_new_tokens=96, window_size=256, num_slots=64)
            tel_3way = store.get_last_telemetry()
            gates_3way = dict(store.last_active_gates)
            mmap_3way = store.last_mmap_bytes_read
            store.detach()

            assert len(tel_2way.active_capsules) == 2, f"Expected 2 active heads on {model_id}, got {tel_2way.active_capsules}"
            assert len(gates_3way) == 3, f"Expected all 3 active heads on 3-way query on {model_id}, got {gates_3way}"

            core_report["attacks"]["attack_4_multicartridge_collision"] = {
                "passed": True,
                "two_way_landauer_x_immerman": {
                    "active_capsules": tel_2way.active_capsules,
                    "num_expanded_heads": tel_2way.num_expanded_heads,
                    "max_inter_head_spread": tel_2way.max_inter_head_spread,
                    "per_capsule_min_weight": tel_2way.per_capsule_min_weight,
                    "per_capsule_max_weight": tel_2way.per_capsule_max_weight,
                    "generated_text": gen_2way["text"],
                    "tok_per_s": gen_2way["tok_per_s"],
                },
                "three_way_simultaneous_all_domains": {
                    "active_gates": {k: round(float(v), 4) for k, v in gates_3way.items()},
                    "num_expanded_heads": len(gates_3way),
                    "mmap_bytes_paged": mmap_3way,
                    "max_inter_head_spread": tel_3way.max_inter_head_spread,
                    "per_capsule_min_weight": tel_3way.per_capsule_min_weight,
                    "per_capsule_max_weight": tel_3way.per_capsule_max_weight,
                    "generated_text": gen_3way["text"],
                    "tok_per_s": gen_3way["tok_per_s"],
                },
            }

            # ==================================================================
            # [ATTACK 5] FULL-STACK SIMULTANEOUS `MoS` + `SCache O(1)` + `HeadExpandedScarStore`
            # ==================================================================
            print(f"[Attack 5 | {model_id}] Full-Stack Simultaneous MoS + SCache O(1) + ScarStore...", flush=True)
            calib_ids = tok(ADVERSARIAL_DOMAIN_TEXTS["python_code"], return_tensors="pt").input_ids
            l_start = max(0, n_layers - mos_tail_layers)
            # Compress the top `mos_tail_layers` SwiGLU MLPs in-place on a lightweight shallow module clone on CPU/MPS
            # without duplicating all untouched layers in RAM!
            orig_tail = copy.deepcopy(nn.ModuleList([slm.model.layers[i].mlp for i in range(l_start, n_layers)])).cpu()
            try:
                # Collect RMS activations on `[l_start : n_layers)`
                rms_in: Dict[int, torch.Tensor] = {}
                rms_inter: Dict[int, torch.Tensor] = {}
                hks_m = []
                for idx in range(l_start, n_layers):
                    def _mk_in(i: int):
                        return lambda _m, inp, _o: (
                            rms_in.__setitem__(
                                i,
                                torch.sqrt(inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1]).pow(2).mean(0) + 1e-6),
                            ),
                            None,
                        )[1]

                    def _mk_dn(i: int):
                        return lambda _m, inp, _o: (
                            rms_inter.__setitem__(
                                i,
                                torch.sqrt(inp[0].detach().float().cpu().reshape(-1, inp[0].shape[-1]).pow(2).mean(0) + 1e-6),
                            ),
                            None,
                        )[1]

                    hks_m.append(slm.model.layers[idx].mlp.gate_proj.register_forward_hook(_mk_in(idx)))
                    hks_m.append(slm.model.layers[idx].mlp.down_proj.register_forward_hook(_mk_dn(idx)))
                _ = slm(calib_ids.to(dev), use_cache=False)
                for h in hks_m:
                    h.remove()

                from scripts.adversarial_mos_scache_suite import (
                    GenuineMoSQuantScarLinear,
                    dequantize_int4_blockwise,
                    quantize_int4_blockwise,
                )

                dense_mlp_bytes = 0
                mos_mlp_bytes = 0
                for idx in range(l_start, n_layers):
                    for pname in ("gate_proj", "up_proj", "down_proj"):
                        lin: nn.Linear = getattr(slm.model.layers[idx].mlp, pname)
                        w_f = lin.weight.detach().float().cpu()
                        dense_mlp_bytes += w_f.numel() * 2
                        pk, sc_w, shp = quantize_int4_blockwise(w_f, block_size=64)
                        w_deq = dequantize_int4_blockwise(pk, sc_w, shp, dtype=torch.float32)
                        s_v = (rms_inter[idx] if pname == "down_proj" else rms_in[idx]).clamp(min=1e-4)
                        res = (w_f - w_deq) * s_v.unsqueeze(0)
                        u_m, s_m, vh_m = torch.linalg.svd(res, full_matrices=False)
                        r_e = min(48, int(s_m.shape[0]))
                        sq = torch.sqrt(s_m[:r_e])
                        b_sc = u_m[:, :r_e] * sq.unsqueeze(0)
                        a_sc = (sq.unsqueeze(1) * vh_m[:r_e, :]) / s_v.unsqueeze(0)
                        mos_mod = GenuineMoSQuantScarLinear(
                            packed_base=pk,
                            scales_base=sc_w,
                            orig_shape=shp,
                            A_scar=a_sc,
                            B_scar=b_sc,
                            bias=lin.bias.detach().cpu() if lin.bias is not None else None,
                        ).to(dev)
                        mos_mlp_bytes += mos_mod.physical_bytes
                        setattr(slm.model.layers[idx].mlp, pname, mos_mod)

                # Verify active MoS ASVD Scar causality
                probe_m_ids = tok(ADVERSARIAL_DOMAIN_TEXTS["physics_math"], return_tensors="pt").input_ids.to(dev)
                lg_mos_on = slm(probe_m_ids, use_cache=False).logits.float()
                for idx in range(l_start, n_layers):
                    slm.model.layers[idx].mlp.gate_proj.scar_enabled = False
                    slm.model.layers[idx].mlp.up_proj.scar_enabled = False
                    slm.model.layers[idx].mlp.down_proj.scar_enabled = False
                lg_mos_off = slm(probe_m_ids, use_cache=False).logits.float()
                for idx in range(l_start, n_layers):
                    slm.model.layers[idx].mlp.gate_proj.scar_enabled = True
                    slm.model.layers[idx].mlp.up_proj.scar_enabled = True
                    slm.model.layers[idx].mlp.down_proj.scar_enabled = True
                mos_scar_shift = float((lg_mos_on - lg_mos_off).abs().max().item())
                assert mos_scar_shift > 0.1, f"MoS ASVD scars inactive on {model_id}: {mos_scar_shift}"

                # Evaluate Full-Stack (`MoS` + `SCache O(1)` + `HeadExpandedScarStore`) on unseen Immerman 72B!
                imm = payload["domains"]["titan72b_immerman_szelepcsenyi"]
                u_p_imm = imm["unseen_prompt_ids"].to(dev)
                u_f_imm = imm["unseen_full_ids"].unsqueeze(0).to(dev)
                lp_imm = int(u_p_imm.numel())
                la_imm = int(imm["unseen_teacher_logits"].shape[0])

                store.attach(slm)
                with store.teacher_forcing_scope(prompt_len=lp_imm):
                    lg_fs = slm(u_f_imm, use_cache=False).logits[0, lp_imm - 1 : lp_imm - 1 + la_imm].detach().cpu().float()
                lg_t_imm = imm["unseen_teacher_logits"][:la_imm].float()
                vc = min(int(lg_fs.shape[-1]), int(lg_t_imm.shape[-1]))
                fs_top1 = float((lg_fs[:, :vc].argmax(-1) == lg_t_imm[:, :vc].argmax(-1)).float().mean().item() * 100.0)
                fs_kl = _kl_div_top_union(lg_t_imm[:, :vc], lg_fs[:, :vc], top_k=256)
                fs_gen = _scache_generate(slm, tok, u_p_imm.unsqueeze(0), max_new_tokens=la_imm, window_size=128, num_slots=32)
                store.detach()

                assert fs_top1 >= 90.0, f"Full-stack MoS+SCache+ScarStore degraded Top-1 on {model_id}: {fs_top1}%"

                core_report["attacks"]["attack_5_fullstack_mos_scache_scarstore"] = {
                    "passed": True,
                    "mos_compressed_layers": f"L{l_start}..L{n_layers - 1} ({mos_tail_layers * 3} SwiGLU matrices)",
                    "dense_tail_mlp_mb": round(dense_mlp_bytes / (1024**2), 2),
                    "mos_tail_mlp_mb": round(mos_mlp_bytes / (1024**2), 2),
                    "tail_mlp_compression_x": round(dense_mlp_bytes / max(mos_mlp_bytes, 1), 2),
                    "mos_active_scar_logit_shift": round(mos_scar_shift, 4),
                    "immerman_72b_fullstack_top1_pct": round(fs_top1, 2),
                    "immerman_72b_fullstack_kl_div": round(fs_kl, 4),
                    "immerman_fullstack_scache_text": fs_gen["text"],
                    "scache_actual_kb": fs_gen["scache_actual_kb"],
                }
            finally:
                for offset, idx in enumerate(range(l_start, n_layers)):
                    slm.model.layers[idx].mlp = orig_tail[offset].to(dev)
                del orig_tail
                gc.collect()

            # ==================================================================
            # [ATTACK 6] EPISTEMIC MISSING-DOMAIN TRAP (`MissingCapsuleAlert`) + `< 2 ms` GDPR `unlearn()`
            # ==================================================================
            print(f"[Attack 6 | {model_id}] MissingCapsuleAlert Trap + HTTP-Range Fetch + <2ms GDPR unlearn()...", flush=True)
            sub_path = tmp_root / "consumer_subset.scarstore"
            sub_store = store.export_subset(
                sub_path,
                include_capsule_ids=["titan72b_kabatiansky_levenshtein", "titan72b_landauer_principle"],
            )
            try:
                # 6A: Strict missing guard must block hallucination on uninstalled Immerman coordinate!
                sub_store.strict_missing_guard = True
                imm_prompt_2d = payload["domains"]["titan72b_immerman_szelepcsenyi"]["unseen_prompt_ids"].unsqueeze(0).to(dev)
                blocked_missing = False
                alert_coord = None
                sub_store.attach(slm)
                try:
                    _ = slm(imm_prompt_2d, use_cache=False)
                except RuntimeError as exc:
                    if "MissingCapsuleAlert" in str(exc):
                        blocked_missing = True
                        alert_coord = sub_store.last_missing_alerts[0].coord if sub_store.last_missing_alerts else None
                finally:
                    sub_store.detach()
                assert blocked_missing, f"MissingCapsuleAlert failed to block uninstalled domain on {model_id}!"

                # 6B: Attach live `on_demand_fetcher` streaming exact `62,720 B` / `143,360 B` record from cloud/full store
                fetched_log: List[int] = []

                def _cloud_range_fetcher(coord: Tuple[int, int], _cid: str) -> bytes:
                    blob = store.read_raw_capsule_bytes_by_coord(coord)
                    fetched_log.append(len(blob))
                    return blob

                sub_store.on_demand_fetcher = _cloud_range_fetcher
                sub_store.attach(slm)
                streamed_gen = _scache_generate(slm, tok, imm_prompt_2d, max_new_tokens=24, window_size=128, num_slots=32)
                sub_store.detach()
                assert fetched_log == [expected_capsule_bytes], f"Unexpected streamed bytes: {fetched_log}"
            finally:
                sub_store.close()

            # 6C: Sub-millisecond GDPR `unlearn("titan72b_landauer_principle")` with zero collateral bleed!
            kl_prompt_2d = payload["domains"]["titan72b_kabatiansky_levenshtein"]["unseen_prompt_ids"].unsqueeze(0).to(dev)
            lnd_prompt_2d = payload["domains"]["titan72b_landauer_principle"]["unseen_prompt_ids"].unsqueeze(0).to(dev)
            lg_lnd_raw_base = slm(lnd_prompt_2d, use_cache=False).logits.detach().clone()

            store.attach(slm)
            lg_kl_before_unlearn = slm(kl_prompt_2d, use_cache=False).logits.detach().clone()
            unlearn_info = store.unlearn("titan72b_landauer_principle")
            lg_lnd_after_unlearn = slm(lnd_prompt_2d, use_cache=False).logits.detach().clone()
            lnd_heads_after = store.last_num_expanded_heads
            lg_kl_after_unlearn = slm(kl_prompt_2d, use_cache=False).logits.detach().clone()
            store.detach()

            exact_landauer_erased = bool(torch.equal(lg_lnd_after_unlearn, lg_lnd_raw_base)) and (lnd_heads_after == 0)
            exact_kl_unharmed = bool(torch.equal(lg_kl_after_unlearn, lg_kl_before_unlearn))
            assert unlearn_info["unlearn_ms"] < 2.0, f"unlearn() exceeded 2 ms: {unlearn_info}"
            assert exact_landauer_erased and exact_kl_unharmed, (
                f"GDPR unlearn failed invariance on {model_id}: erased={exact_landauer_erased}, unharmed={exact_kl_unharmed}"
            )

            core_report["attacks"]["attack_6_missing_capsule_guard_and_gdpr_unlearn"] = {
                "passed": True,
                "kronecker_grid_cells": store.num_grid_cells,
                "presence_bitmap_kb": round(store.presence_bitmap_bytes / 1024.0, 2),
                "missing_capsule_blocked_before_hallucination": blocked_missing,
                "missing_alert_coordinate": list(alert_coord) if alert_coord is not None else None,
                "http_range_streamed_bytes": fetched_log[0],
                "post_stream_recovered_text": streamed_gen["text"],
                "gdpr_unlearn_ms": unlearn_info["unlearn_ms"],
                "exact_torch_equal_to_base_after_unlearn": exact_landauer_erased,
                "exact_zero_collateral_damage_on_other_capsules": exact_kl_unharmed,
            }
        finally:
            store.close()

    core_report["all_attacks_passed"] = all(v["passed"] for v in core_report["attacks"].values())
    core_report["elapsed_s"] = round(time.perf_counter() - t_core0, 2)

    del slm, head_bank
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return core_report


def run_adversarial_0p5b_vs_3b_suite(output_json_path: Path | None = None) -> Dict[str, Any]:
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    try:
        rep_0p5b = audit_single_core_adversarial(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            expected_capsule_bytes=62720,
            expected_grid_cells=802816,
            expected_bitmap_bytes=100352,
            mos_tail_layers=6,
        )
        rep_3b = audit_single_core_adversarial(
            model_id="Qwen/Qwen2.5-3B-Instruct",
            expected_capsule_bytes=143360,
            expected_grid_cells=4194304,
            expected_bitmap_bytes=524288,
            mos_tail_layers=6,
        )

        a1_0 = rep_0p5b["attacks"]["attack_1_anti_facade_and_causality"]
        a1_3 = rep_3b["attacks"]["attack_1_anti_facade_and_causality"]
        a2_0 = rep_0p5b["attacks"]["attack_2_lexical_keyword_trap"]
        a2_3 = rep_3b["attacks"]["attack_2_lexical_keyword_trap"]
        a3_0 = rep_0p5b["attacks"]["attack_3_poisoned_haystack_and_evicted_passkey"]
        a3_3 = rep_3b["attacks"]["attack_3_poisoned_haystack_and_evicted_passkey"]
        a4_0 = rep_0p5b["attacks"]["attack_4_multicartridge_collision"]
        a4_3 = rep_3b["attacks"]["attack_4_multicartridge_collision"]
        a5_0 = rep_0p5b["attacks"]["attack_5_fullstack_mos_scache_scarstore"]
        a5_3 = rep_3b["attacks"]["attack_5_fullstack_mos_scache_scarstore"]
        a6_0 = rep_0p5b["attacks"]["attack_6_missing_capsule_guard_and_gdpr_unlearn"]
        a6_3 = rep_3b["attacks"]["attack_6_missing_capsule_guard_and_gdpr_unlearn"]

        head_to_head = {
            "all_passed_both_cores": bool(rep_0p5b["all_attacks_passed"] and rep_3b["all_attacks_passed"]),
            "capsule_footprint": {
                "0p5b_bytes": 62720,
                "3b_bytes": 143360,
                "3b_exact_35_os_pages_4kb": True,
            },
            "kronecker_addressable_domains": {
                "0p5b_cells": 802816,
                "3b_cells": 4194304,
                "capacity_expansion_x": round(4194304 / 802816, 2),
            },
            "scache_o1_drift_bytes_128_to_1536": {
                "0p5b": a1_0["scache_drift_bytes_128_to_1536"],
                "3b": a1_3["scache_drift_bytes_128_to_1536"],
            },
            "lexical_trap_max_cosine_vs_tau_0p62": {
                "0p5b_max_cos": a2_0["max_adversarial_trap_cosine"],
                "3b_max_cos": a2_3["max_adversarial_trap_cosine"],
                "0p5b_safety_margin": a2_0["margin_below_tau_0p62"],
                "3b_safety_margin": a2_3["margin_below_tau_0p62"],
                "both_zero_false_positives": (
                    a2_0["false_positive_activations"] == 0 and a2_3["false_positive_activations"] == 0
                ),
            },
            "evicted_passkey_likelihood_gain_vs_streaming_llm": {
                "0p5b_gain_x": a3_0["scache_vs_streaming_likelihood_gain_x"],
                "3b_gain_x": a3_3["scache_vs_streaming_likelihood_gain_x"],
                "0p5b_scache_nll": a3_0["scache_o1_nll"],
                "3b_scache_nll": a3_3["scache_o1_nll"],
            },
            "poisoned_1500tok_haystack_synthesis": {
                "0p5b_text": a3_0["post_haystack_scache_text"],
                "3b_text": a3_3["post_haystack_scache_text"],
            },
            "two_way_interdisciplinary_synthesis": {
                "0p5b_spread": a4_0["two_way_landauer_x_immerman"]["max_inter_head_spread"],
                "3b_spread": a4_3["two_way_landauer_x_immerman"]["max_inter_head_spread"],
                "0p5b_text": a4_0["two_way_landauer_x_immerman"]["generated_text"],
                "3b_text": a4_3["two_way_landauer_x_immerman"]["generated_text"],
            },
            "three_way_all_domains_simultaneous": {
                "0p5b_text": a4_0["three_way_simultaneous_all_domains"]["generated_text"],
                "3b_text": a4_3["three_way_simultaneous_all_domains"]["generated_text"],
            },
            "fullstack_mos_scache_scarstore_immerman_top1_pct": {
                "0p5b": a5_0["immerman_72b_fullstack_top1_pct"],
                "3b": a5_3["immerman_72b_fullstack_top1_pct"],
            },
            "gdpr_unlearn_latency_ms": {
                "0p5b_ms": a6_0["gdpr_unlearn_ms"],
                "3b_ms": a6_3["gdpr_unlearn_ms"],
            },
        }

        summary = {
            "suite": "Adversarial Red-Team Core Comparison (v2.2.0): Qwen2.5-0.5B vs Qwen2.5-3B + MoS + SCache O(1) + HeadExpandedScarStore",
            "zero_autograd_verified": (not torch.is_grad_enabled()),
            "all_passed": head_to_head["all_passed_both_cores"],
            "total_elapsed_s": round(time.perf_counter() - t0, 2),
            "head_to_head_summary": head_to_head,
            "core_0p5b": rep_0p5b,
            "core_3b": rep_3b,
        }
        out_p = output_json_path or DEFAULT_OUTPUT_JSON
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\n" + "=" * 96, flush=True)
        print(
            f"🏆 RED-TEAM VERDICT: all_passed={summary['all_passed']} across BOTH 0.5B and 3B in {summary['total_elapsed_s']}s!",
            flush=True,
        )
        print(f"📄 Audit JSON written to: {out_p}", flush=True)
        print("=" * 96, flush=True)
        return summary
    finally:
        torch.set_grad_enabled(prev_grad)


if __name__ == "__main__":
    run_adversarial_0p5b_vs_3b_suite()
