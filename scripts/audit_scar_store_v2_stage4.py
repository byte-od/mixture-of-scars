"""Controlled Head-to-Head Stage 4 (`v2.1.0`) vs Stage 3 (`v2.0.0`) Live Audit on `Qwen2.5-0.5B-Instruct` + `72.71B` Episodes.

Directly compares:
  - **Stage 3 (`Dual-Depth`)**: Cutting the Transformer at `Layer 12/24` + static scalar cone gates (`0.50 / 0.50` pre-sleep).
  - **Stage 4 (`HeadExpandedAttentionBank`)**: **Zero Mid-Transformer Cut** (`Layers 1..24` blocks 100% intact) +
    **On-Demand Dynamic Scar-Attention Head Expansion** (`+M` heads using `Layer 24`'s `14` pretrained `q_proj` heads
    against `P = 8` Per-Head Key Prototypes inside exact `62,720 B = 61.25 KB INT4` capsules!).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
)
from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.scar_grafting import SleepEpisode, find_subsequence_last_end
from scripts.benchmark_scar_grafting_v2 import (
    BACKGROUND_CONTROL_PROMPTS,
    EPISODE_CACHE_PATH,
    RESULTS_DIR,
    _kl_div_top_union,
    format_chat,
)

STAGE3_AUDIT_PATH = RESULTS_DIR / "scar_store_v2_stage3_audit.json"
STAGE4_AUDIT_PATH = RESULTS_DIR / "scar_store_v2_stage4_audit.json"


@torch.no_grad()
def _scache_greedy_generate(
    model: Any,
    tok: Any,
    prompt_ids_2d: torch.Tensor,
    max_new_tokens: int = 36,
) -> Dict[str, Any]:
    dev = next(model.parameters()).device
    scache = SCacheDynamicCache(window_size=256, num_slots=128)
    eos_id = tok.eos_token_id
    t0 = time.perf_counter()
    out = model(prompt_ids_2d.to(dev), past_key_values=scache, use_cache=True)
    nxt = int(out.logits[0, -1].argmax(dim=-1).item())
    gen: List[int] = [nxt]
    while len(gen) < max_new_tokens and nxt != eos_id:
        step_in = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        out = model(step_in, past_key_values=scache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax(dim=-1).item())
        if nxt == eos_id:
            break
        gen.append(nxt)
    dt = max(time.perf_counter() - t0, 1e-6)
    return {
        "text": tok.decode(gen, skip_special_tokens=True).strip(),
        "tokens": len(gen),
        "tok_per_s": round(len(gen) / dt, 2),
    }


def main() -> None:
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        print("[Stage 4 Audit] Loading frozen Student Qwen/Qwen2.5-0.5B-Instruct (24 layers, 14 heads x 64, d=896)...", flush=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        slm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            dtype=torch.float32,
        ).eval()
        payload = torch.load(EPISODE_CACHE_PATH, map_location="cpu", weights_only=False)
        stage3_ref = json.loads(STAGE3_AUDIT_PATH.read_text()) if STAGE3_AUDIT_PATH.exists() else {}

        t_cal0 = time.perf_counter()
        head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
            student_model=slm,
            calibration_prompts=BACKGROUND_CONTROL_PROMPTS,
            tokenizer=tok,
            chat_formatter=format_chat,
            preserved_rank=4,
            attention_beta=14.0,
        )
        cal_ms = round((time.perf_counter() - t_cal0) * 1000.0, 2)

        # Measure raw base SLM unseen baselines BEFORE attaching HeadExpandedAttentionBank
        raw_baselines: Dict[str, Dict[str, Any]] = {}
        for cid, dinfo in payload["domains"].items():
            u_full = dinfo["unseen_full_ids"].unsqueeze(0)
            lp_u = int(dinfo["unseen_prompt_ids"].numel())
            la_u = int(dinfo["unseen_teacher_logits"].shape[0])
            lg_raw = slm(u_full, use_cache=False).logits[0, lp_u - 1 : lp_u - 1 + la_u].float()
            lg_t = dinfo["unseen_teacher_logits"][:la_u].float()
            v_c = min(int(lg_raw.shape[-1]), int(lg_t.shape[-1]))
            top1_raw = float((lg_raw[:, :v_c].argmax(-1) == lg_t[:, :v_c].argmax(-1)).float().mean().item() * 100.0)
            kl_raw = _kl_div_top_union(lg_t[:, :v_c], lg_raw[:, :v_c], top_k=256)
            raw_baselines[cid] = {"top1_pct": round(top1_raw, 2), "kl_div": round(kl_raw, 4)}

        # Capture clean control states BEFORE attaching HeadExpandedAttentionBank
        clean_controls: List[Dict[str, torch.Tensor]] = []
        for cq in BACKGROUND_CONTROL_PROMPTS:
            c_ids = tok(format_chat(cq), return_tensors="pt").input_ids
            cap_l12: Dict[str, torch.Tensor] = {}
            cap_l20: Dict[str, torch.Tensor] = {}
            hk12 = slm.model.layers[11].register_forward_hook(
                lambda _m, _i, out: (cap_l12.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()), None)[1]
            )
            hk20 = slm.model.layers[19].register_forward_hook(
                lambda _m, _i, out: (cap_l20.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()), None)[1]
            )
            c_lg = slm(c_ids, use_cache=False).logits.detach().clone()
            hk12.remove()
            hk20.remove()
            clean_controls.append({"ids": c_ids, "logits": c_lg, "layer12": cap_l12["h"], "layer20": cap_l20["h"]})

        # Solve 3 Head-Expanded Capsules (`rank=60, num_key_prototypes=8` -> exact `62,720 B` = `61.25 KB` INT4!)
        capsule_metrics: Dict[str, Any] = {}
        for cid, dinfo in payload["domains"].items():
            cap_fp16 = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
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
            raw_int4 = cap_fp16.to_int4_bytes()
            assert len(raw_int4) == 62720
            cap_int4 = HeadExpandedScarCapsule.from_int4_bytes(
                raw_int4,
                capsule_id=cid,
                hidden_dim=896,
                num_heads=14,
                head_dim=64,
                rank=60,
                num_key_prototypes=8,
                tau=0.62,
                ramp=0.16,
                metadata=cap_fp16.metadata,
            )
            head_bank.add_capsule(cap_int4)
            capsule_metrics[cid] = {
                "int4_bytes": len(raw_int4),
                "int4_kb": round(len(raw_int4) / 1024.0, 2),
                "rank": cap_int4.rank,
                "num_key_prototypes": cap_int4.num_key_prototypes,
                "total_per_head_keys": cap_int4.num_key_prototypes * cap_int4.num_heads,
                "solve_ms": cap_fp16.metadata["solve_ms"],
                "zero_backpropagation": True,
                "zero_mid_layer_cut": True,
            }

        inter_unseen_q = (
            "By uniting Landauer's principle at temperature T with the Immerman-Szelepcsenyi theorem for "
            "NSPACE(s(n)), what minimum thermodynamic heat must be dissipated to irreversibly erase m bits "
            "during nondeterministic space complementation?"
        )
        unseen_inter_ids = tok(format_chat(inter_unseen_q), return_tensors="pt").input_ids

        # Capture clean internal outputs across ALL 24 decoder blocks on the active interdisciplinary query!
        clean_all_24_blocks: List[torch.Tensor] = []
        blk_hooks_clean = [
            slm.model.layers[idx].register_forward_hook(
                lambda _m, _i, out, _lst=clean_all_24_blocks: (
                    _lst.append((out[0] if isinstance(out, tuple) else out).detach().clone()),
                    None,
                )[1]
            )
            for idx in range(24)
        ]
        _ = slm(unseen_inter_ids, use_cache=False)
        for hk in blk_hooks_clean:
            hk.remove()

        head_bank.attach(slm)
        try:
            # 1. Out-of-cone invariance (`0` extra heads instantiated, `torch.equal == True` across all layers & logits)
            exact_ctrl_logits = True
            exact_ctrl_layer12 = True
            exact_ctrl_layer20 = True
            zero_extra_heads_on_controls = True
            max_ctrl_cos = 0.0
            for item in clean_controls:
                cap_l12_after: Dict[str, torch.Tensor] = {}
                cap_l20_after: Dict[str, torch.Tensor] = {}
                hk12 = slm.model.layers[11].register_forward_hook(
                    lambda _m, _i, out: (cap_l12_after.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()), None)[1]
                )
                hk20 = slm.model.layers[19].register_forward_hook(
                    lambda _m, _i, out: (cap_l20_after.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()), None)[1]
                )
                after_lg = slm(item["ids"], use_cache=False).logits.detach()
                hk12.remove()
                hk20.remove()
                exact_ctrl_logits = exact_ctrl_logits and bool(torch.equal(after_lg, item["logits"]))
                exact_ctrl_layer12 = exact_ctrl_layer12 and bool(torch.equal(cap_l12_after["h"], item["layer12"]))
                exact_ctrl_layer20 = exact_ctrl_layer20 and bool(torch.equal(cap_l20_after["h"], item["layer20"]))
                zero_extra_heads_on_controls = zero_extra_heads_on_controls and (head_bank.last_num_expanded_heads == 0)
                max_ctrl_cos = max(max_ctrl_cos, max(head_bank.last_routing_cosines.values()))

            # Verify ALL 24 Transformer decoder blocks remain bit-for-bit untouched EVEN when M=2 heads fire!
            active_all_24_blocks: List[torch.Tensor] = []
            blk_hooks_active = [
                slm.model.layers[idx].register_forward_hook(
                    lambda _m, _i, out, _lst=active_all_24_blocks: (
                        _lst.append((out[0] if isinstance(out, tuple) else out).detach().clone()),
                        None,
                    )[1]
                )
                for idx in range(24)
            ]
            _ = slm(unseen_inter_ids, use_cache=False)
            for hk in blk_hooks_active:
                hk.remove()
            exact_zero_mid_cut_all_24_blocks = all(
                bool(torch.equal(active_all_24_blocks[i], clean_all_24_blocks[i])) for i in range(24)
            )

            # 2. Evaluate unseen `72.71B` paraphrases (`61.25 KB INT4` + `SCache O(1)`)
            unseen_eval: Dict[str, Any] = {}
            zero_off_diagonal_bleed = True
            for cid, dinfo in payload["domains"].items():
                u_p = dinfo["unseen_prompt_ids"]
                u_full = dinfo["unseen_full_ids"].unsqueeze(0)
                lp_u = int(u_p.numel())
                la_u = int(dinfo["unseen_teacher_logits"].shape[0])
                with head_bank.teacher_forcing_scope(prompt_len=lp_u):
                    lg_he = slm(u_full, use_cache=False).logits[0, lp_u - 1 : lp_u - 1 + la_u].float()
                active_g = dict(head_bank.last_active_gates)
                if list(active_g.keys()) != [cid]:
                    zero_off_diagonal_bleed = False
                lg_t = dinfo["unseen_teacher_logits"][:la_u].float()
                v_c = min(int(lg_he.shape[-1]), int(lg_t.shape[-1]))
                top1_he = float((lg_he[:, :v_c].argmax(-1) == lg_t[:, :v_c].argmax(-1)).float().mean().item() * 100.0)
                kl_he = _kl_div_top_union(lg_t[:, :v_c], lg_he[:, :v_c], top_k=256)
                gen_info = _scache_greedy_generate(slm, tok, u_p.unsqueeze(0), max_new_tokens=la_u)
                unseen_eval[cid] = {
                    "routing_cosines_layer24_in": {k: round(float(v), 4) for k, v in head_bank.last_routing_cosines.items()},
                    "active_gates": active_g,
                    "num_expanded_heads": head_bank.last_num_expanded_heads,
                    "base_top1_pct": raw_baselines[cid]["top1_pct"],
                    "head_expanded_int4_top1_pct": round(top1_he, 2),
                    "base_kl_div": raw_baselines[cid]["kl_div"],
                    "head_expanded_int4_kl_div": round(kl_he, 4),
                    "live_scache_text": gen_info["text"],
                    "titan_72b_ref_text": dinfo["unseen_ref_text"].strip(),
                    "live_tok_per_s": gen_info["tok_per_s"],
                }

            # 3. Pre-Sleep Interdisciplinary Synthesis (`Landauer x Immerman`, ZERO Sleep Bridge!)
            dec_pre, tel_pre_probe8 = HeadExpandedAttentionEngine.assess_reasoning_epistemic_uncertainty(
                student_model=slm,
                head_bank=head_bank,
                prompt_ids_2d=unseen_inter_ids,
                probe_steps=8,
                margin_threshold=0.65,
            )
            gen_pre = _scache_greedy_generate(slm, tok, unseen_inter_ids, max_new_tokens=56)
            tel_pre_full = head_bank.get_last_telemetry()

            # 4. Optional Post-Sleep Interdisciplinary Bridge Consolidation (`61.25 KB INT4`, zero mid-cut!)
            inter_train_q1 = (
                "Combining Landauer's principle for bit erasure at temperature T with the Immerman-Szelepcsenyi "
                "theorem for nondeterministic space, what minimum heat is dissipated if complementing an "
                "NSPACE(s(n)) configuration graph irreversibly erases m bits?"
            )
            inter_train_q2 = (
                "Using Landauer's thermodynamic bound at temperature T and the Immerman-Szelepcsenyi closure "
                "theorem, state the minimum heat dissipated when an NSPACE(s(n)) complementation procedure "
                "irreversibly erases m bits of information."
            )
            inter_ref_answer = (
                "By the Immerman-Szelepcsenyi theorem, NSPACE(s(n)) = coNSPACE(s(n)) for s(n) >= log n, "
                "and by Landauer's principle irreversibly erasing m bits during that complementation "
                "dissipates at least m k_B T ln 2 of heat."
            )
            ans_ids = tok(inter_ref_answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
            ans_ids = torch.cat([ans_ids, torch.tensor([tok.eos_token_id], dtype=torch.long)], dim=0)
            bridge_eps: List[SleepEpisode] = []
            for q_txt in (inter_train_q1, inter_train_q2):
                p_ids = tok(format_chat(q_txt), return_tensors="pt").input_ids[0]
                full_ids = torch.cat([p_ids, ans_ids], dim=0)
                with head_bank.paused():
                    base_lg = slm(full_ids.unsqueeze(0), use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
                synth_lg = base_lg.clone()
                for step_i, target_tok in enumerate(ans_ids.tolist()):
                    cur_max = float(synth_lg[step_i].max().item())
                    synth_lg[step_i, target_tok] = cur_max + 7.5
                qs_i = find_subsequence_last_end(p_ids.tolist(), head_bank.user_turn_prefix_ids)
                bridge_eps.append(
                    SleepEpisode(
                        capsule_id="bridge_landauer__immerman",
                        prompt_ids=p_ids,
                        full_ids=full_ids,
                        teacher_logits=synth_lg.to(torch.float16),
                        query_start_idx=qs_i,
                        surprise_score=2.0,
                        metadata={"prompt": q_txt},
                    )
                )

            bridge_fp16 = HeadExpandedAttentionEngine.consolidate_interdisciplinary_bridge(
                bridge_capsule_id="bridge_landauer__immerman",
                bridge_episodes=bridge_eps,
                student_model=slm,
                head_bank=head_bank,
                rank=60,
                num_key_prototypes=8,
                tau=0.62,
                ramp=0.16,
            )
            bridge_raw = bridge_fp16.to_int4_bytes()
            assert len(bridge_raw) == 62720
            bridge_int4 = HeadExpandedScarCapsule.from_int4_bytes(
                bridge_raw,
                capsule_id="bridge_landauer__immerman",
                hidden_dim=896,
                num_heads=14,
                head_dim=64,
                rank=60,
                num_key_prototypes=8,
                tau=0.62,
                ramp=0.16,
                parent_capsules=bridge_fp16.parent_capsules,
                metadata=bridge_fp16.metadata,
            )
            head_bank.add_capsule(bridge_int4)
            head_bank.attach(slm)

            dec_post, _ = HeadExpandedAttentionEngine.assess_reasoning_epistemic_uncertainty(
                student_model=slm,
                head_bank=head_bank,
                prompt_ids_2d=unseen_inter_ids,
                probe_steps=8,
                margin_threshold=0.65,
            )
            gen_post = _scache_greedy_generate(slm, tok, unseen_inter_ids, max_new_tokens=48)

            # Verify zero single-domain bleed after mounting `bridge_landauer__immerman`
            landauer_info = payload["domains"]["titan72b_landauer_principle"]
            immerman_info = payload["domains"]["titan72b_immerman_szelepcsenyi"]
            gen_landauer_after = _scache_greedy_generate(
                slm, tok, landauer_info["unseen_prompt_ids"].unsqueeze(0), max_new_tokens=int(landauer_info["unseen_teacher_logits"].shape[0])
            )
            gates_landauer_after = dict(head_bank.last_active_gates)
            gen_immerman_after = _scache_greedy_generate(
                slm, tok, immerman_info["unseen_prompt_ids"].unsqueeze(0), max_new_tokens=int(immerman_info["unseen_teacher_logits"].shape[0])
            )
            gates_immerman_after = dict(head_bank.last_active_gates)

            # Build explicit head-to-head Stage 3 vs Stage 4 comparison
            s3_pre = stage3_ref.get("interdisciplinary_bridge_audit", {}).get("pre_sleep_decision", {})
            s3_unseen = stage3_ref.get("unseen_paraphrase_72b_fidelity", {})
            s3_caps = stage3_ref.get("capsules", {})
            mean_s3_solve_ms = round(
                sum(v.get("solve_ms", 0.0) for v in s3_caps.values()) / max(1, len(s3_caps)), 2
            ) if s3_caps else None
            mean_s4_solve_ms = round(
                sum(v["solve_ms"] for v in capsule_metrics.values()) / len(capsule_metrics), 2
            )

            comparison = {
                "capsule_int4_bytes_stage3_vs_stage4": [62720, 62720],
                "cuts_transformer_mid_layer": {"stage3_dual_depth": True, "stage4_head_expanded": False},
                "all_24_blocks_bit_exact_during_reasoning": {
                    "stage3_dual_depth": False,
                    "stage4_head_expanded": exact_zero_mid_cut_all_24_blocks,
                },
                "single_domain_off_diagonal_bleed": {
                    "stage3_kabatiansky_immerman_bleed": round(
                        s3_unseen.get("titan72b_kabatiansky_levenshtein", {})
                        .get("active_gates", {})
                        .get("titan72b_immerman_szelepcsenyi", 0.0),
                        4,
                    ),
                    "stage4_off_diagonal_bleed": 0.0 if zero_off_diagonal_bleed else -1.0,
                },
                "mean_closed_form_solve_ms_per_capsule": {
                    "stage3_two_pass_ms": mean_s3_solve_ms,
                    "stage4_single_pass_ms": mean_s4_solve_ms,
                    "speedup_x": round(mean_s3_solve_ms / max(mean_s4_solve_ms, 1e-6), 2) if mean_s3_solve_ms else None,
                },
                "unseen_72b_top1_pct": {
                    cid: {
                        "stage3": s3_unseen.get(cid, {}).get("dual_depth_int4_top1_pct"),
                        "stage4": unseen_eval[cid]["head_expanded_int4_top1_pct"],
                    }
                    for cid in unseen_eval
                },
                "unseen_72b_kl_div": {
                    cid: {
                        "stage3": s3_unseen.get(cid, {}).get("dual_depth_int4_kl_div"),
                        "stage4": unseen_eval[cid]["head_expanded_int4_kl_div"],
                    }
                    for cid in unseen_eval
                },
                "pre_sleep_interdisciplinary_synthesis": {
                    "stage3_confidence": s3_pre.get("confidence"),
                    "stage4_confidence": dec_pre.confidence,
                    "stage3_margin": s3_pre.get("margin"),
                    "stage4_margin": dec_pre.margin,
                    "stage3_uncertainty": s3_pre.get("uncertainty"),
                    "stage4_uncertainty": dec_pre.uncertainty,
                    "stage3_is_uncertain": s3_pre.get("is_uncertain"),
                    "stage4_is_uncertain": dec_pre.is_uncertain,
                    "stage4_max_14_head_spread": tel_pre_full.max_inter_head_spread,
                    "stage4_landauer_dynamic_weight_range": [
                        tel_pre_full.per_capsule_min_weight.get("titan72b_landauer_principle", 0.0),
                        tel_pre_full.per_capsule_max_weight.get("titan72b_landauer_principle", 0.0),
                    ],
                    "stage4_immerman_dynamic_weight_range": [
                        tel_pre_full.per_capsule_min_weight.get("titan72b_immerman_szelepcsenyi", 0.0),
                        tel_pre_full.per_capsule_max_weight.get("titan72b_immerman_szelepcsenyi", 0.0),
                    ],
                },
            }

            w_landauer_trace = [
                sw.get("titan72b_landauer_principle", 0.0) for sw in tel_pre_full.step_weights
            ]

            report = {
                "stage": "Stage 4 (v2.1.0) — On-Demand Dynamic Scar-Attention Head Expansion (Zero Mid-Transformer Cut)",
                "student_model": "Qwen/Qwen2.5-0.5B-Instruct (24 layers, 14 Q-heads x 64, d=896)",
                "teacher_source": "Qwen2.5-72B-Instruct-MoS (72.71B real cached episodes)",
                "zero_autograd_verified": (not torch.is_grad_enabled()),
                "zero_mid_layer_cut_all_24_blocks_verified": exact_zero_mid_cut_all_24_blocks,
                "calibration_ms": cal_ms,
                "capsules": capsule_metrics,
                "out_of_cone_invariance": {
                    "num_controls": len(BACKGROUND_CONTROL_PROMPTS),
                    "zero_extra_heads_allocated": zero_extra_heads_on_controls,
                    "exact_torch_equal_layer12": exact_ctrl_layer12,
                    "exact_torch_equal_layer20": exact_ctrl_layer20,
                    "exact_torch_equal_logits": exact_ctrl_logits,
                    "max_control_cosine_layer24_in": round(max_ctrl_cos, 4),
                    "tau": 0.62,
                },
                "zero_off_diagonal_single_domain_bleed": zero_off_diagonal_bleed,
                "unseen_paraphrase_72b_fidelity": unseen_eval,
                "interdisciplinary_reasoning_audit": {
                    "unseen_synthesis_query": inter_unseen_q,
                    "pre_sleep_dynamic_head_expansion": {
                        "active_capsules": dec_pre.active_capsules,
                        "num_expanded_scar_heads": tel_pre_full.num_expanded_heads,
                        "confidence": dec_pre.confidence,
                        "uncertainty": dec_pre.uncertainty,
                        "margin": dec_pre.margin,
                        "is_uncertain": dec_pre.is_uncertain,
                        "probe8_max_inter_head_spread": tel_pre_probe8.max_inter_head_spread,
                        "full_rollout_max_inter_head_spread": tel_pre_full.max_inter_head_spread,
                        "per_capsule_mean_weight": tel_pre_full.per_capsule_mean_weight,
                        "per_capsule_min_weight": tel_pre_full.per_capsule_min_weight,
                        "per_capsule_max_weight": tel_pre_full.per_capsule_max_weight,
                        "landauer_step_weight_trace": w_landauer_trace,
                        "pre_sleep_scache_text": gen_pre["text"],
                        "pre_sleep_tok_per_s": gen_pre["tok_per_s"],
                    },
                    "optional_post_sleep_bridge": {
                        "capsule_id": "bridge_landauer__immerman",
                        "parent_capsules": list(bridge_int4.parent_capsules),
                        "int4_bytes": len(bridge_raw),
                        "int4_kb": round(len(bridge_raw) / 1024.0, 2),
                        "solve_ms": bridge_fp16.metadata["solve_ms"],
                        "post_sleep_confidence": dec_post.confidence,
                        "post_sleep_uncertainty": dec_post.uncertainty,
                        "post_sleep_margin": dec_post.margin,
                        "post_sleep_is_uncertain": dec_post.is_uncertain,
                        "post_sleep_scache_text": gen_post["text"],
                        "post_sleep_tok_per_s": gen_post["tok_per_s"],
                    },
                    "zero_single_domain_bleed_after_bridge": {
                        "landauer_active_gates": gates_landauer_after,
                        "landauer_exact_text_match": (
                            gen_landauer_after["text"] == unseen_eval["titan72b_landauer_principle"]["live_scache_text"]
                        ),
                        "immerman_active_gates": gates_immerman_after,
                        "immerman_exact_text_match": (
                            gen_immerman_after["text"] == unseen_eval["titan72b_immerman_szelepcsenyi"]["live_scache_text"]
                        ),
                    },
                },
                "stage3_vs_stage4_comparison": comparison,
            }

            STAGE4_AUDIT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        finally:
            head_bank.detach()
    finally:
        torch.set_grad_enabled(prev_grad)


if __name__ == "__main__":
    main()
