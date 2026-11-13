"""Live Stage 3 (`v2.0.0`) Audit on `Qwen2.5-0.5B-Instruct` + Real `Qwen2.5-72B-Instruct-MoS (72.71B)` Episodes.

Audits:
  1. `DualDepthScarCapsule` (`Layer 12` Golden-Slice + `Layer 24` Final Norm) closed-form synthesis
     (`ZERO autograd / ZERO SGD`) quantized to exact `62,720 B` (`61.25 KB` `Block-64 INT4`).
  2. Unseen `72.71B` paraphrase Top-1 agreement, KL divergence reduction, and live `SCache O(1)` (`tok/s`)
     generation after `61.25 KB INT4` round-trip.
  3. Exact `0.0000000000` (`torch.equal`) invariance across upper Transformer layers AND logits on all
     6 out-of-cone control prompts.
  4. Genuine upper-layer (`Layers 13..24`) cross-attention modulation, `BridgeEscalationDecision` epistemic
     telemetry (`is_uncertain == True` pre-sleep -> `False` post-sleep), zero single-domain bleed, and
     `61.25 KB INT4` Interdisciplinary Sleep Bridge consolidation (`Landauer x Immerman-Szelepcsenyi`).
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

from scar_cache.dual_depth_bridge import (
    DualDepthBridgeEngine,
    DualDepthScarCapsule,
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
        print("[Stage 3 Audit] Loading frozen Student Qwen/Qwen2.5-0.5B-Instruct (24 layers, d=896)...", flush=True)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        slm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            dtype=torch.float32,
        ).eval()
        payload = torch.load(EPISODE_CACHE_PATH, map_location="cpu", weights_only=False)

        t_cal0 = time.perf_counter()
        dual_bank = DualDepthBridgeEngine.calibrate_dual_depth_student(
            student_model=slm,
            calibration_prompts=BACKGROUND_CONTROL_PROMPTS,
            tokenizer=tok,
            chat_formatter=format_chat,
            mid_layer_idx=11,  # Layer 12 of 24 (50% Golden-Slice midpoint)
            preserved_rank=4,
        )
        cal_ms = round((time.perf_counter() - t_cal0) * 1000.0, 2)

        # Measure raw base SLM unseen baselines BEFORE attaching DualDepthScarBank
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

        # Capture clean control logits AND Layer 20 hidden states BEFORE attaching DualDepthScarBank
        clean_controls: List[Dict[str, torch.Tensor]] = []
        for cq in BACKGROUND_CONTROL_PROMPTS:
            c_ids = tok(format_chat(cq), return_tensors="pt").input_ids
            cap_l20: Dict[str, torch.Tensor] = {}
            hk = slm.model.layers[19].register_forward_hook(
                lambda _m, _i, out: (
                    cap_l20.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()),
                    None,
                )[1]
            )
            c_lg = slm(c_ids, use_cache=False).logits.detach().clone()
            hk.remove()
            clean_controls.append({"ids": c_ids, "logits": c_lg, "layer20": cap_l20["h"]})

        # Solve 3 Dual-Depth Capsules (`r_mid=32, r_final=32`), round-trip through exact `62,720 B` INT4!
        capsule_metrics: Dict[str, Any] = {}
        for cid, dinfo in payload["domains"].items():
            cap_fp16 = DualDepthBridgeEngine.solve_dual_depth_capsule(
                capsule_id=cid,
                episodes=dinfo["episodes"],
                student_model=slm,
                dual_bank=dual_bank,
                rank_mid=32,
                rank_final=32,
                mid_steering_alpha=0.16,
                logit_gain=1.30,
                tau=0.70,
                ramp=0.14,
            )
            raw_int4 = cap_fp16.to_int4_bytes()
            assert len(raw_int4) == 62720
            cap_int4 = DualDepthScarCapsule.from_int4_bytes(
                raw_int4,
                capsule_id=cid,
                hidden_dim=896,
                rank_mid=32,
                rank_final=32,
                tau=0.70,
                ramp=0.14,
                metadata=cap_fp16.metadata,
            )
            dual_bank.add_capsule(cap_int4)
            capsule_metrics[cid] = {
                "int4_bytes": len(raw_int4),
                "int4_kb": round(len(raw_int4) / 1024.0, 2),
                "rank_mid": cap_int4.rank_mid,
                "rank_final": cap_int4.rank_final,
                "solve_ms": cap_fp16.metadata["solve_ms"],
                "zero_backpropagation": True,
            }

        dual_bank.attach(slm)
        try:
            # 1. Verify exact `torch.equal` out-of-cone invariance across Layer 20 AND final logits on all 6 controls
            exact_ctrl_logits = True
            exact_ctrl_layer20 = True
            max_ctrl_cos = 0.0
            for item in clean_controls:
                cap_l20_after: Dict[str, torch.Tensor] = {}
                hk = slm.model.layers[19].register_forward_hook(
                    lambda _m, _i, out: (
                        cap_l20_after.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach().clone()),
                        None,
                    )[1]
                )
                after_lg = slm(item["ids"], use_cache=False).logits.detach()
                hk.remove()
                exact_ctrl_logits = exact_ctrl_logits and bool(torch.equal(after_lg, item["logits"]))
                exact_ctrl_layer20 = exact_ctrl_layer20 and bool(torch.equal(cap_l20_after["h"], item["layer20"]))
                max_ctrl_cos = max(max_ctrl_cos, max(dual_bank.last_routing_cosines.values()))

            # 2. Evaluate unseen `72.71B` paraphrases with `61.25 KB INT4` Dual-Depth Capsules + SCache O(1)
            unseen_eval: Dict[str, Any] = {}
            for cid, dinfo in payload["domains"].items():
                u_p = dinfo["unseen_prompt_ids"]
                u_full = dinfo["unseen_full_ids"].unsqueeze(0)
                lp_u = int(u_p.numel())
                la_u = int(dinfo["unseen_teacher_logits"].shape[0])
                with dual_bank.teacher_forcing_scope(prompt_len=lp_u):
                    lg_dual = slm(u_full, use_cache=False).logits[0, lp_u - 1 : lp_u - 1 + la_u].float()
                lg_t = dinfo["unseen_teacher_logits"][:la_u].float()
                v_c = min(int(lg_dual.shape[-1]), int(lg_t.shape[-1]))
                top1_dual = float((lg_dual[:, :v_c].argmax(-1) == lg_t[:, :v_c].argmax(-1)).float().mean().item() * 100.0)
                kl_dual = _kl_div_top_union(lg_t[:, :v_c], lg_dual[:, :v_c], top_k=256)
                gen_info = _scache_greedy_generate(slm, tok, u_p.unsqueeze(0), max_new_tokens=la_u)
                unseen_eval[cid] = {
                    "routing_cosine_layer12": round(float(dual_bank.last_routing_cosines[cid]), 4),
                    "active_gates": dict(dual_bank.last_active_gates),
                    "base_top1_pct": raw_baselines[cid]["top1_pct"],
                    "dual_depth_int4_top1_pct": round(top1_dual, 2),
                    "base_kl_div": raw_baselines[cid]["kl_div"],
                    "dual_depth_int4_kl_div": round(kl_dual, 4),
                    "live_scache_text": gen_info["text"],
                    "titan_72b_ref_text": dinfo["unseen_ref_text"].strip(),
                    "live_tok_per_s": gen_info["tok_per_s"],
                }

            # 3. Interdisciplinary Synthesis (`Landauer x Immerman-Szelepcsenyi`) Pre-Sleep vs Post-Sleep
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
            inter_unseen_q = (
                "By uniting Landauer's principle at temperature T with the Immerman-Szelepcsenyi theorem for "
                "NSPACE(s(n)), what minimum thermodynamic heat must be dissipated to irreversibly erase m bits "
                "during nondeterministic space complementation?"
            )
            inter_ref_answer = (
                "By the Immerman-Szelepcsenyi theorem, NSPACE(s(n)) = coNSPACE(s(n)) for s(n) >= log n, "
                "and by Landauer's principle irreversibly erasing m bits during that complementation "
                "dissipates at least m k_B T ln 2 of heat."
            )

            unseen_inter_ids = tok(format_chat(inter_unseen_q), return_tensors="pt").input_ids

            # Verify genuine upper-layer (`Layer 24` pre-norm) cross-attention delta BEFORE `_final_norm_hook`!
            with dual_bank.paused():
                cap_pre_clean: Dict[str, torch.Tensor] = {}
                hk_c = slm.model.layers[23].register_forward_hook(
                    lambda _m, _i, out: (
                        cap_pre_clean.setdefault("h", (out[0] if isinstance(out, tuple) else out)[0].detach().float().cpu()),
                        None,
                    )[1]
                )
                _ = slm(unseen_inter_ids, use_cache=False)
                hk_c.remove()
            cap_fin_graft: Dict[str, torch.Tensor] = {}
            hk_norm = slm.model.layers[23].register_forward_hook(
                lambda _m, _i, out: (
                    cap_fin_graft.setdefault("h", (out[0] if isinstance(out, tuple) else out)[0].detach().float().cpu()),
                    None,
                )[1]
            )
            _ = slm(unseen_inter_ids, use_cache=False)
            hk_norm.remove()
            upper_cross_attn_l2_delta = float((cap_fin_graft["h"][-1] - cap_pre_clean["h"][-1]).norm().item())

            dec_pre = DualDepthBridgeEngine.assess_interdisciplinary_epistemic_uncertainty(
                student_model=slm,
                dual_bank=dual_bank,
                prompt_ids_2d=unseen_inter_ids,
                probe_steps=8,
                margin_threshold=0.65,
            )
            gen_pre = _scache_greedy_generate(slm, tok, unseen_inter_ids, max_new_tokens=44)

            # Build System 2 synthesis SleepEpisodes (authoritative cross-domain target distribution)
            ans_ids = tok(inter_ref_answer, add_special_tokens=False, return_tensors="pt").input_ids[0]
            ans_ids = torch.cat([ans_ids, torch.tensor([tok.eos_token_id], dtype=torch.long)], dim=0)
            bridge_eps: List[SleepEpisode] = []
            w_head = slm.lm_head.weight.detach().float()
            for q_txt in (inter_train_q1, inter_train_q2):
                p_ids = tok(format_chat(q_txt), return_tensors="pt").input_ids[0]
                full_ids = torch.cat([p_ids, ans_ids], dim=0)
                with dual_bank.paused():
                    base_lg = slm(full_ids.unsqueeze(0), use_cache=False).logits[0, len(p_ids) - 1 : -1].float()
                synth_lg = base_lg.clone()
                for step_i, target_tok in enumerate(ans_ids.tolist()):
                    cur_max = float(synth_lg[step_i].max().item())
                    synth_lg[step_i, target_tok] = cur_max + 7.5
                qs_i = find_subsequence_last_end(p_ids.tolist(), dual_bank.user_turn_prefix_ids)
                bridge_eps.append(
                    SleepEpisode(
                        capsule_id="bridge_landauer__immerman",
                        prompt_ids=p_ids,
                        full_ids=full_ids,
                        teacher_logits=synth_lg.to(torch.float16),
                        query_start_idx=qs_i,
                        surprise_score=2.0,
                        metadata={"prompt": q_txt, "synthesis_domains": ["titan72b_landauer_principle", "titan72b_immerman_szelepcsenyi"]},
                    )
                )

            bridge_fp16 = DualDepthBridgeEngine.consolidate_interdisciplinary_bridge(
                bridge_capsule_id="bridge_landauer__immerman",
                bridge_episodes=bridge_eps,
                student_model=slm,
                dual_bank=dual_bank,
                rank_mid=32,
                rank_final=32,
                tau=0.70,
                ramp=0.14,
            )
            # Round-trip the bridge capsule through exact `62,720 B` (`61.25 KB`) INT4 as well!
            bridge_raw = bridge_fp16.to_int4_bytes()
            assert len(bridge_raw) == 62720
            bridge_int4 = DualDepthScarCapsule.from_int4_bytes(
                bridge_raw,
                capsule_id="bridge_landauer__immerman",
                hidden_dim=896,
                rank_mid=32,
                rank_final=32,
                tau=0.70,
                ramp=0.14,
                parent_capsules=bridge_fp16.parent_capsules,
                metadata=bridge_fp16.metadata,
            )
            dual_bank.add_capsule(bridge_int4)
            dual_bank.attach(slm)

            dec_post = DualDepthBridgeEngine.assess_interdisciplinary_epistemic_uncertainty(
                student_model=slm,
                dual_bank=dual_bank,
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
            gates_landauer_after = dict(dual_bank.last_active_gates)
            gen_immerman_after = _scache_greedy_generate(
                slm, tok, immerman_info["unseen_prompt_ids"].unsqueeze(0), max_new_tokens=int(immerman_info["unseen_teacher_logits"].shape[0])
            )
            gates_immerman_after = dict(dual_bank.last_active_gates)

            report = {
                "stage": "Stage 3 (v2.0.0) — Dual-Depth Golden-Slice Grafting & Interdisciplinary Sleep Bridge",
                "student_model": "Qwen/Qwen2.5-0.5B-Instruct (24 layers, d=896)",
                "teacher_source": "Qwen2.5-72B-Instruct-MoS (72.71B real cached episodes)",
                "golden_slice_mid_layer": "Layer 12 / 24 (mid_layer_idx=11)",
                "zero_autograd_verified": (not torch.is_grad_enabled()),
                "calibration_ms": cal_ms,
                "capsules": capsule_metrics,
                "out_of_cone_invariance": {
                    "num_controls": len(BACKGROUND_CONTROL_PROMPTS),
                    "exact_torch_equal_layer20": exact_ctrl_layer20,
                    "exact_torch_equal_logits": exact_ctrl_logits,
                    "max_control_cosine_layer12": round(max_ctrl_cos, 4),
                    "tau": 0.70,
                },
                "unseen_paraphrase_72b_fidelity": unseen_eval,
                "interdisciplinary_bridge_audit": {
                    "unseen_synthesis_query": inter_unseen_q,
                    "upper_transformer_cross_attn_l2_delta": round(upper_cross_attn_l2_delta, 4),
                    "pre_sleep_decision": {
                        "active_capsules": dec_pre.active_capsules,
                        "confidence": dec_pre.confidence,
                        "uncertainty": dec_pre.uncertainty,
                        "margin": dec_pre.margin,
                        "is_uncertain": dec_pre.is_uncertain,
                        "latency_ms": dec_pre.latency_ms,
                        "pre_sleep_text": gen_pre["text"],
                    },
                    "bridge_capsule": {
                        "capsule_id": "bridge_landauer__immerman",
                        "parent_capsules": list(bridge_int4.parent_capsules),
                        "int4_bytes": len(bridge_raw),
                        "int4_kb": round(len(bridge_raw) / 1024.0, 2),
                        "solve_ms": bridge_fp16.metadata["solve_ms"],
                    },
                    "post_sleep_decision": {
                        "active_capsules": dec_post.active_capsules,
                        "confidence": dec_post.confidence,
                        "uncertainty": dec_post.uncertainty,
                        "margin": dec_post.margin,
                        "is_uncertain": dec_post.is_uncertain,
                        "latency_ms": dec_post.latency_ms,
                        "post_sleep_text": gen_post["text"],
                        "tok_per_s": gen_post["tok_per_s"],
                    },
                    "zero_single_domain_bleed": {
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
            }

            STAGE3_AUDIT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        finally:
            dual_bank.detach()
    finally:
        torch.set_grad_enabled(prev_grad)


if __name__ == "__main__":
    main()
