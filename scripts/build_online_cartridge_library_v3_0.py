#!/usr/bin/env python3
"""Builds and Audits the Official Online Cartridge Library (`v3.0.0`) from Pure `Qwen2.5-72B-Instruct MoS v2`.

Outputs (ready for direct upload to GitHub Release `v3.0.0` with zero-server HTTP/1.1 206 Range streaming):
  - `results/online_library_v3_0/scarstore_catalog_v3_0.json`
  - `results/online_library_v3_0/qwen3b_v3_0.scarstore` (`9` cartridges x `143,360 B = 140.00 KB = 35 x 4 KB`)
  - `results/online_library_v3_0/qwen3b_v3_0.manifest.json`
  - `results/online_library_v3_0/qwen3b_v3_0.shell.scarstore.gz` (Zero-payload Cloud Shell)
  - `results/online_library_v3_0/qwen0p5b_v3_0.scarstore` (`9` cartridges x `62,720 B = 61.25 KB`)
  - `results/online_library_v3_0/qwen0p5b_v3_0.manifest.json`
  - `results/online_library_v3_0/qwen0p5b_v3_0.shell.scarstore.gz` (Zero-payload Cloud Shell)
  - `results/online_cartridge_library_v3_0_audit.json`
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scar_cache.cli_72b import CanonicalMoSEngine
from scar_cache.cli_scar_store import (
    DEFAULT_CALIBRATION_PROMPTS,
    OFFICIAL_RELEASE_BASE_URL,
    OFFICIAL_RELEASE_TAG,
    ScarStoreHTTPRangeServer,
    format_chatml_prompt,
    generate_with_scar_store,
    load_episodes_bundle,
    make_http_range_capsule_fetcher,
)
from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarStore,
)
from scar_cache.scar_grafting import (
    SleepEpisode,
    detect_chatml_user_prefix_ids,
    find_subsequence_last_end,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MOS_72B_DIR = REPO_ROOT / "models" / "qwen25_72b_mos_v2"
LEGACY_EPISODES_PT = REPO_ROOT / "results" / "titan72b_episodes_v2.pt"
LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"
EPISODES_V3_PT = LIB_DIR / "titan72b_online_library_episodes_v3_0.pt"
AUDIT_JSON = REPO_ROOT / "results" / "online_cartridge_library_v3_0_audit.json"


CARTRIDGE_SPECS: List[Dict[str, Any]] = [
    {
        "capsule_id": "titan72b_kabatiansky_levenshtein",
        "title": "High-Dimensional Sphere Packing — Kabatiansky-Levenshtein Bound",
        "title_pt": "Empacotamento Esférico em Alta Dimensão — Cota de Kabatiansky-Levenshtein",
        "category": "Geometry & Coding Theory",
        "summary": "Asymptotic exponential upper bound on spherical codes and kissing numbers in R^n.",
        "train_prompts": [
            "What does the Kabatiansky-Levenshtein theorem prove about unit spheres in high-dimensional spaces?",
            "In high-dimensional geometry, what upper bound does the Kabatiansky-Levenshtein theorem prove for unit spheres?",
        ],
        "example_prompt": "Explain concisely what the Kabatiansky-Levenshtein theorem proves for high-dimensional unit spheres.",
    },
    {
        "capsule_id": "titan72b_landauer_principle",
        "title": "Thermodynamics of Computation — Landauer's Principle (k_B T ln 2)",
        "title_pt": "Termodinâmica da Computação — Princípio de Landauer (k_B T ln 2)",
        "category": "Physics of Information",
        "summary": "Minimum irreversible heat dissipation bound k_B T ln 2 per erased bit at temperature T.",
        "train_prompts": [
            "What does Landauer's principle state about the minimum energy dissipated when erasing one bit of information at temperature T?",
            "In physics of computation, what minimum heat dissipation bound does Landauer's principle establish for erasing one bit at temperature T?",
        ],
        "example_prompt": "Explain concisely what Landauer's principle states for the minimum energy dissipated when erasing one bit at temperature T.",
    },
    {
        "capsule_id": "titan72b_immerman_szelepcsenyi",
        "title": "Space Complexity — Immerman-Szelepcsényi Theorem (NSPACE = co-NSPACE)",
        "title_pt": "Complexidade Espacial — Teorema de Immerman-Szelepcsényi (NSPACE = co-NSPACE)",
        "category": "Theoretical Computer Science",
        "summary": "Closure of nondeterministic space complexity classes under complementation via inductive counting.",
        "train_prompts": [
            "What does the Immerman-Szelepcsenyi theorem prove about nondeterministic space complexity classes?",
            "In complexity theory, what closure property does the Immerman-Szelepcsenyi theorem establish for nondeterministic space classes?",
        ],
        "example_prompt": "Explain concisely what the Immerman-Szelepcsenyi theorem proves about nondeterministic space complexity classes.",
    },
    {
        "capsule_id": "titan72b_fischer_lynch_paterson",
        "title": "Distributed Consensus — FLP Impossibility Theorem",
        "title_pt": "Sistemas Distribuídos — Teorema da Impossibilidade FLP",
        "category": "Distributed Systems",
        "summary": "Impossibility of deterministic fault-tolerant consensus in asynchronous message-passing systems with one crash.",
        "train_prompts": [
            "What does the Fischer-Lynch-Paterson (FLP) impossibility theorem prove about deterministic consensus in asynchronous distributed systems?",
            "In distributed computing, what impossibility result does the Fischer-Lynch-Paterson (FLP) theorem establish under a single unannounced process crash?",
        ],
        "example_prompt": "Explain concisely what the Fischer-Lynch-Paterson impossibility theorem proves about asynchronous consensus.",
    },
    {
        "capsule_id": "titan72b_chaitin_incompleteness",
        "title": "Algorithmic Information Theory — Chaitin's Incompleteness Theorem",
        "title_pt": "Teoria Algorítmica da Informação — Incompletude de Chaitin (Kolmogorov)",
        "category": "Logic & Information Theory",
        "summary": "Upper bound L beyond which no formal axiomatic system can prove lower bounds K(x) > L on Kolmogorov complexity.",
        "train_prompts": [
            "What does Chaitin's incompleteness theorem prove about proving lower bounds on Kolmogorov complexity in formal axiomatic systems?",
            "In algorithmic information theory, what explicit limit does Chaitin's incompleteness theorem establish on provable Kolmogorov complexity?",
        ],
        "example_prompt": "Explain concisely what Chaitin's incompleteness theorem proves regarding Kolmogorov complexity.",
    },
    {
        "capsule_id": "titan72b_regev_lwe_lattice",
        "title": "Post-Quantum Cryptography — Regev's LWE Worst-Case Lattice Reduction",
        "title_pt": "Criptografia Pós-Quântica — Redução Quântica LWE de Regev",
        "category": "Cryptography & Security",
        "summary": "Quantum worst-case to average-case reduction from GapSVP/SIVP lattice problems to Learning With Errors (LWE).",
        "train_prompts": [
            "What worst-case to average-case reduction did Oded Regev prove for the Learning With Errors (LWE) problem in lattice cryptography?",
            "In post-quantum lattice cryptography, how does Regev's theorem connect average-case Learning With Errors (LWE) to worst-case lattice problems?",
        ],
        "example_prompt": "Explain concisely what Oded Regev's theorem proves connecting Learning With Errors (LWE) to worst-case lattice problems.",
    },
    {
        "capsule_id": "titan72b_marchenko_pastur_law",
        "title": "Random Matrix Theory — Marchenko-Pastur Spectral Distribution",
        "title_pt": "Matrizes Aleatórias em IA — Lei Espectral de Marchenko-Pastur",
        "category": "Spectral Math & Deep Learning",
        "summary": "Limiting eigenvalue density and compact support [(1 - sqrt(gamma))^2, (1 + sqrt(gamma))^2] of Wishart covariance matrices.",
        "train_prompts": [
            "What does the Marchenko-Pastur law state about the limiting eigenvalue distribution of large random sample covariance matrices?",
            "In random matrix theory, what asymptotic spectral distribution does the Marchenko-Pastur theorem establish as matrix dimensions grow proportionally?",
        ],
        "example_prompt": "Explain concisely what the Marchenko-Pastur law establishes for the eigenvalues of large random covariance matrices.",
    },
    {
        "capsule_id": "titan72b_shannon_hartley_capacity",
        "title": "Communications & Signal Theory — Shannon-Hartley Channel Capacity",
        "title_pt": "Engenharia de Comunicações — Lei de Capacidade de Shannon-Hartley",
        "category": "Communications & Information",
        "summary": "Tight upper rate C = B log2(1 + S/N) for arbitrarily reliable communication over an AWGN channel.",
        "train_prompts": [
            "What formula and theoretical limit does the Shannon-Hartley theorem establish for error-free information capacity over a noisy AWGN channel?",
            "In information theory, how does the Shannon-Hartley theorem relate channel capacity C to bandwidth B and signal-to-noise ratio S/N?",
        ],
        "example_prompt": "State concisely the Shannon-Hartley channel capacity formula and what it proves for noisy AWGN channels.",
    },
]

BRIDGE_SPEC: Dict[str, Any] = {
    "capsule_id": "bridge_landauer_immerman",
    "title": "Interdisciplinary Sleep Bridge — Thermodynamics × NSPACE Complexity",
    "title_pt": "Ponte Interdisciplinar — Termodinâmica de Landauer × Complexidade NSPACE",
    "category": "Interdisciplinary Synthesis",
    "summary": "Conjunctive Sleep Bridge uniting Landauer's k_B T ln 2 irreversible erasure bound with Immerman-Szelepcsényi NSPACE complementation.",
    "parents": ["titan72b_landauer_principle", "titan72b_immerman_szelepcsenyi"],
    "bridge_prompt": (
        "In one concise sentence, connect Landauer's minimum energy bound for erasing one bit at temperature T "
        "with what the Immerman-Szelepcsenyi theorem proves about nondeterministic space classes."
    ),
    "example_prompt": (
        "State Landauer's thermodynamic bound for erasing a bit at temperature T and connect it to the "
        "Immerman-Szelepcsenyi closure property of nondeterministic space classes."
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def extract_or_load_all_72b_episodes(dev: torch.device, max_new_tokens: int = 26) -> Dict[str, List[SleepEpisode]]:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    episodes_map: Dict[str, List[SleepEpisode]] = {}

    if EPISODES_V3_PT.exists():
        episodes_map = load_episodes_bundle(EPISODES_V3_PT)
        needed = {s["capsule_id"] for s in CARTRIDGE_SPECS} | {BRIDGE_SPEC["capsule_id"]}
        if needed.issubset(set(episodes_map.keys())):
            print(f"[Phase 1] Loaded complete 9-domain 72.71B episode cache from {EPISODES_V3_PT}", flush=True)
            return episodes_map

    if LEGACY_EPISODES_PT.exists():
        legacy = load_episodes_bundle(LEGACY_EPISODES_PT)
        episodes_map.update(legacy)
        print(f"[Phase 1] Loaded {len(legacy)} cached 72.71B domains from {LEGACY_EPISODES_PT}", flush=True)

    missing_specs = [s for s in CARTRIDGE_SPECS if s["capsule_id"] not in episodes_map]
    need_bridge = BRIDGE_SPEC["capsule_id"] not in episodes_map

    if missing_specs or need_bridge:
        print(
            f"[Phase 1] Launching pure standalone Qwen2.5-72B-Instruct MoS v2 ({MOS_72B_DIR}) "
            f"for {len(missing_specs)} new core domains + {'1 bridge' if need_bridge else '0 bridges'} ...",
            flush=True,
        )
        t0 = time.perf_counter()
        titan_72b = CanonicalMoSEngine(MOS_72B_DIR, device=str(dev), streaming=True)
        tok = titan_72b.tokenizer
        prefix_ids = detect_chatml_user_prefix_ids(tok)
        eos_ids = [tok.eos_token_id, 151645, 151643]

        gen_prompts_1d: List[torch.Tensor] = []
        p1_prompts_1d: List[torch.Tensor] = []
        for s in missing_specs:
            gen_prompts_1d.append(tok(format_chatml_prompt(s["train_prompts"][0]), return_tensors="pt").input_ids[0].cpu())
            p1_prompts_1d.append(tok(format_chatml_prompt(s["train_prompts"][1]), return_tensors="pt").input_ids[0].cpu())
        if need_bridge:
            gen_prompts_1d.append(tok(format_chatml_prompt(BRIDGE_SPEC["bridge_prompt"]), return_tensors="pt").input_ids[0].cpu())

        print(f"[Phase 1A] Running packed parallel 72.71B generation across {len(gen_prompts_1d)} streams ({max_new_tokens} tok) ...", flush=True)
        gen_results = titan_72b.generate_packed_greedy(
            prompt_1d_list=gen_prompts_1d,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
        )

        if missing_specs:
            p1_full_seqs: List[torch.Tensor] = []
            for idx in range(len(missing_specs)):
                ans0_ids, _ = gen_results[idx]
                p1_full_seqs.append(torch.cat([p1_prompts_1d[idx], ans0_ids], dim=0))
            print(f"[Phase 1B] Scoring {len(p1_full_seqs)} secondary phrasings in 1 packed 72.71B sweep ...", flush=True)
            p1_logits_list = titan_72b.forward_packed_sequences(p1_full_seqs)

            for idx, s in enumerate(missing_specs):
                cid = s["capsule_id"]
                p0_ids = gen_prompts_1d[idx]
                p1_ids = p1_prompts_1d[idx]
                ans0_ids, lg0_ans = gen_results[idx]
                lp1 = int(p1_ids.numel())
                la0 = int(ans0_ids.numel())
                lg1_ans = p1_logits_list[idx][lp1 - 1 : lp1 - 1 + la0, :].detach().to(torch.float16).cpu()
                ans0_txt = tok.decode(ans0_ids, skip_special_tokens=True)
                print(f"  [72.71B Teacher -> {cid}]: {ans0_txt}", flush=True)

                ep0 = SleepEpisode(
                    capsule_id=cid,
                    prompt_ids=p0_ids,
                    full_ids=torch.cat([p0_ids, ans0_ids], dim=0),
                    teacher_logits=lg0_ans.detach().to(torch.float16).cpu(),
                    query_start_idx=find_subsequence_last_end(p0_ids.tolist(), prefix_ids),
                    surprise_score=1.0,
                    metadata={"prompt": s["train_prompts"][0], "teacher": "Qwen2.5-72B-Instruct-MoS (72.71B)", "answer": ans0_txt},
                )
                ep1 = SleepEpisode(
                    capsule_id=cid,
                    prompt_ids=p1_ids,
                    full_ids=p1_full_seqs[idx],
                    teacher_logits=lg1_ans,
                    query_start_idx=find_subsequence_last_end(p1_ids.tolist(), prefix_ids),
                    surprise_score=1.0,
                    metadata={"prompt": s["train_prompts"][1], "teacher": "Qwen2.5-72B-Instruct-MoS (72.71B)", "answer": ans0_txt},
                )
                episodes_map[cid] = [ep0, ep1]

        if need_bridge:
            b_idx = len(missing_specs)
            pb_ids = gen_prompts_1d[b_idx]
            ansb_ids, lgb_ans = gen_results[b_idx]
            ansb_txt = tok.decode(ansb_ids, skip_special_tokens=True)
            print(f"  [72.71B Bridge  -> {BRIDGE_SPEC['capsule_id']}]: {ansb_txt}", flush=True)
            ep_b = SleepEpisode(
                capsule_id=BRIDGE_SPEC["capsule_id"],
                prompt_ids=pb_ids,
                full_ids=torch.cat([pb_ids, ansb_ids], dim=0),
                teacher_logits=lgb_ans.detach().to(torch.float16).cpu(),
                query_start_idx=find_subsequence_last_end(pb_ids.tolist(), prefix_ids),
                surprise_score=1.0,
                metadata={"prompt": BRIDGE_SPEC["bridge_prompt"], "teacher": "Qwen2.5-72B-Instruct-MoS (72.71B)", "answer": ansb_txt},
            )
            episodes_map[BRIDGE_SPEC["capsule_id"]] = [ep_b]

        print(f"[Phase 1 Done] Pure 72.71B extraction completed in {time.perf_counter() - t0:.1f}s", flush=True)
        del titan_72b
        gc.collect()
        if dev.type == "mps":
            torch.mps.empty_cache()

    torch.save({"episodes_by_capsule": episodes_map}, EPISODES_V3_PT)
    return episodes_map


@torch.no_grad()
def build_tier_library(
    tier_tag: str,
    student_id: str,
    episodes_map: Dict[str, List[SleepEpisode]],
    dev: torch.device,
) -> Dict[str, Any]:
    print(f"\n[Phase 2 - {tier_tag}] Synthesizing 9 Closed-Form Cartridges for {student_id} ...", flush=True)
    t0 = time.perf_counter()
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(student_id, local_files_only=True)
    slm = AutoModelForCausalLM.from_pretrained(student_id, dtype=dtype, local_files_only=True).to(dev).eval()

    head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=DEFAULT_CALIBRATION_PROMPTS,
        tokenizer=tok,
        chat_formatter=format_chatml_prompt,
        preserved_rank=4,
    )

    tau_tier = 0.64 if tier_tag == "3b" else 0.62
    for spec in CARTRIDGE_SPECS:
        cid = spec["capsule_id"]
        eps = episodes_map[cid]
        teacher_ans = str(eps[0].metadata.get("answer") or tok.decode(eps[0].full_ids[eps[0].prompt_ids.numel():], skip_special_tokens=True))
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=eps,
            student_model=slm,
            head_bank=head_bank,
            rank=60,
            num_key_prototypes=8,
            tau=tau_tier,
            ramp=0.16,
        )
        cap.metadata.update(
            {
                "title": spec["title"],
                "title_pt": spec["title_pt"],
                "category": spec["category"],
                "summary": spec["summary"],
                "example_prompt": spec["example_prompt"],
                "teacher_model": "Qwen2.5-72B-Instruct MoS v2 (72.71B)",
                "teacher_reference": teacher_ans,
            }
        )
        head_bank.add_capsule(cap)

    bridge_eps = episodes_map[BRIDGE_SPEC["capsule_id"]]
    bridge_ans = str(bridge_eps[0].metadata.get("answer") or tok.decode(bridge_eps[0].full_ids[bridge_eps[0].prompt_ids.numel():], skip_special_tokens=True))
    bridge_cap = HeadExpandedAttentionEngine.consolidate_interdisciplinary_bridge(
        bridge_capsule_id=BRIDGE_SPEC["capsule_id"],
        bridge_episodes=bridge_eps,
        student_model=slm,
        head_bank=head_bank,
        parent_capsules=BRIDGE_SPEC["parents"],
        rank=60,
        num_key_prototypes=8,
        tau=0.62,
        ramp=0.16,
    )
    bridge_cap.metadata.update(
        {
            "title": BRIDGE_SPEC["title"],
            "title_pt": BRIDGE_SPEC["title_pt"],
            "category": BRIDGE_SPEC["category"],
            "summary": BRIDGE_SPEC["summary"],
            "example_prompt": BRIDGE_SPEC["example_prompt"],
            "teacher_model": "Qwen2.5-72B-Instruct MoS v2 (72.71B)",
            "teacher_reference": bridge_ans,
        }
    )
    head_bank.add_capsule(bridge_cap)

    prefix = f"qwen{tier_tag}_v3_0"
    store_path = LIB_DIR / f"{prefix}.scarstore"
    manifest_path = LIB_DIR / f"{prefix}.manifest.json"
    shell_path = LIB_DIR / f"{prefix}.shell.scarstore"
    shell_gz_path = LIB_DIR / f"{prefix}.shell.scarstore.gz"

    store = HeadExpandedScarStore.build_from_bank(head_bank, store_path, force_kronecker=True)
    summary = store.inspect_summary()
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export Zero-Payload Cloud Shell (`0` installed cartridges, `9` global directory entries)
    shell_store = store.export_subset(shell_path, include_capsule_ids=[])
    shell_summary = shell_store.inspect_summary()
    shell_store.close()
    with open(shell_path, "rb") as sf, gzip.open(shell_gz_path, "wb", compresslevel=9) as gf:
        gf.write(sf.read())

    # Live HTTP/1.1 206 Range verification from Zero-Payload Shell -> Master Store!
    sample_verify: List[Dict[str, Any]] = []
    with ScarStoreHTTPRangeServer(store_path) as srv:
        fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
        live_shell = HeadExpandedScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
        try:
            for test_spec in (CARTRIDGE_SPECS[1], CARTRIDGE_SPECS[3], CARTRIDGE_SPECS[6]):
                res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=test_spec["example_prompt"],
                    max_new_tokens=36,
                    http_client=http_client,
                    persist_fetched=False,
                )
                assert res.active_gates and max(res.active_gates, key=res.active_gates.get) == test_spec["capsule_id"] and res.active_gates[test_spec["capsule_id"]] >= 0.80, (
                    f"[{tier_tag}] Expected dominant {test_spec['capsule_id']}, got {res.active_gates}"
                )
                assert len(res.http_range_events) >= 1 and all(e["http_status"] == 206 for e in res.http_range_events)
                matched_ev = next(e for e in res.http_range_events if e["capsule_id"] == test_spec["capsule_id"])
                assert matched_ev["bytes_received"] == store.record_bytes
                sample_verify.append(
                    {
                        "capsule_id": test_spec["capsule_id"],
                        "answer": res.text,
                        "active_gates": res.active_gates,
                        "http_206_event": matched_ev,
                        "total_http_206_fetches_in_turn": len(res.http_range_events),
                    }
                )
            # Control out-of-cone verification: 0 HTTP requests, 0 mmap bytes!
            ctrl_res, _ = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_shell,
                prompt_text_or_ids="What is the capital of France?",
                max_new_tokens=16,
                http_client=http_client,
                persist_fetched=False,
            )
            assert len(ctrl_res.active_gates) == 0 and len(ctrl_res.http_range_events) == 0
        finally:
            live_shell.close()

    store.close()
    del slm
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()

    elapsed_s = round(time.perf_counter() - t0, 2)
    print(
        f"[Phase 2 - {tier_tag} Complete] {store_path.name} ({summary['file_mib']} MiB) | "
        f"shell.gz={shell_gz_path.stat().st_size / (1024**2):.2f} MiB | solved+verified in {elapsed_s}s",
        flush=True,
    )
    return {
        "tier": tier_tag,
        "student_model": student_id,
        "solve_and_verify_s": elapsed_s,
        "summary": summary,
        "shell_summary": shell_summary,
        "sample_http206_verifications": sample_verify,
        "assets": {
            "store_file": store_path.name,
            "store_bytes": store_path.stat().st_size,
            "store_sha256": sha256_file(store_path),
            "manifest_file": manifest_path.name,
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256_file(manifest_path),
            "shell_gz_file": shell_gz_path.name,
            "shell_gz_bytes": shell_gz_path.stat().st_size,
            "shell_gz_sha256": sha256_file(shell_gz_path),
        },
    }


def main() -> int:
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    episodes_map = extract_or_load_all_72b_episodes(dev=dev, max_new_tokens=26)

    res_0p5b = build_tier_library("0p5b", "Qwen/Qwen2.5-0.5B-Instruct", episodes_map, dev)
    res_3b = build_tier_library("3b", "Qwen/Qwen2.5-3B-Instruct", episodes_map, dev)

    slots_3b = {s["capsule_id"]: s for s in res_3b["summary"]["slots"]}
    slots_0p5b = {s["capsule_id"]: s for s in res_0p5b["summary"]["slots"]}

    all_specs = list(CARTRIDGE_SPECS) + [BRIDGE_SPEC]
    catalog_items: List[Dict[str, Any]] = []
    for sp in all_specs:
        cid = sp["capsule_id"]
        s3 = slots_3b[cid]
        s0 = slots_0p5b[cid]
        catalog_items.append(
            {
                "capsule_id": cid,
                "title": sp["title"],
                "title_pt": sp["title_pt"],
                "category": sp["category"],
                "summary": sp["summary"],
                "example_prompt": sp["example_prompt"],
                "parent_capsules": sp.get("parents", []),
                "teacher_reference": s3["metadata"].get("teacher_reference", ""),
                "coord_3b": s3["coord"],
                "byte_offset_3b": s3["byte_offset"],
                "record_bytes_3b": s3["record_bytes"],
                "coord_0p5b": s0["coord"],
                "byte_offset_0p5b": s0["byte_offset"],
                "record_bytes_0p5b": s0["record_bytes"],
            }
        )

    catalog_payload = {
        "library_name": "Mixture of Scars (MoS) Official Online Cartridge Library",
        "version": OFFICIAL_RELEASE_TAG,
        "teacher_model": "Qwen2.5-72B-Instruct MoS v2 (72.71B, 80 Layers)",
        "synthesis_method": "100% Closed-Form Woodbury Rosetta + AlphaEdit Null-Space SVD (ZERO Backpropagation / ZERO SGD)",
        "release_base_url": OFFICIAL_RELEASE_BASE_URL,
        "tiers": {
            "3b": {
                "student_model": "Qwen/Qwen2.5-3B-Instruct",
                "hidden_dim": 2048,
                "record_bytes": 143360,
                "record_kb": 140.0,
                "os_4kb_pages": 35,
                "assets": res_3b["assets"],
            },
            "0.5b": {
                "student_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "hidden_dim": 896,
                "record_bytes": 62720,
                "record_kb": 61.25,
                "assets": res_0p5b["assets"],
            },
        },
        "cartridges": catalog_items,
    }
    catalog_path = LIB_DIR / "scarstore_catalog_v3_0.json"
    catalog_path.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    audit_report = {
        "version": OFFICIAL_RELEASE_TAG,
        "num_online_cartridges": len(catalog_items),
        "catalog_file": str(catalog_path.relative_to(REPO_ROOT)),
        "catalog_sha256": sha256_file(catalog_path),
        "tier_3b_sample_http206": res_3b["sample_http206_verifications"],
        "tier_0p5b_sample_http206": res_0p5b["sample_http206_verifications"],
        "tiers": catalog_payload["tiers"],
    }
    AUDIT_JSON.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Built Official Online Cartridge Library ({len(catalog_items)} cartridges across 3B & 0.5B) -> {LIB_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
