"""Builds the 16-Cartridge Bilingual Online Library (`v3.1.0`) for `0.5B` & `3B` (`9` EN `72.71B` + `7` PT-BR).

Preserves `qwen0p5b_v3_0.scarstore` and `qwen3b_v3_0.scarstore` untouched for `v3.0.0` SHA-256 parity,
while generating:
  - `results/online_library_v3_0/qwen0p5b_bilingual_v3_1.scarstore` + `.shell.scarstore`
  - `results/online_library_v3_0/qwen3b_bilingual_v3_1.scarstore` + `.shell.scarstore`
  - `results/online_library_v3_0/scarstore_bilingual_catalog_v3_1.json`
  - Exports `qwen0p5b_bilingual_v3_1.scarstore` to `webgpu_dist/qwen0p5b/` (`capsules.scarstore` + `scarstore_router.*`)
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_cache.cli_scar_store import (
    DEFAULT_BILINGUAL_CALIBRATION_PROMPTS,
    ScarStoreHTTPRangeServer,
    format_chatml_prompt,
    generate_with_scar_store,
    make_http_range_capsule_fetcher,
    select_chat_system_prompt,
)
from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarStore,
)
from scar_cache.mos_student import load_mos_student_model
from scar_cache.open_weights_factory import (
    PT_BR_SPECIALIST_CARTRIDGE_SPECS,
    mint_capsules_from_specs,
)
from scar_cache.webgpu_engine import export_scarstore_for_webgpu
from scripts.build_online_cartridge_library_v3_0 import (
    BRIDGE_SPEC,
    CARTRIDGE_SPECS,
)

LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"
EPISODES_PT = LIB_DIR / "titan72b_online_library_episodes_v3_0.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def build_bilingual_tier(
    tier_tag: str,
    student_id: str,
    episodes_map: Dict[str, Any],
    dev: torch.device,
) -> Dict[str, Any]:
    print(f"\n[Bilingual v3.1 - {tier_tag}] Building 16-Cartridge (9 EN 72.71B + 7 PT-BR) Store for {student_id} ...", flush=True)
    t0 = time.perf_counter()
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(student_id, local_files_only=True)
    slm = load_mos_student_model(student_id, tokenizer=tok, device=dev, dtype=dtype, local_files_only=True)

    fmt = lambda q: format_chatml_prompt(q, system_prompt=select_chat_system_prompt(q))
    head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=DEFAULT_BILINGUAL_CALIBRATION_PROMPTS,
        tokenizer=tok,
        chat_formatter=fmt,
        preserved_rank=4,
    )

    tau_en = 0.72 if tier_tag == "3b" else 0.70
    for spec in CARTRIDGE_SPECS:
        cid = spec["capsule_id"]
        eps = episodes_map[cid]
        teacher_ans = str(
            eps[0].metadata.get("answer")
            or tok.decode(eps[0].full_ids[eps[0].prompt_ids.numel() :], skip_special_tokens=True)
        )
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=eps,
            student_model=slm,
            head_bank=head_bank,
            rank=60,
            num_key_prototypes=8,
            tau=tau_en,
            ramp=0.14,
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
    bridge_ans = str(
        bridge_eps[0].metadata.get("answer")
        or tok.decode(bridge_eps[0].full_ids[bridge_eps[0].prompt_ids.numel() :], skip_special_tokens=True)
    )
    bridge_cap = HeadExpandedAttentionEngine.consolidate_interdisciplinary_bridge(
        bridge_capsule_id=BRIDGE_SPEC["capsule_id"],
        bridge_episodes=bridge_eps,
        student_model=slm,
        head_bank=head_bank,
        parent_capsules=BRIDGE_SPEC["parents"],
        rank=60,
        num_key_prototypes=8,
        tau=0.68,
        ramp=0.14,
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

    pt_caps = mint_capsules_from_specs(
        specs=PT_BR_SPECIALIST_CARTRIDGE_SPECS,
        student_model=slm,
        student_tokenizer=tok,
        head_bank=head_bank,
        chat_formatter=fmt,
        teacher_model=None,
        teacher_tokenizer=None,
        teacher_label="Curated PT-BR Specialist Trajectory (Woodbury + AlphaEdit)",
        rank=60,
        num_key_prototypes=8,
    )
    for pt_cap, pt_spec in zip(pt_caps, PT_BR_SPECIALIST_CARTRIDGE_SPECS):
        pt_cap.metadata["teacher_reference"] = pt_spec["episodes"][0]["answer"]
        head_bank.add_capsule(pt_cap)

    prefix = f"qwen{tier_tag}_bilingual_v3_1"
    store_path = LIB_DIR / f"{prefix}.scarstore"
    manifest_path = LIB_DIR / f"{prefix}.manifest.json"
    shell_path = LIB_DIR / f"{prefix}.shell.scarstore"
    shell_gz_path = LIB_DIR / f"{prefix}.shell.scarstore.gz"

    store = HeadExpandedScarStore.build_from_bank(head_bank, store_path, force_kronecker=True)
    summary = store.inspect_summary()
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    shell_store = store.export_subset(shell_path, include_capsule_ids=[])
    shell_summary = shell_store.inspect_summary()
    shell_store.close()
    with open(shell_path, "rb") as sf, gzip.open(shell_gz_path, "wb", compresslevel=9) as gf:
        gf.write(sf.read())

    # Refresh cached user shell if present so `mos-store ui` picks up the latest 16-slot shell cleanly
    user_cache_shell = Path.home() / ".cache" / "mixture_of_scars" / "scarstore" / f"{prefix}.shell.scarstore"
    if user_cache_shell.exists():
        shutil.copyfile(shell_path, user_cache_shell)

    # Live HTTP 206 verification across PT-BR user prompts, EN 72B prompts, and out-of-cone PT-BR greetings!
    verifications: List[Dict[str, Any]] = []
    with ScarStoreHTTPRangeServer(store_path) as srv:
        test_cases = [
            ("me sabe sobre o princípio de Landauer", "ptbr_principio_landauer", "k_B T ln 2"),
            ("O que é a constante de Boltzmann e onde ela se aplica?", "ptbr_constante_boltzmann", "1,380649"),
            ("Como funciona o direito trabalhista para faxineiras no brasil?", "ptbr_direito_domestico_lc150", "150/2015"),
            ("Qual arquitetura você usa?", "ptbr_arquitetura_mos_scache", "Mixture of Scars"),
            ("Detalhe sua arquitetura", "ptbr_arquitetura_mos_scache", "SCache"),
            ("Como funciona a sua arquitetura?", "ptbr_arquitetura_mos_scache", "Mixture of Scars"),
            (
                "Explique o teorema da impossibilidade FLP (Fischer-Lynch-Paterson) em sistemas distribuídos assíncronos.",
                "ptbr_teorema_flp_consenso",
                "Fischer-Lynch-Paterson",
            ),
            (
                "O que prova o teorema de Immerman-Szelepcsényi sobre classes de complexidade espacial não-determinística?",
                "ptbr_immerman_szelepcsenyi",
                "co-NSPACE",
            ),
            (
                "O que estabelece a lei espectral de Marchenko-Pastur para matrizes aleatórias de covariância?",
                "ptbr_lei_marchenko_pastur",
                "Marchenko-Pastur",
            ),
            (
                "Explain concisely what Landauer's principle states for the minimum energy dissipated when erasing one bit at temperature T.",
                "titan72b_landauer_principle",
                "ln 2",
            ),
        ]
        for prompt_str, expected_cid, expected_substr in test_cases:
            fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
            live_shell = HeadExpandedScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
            try:
                res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=prompt_str,
                    max_new_tokens=96,
                    system_prompt=select_chat_system_prompt(prompt_str),
                    http_client=http_client,
                    persist_fetched=False,
                )
            finally:
                live_shell.close()
            assert res.active_gates and max(res.active_gates, key=res.active_gates.get) == expected_cid, (
                f"[{tier_tag}] Expected dominant {expected_cid} on {prompt_str!r}, got {res.active_gates} (cos={res.routing_cosines})"
            )
            assert len(res.http_range_events) >= 1 and res.http_range_events[0]["http_status"] == 206, (
                f"[{tier_tag}] Expected HTTP 206 fetch for {expected_cid}, got {res.http_range_events}"
            )
            assert expected_substr in res.text, (
                f"[{tier_tag}] Expected {expected_substr!r} in answer for {expected_cid}, got {res.text!r}"
            )
            print(
                f"  ✓ [{tier_tag}] {expected_cid} | gates={res.active_gates} | "
                f"HTTP206={len(res.http_range_events)} | ans={res.text!r}",
                flush=True,
            )
            verifications.append(
                {
                    "prompt": prompt_str,
                    "expected_capsule_id": expected_cid,
                    "active_gates": res.active_gates,
                    "http_206_events": res.http_range_events,
                    "answer": res.text,
                }
            )

        # Out-of-cone PT-BR greeting check: strictly 0 active cartridges & 0 HTTP 206 bytes!
        fetcher_c, http_client_c = make_http_range_capsule_fetcher(srv.base_url)
        live_shell_c = HeadExpandedScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher_c)
        try:
            ctrl_res, _ = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_shell_c,
                prompt_text_or_ids="Olá, tudo bem? Quem é você?",
                max_new_tokens=64,
                system_prompt=select_chat_system_prompt("Olá, tudo bem? Quem é você?"),
                http_client=http_client_c,
                persist_fetched=False,
            )
            assert len(ctrl_res.active_gates) == 0 and len(ctrl_res.http_range_events) == 0, (
                f"[{tier_tag}] Expected 0 active cartridges on casual PT-BR greeting, got {ctrl_res.active_gates}"
            )
            print(f"  ✓ [{tier_tag}] Out-of-cone PT-BR control: 0 cartridges, 0 B HTTP 206 | ans={ctrl_res.text[:80]!r}", flush=True)
        finally:
            live_shell_c.close()

    store.close()
    del slm
    gc.collect()
    if dev.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    elapsed_s = round(time.perf_counter() - t0, 2)
    print(f"[Bilingual v3.1 - {tier_tag} Complete] {store_path.name} ({len(summary['slots'])} cartridges) in {elapsed_s}s", flush=True)
    return {
        "tier": tier_tag,
        "student_model": student_id,
        "solve_and_verify_s": elapsed_s,
        "summary": summary,
        "shell_summary": shell_summary,
        "verifications": verifications,
        "assets": {
            "store_file": store_path.name,
            "store_bytes": store_path.stat().st_size,
            "store_sha256": sha256_file(store_path),
            "manifest_file": manifest_path.name,
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256_file(manifest_path),
            "shell_file": shell_path.name,
            "shell_bytes": shell_path.stat().st_size,
            "shell_gz_file": shell_gz_path.name,
            "shell_gz_bytes": shell_gz_path.stat().st_size,
        },
    }


def main() -> int:
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_ep = torch.load(EPISODES_PT, map_location="cpu", weights_only=False)
    episodes_map = raw_ep["episodes_by_capsule"]

    res_0p5b = build_bilingual_tier("0p5b", "Qwen/Qwen2.5-0.5B-Instruct", episodes_map, dev)
    res_3b = build_bilingual_tier("3b", "Qwen/Qwen2.5-3B-Instruct", episodes_map, dev)

    slots_3b = {s["capsule_id"]: s for s in res_3b["summary"]["slots"]}
    slots_0p5b = {s["capsule_id"]: s for s in res_0p5b["summary"]["slots"]}

    all_specs = list(CARTRIDGE_SPECS) + [BRIDGE_SPEC] + list(PT_BR_SPECIALIST_CARTRIDGE_SPECS)
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
                "teacher_model": s3["metadata"].get("teacher_model", "Qwen2.5-72B-Instruct MoS v2"),
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
        "library_name": "Mixture of Scars (MoS) Bilingual Online Cartridge Library (EN 72.71B + PT-BR)",
        "version": "v3.1.0",
        "synthesis_method": "100% Closed-Form Woodbury Rosetta + AlphaEdit Null-Space SVD (ZERO Backpropagation / ZERO SGD)",
        "tiers": {
            "3b": {
                "student_model": "Qwen/Qwen2.5-3B-Instruct",
                "hidden_dim": 2048,
                "record_bytes": 143360,
                "record_kb": 140.0,
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
    cat_path = LIB_DIR / "scarstore_bilingual_catalog_v3_1.json"
    cat_path.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    webgpu_bundle = REPO_ROOT / "webgpu_dist" / "qwen0p5b"
    if webgpu_bundle.exists():
        export_scarstore_for_webgpu(
            LIB_DIR / "qwen0p5b_bilingual_v3_1.scarstore",
            webgpu_bundle,
            copy_payload_store=True,
        )
        print(f"✓ Exported 16-cartridge bilingual ScarStore router + payload to {webgpu_bundle}", flush=True)

    print(f"\n✓ Built 16-cartridge Bilingual ScarStore v3.1.0 -> {cat_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
