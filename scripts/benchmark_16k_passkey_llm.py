#!/usr/bin/env python3
"""
Official 16k–32k+ Token Passkey / Needle-in-a-Haystack Benchmark on Real Pretrained LLMs.

Compares side-by-side on 100% non-repeating technical prose + source documentation:
  1. Normal KV Cache O(T)  — HuggingFace `DynamicCache`
  2. StreamingLLM O(1)     — `SCacheDynamicCache(window_size=256, num_slots=0, num_sink_tokens=4)`
  3. Scar-Cache O(1)       — `SCacheDynamicCache(window_size=256, num_slots=128, num_sink_tokens=4)`

Audits:
  - Exact 6-digit passkey retrieval across needle depths (5%, 25%, 50%, 75%, 95%)
  - End-to-end throughput (`tok/s`) & wall-clock latency (`s`)
  - Exact physical KV memory (`MiB`) + recursive `untyped_storage().data_ptr()` audit proving
    0.00 Bytes of growth from 16k to 32k+ tokens and 0.00 Bytes of hidden tensor discrepancy.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent
import datetime
import decimal
import email
import functools
import glob
import http
import importlib
import inspect
import json
import logging
import os
import pathlib
import re
import sqlite3
import ssl
import subprocess
import sys
import tarfile
import threading
import time
import unittest
import urllib
import xml
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scar_cache.hf_cache import SCacheDynamicCache


def recursive_tensor_storage_bytes(obj: Any, seen: set[int] | None = None) -> int:
    """Walks all Python attributes/containers and sums unique underlying `untyped_storage` bytes."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, torch.Tensor):
        st = obj.untyped_storage()
        sid = int(st.data_ptr())
        if sid in seen:
            return 0
        seen.add(sid)
        return int(st.nbytes())
    if isinstance(obj, dict):
        return sum(recursive_tensor_storage_bytes(k, seen) + recursive_tensor_storage_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(recursive_tensor_storage_bytes(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        return recursive_tensor_storage_bytes(vars(obj), seen)
    return 0


_BOILERPLATE_MARKERS = (
    "IMPLIED WARRANTIES",
    "CONSEQUENTIAL DAMAGES",
    "LIABILITY, WHETHER IN CONTRACT",
    "COPYRIGHT HOLDERS",
    "REDISTRIBUTION AND USE",
    "SPDX-LICENSE",
)


def build_non_repeating_stdlib_corpus(min_chars: int = 280_000) -> str:
    """Builds a deterministic, 100% non-repeating technical prose haystack from Python's standard library docstrings."""
    priority_mods = [
        asyncio, collections, concurrent, datetime, decimal, email, functools,
        http, importlib, json, logging, pathlib, sqlite3, ssl, subprocess,
        tarfile, threading, unittest, urllib, xml, zipfile,
    ]
    seen_blocks: set[str] = set()
    seen_files: set[str] = set()
    blocks: List[str] = []

    def _harvest_py_file(py_path: Path) -> None:
        rp = str(py_path.resolve())
        if rp in seen_files or not py_path.exists():
            return
        seen_files.add(rp)
        src = py_path.read_text(encoding="utf-8", errors="ignore")
        for doc in re.findall(r'"""(.*?)"""', src, flags=re.S):
            clean = " ".join(doc.split())
            if len(clean) <= 120 or clean in seen_blocks:
                continue
            up = clean.upper()
            if any(m in up for m in _BOILERPLATE_MARKERS):
                continue
            seen_blocks.add(clean)
            blocks.append(clean)

    for mod in priority_mods:
        src_path = inspect.getsourcefile(mod)
        if src_path:
            _harvest_py_file(Path(src_path))

    # Deterministically traverse the rest of Python's standard library root (excluding test suites / site-packages)
    stdlib_root = Path(os.__file__).resolve().parent
    for py_file in sorted(stdlib_root.rglob("*.py")):
        if sum(len(b) for b in blocks) >= min_chars:
            break
        parts = set(py_file.relative_to(stdlib_root).parts)
        if parts & {"test", "tests", "site-packages", "idlelib", "turtledemo", "tkinter", "__pycache__"}:
            continue
        _harvest_py_file(py_file)

    return "\n\n".join(blocks)


def insert_needle_at_sentence_boundary(
    tok: Any, full_ids: List[int], target_hay_tokens: int, depth: float, needle: str, question: str
) -> List[int]:
    """Slices non-repeating haystack to `target_hay_tokens` and inserts `needle` at the nearest sentence break."""
    if len(full_ids) < target_hay_tokens:
        raise RuntimeError(f"Non-repeating stdlib corpus has {len(full_ids)} tokens < requested {target_hay_tokens}")
    hay_text = tok.decode(full_ids[:target_hay_tokens], skip_special_tokens=True)

    raw_pos = int(depth * len(hay_text))
    lo = max(64, raw_pos - 300)
    hi = min(len(hay_text) - 64, raw_pos + 300)
    window = hay_text[lo:hi]

    # Prefer paragraph break `\n\n`, then sentence end `. `, else fallback to whitespace
    best_pos = raw_pos
    para_matches = [lo + m.end() for m in re.finditer(r"\n\n", window)]
    sent_matches = [lo + m.end() for m in re.finditer(r"\.\s+", window)]
    space_matches = [lo + m.end() for m in re.finditer(r"\s+", window)]
    for candidates in (para_matches, sent_matches, space_matches):
        if candidates:
            best_pos = min(candidates, key=lambda idx: abs(idx - raw_pos))
            break

    composed = hay_text[:best_pos].rstrip() + needle + hay_text[best_pos:].lstrip() + question
    return tok(composed, add_special_tokens=False).input_ids


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _empty_device_cache(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


@torch.no_grad()
def run_single_passkey_trial(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    passkey: str,
    mode: str,
    window_size: int = 256,
    num_slots: int = 128,
    num_sinks: int = 4,
    max_new_tokens: int = 8,
) -> Dict[str, Any]:
    dev = input_ids.device
    seq_len = int(input_ids.shape[1])

    if mode == "normal_kv":
        cache: Any = DynamicCache()
        chunk_sz = 512
    elif mode == "streaming_llm":
        cache = SCacheDynamicCache(
            window_size=window_size, num_slots=0, num_sink_tokens=num_sinks, project_slots=True
        )
        chunk_sz = cache.prefill_chunk_size
    elif mode == "scache_o1":
        cache = SCacheDynamicCache(
            window_size=window_size, num_slots=num_slots, num_sink_tokens=num_sinks, project_slots=True
        )
        chunk_sz = cache.prefill_chunk_size
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _sync_device(dev)
    t0 = time.perf_counter()
    out = None
    for s0 in range(0, seq_len, chunk_sz):
        out = model(input_ids[:, s0 : s0 + chunk_sz], past_key_values=cache, use_cache=True)

    assert out is not None
    gen_ids: List[int] = []
    nxt = int(out.logits[0, -1].argmax().item())
    for _ in range(max_new_tokens):
        gen_ids.append(nxt)
        out = model(torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax().item())

    _sync_device(dev)
    elapsed_s = time.perf_counter() - t0
    decoded = tokenizer.decode(gen_ids)
    hit = passkey in decoded

    if isinstance(cache, SCacheDynamicCache):
        reported_bytes = int(cache.total_memory_bytes)
        recursive_bytes = int(recursive_tensor_storage_bytes(cache))
        discrepancy_bytes = abs(recursive_bytes - reported_bytes)
        assert discrepancy_bytes == 0, f"Anti-facade violation: recursive={recursive_bytes} != reported={reported_bytes}"
    else:
        reported_bytes = sum(int(l.keys.nbytes + l.values.nbytes) for l in cache.layers if l.keys is not None)
        recursive_bytes = int(recursive_tensor_storage_bytes(cache))
        discrepancy_bytes = abs(recursive_bytes - reported_bytes)

    res = {
        "mode": mode,
        "context_tokens": seq_len,
        "elapsed_s": round(elapsed_s, 3),
        "tok_per_s": round(seq_len / max(elapsed_s, 1e-6), 1),
        "kv_mib": round(reported_bytes / (1024**2), 3),
        "kv_bytes": reported_bytes,
        "recursive_storage_bytes": recursive_bytes,
        "storage_discrepancy_bytes": discrepancy_bytes,
        "hit": bool(hit),
        "generated": decoded,
    }
    del cache
    _empty_device_cache(dev)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="16k–32k+ Passkey / NIAH Real-LLM Benchmark (SCache O(1) vs Normal KV)")
    p.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
        help="HuggingFace causal LM IDs to evaluate",
    )
    p.add_argument("--lengths", nargs="+", type=int, default=[16500, 32800], help="Haystack token lengths (>=16k)")
    p.add_argument("--depths", nargs="+", type=float, default=[0.05, 0.25, 0.50, 0.75, 0.95], help="Needle depths")
    p.add_argument("--passkey", type=str, default="739184", help="Exact passkey digits")
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--num-slots", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "results" / "passkey_16k_32k_llm_audit.json"),
        help="Output JSON telemetry path",
    )
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    corpus_text = build_non_repeating_stdlib_corpus(min_chars=max(args.lengths) * 9)
    needle = f"\n\n[CRITICAL SYSTEM RECORD: The emergency vault passkey is {args.passkey}. Remember passkey {args.passkey}.]\n\n"
    question = (
        "\n\nBased on the critical system record in the document above, what is the 6-digit emergency vault passkey? "
        "Reply with the exact digits only:\nThe emergency vault passkey is "
    )

    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    summary: Dict[str, Any] = {
        "device": str(dev),
        "dtype": str(dtype),
        "window_size": args.window_size,
        "num_slots": args.num_slots,
        "passkey": args.passkey,
        "models": {},
    }

    for model_id in args.models:
        print(f"\n{'=' * 88}\nEvaluating Real Pretrained LLM: {model_id} on {dev} ({dtype})\n{'=' * 88}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        full_corpus_ids = tok(corpus_text, add_special_tokens=False).input_ids
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=local_only).to(dev).eval()
        model_res: Dict[str, Any] = {}

        for hay_len in args.lengths:
            len_key = f"haystack_{hay_len}"
            len_entry: Dict[str, Any] = {"baselines_at_25pct": {}, "scache_depths": []}

            # Baselines at 25% depth (clean sentence boundary well outside local window W=256)
            ids25_list = insert_needle_at_sentence_boundary(tok, full_corpus_ids, hay_len, 0.25, needle, question)
            ids25 = torch.tensor([ids25_list], device=dev)
            for bmode in ("normal_kv", "streaming_llm"):
                r_b = run_single_passkey_trial(
                    model, tok, ids25, args.passkey, mode=bmode, window_size=args.window_size, num_slots=args.num_slots
                )
                len_entry["baselines_at_25pct"][bmode] = r_b
                print(
                    f"  [{bmode:13s}] @ 25% (L={r_b['context_tokens']:5d}) | "
                    f"{r_b['elapsed_s']:6.2f}s ({r_b['tok_per_s']:6.1f} tok/s) | "
                    f"KV={r_b['kv_mib']:7.2f} MiB | hit={str(r_b['hit']):5s} | out={r_b['generated']!r}",
                    flush=True,
                )

            # For >=30k on multi-billion models on laptop MPS, evaluate representative depths [5%, 25%, 75%]
            eval_depths = [0.05, 0.25, 0.75] if (hay_len > 24000 and "0.5B" not in model_id) else args.depths
            for depth in eval_depths:
                ids_d_list = insert_needle_at_sentence_boundary(tok, full_corpus_ids, hay_len, depth, needle, question)
                ids_d = torch.tensor([ids_d_list], device=dev)
                r_sc = run_single_passkey_trial(
                    model, tok, ids_d, args.passkey, mode="scache_o1", window_size=args.window_size, num_slots=args.num_slots
                )
                r_sc["depth"] = depth
                len_entry["scache_depths"].append(r_sc)
                print(
                    f"  [scache_o1    ] @ {int(depth*100):2d}% (L={r_sc['context_tokens']:5d}) | "
                    f"{r_sc['elapsed_s']:6.2f}s ({r_sc['tok_per_s']:6.1f} tok/s) | "
                    f"KV={r_sc['kv_mib']:7.2f} MiB (Δrec={r_sc['storage_discrepancy_bytes']} B) | "
                    f"hit={str(r_sc['hit']):5s} | out={r_sc['generated']!r}",
                    flush=True,
                )

            hits = sum(1 for d in len_entry["scache_depths"] if d["hit"])
            len_entry["scache_passkey_recall_pct"] = round(100.0 * hits / max(len(len_entry["scache_depths"]), 1), 2)
            len_entry["kv_reduction_vs_normal_x"] = round(
                len_entry["baselines_at_25pct"]["normal_kv"]["kv_bytes"]
                / max(len_entry["scache_depths"][0]["kv_bytes"], 1),
                2,
            )
            len_entry["speedup_vs_normal_x"] = round(
                len_entry["baselines_at_25pct"]["normal_kv"]["elapsed_s"]
                / max(len_entry["scache_depths"][0]["elapsed_s"], 1e-6),
                2,
            )
            model_res[len_key] = len_entry

        # Verify strict O(1) 0-byte growth across all evaluated context lengths!
        if len(args.lengths) >= 2:
            b_list = [model_res[f"haystack_{L}"]["scache_depths"][0]["kv_bytes"] for L in args.lengths]
            delta_b = max(b_list) - min(b_list)
            model_res["scache_byte_growth_across_lengths"] = delta_b
            print(f"  ✓ Strict O(1) Audit across lengths {args.lengths}: ΔBytes = {delta_b} B!", flush=True)
            assert delta_b == 0, f"Expected 0 B growth across lengths, got {b_list}"

        summary["models"][model_id] = model_res
        del model
        _empty_device_cache(dev)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Saved audited 16k–32k+ Passkey JSON report to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
