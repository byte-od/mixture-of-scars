"""Forensic Architectural Integrity & Release Synchronization Suite for `v3.0.0`.

Enforces the Zero-Facades / Zero-Shortcuts engineering mandate across the entire repository:
  1. Exact `3.0.0` version parity across `scar_cache`, `mixture_of_scars`, `pyproject.toml`,
     `CITATION.cff`, `paper_latex/main.tex`, and `huggingface_space/scar_cache/__init__.py`.
  2. Exact `100%` public symbol parity (`__all__` & object identity) between `scar_cache` and `mixture_of_scars`.
  3. Live `--help` smoke verification across all `5` registered CLI entry points (`mos-run`, `mos-store`,
     `mos-graft`, `mos-rebasin`, `mos-transcode-sub2bit`).
  4. Anti-Facade byte-level invariants:
     - `SCacheDynamicCache`: `0 B` growth across `16x` window overflow (`64 -> 512` tokens) and `0 B`
       discrepancy via recursive `untyped_storage().data_ptr()` object-graph walk.
     - `Sub2BitMoSLinear`: genuine `uint8` 4-weights-per-byte (`2-bit`) storage (`packed_base.numel() * 4 == N * K`).
     - `HeadExpandedScarCapsule`: exact `35 x 4,096 B = 143,360 B` (`0 B` internal padding) at `3B` geometry.
     - `SharedGridMoSLinear`: genuine `untyped_storage().data_ptr()` sharing (`0 B` unique-byte discrepancy).
  5. Integrity of all committed JSON audit proofs in `results/` and `benchmark_results/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import mixture_of_scars
import scar_cache
from scar_cache import (
    HeadExpandedScarCapsule,
    SCacheDynamicCache,
    Sub2BitMoSLinear,
    audit_unique_module_bytes,
    build_activation_whitener,
    build_rebasin_shared_swiglu_pair,
)
from scar_cache.cli_72b import main as mos_run_main
from scar_cache.cli_scar_store import main as mos_store_main
from scar_cache.scar_grafting import main_cli as mos_graft_main
from scar_cache.sinkhorn_rebasin import main_cli as mos_rebasin_main
from scar_cache.speculative_ghost_jump import recursive_scache_storage_bytes
from scar_cache.sub2bit_mos import main_cli as mos_sub2bit_main
from scar_cache.webgpu_engine import main_cli as mos_webgpu_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v3_version_and_metadata_synchronization() -> None:
    """Verifies `3.0.0` version parity across packages, metadata files, LaTeX paper, and HF Space."""
    assert scar_cache.__version__ == "3.0.0"
    assert mixture_of_scars.__version__ == "3.0.0"

    pyproject_txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "3.0.0"' in pyproject_txt

    cff_txt = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version: "3.0.0"' in cff_txt

    tex_txt = (REPO_ROOT / "paper_latex" / "main.tex").read_text(encoding="utf-8")
    assert "Revision v3.0" in tex_txt
    assert "v3.0" in tex_txt

    hf_init = (REPO_ROOT / "huggingface_space" / "scar_cache" / "__init__.py").read_text(encoding="utf-8")
    assert '__version__ = "3.0.0"' in hf_init

    # Verify zero broken relative markdown links in README.md, Audit Guide & Archive Index
    for md_path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "AUDIT_AND_VERIFICATION_GUIDE.md",
        REPO_ROOT / "docs" / "ONLINE_CARTRIDGE_LIBRARY.md",
        REPO_ROOT / "archive" / "README.md",
    ):
        assert md_path.exists(), f"Missing documentation file: {md_path}"
        md_no_fences = re.sub(r"```.*?```", "", md_path.read_text(encoding="utf-8"), flags=re.DOTALL)
        for m in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", md_no_fences):
            target = m.group(1).strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "file://")):
                continue
            resolved = (md_path.parent / target).resolve()
            assert resolved.exists(), f"Broken relative link in {md_path.name}: {target} -> {resolved}"


def test_scar_cache_and_mixture_of_scars_100pct_symbol_parity() -> None:
    """Ensures `mixture_of_scars` exports 100% of `scar_cache.__all__` with identical object references."""
    sc_all = set(scar_cache.__all__)
    mos_all = set(mixture_of_scars.__all__)
    assert sc_all == mos_all, f"Symbol mismatch: diff={sc_all ^ mos_all}"
    for sym in sc_all:
        assert getattr(scar_cache, sym) is getattr(mixture_of_scars, sym), f"Object mismatch on {sym}"


def test_all_five_v3_cli_entrypoints_load_and_respond_to_help(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-tests all 5 `pyproject.toml` console scripts via `--help` with zero import or argparse errors."""
    cli_funcs = [
        ("mos-run", mos_run_main),
        ("mos-store", mos_store_main),
        ("mos-graft", mos_graft_main),
        ("mos-rebasin", mos_rebasin_main),
        ("mos-transcode-sub2bit", mos_sub2bit_main),
        ("mos-webgpu", mos_webgpu_main),
    ]
    for name, fn in cli_funcs:
        monkeypatch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as exc_info:
            fn()
        assert exc_info.value.code == 0, f"{name} --help exited with {exc_info.value.code}"
        captured = capsys.readouterr().out
        assert "usage:" in captured.lower(), f"{name} --help produced empty usage output"


def test_anti_facade_no_hidden_kv_growth_or_fake_quantization() -> None:
    """Proves byte-for-byte physical storage honesty across SCache O(1), Sub-2-Bit, ScarStore, and Re-Basin."""
    torch.manual_seed(2026)

    # 1. SCache O(1): Push 512 tokens through a window_size=32 cache (16x overflow) -> 0 B growth & 0 B discrepancy!
    cache = SCacheDynamicCache(window_size=32, num_slots=16, num_sink_tokens=4, project_slots=True)
    bytes_at_64 = None
    for step in range(32):  # 32 chunks x 16 tokens = 512 tokens
        k = torch.randn(1, 4, 16, 32, dtype=torch.float16)
        v = torch.randn(1, 4, 16, 32, dtype=torch.float16)
        for l_idx in range(2):
            cache.update(k, v, l_idx)
        if step == 3:  # 64 tokens (already beyond window_size=32 + num_slots=16)
            bytes_at_64 = cache.total_memory_bytes
            assert recursive_scache_storage_bytes(cache) == bytes_at_64

    bytes_at_512 = cache.total_memory_bytes
    assert bytes_at_64 is not None
    assert bytes_at_512 == bytes_at_64, f"O(1) violated: {bytes_at_64} B at 64t vs {bytes_at_512} B at 512t"
    assert recursive_scache_storage_bytes(cache) == bytes_at_512

    # 2. Sub-2-Bit MoS: Genuine 2-bit uint8 packing (4 weights per byte), never disguised FP16/FP32!
    lin = nn.Linear(128, 256, bias=False)
    act_s = torch.ones(128)
    sub2, _ = Sub2BitMoSLinear.from_linear(lin, act_scales=act_s, rank=16, n_alt_iters=1)
    assert sub2.packed_2bit.dtype == torch.uint8
    assert sub2.packed_2bit.numel() * 4 == 256 * 128

    # 3. HeadExpandedScarCapsule (3B geometry): Exact 35 x 4,096 B = 143,360 B OS mmap alignment
    cap_3b = HeadExpandedScarCapsule(
        capsule_id="audit_3b",
        z_anchor=F.normalize(torch.randn(2048), dim=-1).to(torch.float16),
        A_scar=torch.randn(60, 2048, dtype=torch.float16) * 0.02,
        B_scar=torch.randn(2048, 60, dtype=torch.float16) * 0.02,
        K_heads=F.normalize(torch.randn(8, 16, 128), dim=-1).to(torch.float16),
    )
    raw = cap_3b.to_int4_bytes()
    assert len(raw) == 143360
    assert len(raw) % 4096 == 0 and (len(raw) // 4096) == 35

    # 4. Sinkhorn SharedGridMoSLinear: Exact shared storage pointer (`0 B` unique discrepancy)
    class _MiniMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(64, 128, bias=False)
            self.up_proj = nn.Linear(64, 128, bias=False)
            self.down_proj = nn.Linear(128, 64, bias=False)

    m_a, m_b = _MiniMLP().eval(), _MiniMLP().eval()
    wh_in = build_activation_whitener(torch.randn(32, 64))
    wh_mid = build_activation_whitener(torch.randn(32, 128))
    rb_a, rb_b, _ = build_rebasin_shared_swiglu_pair(m_a, m_b, wh_in, wh_in, wh_mid, wh_mid, rank=8)
    holder = nn.ModuleDict({"a": nn.ModuleDict(rb_a), "b": nn.ModuleDict(rb_b)})
    reported = sum(m.physical_bytes for m in rb_a.values()) + sum(m.physical_bytes for m in rb_b.values())
    assert reported == audit_unique_module_bytes(holder)
    assert rb_a["gate_proj"].packed_shared.untyped_storage().data_ptr() == rb_b["gate_proj"].packed_shared.untyped_storage().data_ptr()


def test_published_json_audit_reports_exist_and_verify_zero_discrepancy() -> None:
    """Audits every committed JSON empirical report in `results/` and `benchmark_results/`."""
    required_reports = [
        REPO_ROOT / "benchmark_results" / "adversarial_mos_scache_report.json",
        REPO_ROOT / "results" / "passkey_16k_32k_llm_audit.json",
        REPO_ROOT / "results" / "horizon1_v1_4_audit.json",
        REPO_ROOT / "results" / "horizon2_scar_grafting_v2_audit.json",
        REPO_ROOT / "results" / "scar_store_v2_stage5_3b_vs_0p5b_audit.json",
        REPO_ROOT / "results" / "adversarial_0p5b_vs_3b_audit.json",
        REPO_ROOT / "results" / "mos_store_v2_3_cli_http206_audit.json",
        REPO_ROOT / "results" / "online_cartridge_library_v3_0_audit.json",
        REPO_ROOT / "results" / "online_library_v3_0" / "scarstore_catalog_v3_0.json",
        REPO_ROOT / "results" / "webgpu_wgsl_v3_0_audit.json",
        REPO_ROOT / "results" / "sinkhorn_rebasin_v2_4_audit.json",
        REPO_ROOT / "results" / "scale_7b_audit" / "adversarial_scale_7b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "adversarial_scale_72b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_72b_speed_memory_audit.json",
    ]
    for rp in required_reports:
        assert rp.exists(), f"Missing committed empirical audit report: {rp}"
        payload = json.loads(rp.read_text(encoding="utf-8"))
        assert isinstance(payload, (dict, list)) and len(payload) > 0

        # Recursively check any field ending with `discrepancy_bytes` or `diff_bytes` is strictly 0
        def _walk_zero_discrepancy(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if re.search(r"(discrepancy_bytes|reported_vs_unique_diff_bytes|discrepancy_kb)$", k):
                        assert float(v) == 0.0, f"Non-zero storage discrepancy in {rp.name}:{path}.{k} = {v}"
                    _walk_zero_discrepancy(v, f"{path}.{k}")
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    _walk_zero_discrepancy(item, f"{path}[{idx}]")

        _walk_zero_discrepancy(payload)
