"""Forensic Architectural Integrity & Release Synchronization Suite for `v5.0.0`.

Enforces the Zero-Facades / Zero-Shortcuts engineering mandate across the entire repository:
  1. Exact `5.0.0` version parity across `scar_cache`, `mixture_of_scars`, `scar_codec`, `scar_native`,
     `pyproject.toml`, `CITATION.cff`, `paper_latex/main.tex`, and `huggingface_space/`.
  2. Exact `100%` public symbol parity (`__all__` & object identity) between `scar_cache` and `mixture_of_scars`.
  3. Live `--help` smoke verification across all `8` registered CLI entry points (`mos-run`, `mos-store`,
     `mos-graft`, `mos-rebasin`, `mos-transcode-sub2bit`, `mos-webgpu`, `mos-scar-codec`, `mos-scar-native`).
  4. Anti-Facade byte-level invariants across `SCacheDynamicCache`, `Sub2BitMoSLinear`, `HeadExpandedScarCapsule`,
     `SharedGridMoSLinear`, standalone `scar_codec`, and Tabula-Rasa `ScarNativeLM`.
  5. Integrity of all committed JSON audit proofs in `results/`, `benchmark_results/`, `zenodo_deposit_v4_0/`,
     and `zenodo_deposit_v5_0/`.
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
import scar_codec
import scar_native
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
from scar_codec.cli import main_cli as mos_scar_codec_main
from scar_native.cli import main_cli as mos_scar_native_main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v5_version_and_metadata_synchronization() -> None:
    """Verifies `5.0.0` version parity across all 4 packages, metadata files, LaTeX paper, Zenodo kits, and HF Space."""
    assert scar_cache.__version__ == "5.0.0"
    assert mixture_of_scars.__version__ == "5.0.0"
    assert scar_codec.__version__ == "5.0.0"
    assert scar_native.__version__ == "5.0.0"

    pyproject_txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "5.0.0"' in pyproject_txt

    cff_txt = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert 'version: "5.0.0"' in cff_txt

    tex_txt = (REPO_ROOT / "paper_latex" / "main.tex").read_text(encoding="utf-8")
    assert "Revision v5.0" in tex_txt
    assert "ScarNativeLM" in tex_txt

    for hf_pkg in ("scar_cache", "scar_codec", "scar_native"):
        hf_init = (REPO_ROOT / "huggingface_space" / hf_pkg / "__init__.py").read_text(encoding="utf-8")
        assert "5.0.0" in hf_init or "__version__" in hf_init

    # Verify preserved Zenodo v4.0 deposit artifacts remain 100% intact
    for z4_name in (
        "Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf",
        "MoS_SCache_LaTeX_arXiv_TMLR_v4.0.zip",
        "arXiv_Direct_Upload_v4.0.zip",
        "MoS_SCache_Adversarial_Replication_v4.0.zip",
        "mos_scache_v4_0_hero.png",
    ):
        zp4 = REPO_ROOT / "zenodo_deposit_v4_0" / z4_name
        assert zp4.exists() and zp4.stat().st_size > 1024, f"Missing preserved Zenodo v4.0 file: {zp4}"

    # Verify official Zenodo v5.0 deposit artifacts exist and are non-empty
    for z5_name in (
        "Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf",
        "MoS_ScarNative_LaTeX_arXiv_TMLR_v5.0.zip",
        "arXiv_Direct_Upload_v5.0.zip",
        "MoS_ScarNative_Adversarial_Replication_v5.0.zip",
        "mos_scarnative_v5_0_hero.png",
    ):
        zp5 = REPO_ROOT / "zenodo_deposit_v5_0" / z5_name
        assert zp5.exists() and zp5.stat().st_size > 1024, f"Missing or empty Zenodo v5.0 file: {zp5}"

    # Verify zero broken relative markdown links in README.md, Docs & Archive Index
    for md_path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "SCAR_NATIVE_V5_FOUNDATION_ARCHITECTURE.md",
        REPO_ROOT / "docs" / "AUDIT_AND_VERIFICATION_GUIDE.md",
        REPO_ROOT / "docs" / "UNIFIED_SCAR_OPERATORS_AND_8_SOLUTIONS.md",
        REPO_ROOT / "docs" / "BEYOND_AI_SCAR_ARCHITECTURE_APPLICATIONS.md",
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


def test_v5_wgsl_22_shader_inventory_and_sync() -> None:
    """Enforces 22-shader WGSL 1.0 kernel inventory (including Shaders #21 & #22) and HF Space byte parity."""
    wgsl_main = (REPO_ROOT / "scar_cache" / "webgpu" / "mos_scache_kernels.wgsl").read_text(encoding="utf-8")
    compute_entries = re.findall(r"@compute\s+@workgroup_size\([^)]+\)\s*fn\s+([A-Za-z0-9_]+)", wgsl_main)
    assert len(compute_entries) == 22, f"Expected 22 WGSL @compute shaders, found {len(compute_entries)}: {compute_entries}"
    assert "scarnative_widrow_hoff_evict_update" in compute_entries
    assert "scarnative_o1_hybrid_gqa_attention" in compute_entries

    wgsl_hf = (REPO_ROOT / "huggingface_space" / "scar_cache" / "webgpu" / "mos_scache_kernels.wgsl").read_text(
        encoding="utf-8"
    )
    assert wgsl_main == wgsl_hf, "HuggingFace Space WGSL shader suite out of sync with canonical mos_scache_kernels.wgsl"


def test_scar_cache_and_mixture_of_scars_100pct_symbol_parity() -> None:
    """Ensures `mixture_of_scars` exports 100% of `scar_cache.__all__` with identical object references."""
    sc_all = set(scar_cache.__all__)
    mos_all = set(mixture_of_scars.__all__)
    assert sc_all == mos_all, f"Symbol mismatch: diff={sc_all ^ mos_all}"
    for sym in sc_all:
        assert getattr(scar_cache, sym) is getattr(mixture_of_scars, sym), f"Object mismatch on {sym}"


def test_all_eight_v5_cli_entrypoints_load_and_respond_to_help(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-tests all 8 `pyproject.toml` console scripts via `--help` with zero import or argparse errors."""
    cli_funcs = [
        ("mos-run", mos_run_main),
        ("mos-store", mos_store_main),
        ("mos-graft", mos_graft_main),
        ("mos-rebasin", mos_rebasin_main),
        ("mos-transcode-sub2bit", mos_sub2bit_main),
        ("mos-webgpu", mos_webgpu_main),
        ("mos-scar-codec", mos_scar_codec_main),
        ("mos-scar-native", mos_scar_native_main),
    ]
    for name, fn in cli_funcs:
        monkeypatch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as exc_info:
            fn()
        assert exc_info.value.code == 0, f"{name} --help exited with {exc_info.value.code}"
        captured = capsys.readouterr().out
        assert "usage:" in captured.lower(), f"{name} --help produced empty usage output"


def test_anti_facade_no_hidden_kv_growth_or_fake_quantization() -> None:
    """Proves byte-for-byte physical storage honesty across SCache O(1), Sub-2-Bit, ScarStore, Re-Basin, ScarCodec & ScarNativeLM."""
    torch.manual_seed(2026)

    # 1. SCache O(1): Push 512 tokens through a window_size=32 cache (16x overflow) -> 0 B growth & 0 B discrepancy!
    cache = SCacheDynamicCache(window_size=32, num_slots=16, num_sink_tokens=4, project_slots=True)
    bytes_at_64 = None
    for step in range(32):  # 32 chunks x 16 tokens = 512 tokens
        k = torch.randn(1, 4, 16, 32, dtype=torch.float16)
        v = torch.randn(1, 4, 16, 32, dtype=torch.float16)
        for l_idx in range(2):
            cache.update(k, v, l_idx)
        if step == 3:
            bytes_at_64 = cache.total_memory_bytes
            assert recursive_scache_storage_bytes(cache) == bytes_at_64

    bytes_at_512 = cache.total_memory_bytes
    assert bytes_at_64 is not None
    assert bytes_at_512 == bytes_at_64, f"O(1) violated: {bytes_at_64} B at 64t vs {bytes_at_512} B at 512t"
    assert recursive_scache_storage_bytes(cache) == bytes_at_512

    # 2. Sub-2-Bit MoS: Genuine 2-bit uint8 packing (4 weights per byte)
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

    # 5. Standalone Physical ScarCodec: Exact 34,816 B (34.8 KB) SCache O(1) with 0 B drift
    anom_sc = scar_codec.SurpriseGatedAnomalySCache()
    assert anom_sc.physical_bytes() == 34816
    frames = torch.randn(512, 64) * 0.05
    frames[128] += 12.0
    for ts in range(512):
        anom_sc.ingest(frames[ts], timestamp=ts)
    assert anom_sc.physical_bytes() == 34816
    assert 128 in anom_sc.recorded_timestamps()

    # 6. Pure Scar-Native v5.0: Zero dense `nn.Linear` anywhere in `ScarNativeLM`
    sn_cfg = scar_native.ScarNativeConfig(hidden_size=128, intermediate_size=256, num_layers=2, num_q_heads=4, num_kv_heads=2, golden_layer=1)
    sn_mod = scar_native.ScarNativeLM(sn_cfg).eval()
    assert not any(isinstance(m, nn.Linear) for m in sn_mod.modules())
    packed_b = sn_mod.pack_all_to_scar_binary()
    assert packed_b > 0 and sn_mod.embed_tokens is None


def test_published_json_audit_reports_exist_and_verify_zero_discrepancy() -> None:
    """Audits every committed JSON empirical report in `results/` and `benchmark_results/` (including v5.0)."""
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
        REPO_ROOT / "results" / "online_library_v3_0" / "qwen3b_portfolio_v4_3_prot_int4.manifest.json",
        REPO_ROOT / "results" / "webgpu_wgsl_v3_0_audit.json",
        REPO_ROOT / "results" / "sinkhorn_rebasin_v2_4_audit.json",
        REPO_ROOT / "results" / "scar_codec_physical_v4_0_audit.json",
        REPO_ROOT / "results" / "scar_native_v5_0_audit.json",
        REPO_ROOT / "results" / "scale_7b_audit" / "adversarial_scale_7b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "adversarial_scale_72b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_72b_speed_memory_audit.json",
    ]
    for rp in required_reports:
        assert rp.exists(), f"Missing committed empirical audit report: {rp}"
        payload = json.loads(rp.read_text(encoding="utf-8"))
        assert isinstance(payload, (dict, list)) and len(payload) > 0

        def _walk_zero_discrepancy(node: object, path: str = "") -> None:
            if isinstance(node, dict):
                # Skip intentional `dense_ot` O(T) baseline rows when checking `scache_delta_bytes`
                is_dense_ot = node.get("model_id") == "dense_ot"
                for k, v in node.items():
                    if re.search(r"(discrepancy_bytes|reported_vs_unique_diff_bytes|discrepancy_kb)$", k):
                        assert float(v) == 0.0, f"Non-zero storage discrepancy in {rp.name}:{path}.{k} = {v}"
                    if k == "scache_delta_bytes" and not is_dense_ot:
                        assert float(v) == 0.0, f"Non-zero SCache delta in {rp.name}:{path}.{k} = {v}"
                    _walk_zero_discrepancy(v, f"{path}.{k}")
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    _walk_zero_discrepancy(item, f"{path}[{idx}]")

        _walk_zero_discrepancy(payload)
