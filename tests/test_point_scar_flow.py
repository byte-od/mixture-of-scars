"""Automated Verification Suite for `PointScarFlow` v1.0 (`ScarNative` Continuous 3D Point-Cloud Generator).

Verifies:
  1. Zero `torch.nn.Linear` dense modules, `LUT256[0] == (0,0,0,0)`, deterministic Fibonacci sphere `X_0`,
     and machine-epsilon (`< 1e-5`) PTQ shock upon `pack_all_vq4_scar(drop_master=True)`.
  2. Physical Apple M2 WebGPU (`PointScarFlowWebGPUEngine`) vs PyTorch (`PointScarFlow` & `load_pointscar_from_webgpu_bundle`)
     bit-level parity (`cosine > 0.99999`, `max_abs_err < 5e-5`) for both `guidance_scale = 1.0` and `1.75`.
  3. Byte-for-byte identity across all Native and Portfolio copies of `point_scar_flow.wgsl` (`12` `@compute` kernels).
  4. Trained Colab `A100` `pointscar_v1` bundle (`3.87 MiB`) evaluation metrics, cross-lingual (`PT, EN, ES, FR`)
     3D manifold alignment, inter-concept geometric distinctness, and physical WebGPU parity (`cosine > 0.99999`).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_native.model import ScarNativeLinear
from scar_native.point_scar_flow import (
    POINT_SCAR_WGSL_PATH,
    PointScarFlow,
    PointScarFlowConfig,
    PointScarFlowWebGPUEngine,
    build_canonical_fibonacci_sphere,
    export_pointscar_webgpu_bundle,
    load_pointscar_from_webgpu_bundle,
)

SC_ROOT = Path(__file__).resolve().parent.parent
PF_ROOT = SC_ROOT.parent / "portfolio"
TRAINED_BUNDLE_DIR = SC_ROOT / "artifacts/pointscar_v1"


def test_point_scar_flow_zero_dense_linear_and_zero_ptq_shock() -> None:
    torch.manual_seed(20261115)
    cfg = PointScarFlowConfig(
        cond_in_dim=2048,
        cond_dim=256,
        d_model=128,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        num_parts=8,
        num_points=256,
        scar_rank=8,
        cond_scar_rank=16,
        head_scar_rank=8,
    )
    model = PointScarFlow(cfg).eval()
    model.assert_zero_dense_linear()

    scar_mods = [m for m in model.modules() if isinstance(m, ScarNativeLinear)]
    assert len(scar_mods) > 15
    for m in scar_mods:
        assert not isinstance(m, nn.Linear)
        assert float(m.lut256_f16[0].abs().max().item()) == 0.0

    # Deterministic canonical Fibonacci sphere (`0` randomness)
    s1 = build_canonical_fibonacci_sphere(256)
    s2 = build_canonical_fibonacci_sphere(256)
    assert torch.equal(s1, s2)
    radii = s1.norm(dim=-1)
    assert float((radii - 1.0).abs().max().item()) < 1e-5

    z = F.normalize(torch.randn(2, 2048), dim=-1)
    y_pre = model.sample_euler(z, num_steps=4, guidance_scale=1.0)
    phys_bytes = model.pack_all_vq4_scar(drop_master=True)
    assert phys_bytes > 0
    for m in scar_mods:
        assert m.w_master is None and m.packed is not None

    y_post = model.sample_euler(z, num_steps=4, guidance_scale=1.0)
    ptq_shock = float((y_pre - y_post).abs().max().item())
    assert ptq_shock < 1e-5, f"Expected near-zero PTQ shock (< 1e-5), got {ptq_shock}"


def test_point_scar_flow_physical_webgpu_parity(tmp_path: Path) -> None:
    torch.manual_seed(424242)
    cfg = PointScarFlowConfig(
        cond_in_dim=2048,
        cond_dim=256,
        d_model=128,
        d_ff=256,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        num_parts=8,
        num_points=256,
        scar_rank=8,
        cond_scar_rank=16,
        head_scar_rank=8,
    )
    model = PointScarFlow(cfg).eval()
    # Give velocity head & AdaLN realistic trained magnitudes so non-trivial 3D trajectories are tested!
    with torch.no_grad():
        model.vel_head.w_master.mul_(8.0)
        for blk in model.blocks:
            blk.ada_mod.w_master.mul_(6.0)

    export_pointscar_webgpu_bundle(model, tmp_path)
    reloaded = load_pointscar_from_webgpu_bundle(tmp_path)
    wgpu_eng = PointScarFlowWebGPUEngine(tmp_path)

    z = F.normalize(torch.randn(2048), dim=-1)
    for g_scale in (1.0, 1.75):
        pt_cloud = model.sample_euler(z, num_steps=4, guidance_scale=g_scale).numpy()
        re_cloud = reloaded.sample_euler(z, num_steps=4, guidance_scale=g_scale).numpy()
        gpu_cloud = wgpu_eng.sample_euler(z.numpy(), num_steps=4, guidance_scale=g_scale)

        assert np.max(np.abs(pt_cloud - re_cloud)) < 1e-5
        cos = float(
            np.dot(pt_cloud.reshape(-1), gpu_cloud.reshape(-1))
            / (np.linalg.norm(pt_cloud) * np.linalg.norm(gpu_cloud) + 1e-12)
        )
        max_err = float(np.max(np.abs(pt_cloud - gpu_cloud)))
        assert cos > 0.99999, f"WebGPU parity cosine too low at CFG={g_scale}: {cos:.8f}"
        assert max_err < 5e-5, f"WebGPU max_abs_err too high at CFG={g_scale}: {max_err:.6e}"


def test_point_scar_flow_wgsl_byte_identical_mirror() -> None:
    wgsl_canon = POINT_SCAR_WGSL_PATH.read_text(encoding="utf-8")
    pf_src = (PF_ROOT / "src/lib/shaders/point_scar_flow.wgsl").read_text(encoding="utf-8")
    pf_pub = (PF_ROOT / "public/webgpu/pointscar_v1/point_scar_flow.wgsl").read_text(encoding="utf-8")

    assert wgsl_canon == pf_src, "portfolio/src/lib/shaders/point_scar_flow.wgsl != canonical point_scar_flow.wgsl"
    assert wgsl_canon == pf_pub, "portfolio/public/webgpu/pointscar_v1/point_scar_flow.wgsl != canonical point_scar_flow.wgsl"

    entrypoints = re.findall(r"@compute\s+@workgroup_size\([^)]+\)\s*fn\s+([a-zA-Z0-9_]+)", wgsl_canon)
    expected = [
        "psf_build_point_features_f32",
        "psf_time_embed_64_f32",
        "psf_vq4_batched_equilibrate_fwht64",
        "psf_batched_int4_ascar_gemm",
        "psf_batched_vq4_int4scar_gemm",
        "psf_rmsnorm_adaln_f32",
        "psf_part_from_point_attn_hd32",
        "psf_point_from_part_attn_hd32",
        "psf_add_silu_f32",
        "psf_swiglu_mul_f32",
        "psf_gated_residual_add_f32",
        "psf_euler_step_cfg_f32",
    ]
    assert entrypoints == expected, f"Expected 12 entrypoints {expected}, got {entrypoints}"


def test_trained_pointscar_v1_bundle_polyglot_and_webgpu_parity() -> None:
    assert (TRAINED_BUNDLE_DIR / "pointscar_manifest.json").exists(), (
        f"Missing trained Colab A100 bundle at {TRAINED_BUNDLE_DIR}"
    )
    report = json.loads((TRAINED_BUNDLE_DIR / "eval_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((TRAINED_BUNDLE_DIR / "pointscar_manifest.json").read_text(encoding="utf-8"))
    shard_path = TRAINED_BUNDLE_DIR / manifest["shard_file"]
    assert shard_path.stat().st_size == int(manifest["shard_bytes"])
    assert 3_000_000 <= shard_path.stat().st_size <= 5_000_000

    # Portfolio mirror must hold exact identical binary shard and manifest!
    pf_bundle = PF_ROOT / "public/webgpu/pointscar_v1"
    assert (pf_bundle / manifest["shard_file"]).read_bytes() == shard_path.read_bytes()

    assert float(report["router_anchor0_alignment_cosine"]) > 0.70
    # Note: on NVIDIA A100 with TF32 Tensor Cores enabled, `_fast_fwht64` (`@ h64_t` in TF32) vs
    # exact FP32 butterfly `fwht_block64_orthonormal` across 4 ODE steps differs by ~1.15e-3 (`< 2e-3`).
    assert float(report["ptq_max_abs_shock"]) < 2e-3
    chamfer_4 = float(report["held_out_chamfer_l2_4step_1pass"])
    chamfer_sph = float(report["held_out_chamfer_l2_sphere_baseline"])
    assert chamfer_4 < chamfer_sph * 0.75, (
        f"Expected strong 3D deformation below sphere baseline: {chamfer_4} vs {chamfer_sph}"
    )

    xling = report["cross_lingual_3d_shape_cosine"]
    inter_c = float(report["inter_concept_3d_cosine"])
    for key in ("pt_en", "es_en", "fr_en"):
        assert float(xling[key]) > inter_c + 0.25, (
            f"Expected cross-lingual 3D cosine ({key}={xling[key]}) >> inter-concept ({inter_c})"
        )

    # Run physical Apple M2 WebGPU vs PyTorch across all 16 held-out polyglot test latents (`4 concepts x PT/EN/ES/FR`)!
    poly_latents = json.loads((TRAINED_BUNDLE_DIR / "polyglot_test_latents.json").read_text(encoding="utf-8"))
    pt_model = load_pointscar_from_webgpu_bundle(TRAINED_BUNDLE_DIR)
    wgpu_eng = PointScarFlowWebGPUEngine(TRAINED_BUNDLE_DIR)
    x0_xyz = build_canonical_fibonacci_sphere(pt_model.cfg.num_points).numpy()

    def _chamfer_l2_np(a_xyz: np.ndarray, b_xyz: np.ndarray) -> float:
        pa = a_xyz.reshape(-1, 3)
        pb = b_xyz.reshape(-1, 3)
        diff = pa[:, None, :] - pb[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        return float(0.5 * (np.mean(np.min(dist, axis=1)) + np.mean(np.min(dist, axis=0))))

    en_clouds_by_concept: dict[str, np.ndarray] = {}
    min_intra_xling_cos: dict[str, float] = {}
    max_intra_xling_chamfer: dict[str, float] = {}
    for concept_id, by_lang in poly_latents.items():
        lang_xyz: dict[str, np.ndarray] = {}
        for lg in ("pt", "en", "es", "fr"):
            z_np = np.asarray(by_lang[lg]["z_23"], dtype=np.float32)
            y_pt = pt_model.sample_euler(torch.from_numpy(z_np), num_steps=4, guidance_scale=1.0).numpy()
            y_gpu = wgpu_eng.sample_euler(z_np, num_steps=4, guidance_scale=1.0)

            cos_parity = float(
                np.dot(y_pt.reshape(-1), y_gpu.reshape(-1))
                / (np.linalg.norm(y_pt) * np.linalg.norm(y_gpu) + 1e-12)
            )
            max_err = float(np.max(np.abs(y_pt - y_gpu)))
            assert cos_parity > 0.99999, f"[{concept_id}/{lg}] WebGPU parity cosine={cos_parity:.8f}"
            assert max_err < 5e-5, f"[{concept_id}/{lg}] WebGPU max_err={max_err:.6e}"

            # Must genuinely deform away from initial canonical Fibonacci sphere `X_0`!
            disp_rms = float(np.sqrt(np.mean((y_gpu[:, :3] - x0_xyz) ** 2)))
            assert disp_rms > 0.18, f"[{concept_id}/{lg}] Insufficient 3D OT displacement from sphere: {disp_rms:.4f}"
            lang_xyz[lg] = y_gpu[:, :3].reshape(-1)

        en_vec = lang_xyz["en"]
        en_clouds_by_concept[concept_id] = en_vec
        xl_cos_list: list[float] = []
        xl_chamf_list: list[float] = []
        for lg_other in ("pt", "es", "fr"):
            v_o = lang_xyz[lg_other]
            c_xl = float(np.dot(en_vec, v_o) / (np.linalg.norm(en_vec) * np.linalg.norm(v_o) + 1e-12))
            assert c_xl > 0.95, f"[{concept_id}] Cross-lingual 3D cosine EN vs {lg_other} too low: {c_xl:.4f}"
            xl_cos_list.append(c_xl)
            xl_chamf_list.append(_chamfer_l2_np(en_vec, v_o))
        min_intra_xling_cos[concept_id] = min(xl_cos_list)
        max_intra_xling_chamfer[concept_id] = max(xl_chamf_list)

    # Distinct concepts MUST produce distinct 3D point-cloud geometries (both in OT cosine margin & 3D Chamfer-L2)!
    c_keys = list(en_clouds_by_concept.keys())
    for i in range(len(c_keys)):
        for j in range(i + 1, len(c_keys)):
            ka, kb = c_keys[i], c_keys[j]
            va = en_clouds_by_concept[ka]
            vb = en_clouds_by_concept[kb]
            c_ij = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            min_intra = min(min_intra_xling_cos[ka], min_intra_xling_cos[kb])
            assert (1.0 - c_ij) > (1.0 - min_intra) * 5.0, (
                f"Concepts {ka} and {kb} insufficiently separated vs cross-lingual invariance "
                f"(inter_cos={c_ij:.4f}, min_intra_xling_cos={min_intra:.4f})!"
            )
            inter_chamfer = _chamfer_l2_np(va, vb)
            max_intra_ch = max(max_intra_xling_chamfer[ka], max_intra_xling_chamfer[kb])
            assert inter_chamfer > max_intra_ch * 1.5, (
                f"Concepts {ka} and {kb} 3D Chamfer-L2 too close: inter={inter_chamfer:.4f} vs intra_xling={max_intra_ch:.4f}"
            )
