"""Dual-Engine Verification Suite (`Native PyTorch CPU/MPS` + `Physical WebGPU WGSL 1.0`) for `VQ4-Scar`.

Enforces Ecosystem Rule #5:
  Every `VQ4-Scar` (`4D LUT256` + `d_eq` + `FWHT-64` + `2-Pass INT4-ASVD`) operator MUST run
  with verified numerical parity (`cosine > 0.999999`) across BOTH:
    1. Native Engine (`PyTorch CPU / Metal MPS / CUDA` via `VQ4ScarLinear`)
    2. Web Engine (`Physical WebGPU WGSL 1.0` via `vq4_equilibrate_fwht64` & `mos_fused_vq4_int4scar_gemv`)
  and across all 4 target languages (`PT`, `EN`, `ES`, `FR`) on strictly disjoint held-out tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.sub2bit_mos import dequantize_sub2bit_coalesced, quantize_sub2bit_block64
from scar_cache.vq4_scar_mos import (
    VQ4ScarLinear,
    WebGPUVQ4ScarExecutor,
    quantize_matrix_vq4_scar_2pass,
)


def test_vq4_scar_exact_zero_and_byte_budget() -> None:
    torch.manual_seed(42)
    lin = nn.Linear(512, 1024, bias=True)
    # Zero out the first 64-weight block of row 0 to verify exact zero preservation via LUT[0] = (0,0,0,0)
    with torch.no_grad():
        lin.weight[0, :64].zero_()

    packed = quantize_matrix_vq4_scar_2pass(lin.weight, col_rms=None, bias=lin.bias, rank=32, two_pass=True)
    assert torch.all(packed.lut256_f16[0] == 0.0), "LUT256[0] must be hard-pinned to exact (0,0,0,0)!"

    expected_bytes = (
        (1024 * 512) // 4             # VQ4 1-byte indices (2.00 bpw)
        + ((1024 * 512) // 64) * 2    # FP16 block-64 scales (0.25 bpw)
        + 256 * 4 * 2                 # 2 KB LUT256
        + 512 * 2                     # FP16 channel equilibration d_eq
        + (32 * 512) // 2             # INT4 A_scar nibbles
        + ((32 * 512) // 64) * 2      # FP16 A_scar block scales
        + (1024 * 32) // 2            # INT4 B_scar nibbles
        + 1024 * 2                    # FP16 B_scar row scales
        + 1024 * 2                    # FP16 bias
    )
    assert packed.physical_bytes == expected_bytes


def test_vq4_scar_native_cpu_and_mps_outlier_healing() -> None:
    torch.manual_seed(2026)
    in_dim, out_dim = 512, 1024
    lin = nn.Linear(in_dim, out_dim, bias=False)

    # Disjoint calibration vs held-out test activations with realistic 300x channel outliers!
    outlier_profile = torch.ones(in_dim)
    outlier_profile[7] = 280.0
    outlier_profile[131] = 420.0
    outlier_profile[389] = 190.0

    with torch.no_grad():
        # Counter-scale outlier columns in W (just like real Transformer layers!)
        lin.weight.mul_(0.04 / outlier_profile.sqrt().unsqueeze(0))
        x_calib = torch.randn(256, in_dim) * outlier_profile.unsqueeze(0)
        x_heldout = torch.randn(192, in_dim) * outlier_profile.unsqueeze(0)

        col_rms = x_calib.square().mean(dim=0).sqrt()
        y_ref = lin(x_heldout)
        ref_norm = y_ref.norm().item()

        # Legacy scalar 2-bit baseline
        p2, s2 = quantize_sub2bit_block64(lin.weight.float())
        w_s2 = dequantize_sub2bit_coalesced(p2, s2, lin.weight.shape, torch.float32)
        err_scalar2 = (x_heldout @ w_s2.T - y_ref).norm().item() / ref_norm

        # VQ4-Scar 1-Pass vs 2-Pass on Native CPU
        vq_1pass = VQ4ScarLinear.from_linear(lin, col_rms=col_rms, rank=32, two_pass=False)
        vq_2pass = VQ4ScarLinear.from_linear(lin, col_rms=col_rms, rank=32, two_pass=True)

        y_cpu_1 = vq_1pass(x_heldout)
        y_cpu_2 = vq_2pass(x_heldout)
        err_1pass = (y_cpu_1 - y_ref).norm().item() / ref_norm
        err_2pass = (y_cpu_2 - y_ref).norm().item() / ref_norm

        assert err_2pass < err_1pass, f"2-Pass ({err_2pass:.4f}) should beat 1-Pass ({err_1pass:.4f})"
        assert err_2pass < 0.35 * err_scalar2, (
            f"VQ4-Scar 2-Pass ({err_2pass:.4f}) must crush scalar 2-bit ({err_scalar2:.4f}) on outlier channels!"
        )

        # Native Metal MPS parity check (when MPS is available on Apple Silicon)
        if torch.backends.mps.is_available():
            dev_mps = torch.device("mps")
            vq_mps = vq_2pass.to(dev_mps)
            y_mps = vq_mps(x_heldout.to(dev_mps)).cpu()
            cos_cpu_mps = F.cosine_similarity(y_cpu_2.flatten(), y_mps.flatten(), dim=0).item()
            assert cos_cpu_mps > 0.999999, f"Native CPU vs MPS cosine drift: {cos_cpu_mps}"


def test_vq4_scar_physical_webgpu_wgsl_parity_with_native() -> None:
    pytest.importorskip("wgpu")
    torch.manual_seed(777)

    executor = WebGPUVQ4ScarExecutor()
    for rank, has_bias in ((0, False), (48, True)):
        lin = nn.Linear(768, 1536, bias=has_bias)
        col_rms = torch.linspace(0.5, 25.0, 768)
        col_rms[19] = 310.0

        vq_lin = VQ4ScarLinear.from_linear(lin, col_rms=col_rms, rank=rank, two_pass=True)
        x_test = torch.randn(6, 768) * col_rms.sqrt().unsqueeze(0)

        y_native = vq_lin(x_test).detach().cpu().numpy()
        y_webgpu = executor.execute_gemv(vq_lin.as_packed(), x_test.numpy())

        cos = float(
            np.dot(y_native.ravel(), y_webgpu.ravel())
            / (np.linalg.norm(y_native.ravel()) * np.linalg.norm(y_webgpu.ravel()) + 1e-12)
        )
        max_abs = float(np.max(np.abs(y_native - y_webgpu)))
        rel_l2 = float(np.linalg.norm(y_native - y_webgpu) / (np.linalg.norm(y_native) + 1e-12))

        assert cos > 0.999999, f"WebGPU WGSL vs Native PyTorch cosine={cos} (rank={rank})"
        assert rel_l2 < 1e-5, f"WebGPU WGSL vs Native PyTorch rel_l2={rel_l2} (rank={rank})"
        assert max_abs < 2e-3, f"WebGPU WGSL vs Native PyTorch max_abs={max_abs} (rank={rank})"


def test_vq4_scar_real_qwen_polyglot_native_and_webgpu() -> None:
    """End-to-end polyglot (`PT / EN / ES / FR`) test on real Qwen2.5 weights across BOTH Native & WebGPU!"""
    pytest.importorskip("wgpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, local_files_only=True)
    except Exception:
        pytest.skip(f"{model_id} not cached locally")

    model.eval()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)

    # Strictly disjoint 4-language calibration texts vs 4-language held-out test texts
    calib_texts = [
        "A fotossíntese converte luz solar em energia química dentro dos cloroplastos das plantas.",
        "Binary search halves a sorted array at each step to achieve logarithmic lookup complexity.",
        "Las estrellas fusionan hidrógeno en helio manteniendo el equilibrio frente a la gravedad.",
        "Les récifs coralliens abritent une biodiversité marine exceptionnelle dans les océans.",
    ]
    heldout_by_lang = {
        "PT": "Explique a mecânica quântica e o limite termodinâmico de Landauer com a constante de Boltzmann.",
        "EN": "Explain quantum mechanics and Landauer's thermodynamic limit via Boltzmann's constant.",
        "ES": "Explica la mecánica cuántica y el límite termodinámico de Landauer con la constante de Boltzmann.",
        "FR": "Expliquez la mécanique quantique et la limite thermodynamique de Landauer via la constante de Boltzmann.",
    }

    target_mlp = model.model.layers[12].mlp
    gate_sq = torch.zeros(target_mlp.gate_proj.in_features, device=dev)
    down_sq = torch.zeros(target_mlp.down_proj.in_features, device=dev)
    n_cal = 0

    def _h_gate(_m, inp, _o):
        nonlocal gate_sq, n_cal
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        gate_sq += x.square().sum(dim=0)
        n_cal += x.shape[0]

    def _h_down(_m, inp, _o):
        nonlocal down_sq
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
        down_sq += x.square().sum(dim=0)

    hg = target_mlp.gate_proj.register_forward_hook(_h_gate)
    hd = target_mlp.down_proj.register_forward_hook(_h_down)
    with torch.no_grad():
        for txt in calib_texts:
            model(tok(txt, return_tensors="pt")["input_ids"].to(dev), use_cache=False)
    hg.remove()
    hd.remove()

    gate_rms = (gate_sq / n_cal).sqrt().clamp(min=1e-6)
    down_rms = (down_sq / n_cal).sqrt().clamp(min=1e-6)

    # Quantize real Qwen `gate_proj` & `down_proj` into unified `VQ4ScarLinear` (`rank=48`, 2-Pass)
    vq_gate = VQ4ScarLinear.from_linear(target_mlp.gate_proj, col_rms=gate_rms, rank=48, two_pass=True).to(dev)
    vq_down = VQ4ScarLinear.from_linear(target_mlp.down_proj, col_rms=down_rms, rank=48, two_pass=True).to(dev)

    # Legacy scalar 2-bit reference for comparison
    pg, sg = quantize_sub2bit_block64(target_mlp.gate_proj.weight.detach().float())
    wg_s2 = dequantize_sub2bit_coalesced(pg, sg, target_mlp.gate_proj.weight.shape, torch.float32).to(dev)
    pd, sd = quantize_sub2bit_block64(target_mlp.down_proj.weight.detach().float())
    wd_s2 = dequantize_sub2bit_coalesced(pd, sd, target_mlp.down_proj.weight.shape, torch.float32).to(dev)

    executor = WebGPUVQ4ScarExecutor()

    for lang, prompt in heldout_by_lang.items():
        cap: dict[str, torch.Tensor] = {}

        def _cap_g(_m, inp, _o):
            cap["xg"] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])

        def _cap_d(_m, inp, _o):
            cap["xd"] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])

        h1 = target_mlp.gate_proj.register_forward_hook(_cap_g)
        h2 = target_mlp.down_proj.register_forward_hook(_cap_d)
        with torch.no_grad():
            model(tok(prompt, return_tensors="pt")["input_ids"].to(dev), use_cache=False)
        h1.remove()
        h2.remove()

        xg, xd = cap["xg"], cap["xd"]
        yg_ref = target_mlp.gate_proj(xg)
        yd_ref = target_mlp.down_proj(xd)

        # 1. Native PyTorch (`MPS/CPU`) accuracy across all 4 languages
        yg_nat = vq_gate(xg)
        yd_nat = vq_down(xd)

        err_gate_vq4 = (yg_nat - yg_ref).norm().item() / yg_ref.norm().item()
        err_gate_s2 = (xg @ wg_s2.T - yg_ref).norm().item() / yg_ref.norm().item()
        err_down_vq4 = (yd_nat - yd_ref).norm().item() / yd_ref.norm().item()
        err_down_s2 = (xd @ wd_s2.T - yd_ref).norm().item() / yd_ref.norm().item()

        assert err_gate_vq4 < err_gate_s2, f"[{lang}] VQ4-Scar gate ({err_gate_vq4:.3f}) must beat scalar 2-bit ({err_gate_s2:.3f})"
        assert err_down_vq4 < err_down_s2, f"[{lang}] VQ4-Scar down ({err_down_vq4:.3f}) must beat scalar 2-bit ({err_down_s2:.3f})"

        # 2. Physical WebGPU WGSL 1.0 parity against Native PyTorch on the exact language tokens!
        yg_wgpu = executor.execute_gemv(vq_gate.as_packed(), xg.cpu().numpy())
        yd_wgpu = executor.execute_gemv(vq_down.as_packed(), xd.cpu().numpy())

        yg_nat_np = yg_nat.cpu().numpy()
        yd_nat_np = yd_nat.cpu().numpy()

        cos_g = float(
            np.dot(yg_nat_np.ravel(), yg_wgpu.ravel())
            / (np.linalg.norm(yg_nat_np.ravel()) * np.linalg.norm(yg_wgpu.ravel()) + 1e-12)
        )
        cos_d = float(
            np.dot(yd_nat_np.ravel(), yd_wgpu.ravel())
            / (np.linalg.norm(yd_nat_np.ravel()) * np.linalg.norm(yd_wgpu.ravel()) + 1e-12)
        )
        assert cos_g > 0.999999, f"[{lang}] Native vs WebGPU gate cosine={cos_g}"
        assert cos_d > 0.999999, f"[{lang}] Native vs WebGPU down cosine={cos_d}"
