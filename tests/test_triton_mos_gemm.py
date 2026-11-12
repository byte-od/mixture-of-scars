"""Unit & Numerical Parity Tests for Fused MoS Kernels (Dense-Base + Canonical Block-64 W4A16 + ASVD Scar)."""

import ast
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.cli_72b import (
    _pack_block64_for_cuda_mode,
    _probe_cuda_int4pack_abi,
    dequantize_int4_coalesced,
    safe_gemm,
)
from scar_cache.real_70b import MoSResidualLinear
from scar_cache.triton_mos_gemm import (
    HAS_TRITON,
    fused_int4_mos_linear,
    fused_mos_linear,
    pytorch_fused_mos_linear,
    pytorch_int4_mos_linear,
    triton_fused_int4_mos_linear,
)
from scripts.export_canonical_72b_mos import quantize_int4_blockwise


def test_fused_mos_linear_numerical_equivalence_2d():
    torch.manual_seed(42)
    M, K, N, R = 32, 256, 512, 16

    x = torch.randn(M, K)
    W_base = torch.randn(N, K)
    A = torch.randn(R, K)
    B = torch.randn(N, R)

    out_ref = F.linear(x, W_base) + F.linear(F.linear(x, A), B)
    out_fused = pytorch_fused_mos_linear(x, W_base, A, B)

    rel_error = (torch.norm(out_ref - out_fused) / torch.norm(out_ref)).item()
    assert rel_error < 1e-6, f"Relative error {rel_error:.2e} exceeded 2D threshold"


def test_fused_mos_linear_3d_and_scaling():
    torch.manual_seed(42)
    batch, seq, in_dim, out_dim, rank = 2, 8, 128, 256, 16
    scaling = 0.5

    x = torch.randn(batch, seq, in_dim)
    base = nn.Linear(in_dim, out_dim, bias=False)

    mos_proj = MoSResidualLinear(
        base_linear=base,
        in_features=in_dim,
        out_features=out_dim,
        rank=rank,
        scaling=scaling,
    )

    y_unfused = mos_proj(x, use_fused=False)
    y_fused = mos_proj(x, use_fused=True)

    rel_error = (torch.norm(y_unfused - y_fused) / torch.norm(y_unfused)).item()
    assert rel_error < 1e-6, f"Relative error {rel_error:.2e} exceeded 3D scaling threshold"


def test_fused_interface_fallback():
    """Validates that the unified dense-base interface works cleanly on any hardware."""
    torch.manual_seed(42)
    x = torch.randn(4, 64)
    W_base = torch.randn(128, 64)
    A = torch.randn(8, 64)
    B = torch.randn(128, 8)

    out = fused_mos_linear(x, W_base, A, B)
    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()


def test_block64_int4_mos_decode_and_prefill_parity_and_bias():
    """Verifies exact Block-64 even/odd nibble unpacking + ASVD Scar + bias across M=1, M=5, and M=40."""
    torch.manual_seed(123)
    # Use non-power-of-two multiples of 64 (like Qwen-72B 29568 = 462 * 64)
    out_f, in_f, rank = 384, 192, 16
    w_orig = torch.randn(out_f, in_f) * 0.05
    pb, sb, shape = quantize_int4_blockwise(w_orig, block_size=64)
    a_scar = (torch.randn(rank, in_f) * 0.04).to(torch.float16)
    b_scar = (torch.randn(out_f, rank) * 0.04).to(torch.float16)
    bias = (torch.randn(out_f) * 0.1).to(torch.float16)

    w_deq = dequantize_int4_coalesced(pb, sb, shape, torch.float32)

    for inp_shape in [(1, 1, in_f), (5, in_f), (2, 20, in_f)]:
        x = torch.randn(*inp_shape, dtype=torch.float32) * 1.5
        y_oracle = F.linear(x, w_deq, bias.float()) + F.linear(F.linear(x, a_scar.float()), b_scar.float())
        y_split = fused_int4_mos_linear(
            x=x,
            packed_base=pb,
            scales_base=sb,
            out_features=out_f,
            in_features=in_f,
            A_scar=a_scar,
            B_scar=b_scar,
            bias=bias,
        )
        assert y_split.shape == y_oracle.shape
        rel_err = float((torch.norm(y_oracle - y_split) / torch.norm(y_oracle)).item())
        cos = float(F.cosine_similarity(y_oracle.reshape(1, -1), y_split.reshape(1, -1)).item())
        assert rel_err < 1e-5, f"Expected <1e-5 rel_err on shape {inp_shape}, got {rel_err}"
        assert cos > 0.999999, f"Expected >0.999999 cosine on shape {inp_shape}, got {cos}"

    # Verify rank-0 empty scar + bias=None edge case
    empty_a = torch.empty((0, in_f), dtype=torch.float16)
    empty_b = torch.empty((out_f, 0), dtype=torch.float16)
    x_r0 = torch.randn(3, in_f)
    y_r0_ref = F.linear(x_r0, w_deq)
    y_r0_got = pytorch_int4_mos_linear(x_r0, pb, sb, out_f, in_f, A_scar=empty_a, B_scar=empty_b, bias=None)
    assert float((torch.norm(y_r0_ref - y_r0_got) / torch.norm(y_r0_ref)).item()) < 1e-5


def test_block64_int4_mos_extreme_swiglu_outlier_fp16_safety():
    """Verifies zero Inf/NaN when Qwen-72B SwiGLU activations spike to 195,720x (>3x FP16 max 65,504)."""
    torch.manual_seed(77)
    out_f, in_f, rank = 256, 192, 16
    pb, sb, shape = quantize_int4_blockwise(torch.randn(out_f, in_f) * 0.02, block_size=64)
    a_scar = (torch.randn(rank, in_f) * 0.01).to(torch.float16)
    b_scar = (torch.randn(out_f, rank) * 0.01).to(torch.float16)

    x_outlier = torch.randn(2, 16, in_f, dtype=torch.float32)
    x_outlier[0, 3, 7] = 195720.0
    x_outlier[1, 11, 42] = -184000.0

    w_deq = dequantize_int4_coalesced(pb, sb, shape, torch.float32)
    y_ref = safe_gemm(x_outlier, w_deq) + F.linear(F.linear(x_outlier, a_scar.float()), b_scar.float())
    y_got = fused_int4_mos_linear(x_outlier, pb, sb, out_f, in_f, A_scar=a_scar, B_scar=b_scar)

    assert torch.isfinite(y_got).all(), "Expected 100% finite outputs under 195,720x SwiGLU outlier"
    cos = float(F.cosine_similarity(y_ref.reshape(1, -1), y_got.reshape(1, -1)).item())
    assert cos > 0.99999, f"Expected >0.99999 cosine under extreme SwiGLU outlier, got {cos}"

    # If CUDA + Triton are present, also validate live GPU kernel parity across decode (M=1) and prefill (M=32)
    if torch.cuda.is_available() and HAS_TRITON:
        dev = torch.device("cuda:0")
        pb_c, sb_c = pb.to(dev), sb.to(dev)
        a_c, b_c = a_scar.to(dev), b_scar.to(dev)
        for m_sz in (1, 32):
            x_c = x_outlier.reshape(-1, in_f)[:m_sz].to(dev)
            y_gpu = triton_fused_int4_mos_linear(x_c, pb_c, sb_c, out_f, in_f, A_scar=a_c, B_scar=b_c)
            y_cpu = y_ref.reshape(-1, out_f)[:m_sz].to(dev)
            assert torch.isfinite(y_gpu).all()
            cos_gpu = float(F.cosine_similarity(y_cpu.reshape(1, -1), y_gpu.float().reshape(1, -1)).item())
            assert cos_gpu > 0.9995, f"Expected >0.9995 CUDA Triton parity for M={m_sz}, got {cos_gpu}"


def test_cuda_int4pack_packer_modes_and_kernel_definitions():
    """Verifies CUDA int4pack nibble packers, non-CUDA probe safety, and Triton kernel definitions."""
    pb_2d = torch.randint(0, 256, (16, 64), dtype=torch.uint8)  # N=16, K=128
    assert _pack_block64_for_cuda_mode(pb_2d, "u8_swap").shape == (16, 64)
    assert _pack_block64_for_cuda_mode(pb_2d, "u8_noswap").shape == (16, 64)
    assert _pack_block64_for_cuda_mode(pb_2d, "i32_msb").shape == (16, 16)
    assert _pack_block64_for_cuda_mode(pb_2d, "i32_lsb").shape == (16, 16)

    assert _probe_cuda_int4pack_abi(torch.device("cpu")) is None

    src_path = Path(__file__).resolve().parents[1] / "scar_cache" / "triton_mos_gemm.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    fn_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_fused_int4_mos_gemv_decode_kernel" in fn_names
    assert "_fused_int4_mos_gemm_prefill_kernel" in fn_names
    assert "triton_fused_int4_mos_linear" in fn_names


if __name__ == "__main__":
    test_fused_mos_linear_numerical_equivalence_2d()
    test_fused_mos_linear_3d_and_scaling()
    test_fused_interface_fallback()
    test_block64_int4_mos_decode_and_prefill_parity_and_bias()
    test_block64_int4_mos_extreme_swiglu_outlier_fp16_safety()
    test_cuda_int4pack_packer_modes_and_kernel_definitions()
    print("✅ ALL FUSED MoS + BLOCK-64 W4A16 KERNEL TESTS PASSED!")
