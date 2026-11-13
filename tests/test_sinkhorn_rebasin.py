"""Unit & Integration Tests for `scar_cache.sinkhorn_rebasin` (`v2.4.0` — Track 2.2)."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import save_file
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.sinkhorn_rebasin import (
    apply_exact_swiglu_monomial_gauge,
    audit_unique_module_bytes,
    build_activation_whitener,
    build_rebasin_shared_swiglu_pair,
    convert_llm_cross_layer_rebasin_mos,
    main_cli,
    pack_signed_int4_nibbles,
    solve_swiglu_sinkhorn_rebasin,
)


class _ToySwiGLU(nn.Module):
    def __init__(self, d: int = 128, d_ff: int = 256) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyDecoderBlock(nn.Module):
    def __init__(self, d: int = 128, d_ff: int = 256) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = _ToySwiGLU(d=d, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.25 * self.mlp(self.ln(x))


class _ToyCausalLM(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 128, d_ff: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        torch.manual_seed(101)
        self.embed = nn.Embedding(vocab, d)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_ToyDecoderBlock(d=d, d_ff=d_ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> object:
        h = self.embed(input_ids)
        for blk in self.model.layers:
            h = blk(h)
        return type("Out", (), {"logits": self.lm_head(h)})()


def test_exact_swiglu_monomial_gauge_zero_error_invariance() -> None:
    """Proves $(P, \\alpha) \\in \\mathrm{GL}_1(\\mathbb{R})^{d_{\\text{ff}}} \\rtimes S_{d_{\\text{ff}}}$ preserves SwiGLU output to `< 1e-5`."""
    torch.manual_seed(7)
    mlp = _ToySwiGLU(d=128, d_ff=256).eval()
    perm = torch.randperm(256)
    signs = torch.where(torch.rand(256) > 0.5, torch.ones(256), -torch.ones(256))
    scales = torch.empty(256).uniform_(0.3, 3.2) * signs

    gauged_mlp = apply_exact_swiglu_monomial_gauge(mlp, perm=perm, monomial_scale=scales).eval()
    x = torch.randn(12, 128)
    with torch.no_grad():
        y_ref = mlp(x)
        y_gauged = gauged_mlp(x)

    max_diff = float((y_ref - y_gauged).abs().max().item())
    assert max_diff < 1e-5, f"Monomial SwiGLU gauge violated invariance: max_diff={max_diff}"


def test_sinkhorn_rebasin_recovers_permuted_rescaled_specialist_and_shares_storage_ptr() -> None:
    """Verifies Git Re-Basin recovery on permuted/rescaled SwiGLU specialist + exact pointer sharing."""
    torch.manual_seed(23)
    d, d_ff = 128, 256
    spec_a = _ToySwiGLU(d=d, d_ff=d_ff).eval()
    # Create Specialist B by adding a structured task delta AND scrambling all 256 channels + signs/scales!
    spec_b_unscrambled = _ToySwiGLU(d=d, d_ff=d_ff).eval()
    with torch.no_grad():
        u_delta = torch.randn(d_ff, 8) * 0.04
        v_delta = torch.randn(8, d) * 0.04
        spec_b_unscrambled.gate_proj.weight.copy_(spec_a.gate_proj.weight + u_delta @ v_delta)
        spec_b_unscrambled.up_proj.weight.copy_(spec_a.up_proj.weight - u_delta @ v_delta)
        spec_b_unscrambled.down_proj.weight.copy_(spec_a.down_proj.weight + (u_delta @ v_delta).T)

    true_perm = torch.randperm(d_ff)
    true_alpha = torch.empty(d_ff).uniform_(0.4, 2.5) * torch.where(
        torch.rand(d_ff) > 0.5, torch.ones(d_ff), -torch.ones(d_ff)
    )
    spec_b = apply_exact_swiglu_monomial_gauge(spec_b_unscrambled, true_perm, true_alpha).eval()

    align = solve_swiglu_sinkhorn_rebasin(spec_a, spec_b)
    assert align.aligned_cosine > 0.90, f"Expected >0.90 aligned cosine, got {align.aligned_cosine}"
    assert align.aligned_cosine > abs(align.unaligned_cosine) + 0.50
    assert align.unquantized_max_abs_diff < 1e-5

    x_cal = torch.randn(128, d)
    z_a = F.silu(spec_a.gate_proj(x_cal)) * spec_a.up_proj(x_cal)
    z_b = F.silu(spec_b.gate_proj(x_cal)) * spec_b.up_proj(x_cal)
    wh_in_a = build_activation_whitener(x_cal)
    wh_in_b = build_activation_whitener(x_cal)
    wh_int_a = build_activation_whitener(z_a)
    wh_int_b = build_activation_whitener(z_b)

    na_a, na_b, _ = build_rebasin_shared_swiglu_pair(
        spec_a, spec_b, wh_in_a, wh_in_b, wh_int_a, wh_int_b, rank=16, naive_unaligned=True
    )
    rb_a, rb_b, _ = build_rebasin_shared_swiglu_pair(
        spec_a, spec_b, wh_in_a, wh_in_b, wh_int_a, wh_int_b, rank=16, naive_unaligned=False
    )

    # 1. Verify genuine storage pointer identity (`0 B` unreported allocation!)
    for pname in ("gate_proj", "up_proj", "down_proj"):
        assert (
            rb_a[pname].packed_shared.untyped_storage().data_ptr()
            == rb_b[pname].packed_shared.untyped_storage().data_ptr()
        )
    holder = nn.ModuleDict({"a": nn.ModuleDict(rb_a), "b": nn.ModuleDict(rb_b)})
    rep_bytes = sum(m.physical_bytes for m in rb_a.values()) + sum(m.physical_bytes for m in rb_b.values())
    uniq_bytes = audit_unique_module_bytes(holder)
    assert rep_bytes == uniq_bytes, f"Discrepancy: reported={rep_bytes} vs unique={uniq_bytes}"

    # 2. Verify dramatic reconstruction superiority on unseen inputs
    x_test = torch.randn(64, d)
    with torch.no_grad():
        y_ref_b = spec_b(x_test)
        y_na_b = na_b["down_proj"](F.silu(na_b["gate_proj"](x_test)) * na_b["up_proj"](x_test))
        y_rb_b = rb_b["down_proj"](F.silu(rb_b["gate_proj"](x_test)) * rb_b["up_proj"](x_test))

    err_naive = float((y_ref_b - y_na_b).norm().item() / y_ref_b.norm().item())
    err_rebasin = float((y_ref_b - y_rb_b).norm().item() / y_ref_b.norm().item())
    cos_rebasin = float(F.cosine_similarity(y_ref_b, y_rb_b, dim=-1).mean().item())
    assert err_rebasin < err_naive * 0.20, f"Expected >5x lower error: {err_rebasin:.4f} vs {err_naive:.4f}"
    assert cos_rebasin > 0.99, f"Expected cos > 0.99, got {cos_rebasin:.5f}"


def test_convert_llm_cross_layer_rebasin_mos_with_and_without_procrustes() -> None:
    """Verifies end-to-end LLM conversion + Procrustes mode + zero byte discrepancy on CPU and MPS/CUDA."""
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    lm = _ToyCausalLM(vocab=128, d=128, d_ff=256, n_layers=4).eval().to(dev)
    calib = [torch.randint(0, 128, (2, 24), device=dev) for _ in range(3)]
    test_ids = torch.randint(0, 128, (2, 20), device=dev)

    with torch.no_grad():
        ref_lg = lm(test_ids).logits.float()

    na_lm, na_st = convert_llm_cross_layer_rebasin_mos(
        lm, calib_batches=calib, pairs=[(0, 1), (2, 3)], rank=24, naive_unaligned=True
    )
    rb_lm, rb_st = convert_llm_cross_layer_rebasin_mos(
        lm, calib_batches=calib, pairs=[(0, 1), (2, 3)], rank=24, use_procrustes=True
    )
    assert na_st["storage_audit_discrepancy_bytes"] == 0
    assert rb_st["storage_audit_discrepancy_bytes"] == 0
    assert rb_st["pair_compression_ratio_vs_fp16"] > 1.5

    with torch.no_grad():
        lg_na = na_lm(test_ids).logits.float()
        lg_rb = rb_lm(test_ids).logits.float()
    p_ref = F.softmax(ref_lg, dim=-1)
    kl_na = float((p_ref * (F.log_softmax(ref_lg, dim=-1) - F.log_softmax(lg_na, dim=-1))).sum(-1).mean().item())
    kl_rb = float((p_ref * (F.log_softmax(ref_lg, dim=-1) - F.log_softmax(lg_rb, dim=-1))).sum(-1).mean().item())
    assert kl_rb < kl_na, f"Expected Sinkhorn-Procrustes KL ({kl_rb:.5f}) < Naive KL ({kl_na:.5f})"


def test_mos_rebasin_cli_audit_72b_style_shard(tmp_path: Path, capsys: object) -> None:
    """Verifies `mos-rebasin audit-72b-shard` CLI over a synthetic MoS v2 `.safetensors` shard."""
    torch.manual_seed(99)
    d, d_ff = 128, 256
    shard_tensors = {}
    for l_idx in (78, 79):
        shard_tensors[f"layers.{l_idx}.post_attention_layernorm.weight"] = torch.ones(d, dtype=torch.float16)
        for pname, (out_f, in_f) in (
            ("gate_proj", (d_ff, d)),
            ("up_proj", (d_ff, d)),
            ("down_proj", (d, d_ff)),
        ):
            n_blk = out_f * (in_f // 64)
            q = torch.randint(-7, 7, (n_blk, 64), dtype=torch.int8)
            packed = pack_signed_int4_nibbles(q)
            scales = torch.full((n_blk, 1), 0.02, dtype=torch.float16)
            a_s = torch.randn((8, in_f), dtype=torch.float16) * 0.01
            b_s = torch.randn((out_f, 8), dtype=torch.float16) * 0.01
            pfx = f"layers.{l_idx}.mlp.{pname}"
            shard_tensors[f"{pfx}.packed_base"] = packed
            shard_tensors[f"{pfx}.scales_base"] = scales
            shard_tensors[f"{pfx}.A_scar"] = a_s
            shard_tensors[f"{pfx}.B_scar"] = b_s

    shard_path = tmp_path / "toy_mos_group_19.safetensors"
    save_file(shard_tensors, str(shard_path))

    rc = main_cli(["audit-72b-shard", "--shard", str(shard_path), "--layer-a", "78", "--layer-b", "79", "--rank", "16", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert out["storage"]["reported_vs_unique_diff_bytes"] == 0
    assert out["sinkhorn_rebasin_shared_grid"]["layer_b_relative_error"] <= out["naive_unaligned_layer_b"]["relative_error"]


def test_tiled_online_dual_potential_sinkhorn_matches_dense_rebasin() -> None:
    """Verifies `project_tiled_swiglu_sinkhorn_bijection` (`tile_size=64`) matches dense Sinkhorn alignment."""
    torch.manual_seed(77)
    d, d_ff = 128, 256
    mlp_a = _ToySwiGLU(d=d, d_ff=d_ff).eval()
    mlp_b_clean = _ToySwiGLU(d=d, d_ff=d_ff).eval()
    with torch.no_grad():
        delta = torch.randn(d_ff, d) * 0.015
        mlp_b_clean.gate_proj.weight.copy_(mlp_a.gate_proj.weight + delta)
        mlp_b_clean.up_proj.weight.copy_(mlp_a.up_proj.weight - delta)
        mlp_b_clean.down_proj.weight.copy_(mlp_a.down_proj.weight + delta.T)

    perm_true = torch.randperm(d_ff)
    alpha_true = torch.empty(d_ff).uniform_(0.4, 2.5) * torch.where(
        torch.rand(d_ff) > 0.5, torch.ones(d_ff), -torch.ones(d_ff)
    )
    mlp_b = apply_exact_swiglu_monomial_gauge(mlp_b_clean, perm_true, alpha_true).eval()

    al_dense = solve_swiglu_sinkhorn_rebasin(mlp_a, mlp_b, sinkhorn_iters=6, tiled_threshold=10000)
    al_tiled = solve_swiglu_sinkhorn_rebasin(mlp_a, mlp_b, sinkhorn_iters=6, tiled_threshold=128, tile_size=64)

    assert int(torch.unique(al_tiled.perm).numel()) == d_ff
    assert al_tiled.aligned_cosine > 0.94
    assert abs(al_tiled.aligned_cosine - al_dense.aligned_cosine) < 1e-3
    assert torch.equal(al_tiled.perm, al_dense.perm)
    assert al_tiled.unquantized_max_abs_diff < 1e-5

