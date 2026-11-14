"""Forensic Anatomy of Sub-2-Bit Quantization Noise on Qwen2.5-3B-Instruct.

Measures with zero handwaving:
  1. Exact Activation & Weight Kurtosis across `q, k, v, o, gate, up, down` across depths.
  2. Single-Projection End-to-End KL Divergence & Top-1 Agreement Ablation:
     Isolates the exact damage of 2-bit on ONLY `q_proj`, ONLY `k/v_proj`, ONLY `o_proj`,
     ONLY `gate_proj`, ONLY `up_proj`, and ONLY `down_proj` across all 36 layers.
  3. Layer-Group Depth Sensitivity (`L0..5`, `L6..17`, `L18..29`, `L30..35`):
     Shows exactly where along the residual stream quantization noise is born vs amplified.
  4. Mathematical Mechanism Comparison on Real Calibration Activations `X`:
     Compares relative output error `||(W - W_hat) X||_F / ||W X||_F` at identical bit-budgets:
       - Naive Uniform 2-bit (`2.25 bpw`)
       - Sub-2-Bit Lloyd-Max (`2.25 bpw`)
       - Sub-2-Bit + ASVD r=32 (`~2.75 bpw`)
       - Hadamard-Rotated Sub-2-Bit (`WHT-64/2048`, exact `2.25 bpw`, +0 bytes!)
       - Hadamard + Cholesky GPTQ Error-Feedback Sub-2-Bit (`exact 2.25 bpw`, +0 bytes!)
       - Hadamard + GPTQ + ASVD r=32
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/juliocesarreisfilho/Projects/scar-cache")
from scar_cache.mos_student import _DEFAULT_ASVD_CALIBRATION_TEXTS
from scar_cache.sub2bit_mos import (
    SUB2BIT_BETA,
    dequantize_sub2bit_coalesced,
    quantize_naive_uniform_2bit_block64,
    quantize_sub2bit_block64,
)


def fwht_last_dim_block64(x: torch.Tensor) -> torch.Tensor:
    """Exact orthonormal Fast Walsh-Hadamard Transform inside each 64-element block (`H^T = H, H^2 = I`)."""
    orig_shape = x.shape
    y = x.reshape(-1, 64).clone()
    h = 1
    while h < 64:
        y = y.view(-1, 64 // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    return (y.view(-1, 64) * (1.0 / 8.0)).view(orig_shape)


def random_orthogonal_block_hadamard(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Randomized Block-256 Walsh-Hadamard Transform (`QuIP#`-style incoherence: `H_256 diag(signs)`)."""
    orig_shape = x.shape
    y = (x * signs).reshape(-1, 256).clone()
    h = 1
    while h < 256:
        y = y.view(-1, 256 // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    return (y.view(-1, 256) * (1.0 / 16.0)).view(orig_shape)


@torch.no_grad()
def gptq_sub2bit_quantize_matrix(
    w_f32: torch.Tensor,
    x_calib: torch.Tensor,
    beta: float = SUB2BIT_BETA,
    damp_rel: float = 0.02,
) -> torch.Tensor:
    """Exact Cholesky GPTQ Sequential Error-Feedback Quantizer on our 4-level `{+-beta, +-1.0}` grid (`2.25 bpw`)."""
    N, K = w_f32.shape
    # Empirical Hessian H = (1/T) X^T X + damp * diag
    xf = x_calib.reshape(-1, K).float()
    H = (xf.T @ xf) / max(1, xf.shape[0])
    diag_mean = torch.diagonal(H).mean().clamp(min=1e-6)
    H.diagonal().add_(damp_rel * diag_mean)

    # Upper Cholesky of H^{-1}: GPTQ classic formulation
    L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    W = w_f32.clone().float()
    Q = torch.zeros_like(W)
    mid_ratio = 0.5 * (1.0 + beta)

    for b_start in range(0, K, 64):
        b_end = b_start + 64
        W_blk = W[:, b_start:b_end].clone()
        Hinv_blk = H_inv_chol[b_start:b_end, b_start:b_end]
        Err_blk = torch.zeros_like(W_blk)

        # Calibrate block scale alpha on current compensated block weights (weighted by diag(H))
        h_diag = torch.diagonal(H)[b_start:b_end].clamp(min=1e-8).unsqueeze(0)
        rms = W_blk.square().mean(dim=1, keepdim=True).sqrt()
        alpha = torch.minimum(W_blk.abs().amax(dim=1, keepdim=True), rms * 1.48).clamp(min=1e-7)
        for _ in range(4):
            b1 = (W_blk >= 0.0).float()
            b0 = (W_blk.abs() > (mid_ratio * alpha)).float()
            c = (2.0 * b1 - 1.0) * (beta + (1.0 - beta) * b0)
            alpha = ((h_diag * W_blk * c).sum(dim=1, keepdim=True) / (h_diag * c.square()).sum(dim=1, keepdim=True).clamp(min=1e-12)).clamp(min=1e-7)
        alpha = alpha.to(torch.float16).float()

        # Sequential column-by-column quantization + exact Cholesky error feedback inside block!
        for j in range(64):
            w_col = W_blk[:, j]
            d_jj = Hinv_blk[j, j].clamp(min=1e-8)
            b1_c = (w_col >= 0.0).float()
            b0_c = (w_col.abs() > (mid_ratio * alpha.squeeze(1))).float()
            q_col = (2.0 * b1_c - 1.0) * (beta + (1.0 - beta) * b0_c) * alpha.squeeze(1)
            Q[:, b_start + j] = q_col
            err = (w_col - q_col) / d_jj
            Err_blk[:, j] = err
            if j + 1 < 64:
                W_blk[:, j + 1 :] -= err.unsqueeze(1) * Hinv_blk[j, j + 1 :].unsqueeze(0)

        # Propagate remaining block error to ALL future blocks b_end..K!
        if b_end < K:
            W[:, b_end:] -= Err_blk @ H_inv_chol[b_start:b_end, b_end:]

    return Q


@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Forensic Noise Lab] Loading {model_id} FP16 on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(dev)
    model.eval()

    eval_prompts = list(_DEFAULT_ASVD_CALIBRATION_TEXTS) + [
        "<|im_start|>system\nResponda em português.<|im_end|>\n"
        "<|im_start|>user\nMe fale sobre física quântica e a constante de Boltzmann.<|im_end|>\n"
        "<|im_start|>assistant\nA física quântica estuda fenômenos na escala atômica e subatômica, enquanto a constante de Boltzmann relaciona energia térmica e temperatura na mecânica estatística.<|im_end|>"
    ]
    batches = [tok(t, return_tensors="pt")["input_ids"].to(dev) for t in eval_prompts]

    # Capture FP16 reference log-probs AND real input activations X on representative layers (L0, L18, L34)
    captured_x: Dict[str, List[torch.Tensor]] = {}
    probe_targets = {
        "model.layers.0.mlp.down_proj",
        "model.layers.18.self_attn.q_proj",
        "model.layers.18.self_attn.o_proj",
        "model.layers.18.mlp.gate_proj",
        "model.layers.18.mlp.up_proj",
        "model.layers.18.mlp.down_proj",
        "model.layers.34.mlp.down_proj",
        "model.layers.35.mlp.down_proj",
    }
    hooks = []
    for name, mod in model.named_modules():
        if name in probe_targets:
            captured_x[name] = []
            def _mk(n):
                def _hk(_m, inp, _o):
                    captured_x[n].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu().float())
                return _hk
            hooks.append(mod.register_forward_hook(_mk(name)))

    ref_logits: List[torch.Tensor] = []
    for b in batches:
        out = model(b, use_cache=False).logits[:, :-1, :].float().cpu()
        ref_logits.append(out)
    for h in hooks:
        h.remove()

    print("\n=== EXP 1: WHERE DO ACTIVATION & WEIGHT OUTLIERS LIVE? (KURTOSIS & MAX/RMS SPIKE) ===", flush=True)
    for name in sorted(probe_targets):
        X = torch.cat(captured_x[name], dim=0)
        mod = model.get_submodule(name)
        W = mod.weight.detach().cpu().float()
        x_rms = X.square().mean().sqrt().item()
        x_max = X.abs().amax().item()
        x_kurt = ((X / X.std().clamp(min=1e-8)).pow(4).mean()).item()
        ch_rms = X.square().mean(dim=0).sqrt()
        ch_ratio = (ch_rms.max() / ch_rms.median().clamp(min=1e-8)).item()
        w_kurt = ((W / W.std().clamp(min=1e-8)).pow(4).mean()).item()
        print(
            f"  {name:34s} | X max/rms={x_max/max(x_rms,1e-8):7.1f}x | ch_max/med={ch_ratio:6.1f}x "
            f"| X_kurt={x_kurt:8.1f} | W_kurt={w_kurt:5.2f}",
            flush=True,
        )

    print("\n=== EXP 2: MATHEMATICAL MECHANISM BENCHMARK ON REAL LAYER OUTPUT `|| (W - W_hat) X || / || W X ||` ===", flush=True)
    torch.manual_seed(42)
    for name in [
        "model.layers.18.self_attn.q_proj",
        "model.layers.18.mlp.gate_proj",
        "model.layers.18.mlp.down_proj",
        "model.layers.34.mlp.down_proj",
    ]:
        X = torch.cat(captured_x[name], dim=0)
        W = model.get_submodule(name).weight.detach().cpu().float()
        N, K = W.shape
        Y_ref = X @ W.T
        ref_norm = Y_ref.norm().item()
        s_ch = X.square().mean(dim=0).sqrt().clamp(min=1e-5)

        # (a) Naive 2-bit
        w_naive = quantize_naive_uniform_2bit_block64(W)
        err_naive = (X @ w_naive.T - Y_ref).norm().item() / ref_norm

        # (b) Sub-2-Bit Lloyd-Max (act-weighted)
        p2, s2 = quantize_sub2bit_block64(W, act_scales=s_ch)
        w_sub2 = dequantize_sub2bit_coalesced(p2, s2, (N, K), torch.float32)
        err_sub2 = (X @ w_sub2.T - Y_ref).norm().item() / ref_norm

        # (c) Sub-2-Bit + ASVD r=32
        u1, sv1, v1 = torch.svd_lowrank(W * s_ch.unsqueeze(0), q=44, niter=3)
        w_sm = W - (u1[:, :32] * sv1[:32].unsqueeze(0)) @ (v1[:, :32].T / s_ch.unsqueeze(0))
        p2a, s2a = quantize_sub2bit_block64(w_sm, act_scales=s_ch)
        w_2ba = dequantize_sub2bit_coalesced(p2a, s2a, (N, K), torch.float32)
        u2, sv2, v2 = torch.svd_lowrank((W - w_2ba) * s_ch.unsqueeze(0), q=44, niter=3)
        w_asvd = w_2ba + (u2[:, :32] * sv2[:32].unsqueeze(0)) @ (v2[:, :32].T / s_ch.unsqueeze(0))
        err_asvd = (X @ w_asvd.T - Y_ref).norm().item() / ref_norm

        # (d) Block-256 Randomized Walsh-Hadamard Incoherence (`W_rot = W H^T`, `X_rot = X H^T`) + Sub-2-Bit (+0 bytes!)
        signs = torch.where(torch.rand(K) > 0.5, 1.0, -1.0)
        W_rot = random_orthogonal_block_hadamard(W, signs)
        X_rot = random_orthogonal_block_hadamard(X, signs)
        s_rot = X_rot.square().mean(dim=0).sqrt().clamp(min=1e-5)
        p2h, s2h = quantize_sub2bit_block64(W_rot, act_scales=s_rot)
        W_rot_q = dequantize_sub2bit_coalesced(p2h, s2h, (N, K), torch.float32)
        err_wht = (X_rot @ W_rot_q.T - Y_ref).norm().item() / ref_norm

        # (e) Hadamard + Cholesky GPTQ Sequential Error-Feedback (`exact 2.25 bpw`, +0 bytes!)
        W_rot_gptq = gptq_sub2bit_quantize_matrix(W_rot, X_rot)
        err_wht_gptq = (X_rot @ W_rot_gptq.T - Y_ref).norm().item() / ref_norm

        # (f) Block-64 INT4 baseline (`4.25 bpw`)
        blks = W.reshape(-1, 64)
        sc4 = torch.maximum((-blks.amin(dim=1, keepdim=True)) / 8.0, blks.amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-7)
        w_int4 = (torch.clamp(torch.round(blks / sc4), -8.0, 7.0) * sc4).reshape(N, K)
        err_int4 = (X @ w_int4.T - Y_ref).norm().item() / ref_norm

        print(f"\n  [{name}]")
        print(f"    Naive Uniform 2-bit (2.25 bpw)        : {err_naive*100:6.2f}% output error")
        print(f"    Sub-2-Bit Lloyd-Max (2.25 bpw)        : {err_sub2*100:6.2f}% output error")
        print(f"    Sub-2-Bit + ASVD r=32 (~2.75 bpw)     : {err_asvd*100:6.2f}% output error")
        print(f"    Hadamard WHT + Sub-2-Bit (2.25 bpw)   : {err_wht*100:6.2f}% output error")
        print(f"    Hadamard WHT + GPTQ 2-Bit (2.25 bpw!) : {err_wht_gptq*100:6.2f}% output error")
        print(f"    Block-64 INT4 Reference (4.25 bpw)    : {err_int4*100:6.2f}% output error", flush=True)

    # Helper to evaluate end-to-end KL & Top-1 when ONLY a subset of modules/layers is 2-bit (with WHT!)
    def eval_end_to_end(desc: str, layer_range: range, suffixes: Tuple[str, ...], use_wht: bool = False) -> None:
        saved: List[Tuple[nn.Parameter, torch.Tensor]] = []
        torch.manual_seed(123)
        for l_i in layer_range:
            layer = model.model.layers[l_i]
            for sfx in suffixes:
                mod = getattr(layer.self_attn, sfx) if sfx in ("q_proj", "k_proj", "v_proj", "o_proj") else getattr(layer.mlp, sfx)
                saved.append((mod.weight, mod.weight.data.clone()))
                wf = mod.weight.detach().cpu().float()
                N, K = wf.shape
                if use_wht:
                    sg = torch.where(torch.rand(K) > 0.5, 1.0, -1.0)
                    wr = random_orthogonal_block_hadamard(wf, sg)
                    p2, s2 = quantize_sub2bit_block64(wr)
                    wq_r = dequantize_sub2bit_coalesced(p2, s2, (N, K), torch.float32)
                    # Inverse of diag(sg) * H_256: apply H_256 then * sg!
                    y = wq_r.reshape(-1, 256).clone()
                    h = 1
                    while h < 256:
                        y = y.view(-1, 256 // (2 * h), 2, h)
                        a = y[:, :, 0, :].clone()
                        b = y[:, :, 1, :].clone()
                        y[:, :, 0, :] = a + b
                        y[:, :, 1, :] = a - b
                        h *= 2
                    wq = (y.view(-1, 256) * (1.0 / 16.0)).view(N, K) * sg
                else:
                    p2, s2 = quantize_sub2bit_block64(wf)
                    wq = dequantize_sub2bit_coalesced(p2, s2, (N, K), torch.float32)
                mod.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))

        kl_sum = 0.0
        top1_match = 0
        tot_tok = 0
        for b_idx, b in enumerate(batches):
            lg = model(b, use_cache=False).logits[:, :-1, :].float().cpu()
            p_ref = F.softmax(ref_logits[b_idx], dim=-1)
            log_p_ref = F.log_softmax(ref_logits[b_idx], dim=-1)
            log_q = F.log_softmax(lg, dim=-1)
            kl = (p_ref * (log_p_ref - log_q)).sum(dim=-1)
            kl_sum += float(kl.sum().item())
            top1_match += int((lg.argmax(dim=-1) == ref_logits[b_idx].argmax(dim=-1)).sum().item())
            tot_tok += int(kl.numel())

        for param, orig in saved:
            param.data.copy_(orig)
        print(f"  {desc:46s} | KL={kl_sum/tot_tok:6.4f} | Top-1 Agreement={100.0*top1_match/tot_tok:5.1f}%", flush=True)

    print("\n=== EXP 3: MODULE-BY-MODULE ABLATION ACROSS ALL 36 LAYERS (WHICH PROJECTION BREAKS?) ===", flush=True)
    for sfx in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        eval_end_to_end(f"ONLY `{sfx}` in 2-bit (all 36L, plain)", range(36), (sfx,), use_wht=False)
    for sfx in ("gate_proj", "up_proj", "down_proj"):
        eval_end_to_end(f"ONLY `{sfx}` in 2-bit + Hadamard WHT (36L)", range(36), (sfx,), use_wht=True)

    print("\n=== EXP 4: DEPTH-BY-DEPTH ABLATION (WHERE ALONG LAYERS 0..35 IS NOISE AMPLIFIED?) ===", flush=True)
    for lbl, rng in [
        ("Early Layers L0..L5 (6 layers MLP 2-bit)", range(0, 6)),
        ("Early-Mid L6..L14 (9 layers MLP 2-bit)", range(6, 15)),
        ("Mid-Deep L15..L23 (9 layers MLP 2-bit)", range(15, 24)),
        ("Late Layers L24..L32 (9 layers MLP 2-bit)", range(24, 33)),
        ("Final Layers L33..L35 (3 layers MLP 2-bit)", range(33, 36)),
    ]:
        eval_end_to_end(lbl, rng, ("gate_proj", "up_proj", "down_proj"), use_wht=False)
        eval_end_to_end(lbl + " + WHT", rng, ("gate_proj", "up_proj", "down_proj"), use_wht=True)

    print(f"\n[Forensic Noise Lab] Completed in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
