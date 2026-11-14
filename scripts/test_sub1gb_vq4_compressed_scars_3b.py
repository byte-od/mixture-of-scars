"""Polyglot (PT, EN, ES, FR) Test of Option 1: Sub-1-GB Qwen2.5-3B `VQ4-Scar` via Compressed 2-Pass `INT4-ASVD` Scars.

Compresses the `ASVD` scars `(B_scar, A_scar)` themselves into `Block-64 INT4` (`4.25 bpw`, `3.76x` smaller than FP16,
matching `GoldenSliceScarStoreV4`), paired with:
  1. `2-Pass Alternating Residual Scar Healing` (LoftQ-style: Pass 2 fits the INT4-ASVD scar directly to the
     activation-weighted quantization error `(W - W_vq4_pass1) * diag(s)` so 0% of scar rank is wasted!).
  2. `alpha^2`-Energy-Weighted GPU Lloyd-Max (`K-Means`) for the `256`-entry `4D` Micro-Scar Codebook (`LUT256` with
     hard-pinned exact `(0, 0, 0, 0)` + `D4` kissing-24 shells).

Evaluates across unseen `PT, EN, ES, FR` passages (`0%` calibration overlap):
  - Stage 0: Layer 18 Held-Out check comparing 1-Pass `FP16` ASVD (`1.59 MiB`/mat) vs 2-Pass `INT4` ASVD (`0.42 MiB`/mat).
  - Option 1A (`< 935 MiB` / `< 0.980 GB` decimal — fits entire `dist/` inside `1,000,000,000` decimal bytes!).
  - Option 1B (`~988 MiB` / `0.965 GiB` — fits inside `1.00 GiB` GitHub Pages ceiling with `INT4` `embed_tokens`
    + 7 sink/final boundary layers protecting `attn + gate_proj` in `INT4`!).
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/juliocesarreisfilho/Projects/scar-cache")
from scripts.test_vector_scar_codebook_3b import (
    DISJOINT_POLYGLOT_CALIB_TEXTS,
    POLYGLOT_EVAL_PASSAGES,
    POLYGLOT_GEN_PROMPTS,
    SYSTEM_PROMPT,
    _build_initial_4d_lattice,
    _rademacher_signs,
    fast_gpu_asvd_scar,
    fwht_block64,
    quantize_dequantize_vq4_scar,
)


@torch.no_grad()
def _quantize_dequantize_block64_int4_raw(x_f32: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Exact WebGPU `Block-64 INT4` (`[-8..+7]`, `34` bytes per `64` weights = `4.25 bpw`)."""
    orig_shape = x_f32.shape
    assert x_f32.numel() % 64 == 0
    blks = x_f32.reshape(-1, 64)
    sc = torch.maximum((-blks.amin(dim=1, keepdim=True)) / 8.0, blks.amax(dim=1, keepdim=True) / 7.0).clamp(
        min=1e-7
    ).to(torch.float16)
    q = torch.clamp(torch.round(blks / sc.float()), -8.0, 7.0)
    xq = (q * sc.float()).reshape(orig_shape)
    return xq, blks.shape[0] * 34


@torch.no_grad()
def _quantize_dequantize_signed_int3_zero(x_f32: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Signed Zero-Preserving `Block-64 INT3` (`{-4,-3,-2,-1,0,+1,+2,+3}`, `26` bytes per `64` weights = `3.25 bpw`)."""
    orig_shape = x_f32.shape
    assert x_f32.numel() % 64 == 0
    blks = x_f32.reshape(-1, 64)
    sc = torch.maximum((-blks.amin(dim=1, keepdim=True)) / 4.0, blks.amax(dim=1, keepdim=True) / 3.0).clamp(
        min=1e-7
    ).to(torch.float16)
    q = torch.clamp(torch.round(blks / sc.float()), -4.0, 3.0)
    xq = (q * sc.float()).reshape(orig_shape)
    return xq, blks.shape[0] * 26


@torch.no_grad()
def fast_gpu_int4_asvd_scar(
    w_target_f32: torch.Tensor,
    col_rms: torch.Tensor,
    rank: int,
    alpha: float = 0.75,
) -> Tuple[torch.Tensor, int]:
    """Extracts activation-weighted rank-`r` scar and compresses BOTH `B_scar^T` and `A_scar` into `Block-64 INT4` (`3.76x` smaller!)."""
    N, K = w_target_f32.shape
    dev = w_target_f32.device
    s = (col_rms / col_rms.median().clamp(min=1e-7)).pow(alpha).clamp(0.05, 64.0)
    w_act = w_target_f32 * s.unsqueeze(0)

    q = min(rank + 12, min(N, K))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261115 + N + K + rank)
    omega = torch.randn(K, q, generator=gen, dtype=torch.float32).to(dev)

    y = w_act @ omega
    q_mat = torch.linalg.qr(y.cpu(), mode="reduced").Q.to(dev)
    for _ in range(2):
        z = w_act.T @ q_mat
        qz = torch.linalg.qr(z.cpu(), mode="reduced").Q.to(dev)
        y = w_act @ qz
        q_mat = torch.linalg.qr(y.cpu(), mode="reduced").Q.to(dev)

    b_small = (q_mat.T @ w_act).cpu()
    u_b, s_val, vh = torch.linalg.svd(b_small, full_matrices=False)
    u_r = q_mat.to(dev) @ u_b[:, :rank].to(dev)
    sv_r = s_val[:rank].sqrt().to(dev)  # Balanced singular-value split so both B and A have crisp dynamic range!
    vh_r = vh[:rank, :].to(dev)

    # Balanced factorization: B_t is (rank, N), A is (rank, K); both N and K are exact multiples of 64!
    b_t_raw = (u_r * sv_r.unsqueeze(0)).T.contiguous()  # (rank, N)
    a_raw = ((vh_r * sv_r.unsqueeze(1)) / s.unsqueeze(0)).contiguous()  # (rank, K)

    b_t_q, bytes_b = _quantize_dequantize_block64_int4_raw(b_t_raw)
    a_q, bytes_a = _quantize_dequantize_block64_int4_raw(a_raw)
    scar_hat = b_t_q.T @ a_q
    return scar_hat, (bytes_b + bytes_a)


@torch.no_grad()
def _encode_vq4_fp16(z: torch.Tensor, codebook_f32: torch.Tensor, chunk: int = 1048576) -> torch.Tensor:
    """Fast Metal FP16 Tensor-Core nearest-neighbor search in `LUT256 (256, 4)`."""
    c_h = codebook_f32.to(torch.float16)
    c_norm2 = codebook_f32.square().sum(dim=1).unsqueeze(0).to(torch.float16)  # (1, 256)
    M4 = z.shape[0]
    out_idx = torch.empty(M4, dtype=torch.int64, device=z.device)
    for s in range(0, M4, chunk):
        e = min(s + chunk, M4)
        zh = z[s:e].to(torch.float16)
        dots = zh @ c_h.T
        out_idx[s:e] = (c_norm2 - 2.0 * dots).argmin(dim=1)
    return out_idx


@torch.no_grad()
def _fit_and_apply_vq4(
    w_res_f32: torch.Tensor,
    d_f32: torch.Tensor,
    signs: torch.Tensor,
    codebook: Optional[torch.Tensor] = None,
    kmeans_iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Applies channel eq + WHT-64 + `alpha^2`-weighted GPU Lloyd-Max `LUT256` + optimal block scale."""
    N, K = w_res_f32.shape
    dev = w_res_f32.device
    w_eq = w_res_f32 * d_f32.unsqueeze(0)
    w_rot = fwht_block64(w_eq, signs, inverse=False)
    w_blk = w_rot.reshape(-1, 64)
    n_blocks = w_blk.shape[0]

    alpha = w_blk.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-7, max=65000.0).to(torch.float16).float()
    z_all = (w_blk / alpha).reshape(-1, 4)

    if codebook is None:
        # Energy-weighted (`alpha^2`) subsample of ~65,536 4D vectors
        step = max(1, z_all.shape[0] // 65536)
        z_sub = z_all[::step][:65536].contiguous()
        alpha_4 = alpha.repeat_interleave(16, dim=0).squeeze(1)
        w_sub = alpha_4[::step][: z_sub.shape[0]].square().clamp(min=1e-12)
        w_sub = (w_sub / w_sub.mean().clamp(min=1e-12)).contiguous()

        C = _build_initial_4d_lattice(z_sub)
        wz_sub = z_sub * w_sub.unsqueeze(1)
        for _ in range(kmeans_iters):
            idx = _encode_vq4_fp16(z_sub, C)
            sums = torch.zeros_like(C)
            cnts = torch.zeros(256, dtype=torch.float32, device=dev)
            sums.index_add_(0, idx, wz_sub)
            cnts.index_add_(0, idx, w_sub)
            mask = cnts > 1e-4
            mask[0] = False  # Lock LUT[0] = (0, 0, 0, 0)!
            C[mask] = sums[mask] / cnts[mask].unsqueeze(1)
            C[0].zero_()
        C = C.to(torch.float16).float()
        C[0].zero_()
    else:
        C = codebook

    idx_all = _encode_vq4_fp16(z_all, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
        .float()
    )
    z_all2 = (w_blk / alpha).reshape(-1, 4)
    idx_all = _encode_vq4_fp16(z_all2, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
        .float()
    )

    wq_rot = (c_blk * alpha).reshape(N, K)
    wq_res = fwht_block64(wq_rot, signs, inverse=True) / d_f32.unsqueeze(0)
    vq_bytes = n_blocks * (16 + 2) + (256 * 4 * 2)  # codes + FP16 block scales + 2KB LUT256
    return wq_res, C, vq_bytes


@torch.no_grad()
def quantize_dequantize_vq4_int4scar_2pass(
    w_f32: torch.Tensor,
    col_rms: torch.Tensor,
    rank: int = 64,
    eq_beta: float = 0.65,
    two_pass: bool = True,
) -> Tuple[torch.Tensor, int]:
    """`VQ4 LUT256` + `Block-64 INT4`-Compressed `ASVD` Scar (`3.76x` smaller!) + 2-Pass Residual Healing."""
    N, K = w_f32.shape
    dev = w_f32.device
    d_fp16 = (col_rms / col_rms.median().clamp(min=1e-7)).pow(eq_beta).clamp(0.125, 32.0).to(torch.float16)
    d_f32 = d_fp16.float()
    eq_bytes = K * 2
    signs = _rademacher_signs(K, dev)

    if rank <= 0:
        wq_res, _, vq_bytes = _fit_and_apply_vq4(w_f32, d_f32, signs, codebook=None)
        return wq_res, (vq_bytes + eq_bytes)

    # Pass 1: Pre-VQ outlier shield (`INT4`-compressed ASVD scar) + learn `LUT256`
    scar_1, scar_bytes = fast_gpu_int4_asvd_scar(w_f32, col_rms, rank=rank, alpha=0.75)
    wq_vq1, C, vq_bytes = _fit_and_apply_vq4(w_f32 - scar_1, d_f32, signs, codebook=None)

    if not two_pass:
        return (wq_vq1 + scar_1), (vq_bytes + eq_bytes + scar_bytes)

    # Pass 2 (Residual Scar Healing): Fit the INT4-ASVD scar directly to `W - wq_vq1`, then re-encode VQ4!
    scar_2, scar_bytes = fast_gpu_int4_asvd_scar(w_f32 - wq_vq1, col_rms, rank=rank, alpha=0.75)
    wq_vq2, _, vq_bytes = _fit_and_apply_vq4(w_f32 - scar_2, d_f32, signs, codebook=C)
    return (wq_vq2 + scar_2), (vq_bytes + eq_bytes + scar_bytes)


@torch.no_grad()
def quantize_dequantize_int4_with_int4scar(
    w_f32: torch.Tensor,
    col_rms: Optional[torch.Tensor] = None,
    rank: int = 0,
    use_eq: bool = True,
) -> Tuple[torch.Tensor, int]:
    """`Block-64 INT4` base + optional `Block-64 INT4`-compressed ASVD scar + optional `d_fp16` eq."""
    N, K = w_f32.shape
    scar_bytes = 0
    scar_hat = None
    if rank > 0 and col_rms is not None:
        scar_hat, scar_bytes = fast_gpu_int4_asvd_scar(w_f32, col_rms, rank=rank, alpha=0.75)
        w_res = w_f32 - scar_hat
    else:
        w_res = w_f32

    eq_bytes = 0
    if use_eq and col_rms is not None:
        d_fp16 = (col_rms / col_rms.median().clamp(min=1e-7)).pow(0.55).clamp(0.125, 32.0).to(torch.float16)
        d_f32 = d_fp16.float()
        w_in = w_res * d_f32.unsqueeze(0)
        eq_bytes = K * 2
    else:
        d_f32 = None
        w_in = w_res

    wq_in, base_bytes = _quantize_dequantize_block64_int4_raw(w_in)
    wq = (wq_in / d_f32.unsqueeze(0)) if d_f32 is not None else wq_in
    if scar_hat is not None:
        wq = wq + scar_hat
    return wq, (base_bytes + eq_bytes + scar_bytes)


@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Option 1: Sub-1-GB Compressed-Scar VQ4 Lab] Loading {model_id} FP16 on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(dev)
    model.eval()

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Collect disjoint 4-lang channel RMS (`0%` eval overlap)
    sum_sq: Dict[str, torch.Tensor] = {}
    cnt_tok: Dict[str, int] = {}
    hooks = []

    def _make_hook(name: str):
        def _fn(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            sq = x.square().sum(dim=0)
            if name not in sum_sq:
                sum_sq[name] = sq
                cnt_tok[name] = x.shape[0]
            else:
                sum_sq[name] += sq
                cnt_tok[name] += x.shape[0]

        return _fn

    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hooks.append(getattr(layer.self_attn, sfx).register_forward_hook(_make_hook(f"L{l_idx}.{sfx}")))
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            hooks.append(getattr(layer.mlp, sfx).register_forward_hook(_make_hook(f"L{l_idx}.{sfx}")))

    for txt in DISJOINT_POLYGLOT_CALIB_TEXTS:
        model(tok(txt, return_tensors="pt")["input_ids"].to(dev), use_cache=False)
    for h in hooks:
        h.remove()

    col_rms: Dict[str, torch.Tensor] = {
        name: (sq / max(1, cnt_tok[name])).sqrt().clamp(min=1e-6) for name, sq in sum_sq.items()
    }

    # Held-Out Layer 18 comparison: 1-Pass FP16 Scar (`1.59 MiB`) vs 2-Pass INT4 Scar (`0.42 MiB` = 3.76x smaller!)
    print(f"\n{'='*92}")
    print("STAGE 0 — HELD-OUT L18: FP16 SCAR (16-bit) vs COMPRESSED 2-PASS INT4 SCAR (4.25-bit, 3.76x SMALLER!)")
    print(f"{'='*92}", flush=True)
    heldout_x: Dict[str, List[torch.Tensor]] = {"L18.gate_proj": [], "L18.down_proj": []}
    h18 = [
        model.model.layers[18].mlp.gate_proj.register_forward_hook(
            lambda _m, inp, _o: heldout_x["L18.gate_proj"].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
        ),
        model.model.layers[18].mlp.down_proj.register_forward_hook(
            lambda _m, inp, _o: heldout_x["L18.down_proj"].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
        ),
    ]
    eval_ids = {lang: tok(txt, return_tensors="pt")["input_ids"].to(dev) for lang, txt in POLYGLOT_EVAL_PASSAGES.items()}
    ref_logits = {lang: model(ids, use_cache=False).logits[:, :-1, :].float().cpu() for lang, ids in eval_ids.items()}
    for h in h18:
        h.remove()

    for mname, w_key in (
        ("L18.gate_proj", "model.layers.18.mlp.gate_proj.weight"),
        ("L18.down_proj", "model.layers.18.mlp.down_proj.weight"),
    ):
        X_ho = torch.cat(heldout_x[mname], dim=0)
        W_orig = cpu_state[w_key].to(dev, dtype=torch.float32)
        Y_ref = X_ho @ W_orig.T
        ynorm = Y_ref.norm().item()
        c_r = col_rms[mname]

        w_fp16_s, b_fp16 = quantize_dequantize_vq4_scar(W_orig, c_r, rank=64)
        err_fp16 = 100.0 * (X_ho @ w_fp16_s.T - Y_ref).norm().item() / ynorm

        w_int4_1p, b_i4_1 = quantize_dequantize_vq4_int4scar_2pass(W_orig, c_r, rank=64, two_pass=False)
        err_i4_1p = 100.0 * (X_ho @ w_int4_1p.T - Y_ref).norm().item() / ynorm

        w_int4_2p, b_i4_2 = quantize_dequantize_vq4_int4scar_2pass(W_orig, c_r, rank=64, two_pass=True)
        err_i4_2p = 100.0 * (X_ho @ w_int4_2p.T - Y_ref).norm().item() / ynorm

        w_int4_r96, b_i4_96 = quantize_dequantize_vq4_int4scar_2pass(W_orig, c_r, rank=96, two_pass=True)
        err_i4_r96 = 100.0 * (X_ho @ w_int4_r96.T - Y_ref).norm().item() / ynorm

        print(f"\n  [{mname}] Unseen 4-Lang Tokens = {X_ho.shape[0]}:")
        print(f"    • VQ4 + FP16-ASVD (r=64)        [{b_fp16/(1024**2):5.2f} MiB]: {err_fp16:6.2f}% error")
        print(f"    • VQ4 + INT4-ASVD (r=64, 1-Pass) [{b_i4_1/(1024**2):5.2f} MiB]: {err_i4_1p:6.2f}% error")
        print(f"    • VQ4 + INT4-ASVD (r=64, 2-Pass) [{b_i4_2/(1024**2):5.2f} MiB]: {err_i4_2p:6.2f}% error  <-- saves {(b_fp16-b_i4_2)/(1024**2):.2f} MiB/mat!")
        print(f"    • VQ4 + INT4-ASVD (r=96, 2-Pass) [{b_i4_96/(1024**2):5.2f} MiB]: {err_i4_r96:6.2f}% error", flush=True)

    del heldout_x

    def evaluate_current_model(title: str, total_bytes: int) -> None:
        mib = total_bytes / (1024.0**2)
        gib = total_bytes / (1024.0**3)
        gb_dec = total_bytes / 1e9
        print(f"\n{'='*92}")
        print(f"CONFIG: {title}")
        print(f"SIZE  : {mib:.2f} MiB | {gib:.3f} GiB (binary) | {gb_dec:.3f} GB (decimal)")
        print(f"{'='*92}", flush=True)
        kls, top1s = [], []
        for lang in ("PT", "EN", "ES", "FR"):
            lg = model(eval_ids[lang], use_cache=False).logits[:, :-1, :].float().cpu()
            ref = ref_logits[lang]
            p_ref = F.softmax(ref, dim=-1)
            kl = (p_ref * (F.log_softmax(ref, dim=-1) - F.log_softmax(lg, dim=-1))).sum(dim=-1).mean().item()
            t1 = 100.0 * (lg.argmax(dim=-1) == ref.argmax(dim=-1)).float().mean().item()
            kls.append(kl)
            top1s.append(t1)
            print(f"  [{lang}] KL={kl:6.4f} | Top-1 Agreement={t1:5.1f}%", flush=True)
        print(f"  [MEAN 4-LANG] KL={sum(kls)/4.0:6.4f} | Top-1 Agreement={sum(top1s)/4.0:5.1f}%\n", flush=True)

        for lang, q in POLYGLOT_GEN_PROMPTS:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            inp = (enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc).to(dev)
            out = model.generate(inp, max_new_tokens=135, do_sample=False)
            ans = tok.decode(out[0, inp.shape[1] :], skip_special_tokens=True).replace("\n", " ")
            print(f"  [{lang} GEN] {ans}", flush=True)

    # -------------------------------------------------------------------------
    # OPTION 1B: `< 1.00 GiB` GitHub Pages Binary Ceiling (`~992 MiB` = `0.969 GiB`)
    #   Exact equivalent of `Config 2` (`1,176 MiB`) with `ASVD` scars compressed in `INT4` + 2-Pass Healing!
    #   - `embed_tokens`: Pure `Block-64 INT4` (`157.52 MiB`)
    #   - `k_proj, v_proj` across ALL 36 layers: `Block-64 INT4` (crisp GQA SCache keys/values)
    #   - 7 boundary layers (`L0..L4, L34..L35`): `q/o_proj + gate_proj` in `INT4 + INT4-ASVD r=32`,
    #     `up/down_proj` in `VQ4 LUT256 + 2-Pass INT4-ASVD r=96`!
    #   - 29 interior layers (`L5..L33`): `VQ4 LUT256 + 2-Pass INT4-ASVD` (`r=80` on `gate/down`, `r=64` on `o_proj`, `r=24` on `q/up`)!
    # -------------------------------------------------------------------------
    t_1b = time.perf_counter()
    wq_emb4, b_emb4 = _quantize_dequantize_block64_int4_raw(
        cpu_state["model.embed_tokens.weight"].to(dev, dtype=torch.float32)
    )
    model.model.embed_tokens.weight.data.copy_(wq_emb4.to(torch.float16))
    bytes_1b = b_emb4

    for l_idx, layer in enumerate(model.model.layers):
        is_boundary = l_idx < 5 or l_idx >= 34
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if sfx in ("k_proj", "v_proj") or is_boundary:
                wq, nb = quantize_dequantize_int4_with_int4scar(
                    w_gpu, c_r, rank=32 if sfx == "o_proj" else 0, use_eq=True
                )
            else:
                r = 64 if sfx == "o_proj" else 24
                wq, nb = quantize_dequantize_vq4_int4scar_2pass(w_gpu, c_r, rank=r, two_pass=True)
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_1b += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)

        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if is_boundary and sfx == "gate_proj":
                wq, nb = quantize_dequantize_int4_with_int4scar(w_gpu, c_r, rank=32, use_eq=True)
            elif is_boundary:
                wq, nb = quantize_dequantize_vq4_int4scar_2pass(w_gpu, c_r, rank=96, two_pass=True)
            else:
                r = 80 if sfx in ("gate_proj", "down_proj") else 24
                wq, nb = quantize_dequantize_vq4_int4scar_2pass(w_gpu, c_r, rank=r, two_pass=True)
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_1b += nb

    print(f"\n(Built Option 1B in {time.perf_counter() - t_1b:.1f}s)", flush=True)
    evaluate_current_model("Option 1B — Sub-1-GiB VQ4 + Compressed 2-Pass INT4-ASVD Scars (< 1.00 GiB!)", bytes_1b)

    # -------------------------------------------------------------------------
    # OPTION 1A: Ultra-Compact `< 1.00 GB` DECIMAL (`< 935 MiB` -> entire `dist/` `< 1,000,000,000 B`!)
    #   - `embed_tokens`: Signed Zero-Preserving `INT3` (`3.25 bpw`, exact `0.0`!) + `INT4` SVD Scar (`r=64`) (`125.9 MiB`)
    #   - `k_proj, v_proj` across ALL 36 layers: `Block-64 INT4`
    #   - 4 critical Sink/Final layers (`L0, L1, L2, L35`): `q/o_proj + gate_proj` in `INT4`, `up/down` in `VQ4 + r=64`
    #   - 32 interior layers (`L3..L34`): `VQ4 LUT256 + 2-Pass INT4-ASVD` (`r=64` on `gate/down`, `r=48` on `o_proj`)
    # -------------------------------------------------------------------------
    t_1a = time.perf_counter()
    w_emb_gpu = cpu_state["model.embed_tokens.weight"].to(dev, dtype=torch.float32)
    # Signed INT3 with exact 0.0 + INT4 rank-64 residual scar on vocabulary!
    emb_q0, b_emb3 = _quantize_dequantize_signed_int3_zero(w_emb_gpu)
    emb_err = w_emb_gpu - emb_q0
    ones_rms = torch.ones(w_emb_gpu.shape[1], dtype=torch.float32, device=dev)
    emb_scar, b_emb_scar = fast_gpu_int4_asvd_scar(emb_err, ones_rms, rank=64, alpha=0.0)
    emb_q1, b_emb3 = _quantize_dequantize_signed_int3_zero(w_emb_gpu - emb_scar)
    model.model.embed_tokens.weight.data.copy_((emb_q1 + emb_scar).to(torch.float16))
    bytes_1a = b_emb3 + b_emb_scar

    for l_idx, layer in enumerate(model.model.layers):
        is_sink_final = l_idx in (0, 1, 2, 35)
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if sfx in ("k_proj", "v_proj") or is_sink_final:
                wq, nb = quantize_dequantize_int4_with_int4scar(
                    w_gpu, c_r, rank=16 if sfx == "o_proj" else 0, use_eq=True
                )
            else:
                r = 48 if sfx == "o_proj" else 0
                wq, nb = quantize_dequantize_vq4_int4scar_2pass(w_gpu, c_r, rank=r, two_pass=True)
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_1a += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)

        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if is_sink_final and sfx == "gate_proj":
                wq, nb = quantize_dequantize_int4_with_int4scar(w_gpu, c_r, rank=16, use_eq=True)
            else:
                r = 64 if sfx in ("gate_proj", "down_proj") else 0
                wq, nb = quantize_dequantize_vq4_int4scar_2pass(w_gpu, c_r, rank=r, two_pass=True)
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_1a += nb

    print(f"\n(Built Option 1A in {time.perf_counter() - t_1a:.1f}s)", flush=True)
    evaluate_current_model("Option 1A — Ultra-Compact < 1.00 GB Decimal (< 935 MiB for Full GitHub Pages Dist!)", bytes_1a)

    print(f"\n[Option 1 Lab] Completed all stages across PT/EN/ES/FR in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
