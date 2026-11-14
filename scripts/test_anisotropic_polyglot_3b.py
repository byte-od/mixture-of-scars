"""Polyglot (PT, EN, ES, FR) Benchmark of Surgical Anisotropic MoS + Hadamard WHT + NF3/Sub-2-Bit on Qwen2.5-3B.

Tests 3 ultra-compact Zero-SGD configurations against FP16 across PT, EN, ES, and FR:
  1. `NF3 + Block-64 WHT` (`3.25 bpw` across all 36L + embed -> `1,196 MiB` / `1.17 GiB`)
  2. `Balanced Surgical 2/3-Bit MoS + WHT` (`1,068 MiB` / `1.04 GiB`)
  3. `Sub-1-GB Surgical 2/3-Bit MoS + WHT` (`946 MiB` / `0.924 GiB` — strictly `< 1 GB`!)
"""

from __future__ import annotations

import copy
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/juliocesarreisfilho/Projects/scar-cache")
from scar_cache.sub2bit_mos import dequantize_sub2bit_coalesced, quantize_sub2bit_block64

# Optimal symmetric 8-level Gaussian Lloyd-Max centroids normalized to outer level = 1.0:
# Raw N(0,1) levels: +-0.24509, +-0.75601, +-1.34391, +-2.15195 -> / 2.15195:
NF3_POS_LEVELS = (0.11389, 0.35131, 0.62451, 1.00000)
NF3_THRESHOLDS = (
    0.5 * (0.11389 + 0.35131),
    0.5 * (0.35131 + 0.62451),
    0.5 * (0.62451 + 1.00000),
)


def _rademacher_signs(k: int, device: torch.device) -> torch.Tensor:
    """Deterministic zero-storage Rademacher signs `s_j in {-1, +1}` from golden-ratio bit hash."""
    idx = torch.arange(k, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) * 0x45D9F3B & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


@torch.no_grad()
def fwht_block64(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Orthonormal Fast Walsh-Hadamard Transform on 64-element blocks with Rademacher pre-signs."""
    orig = x.shape
    y = x if inverse else (x * signs)
    y = y.reshape(-1, 64).clone()
    h = 1
    while h < 64:
        y = y.view(-1, 64 // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    out = (y.view(-1, 64) * 0.125).view(orig)
    return (out * signs) if inverse else out


@torch.no_grad()
def quantize_dequantize_nf3_wht(w_f32: torch.Tensor, use_wht: bool = True) -> Tuple[torch.Tensor, int]:
    """Symmetric 8-level Gaussian Lloyd-Max (`3.25 bpw` = `24B` codes + `2B` FP16 scale per 64 weights)."""
    N, K = w_f32.shape
    assert K % 64 == 0
    signs = _rademacher_signs(K, w_f32.device)
    wr = fwht_block64(w_f32, signs, inverse=False) if use_wht else w_f32
    w_blk = wr.reshape(-1, 64)

    # Optimal Gaussian outer centroid sits at `2.152 * rms`
    rms = w_blk.square().mean(dim=1, keepdim=True).sqrt()
    amax = w_blk.abs().amax(dim=1, keepdim=True)
    alpha = torch.minimum(amax, rms * 2.152).clamp(min=1e-7, max=65000.0)

    sgn = torch.where(w_blk >= 0.0, 1.0, -1.0)
    absw = w_blk.abs()
    t0, t1, t2 = NF3_THRESHOLDS
    l0, l1, l2, l3 = NF3_POS_LEVELS

    for _ in range(4):
        mag = torch.where(
            absw <= t0 * alpha,
            l0,
            torch.where(absw <= t1 * alpha, l1, torch.where(absw <= t2 * alpha, l2, l3)),
        )
        c = sgn * mag
        alpha = ((w_blk * c).sum(dim=1, keepdim=True) / c.square().sum(dim=1, keepdim=True).clamp(min=1e-12)).clamp(
            min=1e-7, max=65000.0
        )

    alpha_fp16 = alpha.to(torch.float16).float()
    mag_f = torch.where(
        absw <= t0 * alpha_fp16,
        l0,
        torch.where(absw <= t1 * alpha_fp16, l1, torch.where(absw <= t2 * alpha_fp16, l2, l3)),
    )
    wq_rot = (sgn * mag_f * alpha_fp16).reshape(N, K)
    wq = fwht_block64(wq_rot, signs, inverse=True) if use_wht else wq_rot
    n_blocks = (N * K) // 64
    phys_bytes = n_blocks * (24 + 2)  # 192 bits (24 B) + 1 FP16 scale (2 B) = 26 B / 64 weights
    return wq, phys_bytes


@torch.no_grad()
def quantize_dequantize_sub2bit_wht(
    w_f32: torch.Tensor, rank: int = 0
) -> Tuple[torch.Tensor, int]:
    """Sub-2-Bit (`2.25 bpw`) + Block-64 WHT (+ optional rank-r SVD scar before rotation)."""
    N, K = w_f32.shape
    scar_bytes = 0
    if rank > 0:
        u1, sv1, v1 = torch.svd_lowrank(w_f32, q=rank + 8, niter=2)
        b_s = (u1[:, :rank] * sv1[:rank].unsqueeze(0)).to(torch.float16)
        a_s = v1[:, :rank].T.to(torch.float16)
        w_res = w_f32 - (b_s.float() @ a_s.float())
        scar_bytes = (b_s.numel() + a_s.numel()) * 2
    else:
        w_res = w_f32
        b_s = a_s = None

    signs = _rademacher_signs(K, w_f32.device)
    wr = fwht_block64(w_res, signs, inverse=False)
    p2, s2 = quantize_sub2bit_block64(wr)
    wq_rot = dequantize_sub2bit_coalesced(p2, s2, (N, K), torch.float32)
    wq = fwht_block64(wq_rot, signs, inverse=True)
    if b_s is not None and a_s is not None:
        wq = wq + (b_s.float() @ a_s.float())
    n_blocks = (N * K) // 64
    phys_bytes = n_blocks * (16 + 2) + scar_bytes
    return wq, phys_bytes


@torch.no_grad()
def quantize_dequantize_int4(w_f32: torch.Tensor) -> Tuple[torch.Tensor, int]:
    N, K = w_f32.shape
    blks = w_f32.reshape(-1, 64)
    sc = torch.maximum((-blks.amin(dim=1, keepdim=True)) / 8.0, blks.amax(dim=1, keepdim=True) / 7.0).clamp(
        min=1e-7
    ).to(torch.float16)
    q = torch.clamp(torch.round(blks / sc.float()), -8.0, 7.0)
    wq = (q * sc.float()).reshape(N, K)
    return wq, ((N * K) // 64) * 34


POLYGLOT_EVAL_PASSAGES: Dict[str, str] = {
    "PT": (
        "<|im_start|>system\nResponda sempre no mesmo idioma do usuário com precisão técnica.<|im_end|>\n"
        "<|im_start|>user\nExplique o princípio de Landauer, a constante de Boltzmann e os avanços da computação quântica.<|im_end|>\n"
        "<|im_start|>assistant\nO princípio de Landauer estabelece que apagar irreversivelmente um bit de informação em um reservatório térmico à temperatura absoluta T dissipa no mínimo k_B T ln 2 de calor, onde k_B é a constante de Boltzmann que conecta temperatura e energia microscópica. Na computação quântica recente, portas lógicas unitárias reversíveis e códigos de correção de erros topológicos permitem manipular superposições coerentes com menor dissipação.<|im_end|>"
    ),
    "EN": (
        "<|im_start|>system\nAlways reply in the same language as the user with technical precision.<|im_end|>\n"
        "<|im_start|>user\nExplain Landauer's principle, Boltzmann's constant, and recent advances in quantum computing.<|im_end|>\n"
        "<|im_start|>assistant\nLandauer's principle states that irreversibly erasing one bit of information at absolute temperature T dissipates at least k_B T ln 2 of heat, where Boltzmann's constant k_B bridges macroscopic temperature and microscopic kinetic energy. Recent advances in fault-tolerant quantum computing leverage unitary reversible gates and surface error-correcting codes to preserve coherence across logical qubits.<|im_end|>"
    ),
    "ES": (
        "<|im_start|>system\nResponde siempre en el mismo idioma del usuario con precisión técnica.<|im_end|>\n"
        "<|im_start|>user\nExplica el principio de Landauer, la constante de Boltzmann y los avances recientes en computación cuántica.<|im_end|>\n"
        "<|im_start|>assistant\nEl principio de Landauer establece que borrar irreversiblemente un bit de información a una temperatura absoluta T disipa al menos k_B T ln 2 de calor, donde la constante de Boltzmann k_B relaciona la temperatura termodinámica con la energía térmica media. En la computación cuántica reciente, las compuertas unitarias reversibles y la corrección de errores cuánticos permiten operar qubits lógicos con alta fidelidad.<|im_end|>"
    ),
    "FR": (
        "<|im_start|>system\nRépondez toujours dans la même langue que l'utilisateur avec précision technique.<|im_end|>\n"
        "<|im_start|>user\nExpliquez le principe de Landauer, la constante de Boltzmann et les avancées récentes de l'informatique quantique.<|im_end|>\n"
        "<|im_start|>assistant\nLe principe de Landauer stipule que l'effacement irréversible d'un bit d'information à la température absolue T dissipe au minimum k_B T ln 2 de chaleur, où la constante de Boltzmann k_B relie la température macroscopique à l'énergie microscopique. Les avancées récentes en informatique quantique exploitent des portes unitaires réversibles et des codes correcteurs d'erreurs pour stabiliser les qubits logiques.<|im_end|>"
    ),
}

POLYGLOT_GEN_PROMPTS: List[Tuple[str, str]] = [
    ("PT", "Me fale sobre física quântica e a constante de Boltzmann."),
    ("EN", "Explain quantum physics and where Boltzmann's constant applies."),
    ("ES", "¿Qué es la física cuántica y dónde se aplica la constante de Boltzmann?"),
    ("FR", "Parlez-moi de la physique quantique et de la constante de Boltzmann."),
]

SYSTEM_PROMPT = (
    "You are System One, the official AI assistant for Julio Reis's portfolio (byte-od.github.io), "
    "running locally via WebGPU WGSL, MoS, and SCache O(1). Never claim to be Julio; refer to Julio Reis "
    "in the third person. Always reply in the same language as the user's message in 2 to 4 complete sentences."
)


@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Polyglot Surgical Lab] Loading {model_id} FP16 on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(dev)
    model.eval()

    # Snapshot clean FP16 state dict on CPU (6.1 GB RAM, allows instant clean resets between configs!)
    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Reference FP16 logits per language
    eval_ids = {lang: tok(txt, return_tensors="pt")["input_ids"].to(dev) for lang, txt in POLYGLOT_EVAL_PASSAGES.items()}
    ref_logits = {lang: model(ids, use_cache=False).logits[:, :-1, :].float().cpu() for lang, ids in eval_ids.items()}

    def restore_fp16() -> None:
        for k, v in model.state_dict().items():
            v.copy_(cpu_state[k].to(device=v.device, dtype=v.dtype))

    def evaluate_current_model(title: str, total_bytes: int) -> None:
        mib = total_bytes / (1024.0**2)
        gib = total_bytes / (1024.0**3)
        print(f"\n{'='*86}")
        print(f"CONFIG: {title} | SIZE = {mib:.2f} MiB ({gib:.3f} GiB)")
        print(f"{'='*86}", flush=True)
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
            out = model.generate(inp, max_new_tokens=115, do_sample=False)
            ans = tok.decode(out[0, inp.shape[1] :], skip_special_tokens=True).replace("\n", " ")
            print(f"  [{lang} GEN] {ans}", flush=True)

    # -------------------------------------------------------------------------
    # REGIME 1: Pure `NF3 (3-Bit Lloyd-Max) + Block-64 Hadamard WHT` (~1,196 MiB)
    # -------------------------------------------------------------------------
    t_r1 = time.perf_counter()
    b_r1 = 0
    wq_e, nb = quantize_dequantize_nf3_wht(cpu_state["model.embed_tokens.weight"].float(), use_wht=True)
    model.model.embed_tokens.weight.data.copy_(wq_e.to(device=dev, dtype=torch.float16))
    b_r1 += nb
    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].float()
            wq, nb = quantize_dequantize_nf3_wht(w_cpu, use_wht=True)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r1 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].float()
            wq, nb = quantize_dequantize_nf3_wht(w_cpu, use_wht=True)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r1 += nb
    print(f"\n(Built Regime 1 in {time.perf_counter() - t_r1:.1f}s)", flush=True)
    evaluate_current_model("Regime 1 — Pure 3-Bit NF3 + Hadamard WHT (All 36L)", b_r1)

    # -------------------------------------------------------------------------
    # REGIME 2: Sub-1-GB Surgical Anisotropic 2/3-Bit MoS + WHT (< 975 MiB = < 1.00 GB!)
    # -------------------------------------------------------------------------
    t_r2 = time.perf_counter()
    b_r2 = 0
    wq_e, nb = quantize_dequantize_nf3_wht(cpu_state["model.embed_tokens.weight"].float(), use_wht=True)
    model.model.embed_tokens.weight.data.copy_(wq_e.to(device=dev, dtype=torch.float16))
    b_r2 += nb
    for l_idx, layer in enumerate(model.model.layers):
        is_protected_depth = l_idx < 8 or l_idx >= 33
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].float()
            if is_protected_depth or sfx in ("k_proj", "v_proj", "o_proj"):
                wq, nb = quantize_dequantize_nf3_wht(w_cpu, use_wht=True)
            else:
                wq, nb = quantize_dequantize_sub2bit_wht(w_cpu, rank=0)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r2 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)

        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].float()
            if is_protected_depth:
                wq, nb = quantize_dequantize_nf3_wht(w_cpu, use_wht=True)
            elif sfx == "gate_proj":
                if 14 <= l_idx <= 27:
                    wq, nb = quantize_dequantize_sub2bit_wht(w_cpu, rank=16)
                else:
                    wq, nb = quantize_dequantize_nf3_wht(w_cpu, use_wht=True)
            elif sfx == "down_proj":
                wq, nb = quantize_dequantize_sub2bit_wht(w_cpu, rank=16)
            else:
                wq, nb = quantize_dequantize_sub2bit_wht(w_cpu, rank=0)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r2 += nb
    print(f"\n(Built Regime 2 in {time.perf_counter() - t_r2:.1f}s)", flush=True)
    evaluate_current_model("Regime 2 — Sub-1-GB Surgical 2/3-Bit MoS + WHT (< 1 GB GitHub Pages!)", b_r2)

    # -------------------------------------------------------------------------
    # REGIME 3: Pure Block-64 INT4 Baseline (`1,563.5 MiB`) for direct 4-language comparison!
    # -------------------------------------------------------------------------
    b_r3 = 0
    wq_e, nb = quantize_dequantize_int4(cpu_state["model.embed_tokens.weight"].float())
    model.model.embed_tokens.weight.data.copy_(wq_e.to(device=dev, dtype=torch.float16))
    b_r3 += nb
    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].float()
            wq, nb = quantize_dequantize_int4(w_cpu)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r3 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_cpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].float()
            wq, nb = quantize_dequantize_int4(w_cpu)
            lin.weight.data.copy_(wq.to(device=dev, dtype=torch.float16))
            b_r3 += nb
    evaluate_current_model("Regime 3 — Pure Block-64 INT4 Reference", b_r3)

    print(f"\n[Polyglot Surgical Lab] All 3 regimes x 4 languages completed in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
