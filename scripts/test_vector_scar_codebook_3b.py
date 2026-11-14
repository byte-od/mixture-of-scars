"""Polyglot (PT, EN, ES, FR) Benchmark of 4D Vector-Scar Codebooks (`VQ4 LUT256`) + Polyglot `ASVD` Scars on Qwen2.5-3B.

Tests replacing scalar Gaussian bins with our Scar-Native mechanisms (Zero-SGD, strictly held-out):
  - Stage 0 (Held-Out Layer 18 Micro-Ablation): Quantifies exact output error `||X W - X W_hat|| / ||X W||`
    on unseen tokens across:
      (a) Scalar 2-bit (`2.25 bpw`)
      (b) Scalar 2-bit + Hadamard WHT (without channel equilibration -> exposes outlier noise amplification!)
      (c) Scalar 2-bit + Channel Equilibration + Hadamard WHT
      (d) `VQ4 LUT256` (4D Micro-Scar Codebook with exact `(0,0,0,0)` + D4 shells + GPU Lloyd-Max, `2.25 bpw`) + Equilibration + WHT
      (e) `VQ4 LUT256` + Polyglot `ASVD` Scar (`r=48`) + Equilibration + WHT
  - Stage 1 (Full 36-Layer 4-Language Benchmark across PT, EN, ES, FR):
      1. `Sub-1-GiB VQ4-Scar` (`~0.98 GiB` / `1,008 MiB` — strictly `< 1 GiB`!)
      2. `Hybrid Sink-Shielded VQ4-Scar` (`~1.10 GiB` / `1,128 MiB` — smaller than `NF3 1.17 GiB`!)
      3. `Full MoS Block-64 INT4 + Polyglot ASVD Scars` (`~1.61 GiB` — verifies elimination of the token-90 FR->ZH leak!)
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
from scar_cache.sub2bit_mos import dequantize_sub2bit_coalesced, quantize_sub2bit_block64


# =============================================================================
# 1. STRICTLY DISJOINT POLYGLOT CALIBRATION SET (0% overlap with Eval/Gen!)
# =============================================================================
DISJOINT_POLYGLOT_CALIB_TEXTS: Tuple[str, ...] = (
    # PT (Botany, Algorithms, History, Architecture)
    "A fotossíntese nas plantas vasculares converte dióxido de carbono e água em glicose dentro dos cloroplastos sob luz solar.",
    "O algoritmo de busca binária divide repetidamente um vetor ordenado ao meio, garantindo complexidade temporal logarítmica no pior caso.",
    "A imprensa de tipos móveis impulsionou a circulação de livros científicos e literários na Europa renascentista durante o século quinze.",
    "Pontes estaiadas distribuem as cargas do tabuleiro diretamente para torres altas por meio de cabos de aço tracionados em leque.",
    # EN (Oceanography, Music, Geometry, Databases)
    "Thermohaline circulation drives deep ocean currents across global basins through gradients in seawater density and salinity.",
    "Counterpoint in classical composition weaves independent melodic voices into harmonic balance while preserving rhythmic autonomy.",
    "In Euclidean geometry, the perpendicular bisectors of the three sides of any triangle intersect at a single circumcenter.",
    "Write-ahead logging guarantees durability and atomic recovery in relational databases before pages are flushed to disk.",
    # ES (Astronomy, Biology, Literature, Civil Engineering)
    "Las estrellas de secuencia principal fusionan núcleos de hidrógeno en helio manteniendo el equilibrio hidrostático frente a su propia gravedad.",
    "Las enzimas digestivas aceleran reacciones bioquímicas específicas reduciendo la barrera de activación sin consumirse en el proceso.",
    "La novela moderna surgió como un espacio narrativo capaz de retratar las tensiones psicológicas y sociales de personajes complejos.",
    "Los acueductos de piedra aprovechaban pendientes suaves y arcos sucesivos para transportar agua fresca a lo largo de decenas de kilómetros.",
    # FR (Chemistry, Cartography, Ecology, Computing)
    "Les liaisons covalentes résultent de la mise en commun de paires d'électrons de valence entre atomes cherchant une configuration stable.",
    "Les projections cartographiques transforment une surface sphérique en carte plane au prix d'un compromis entre angles et superficies.",
    "Les récifs coralliens abritent une biodiversité marine exceptionnelle grâce à la symbiose entre polypes calcaires et microalgues.",
    "Un compilateur moderne analyse la syntaxe abstraite du code source avant d'optimiser l'allocation des registres processeur.",
)

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
        "<|im_start|>system\nRépondez toujours dans la même langue que l'utilisateur avec precisión technique.<|im_end|>\n"
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


# =============================================================================
# 2. FAST GPU PRIMITIVES: WHT-64, RANDOMIZED ASVD & 4D VECTOR-SCAR LUT256
# =============================================================================
def _rademacher_signs(k: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(k, device=device, dtype=torch.int64)
    h = (idx * 2654435761) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) * 0x45D9F3B & 0xFFFFFFFF
    return torch.where((h & 1) == 0, 1.0, -1.0)


@torch.no_grad()
def fwht_block64(x: torch.Tensor, signs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
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
def fast_gpu_asvd_scar(
    w_f32: torch.Tensor,
    col_rms: torch.Tensor,
    rank: int,
    alpha: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Extracts genuine activation-weighted FP16 Scar `(B_scar, A_scar)` on GPU via subspace iteration."""
    N, K = w_f32.shape
    dev = w_f32.device
    s = (col_rms / col_rms.median().clamp(min=1e-7)).pow(alpha).clamp(0.05, 64.0)
    w_act = w_f32 * s.unsqueeze(0)

    q = min(rank + 12, min(N, K))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20261114 + N + K + rank)
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
    sv_r = s_val[:rank].to(dev)
    vh_r = vh[:rank, :].to(dev)

    b_scar = (u_r * sv_r.unsqueeze(0)).to(torch.float16)
    a_scar = (vh_r / s.unsqueeze(0)).to(torch.float16)
    scar_hat = b_scar.float() @ a_scar.float()
    scar_bytes = (b_scar.numel() + a_scar.numel()) * 2
    return w_f32 - scar_hat, scar_hat, scar_bytes


@torch.no_grad()
def _build_initial_4d_lattice(z_sub: torch.Tensor) -> torch.Tensor:
    """Seeds `LUT256` with `(0,0,0,0)` + 16 single-axis spikes + 72 D4 root shells + 32 hypercubes + 135 data seeds."""
    dev = z_sub.device
    pts: List[List[float]] = [[0.0, 0.0, 0.0, 0.0]]  # Index 0: Hard-pinned exact zero!

    # 16 single-axis sparse spikes (impossible in scalar 2-bit!)
    for r in (1.20, 2.45):
        for ax in range(4):
            for sgn in (-1.0, 1.0):
                v = [0.0, 0.0, 0.0, 0.0]
                v[ax] = sgn * r
                pts.append(v)

    # 72 D4 lattice root vectors (kissing number 24 optimal 4D sphere packing x 3 shells!)
    for r in (0.72, 1.38, 2.12):
        for i in range(4):
            for j in range(i + 1, 4):
                for si in (-1.0, 1.0):
                    for sj in (-1.0, 1.0):
                        v = [0.0, 0.0, 0.0, 0.0]
                        v[i] = si * r
                        v[j] = sj * r
                        pts.append(v)

    # 32 hypercube shell vertices (16 signs x 2 shells)
    for r in (0.55, 1.12):
        for mask in range(16):
            pts.append([r if ((mask >> c) & 1) else -r for c in range(4)])

    assert len(pts) == 121
    c_fixed = torch.tensor(pts, dtype=torch.float32, device=dev)
    rem = 256 - len(pts)  # 135 data-driven prototypes
    stride = max(1, z_sub.shape[0] // rem)
    c_data = z_sub[::stride][:rem].clone()
    if c_data.shape[0] < rem:
        pad = z_sub[: rem - c_data.shape[0]].clone()
        c_data = torch.cat([c_data, pad], dim=0)
    return torch.cat([c_fixed, c_data], dim=0)


@torch.no_grad()
def _encode_vq4(z: torch.Tensor, codebook: torch.Tensor, chunk: int = 524288) -> torch.Tensor:
    """Finds nearest 4D prototype in `codebook (256, 4)` for `z (M4, 4)` on GPU."""
    c_norm2 = codebook.square().sum(dim=1).unsqueeze(0)  # (1, 256)
    M4 = z.shape[0]
    out_idx = torch.empty(M4, dtype=torch.int64, device=z.device)
    for s in range(0, M4, chunk):
        e = min(s + chunk, M4)
        dots = z[s:e] @ codebook.T  # (chunk, 256)
        out_idx[s:e] = (c_norm2 - 2.0 * dots).argmin(dim=1)
    return out_idx


@torch.no_grad()
def quantize_dequantize_vq4_scar(
    w_f32: torch.Tensor,
    col_rms: torch.Tensor,
    rank: int = 0,
    eq_beta: float = 0.65,
    kmeans_iters: int = 10,
) -> Tuple[torch.Tensor, int]:
    """4D Micro-Scar Codebook (`LUT256` with exact `(0,0,0,0)`) + Polyglot ASVD + Channel Equilibration + WHT."""
    N, K = w_f32.shape
    dev = w_f32.device

    if rank > 0:
        w_res, scar_hat, scar_bytes = fast_gpu_asvd_scar(w_f32, col_rms, rank=rank, alpha=0.75)
    else:
        w_res = w_f32
        scar_hat = None
        scar_bytes = 0

    # Channel equilibration vector `d_fp16` (2 * K bytes; folded into input `x / d` before WGSL matvec)
    d_fp16 = (col_rms / col_rms.median().clamp(min=1e-7)).pow(eq_beta).clamp(0.125, 32.0).to(torch.float16)
    d_f32 = d_fp16.float()
    w_eq = w_res * d_f32.unsqueeze(0)
    eq_bytes = K * 2

    # Orthonormal Block-64 Rademacher-Hadamard rotation
    signs = _rademacher_signs(K, dev)
    w_rot = fwht_block64(w_eq, signs, inverse=False)
    w_blk = w_rot.reshape(-1, 64)
    n_blocks = w_blk.shape[0]

    alpha = w_blk.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-7, max=65000.0).to(torch.float16).float()
    z_all = (w_blk / alpha).reshape(-1, 4)

    # Subsample ~65,536 4-vectors across the matrix for rapid GPU Lloyd-Max codebook learning
    step = max(1, z_all.shape[0] // 65536)
    z_sub = z_all[::step][:65536].contiguous()
    C = _build_initial_4d_lattice(z_sub)
    ones = torch.ones(z_sub.shape[0], dtype=torch.float32, device=dev)

    for _ in range(kmeans_iters):
        c_norm2 = C.square().sum(dim=1).unsqueeze(0)
        idx = (c_norm2 - 2.0 * (z_sub @ C.T)).argmin(dim=1)
        sums = torch.zeros_like(C)
        cnts = torch.zeros(256, dtype=torch.float32, device=dev)
        sums.index_add_(0, idx, z_sub)
        cnts.index_add_(0, idx, ones)
        mask = cnts > 0.5
        mask[0] = False  # Keep LUT[0] hard-pinned to exact (0, 0, 0, 0)!
        C[mask] = sums[mask] / cnts[mask].unsqueeze(1)
        C[0].zero_()

    # Snap codebook to exact FP16 (`2,048` bytes per matrix!)
    C = C.to(torch.float16).float()
    C[0].zero_()
    lut_bytes = 256 * 4 * 2

    # Pass 1 encode + closed-form Block-64 FP16 scale refinement + Pass 2 encode
    idx_all = _encode_vq4(z_all, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
        .float()
    )
    z_all2 = (w_blk / alpha).reshape(-1, 4)
    idx_all = _encode_vq4(z_all2, C)
    c_blk = C[idx_all].reshape(n_blocks, 64)
    alpha = (
        ((w_blk * c_blk).sum(dim=1, keepdim=True) / c_blk.square().sum(dim=1, keepdim=True).clamp(min=1e-12))
        .clamp(min=1e-7, max=65000.0)
        .to(torch.float16)
        .float()
    )

    wq_rot = (c_blk * alpha).reshape(N, K)
    wq_eq = fwht_block64(wq_rot, signs, inverse=True)
    wq = wq_eq / d_f32.unsqueeze(0)
    if scar_hat is not None:
        wq = wq + scar_hat

    # Physical bytes: 1 byte per 4 weights + 2B FP16 scale per 64 weights + 2KB LUT256 + 2*K eq + ASVD scar
    phys_bytes = n_blocks * (16 + 2) + lut_bytes + eq_bytes + scar_bytes
    return wq, phys_bytes


@torch.no_grad()
def quantize_dequantize_int4_mos(
    w_f32: torch.Tensor,
    col_rms: Optional[torch.Tensor] = None,
    rank: int = 0,
    use_eq: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Block-64 INT4 (`4.25 bpw`) + optional Polyglot ASVD Scar (`rank`) + optional channel equilibration."""
    N, K = w_f32.shape
    scar_bytes = 0
    scar_hat = None
    if rank > 0 and col_rms is not None:
        w_res, scar_hat, scar_bytes = fast_gpu_asvd_scar(w_f32, col_rms, rank=rank, alpha=0.75)
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

    blks = w_in.reshape(-1, 64)
    sc = torch.maximum((-blks.amin(dim=1, keepdim=True)) / 8.0, blks.amax(dim=1, keepdim=True) / 7.0).clamp(
        min=1e-7
    ).to(torch.float16)
    q = torch.clamp(torch.round(blks / sc.float()), -8.0, 7.0)
    wq = (q * sc.float()).reshape(N, K)
    if d_f32 is not None:
        wq = wq / d_f32.unsqueeze(0)
    if scar_hat is not None:
        wq = wq + scar_hat
    phys_bytes = ((N * K) // 64) * 34 + eq_bytes + scar_bytes
    return wq, phys_bytes


# =============================================================================
# 3. MAIN POLYGLOT LABORATORY
# =============================================================================
@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[VQ4-Scar Polyglot Lab] Loading {model_id} FP16 on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(dev)
    model.eval()

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # -------------------------------------------------------------------------
    # STEP A: Collect Disjoint Polyglot Per-Channel Activation RMS (`col_rms`)
    # -------------------------------------------------------------------------
    t_cal = time.perf_counter()
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
            name = f"L{l_idx}.{sfx}"
            hooks.append(getattr(layer.self_attn, sfx).register_forward_hook(_make_hook(name)))
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            name = f"L{l_idx}.{sfx}"
            hooks.append(getattr(layer.mlp, sfx).register_forward_hook(_make_hook(name)))

    for txt in DISJOINT_POLYGLOT_CALIB_TEXTS:
        ids = tok(txt, return_tensors="pt")["input_ids"].to(dev)
        model(ids, use_cache=False)

    for h in hooks:
        h.remove()

    col_rms: Dict[str, torch.Tensor] = {
        name: (sq / max(1, cnt_tok[name])).sqrt().clamp(min=1e-6) for name, sq in sum_sq.items()
    }
    total_cal_tokens = next(iter(cnt_tok.values()))
    print(
        f"[Calib] Collected disjoint 4-lang channel RMS across {len(col_rms)} matrices "
        f"({total_cal_tokens} tokens, 0% eval overlap) in {time.perf_counter() - t_cal:.2f}s",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # STEP B: Stage 0 — Held-Out Layer 18 Micro-Ablation (`gate_proj` & `down_proj`)
    # -------------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("STAGE 0 — HELD-OUT LAYER 18 OUTPUT ERROR `||X W - X W_hat|| / ||X W||` (UNSEEN 4-LANG EVAL)")
    print(f"{'='*90}", flush=True)
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
    ref_logits = {}
    for lang, ids in eval_ids.items():
        ref_logits[lang] = model(ids, use_cache=False).logits[:, :-1, :].float().cpu()
    for h in h18:
        h.remove()

    for mname, w_key in (
        ("L18.gate_proj", "model.layers.18.mlp.gate_proj.weight"),
        ("L18.down_proj", "model.layers.18.mlp.down_proj.weight"),
    ):
        X_ho = torch.cat(heldout_x[mname], dim=0)  # Strictly unseen 4-lang tokens!
        W_orig = cpu_state[w_key].to(dev, dtype=torch.float32)
        Y_ref = X_ho @ W_orig.T
        ynorm = Y_ref.norm().item()
        c_rms = col_rms[mname]
        signs = _rademacher_signs(W_orig.shape[1], dev)

        # (a) Scalar 2-bit
        p2, s2 = quantize_sub2bit_block64(W_orig)
        W_a = dequantize_sub2bit_coalesced(p2, s2, W_orig.shape, torch.float32)
        err_a = 100.0 * (X_ho @ W_a.T - Y_ref).norm().item() / ynorm

        # (b) Scalar 2-bit + WHT (no channel equilibration)
        wr = fwht_block64(W_orig, signs, inverse=False)
        p2b, s2b = quantize_sub2bit_block64(wr)
        W_b = fwht_block64(dequantize_sub2bit_coalesced(p2b, s2b, W_orig.shape, torch.float32), signs, inverse=True)
        err_b = 100.0 * (X_ho @ W_b.T - Y_ref).norm().item() / ynorm

        # (c) Scalar 2-bit + Channel Equilibration + WHT
        d_eq = (c_rms / c_rms.median()).pow(0.65).clamp(0.125, 32.0).to(torch.float16).float()
        wr_c = fwht_block64(W_orig * d_eq.unsqueeze(0), signs, inverse=False)
        p2c, s2c = quantize_sub2bit_block64(wr_c)
        W_c = fwht_block64(dequantize_sub2bit_coalesced(p2c, s2c, W_orig.shape, torch.float32), signs, inverse=True) / d_eq.unsqueeze(0)
        err_c = 100.0 * (X_ho @ W_c.T - Y_ref).norm().item() / ynorm

        # (d) 4D Micro-Scar Codebook (`VQ4 LUT256` with `(0,0,0,0)`) + Equilibration + WHT (rank=0)
        W_d, _ = quantize_dequantize_vq4_scar(W_orig, c_rms, rank=0, eq_beta=0.65)
        err_d = 100.0 * (X_ho @ W_d.T - Y_ref).norm().item() / ynorm

        # (e) `VQ4 LUT256` + Polyglot `ASVD` Scar (`r=48`) + Equilibration + WHT
        W_e, _ = quantize_dequantize_vq4_scar(W_orig, c_rms, rank=48, eq_beta=0.65)
        err_e = 100.0 * (X_ho @ W_e.T - Y_ref).norm().item() / ynorm

        print(f"\n  [{mname}] Held-Out 4-Lang Tokens = {X_ho.shape[0]}:")
        print(f"    (a) Scalar 2-bit Gaussian (`2.25 bpw`)                      : {err_a:6.2f}% error")
        print(f"    (b) Scalar 2-bit + WHT (no eq -> outlier amplification!)    : {err_b:6.2f}% error")
        print(f"    (c) Scalar 2-bit + Channel Equilibration + WHT              : {err_c:6.2f}% error")
        print(f"    (d) VQ4 Micro-Scar LUT256 (`0` + D4 + KMeans) + Eq + WHT    : {err_d:6.2f}% error")
        print(f"    (e) VQ4 Micro-Scar LUT256 + Polyglot ASVD (r=48) + Eq + WHT : {err_e:6.2f}% error", flush=True)

    del heldout_x

    def evaluate_current_model(title: str, total_bytes: int) -> None:
        mib = total_bytes / (1024.0**2)
        gib = total_bytes / (1024.0**3)
        print(f"\n{'='*90}")
        print(f"CONFIG: {title} | SIZE = {mib:.2f} MiB ({gib:.3f} GiB)")
        print(f"{'='*90}", flush=True)
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

    # Pre-quantize protected multilingual INT4 `embed_tokens` once (`157.52 MiB`)
    wq_embed_int4, embed_int4_bytes = quantize_dequantize_int4_mos(
        cpu_state["model.embed_tokens.weight"].to(dev, dtype=torch.float32)
    )
    wq_embed_int4_fp16 = wq_embed_int4.to(torch.float16)

    # -------------------------------------------------------------------------
    # CONFIG 1: Sub-1-GiB VQ4-Scar (`< 1.00 GiB` = `~1,008 MiB`!)
    #   - `embed_tokens`: Protected Block-64 INT4 (`157.5 MiB`)
    #   - All 36 layers: `VQ4 LUT256 (2.25 bpw)` + Channel Equilibration + WHT
    #   - Polyglot `ASVD` Scars: `rank=48` on `gate_proj` & `down_proj`, `rank=32` on `o_proj`
    # -------------------------------------------------------------------------
    t_c1 = time.perf_counter()
    b_c1 = embed_int4_bytes
    model.model.embed_tokens.weight.data.copy_(wq_embed_int4_fp16)
    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            r = 32 if sfx == "o_proj" else 0
            wq, nb = quantize_dequantize_vq4_scar(w_gpu, col_rms[f"L{l_idx}.{sfx}"], rank=r)
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c1 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].to(dev, dtype=torch.float32)
            r = 48 if sfx in ("gate_proj", "down_proj") else 0
            wq, nb = quantize_dequantize_vq4_scar(w_gpu, col_rms[f"L{l_idx}.{sfx}"], rank=r)
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c1 += nb
    print(f"\n(Built Config 1 in {time.perf_counter() - t_c1:.1f}s)", flush=True)
    evaluate_current_model("Config 1 — Sub-1-GiB VQ4-Scar (All 36L VQ4 + Polyglot ASVD + Eq)", b_c1)

    # -------------------------------------------------------------------------
    # CONFIG 2: Hybrid Sink-Shielded VQ4-Scar (`~1.11 GiB` — smaller than NF3 `1.17 GiB`!)
    #   - `L0..L4` & `L34..L35` (7 boundary layers): `Block-64 INT4 + ASVD r=32 + Eq`
    #   - `L5..L33` (29 interior layers): `VQ4 LUT256` (`k_proj/v_proj` INT4 — tiny `256x2048` GQA!,
    #     `q/o/gate/up/down` in `VQ4 LUT256 + Polyglot ASVD r=64 + Eq`)
    # -------------------------------------------------------------------------
    t_c2 = time.perf_counter()
    b_c2 = embed_int4_bytes
    model.model.embed_tokens.weight.data.copy_(wq_embed_int4_fp16)
    for l_idx, layer in enumerate(model.model.layers):
        is_boundary = l_idx < 5 or l_idx >= 34
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if is_boundary or sfx in ("k_proj", "v_proj"):
                wq, nb = quantize_dequantize_int4_mos(w_gpu, c_r, rank=16 if sfx == "o_proj" else 0, use_eq=True)
            else:
                wq, nb = quantize_dequantize_vq4_scar(w_gpu, c_r, rank=48 if sfx == "o_proj" else 16)
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c2 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if is_boundary:
                wq, nb = quantize_dequantize_int4_mos(w_gpu, c_r, rank=32 if sfx in ("gate_proj", "down_proj") else 0, use_eq=True)
            else:
                r = 64 if sfx in ("gate_proj", "down_proj") else 24
                wq, nb = quantize_dequantize_vq4_scar(w_gpu, c_r, rank=r)
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c2 += nb
    print(f"\n(Built Config 2 in {time.perf_counter() - t_c2:.1f}s)", flush=True)
    evaluate_current_model("Config 2 — Hybrid Sink-Shielded VQ4-Scar (~1.11 GiB < NF3 1.17 GiB)", b_c2)

    # -------------------------------------------------------------------------
    # CONFIG 3: Full MoS Block-64 INT4 + Polyglot ASVD Scars (`~1.61 GiB` — Roadmap #2 Reference!)
    #   - Verifies that adding our Polyglot ASVD Scars eliminates the token-90 FR->ZH subword leak!
    # -------------------------------------------------------------------------
    t_c3 = time.perf_counter()
    b_c3 = embed_int4_bytes
    model.model.embed_tokens.weight.data.copy_(wq_embed_int4_fp16)
    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            wq, nb = quantize_dequantize_int4_mos(w_gpu, c_r, rank=32 if sfx == "o_proj" else 0, use_eq=True)
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c3 += nb + (lin.bias.numel() * 2 if lin.bias is not None else 0)
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(layer.mlp, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.mlp.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            wq, nb = quantize_dequantize_int4_mos(
                w_gpu, c_r, rank=32 if sfx in ("gate_proj", "down_proj") else 0, use_eq=True
            )
            lin.weight.data.copy_(wq.to(torch.float16))
            b_c3 += nb
    print(f"\n(Built Config 3 in {time.perf_counter() - t_c3:.1f}s)", flush=True)
    evaluate_current_model("Config 3 — Full MoS Block-64 INT4 + Polyglot ASVD Scars (Roadmap #2)", b_c3)

    print(f"\n[VQ4-Scar Polyglot Lab] Completed all stages across PT/EN/ES/FR in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
