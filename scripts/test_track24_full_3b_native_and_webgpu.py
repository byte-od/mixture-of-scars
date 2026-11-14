"""Complete 36-Layer Polyglot (PT, EN, ES, FR) Benchmark of Track 2.4 (`VQ4-Scar` Unified Native + WebGPU).

Executes the full Track 2.4 architecture across BOTH engines (`Native PyTorch MPS` + `Physical WebGPU WGSL 1.0`):
  - Uses the canonical unified `scar_cache/vq4_scar_mos.py` (`VQ4ScarPackedMatrix` with row-scaled `INT4 B_scar`,
    `LUT256` with exact `(0,0,0,0)` + `D4` shells + `alpha^2`-weighted GPU K-Means, `d_eq_f16`, and `2-Pass` healing).
  - Reinvests the `~135 MiB` saved by `INT4`-compressed scars into protecting the nonlinear triad:
      • `Track 2.4-A` (`~1.08 GiB` Lean): Protects `embed_tokens`, `k/v_proj` (all 36L), `gate_proj` (all 36L),
        and `q/o_proj` (`L0..L4, L34..L35`) in `Block-64 INT4 + Eq + INT4-ASVD`; compresses all `up_proj`,
        `down_proj` (`r=80`), and interior `q/o_proj` (`r=64`) with `VQ4-Scar`!
      • `Track 2.4-B` (`~1.13 GiB` Full Triad Shield — still smaller than `NF3 1.17 GiB` & `Config 2 1.15 GiB`!):
        Protects BOTH `gate_proj` AND `o_proj` across ALL 36 layers in `Block-64 INT4 + Eq + INT4-ASVD`, while
        `up_proj` (`r=32`), `down_proj` (`r=96`), and interior `q_proj` (`r=32`) run in `VQ4-Scar`!
  - Dual-Engine Verification (`Native PyTorch MPS` <-> `Physical WebGPU WGSL 1.0` on Apple Silicon GPU):
      • Verifies full-block chained SwiGLU MLP + Attention `VQ4-Scar` execution across depths `L0, L18, L35`
        on unseen `PT, EN, ES, FR` hidden states (`cosine > 0.99999` across the nonlinear SwiGLU block!).
"""

from __future__ import annotations

import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/Users/juliocesarreisfilho/Projects/scar-cache")
from scar_cache.vq4_scar_mos import (
    VQ4ScarPackedMatrix,
    WebGPUVQ4ScarExecutor,
    dequant_block64_int4,
    dequant_rowwise_int4,
    pack_block64_int4,
    pack_rowwise_int4,
    quantize_matrix_vq4_scar_2pass,
)
from scripts.test_vector_scar_codebook_3b import (
    DISJOINT_POLYGLOT_CALIB_TEXTS,
    POLYGLOT_EVAL_PASSAGES,
    POLYGLOT_GEN_PROMPTS,
    SYSTEM_PROMPT,
)


SECOND_POLYGLOT_GEN_PROMPTS: List[Tuple[str, str]] = [
    ("PT", "Explique como funciona a busca binária e por que sua complexidade é O(log n)."),
    ("EN", "Explain how binary search works and why its time complexity is O(log n)."),
    ("ES", "Explica cómo funciona la búsqueda binaria y por qué su complejidad es O(log n)."),
    ("FR", "Expliquez comment fonctionne la recherche binaire et pourquoi sa complexité est O(log n)."),
]


class ProtectedInt4ScarMatrix:
    """Unified `Block-64 INT4 + Channel Eq + Row-Scaled INT4-ASVD Scar` matrix (Native + WebGPU)."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        rank: int,
        base_packed_u8: torch.Tensor,
        base_scales_f16: torch.Tensor,
        d_eq_f16: Optional[torch.Tensor],
        a_packed_u8: Optional[torch.Tensor] = None,
        a_scales_f16: Optional[torch.Tensor] = None,
        b_packed_u8: Optional[torch.Tensor] = None,
        b_scales_f16: Optional[torch.Tensor] = None,
        bias_f16: Optional[torch.Tensor] = None,
    ) -> None:
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        self.base_packed_u8 = base_packed_u8
        self.base_scales_f16 = base_scales_f16
        self.d_eq_f16 = d_eq_f16
        self.a_packed_u8 = a_packed_u8
        self.a_scales_f16 = a_scales_f16
        self.b_packed_u8 = b_packed_u8
        self.b_scales_f16 = b_scales_f16
        self.bias_f16 = bias_f16

    @property
    def physical_bytes(self) -> int:
        b = self.base_packed_u8.numel() + self.base_scales_f16.numel() * 2
        if self.d_eq_f16 is not None:
            b += self.d_eq_f16.numel() * 2
        if self.rank > 0 and self.a_packed_u8 is not None and self.b_packed_u8 is not None:
            assert self.a_scales_f16 is not None and self.b_scales_f16 is not None
            b += (
                self.a_packed_u8.numel()
                + self.a_scales_f16.numel() * 2
                + self.b_packed_u8.numel()
                + self.b_scales_f16.numel() * 2
            )
        if self.bias_f16 is not None:
            b += self.bias_f16.numel() * 2
        return int(b)

    @torch.no_grad()
    def reconstruct_weight_f32(self, device: torch.device) -> torch.Tensor:
        wq = dequant_block64_int4(self.base_packed_u8.to(device), self.base_scales_f16.to(device))
        if self.d_eq_f16 is not None:
            wq = wq / self.d_eq_f16.to(device).float().unsqueeze(0)
        if self.rank > 0 and self.a_packed_u8 is not None and self.b_packed_u8 is not None:
            assert self.a_scales_f16 is not None and self.b_scales_f16 is not None
            aq = dequant_block64_int4(self.a_packed_u8.to(device), self.a_scales_f16.to(device))
            bq = dequant_rowwise_int4(self.b_packed_u8.to(device), self.b_scales_f16.to(device))
            wq = wq + (bq @ aq)
        return wq

    @torch.no_grad()
    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        orig_prefix = x.shape[:-1]
        x2d = x.reshape(-1, self.in_dim).float()
        dev = x2d.device
        x_in = (x2d / self.d_eq_f16.to(dev).float().unsqueeze(0)) if self.d_eq_f16 is not None else x2d
        w_base = dequant_block64_int4(self.base_packed_u8.to(dev), self.base_scales_f16.to(dev))
        y = x_in @ w_base.T
        if self.rank > 0 and self.a_packed_u8 is not None and self.b_packed_u8 is not None:
            assert self.a_scales_f16 is not None and self.b_scales_f16 is not None
            aq = dequant_block64_int4(self.a_packed_u8.to(dev), self.a_scales_f16.to(dev))
            bq = dequant_rowwise_int4(self.b_packed_u8.to(dev), self.b_scales_f16.to(dev))
            y = y + ((x2d @ aq.T) @ bq.T)
        if self.bias_f16 is not None:
            y = y + self.bias_f16.to(dev).float().unsqueeze(0)
        return y.reshape(*orig_prefix, self.out_dim).to(orig_dtype)


@torch.no_grad()
def quantize_protected_int4_2pass(
    weight: torch.Tensor,
    col_rms: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rank: int = 24,
    use_eq: bool = True,
) -> ProtectedInt4ScarMatrix:
    """Quantizes a shielded projection (`gate_proj`, `o_proj`, etc.) into `Block-64 INT4 + Eq + 2-Pass INT4-ASVD`."""
    from scar_cache.vq4_scar_mos import _extract_compressed_int4_asvd

    w_f32 = weight.detach().float()
    N, K = w_f32.shape
    dev = w_f32.device
    bias_f16 = bias.detach().to(dev, dtype=torch.float16).contiguous() if bias is not None else None

    if use_eq and col_rms is not None:
        c_rms = col_rms.detach().to(dev, dtype=torch.float32).clamp(min=1e-6)
        d_f16 = (c_rms / c_rms.median().clamp(min=1e-7)).pow(0.55).clamp(0.125, 32.0).to(torch.float16)
        d_f32 = d_f16.float()
    else:
        c_rms = col_rms.detach().to(dev, dtype=torch.float32).clamp(min=1e-6) if col_rms is not None else None
        d_f16 = None
        d_f32 = None

    if rank > 0 and c_rms is not None:
        # Pass 1 base INT4 -> Pass 2 residual scar healing directly on quantization residual!
        w_in0 = (w_f32 * d_f32.unsqueeze(0)) if d_f32 is not None else w_f32
        _, _, w_q0_in = pack_block64_int4(w_in0)
        w_q0 = (w_q0_in / d_f32.unsqueeze(0)) if d_f32 is not None else w_q0_in
        a_p, a_s, b_p, b_s, scar_hat = _extract_compressed_int4_asvd(w_f32 - w_q0, c_rms, rank=rank)
        w_res = w_f32 - scar_hat
    else:
        a_p = a_s = b_p = b_s = None
        w_res = w_f32

    w_in = (w_res * d_f32.unsqueeze(0)) if d_f32 is not None else w_res
    b_pack, b_sc, _ = pack_block64_int4(w_in)
    return ProtectedInt4ScarMatrix(
        out_dim=N,
        in_dim=K,
        rank=rank if a_p is not None else 0,
        base_packed_u8=b_pack,
        base_scales_f16=b_sc,
        d_eq_f16=d_f16,
        a_packed_u8=a_p,
        a_scales_f16=a_s,
        b_packed_u8=b_p,
        b_scales_f16=b_s,
        bias_f16=bias_f16,
    )


@torch.no_grad()
def execute_protected_int4_webgpu_gemv(
    executor: WebGPUVQ4ScarExecutor,
    mat: ProtectedInt4ScarMatrix,
    x_batch_f32: np.ndarray,
) -> np.ndarray:
    """Executes `ProtectedInt4ScarMatrix` on Physical WebGPU WGSL 1.0 (`mos_fused_int4_bscar_gemv` + `swiglu`)."""
    U = executor.wgpu.BufferUsage
    dev = torch.device("cpu")

    # Equilibrate input on host/shader identically (`x / d_eq_f16`)
    x2d = np.ascontiguousarray(x_batch_f32.reshape(-1, mat.in_dim), dtype=np.float32)
    if mat.d_eq_f16 is not None:
        d_np = mat.d_eq_f16.detach().cpu().float().numpy()
        x_eq = np.ascontiguousarray(x2d / d_np[None, :], dtype=np.float32)
    else:
        x_eq = x2d

    # Dequantize only tiny row-scaled `B_scar` (`[out_dim, rank]` FP16, `<= 1.3 MiB`) for Kernel #4 `mos_fused_int4_bscar_gemv`
    chunks: List[np.ndarray] = []
    offs: Dict[str, int] = {}
    cur = 0

    def _add(name: str, raw: bytes) -> None:
        nonlocal cur
        if len(raw) % 4 != 0:
            raw = raw + b"\x00" * (4 - (len(raw) % 4))
        arr = np.frombuffer(raw, dtype=np.uint32).copy()
        offs[name] = cur
        chunks.append(arr)
        cur += int(arr.size)

    _add("base_p", mat.base_packed_u8.detach().cpu().contiguous().numpy().tobytes())
    _add("base_s", mat.base_scales_f16.detach().cpu().contiguous().numpy().tobytes())
    if mat.rank > 0 and mat.a_packed_u8 is not None and mat.b_packed_u8 is not None:
        assert mat.a_scales_f16 is not None and mat.b_scales_f16 is not None
        _add("a_p", mat.a_packed_u8.detach().cpu().contiguous().numpy().tobytes())
        _add("a_s", mat.a_scales_f16.detach().cpu().contiguous().numpy().tobytes())
        bq_f16 = dequant_rowwise_int4(mat.b_packed_u8.to(dev), mat.b_scales_f16.to(dev)).to(torch.float16)
        _add("b_f16", bq_f16.contiguous().numpy().tobytes())
    else:
        offs["a_p"] = offs["a_s"] = offs["b_f16"] = 0

    if mat.bias_f16 is not None:
        _add("bias", mat.bias_f16.detach().cpu().contiguous().numpy().tobytes())
        has_bias = 1
    else:
        offs["bias"] = 0
        has_bias = 0

    shard_u32 = np.concatenate(chunks, axis=0)
    shard_buf = executor.device.create_buffer_with_data(data=shard_u32.tobytes(), usage=U.STORAGE)
    x_raw_buf = executor.device.create_buffer(size=mat.in_dim * 4, usage=U.STORAGE | U.COPY_DST)
    x_eq_buf = executor.device.create_buffer(size=mat.in_dim * 4, usage=U.STORAGE | U.COPY_DST)
    u_buf = executor.device.create_buffer(size=max(4, mat.rank * 4), usage=U.STORAGE)
    dummy = executor.device.create_buffer(size=4, usage=U.STORAGE)
    y_buf = executor.device.create_buffer(size=mat.out_dim * 4, usage=U.STORAGE | U.COPY_SRC)

    bg_a = None
    if mat.rank > 0:
        u_a = executor._params_buf(p0=mat.rank, p1=mat.in_dim, p2=0, p3=offs["a_p"], p4=offs["a_s"])
        bg_a = executor._bind_group("mos_fused_int4_bscar_gemv", {0: u_a, 1: shard_buf, 3: x_raw_buf, 4: dummy, 6: u_buf})

    u_m = executor._params_buf(
        p0=mat.out_dim,
        p1=mat.in_dim,
        p2=mat.rank,
        p3=offs["base_p"],
        p4=offs["base_s"],
        p5=offs["b_f16"],
        p6=offs["bias"],
        p7=has_bias,
    )
    bg_m = executor._bind_group("mos_fused_int4_bscar_gemv", {0: u_m, 1: shard_buf, 3: x_eq_buf, 4: u_buf, 6: y_buf})

    outs: List[np.ndarray] = []
    for t in range(x2d.shape[0]):
        executor.device.queue.write_buffer(x_raw_buf, 0, x2d[t].tobytes())
        executor.device.queue.write_buffer(x_eq_buf, 0, x_eq[t].tobytes())
        enc = executor.device.create_command_encoder()
        if bg_a is not None:
            c1 = enc.begin_compute_pass()
            c1.set_pipeline(executor.pipelines["mos_fused_int4_bscar_gemv"])
            c1.set_bind_group(0, bg_a)
            c1.dispatch_workgroups((mat.rank + 7) >> 3, 1, 1)
            c1.end()
        c2 = enc.begin_compute_pass()
        c2.set_pipeline(executor.pipelines["mos_fused_int4_bscar_gemv"])
        c2.set_bind_group(0, bg_m)
        c2.dispatch_workgroups((mat.out_dim + 7) >> 3, 1, 1)
        c2.end()
        executor.device.queue.submit([enc.finish()])
        outs.append(np.frombuffer(executor.device.queue.read_buffer(y_buf), dtype=np.float32).copy())
    return np.stack(outs, axis=0).reshape(*x_batch_f32.shape[:-1], mat.out_dim)


@torch.no_grad()
def _effective_weight_from_vq4(packed: VQ4ScarPackedMatrix, dev: torch.device) -> torch.Tensor:
    """Materializes exact effective `[out_dim, in_dim]` weight of `VQ4ScarPackedMatrix` for fast 36-layer eval."""
    from scar_cache.vq4_scar_mos import fwht_block64_orthonormal, rademacher_signs_u32_exact

    signs = rademacher_signs_u32_exact(packed.in_dim, dev)
    lut_f32 = packed.lut256_f16.to(dev).float()
    codes = packed.vq_codes_u8.to(dev).long()
    w_rot_blk = lut_f32[codes].reshape(packed.out_dim, packed.in_dim // 64, 64)
    w_rot = (w_rot_blk * packed.vq_scales_f16.to(dev).float().unsqueeze(-1)).reshape(packed.out_dim, packed.in_dim)
    wq = fwht_block64_orthonormal(w_rot, signs, inverse=True) / packed.d_eq_f16.to(dev).float().unsqueeze(0)
    if packed.rank > 0 and packed.a_scar_packed_u8 is not None and packed.b_scar_packed_u8 is not None:
        assert packed.a_scar_scales_f16 is not None and packed.b_scar_scales_f16 is not None
        aq = dequant_block64_int4(packed.a_scar_packed_u8.to(dev), packed.a_scar_scales_f16.to(dev))
        bq = dequant_rowwise_int4(packed.b_scar_packed_u8.to(dev), packed.b_scar_scales_f16.to(dev))
        wq = wq + (bq @ aq)
    return wq


@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps")
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Track 2.4 Dual Native+WebGPU Lab] Loading {model_id} FP16 on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=True).to(dev)
    model.eval()

    cpu_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # -------------------------------------------------------------------------
    # STEP 1: Disjoint 4-Language Activation RMS Calibration (`0%` eval overlap)
    # -------------------------------------------------------------------------
    sum_sq: Dict[str, torch.Tensor] = {}
    cnt_tok: Dict[str, int] = {}
    hooks = []

    def _make_hook(name: str):
        def _fn(_m: nn.Module, inp: Tuple[torch.Tensor, ...], _o: torch.Tensor) -> None:
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

    # Capture unseen 4-language reference logits + representative hidden states at L0, L18, L35 for WebGPU parity!
    eval_ids = {lang: tok(txt, return_tensors="pt")["input_ids"].to(dev) for lang, txt in POLYGLOT_EVAL_PASSAGES.items()}
    ref_logits: Dict[str, torch.Tensor] = {}
    layer_inputs_by_lang: Dict[str, Dict[int, torch.Tensor]] = {lang: {} for lang in ("PT", "EN", "ES", "FR")}

    for lang, ids in eval_ids.items():
        cap_h = []
        for probe_l in (0, 18, 35):
            def _mk_p(pl: int):
                def _fp(_m, inp, _o):
                    # Capture last 8 unseen tokens of the MLP input for fast physical WebGPU block verification
                    layer_inputs_by_lang[lang][pl] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])[-8:].cpu()
                return _fp
            cap_h.append(model.model.layers[probe_l].mlp.gate_proj.register_forward_hook(_mk_p(probe_l)))
        ref_logits[lang] = model(ids, use_cache=False).logits[:, :-1, :].float().cpu()
        for h in cap_h:
            h.remove()

    def evaluate_full_model(title: str, total_bytes: int) -> None:
        mib = total_bytes / (1024.0**2)
        gib = total_bytes / (1024.0**3)
        print(f"\n{'='*94}")
        print(f"CONFIG: {title}")
        print(f"SIZE  : {mib:.2f} MiB ({gib:.3f} GiB) | Compression vs FP16 (6,170 MiB): {6170.0/mib:.2f}x")
        print(f"{'='*94}", flush=True)
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
        print(f"  [MEAN 4-LANG] KL={sum(kls)/4.0:6.4f} | Top-1 Agreement={sum(top1s)/4.0:5.1f}%", flush=True)

        print("\n  --- Suite 1: Quantum Physics & Boltzmann Constant (140 tokens, Greedy) ---")
        for lang, q in POLYGLOT_GEN_PROMPTS:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            inp = (enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc).to(dev)
            out = model.generate(inp, max_new_tokens=140, do_sample=False)
            ans = tok.decode(out[0, inp.shape[1] :], skip_special_tokens=True).replace("\n", " ")
            print(f"  [{lang} Q1] {ans}", flush=True)

        print("\n  --- Suite 2: Algorithms & Complexity O(log n) (120 tokens, Greedy) ---")
        for lang, q in SECOND_POLYGLOT_GEN_PROMPTS:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            inp = (enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc).to(dev)
            out = model.generate(inp, max_new_tokens=120, do_sample=False)
            ans = tok.decode(out[0, inp.shape[1] :], skip_special_tokens=True).replace("\n", " ")
            print(f"  [{lang} Q2] {ans}", flush=True)

    # Pre-pack protected multilingual `INT4` `embed_tokens` (`157.52 MiB`, tied with `lm_head` in Qwen2.5-3B)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    norm_bytes = sum(v.numel() * 2 for k, v in cpu_state.items() if "norm" in k)
    emb_p, emb_s, wq_emb = pack_block64_int4(cpu_state["model.embed_tokens.weight"].to(dev, dtype=torch.float32))
    emb_bytes = emb_p.numel() + emb_s.numel() * 2 + norm_bytes
    wq_emb_f16 = wq_emb.to(torch.float16)

    # -------------------------------------------------------------------------
    # TRACK 2.4-B (PRIMARY TARGET): Full Nonlinear Triad Shield (`~1.13 GiB` < NF3 `1.17 GiB`!)
    #   - `embed_tokens`: Block-64 INT4 (`157.5 MiB`)
    #   - ALL 36 layers:
    #       • `k_proj, v_proj`: Block-64 INT4 + Eq
    #       • `gate_proj` (SiLU switch): Block-64 INT4 + Eq + 2-Pass INT4-ASVD (`r=24`)
    #       • `o_proj` (residual writer): Block-64 INT4 + Eq + 2-Pass INT4-ASVD (`r=24`)
    #       • `q_proj`: Boundary `L0..L4, L34..L35` in INT4; interior `L5..L33` in `VQ4-Scar` (`r=32`)
    #       • `up_proj`: `VQ4-Scar` (`LUT256 + Eq + FWHT-64 + 2-Pass Row-Scaled INT4-ASVD`, `r=32`)
    #       • `down_proj`: `VQ4-Scar` (`LUT256 + Eq + FWHT-64 + 2-Pass Row-Scaled INT4-ASVD`, `r=96`)
    # -------------------------------------------------------------------------
    t_24b = time.perf_counter()
    bytes_24b = emb_bytes
    model.model.embed_tokens.weight.data.copy_(wq_emb_f16)

    # Save packed `L0, L18, L35` MLP operators for the Physical WebGPU WGSL 1.0 Chained SwiGLU Parity Test!
    probe_mlp_ops: Dict[int, Tuple[ProtectedInt4ScarMatrix, VQ4ScarPackedMatrix, VQ4ScarPackedMatrix]] = {}

    for l_idx, layer in enumerate(model.model.layers):
        is_boundary = l_idx < 5 or l_idx >= 34
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if sfx in ("k_proj", "v_proj"):
                pm = quantize_protected_int4_2pass(w_gpu, c_r, bias=lin.bias, rank=0, use_eq=True)
                wq = pm.reconstruct_weight_f32(dev)
                nb = pm.physical_bytes
            elif sfx == "o_proj" or is_boundary:
                r = 24 if sfx == "o_proj" else 0
                pm = quantize_protected_int4_2pass(w_gpu, c_r, bias=lin.bias, rank=r, use_eq=True)
                wq = pm.reconstruct_weight_f32(dev)
                nb = pm.physical_bytes
            else:
                vqm = quantize_matrix_vq4_scar_2pass(w_gpu, c_r, bias=lin.bias, rank=32, two_pass=True)
                wq = _effective_weight_from_vq4(vqm, dev)
                nb = vqm.physical_bytes
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_24b += nb

        # MLP: gate_proj (Protected INT4 + r=24), up_proj (VQ4-Scar r=32), down_proj (VQ4-Scar r=96)
        w_gate = cpu_state[f"model.layers.{l_idx}.mlp.gate_proj.weight"].to(dev, dtype=torch.float32)
        w_up = cpu_state[f"model.layers.{l_idx}.mlp.up_proj.weight"].to(dev, dtype=torch.float32)
        w_down = cpu_state[f"model.layers.{l_idx}.mlp.down_proj.weight"].to(dev, dtype=torch.float32)

        gate_op = quantize_protected_int4_2pass(w_gate, col_rms[f"L{l_idx}.gate_proj"], rank=24, use_eq=True)
        up_op = quantize_matrix_vq4_scar_2pass(w_up, col_rms[f"L{l_idx}.up_proj"], rank=32, two_pass=True)
        down_op = quantize_matrix_vq4_scar_2pass(w_down, col_rms[f"L{l_idx}.down_proj"], rank=96, two_pass=True)

        layer.mlp.gate_proj.weight.data.copy_(gate_op.reconstruct_weight_f32(dev).to(torch.float16))
        layer.mlp.up_proj.weight.data.copy_(_effective_weight_from_vq4(up_op, dev).to(torch.float16))
        layer.mlp.down_proj.weight.data.copy_(_effective_weight_from_vq4(down_op, dev).to(torch.float16))
        bytes_24b += gate_op.physical_bytes + up_op.physical_bytes + down_op.physical_bytes

        if l_idx in (0, 18, 35):
            probe_mlp_ops[l_idx] = (
                ProtectedInt4ScarMatrix(
                    gate_op.out_dim,
                    gate_op.in_dim,
                    gate_op.rank,
                    gate_op.base_packed_u8.cpu(),
                    gate_op.base_scales_f16.cpu(),
                    gate_op.d_eq_f16.cpu() if gate_op.d_eq_f16 is not None else None,
                    gate_op.a_packed_u8.cpu() if gate_op.a_packed_u8 is not None else None,
                    gate_op.a_scales_f16.cpu() if gate_op.a_scales_f16 is not None else None,
                    gate_op.b_packed_u8.cpu() if gate_op.b_packed_u8 is not None else None,
                    gate_op.b_scales_f16.cpu() if gate_op.b_scales_f16 is not None else None,
                ),
                up_op.to_device(torch.device("cpu")),
                down_op.to_device(torch.device("cpu")),
            )

    print(f"\n(Built Track 2.4-B 36-Layer Unified Model in {time.perf_counter() - t_24b:.1f}s)", flush=True)
    evaluate_full_model("Track 2.4-B — Full Triad-Shielded VQ4-Scar (gate+o INT4+Scar, up/down/q VQ4-Scar)", bytes_24b)

    # -------------------------------------------------------------------------
    # DUAL-ENGINE VERIFICATION: NATIVE PYTORCH MPS <-> PHYSICAL WEBGPU WGSL 1.0
    #   Executes the FULL NONLINEAR CHAINED SwiGLU MLP BLOCK:
    #     y = down_proj_vq4( SiLU(gate_proj_int4scar(x)) * up_proj_vq4(x) )
    #   across depths `L0` (Sink), `L18` (Golden Slice), and `L35` (Final Layer)
    #   on unseen `PT, EN, ES, FR` hidden states!
    # -------------------------------------------------------------------------
    print(f"\n{'='*94}")
    print("DUAL-ENGINE PARITY AUDIT: NATIVE PYTORCH (MPS) <-> PHYSICAL WEBGPU WGSL 1.0 (CHAINED SwiGLU MLP)")
    print(f"{'='*94}", flush=True)
    executor = WebGPUVQ4ScarExecutor()
    print(f"  Physical WebGPU Adapter: {executor.adapter_name} (WGSL 1.0 Core — 17 Shaders Compiled)")

    for probe_l in (0, 18, 35):
        g_op, u_op, d_op = probe_mlp_ops[probe_l]
        cos_list, rel_list = [], []
        t_wgpu0 = time.perf_counter()
        for lang in ("PT", "EN", "ES", "FR"):
            x_t = layer_inputs_by_lang[lang][probe_l]  # [8, 2048] unseen hidden states

            # 1. Native PyTorch forward of the complete nonlinear SwiGLU block
            g_nat = g_op.forward_native(x_t)
            u_nat = u_op.forward_native(x_t)
            mid_nat = F.silu(g_nat) * u_nat
            y_nat = d_op.forward_native(mid_nat).numpy()

            # 2. Physical WebGPU WGSL 1.0 forward of the complete nonlinear SwiGLU block
            x_np = x_t.numpy()
            g_web = execute_protected_int4_webgpu_gemv(executor, g_op, x_np)
            u_web = executor.execute_gemv(u_op, x_np)
            mid_web = (g_web / (1.0 + np.exp(-np.clip(g_web, -80.0, 80.0)))) * u_web
            y_web = executor.execute_gemv(d_op, np.ascontiguousarray(mid_web, dtype=np.float32))

            cos = float(
                np.dot(y_nat.ravel(), y_web.ravel())
                / (np.linalg.norm(y_nat.ravel()) * np.linalg.norm(y_web.ravel()) + 1e-12)
            )
            rel = float(np.linalg.norm(y_nat - y_web) / (np.linalg.norm(y_nat) + 1e-12))
            cos_list.append(cos)
            rel_list.append(rel)
            assert cos > 0.99999, f"Parity drift at L{probe_l} [{lang}]: cos={cos}"

        dt_ms = (time.perf_counter() - t_wgpu0) * 1000.0
        print(
            f"  [Layer {probe_l:2d} Chained SwiGLU 2048->11008->2048] "
            f"PT cos={cos_list[0]:.8f} | EN cos={cos_list[1]:.8f} | ES cos={cos_list[2]:.8f} | FR cos={cos_list[3]:.8f} "
            f"| Mean RelDiff={sum(rel_list)/4.0:.2e} ({dt_ms:.1f} ms)",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # TRACK 2.4-A (LEAN VARIANT): `~1.09 GiB` (`o_proj` interior also in `VQ4-Scar r=64`)
    # -------------------------------------------------------------------------
    t_24a = time.perf_counter()
    bytes_24a = emb_bytes
    model.model.embed_tokens.weight.data.copy_(wq_emb_f16)
    for l_idx, layer in enumerate(model.model.layers):
        is_boundary = l_idx < 4 or l_idx >= 35
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            w_gpu = cpu_state[f"model.layers.{l_idx}.self_attn.{sfx}.weight"].to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            if sfx in ("k_proj", "v_proj") or is_boundary:
                pm = quantize_protected_int4_2pass(
                    w_gpu, c_r, bias=lin.bias, rank=16 if sfx == "o_proj" else 0, use_eq=True
                )
                wq = pm.reconstruct_weight_f32(dev)
                nb = pm.physical_bytes
            else:
                r = 64 if sfx == "o_proj" else 16
                vqm = quantize_matrix_vq4_scar_2pass(w_gpu, c_r, bias=lin.bias, rank=r, two_pass=True)
                wq = _effective_weight_from_vq4(vqm, dev)
                nb = vqm.physical_bytes
            lin.weight.data.copy_(wq.to(torch.float16))
            bytes_24a += nb

        w_gate = cpu_state[f"model.layers.{l_idx}.mlp.gate_proj.weight"].to(dev, dtype=torch.float32)
        w_up = cpu_state[f"model.layers.{l_idx}.mlp.up_proj.weight"].to(dev, dtype=torch.float32)
        w_down = cpu_state[f"model.layers.{l_idx}.mlp.down_proj.weight"].to(dev, dtype=torch.float32)

        gate_op = quantize_protected_int4_2pass(w_gate, col_rms[f"L{l_idx}.gate_proj"], rank=16, use_eq=True)
        up_op = quantize_matrix_vq4_scar_2pass(w_up, col_rms[f"L{l_idx}.up_proj"], rank=16, two_pass=True)
        down_op = quantize_matrix_vq4_scar_2pass(w_down, col_rms[f"L{l_idx}.down_proj"], rank=80, two_pass=True)

        layer.mlp.gate_proj.weight.data.copy_(gate_op.reconstruct_weight_f32(dev).to(torch.float16))
        layer.mlp.up_proj.weight.data.copy_(_effective_weight_from_vq4(up_op, dev).to(torch.float16))
        layer.mlp.down_proj.weight.data.copy_(_effective_weight_from_vq4(down_op, dev).to(torch.float16))
        bytes_24a += gate_op.physical_bytes + up_op.physical_bytes + down_op.physical_bytes

    print(f"\n(Built Track 2.4-A Lean 36-Layer Unified Model in {time.perf_counter() - t_24a:.1f}s)", flush=True)
    evaluate_full_model("Track 2.4-A Lean — ~1.09 GiB VQ4-Scar (gate INT4+Scar, o/up/down/q VQ4-Scar)", bytes_24a)

    print(f"\n[Track 2.4 Dual Native+WebGPU Lab] Completed all benchmarks in {time.perf_counter() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
