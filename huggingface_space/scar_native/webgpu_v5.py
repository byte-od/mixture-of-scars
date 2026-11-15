"""Zero-Facade WebGPU WGSL 1.0 Exporter & Execution Engine for `ScarNativeLM` (`v5.0.0`).

Executes 100% of `ScarNativeLM` inference on the physical GPU via the canonical 22-shader WGSL 1.0 suite:
  - `embed_int4_lookup`
  - `rmsnorm_f32`
  - `vq4_equilibrate_fwht64` + `mos_fused_int4_bscar_gemv` + `mos_fused_vq4_int4scar_gemv`
  - `vec_copy_f32` + `qwen_rope_inplace`
  - `scarnative_widrow_hoff_evict_update` (Shader #21)
  - `scarnative_o1_hybrid_gqa_attention` (Shader #22)
  - `golden_slice_scar_inject` (`P_perp_lang` Cartridge Injection)
  - `swiglu_mul_f32` + `vec_add_inplace_f32` + `apply_script_subspace_gate`
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scar_codec.core import dequant_block64_int4, dequant_rowwise_int4, pack_block64_int4
from scar_native.model import DetachedScarCartridge, ScarNativeConfig, ScarNativeLM, build_rope_inv_freq
from scar_native.tokenizer import ScarNativeTokenizer

WGSL_KERNELS_PATH = Path(__file__).resolve().parents[1] / "scar_cache" / "webgpu" / "mos_scache_kernels.wgsl"


def _pack_params(
    p0: int = 0,
    p1: int = 0,
    p2: int = 0,
    p3: int = 0,
    p4: int = 0,
    p5: int = 0,
    p6: int = 0,
    p7: int = 0,
    f0: float = 0.0,
    f1: float = 0.0,
    f2: float = 0.0,
    f3: float = 0.0,
    f4: float = 0.0,
    f5: float = 0.0,
    f6: float = 0.0,
    f7: float = 0.0,
) -> bytes:
    return struct.pack(
        "<8I8f",
        int(p0), int(p1), int(p2), int(p3), int(p4), int(p5), int(p6), int(p7),
        float(f0), float(f1), float(f2), float(f3), float(f4), float(f5), float(f6), float(f7),
    )


class _SingleShardPacker:
    def __init__(self) -> None:
        self.buf = bytearray()

    def append_aligned(self, raw: bytes) -> int:
        rem = len(self.buf) % 4
        if rem:
            self.buf.extend(b"\x00" * (4 - rem))
        u32_off = len(self.buf) // 4
        self.buf.extend(raw)
        rem2 = len(self.buf) % 4
        if rem2:
            self.buf.extend(b"\x00" * (4 - rem2))
        return u32_off


@torch.no_grad()
def export_scar_native_webgpu_bundle(
    model: ScarNativeLM,
    tokenizer: ScarNativeTokenizer,
    out_dir: Path,
) -> Dict[str, Any]:
    """Exports `ScarNativeLM` + `DetachedScarStore` into a compact, self-contained WebGPU WGSL 1.0 bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    packer = _SingleShardPacker()

    # 1. INT4 Embedding
    if model.embed_packed_u8 is not None and model.embed_scales_f16 is not None:
        emb_pk, emb_sc = model.embed_packed_u8.cpu(), model.embed_scales_f16.cpu()
    else:
        emb_pk, emb_sc, _ = pack_block64_int4(model.embed_tokens.weight.detach().cpu().float())

    emb_pk_off = packer.append_aligned(emb_pk.contiguous().numpy().tobytes())
    emb_sc_off = packer.append_aligned(emb_sc.contiguous().to(torch.float16).numpy().tobytes())

    # Shared canonical 2 KB LUT256 (`512 u32` words) stored ONCE for the entire engine!
    first_pkg = model.lm_head.pack_vq4_scar(drop_master=False)
    shared_lut_off = packer.append_aligned(first_pkg.lut256_f16.detach().cpu().contiguous().numpy().tobytes())

    def _pack_scar_linear(lin_mod: Any) -> Dict[str, int]:
        pkg = lin_mod.pack_vq4_scar(drop_master=False)
        vc_off = packer.append_aligned(pkg.codes_u32.detach().cpu().contiguous().numpy().tobytes())
        vs_off = packer.append_aligned(pkg.scales_f16.detach().cpu().contiguous().numpy().tobytes())
        deq_off = packer.append_aligned(pkg.d_eq_f16.detach().cpu().contiguous().numpy().tobytes())
        spec: Dict[str, int] = {
            "out_dim": int(pkg.out_features),
            "in_dim": int(pkg.in_features),
            "rank": int(pkg.rank),
            "vq_codes_u32_off": vc_off,
            "vq_scales_u32_off": vs_off,
            "lut256_u32_off": shared_lut_off,
            "d_eq_u32_off": deq_off,
            "ascar_packed_u32_off": 0,
            "ascar_scales_u32_off": 0,
            "bscar_u32_off": 0,
        }
        if pkg.rank > 0:
            ap_off = packer.append_aligned(pkg.a_scar_u8.detach().cpu().contiguous().numpy().tobytes())
            as_off = packer.append_aligned(pkg.a_scales_f16.detach().cpu().contiguous().numpy().tobytes())
            b_comb = (
                pkg.b_scar_u8.detach().cpu().contiguous().numpy().tobytes()
                + pkg.b_row_scales_f16.detach().cpu().contiguous().numpy().tobytes()
            )
            bs_off = packer.append_aligned(b_comb)
            spec["ascar_packed_u32_off"] = ap_off
            spec["ascar_scales_u32_off"] = as_off
            spec["bscar_u32_off"] = bs_off
        return spec

    layers_meta: List[Dict[str, Any]] = []
    for l_idx, blk in enumerate(model.layers):
        ln1_off = packer.append_aligned(blk.input_layernorm.weight.detach().cpu().to(torch.float16).numpy().tobytes())
        ln2_off = packer.append_aligned(
            blk.post_attention_layernorm.weight.detach().cpu().to(torch.float16).numpy().tobytes()
        )
        attn = blk.self_attn
        lam_h, eta_h, alpha_h = attn.gate_params()
        # Pack per-KV-head gate parameters `[n_kv, head_dim + 3]`
        gate_rows: List[np.ndarray] = []
        for h_kv in range(cfg.num_kv_heads):
            wg = attn.w_gate[h_kv].detach().cpu().float().numpy()
            bg = float(attn.b_gate[h_kv].detach().cpu().item())
            lh = float(lam_h[h_kv].detach().cpu().item())
            eh = float(eta_h[h_kv].detach().cpu().item())
            gate_rows.append(np.concatenate([wg, np.array([bg, lh, eh], dtype=np.float32)]))
        kv_gate_f32 = np.concatenate(gate_rows).astype(np.float32)
        alpha_f32 = alpha_h.detach().cpu().float().numpy().astype(np.float32)

        layers_meta.append(
            {
                "layer_idx": l_idx,
                "ln1_u32_off": ln1_off,
                "ln2_u32_off": ln2_off,
                "kv_gate_f32": kv_gate_f32.tolist(),
                "alpha_assoc_f32": alpha_f32.tolist(),
                "projs": {
                    "q_proj": _pack_scar_linear(attn.q_proj),
                    "k_proj": _pack_scar_linear(attn.k_proj),
                    "v_proj": _pack_scar_linear(attn.v_proj),
                    "o_proj": _pack_scar_linear(attn.o_proj),
                    "gate_proj": _pack_scar_linear(blk.mlp.gate_proj),
                    "up_proj": _pack_scar_linear(blk.mlp.up_proj),
                    "down_proj": _pack_scar_linear(blk.mlp.down_proj),
                },
            }
        )

    final_norm_off = packer.append_aligned(model.norm.weight.detach().cpu().to(torch.float16).numpy().tobytes())
    lm_head_spec = _pack_scar_linear(model.lm_head)

    shard_name = "scarnative_v5_shard_00.bin"
    (out_dir / shard_name).write_bytes(bytes(packer.buf))
    shutil.copyfile(WGSL_KERNELS_PATH, out_dir / "mos_scache_kernels.wgsl")
    tokenizer.save(out_dir / "tokenizer.json")

    # Export `P_perp_lang` nullspace basis & all detached cartridges
    u_lang_np = model.nullspace.u_lang.detach().cpu().float().numpy()
    cart_list: List[Dict[str, Any]] = []
    cart_blob = bytearray()
    for cid, c in model.scar_store.cartridges.items():
        if c.wiped:
            continue
        raw_page = c.to_page_aligned_bytes()
        off = len(cart_blob)
        cart_blob.extend(raw_page)
        a_f32 = dequant_block64_int4(c.a_scar_u8.cpu(), c.a_scales_f16.cpu()).numpy().astype(np.float16)
        b_f32 = dequant_rowwise_int4(c.b_scar_u8.cpu(), c.b_row_scales_f16.cpu()).numpy().astype(np.float16)
        cart_list.append(
            {
                "cartridge_id": cid,
                "byte_offset": off,
                "record_bytes": len(raw_page),
                "tau_cos": float(c.tau_cos),
                "z_anchor": c.z_anchor_f16.detach().cpu().float().tolist(),
                "a_scar_f16_b64": np.frombuffer(a_f32.tobytes(), dtype=np.uint16).tolist(),
                "b_scar_f16_b64": np.frombuffer(b_f32.tobytes(), dtype=np.uint16).tolist(),
            }
        )
    (out_dir / "detached_cartridges.scarstore").write_bytes(bytes(cart_blob))

    manifest: Dict[str, Any] = {
        "format": "scar_native_webgpu_v5_0",
        "version": "5.0.0",
        "config": cfg.to_dict(),
        "shard_file": shard_name,
        "shard_bytes": len(packer.buf),
        "wgsl_shader": "mos_scache_kernels.wgsl",
        "embed": {"packed_u32_off": emb_pk_off, "scales_u32_off": emb_sc_off},
        "final_norm_u32_off": final_norm_off,
        "lm_head": lm_head_spec,
        "layers": layers_meta,
        "nullspace": {
            "is_calibrated": int(model.nullspace.is_calibrated.item()),
            "k_lang": cfg.k_lang_null,
            "u_lang_t": u_lang_np.T.tolist(),  # [k_lang, D]
        },
        "cjk_bitmask_u32": [int(w) for w in tokenizer.build_cjk_bitmask_u32().tolist()],
        "cartridges": cart_list,
    }
    (out_dir / "scarnative_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class ScarNativeWebGPUEngine:
    """Physical GPU (`wgpu` Metal / Vulkan / DX12) Executor for `ScarNativeLM` (`v5.0.0`)."""

    def __init__(self, bundle_dir: Path) -> None:
        import wgpu  # Lazy import so CPU-only environments still import `scar_native`

        self.wgpu = wgpu
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "scarnative_manifest.json").read_text(encoding="utf-8"))
        self.cfg = ScarNativeConfig(**self.manifest["config"])
        self.tokenizer = ScarNativeTokenizer.load(self.bundle_dir / "tokenizer.json")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter available.")
        self.device = adapter.request_device_sync(required_limits={})
        info = adapter.info
        self.adapter_name = str(info.get("device") or info.get("description") or "WebGPU Adapter")

        wgsl_code = (self.bundle_dir / self.manifest["wgsl_shader"]).read_text(encoding="utf-8")
        self.shader_mod = self.device.create_shader_module(code=wgsl_code)
        self.pipelines: Dict[str, Any] = {}
        for ep in (
            "embed_int4_lookup",
            "rmsnorm_f32",
            "vq4_equilibrate_fwht64",
            "mos_fused_int4_bscar_gemv",
            "mos_fused_vq4_int4scar_gemv",
            "vec_copy_f32",
            "qwen_rope_inplace",
            "scarnative_widrow_hoff_evict_update",
            "scarnative_o1_hybrid_gqa_attention",
            "golden_slice_scar_inject",
            "swiglu_mul_f32",
            "vec_add_inplace_f32",
            "apply_script_subspace_gate",
        ):
            self.pipelines[ep] = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": self.shader_mod, "entry_point": ep},
            )

        U = wgpu.BufferUsage
        shard_bytes = (self.bundle_dir / self.manifest["shard_file"]).read_bytes()
        self.shard_buf = self.device.create_buffer_with_data(data=shard_bytes, usage=U.STORAGE)

        c = self.cfg
        q_dim = c.num_q_heads * c.head_dim
        kv_dim = c.num_kv_heads * c.head_dim

        # Strictly O(1) constant-space GPU buffers (NEVER reallocated across infinite context!)
        total_kv_heads = c.num_layers * c.num_kv_heads
        self.counts_u32_total = total_kv_heads * 4
        cap_per_head = c.num_sink + c.slide_len
        vec_u32 = c.head_dim // 2
        kv_u32_words = self.counts_u32_total + total_kv_heads * 3 * cap_per_head * vec_u32

        self.scache_kv_buf = self.device.create_buffer(size=kv_u32_words * 4, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC)
        self.scache_w_scar_buf = self.device.create_buffer(
            size=total_kv_heads * c.head_dim * c.head_dim * 4, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC
        )
        self.scache_total_gpu_bytes = int(self.scache_kv_buf.size + self.scache_w_scar_buf.size)

        # Scratch activation vectors (single token)
        self.step_state_buf = self.device.create_buffer(size=16, usage=U.STORAGE | U.COPY_DST)
        self.h_cur_buf = self.device.create_buffer(size=c.hidden_size * 4, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC)
        self.norm_buf = self.device.create_buffer(size=c.hidden_size * 4, usage=U.STORAGE)
        self.vq4_rot_buf = self.device.create_buffer(size=max(c.hidden_size, c.intermediate_size) * 4, usage=U.STORAGE)
        self.asvd_u_buf = self.device.create_buffer(size=max(16, c.scar_rank) * 4, usage=U.STORAGE)
        self.dummy_buf = self.device.create_buffer(size=16, usage=U.STORAGE)

        self.q_buf = self.device.create_buffer(size=q_dim * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        self.k_buf = self.device.create_buffer(size=kv_dim * 4, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        self.v_buf = self.device.create_buffer(size=kv_dim * 4, usage=U.STORAGE)
        self.attn_out_buf = self.device.create_buffer(size=q_dim * 4, usage=U.STORAGE)
        self.proj_out_buf = self.device.create_buffer(size=c.hidden_size * 4, usage=U.STORAGE)
        self.gate_buf = self.device.create_buffer(size=c.intermediate_size * 4, usage=U.STORAGE)
        self.up_buf = self.device.create_buffer(size=c.intermediate_size * 4, usage=U.STORAGE)
        self.logits_buf = self.device.create_buffer(size=c.vocab_size * 4, usage=U.STORAGE | U.COPY_SRC)
        self.logits_read_buf = self.device.create_buffer(size=c.vocab_size * 4, usage=U.MAP_READ | U.COPY_DST)
        self.golden_read_buf = self.device.create_buffer(size=c.hidden_size * 4, usage=U.MAP_READ | U.COPY_DST)

        inv_freq = build_rope_inv_freq(c.head_dim, c.rope_theta).numpy().astype(np.float32)
        self.inv_freq_buf = self.device.create_buffer_with_data(data=inv_freq.tobytes(), usage=U.STORAGE)

        cjk_mask = np.array(self.manifest["cjk_bitmask_u32"], dtype=np.uint32)
        self.cjk_mask_buf = self.device.create_buffer_with_data(data=cjk_mask.tobytes(), usage=U.STORAGE)

        # Per-layer `k_raw_and_gates` and `q_raw_and_alpha` buffers (pre-populated with static gate tails!)
        self.k_raw_gate_bufs: List[Any] = []
        self.q_raw_alpha_bufs: List[Any] = []
        for l_meta in self.manifest["layers"]:
            tail_kv = np.array(l_meta["kv_gate_f32"], dtype=np.float32)
            full_kv = np.zeros((kv_dim + tail_kv.shape[0],), dtype=np.float32)
            full_kv[kv_dim:] = tail_kv
            self.k_raw_gate_bufs.append(
                self.device.create_buffer_with_data(data=full_kv.tobytes(), usage=U.STORAGE | U.COPY_DST)
            )
            tail_q = np.array(l_meta["alpha_assoc_f32"], dtype=np.float32)
            full_q = np.zeros((q_dim + tail_q.shape[0],), dtype=np.float32)
            full_q[q_dim:] = tail_q
            self.q_raw_alpha_bufs.append(
                self.device.create_buffer_with_data(data=full_q.tobytes(), usage=U.STORAGE | U.COPY_DST)
            )

        # Golden-Slice Cartridge GPU buffers
        zero_mu = np.zeros((2 * c.hidden_size,), dtype=np.float32)
        self.zero_mu_buf = self.device.create_buffer_with_data(data=zero_mu.tobytes(), usage=U.STORAGE)
        cap_elems = 1 + 2 * c.cartridge_rank * c.hidden_size
        self.active_cap_buf = self.device.create_buffer(size=cap_elems * 4, usage=U.STORAGE | U.COPY_DST)
        self.scar_telem_buf = self.device.create_buffer(size=64, usage=U.STORAGE)
        self.active_cartridge_id: Optional[str] = None

        # Parse exported cartridges & nullspace
        self.u_lang_t = np.array(self.manifest["nullspace"]["u_lang_t"], dtype=np.float32)  # [k, D]
        self.is_null_calibrated = bool(self.manifest["nullspace"]["is_calibrated"])
        self.cartridges: Dict[str, Dict[str, Any]] = {}
        for cm in self.manifest["cartridges"]:
            a_u16 = np.array(cm["a_scar_f16_b64"], dtype=np.uint16)
            b_u16 = np.array(cm["b_scar_f16_b64"], dtype=np.uint16)
            a_f32 = np.frombuffer(a_u16.tobytes(), dtype=np.float16).astype(np.float32)
            b_f32 = np.frombuffer(b_u16.tobytes(), dtype=np.float16).astype(np.float32)
            self.cartridges[cm["cartridge_id"]] = {
                "tau_cos": float(cm["tau_cos"]),
                "z_anchor": np.array(cm["z_anchor"], dtype=np.float32),
                "packed_f32": np.concatenate([np.array([1.0], dtype=np.float32), a_f32, b_f32]),
                "wiped": False,
            }

        self.reset_scache()

    def reset_scache(self) -> None:
        self.device.queue.write_buffer(self.scache_kv_buf, 0, bytes(self.scache_kv_buf.size))
        self.device.queue.write_buffer(self.scache_w_scar_buf, 0, bytes(self.scache_w_scar_buf.size))
        self.active_cartridge_id = None

    def set_active_cartridge(self, cid: Optional[str]) -> None:
        if cid is None or cid not in self.cartridges or self.cartridges[cid]["wiped"]:
            self.active_cartridge_id = None
            return
        self.active_cartridge_id = cid
        self.device.queue.write_buffer(self.active_cap_buf, 0, self.cartridges[cid]["packed_f32"].tobytes())

    def unlearn_cartridge(self, cid: str) -> None:
        if cid in self.cartridges:
            self.cartridges[cid]["packed_f32"].fill(0.0)
            self.cartridges[cid]["z_anchor"].fill(0.0)
            self.cartridges[cid]["wiped"] = True
        if self.active_cartridge_id == cid:
            self.device.queue.write_buffer(self.active_cap_buf, 0, bytes(self.active_cap_buf.size))
            self.active_cartridge_id = None

    def _u_buf(self, raw_params: bytes) -> Any:
        U = self.wgpu.BufferUsage
        return self.device.create_buffer_with_data(data=raw_params, usage=U.UNIFORM)

    def _bg(self, ep: str, bindings: Dict[int, Any]) -> Any:
        layout = self.pipelines[ep].get_bind_group_layout(0)
        entries = [{"binding": k, "resource": {"buffer": v, "offset": 0, "size": v.size}} for k, v in bindings.items()]
        return self.device.create_bind_group(layout=layout, entries=entries)

    def _dispatch_vq4_linear(self, cpass: Any, spec: Dict[str, int], x_buf: Any, y_buf: Any) -> None:
        in_dim = int(spec["in_dim"])
        out_dim = int(spec["out_dim"])
        rank = int(spec["rank"])

        u_eq = self._u_buf(_pack_params(p0=in_dim, p1=spec["d_eq_u32_off"]))
        cpass.set_pipeline(self.pipelines["vq4_equilibrate_fwht64"])
        cpass.set_bind_group(
            0, self._bg("vq4_equilibrate_fwht64", {0: u_eq, 1: self.shard_buf, 3: x_buf, 6: self.vq4_rot_buf}), [], 0, 0
        )
        cpass.dispatch_workgroups(in_dim >> 6, 1, 1)

        if rank > 0:
            u_a = self._u_buf(
                _pack_params(
                    p0=rank,
                    p1=in_dim,
                    p2=0,
                    p3=spec["ascar_packed_u32_off"],
                    p4=spec["ascar_scales_u32_off"],
                )
            )
            cpass.set_pipeline(self.pipelines["mos_fused_int4_bscar_gemv"])
            cpass.set_bind_group(
                0,
                self._bg(
                    "mos_fused_int4_bscar_gemv",
                    {0: u_a, 1: self.shard_buf, 3: x_buf, 4: self.dummy_buf, 6: self.asvd_u_buf},
                ),
                [],
                0,
                0,
            )
            cpass.dispatch_workgroups((rank + 7) >> 3, 1, 1)

        u_vq = self._u_buf(
            _pack_params(
                p0=out_dim,
                p1=in_dim,
                p2=rank,
                p3=spec["vq_codes_u32_off"],
                p4=spec["vq_scales_u32_off"],
                p5=spec["lut256_u32_off"],
                p6=spec["bscar_u32_off"],
                p7=0,
            )
        )
        cpass.set_pipeline(self.pipelines["mos_fused_vq4_int4scar_gemv"])
        cpass.set_bind_group(
            0,
            self._bg(
                "mos_fused_vq4_int4scar_gemv",
                {0: u_vq, 1: self.shard_buf, 3: self.vq4_rot_buf, 4: self.asvd_u_buf, 6: y_buf},
            ),
            [],
            0,
            0,
        )
        cpass.dispatch_workgroups((out_dim + 7) >> 3, 1, 1)

    def _read_f32_buffer(self, map_buf: Any, count: int) -> np.ndarray:
        self.device.queue.on_submitted_work_done_sync()
        map_buf.map_sync(self.wgpu.MapMode.READ)
        raw = bytes(map_buf.read_mapped())
        map_buf.unmap()
        return np.frombuffer(raw, dtype=np.float32)[:count].copy()

    def step_token_gpu(
        self,
        token_id: int,
        pos_t: int,
        suppress_cjk: bool = True,
        route_on_golden: bool = False,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Executes a single token in `O(1)` GPU memory (`Δ = 0 B`), optionally routing at Golden Slice."""
        c = self.cfg
        q_dim = c.num_q_heads * c.head_dim
        kv_dim = c.num_kv_heads * c.head_dim
        total_rope_pairs = (c.num_q_heads + c.num_kv_heads) * (c.head_dim // 2)
        attn_scale = 1.0 / math.sqrt(float(c.head_dim))

        step_bytes = struct.pack("<4I", int(token_id), int(pos_t), 0, 0)
        self.device.queue.write_buffer(self.step_state_buf, 0, step_bytes)

        def _encode_layer_range(enc: Any, start_l: int, end_l: int) -> None:
            cp = enc.begin_compute_pass()
            if start_l == 0:
                emb = self.manifest["embed"]
                u_emb = self._u_buf(
                    _pack_params(
                        p0=c.hidden_size,
                        p1=emb["packed_u32_off"],
                        p2=emb["scales_u32_off"],
                        p3=0,
                        p4=c.vocab_size,
                    )
                )
                cp.set_pipeline(self.pipelines["embed_int4_lookup"])
                cp.set_bind_group(
                    0,
                    self._bg(
                        "embed_int4_lookup",
                        {0: u_emb, 1: self.shard_buf, 2: self.step_state_buf, 6: self.h_cur_buf},
                    ),
                    [],
                    0,
                    0,
                )
                cp.dispatch_workgroups((c.hidden_size + 255) >> 8, 1, 1)

            for l_i in range(start_l, end_l):
                lm = self.manifest["layers"][l_i]
                projs = lm["projs"]
                u_ln1 = self._u_buf(_pack_params(p0=c.hidden_size, p1=lm["ln1_u32_off"], f0=c.rms_norm_eps))
                cp.set_pipeline(self.pipelines["rmsnorm_f32"])
                cp.set_bind_group(
                    0, self._bg("rmsnorm_f32", {0: u_ln1, 1: self.shard_buf, 3: self.h_cur_buf, 6: self.norm_buf}), [], 0, 0
                )
                cp.dispatch_workgroups(1, 1, 1)

                self._dispatch_vq4_linear(cp, projs["q_proj"], self.norm_buf, self.q_buf)
                self._dispatch_vq4_linear(cp, projs["k_proj"], self.norm_buf, self.k_buf)
                self._dispatch_vq4_linear(cp, projs["v_proj"], self.norm_buf, self.v_buf)

                # Save Pre-RoPE Q_raw and K_raw into the prefixes of `q_raw_alpha_bufs[l_i]` and `k_raw_gate_bufs[l_i]`
                u_cq = self._u_buf(_pack_params(p0=q_dim))
                cp.set_pipeline(self.pipelines["vec_copy_f32"])
                cp.set_bind_group(0, self._bg("vec_copy_f32", {0: u_cq, 3: self.q_buf, 6: self.q_raw_alpha_bufs[l_i]}), [], 0, 0)
                cp.dispatch_workgroups((q_dim + 255) >> 8, 1, 1)

                u_ck = self._u_buf(_pack_params(p0=kv_dim))
                cp.set_pipeline(self.pipelines["vec_copy_f32"])
                cp.set_bind_group(0, self._bg("vec_copy_f32", {0: u_ck, 3: self.k_buf, 6: self.k_raw_gate_bufs[l_i]}), [], 0, 0)
                cp.dispatch_workgroups((kv_dim + 255) >> 8, 1, 1)

                # Apply RoPE to `q_buf` and `k_buf`
                u_rope = self._u_buf(_pack_params(p0=c.num_q_heads, p1=c.num_kv_heads, p2=c.head_dim))
                cp.set_pipeline(self.pipelines["qwen_rope_inplace"])
                cp.set_bind_group(
                    0,
                    self._bg(
                        "qwen_rope_inplace",
                        {0: u_rope, 2: self.step_state_buf, 3: self.inv_freq_buf, 6: self.q_buf, 7: self.k_buf},
                    ),
                    [],
                    0,
                    0,
                )
                cp.dispatch_workgroups((total_rope_pairs + 63) >> 6, 1, 1)

                # Shader #21: Surprise-Gated Widrow-Hoff Eviction + O(1) Ring Update
                u_wh = self._u_buf(
                    _pack_params(
                        p0=l_i,
                        p1=c.num_kv_heads,
                        p2=c.head_dim,
                        p3=c.num_sink,
                        p4=c.slide_len,
                        p5=self.counts_u32_total,
                    )
                )
                cp.set_pipeline(self.pipelines["scarnative_widrow_hoff_evict_update"])
                cp.set_bind_group(
                    0,
                    self._bg(
                        "scarnative_widrow_hoff_evict_update",
                        {
                            0: u_wh,
                            2: self.step_state_buf,
                            3: self.k_buf,
                            4: self.v_buf,
                            5: self.k_raw_gate_bufs[l_i],
                            6: self.scache_w_scar_buf,
                            8: self.scache_kv_buf,
                        },
                    ),
                    [],
                    0,
                    0,
                )
                cp.dispatch_workgroups(c.num_kv_heads, 1, 1)

                # Shader #22: Hybrid O(1) RoPE Sink+Window + Content-Addressed Fast-Weight Attention
                u_att = self._u_buf(
                    _pack_params(
                        p0=l_i,
                        p1=c.num_q_heads,
                        p2=c.num_kv_heads,
                        p3=c.head_dim,
                        p4=c.num_sink,
                        p5=c.slide_len,
                        p6=self.counts_u32_total,
                        f0=attn_scale,
                    )
                )
                cp.set_pipeline(self.pipelines["scarnative_o1_hybrid_gqa_attention"])
                cp.set_bind_group(
                    0,
                    self._bg(
                        "scarnative_o1_hybrid_gqa_attention",
                        {
                            0: u_att,
                            1: self.scache_kv_buf,
                            3: self.q_buf,
                            4: self.scache_w_scar_buf,
                            5: self.q_raw_alpha_bufs[l_i],
                            6: self.attn_out_buf,
                        },
                    ),
                    [],
                    0,
                    0,
                )
                cp.dispatch_workgroups(c.num_q_heads, 1, 1)

                self._dispatch_vq4_linear(cp, projs["o_proj"], self.attn_out_buf, self.proj_out_buf)

                u_add = self._u_buf(_pack_params(p0=c.hidden_size))
                cp.set_pipeline(self.pipelines["vec_add_inplace_f32"])
                cp.set_bind_group(0, self._bg("vec_add_inplace_f32", {0: u_add, 3: self.proj_out_buf, 6: self.h_cur_buf}), [], 0, 0)
                cp.dispatch_workgroups((c.hidden_size + 255) >> 8, 1, 1)

                # MLP
                u_ln2 = self._u_buf(_pack_params(p0=c.hidden_size, p1=lm["ln2_u32_off"], f0=c.rms_norm_eps))
                cp.set_pipeline(self.pipelines["rmsnorm_f32"])
                cp.set_bind_group(
                    0, self._bg("rmsnorm_f32", {0: u_ln2, 1: self.shard_buf, 3: self.h_cur_buf, 6: self.norm_buf}), [], 0, 0
                )
                cp.dispatch_workgroups(1, 1, 1)

                self._dispatch_vq4_linear(cp, projs["gate_proj"], self.norm_buf, self.gate_buf)
                self._dispatch_vq4_linear(cp, projs["up_proj"], self.norm_buf, self.up_buf)

                u_sw = self._u_buf(_pack_params(p0=c.intermediate_size))
                cp.set_pipeline(self.pipelines["swiglu_mul_f32"])
                cp.set_bind_group(0, self._bg("swiglu_mul_f32", {0: u_sw, 3: self.up_buf, 6: self.gate_buf}), [], 0, 0)
                cp.dispatch_workgroups((c.intermediate_size + 255) >> 8, 1, 1)

                self._dispatch_vq4_linear(cp, projs["down_proj"], self.gate_buf, self.proj_out_buf)
                cp.set_pipeline(self.pipelines["vec_add_inplace_f32"])
                cp.set_bind_group(0, self._bg("vec_add_inplace_f32", {0: u_add, 3: self.proj_out_buf, 6: self.h_cur_buf}), [], 0, 0)
                cp.dispatch_workgroups((c.hidden_size + 255) >> 8, 1, 1)

                # Inject active Golden-Slice Cartridge when not paused for routing
                if (not route_on_golden) and (l_i + 1 == c.golden_layer) and (self.active_cartridge_id is not None):
                    u_gs = self._u_buf(
                        _pack_params(p0=c.hidden_size, p1=c.cartridge_rank, p2=1, p3=0, f0=1.0)
                    )
                    cp.set_pipeline(self.pipelines["golden_slice_scar_inject"])
                    cp.set_bind_group(
                        0,
                        self._bg(
                            "golden_slice_scar_inject",
                            {
                                0: u_gs,
                                3: self.zero_mu_buf,
                                5: self.active_cap_buf,
                                6: self.h_cur_buf,
                                7: self.scar_telem_buf,
                            },
                        ),
                        [],
                        0,
                        0,
                    )
                    cp.dispatch_workgroups(1, 1, 1)
            cp.end()

        if route_on_golden:
            enc1 = self.device.create_command_encoder()
            _encode_layer_range(enc1, 0, c.golden_layer)
            enc1.copy_buffer_to_buffer(self.h_cur_buf, 0, self.golden_read_buf, 0, c.hidden_size * 4)
            self.device.queue.submit([enc1.finish()])
            h_gold = self._read_f32_buffer(self.golden_read_buf, c.hidden_size)

            if self.is_null_calibrated:
                proj = h_gold - (h_gold @ self.u_lang_t.T) @ self.u_lang_t
            else:
                proj = h_gold
            zq = proj / max(float(np.linalg.norm(proj)), 1e-6)
            best_cid: Optional[str] = None
            best_sim = -1.0
            for cid, cm in self.cartridges.items():
                if cm["wiped"]:
                    continue
                s = float(np.dot(zq, cm["z_anchor"]))
                if s > best_sim:
                    best_sim = s
                    best_cid = cid
            if best_cid is not None and best_sim >= self.cartridges[best_cid]["tau_cos"]:
                self.set_active_cartridge(best_cid)
            else:
                self.set_active_cartridge(None)

            enc2 = self.device.create_command_encoder()
            if self.active_cartridge_id is not None:
                cp_inj = enc2.begin_compute_pass()
                u_gs = self._u_buf(_pack_params(p0=c.hidden_size, p1=c.cartridge_rank, p2=1, p3=0, f0=1.0))
                cp_inj.set_pipeline(self.pipelines["golden_slice_scar_inject"])
                cp_inj.set_bind_group(
                    0,
                    self._bg(
                        "golden_slice_scar_inject",
                        {0: u_gs, 3: self.zero_mu_buf, 5: self.active_cap_buf, 6: self.h_cur_buf, 7: self.scar_telem_buf},
                    ),
                    [],
                    0,
                    0,
                )
                cp_inj.dispatch_workgroups(1, 1, 1)
                cp_inj.end()
            _encode_layer_range(enc2, c.golden_layer, c.num_layers)
            self._encode_head_and_copy(enc2, suppress_cjk)
            self.device.queue.submit([enc2.finish()])
            return self._read_f32_buffer(self.logits_read_buf, c.vocab_size), self.active_cartridge_id

        enc = self.device.create_command_encoder()
        _encode_layer_range(enc, 0, c.num_layers)
        self._encode_head_and_copy(enc, suppress_cjk)
        self.device.queue.submit([enc.finish()])
        return self._read_f32_buffer(self.logits_read_buf, c.vocab_size), self.active_cartridge_id

    def _encode_head_and_copy(self, enc: Any, suppress_cjk: bool) -> None:
        c = self.cfg
        cp = enc.begin_compute_pass()
        u_fn = self._u_buf(_pack_params(p0=c.hidden_size, p1=self.manifest["final_norm_u32_off"], f0=c.rms_norm_eps))
        cp.set_pipeline(self.pipelines["rmsnorm_f32"])
        cp.set_bind_group(
            0, self._bg("rmsnorm_f32", {0: u_fn, 1: self.shard_buf, 3: self.h_cur_buf, 6: self.norm_buf}), [], 0, 0
        )
        cp.dispatch_workgroups(1, 1, 1)

        self._dispatch_vq4_linear(cp, self.manifest["lm_head"], self.norm_buf, self.logits_buf)

        u_gate = self._u_buf(_pack_params(p0=c.vocab_size, p1=1 if suppress_cjk else 0, f0=64.0))
        cp.set_pipeline(self.pipelines["apply_script_subspace_gate"])
        cp.set_bind_group(
            0, self._bg("apply_script_subspace_gate", {0: u_gate, 1: self.cjk_mask_buf, 6: self.logits_buf}), [], 0, 0
        )
        cp.dispatch_workgroups((c.vocab_size + 255) >> 8, 1, 1)
        cp.end()

        enc.copy_buffer_to_buffer(self.logits_buf, 0, self.logits_read_buf, 0, c.vocab_size * 4)
