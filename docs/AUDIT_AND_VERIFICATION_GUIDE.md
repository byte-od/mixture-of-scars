# Forensic Audit & Independent Verification Guide (`v4.0.0`)

> **Zero-Facades / Zero-Shortcuts Engineering Mandate:**  
> Every claim in **Mixture of Scars (`MoS`) `v4.0.0`**, **`SCache` $\mathcal{O}(1)$**, and domain-independent **`ScarCodec`** is backed by pure, inspectable PyTorch/Metal/Triton/WGSL source code in [`scar_cache/`](../scar_cache/) and [`scar_codec/`](../scar_codec/), automated physical byte-pointer checks (`untyped_storage().data_ptr()` & fixed `GPUBuffer` bindings), **`114/114`** executable `pytest` tests across `29` suites in [`tests/`](../tests/), and committed raw JSON telemetry in [`results/`](../results/) and [`benchmark_results/`](../benchmark_results/).

---

## 1. 60-Second End-to-End Audit Commands

```bash
# 1. Run the complete 114-test forensic, mathematical, 20-shader WGSL hardware, Physical ScarCodec, and red-team verification suite:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/ -v

# 2. Run ONLY the dedicated v4.0.0 anti-facade, symbol-parity, 7-CLI, scope-boundary, and JSON-proof audit (< 3s):
pytest tests/test_v3_architectural_integrity.py -v

# 3. Verify all 7 installed CLI entry points respond cleanly:
mos-run --help
mos-store --help
mos-graft --help
mos-rebasin --help
mos-transcode-sub2bit --help
mos-webgpu --help
mos-scar-codec --help
```

---

## 2. Claim-to-Code-to-Test Traceability Matrix (`Preprint v4.0`)

| Preprint `v4.0` Pillar & Empirical Metric | Canonical Implementation (`scar_cache/` & `scar_codec/`) | Automated Test (`tests/`) | Replication Script & Raw JSON Proof |
|---|---|---|---|
| **Pillar I-A: Canonical `72.71B` & Zero-Spike Student `Block-64 INT4 + ASVD`** (`72.71B`: `38.12 GiB`, `20 × 4L` shards, `50.38×` Apple Metal `int4pack`, CUDA Triton `W4A16+ASVD`, `100%` Top-5; `3B` Student: `3.49 GiB` VRAM & `0.55 GiB` CPU RSS, `0` load spike; `0.5B`: `0.78 GiB` VRAM) | [`cli_72b.py`](../scar_cache/cli_72b.py), [`mos_student.py`](../scar_cache/mos_student.py), [`triton_mos_gemm.py`](../scar_cache/triton_mos_gemm.py) | [`test_canonical_mos_v2.py`](../tests/test_canonical_mos_v2.py), [`test_triton_mos_gemm.py`](../tests/test_triton_mos_gemm.py), [`test_real_70b_mos_loader.py`](../tests/test_real_70b_mos_loader.py), [`test_cli_scar_store.py`](../tests/test_cli_scar_store.py) | [`benchmark_72b_apple_silicon.py`](../scripts/benchmark_72b_apple_silicon.py), [`adversarial_scale_72b_suite.py`](../scripts/adversarial_scale_72b_suite.py) → [`apple_m2_72b_speed_memory_audit.json`](../results/scale_72b_audit/apple_m2_72b_speed_memory_audit.json), [`adversarial_scale_72b_report.json`](../results/scale_72b_audit/adversarial_scale_72b_report.json) |
| **Pillar I-B: `Sub-2-Bit MoS` (`2.25 bpw` + Two-Pass Alternating `ASVD`)** (`72.71B` in `20.65 GiB` shards / `21.89 GiB` VRAM `< 24 GB` GPU, `cos = 0.99911` on `L79`, `12,290×` PPL reduction on `3B`) | [`sub2bit_mos.py`](../scar_cache/sub2bit_mos.py), [`triton_mos_gemm.py`](../scar_cache/triton_mos_gemm.py) | [`test_horizon1_v1_4.py`](../tests/test_horizon1_v1_4.py), [`test_v3_architectural_integrity.py`](../tests/test_v3_architectural_integrity.py) | [`benchmark_horizon1_v1_4.py`](../scripts/benchmark_horizon1_v1_4.py) → [`horizon1_v1_4_audit.json`](../results/horizon1_v1_4_audit.json) |
| **Pillar II-A: `SCache` $\mathcal{O}(1)$ (`16.6k` & `32.9k` Passkey)** (`22.71 MiB` locked on `3B`, `Δ = +0 B` drift, `50.9×` smaller & `5.84×` faster than Normal KV, `100%` recall, Widrow-Hoff orthogonal erasure) | [`hf_cache.py`](../scar_cache/hf_cache.py), [`core.py`](../scar_cache/core.py), [`multi_slot.py`](../scar_cache/multi_slot.py), [`projections.py`](../scar_cache/projections.py) | [`test_hf_cache.py`](../tests/test_hf_cache.py), [`test_core.py`](../tests/test_core.py), [`test_multi_slot.py`](../tests/test_multi_slot.py), [`test_adversarial_mos_scache.py`](../tests/test_adversarial_mos_scache.py) | [`benchmark_16k_passkey_llm.py`](../scripts/benchmark_16k_passkey_llm.py) → [`passkey_16k_32k_llm_audit.json`](../results/passkey_16k_32k_llm_audit.json) |
| **Pillar II-B: `Ghost-Jump` Speculative Reflex + Atomic `SCache` Rollback** (`1.89×–1.92×` MPS speedup, `KL = 0.0000` exact parity, `0 B` leak) | [`speculative_ghost_jump.py`](../scar_cache/speculative_ghost_jump.py), [`hf_cache.py`](../scar_cache/hf_cache.py) | [`test_horizon1_v1_4.py`](../tests/test_horizon1_v1_4.py) | [`benchmark_horizon1_v1_4.py`](../scripts/benchmark_horizon1_v1_4.py) → [`horizon1_v1_4_audit.json`](../results/horizon1_v1_4_audit.json) |
| **Pillar III: Closed-Form `140.00 KB` `HeadExpandedScarStore` & Language-Nullspace `GoldenSliceScarStore v4.0` ($P_{\perp\text{lang}}$)** (`0` SGD, `72.71B → 3B` `98.61%` unseen Top-1, `41 × 140 KB` `Layer 23/36` `HTTP/1.1 206` cartridges hitting `164/164 = 100.0%` across `PT/EN/ES/FR`, `18 µs` RAM / `7.64 ms` `0x00` SSD GDPR wipe) | [`head_expanded_attention.py`](../scar_cache/head_expanded_attention.py), [`golden_slice_v4.py`](../scar_cache/golden_slice_v4.py), [`cli_scar_store.py`](../scar_cache/cli_scar_store.py), [`ui_scar_store.py`](../scar_cache/ui_scar_store.py), [`open_weights_factory.py`](../scar_cache/open_weights_factory.py), [`scar_grafting.py`](../scar_cache/scar_grafting.py), [`scar_store_v2.py`](../scar_cache/scar_store_v2.py) | [`test_scar_grafting_v2.py`](../tests/test_scar_grafting_v2.py), [`test_scar_store_v2_stage1.py`](../tests/test_scar_store_v2_stage1.py)–[`stage5_adversarial.py`](../tests/test_scar_store_v2_stage5_adversarial.py), [`test_golden_slice_v4.py`](../tests/test_golden_slice_v4.py), [`test_cli_scar_store.py`](../tests/test_cli_scar_store.py) | [`build_universal_scarstore_v4_0.py`](../scripts/build_universal_scarstore_v4_0.py), [`build_online_cartridge_library_v3_0.py`](../scripts/build_online_cartridge_library_v3_0.py) → [`qwen3b_portfolio_v4_3_prot_int4.manifest.json`](../results/online_library_v3_0/qwen3b_portfolio_v4_3_prot_int4.manifest.json), [`scar_store_v2_stage5_3b_vs_0p5b_audit.json`](../results/scar_store_v2_stage5_3b_vs_0p5b_audit.json), [`adversarial_0p5b_vs_3b_audit.json`](../results/adversarial_0p5b_vs_3b_audit.json), [`mos_store_v2_3_cli_http206_audit.json`](../results/mos_store_v2_3_cli_http206_audit.json) |
| **Pillar IV: *Sinkhorn Monomial–Procrustes Re-Basin* & Marchenko-Pastur Law** (`56.36×` specialist KL drop, `KL = 0.0353 < 0.08` on `3B` `L22–L23` at `5.09×` compression & `0` extra FLOPs, `14.68×` on `12` layers, `< 350 MiB` tiled `72B` log-Sinkhorn) | [`sinkhorn_rebasin.py`](../scar_cache/sinkhorn_rebasin.py) | [`test_sinkhorn_rebasin.py`](../tests/test_sinkhorn_rebasin.py), [`test_v3_architectural_integrity.py`](../tests/test_v3_architectural_integrity.py) | [`audit_sinkhorn_rebasin_v2_4.py`](../scripts/audit_sinkhorn_rebasin_v2_4.py) → [`sinkhorn_rebasin_v2_4_audit.json`](../results/sinkhorn_rebasin_v2_4_audit.json) |
| **Pillar V: Unified `Native + 20-Shader WebGPU WGSL 1.0` (`VQ4-Scar`, `ProtectedInt4Scar`, `P_⊥CJK`)** (`0.992 GiB` Sub-1GB `VQ4-Scar`, `1,190.5 MiB` Shielded `VQ4-Scar`, `1,601.6 MiB` `ProtectedInt4Scar` at `KL = 0.6938`, `0%` CJK leak, `cos = 1.00000012` WGSL parity, `~21 tok/s` `3B` / `71.85 tok/s` `0.5B`) | [`vq4_scar_mos.py`](../scar_cache/vq4_scar_mos.py), [`webgpu_engine.py`](../scar_cache/webgpu_engine.py), [`mos_scache_kernels.wgsl`](../scar_cache/webgpu/mos_scache_kernels.wgsl), [`mos_webgpu_engine.js`](../scar_cache/webgpu/app/mos_webgpu_engine.js), [`scarstore_http206_browser.js`](../scar_cache/webgpu/app/scarstore_http206_browser.js) | [`test_vq4_scar_native_and_webgpu.py`](../tests/test_vq4_scar_native_and_webgpu.py), [`test_webgpu_wgsl_engine.py`](../tests/test_webgpu_wgsl_engine.py) | [`verify_track24c_shielded_3b_polyglot.py`](../scripts/verify_track24c_shielded_3b_polyglot.py), `mos-webgpu verify` → [`webgpu_wgsl_v3_0_audit.json`](../results/webgpu_wgsl_v3_0_audit.json) |
| **Pillar VI: Domain-Independent Physical `ScarCodec` (`mos-scar-codec`, 12 Hardware/Space/Science Domains)** (`-60 dB` `1,000×` RFI `cos = 0.9752` at `8.67×`; `34,816 B` `Δ = 0 B` Black-Box `SCache` `5/5` fault recall; `[512,512]` CFD/`500×` HDR `12.05×`, `75.0×` lower RMSE than FFT, `0.0` vacuum error, `9.2×` faster `O(1)` read) | [`scar_codec/core.py`](../scar_codec/core.py), [`scar_codec/cli.py`](../scar_codec/cli.py) | [`test_scar_codec_physical.py`](../tests/test_scar_codec_physical.py), [`test_v3_architectural_integrity.py`](../tests/test_v3_architectural_integrity.py) | [`benchmark_scar_codec_physical_v4_0.py`](../scripts/benchmark_scar_codec_physical_v4_0.py) → [`scar_codec_physical_v4_0_audit.json`](../results/scar_codec_physical_v4_0_audit.json) |

---

## 3. Architectural Scope & Anti-Facade Guarantees

1. **Physical Pointer Auditing (`0.00 B` Discrepancy):**
   - `recursive_scache_storage_bytes(cache)` (in [`speculative_ghost_jump.py`](../scar_cache/speculative_ghost_jump.py)) and `audit_unique_module_bytes(module)` (in [`sinkhorn_rebasin.py`](../scar_cache/sinkhorn_rebasin.py)) walk the entire Python/PyTorch object graph via `tensor.untyped_storage().data_ptr()` to prove that no hidden $\mathcal{O}(T)$ buffers or duplicate weight copies exist behind reported metrics.
2. **Genuine Bit-Packed Tensors (Zero Float Masks):**
   - `CanonicalMoSGroup` & `ProtectedInt4ScarMatrix` (`INT4`) pack 2 weights per `torch.uint8` byte (`packed_base.numel() * 2 == N * K`).
   - `VQ4ScarMatrix` & `ScarCodecPacket` pack 4D lattice vector codes at `1 uint8` per 4 scalars (`lut_codes.numel() * 4 == N * K`) with `LUT256[0] = (0,0,0,0)` exact vacuum invariance.
   - `Sub2BitMoSLinear` (`2.25 bpw`) packs 4 quaternary codes per `torch.uint8` byte (`packed_2bit.numel() * 4 == N * K`).
   - `HeadExpandedScarStore` & `GoldenSliceScarStoreV4` write raw `4,096 B`-aligned binary slots (`143,360 B = 35 × 4,096 B` on `3B`) accessed via OS `mmap` or RFC 7233 `HTTP/1.1 206 Range`, and `mos-store unlearn --persist` overwrites the exact slot and anchor offsets on disk with `0x00` + `os.fsync()`.
3. **Clear Distinction Between `CanonicalMoSEngine` (`72.71B` Real Weights) and `NativeMoS70B` (Recurrent Depth Expansion):**
   - [`scar_cache/cli_72b.py`](../scar_cache/cli_72b.py) (`CanonicalMoSEngine`) executes the **real `80`-layer `Qwen2.5-72B-Instruct` (`72.71B`)** pretrained weights across `20` SHA-256 verified shards ([`models/qwen25_72b_mos_v2/mos_manifest.json`](../models/qwen25_72b_mos_v2/mos_manifest.json)).
   - [`scar_cache/native_70b.py`](../scar_cache/native_70b.py) (`NativeMoS70B`) is explicitly documented and tested ([`tests/test_native_70b.py`](../tests/test_native_70b.py)) as a cyclic Universal-Transformer depth-expansion topology (`base_layers < num_layers`) over compact backbones—never conflated with pretrained `72.71B` benchmarks.
