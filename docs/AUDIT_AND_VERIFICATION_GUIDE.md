# Forensic Audit & Independent Verification Guide (`v3.0.0`)

> **Zero-Facades / Zero-Shortcuts Engineering Mandate:**  
> Every claim in **Mixture of Scars (`MoS`) `v3.0.0`** and **`SCache` $\mathcal{O}(1)$** is backed by pure, inspectable PyTorch/Metal/Triton source code in [`scar_cache/`](../scar_cache/), automated physical byte-pointer checks (`untyped_storage().data_ptr()`), **`88/88`** executable `pytest` tests across `25` suites in [`tests/`](../tests/), and committed raw JSON telemetry in [`results/`](../results/) and [`benchmark_results/`](../benchmark_results/).

---

## 1. 60-Second End-to-End Audit Commands

```bash
# 1. Run the complete 88-test forensic, mathematical, and red-team verification suite:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/ -v

# 2. Run ONLY the dedicated v3.0.0 anti-facade, symbol-parity, CLI, and JSON-proof audit (< 3s):
pytest tests/test_v3_architectural_integrity.py -v

# 3. Verify all 5 installed CLI entry points respond cleanly:
mos-run --help
mos-store --help
mos-graft --help
mos-rebasin --help
mos-transcode-sub2bit --help
```

---

## 2. Claim-to-Code-to-Test Traceability Matrix (`Preprint v3.0`)

| Preprint `v3.0` Pillar & Empirical Metric | Canonical Implementation (`scar_cache/`) | Automated Test (`tests/`) | Replication Script & Raw JSON Proof |
|---|---|---|---|
| **Pillar I-A: Canonical `72.71B` & Zero-Spike Student `Block-64 INT4 + ASVD`** (`72.71B`: `38.12 GiB`, `20 × 4L` shards, `50.38×` Apple Metal `int4pack`, CUDA Triton `W4A16+ASVD`, `100%` Top-5; `3B` Student: `3.49 GiB` VRAM & `0.55 GiB` CPU RSS, `0` load spike; `0.5B`: `0.78 GiB` VRAM) | [`cli_72b.py`](../scar_cache/cli_72b.py), [`mos_student.py`](../scar_cache/mos_student.py), [`triton_mos_gemm.py`](../scar_cache/triton_mos_gemm.py) | [`test_canonical_mos_v2.py`](../tests/test_canonical_mos_v2.py), [`test_triton_mos_gemm.py`](../tests/test_triton_mos_gemm.py), [`test_real_70b_mos_loader.py`](../tests/test_real_70b_mos_loader.py), [`test_cli_scar_store.py`](../tests/test_cli_scar_store.py) | [`benchmark_72b_apple_silicon.py`](../scripts/benchmark_72b_apple_silicon.py), [`adversarial_scale_72b_suite.py`](../scripts/adversarial_scale_72b_suite.py) → [`apple_m2_72b_speed_memory_audit.json`](../results/scale_72b_audit/apple_m2_72b_speed_memory_audit.json), [`adversarial_scale_72b_report.json`](../results/scale_72b_audit/adversarial_scale_72b_report.json) |
| **Pillar I-B: `Sub-2-Bit MoS` (`2.25 bpw` + Two-Pass Alternating `ASVD`)** (`72.71B` in `20.65 GiB` shards / `21.89 GiB` VRAM `< 24 GB` GPU, `cos = 0.99911` on `L79`, `12,290×` PPL reduction on `3B`) | [`sub2bit_mos.py`](../scar_cache/sub2bit_mos.py), [`triton_mos_gemm.py`](../scar_cache/triton_mos_gemm.py) | [`test_horizon1_v1_4.py`](../tests/test_horizon1_v1_4.py), [`test_v3_architectural_integrity.py`](../tests/test_v3_architectural_integrity.py) | [`benchmark_horizon1_v1_4.py`](../scripts/benchmark_horizon1_v1_4.py) → [`horizon1_v1_4_audit.json`](../results/horizon1_v1_4_audit.json) |
| **Pillar II-A: `SCache` $\mathcal{O}(1)$ (`16.6k` & `32.9k` Passkey)** (`22.71 MiB` locked on `3B`, `Δ = +0 B` drift, `50.9×` smaller & `5.84×` faster than Normal KV, `100%` recall, Widrow-Hoff orthogonal erasure) | [`hf_cache.py`](../scar_cache/hf_cache.py), [`core.py`](../scar_cache/core.py), [`multi_slot.py`](../scar_cache/multi_slot.py), [`projections.py`](../scar_cache/projections.py) | [`test_hf_cache.py`](../tests/test_hf_cache.py), [`test_core.py`](../tests/test_core.py), [`test_multi_slot.py`](../tests/test_multi_slot.py), [`test_adversarial_mos_scache.py`](../tests/test_adversarial_mos_scache.py) | [`benchmark_16k_passkey_llm.py`](../scripts/benchmark_16k_passkey_llm.py) → [`passkey_16k_32k_llm_audit.json`](../results/passkey_16k_32k_llm_audit.json) |
| **Pillar II-B: `Ghost-Jump` Speculative Reflex + Atomic `SCache` Rollback** (`1.89×–1.92×` MPS speedup, `KL = 0.0000` exact parity, `0 B` leak) | [`speculative_ghost_jump.py`](../scar_cache/speculative_ghost_jump.py), [`hf_cache.py`](../scar_cache/hf_cache.py) | [`test_horizon1_v1_4.py`](../tests/test_horizon1_v1_4.py) | [`benchmark_horizon1_v1_4.py`](../scripts/benchmark_horizon1_v1_4.py) → [`horizon1_v1_4_audit.json`](../results/horizon1_v1_4_audit.json) |
| **Pillar III: Closed-Form `140.00 KB` (`35 × 4 KB` `mmap`) `HeadExpandedScarStore` (`mos-store` CLI/UI, Live `HTTP 206` Library & Open-Weights Factory)** (`0` SGD, `72.71B → 3B` `98.61%` unseen Top-1, 9-slot GitHub CDN `HTTP/1.1 206` streaming, `same_vocab_exact` + `cross_tokenizer_bridge` minting, `18 µs` RAM / `7.64 ms` `0x00` SSD GDPR wipe, `6/6` red-team pass) | [`head_expanded_attention.py`](../scar_cache/head_expanded_attention.py), [`cli_scar_store.py`](../scar_cache/cli_scar_store.py), [`ui_scar_store.py`](../scar_cache/ui_scar_store.py), [`open_weights_factory.py`](../scar_cache/open_weights_factory.py), [`scar_grafting.py`](../scar_cache/scar_grafting.py), [`scar_store_v2.py`](../scar_cache/scar_store_v2.py), [`dual_depth_bridge.py`](../scar_cache/dual_depth_bridge.py) | [`test_scar_grafting_v2.py`](../tests/test_scar_grafting_v2.py), [`test_scar_store_v2_stage1.py`](../tests/test_scar_store_v2_stage1.py)–[`stage5_adversarial.py`](../tests/test_scar_store_v2_stage5_adversarial.py), [`test_cli_scar_store.py`](../tests/test_cli_scar_store.py) | [`build_online_cartridge_library_v3_0.py`](../scripts/build_online_cartridge_library_v3_0.py), [`audit_scar_store_v2_stage5_3b_vs_0p5b.py`](../scripts/audit_scar_store_v2_stage5_3b_vs_0p5b.py), [`adversarial_core_0p5b_vs_3b_suite.py`](../scripts/adversarial_core_0p5b_vs_3b_suite.py), [`audit_mos_store_cli_v2_3.py`](../scripts/audit_mos_store_cli_v2_3.py) → [`online_cartridge_library_v3_0_audit.json`](../results/online_cartridge_library_v3_0_audit.json), [`scar_store_v2_stage5_3b_vs_0p5b_audit.json`](../results/scar_store_v2_stage5_3b_vs_0p5b_audit.json), [`adversarial_0p5b_vs_3b_audit.json`](../results/adversarial_0p5b_vs_3b_audit.json), [`mos_store_v2_3_cli_http206_audit.json`](../results/mos_store_v2_3_cli_http206_audit.json) |
| **Pillar IV: *Sinkhorn Monomial–Procrustes Re-Basin* & Marchenko-Pastur Law** (`56.36×` specialist KL drop, `KL = 0.0353 < 0.08` on `3B` `L22–L23` at `5.09×` compression & `0` extra FLOPs, `14.68×` on `12` layers, `< 350 MiB` tiled `72B` log-Sinkhorn) | [`sinkhorn_rebasin.py`](../scar_cache/sinkhorn_rebasin.py) | [`test_sinkhorn_rebasin.py`](../tests/test_sinkhorn_rebasin.py), [`test_v3_architectural_integrity.py`](../tests/test_v3_architectural_integrity.py) | [`audit_sinkhorn_rebasin_v2_4.py`](../scripts/audit_sinkhorn_rebasin_v2_4.py) → [`sinkhorn_rebasin_v2_4_audit.json`](../results/sinkhorn_rebasin_v2_4_audit.json) |

---

## 3. Architectural Scope & Anti-Facade Guarantees

1. **Physical Pointer Auditing (`0.00 B` Discrepancy):**
   - `recursive_scache_storage_bytes(cache)` (in [`speculative_ghost_jump.py`](../scar_cache/speculative_ghost_jump.py)) and `audit_unique_module_bytes(module)` (in [`sinkhorn_rebasin.py`](../scar_cache/sinkhorn_rebasin.py)) walk the entire Python/PyTorch object graph via `tensor.untyped_storage().data_ptr()` to prove that no hidden $\mathcal{O}(T)$ buffers or duplicate weight copies exist behind reported metrics.
2. **Genuine Bit-Packed Tensors (Zero Float Masks):**
   - `CanonicalMoSGroup` (`INT4`) packs 2 weights per `torch.uint8` byte (`packed_base.numel() * 2 == N * K`).
   - `Sub2BitMoSLinear` (`2.25 bpw`) packs 4 quaternary codes per `torch.uint8` byte (`packed_2bit.numel() * 4 == N * K`).
   - `HeadExpandedScarStore` writes raw `4,096 B`-aligned binary slots (`143,360 B = 35 × 4,096 B` on `3B`) accessed via OS `mmap`, and `mos-store unlearn --persist` overwrites the exact slot and anchor offsets on disk with `0x00` + `os.fsync()`.
3. **Clear Distinction Between `CanonicalMoSEngine` (`72.71B` Real Weights) and `NativeMoS70B` (Recurrent Depth Expansion):**
   - [`scar_cache/cli_72b.py`](../scar_cache/cli_72b.py) (`CanonicalMoSEngine`) executes the **real `80`-layer `Qwen2.5-72B-Instruct` (`72.71B`)** pretrained weights across `20` SHA-256 verified shards ([`models/qwen25_72b_mos_v2/mos_manifest.json`](../models/qwen25_72b_mos_v2/mos_manifest.json)).
   - [`scar_cache/native_70b.py`](../scar_cache/native_70b.py) (`NativeMoS70B`) is explicitly documented and tested ([`tests/test_native_70b.py`](../tests/test_native_70b.py)) as a cyclic Universal-Transformer depth-expansion topology (`base_layers < num_layers`) over compact backbones—never conflated with pretrained `72.71B` benchmarks.
