# Mixture of Scars (MoS) & SCache $\mathcal{O}(1)$: Breaking the Dual Memory Wall Across `0.5B`–`72.71B` Transformers

[![DOI v3.0.0](https://zenodo.org/badge/DOI/10.5281/zenodo.24172784.svg)](https://doi.org/10.5281/zenodo.24172784)
[![Release: v3.0.0](https://img.shields.io/badge/Release-v3.0.0%20(Preprint%20v3.0%3A%20Sub--2--Bit%20%3C24GB%20%7C%20140KB%20ScarStore%20%7C%20Sinkhorn%20Re--Basin)-brightgreen.svg)](zenodo_deposit_v3_0/Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Drop-in](https://img.shields.io/badge/HuggingFace-DynamicCache%20Drop--in-yellow.svg)](https://huggingface.co/)
[![32.9k Passkey Recall](https://img.shields.io/badge/32.9k%20Passkey-100%25%20%40%2022.7%20MiB%20(5.84x%20Faster)-success.svg)](#1-166k--329k-passkey--needle-in-a-haystack-scache-o1-vs-normal-kv-cache)
[![72.7B Sub-2-Bit](https://img.shields.io/badge/72.7B%20Sub--2--Bit-20.65%20GiB%20Shards%20(%3C24%20GB%20VRAM)-9cf.svg)](#3-horizon-1-v140-ghost-jump-speculative-reflex--sub-2-bit-mos--24-gb-7271b)
[![Fused Metal, Triton & WebGPU WGSL](https://img.shields.io/badge/Fused%20Kernels-Metal%2050.4x%20%7C%20Triton%20W4%2FW2A16%20%7C%20WebGPU%20WGSL%2071.8%20tok%2Fs-blueviolet.svg)](scar_cache/webgpu/mos_scache_kernels.wgsl)
[![Tests Passing](https://img.shields.io/badge/Tests-95%2F95%20Passing-brightgreen.svg)](tests/)
[![Forensic Audit Guide](https://img.shields.io/badge/Audit%20Guide-v3.0%20Zero--Facade-0ea5e9.svg)](docs/AUDIT_AND_VERIFICATION_GUIDE.md)
[![72B MoS Download](https://img.shields.io/badge/72B%20MoS%20Model-38.12%20GiB%20Download-purple.svg)](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
[![Practical 72B Guide](https://img.shields.io/badge/Practical%2072B%20Guide-Docs-orange.svg)](docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
[![v2.x Roadmap](https://img.shields.io/badge/Roadmap-v1.4%20%E2%86%92%20v3.x-00b4d8.svg)](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md)
[![Sponsor Research](https://img.shields.io/badge/Sponsor-Byte--OD%20Research-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/byte-od)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/thetod/mixture-of-scars-benchmark)

> **Unified Scar Operator (`v3.0.0`):** Autoregressive Transformers hit four compounding hardware walls: static parameter bloat $\mathcal{O}(N)$ (`135.43 GiB` FP16 at `72.71B`), linear/quadratic KV-cache growth $\mathcal{O}(T) / \mathcal{O}(T^2)$, memory-bound single-token decode, and backpropagation-bound specialization. **Mixture of Scars (`MoS v3.0`)** and **`SCache`** unify all four regimes *training-free* under one directional residual law:
> 1. **Mixture of Scars (`MoS` & `Sub-2-Bit MoS` — Spatial Weights, `0.5B`–`72.71B`):** Pairs canonical `Block-64 INT4` (`38.12 GiB` at `72.71B`, `100.0%` Top-5, `50.38×` fused Apple Metal & CUDA Triton `W4A16+ASVD`), **Zero-Spike `MoS` Student Execution** (`3.49 GiB` Metal VRAM & `0.55 GiB` CPU RSS on `3B` in `4.1s` load with `0` `2×` CPU→GPU spike; `0.78 GiB` VRAM on `0.5B`), and `2.25 bpw` symmetric ternary-quaternary bases + two-pass alternating `ASVD` shields (`cos = 0.99911` under `195,720×` SwiGLU spikes; `12,290×` PPL reduction vs. naive `2-bit`), shrinking `72.71B` into **`20.65 GiB` shards and `21.89 GiB` GPU Resident VRAM (`< 24 GB` RTX 3090/4090 & `< 32 GB` laptops)**.
> 2. **`SCache` $\mathcal{O}(1)$ & `Ghost-Jump` Speculative Reflex (Temporal Memory & Decode):** Drop-in HuggingFace `DynamicCache` combining attention sinks, rolling FIFO, synchronized deep-layer `[L/2 : L)` Widrow-Hoff prediction-surprise routing over `32`-token micro-pages (`+0.00 Bytes` drift from `16.6k` to `32.9k` tokens; `50.9×` smaller & `5.84×` faster), and atomic $\mathcal{O}(1)$ speculative rollback (`KL = 0.0000`, `1.89×–1.92×` speedup).
> 3. **Closed-Form `140 KB` `HeadExpandedScarStore` (`mos-store` CLI & Web UI — Live `HTTP 206` Online Library & Open-Weights Factory):** Transplants `72.71B` and open-weight teacher derivations with **`0` SGD / `0` backprop** into OS-page-aligned **`140.00 KB` (`35 × 4 KB` `mmap`)** cartridges on `3B` (`61.25 KB` on `0.5B`, `98.61%` unseen Top-1 parity, live 9-slot GitHub CDN `HTTP/1.1 206` Range streaming, zero-dependency `mos-store ui` Visual Store, and `18 µs` RAM / `7.64 ms` `0x00` SSD GDPR unlearning).
> 4. **Sinkhorn Monomial–Procrustes Re-Basin (`mos-rebasin`):** Exploits exact SwiGLU monomial gauge symmetry via tiled single-pass log-Sinkhorn (`< 350 MiB` VRAM at `72B+`, `0` extra runtime FLOPs), slashing shared-base KL by **`56.36×` (`0.1578 → 0.0028`)** on permuted specialists and **`3.37×` (`0.1190 → 0.0353 < 0.08`)** on `3B` cross-layer pairs.
> 5. **Horizon 3.3 — Native Zero-Python `WebGPU / WGSL` Browser Engine (`mos-webgpu`):** Compiles the complete `MoS Block-64 INT4 + ASVD` + `SCache O(1)` (`Widrow-Hoff Delta` + fixed surprise slots, `Δ = +0 B`) + `HeadExpandedScarStore` (`HTTP/1.1 206` `61.25 KB` / `140.00 KB` live cartridge streaming & `0x00` LGPD wipe) stack into `14` pure `WGSL 1.0` compute shaders ([`scar_cache/webgpu/mos_scache_kernels.wgsl`](scar_cache/webgpu/mos_scache_kernels.wgsl)), reaching **`67.7–71.85 tok/s`** on Apple M2 Metal (`526.09 MiB` shards `< 80 MiB`, `2,694.75 KB` locked `SCache O(1)`).

![Multi-Scale v3.0.0 Empirical Audit — 32.9k Passkey O(1) KV, <24 GB 72.71B Sub-2-Bit, 140 KB ScarStore & Sinkhorn Re-Basin](zenodo_deposit_v3_0/mos_scache_v3_0_hero.png)

---

## 🔥 Red-Team Benchmarks (Top-of-README Quick-Look)

### 1. `16.6k` & `32.9k` Passkey / Needle-in-a-Haystack: `SCache O(1)` vs. Normal KV Cache
Measured live on Apple Silicon M2 (`16 GB`, `MPS`, `FP16`) across **100% non-repeating Python standard library documentation** (`5%, 25%, 50%, 75%, 95%` needle depths, **zero fine-tuning**, verified via recursive `untyped_storage().data_ptr()` byte audit with `0.00 B` discrepancy in [`results/passkey_16k_32k_llm_audit.json`](results/passkey_16k_32k_llm_audit.json)):

| Pretrained Model | Context (`L`) | Cache Engine | Physical KV Memory (`Δrec = 0 B`) | Throughput (`tok/s` · Wall Time) | Passkey Recall (`5%–95%` Depths) | KV Reduction & Speed vs. Normal KV |
|---|---|---|---|---|---|---|
| **`Qwen2.5-3B-Instruct`** (`36L`) | `16,568` tok | Normal KV $\mathcal{O}(T)$ (`DynamicCache`) | `582.75 MiB` *(linear growth)* | `245.3 tok/s` (`67.5s`) | `100%` (`5/5`) | `1.0×` RAM · `1.00×` speed |
| **`Qwen2.5-3B-Instruct`** (`36L`) | `16,568` tok | `StreamingLLM` $\mathcal{O}(1)$ ($W=256$) | `9.00 MiB` *(constant)* | `394.0 tok/s` (`42.1s`) | **`0%` (Evicted)** | `64.8×` less RAM · Amnesia outside $W$ |
| **`Qwen2.5-3B-Instruct`** (`36L`) | **`16,568` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`22.71 MiB` (`+0.00 B` drift)** | **`299.0 tok/s` (`55.4s`)** | **`100%` (`5/5` depths)** | **`25.7×` less KV RAM · `1.22×` faster** |
| **`Qwen2.5-3B-Instruct`** (`36L`) | `32,868` tok | Normal KV $\mathcal{O}(T)$ (`DynamicCache`) | `1,155.80 MiB` (`1.13 GiB`) | `51.5 tok/s` (`638.3s` / `10.6m`) | `100%` (`3/3`) | `1.0×` RAM · Quadratic $\mathcal{O}(T^2)$ stall |
| **`Qwen2.5-3B-Instruct`** (`36L`) | `32,868` tok | `StreamingLLM` $\mathcal{O}(1)$ ($W=256$) | `9.00 MiB` *(constant)* | `391.4 tok/s` (`84.0s`) | **`0%` (Evicted)** | `128.4×` less RAM · Amnesia outside $W$ |
| **`Qwen2.5-3B-Instruct`** (`36L`) | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`22.71 MiB` (`Δ = +0 B` vs `16.6k`!)** | **`300.8 tok/s` (`109.3s`)** | **`100%` (`3/3`: `5%, 25%, 75%`)** | **`50.9×` less KV RAM · `5.84×` faster!** |
| **`Qwen2.5-0.5B-Instruct`** (`24L`) | **`16,568` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256 / 128$)** | **`6.86 MiB` (`5.32 MiB` @ $S=128$)** | **`1,159.4 tok/s` (`14.3s` vs `15.0s`)** | **`80%` (`4/5`) / `100%` (`5/5` @ $S=128$)** | **`28.3×–36.5×` less KV RAM · `1.1×` faster** |
| **`Qwen2.5-0.5B-Instruct`** (`24L`) | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`6.86 MiB` (`Δ = +0 B` vs `16.6k`!)** | **`1,176.9 tok/s` (`27.9s` vs `44.5s`)** | **`100%` (`5/5` depths)** | **`56.2×` less KV RAM · `1.60×` faster!** |

```bash
# Reproduce the 16.6k & 32.9k Passkey / NIAH audit on your own GPU or Mac (writes JSON + verifies ΔBytes = 0 B):
python scripts/benchmark_16k_passkey_llm.py --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-3B-Instruct --lengths 16500 32800
```

### 2. `72.71B` (`Qwen2.5-72B-Instruct` Canonical `MoS v2`, `80L`): `tokens/sec` Next to Peak Memory
Measured with fused Apple Metal `_weight_int4pack_mm` + Rank-48 `ASVD` scars (**`cos = 1.000000` parity**, **`50.38×` faster 4-layer compute**: `1,495.3 ms → 29.7 ms` in [`results/scale_72b_audit/apple_m2_72b_speed_memory_audit.json`](results/scale_72b_audit/apple_m2_72b_speed_memory_audit.json)) and fused NVIDIA `CUDA` Triton `Block-64 W4A16 + ASVD` kernels ([`scar_cache/triton_mos_gemm.py`](scar_cache/triton_mos_gemm.py)):

| Hardware & Mode (`72.71B`, `80` Layers, `38.12 GiB` Shards) | Prefill `tok/s` | Decode `tok/s` | Peak Active GPU / MPS Memory | Constant `SCache` $\mathcal{O}(1)$ (`W=256, S=24`) | Hardware Bottleneck & Operational Notes |
|---|---|---|---|---|---|
| **Apple M2 (`16 GB` Unified RAM, `100 GB/s`) — `--streaming`** *(Measured Live)* | **`4.23 tok/s`** | `0.023 tok/s` (`43.3 s/tok`) | **`1.92 GiB` MPS** (`120.6 MiB` post-unload · `0.52 GiB` RSS) | **`120.4–128.5 MiB`** (`+0 B` @ `16k+`) | Runs full `72.7B` on a base `16 GB` laptop (`0 OOM`); decode is NVMe SSD I/O-bound (`38.1 GiB`/tok). |
| **Apple M2 (`100 GB/s` bus) — Fused Metal `INT4` Compute** *(Measured Unpaged)* | **`2.53 tok/s`** *(128-blk)* | **`1.28 tok/s`** | `1.90 GiB`/group (`38.56 GiB` 80L eq.) | **`128.51 MiB`** (`+0 B`) | Direct unpaged Metal `_weight_int4pack_mm` + `ASVD` scar rate (`29.7 ms` / 4-layer group). |
| **Apple M2 Pro (`32 GB` max SKU, `200 GB/s`) — Hybrid Page-Cache / Streaming** | `~5.0–6.5 tok/s` | **`~0.40–0.55 tok/s`** *(2.57 tok/s bus ceiling\*)* | **`1.92 GiB` Active MPS** (`~24 GiB` OS RAM cache) | **`128.51 MiB`** (`+0 B`) | \*M2 Pro hardware maxes out at `32 GB` Unified RAM (`< 38.12 GiB` shards), so `~10–12 GiB` spills to `5.5 GB/s` Pro NVMe; pure `200 GB/s` Metal bus ceiling is **`2.57 tok/s`**. |
| **Apple M4 Pro (`48 GB`, `273 GB/s`) / M2 Max (`64 GB+`, `400 GB/s`) — `--resident`** | `~6.8–9.6 tok/s` | **`~3.46 – 4.87 tok/s`** | **`38.56 GiB` Unified MPS** (`0 MB/tok` SSD) | **`128.51 MiB`** (`+0 B` vs `10.7 GiB` KV @ `32k`) | `100%` of `20 × 4L` shards stay wired in Metal Unified Memory; `SCache` saves **`>10.5 GiB`** RAM at `32k` so `72B + 32k` fits in `48 GB`! |
| **NVIDIA `CUDA` (`sm_75+` T4×2 / L4 / 3090 / 4090 / A100)** *(Fused Triton `W4A16+ASVD`)* | `18+ tok/s` | **`4.03 – 7.27+ tok/s`** | **`14.1 – 38.6 GiB` VRAM** (`1.79 GiB`/shard stream) | **`126.0 – 166.0 MiB`** (`+0 B`) | Zero-swizzle register nibble unpack (`M <= 8` coalesced GEMV + `M > 8` Tensor-Core GEMM) + `sm_80+` `int4pack`. |

```bash
# 1. Run the Apple Silicon 72B Fused-Metal speed + peak memory audit (auto-runs 80L --resident on >=48 GB Macs):
python scripts/benchmark_72b_apple_silicon.py

# 2. Download public SHA-256 verified Qwen2.5-72B-Instruct MoS v2 shards & launch interactive terminal chat:
mos-run --download --chat --model-dir ./models/qwen25_72b_mos_v2
```

### 3. Horizon 1 (`v1.4.0`): *Ghost-Jump* Speculative Reflex & `Sub-2-Bit MoS` (`< 24 GB` `72.71B`)
Audited live on Apple Silicon M2 (`MPS`, `HF_HUB_OFFLINE=1`) across `Qwen2.5-0.5B-Instruct`, `Qwen2.5-3B-Instruct`, and real `Qwen2.5-72B-Instruct` shards ([`results/horizon1_v1_4_audit.json`](results/horizon1_v1_4_audit.json), [`scar_cache/speculative_ghost_jump.py`](scar_cache/speculative_ghost_jump.py), [`scar_cache/sub2bit_mos.py`](scar_cache/sub2bit_mos.py)):

* **Track 1.1 — `System 1 + System 2` *Ghost-Jump* Speculative Reflex + Atomic `SCache O(1)` Rollback:**
  - Pairs `Qwen2.5-0.5B-Instruct` (`System 1` Scout, `SCache O(1)`) with `Qwen2.5-3B-Instruct` (`System 2` Verifier, `SCache O(1)`), dynamically halting drafts at epistemic bifurcations via `GhostJumpReflexDecision` (`confidence`, `margin = P_1 - P_2`).
  - **`1.89× – 1.92×` measured wall-clock decode speedup** on structured Python & mathematical derivations (`12.50 → 23.65 tok/s` and `12.14 → 23.30 tok/s`, `78.8%–79.2%` acceptance, **`4.80–5.05` tokens/sweep**) with **`100.0%` exact bit-for-bit greedy parity (`96/96` tokens, `KL = 0.0000`)** and **`0 Bytes` recursive storage discrepancy** (`untyped_storage().data_ptr()`).
  - **Hardware Sweep Scaling on `72.71B`:** In `16 GB` NVMe streaming (`43.3 s`/sweep), `4.80–5.05` tokens/sweep yields **`~4.7×–4.9×` faster `72.71B` streaming decode**; on NVIDIA CUDA Triton (`M <= 8` coalesced register GEMV), single-sweep verification yields **`~3.8×–4.6×` resident `72.71B` speedup** *(note: PyTorch's built-in Apple Metal `_weight_int4pack_mm` currently loops row-wise for `M > 1` — `32.3 ms` @ `M=1` vs. `140.8 ms` @ `M=7` — making a custom batched Metal `M <= 8` shader a prime target alongside our CUDA Triton kernel)*.
* **Track 1.2 — `MoS Sub-2-Bit` (`2.25 bpw` Base + Two-Pass Alternating `ASVD` Outlier Shield):**
  - Uses an IEEE-exact 4-level symmetric codebook `{-1.0, -0.3125, +0.3125, +1.0}` (`beta = 5/16`, zero LUT in GPU registers, fused CUDA Triton `W2A16+ASVD` kernels in [`scar_cache/triton_mos_gemm.py`](scar_cache/triton_mos_gemm.py)) paired with a **Zero-Data Analytical SwiGLU Salience Oracle** (`||w_gate_j * ln2||_2 * ||w_up_j * ln2||_2 * sqrt(1 + 2 cos^2)`) and automatic GQA `k_proj / v_proj` INT4 shielding.
  - **`Qwen2.5-72B` Layer 79 (`8192 × 29568` under `195,720×` SwiGLU spikes):** Naive `2-bit` degrades to `cos = 0.8831` (`rel_err = 0.5430`), whereas **`MoS Sub-2-Bit + ASVD (r=96)` achieves `cos = 0.99911` (`rel_err = 0.0421` — `12.9×` lower error at `2.489 bpw`)** and **`cos = 0.9633`** across 4 consecutive deep layers (`L76..L79`).
  - **Breaks the `24 GB` GPU & `32 GB` Mac Barrier for `72.71B`:** Transcoding `mos_group_19.safetensors` shrinks each 4-layer shard from `1.79 GiB → 1.03 GiB` (**`20.65 GiB`** across all `20` shards; **`22.97 GiB`** full disk checkpoint `< 32 GB` M2 Pro; **`21.89 GiB` GPU Resident VRAM** with `lm_head` + `SCache O(1)` — **fitting `100%` resident on a single `24 GB` RTX 3090 / 4090 / L4!**).
  - **End-to-End Pretrained `Qwen2.5-3B-Instruct` (`108` SwiGLU MLPs):** Naive uniform `2-bit` collapses completely (`PPL = 1,351,758.2`, `Top-1 = 0.5%`), while **`MoS Sub-2-Bit + ASVD` reduces perplexity by `12,290×` (`PPL = 109.99`, `KL = 1.06`, `79.4%–91.1%` Top-5 recall, `4.66×` MLP compression)**.

```bash
# Reproduce the complete Horizon 1 (v1.4.0) Ghost-Jump + Sub-2-Bit 72B audit:
python scripts/benchmark_horizon1_v1_4.py

# Transcode a 38.12 GiB Qwen2.5-72B MoS v2 checkpoint into a 22.97 GiB (21.89 GiB VRAM) Sub-2-Bit checkpoint:
mos-transcode-sub2bit --in-dir ./models/qwen25_72b_mos_v2 --out-dir ./models/qwen25_72b_mos_sub2bit

# Run speculative System 1 + System 2 Ghost-Jump decoding from the CLI:
mos-run --model-dir ./models/qwen25_72b_mos_sub2bit --draft-model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain Widrow-Hoff orthogonal erasure:"
```

### 4. Horizon 2 COMPLETE (`v2.4.0`): Closed-Form `72.71B → 0.5B & 3B` Scar Grafting, `mos-store` (`HTTP 206` + `0x00` SSD Wipe) & *Sinkhorn Monomial–Procrustes Re-Basin* (`KL = 0.0353 < 0.08`)
Audited live on Apple Silicon M2 (`16 GB`, `MPS`) transplanting **100% pure standalone `Qwen2.5-72B-Instruct MoS` (`72.71B`, `80` layers, `d = 8192`)** scientific knowledge directly into frozen **`Qwen2.5-0.5B-Instruct` (`24L`, `d = 896`)** and **`Qwen2.5-3B-Instruct` (`36L`, `d = 2048`)** cores via closed-form Woodbury Vocabulary Rosetta Inversion + AlphaEdit Null-Space Dual Ridge SVD ([`docs/SCAR_STORE_V2_ARCHITECTURE.md`](docs/SCAR_STORE_V2_ARCHITECTURE.md), [`scar_cache/scar_grafting.py`](scar_cache/scar_grafting.py), [`scar_cache/scar_store_v2.py`](scar_cache/scar_store_v2.py), [`scar_cache/dual_depth_bridge.py`](scar_cache/dual_depth_bridge.py), [`scar_cache/head_expanded_attention.py`](scar_cache/head_expanded_attention.py), [`scar_cache/cli_scar_store.py`](scar_cache/cli_scar_store.py), [`scar_cache/sinkhorn_rebasin.py`](scar_cache/sinkhorn_rebasin.py)):
* **Stage 1 (`ScarStoreV2`):** Exact **`62,720 B` (`61.25 KB` `Block-64 INT4`, `2,318,508.5×` smaller than `72B` BF16)** `mmap`-paged capsules over an `802,816`-cell (`896 × 896`) Dual-Subspace Kronecker Grid (`98.0 KB` presence bitmap, `MissingCapsuleAlert` anti-hallucination guard, consumer pack slicing).
* **Stage 2 (`Span-Max` + Voronoi `SCache O(1)`):** Sliding-window routing (`0.8502–0.8849` cosine vs. `0.6151` diluted global mean) + zero-bleed multi-capsule generation across composite 3-domain queries (`exact_torch_equal = True` on all controls).
* **Stage 3 (`DualDepthScarBank` + Interdisciplinary Sleep Bridges):** Golden-Slice `Layer 12 + Layer 24` causal injection (`91.67%`, `95.83%`, **`100.0%`** unseen `72.71B` Top-1 match at `61.25 KB` INT4) + `BridgeEscalationDecision` epistemic telemetry & `consolidate_interdisciplinary_bridge` (`confidence: 0.8398 → 0.9788`, `margin: 0.7053 → 0.9702`, zero single-domain bleed).
* **Stage 4 (`HeadExpandedAttentionBank` — Zero Mid-Cut Dynamic Scar-Head Expansion):** Keeps **all `24/24` Transformer blocks `100%` bit-exact (`torch.equal = True`)**, synthesizes `2.06×` faster (`942.6 ms`/cap) inside the **exact same `62,720 B` (`61.25 KB` INT4) budget** (`rank=60, P=8`, `112` per-head keys), eliminates `100%` of mid-layer off-diagonal bleed (`0.0000`), and raises **pre-sleep** interdisciplinary margin from **`0.7053 → 0.9607` (`+25.54 p.p.`, `87.02%` 14-head spread, dynamic `0.9979 ↔ 0.0029` token switching)**!
* **Stage 5 (`HeadExpandedScarStore` on `Qwen2.5-3B-Instruct` + `6/6` Adversarial Red-Team Suite):**
  - **Zero-Waste OS Page Alignment:** On `Qwen2.5-3B-Instruct` (`36L`, `16 × 128` Q-heads, `d = 2048`), every `HeadExpandedScarCapsule` (`rank=60, P=8`, `128` per-head keys) packs into **exact `143,360 B` = `140.00 KB` = `35 × 4,096 B` OS `mmap` pages (`0 B` internal padding; `1,015,469×` smaller than `72B` BF16)** over a **`4,194,304`-cell (`2048 × 2048`) Kronecker Grid** (`512 KB` CPU bitmap, `36/36` blocks `torch.equal = True` in flight).
  - **`6.11×` Lower Unseen `72.71B` KL Divergence (`98.61%` Mean Top-1):** Raises unseen `72.71B` Top-1 agreement from `94.44%` (`0.5B`) to **`98.61%` (`71/72` exact tokens: `95.83%` Kabatiansky-Levenshtein [`KL 0.0727`], `100.0%` Landauer [`KL 0.0066` — `97.7×` below base `3B`], `100.0%` Immerman-Szelepcsényi [`KL 0.0079`])** and corrects raw `3B`'s interdisciplinary hallucination (`\log m → k_B T \ln 2`, `93.03%` 16-head spread).
  - **`6/6` Hostile Red-Team Attacks Passed Across Both `0.5B` & `3B` ([`results/adversarial_0p5b_vs_3b_audit.json`](results/adversarial_0p5b_vs_3b_audit.json)):** `0.0 B` `SCache O(1)` drift (`0.0 KB` `untyped_storage().data_ptr()` discrepancy), `0/6` false positives on keyword-stuffed SQL/CSS/Git/baking traps (`0 B` `mmap` read), `100%` resistance to `1,560`-token `"999 Joules"` poisoned multi-turn haystacks (`3B` outputs exact `kT \ln 2`), crystal-clear single-rollout 3-way simultaneous synthesis (`96.99%` 16-head spread on `3B`), **`100.0%` Top-1 (`KL = 0.0029` on `3B`)** under simultaneous `MoS` (`3.41×` MLP) + `SCache O(1)` + `ScarStore`, and **`10.6–18.0 µs` (`0.011–0.018 ms`)** surgical GDPR `unlearn()`!

* **Zero Backpropagation / Zero SGD (`~0.90 s` per capsule):** Calibrates the Student null-space in `276.9 ms` and mints each `225.75 KB` FP16 (`64.75 KB` INT4-eq) `ScarCapsule` `(B_scar, A_scar, z_anchor)` in `< 1.0 s` (`629,052.7×` smaller active transfer than `135.43 GiB` `72B` BF16; `5.25 KB` L1 router index).
* **Breaks the `d = 896` Linear Weight-Editing Ceiling via Spherical-Cap Cones:** Unlike static linear merges (`AlphaEdit`/`MEMIT`) bounded by `d = 896` orthogonal directions, one-sided `ReLU` spherical cones (`g_k(q) = clamp((cos(q, z_k) - tau) / ramp, 0, 1)`, `strictly_disjoint_at_anchors = True`) isolate domains exponentially in $\mathbb{S}^{895}$ (`0.0000000000` out-of-cone perturbation across 6 unrelated controls, `torch.equal = True`; `< 2 ms` surgical `bank.unlearn()`).
* **Unseen Paraphrase Generalization & Self-Accelerating `Ghost-Jump` vs. `72.71B`:**
  - *Immerman-Szelepcsényi Theorem:* `0.5B` hallucination (*"deterministic Turing machines..."*) → **exact `72.71B` theorem** (*"nondeterministic space complexity classes are closed under complementation"*), **`100.0%` Top-1 agreement**, `KL: 0.48 → 0.03` (**`18.5×` lower**), `Ghost-Jump` draft acceptance **`26.0% → 100.0%`** (`11 → 5` sweeps)!
  - *Landauer's Principle & Kabatiansky-Levenshtein Bound:* `Ghost-Jump` acceptance jumps **`44.7% → 100.0%`** and **`13.0% → 60.0%`** (`15 → 7` sweeps).

```bash
# Reproduce the Horizon 2 (v2.0.0a1) 72.71B -> 0.5B Closed-Form Scar Grafting audit:
python scripts/benchmark_scar_grafting_v2.py

# Graft or query portable .scargraft.safetensors knowledge cartridges from the CLI:
mos-graft --query "Explain concisely what the Immerman-Szelepcsenyi theorem proves about nondeterministic space complexity classes." --cartridge results/qwen72b_to_0p5b_v2.scargraft.safetensors

# Launch the Zero-Dependency Visual Cartridge Store + Live SCache O(1) Chat + Open-Weights Factory UI:
mos-store ui --model Qwen/Qwen2.5-3B-Instruct --remote auto

# Browse the live Official Online Cartridge Library (9 slots for 3B [140 KB] & 0.5B [61.25 KB]):
mos-store catalog

# Stream ONLY the exact 140 KB cartridge needed for your prompt live over HTTP/1.1 206 Range:
mos-store query --remote auto --model Qwen/Qwen2.5-3B-Instruct \
  --prompt "Explain concisely what the Fischer-Lynch-Paterson impossibility theorem proves about asynchronous consensus."

# Mint new cartridges in Zero SGD from ANY Open-Weight Teacher (same-vocab or cross-tokenizer) or JSON spec:
mos-store build --student Qwen/Qwen2.5-3B-Instruct --teacher Qwen/Qwen2.5-7B-Instruct \
  --capsule-id ow_custom --title "Custom Open-Weights Specialist" --prompt "Your domain question..." --out ./my_3b_library.scarstore --append

# Or serve your own local .scarstore, consolidate sleep bridges, or cryptographically wipe (0x00) a capsule:
mos-store serve --store results/qwen72b_to_3b_stage5.scarstore --port 8765
mos-store unlearn --store ./my_3b_library.scarstore --id titan72b_fischer_lynch_paterson --persist

# Run Track 2.2 Sinkhorn Monomial-Procrustes Re-Basin audit (0.5B, 3B & 72.71B shards):
python scripts/audit_sinkhorn_rebasin_v2_4.py
mos-rebasin audit-72b-shard --shard ./models/qwen25_72b_mos_v2/mos_group_19.safetensors --layer-a 78 --layer-b 79 --json
```

* **Stage 6 (`mos-store` Production CLI/UI + Live Public `HTTP/1.1 206` Online Library + Zero-Spike `MoS` Student + Open-Weights Factory, `v3.0.0`):**
  - **Unified CLI & Zero-Dependency Visual Store Web UI ([`scar_cache/cli_scar_store.py`](scar_cache/cli_scar_store.py), [`scar_cache/ui_scar_store.py`](scar_cache/ui_scar_store.py)):** Ships `mos-store` (`ui`, `catalog`, `pull`, `build`, `inspect`, `slice`, `serve`, `query`/`--chat`, `sleep-bridge`, `unlearn --persist`, audited in [`results/mos_store_v2_3_cli_http206_audit.json`](results/mos_store_v2_3_cli_http206_audit.json)), launching an App-Store-style local web interface (`http://127.0.0.1:7860`) with live `HTTP 206` byte-range telemetry, `VRAM MoS` & `SCache O(1)` gauges, PT-BR/EN prompt routing, `< 1 s` cartridge minting, and 1-click `0x00` SSD GDPR wipe (`7.64 ms` + `fsync`).
  - **Live Official `9`-Slot Online Cartridge Library ([`docs/ONLINE_CARTRIDGE_LIBRARY.md`](docs/ONLINE_CARTRIDGE_LIBRARY.md), [`releases/tag/v3.0.0`](https://github.com/byte-od/mixture-of-scars/releases/tag/v3.0.0)):** Public 24/7 GitHub Release CDN serving `qwen3b_v3_0.scarstore` (`140.00 KB = 35 × 4 KB`/slot) and `qwen0p5b_v3_0.scarstore` (`61.25 KB`/slot) across `9` complete `72.71B` scientific domains (`100%` single-slot `HTTP 206` precision via Zero-Payload Cloud Shells), plus [`scar_cache/open_weights_factory.py`](scar_cache/open_weights_factory.py) (`same_vocab_exact`, `cross_tokenizer_bridge`, and batch JSON specs).
  - **Zero-Spike `MoS Block-64 INT4 + ASVD (r=128)` Student Engine ([`scar_cache/mos_student.py`](scar_cache/mos_student.py)):** Replaces HuggingFace's `from_pretrained().to(dev)` (`~12.4 GiB` `2×` CPU→GPU duplication spike on `3B`) with direct streaming safetensors transcoding (zero-clipping asymmetric `-8..+7` `Block-64 INT4` + noise-bounded `ASVD r=128` + boundary massive-activation shield + dynamic row-scaled Metal `_weight_int4pack_mm`), running `Qwen2.5-3B-Instruct` in **`3.49 GiB` active Metal VRAM + `0.55 GiB` CPU RSS** (`4.14–5.23 s` cached load, `17.0–17.1 tok/s` decode with `logits_to_keep=1`) and `Qwen2.5-0.5B-Instruct` in **`0.78 GiB` VRAM** (`0.90–1.72 s` load).
* **Track 2.2 (`Sinkhorn Monomial–Procrustes Re-Basin`, `v2.4.0` — Resolves Preprint Section 5 Limitation #1):**
  - Exploits the exact **SwiGLU Monomial Gauge Group** $\mathcal{G}_{\text{SwiGLU}} = \mathrm{GL}_1(\mathbb{R})^{d_{\text{ff}}} \rtimes S_{d_{\text{ff}}}$ (`FP32 max_diff = 5.25e-06`, **`0` runtime FLOPs** by folding $\alpha_i$ + `down_proj` intermediate RMS `100%` into `FP16` `Block-64` scales via `transposed_grid=True`), **Fused Single-Pass Online Dual-Potential Log-Sinkhorn** (`< 350 MiB` tile VRAM at `72B+`, `57×` less GPU memory), `2×2` Gram principal `Block-64 INT4` shared grids (`same untyped_storage().data_ptr()`, **`0 B` byte discrepancy** across CPU/MPS/CUDA), and optional whitened Procrustes rotations ([`results/sinkhorn_rebasin_v2_4_audit.json`](results/sinkhorn_rebasin_v2_4_audit.json)):
    - **Permuted Multi-Specialist *Git Re-Basin* (`0.5B` `L12`):** Alignment cosine `0.0678 → 0.9817`, cutting shared-base KL **`56.36×` (`0.1578 → 0.0028`, `97.41%` Top-1, `100.0%` Top-5)**.
    - **Cross-Layer Shared-Base `Qwen2.5-3B-Instruct` (`5.09×` pair compression, `-80.37%` vs FP16):** Interior `pair_L22_L23` drops from Naive `KL = 0.1190` to **`KL = 0.0395` Monomial (`3.01×` lower, `99.77%` Top-5, `0` extra FLOPs)** and **`KL = 0.0353 < 0.08` Procrustes (`3.37×` lower)**! Across **`12` consecutive layers (`L18..L29`, `1/3` of the network)**, Naive collapses (`KL = 6.4634`) while Sinkhorn Monomial achieves **`KL = 0.4403` (`14.68×` lower KL, `95.02%` Top-5)**!
    - **Colossal `Qwen2.5-72B` (`29,568 × 8,192`, `6.23×` pair compression: `2,772 → 445.21 MiB`) & *Residual Stream Marchenko-Pastur Orthogonality Law*:** Achieves **`51.1×` (`L40-L41`)** and **`109.2×` (`L78-L79`)** inner-product gains, raising `L79` output cosine under `195,720×` SwiGLU spikes from `0.1626 → 0.8835` (`2.73×` lower error), while uncovering that adjacent pretrained residual blocks self-organize near the random Marchenko-Pastur orthogonality floor ($\mathbb{E}[\max_j \cos] = 0.0633\text{–}0.1228$ vs. $\sqrt{2\ln(29568)/8192} = 0.0501$) — establishing mathematically why *Sinkhorn Re-Basin* is optimal for selective interior pairs and multi-specialist banks, whereas full 80-layer standalone compression achieves higher Pareto fidelity via independent `2.25 bpw` `Sub-2-Bit MoS` bases (`cos = 0.99911`, `20.65 GiB`).

### 5. Horizon 3.3 (`mos-webgpu`): Native Zero-Python `WebGPU / WGSL` Browser Engine & `HTTP 206` ScarStore Studio
Audited live on Apple M2 Metal GPU (`wgpu` + pure ES-module browser bundle, zero `@huggingface/transformers.js` or ONNX Runtime wrappers) in [`results/webgpu_wgsl_v3_0_audit.json`](results/webgpu_wgsl_v3_0_audit.json) ([`scar_cache/webgpu/mos_scache_kernels.wgsl`](scar_cache/webgpu/mos_scache_kernels.wgsl), [`scar_cache/webgpu_engine.py`](scar_cache/webgpu_engine.py), [`scar_cache/webgpu/app/mos_webgpu_engine.js`](scar_cache/webgpu/app/mos_webgpu_engine.js), [`scar_cache/webgpu/app/scarstore_http206_browser.js`](scar_cache/webgpu/app/scarstore_http206_browser.js), [`scar_cache/webgpu/app/qwen_bpe_tokenizer.js`](scar_cache/webgpu/app/qwen_bpe_tokenizer.js), [`scar_cache/webgpu/app/index.html`](scar_cache/webgpu/app/index.html)):
* **`14` Pure `WGSL 1.0` Compute Kernels (`cos > 0.999999` vs. PyTorch):** Implements `embed_int4_lookup`, `rmsnorm_f32`, `fp16_gemv_bias`, fused `mos_fused_int4_bscar_gemv` (`Block-64 INT4 + ASVD` via `unpack2x16float` + nibble shifts with zero optional shader extensions), `qwen_rope_inplace`, `scache_o1_update_widrow_hoff`, `scache_o1_gqa_attention`, `swiglu_mul_f32`, `head_expanded_scar_inject` (`+M` dynamic competitive Scar-Heads), `int4_vocab_logits_gemv`, and 2-stage `logits_top2_reduce` (`151,936` logits → **`16 Bytes`/token `mapAsync` readout**).
* **`>10.5×` Faster Decode & Strict `SCache O(1)` (`ΔBytes = 0`):** Exports `Qwen2.5-0.5B-Instruct` in `2.60s` into `8` universal-WebGPU-compliant binary shards (`max = 71.74 MiB < 128 MiB`, `526.09 MiB` total VRAM), sustaining **`67.7–71.85 tok/s`** single-token decode with `SCache O(1)` locked at **`2,759,424 B` (`2,694.75 KB`, `Δ = +0 B`)**.
* **Zero-Python Live `HTTP/1.1 206` Browser Cartridge Streaming & `< 0.2 ms` `0x00` LGPD Wipe:** Boots with `0 B` cartridge payload (`42.0 KB` anchor router), streams exact **`62,720 B` (`61.25 KB`)** / **`143,360 B` (`140.00 KB`)** byte ranges via `fetch(..., { headers: { Range } })` on cone hit (`k_BT \ln 2` `72.71B` derivation at `p_1 = 0.9790`; `95.88%` 14-head spread under simultaneous `M = 2` Landauer + FLP injection), and wipes individual cartridges in browser RAM with `0x00` (`exact_revert_to_base_tokens = True`).

```bash
# 1. Export Qwen2.5-0.5B-Instruct (or 3B) + v3.0 .scarstore into static <80 MiB WebGPU shards:
mos-webgpu export --model Qwen/Qwen2.5-0.5B-Instruct --store results/online_library_v3_0/qwen0p5b_v3_0.scarstore --out ./webgpu_dist/qwen0p5b

# 2. Verify WGSL hardware parity, SCache O(1) ΔBytes = 0, and live ScarStore injection on local GPU:
mos-webgpu verify --dir ./webgpu_dist/qwen0p5b

# 3. Serve the Zero-Python-Inference WebGPU Studio + RFC 7233 HTTP/1.1 206 Range endpoint in Chrome/Edge:
mos-webgpu serve --dir ./webgpu_dist/qwen0p5b --port 8780
```

---

## 📄 Official Preprint (`v3.0.0`), Public `72.71B` Shards & Forensic Audit Guide
- **Permanent Preprint (CERN / Zenodo `v3.0.0` DOI):** [`10.5281/zenodo.24172784`](https://doi.org/10.5281/zenodo.24172784) · *Concept DOI (all versions):* [`10.5281/zenodo.24144512`](https://doi.org/10.5281/zenodo.24144512) ([**Preprint v3.0 PDF**](zenodo_deposit_v3_0/Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf) | [v1.3 PDF](archive/zenodo_deposits/zenodo_deposit_v1_3/Mixture_of_Scars_and_SCache_Preprint_v1.3.pdf) | [v1.2 PDF](archive/zenodo_deposits/zenodo_deposit_v1_2/Mixture_of_Scars_and_SCache_Preprint_v1.2.pdf) | [v1.1 PDF](archive/zenodo_deposits/zenodo_deposit_v1_1/Mixture_of_Scars_and_SCache_Preprint_v1.1.pdf) | [v1.0 PDF](archive/zenodo_deposits/zenodo_deposit_v1/Mixture_of_Scars_and_SCache_Preprint_v1.0.pdf))
- **Academic LaTeX Source & Replication Bundles (`v3.0.0`, arXiv / TMLR / Double-Blind Ready):** [`paper_latex/`](paper_latex/), [`arXiv_Direct_Upload_v3.0.zip`](zenodo_deposit_v3_0/arXiv_Direct_Upload_v3.0.zip), [`MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip`](zenodo_deposit_v3_0/MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip) & [`MoS_SCache_Adversarial_Replication_v3.0.zip`](zenodo_deposit_v3_0/MoS_SCache_Adversarial_Replication_v3.0.zip)
- **Forensic Audit & Zero-Facade Verification Guide (`95/95` Tests):** [`docs/AUDIT_AND_VERIFICATION_GUIDE.md`](docs/AUDIT_AND_VERIFICATION_GUIDE.md)
- **Live Official Online Cartridge Library (`9` Slots, `HTTP/1.1 206` Streaming):** [`docs/ONLINE_CARTRIDGE_LIBRARY.md`](docs/ONLINE_CARTRIDGE_LIBRARY.md) · [`releases/tag/v3.0.0`](https://github.com/byte-od/mixture-of-scars/releases/tag/v3.0.0)
- **Public `Qwen2.5-72B-Instruct MoS v2` Checkpoint (`38.12 GiB`, `22` SHA-256 Shards, Zero Login) & Practical Guide:**
  - **Direct Download (`HTTP 206 Range` + `SHA-256`):** [`github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos`](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
  - **Step-by-Step Local Deployment Guide:** [`docs/GUIA_PRATICO_MOS_SCACHE_72B.md`](docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
- **Official `v1.4 → v3.x` Innovation & Cross-Domain Applications Roadmap:** [`docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md`](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md)
- **Interactive Gradio Red-Team Laboratory (Hugging Face Space):** [`huggingface_space/`](huggingface_space/) · *(Historical `v1.x` deposits, legacy Space drafts & Kaggle bundles preserved cleanly in [`archive/`](archive/README.md))*

---

## 📐 Mathematical Formulation: The Unified Scar Operator

Both `MoS` (in parameter space) and `SCache` (in context time) instantiate the same directional residual operator:

$$F(\mathbf{x}) = F_{\text{base}}(\mathbf{x}) + \sum_{k} \operatorname{ReLU}\!\left(\frac{\mathbf{x}^\top \mathbf{z}_k}{\|\mathbf{x}\|_2 \|\mathbf{z}_k\|_2 + \epsilon}\right) \cdot \Delta_{\text{scar}, k}(\mathbf{x})$$

### 1. One-Sided Spherical-Cap Gate (`0.000000` Out-of-Cone Interference)
Unlike `Softmax` routing ($\sigma_k(\mathbf{x}) > 0$ across all $\mathbb{R}^d$, leaking diffuse cross-talk), the one-sided `ReLU` cosine gate defines an exact $90^\circ$ spherical cap around $\mathbf{z}_k$. Every input in the orthogonal/antipodal half-space ($\mathbf{x}^\top \mathbf{z}_k \le 0$) incurs **identically zero (`0.000000`) activation or write**.

### 2. Spatial Domain — `MoS` (`Block-64 INT4` + In-Register Activation-Aware SVD)
Given pretrained matrix $\mathbf{W} \in \mathbb{R}^{N \times K}$, channel RMS activation scale $s_j = \sqrt{\mathbb{E}[x_j^2]} + \epsilon$, and symmetric `Block-64` 4-bit base $\widehat{\mathbf{W}}_4 = \operatorname{dequant}(\mathbf{W}_q, \boldsymbol{\alpha}_w)$, `MoS` factors the activation-weighted quantization residual $(\mathbf{W} - \widehat{\mathbf{W}}_4)\operatorname{diag}(\mathbf{s}) \approx \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$ ($r \ll \min(N, K)$), setting $\mathbf{B}_{\text{scar}} = \mathbf{U}_r \boldsymbol{\Sigma}_r^{1/2}$ and $\mathbf{A}_{\text{scar}} = \boldsymbol{\Sigma}_r^{1/2} \mathbf{V}_r^\top \operatorname{diag}(\mathbf{s})^{-1}$:

$$\mathbf{y}(\mathbf{x}) = \widehat{\mathbf{W}}_4 \mathbf{x} + \mathbf{B}_{\text{scar}}(\mathbf{A}_{\text{scar}}\mathbf{x})$$

To eliminate the `~960 MB` temporary FP16 dequantization buffer per `72B` SwiGLU matrix while preventing `> 65,504` FP16 overflow under **`195,720×`** SwiGLU intermediate spikes via dynamic per-row scaling $c_m = \max(\|\mathbf{x}_m\|_\infty / 512, 10^{-4})$, `v1.3.0` fuses nibble unpacking and the low-rank scar epilogue directly in GPU registers on both **Apple Silicon Metal** (`_weight_int4pack_mm`) and **NVIDIA CUDA Triton** ([`scar_cache/triton_mos_gemm.py`](scar_cache/triton_mos_gemm.py)).

### 3. Temporal Domain — `SCache` $\mathcal{O}(1)$ (Orthogonal Widrow-Hoff + Deep-Layer Surprise Routing)
Per layer $\ell$ and KV head $h$, `SCache` maintains $N_{\text{sink}}$ attention sinks, a rolling FIFO window of $W_{\text{win}} - N_{\text{sink}}$ tokens, and $S_{\text{slots}}$ fixed anchor slots backed by per-head Widrow-Hoff fast weights $\mathbf{W}_{\text{scar}}^{(h)} \in \mathbb{R}^{d_h \times d_h}$:

$$\mathbf{e}_t = \mathbf{v}_{\text{ev}} - \mathbf{W}_{\text{scar}}^{(t-1)} \hat{\mathbf{k}}_t, \qquad \mathbf{W}_{\text{scar}}^{(t)} = \gamma \mathbf{W}_{\text{scar}}^{(t-1)} + \eta\, g_{m^\star}(\hat{\mathbf{k}}_t)\, \mathbf{e}_t \hat{\mathbf{k}}_t^\top$$

For $\gamma=1, \eta g_{m^\star}=1, \|\hat{\mathbf{k}}_t\|_2=1$, this simplifies to $\mathbf{W}_{\text{scar}}^{(t)} = \mathbf{W}_{\text{scar}}^{(t-1)}(\mathbf{I} - \hat{\mathbf{k}}_t \hat{\mathbf{k}}_t^\top) + \mathbf{v}_{\text{ev}} \hat{\mathbf{k}}_t^\top$: the orthogonal projector $(\mathbf{I} - \hat{\mathbf{k}}_t \hat{\mathbf{k}}_t^\top)$ **surgically erases stale values on $\operatorname{span}(\hat{\mathbf{k}}_t)$ without perturbing $\operatorname{span}(\hat{\mathbf{k}}_t)^\perp$** (zero Hebbian ghosting). Simultaneously, $\|\mathbf{e}_t\|_2$ aggregated synchronously across semantic deep layers $\ell \in [L_{\text{layers}}/2, L_{\text{layers}})$ acts as a **training-free Surprise Oracle**, promoting high-surprise `32`-token contiguous micro-pages into RoPE-aligned anchor slots at `+0.00 Bytes` memory growth.

---

## 🔬 Multi-Scale 8-Vector Adversarial Audit (`0.5B` → `7.62B` → `72.71B`)

### Colossal `72.71B` (`Qwen/Qwen2.5-72B-Instruct`, `80L`, `240` SwiGLU MLPs) & `7.62B` (`28L`) 3D Pareto
Evaluated across a **non-repeating 926-token distributed storage + Python + queueing math + SQL corpus** ([`scripts/adversarial_scale_72b_suite.py`](scripts/adversarial_scale_72b_suite.py), [`scripts/adversarial_scale_7b_suite.py`](scripts/adversarial_scale_7b_suite.py)):

| Scale & Engine | Non-Repeat PPL | Weight / `8×` Multi-Tenant Footprint | KV VRAM (`L = 4,096`) | Disk Read / Tok | Throughput (`tok/s`) | Speedup vs. `AirLLM` FP16 |
|---|---|---|---|---|---|---|
| **`72.71B`** Dense FP16 + Full KV $\mathcal{O}(T)$ | `4.35` | `135.43 GiB` (`866.3 GiB` `8×` OOM) | `1,280.0 MiB` $\mathcal{O}(T)$ | `0 MB` | Multi-A100 Ref | Cluster Only |
| **`72.71B`** `AirLLM` FP16 SSD (`3.5 GiB/s` / Cloud) | `4.35` | `<2 GiB` (I/O-stalled) | `1,280.0 MiB` $\mathcal{O}(T)$ | `135.43 GiB` | `0.026` / `0.0006` | `1.0×` (`38.7 s/tok`) |
| **`72.71B`** `MoS` + `StreamingLLM` ($W=128$) | `6.37` | `38.12 GiB` (**`31.4 GiB 8×`**) | `40.0 MiB` $\mathcal{O}(1)$ | **`0 MB` (VRAM)** | `4.72` / `7.27` | `182×–280×` (Amnesia) |
| **`72.71B`** **`MoS` + `SCache` $\mathcal{O}(1)$ ($W=384, S=16$)** | **`4.75`** | **`38.12 GiB` (`31.4 GiB 8×`, `-96.37%`)** | **`166.0 MiB` $\mathcal{O}(1)$ (`+0 B`)** | **`0 MB` (VRAM)** | **`4.03` / `7.27`** | **`155×–280×` (`6,717×` Cloud)** |
| **`7.62B`** `AirLLM` FP16 SSD (`1.57 GB/s`) | `6.05` | `<2 GiB` (SSD-stalled) | `229.4 MB` $\mathcal{O}(T)$ | `14.2 GiB` | `0.11 tok/s` | `1.0×` (`9.1 s/tok`) |
| **`7.62B`** **`MoS` + `SCache` $\mathcal{O}(1)$ ($W=256, S=16$)** | **`6.67`** | **`2.99 GiB` (`3.72 GiB 8×`, `-95.6%`)** | **`22.0 MB` $\mathcal{O}(1)$ (`+0 B`)** | **`0 MB` (VRAM)** | **`24.7 tok/s`** | **`224.5×` (`62.7×` vs INT4)** |

### Summary of Hostile Red-Team Vectors (`V0`–`V6`)
| Hostile Vector | Conventional Baseline | `MoS` / `SCache` $\mathcal{O}(1)$ Result | Audited Empirical Finding |
|---|---|---|---|
| **V0: Recursive Byte-Pointer Audit** | `72.7B @ 4k`: `1,310,720.0 KB` KV | **`66,281.3 KB` (`19.78×` smaller)** | `0.00 KB` hidden storage discrepancy (`untyped_storage().data_ptr()`) |
| **V1: `195,720×` SwiGLU Outliers (`72.7B`)** | Uniform `INT4` KL: `0.0474–0.3717` | **KL `0.0144–0.1206` (`3.08×–3.29×` lower)** | **`100.0%` Top-5 agreement** across Python, Physics, Portuguese & JSON |
| **V2: Long-Horizon Tail PPL ($W=64$)** | `72.7B` Naive: `127.91` · StreamLLM: `97.46` | **`72.7B` `SCache` Tail PPL: `19.51`** | **`5.00×` lower tail PPL** than `StreamingLLM` (`-86.8%` KV VRAM) |
| **V3: Contradictory State Mutation** | Additive Hebbian: `cos = 0.467` (ghosting) | **Widrow-Hoff: `cos = 1.0000`** | Exact orthogonal erasure; `14.38×` higher evicted passcode prob at `72.7B` |
| **V4: `500×` Noise DoS + Antipodal** | Unbounded linear drift / corruption | **Target `cos = 1.0000` · Leak `0.000000`** | Exact mathematical immunity outside the `ReLU` spherical cap |
| **V5 & V6: Full-Stack `72.7B` / `123B` Stream** | `72.7B`: `135.4 GiB` · `123B`: `246 GB` | **`20 × 1.79 GiB` shards (`1.92 GiB` MPS)** | Simultaneous `-72.7%` MLP & `-86.8%` KV; recalls `7942-OMEGA` on `16 GB` M2 |

---

## ⚖️ Honest Boundary Conditions & Limitations

1. **`32 GB` M2 Pro SKU & `24 GB` Consumer GPUs on `72.71B` (`INT4 v1.2` vs. `Sub-2-Bit v1.4`):** The `80`-layer `INT4+ASVD` (`v1.2`) shards total `38.12 GiB` (`100%` RAM-resident on `>= 48 GB` Macs at `3.46–4.87 tok/s`, paging `~8–10 GiB` over NVMe on `32 GB` M2 Pro). In **`v1.4.0` (`Sub-2-Bit + ASVD`, `mos-transcode-sub2bit`)**, the `20` shards shrink to **`20.65 GiB`** (`22.97 GiB` full disk / **`21.89 GiB` GPU VRAM**), fitting **`100%` resident on both `24 GB` GPUs (`RTX 3090 / 4090 / L4`) and `32 GB` Apple M2 Pro** (`cos = 0.99911` on Layer 79 outlier projections; `cos = 0.9633` across 4 deep layers, trading higher `2-bit` isotropic quantization noise for `1.73×` denser residency).
2. **Cross-Layer Permutation Sensitivity on Pretrained SwiGLU:** Sharing a single dense base across *distinct* pretrained layers without Sinkhorn permutation alignment incurs `KL ≈ 0.78` (as seen in synthetic topology stress scripts `examples/13` & `15`, which achieve `90×` structural compression on unaligned weights). Production off-the-shelf models therefore pair **per-layer `Block-64 INT4` bases with `ASVD` scars** (`38.12 GiB` standalone at `72.7B`), while **single-base sharing** is used for multi-tenant specialist banks anchored to a common backbone (`-96.37%` / `27.59×` across 8 specialists).
3. **Fixed Micro-Page Slot Budget vs. Random Hash Copying:** Synchronized `32`-token micro-page routing (`S = 256`) achieves `100%` passkey recall at `32.9k` tokens; verbatim copying of dozens of independent uncorrelated hashes scales linearly with `S`.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/byte-od/mixture-of-scars.git
cd mixture-of-scars
pip install -e . pytest
```

### 1. Drop-in HuggingFace $\mathcal{O}(1)$ KV-Cache Replacement (`SCacheDynamicCache`)

Works out-of-the-box with HuggingFace `transformers` (`Qwen2.5`, `Llama-3`, `Mistral`, `SmolLM2`) with **zero fine-tuning**:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache import SCacheDynamicCache

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Constant O(1) cache: 4 sinks + 252 FIFO window + 256 surprise-routed micro-page tokens + Widrow-Hoff fast weights
cache = SCacheDynamicCache(window_size=256, num_sink_tokens=4, num_slots=256)

inputs = tokenizer("Your 32k+ token context here...", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)

print(tokenizer.decode(out[0], skip_special_tokens=True))
print(f"Frozen SCache Memory: {cache.total_memory_bytes / (1024**2):.2f} MiB (+0.00 B drift at any length!)")
```

### 2. Fused `Block-64 W4A16 + ASVD` Linear & Multi-Specialist `MoS` Layers

```python
import torch
import torch.nn as nn
from scar_cache import MixtureOfScarsLinear, fused_int4_mos_linear

# Multi-tenant / multi-expert layer sharing a single base with directional ReLU scar experts:
shared_base = nn.Linear(4096, 4096, bias=False)
expert_bank_a = MixtureOfScarsLinear(4096, 4096, num_experts=8, top_k=2, shared_base=shared_base)
expert_bank_b = MixtureOfScarsLinear(4096, 4096, num_experts=8, top_k=2, shared_base=shared_base)

x = torch.randn(4, 4096)
y = expert_bank_a(x)  # Zero-copy shared base + L1/L2-resident ReLU-gated low-rank scars
```

### 3. Download & Run `Qwen2.5-72B-Instruct MoS v2` Locally (`38.12 GiB`, Zero Login)

All `22` **SHA-256 verified** `.safetensors` shards (`< 1.80 GiB` per 4-layer group) are publicly hosted with `HTTP 206 Range` resume support:
- **📦 Direct Model Release:** [`releases/tag/v1.2.0-qwen72b-mos`](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
- **📘 Practical Deployment Guide (Metal / CUDA / CPU):** [`docs/GUIA_PRATICO_MOS_SCACHE_72B.md`](docs/GUIA_PRATICO_MOS_SCACHE_72B.md)

```bash
# Auto-downloads SHA-256 verified 72.71B shards (resumable) & opens interactive O(1)-memory terminal chat:
mos-run --download --chat --model-dir ./models/qwen25_72b_mos_v2
```

---

## 🌍 Beyond LLMs: Open Innovation Frontiers (`v1.4 → v2.x` Roadmap)

The Unified Scar Operator unlocks concrete mechanisms outside text generation — detailed in [`docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md`](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md):
- **Vector Search & Post-RAG (`Scar-Search`):** Nanosecond L1-cache (`34.8 KB`) `ReLU` spherical-cap shard pruning, `20×–50×` smaller surprise-filtered vector indices, and mountable `.scache` $\mathcal{O}(1)$ knowledge cartridges.
- **Data Storage & Lossless Compression (`Scar-Storage`):** Constant-RAM long-range `SCache + ANS` lossless entropy coding, `mos-git` low-rank tensor version control (`-96%+` disk), and **`< 1 ms` surgical *Machine Unlearning*** (GDPR/LGPD erasure via $\mathbf{W}_{\text{scar}}(\mathbf{I} - \hat{\mathbf{k}}\hat{\mathbf{k}}^\top)$).
- **Next-Gen Inference (`v1.4`–`v2.0`):** `System 1` (`0.5B` @ `1,176 tok/s`) + `72B MoS` speculative Ghost-Jump decoding (`3×–5×` speedup), `Sub-2-Bit + ASVD` (`72B` in `~20 GiB` for `24 GB` GPUs), sleep-cycle `SCache → MoS` continual consolidation, and native `WebGPU / WGSL` browser shaders.

---

## 📂 Reproduction Suites, Clean Repository Map & Executable Examples

```bash
# Run all 95 unit, red-team causality, MoS Student, ScarStore HTTP-206/UI/GDPR, Sinkhorn Re-Basin, WebGPU WGSL hardware, and v3.0 architectural integrity tests (26 suites):
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/ -v
```

- **Forensic Audit & Repository Map (`v3.0.0`):** See [`docs/AUDIT_AND_VERIFICATION_GUIDE.md`](docs/AUDIT_AND_VERIFICATION_GUIDE.md) for the complete `26`-file (`95`-test) anti-facade verification matrix, physical byte-pointer audit protocol (`0.00 B` discrepancy), and top-level directory map (`scar_cache/`, `mixture_of_scars/`, `tests/`, `scripts/`, `results/`, `paper_latex/`, `zenodo_deposit_v3_0/`, and [`archive/`](archive/README.md)).
- **Core Verification & Red-Team Suites (`scripts/`):**
  - [`scripts/build_online_cartridge_library_v3_0.py`](scripts/build_online_cartridge_library_v3_0.py) & [`scripts/publish_online_library_github_release.py`](scripts/publish_online_library_github_release.py) — Builds & audits the `9`-slot `72.71B → 3B & 0.5B` Online Cartridge Library (`HTTP/1.1 206` CDN verification).
  - [`scripts/audit_sinkhorn_rebasin_v2_4.py`](scripts/audit_sinkhorn_rebasin_v2_4.py) — Track 2.2 Sinkhorn Monomial–Procrustes Re-Basin (`0.5B`, `3B`, `72.71B` shards & Marchenko-Pastur law).
  - [`scripts/audit_mos_store_cli_v2_3.py`](scripts/audit_mos_store_cli_v2_3.py) & [`scripts/audit_scar_store_v2_stage5_3b_vs_0p5b.py`](scripts/audit_scar_store_v2_stage5_3b_vs_0p5b.py) — `140 KB` `HeadExpandedScarStore` (`HTTP/1.1 206` Range + `0x00` SSD GDPR wipe).
  - [`scripts/benchmark_horizon1_v1_4.py`](scripts/benchmark_horizon1_v1_4.py) — `v1.4.0` Horizon 1 `Ghost-Jump` (`1.89×–1.92×` lossless speedup) + `Sub-2-Bit 72B` (`21.89 GiB` VRAM) live audit.
  - [`scripts/benchmark_16k_passkey_llm.py`](scripts/benchmark_16k_passkey_llm.py) — `16.6k` & `32.9k` Passkey / NIAH live audit (`ΔBytes = 0 B` pointer check).
  - [`scripts/benchmark_72b_apple_silicon.py`](scripts/benchmark_72b_apple_silicon.py) — Apple Silicon Metal `_weight_int4pack_mm` (`50.38×` speedup) + `80L` memory/speed profiler.
  - [`scripts/adversarial_scale_72b_suite.py`](scripts/adversarial_scale_72b_suite.py) & [`scripts/adversarial_scale_7b_suite.py`](scripts/adversarial_scale_7b_suite.py) — Full `72.71B` & `7.62B` 8-vector adversarial audits.
  - [`scripts/adversarial_mos_scache_suite.py`](scripts/adversarial_mos_scache_suite.py) — `0.5B`–`123B` multi-scale red-team suite (`V0`–`V6`).
- **Standalone Tutorials (`examples/01`–`examples/16`):** `16` self-contained scripts in [`examples/`](examples/) covering 3-line drop-ins, RL `4.25 KB` trauma scars (`02`), `< 3 µs` test-time adaptation (`03`), `45k`-token stress tests (`09`), `Qwen2.5-3B` MLP compression (`12`), and unaligned structural topology stress tests (`13`, `15`).
- **Interactive Cloud Notebook:** [**Kaggle `thetod/mixture-of-scars-benchmark`**](https://www.kaggle.com/code/thetod/mixture-of-scars-benchmark) ([`notebooks/mixture_of_scars_official_benchmark.ipynb`](notebooks/mixture_of_scars_official_benchmark.ipynb)).

---

## ❤️ Sponsor Open Research & GPU Compute

**Mixture of Scars (`MoS`)** and **`SCache O(1)`** are developed independently under the permissive **MIT License** — with full source code, fused Apple Metal & CUDA Triton kernels, reproducible red-team benchmarks, and zero-login `72.71B` weights released free to the community.

If this research, its mathematical primitives, or its GPU/Unified-Memory kernels save infrastructure costs or inspire new mechanisms in your team, consider sponsoring our lab via **[GitHub Sponsors (`@byte-od`)](https://github.com/sponsors/byte-od)**:

* **100% of sponsorships fund dedicated multi-GPU & Unified-Memory compute** for our [`v1.4 → v3.x` Open Roadmap](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md):
  1. **`MoS Sub-2-Bit + ASVD` (`72B` in `~20 GiB` & `405B` Shards):** Public SHA-256 checkpoints fitting `100%` resident on single `24 GB` GPUs and `32 GB` Macs.
  2. **`System 1 + 72B` Speculative Reflex Fusion:** `3×–5×` faster lossless decode via Ghost-Jump draft verification.
  3. **`128k–1M` Multi-Needle `SCache` & `.scarstore` Post-RAG Cartridges:** Hierarchical disk-paged `SCache`, automated `mos-digest` corpus ingestion, lossless `SCache + ANS` storage, and native `WebGPU / WGSL` browser kernels.

👉 **[Become a GitHub Sponsor (`github.com/sponsors/byte-od`)](https://github.com/sponsors/byte-od)**

---

## 📄 License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
Developed by **Julio Cesar Reis Filho** (*Byte-OD Research & System One Architecture Lab*, 2026).

```bibtex
@techreport{reisfilho2026mos_scache,
  author      = {Reis Filho, Julio Cesar},
  title       = {Mixture of Scars (MoS) v3.0 and SCache O(1): Sub-2-Bit Colossal Compression (<24 GB 72.7B), Closed-Form 140 KB Scar Cartridges, Sinkhorn Re-Basin, and Constant-Memory 32k Attention},
  institution = {Byte-OD Research \& System One Architecture Lab},
  year        = {2026},
  month       = nov,
  version     = {3.0.0},
  doi         = {10.5281/zenodo.24172784},
  url         = {https://doi.org/10.5281/zenodo.24172784}
}

@misc{reisfilho2026qwen72b_mos_v2,
  author       = {Reis Filho, Julio Cesar},
  title        = {Qwen2.5-72B-Instruct MoS v2 (72.71B, 38.12 GiB Canonical INT4+ASVD Checkpoint) and Official Practical Local Deployment Guide},
  howpublished = {Public SHA-256 Verified Checkpoint Release (\texttt{mos-run --download}) and Practical Guide (\texttt{docs/GUIA\_PRATICO\_MOS\_SCACHE\_72B.md})},
  year         = {2026},
  url          = {https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos},
  note         = {Practical Guide: https://github.com/byte-od/mixture-of-scars/blob/main/docs/GUIA_PRATICO_MOS_SCACHE_72B.md}
}
```
