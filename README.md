# Mixture of Scars (MoS) & SCache $\mathcal{O}(1)$: Breaking the Dual Memory Wall Across `0.5B`–`72.71B` Transformers

[![DOI v1.3.0](https://zenodo.org/badge/DOI/10.5281/zenodo.24172784.svg)](https://doi.org/10.5281/zenodo.24172784)
[![Release: v1.3.0](https://img.shields.io/badge/Release-v1.3.0-brightgreen.svg)](zenodo_deposit_v1_3/Mixture_of_Scars_and_SCache_Preprint_v1.3.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Drop-in](https://img.shields.io/badge/HuggingFace-DynamicCache%20Drop--in-yellow.svg)](https://huggingface.co/)
[![32.9k Passkey Recall](https://img.shields.io/badge/32.9k%20Passkey-100%25%20%40%2022.7%20MiB%20(5.84x%20Faster)-success.svg)](#1-166k--329k-passkey--needle-in-a-haystack-scache-o1-vs-normal-kv-cache)
[![72.7B Multi-Tenant](https://img.shields.io/badge/72.7B%20MoS-38.1%20GiB%20(%E2%88%9296.4%25%208x%20Bank)-9cf.svg)](#2-7271b-qwen25-72b-instruct-canonical-mos-v2-80l-tokenssec-next-to-peak-memory)
[![Fused Metal & Triton](https://img.shields.io/badge/Fused%20Kernels-Metal%2050.4x%20%7C%20CUDA%20Triton-blueviolet.svg)](scar_cache/triton_mos_gemm.py)
[![Tests Passing](https://img.shields.io/badge/Tests-44%2F44%20Passing-brightgreen.svg)](tests/)
[![72B MoS Download](https://img.shields.io/badge/72B%20MoS%20Model-38.12%20GiB%20Download-purple.svg)](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
[![Practical 72B Guide](https://img.shields.io/badge/Practical%2072B%20Guide-Docs-orange.svg)](docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
[![v2.x Roadmap](https://img.shields.io/badge/Roadmap-v1.4%20%E2%86%92%20v2.x-00b4d8.svg)](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md)
[![Sponsor Research](https://img.shields.io/badge/Sponsor-Byte--OD%20Research-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/byte-od)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/thetod/mixture-of-scars-benchmark)

> **Unified Scar Operator (`v1.3.0`):** Autoregressive Transformers hit two compounding hardware walls: static parameter bloat $\mathcal{O}(N)$ (`135.43 GiB` FP16 at `72.71B`) and linear/quadratic KV-cache growth $\mathcal{O}(T) / \mathcal{O}(T^2)$. **Mixture of Scars (`MoS`)** and **`SCache`** attack both walls *training-free* under a single directional residual formulation:
> 1. **Mixture of Scars (`MoS` — Spatial Weights):** Pairs canonical `Block-64 INT4` bases with **Activation-Aware SVD (`ASVD`)** low-rank scar experts, `20 × 4`-layer streaming shards (`1.79 GiB`/shard), and **fused in-register `W4A16+ASVD` GPU kernels** on Apple Silicon Metal (`50.38×` group compute speedup at `cos = 1.000000`) and NVIDIA CUDA Triton (`sm_75+` Turing `T4` through Hopper `H100`) — neutralizing **`195,720×`** SwiGLU outliers at `72.71B` (`135.43 → 38.12 GiB` standalone; **`27.59×` / `-96.37%`** across an 8-specialist bank) with **`100.0%` Top-5 agreement**.
> 2. **`SCache` (`SCacheDynamicCache` — Temporal $\mathcal{O}(1)$ Memory):** Drop-in HuggingFace `DynamicCache` combining attention sinks, a rolling FIFO window, synchronized deep-layer `[L/2 : L)` Widrow-Hoff prediction-surprise routing over `32`-token micro-pages, and orthogonal Delta fast weights (`+0.00 Bytes` drift from `16.6k` to `32.9k` tokens).

![Multi-Scale v1.3.0 Empirical Audit — 32.9k Passkey Recall, 72.71B Outliers & Fused Metal/CUDA Speed](zenodo_deposit_v1_3/mos_scache_v1_3_hero.png)

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

---

## 📄 Official Preprint (`v1.3.0`), Public `72.71B` Shards & Replication Assets
- **Permanent Preprint (CERN / Zenodo `v1.3.0` DOI):** [`10.5281/zenodo.24172784`](https://doi.org/10.5281/zenodo.24172784) · *Concept DOI (all versions):* [`10.5281/zenodo.24144512`](https://doi.org/10.5281/zenodo.24144512) ([**Preprint v1.3 PDF**](zenodo_deposit_v1_3/Mixture_of_Scars_and_SCache_Preprint_v1.3.pdf) | [v1.2 PDF](zenodo_deposit_v1_2/Mixture_of_Scars_and_SCache_Preprint_v1.2.pdf) | [v1.1 PDF](zenodo_deposit_v1_1/Mixture_of_Scars_and_SCache_Preprint_v1.1.pdf) | [v1.0 PDF](zenodo_deposit_v1/Mixture_of_Scars_and_SCache_Preprint_v1.0.pdf))
- **Academic LaTeX Source (arXiv / TMLR / Double-Blind Ready):** [`paper_latex/`](paper_latex/), [`arXiv_Direct_Upload_v1.3.zip`](zenodo_deposit_v1_3/arXiv_Direct_Upload_v1.3.zip) & [`MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip`](zenodo_deposit_v1_3/MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip)
- **Public `Qwen2.5-72B-Instruct MoS v2` Checkpoint (`38.12 GiB`, `22` SHA-256 Shards, Zero Login) & Practical Guide:**
  - **Direct Download (`HTTP 206 Range` + `SHA-256`):** [`github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos`](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
  - **Step-by-Step Local Deployment Guide:** [`docs/GUIA_PRATICO_MOS_SCACHE_72B.md`](docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
- **Official `v1.4 → v2.x` Innovation & Cross-Domain Applications Roadmap:** [`docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md`](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md)
- **Interactive Gradio Red-Team Laboratory (Hugging Face Space):** [`huggingface_space/`](huggingface_space/)

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

1. **`32 GB` M2 Pro SKU vs. `>= 48 GB` Unified Memory on `72.71B`:** The `80`-layer `Qwen2.5-72B MoS v2` shards total `38.12 GiB`. Because Apple caps the M2 Pro hardware SKU at `32 GB` total Unified RAM, `~8–10 GiB` pages over NVMe (`~0.40–0.55 tok/s` hybrid vs. `2.57 tok/s` pure `200 GB/s` bus ceiling). Macs with `>= 48 GB` (`M4 Pro 48 GB`, `M2/M3/M4 Max 64 GB+`) keep all `80` layers + `32k` `SCache` (`128.5 MiB` vs. `10.7 GiB` Normal KV!) **100% RAM-resident** at `3.46–4.87 tok/s` with `0 MB/tok` SSD I/O. (`Sub-2-Bit + ASVD` `~20 GiB` for `24 GB` GPUs & `32 GB` Macs is Track 1.2 of our [`v2.x` Roadmap](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md)).
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

## 📂 Reproduction Suites & Executable Examples

```bash
# Run all 44 unit, red-team causality, and fused Metal/Triton kernel parity tests:
pytest -q
```

- **Core Verification & Red-Team Suites (`scripts/`):**
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

* **100% of sponsorships fund dedicated multi-GPU & Unified-Memory compute** for our [`v1.4 → v2.x` Open Roadmap](docs/ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md):
  1. **`MoS Sub-2-Bit + ASVD` (`72B` in `~20 GiB` & `405B` Shards):** Public SHA-256 checkpoints fitting `100%` resident on single `24 GB` GPUs and `32 GB` Macs.
  2. **`System 1 + 72B` Speculative Reflex Fusion:** `3×–5×` faster lossless decode via Ghost-Jump draft verification.
  3. **`128k–1M` Multi-Needle `SCache` & `.scache` Post-RAG Cartridges:** Continuous long-horizon benchmarks, lossless `SCache + ANS` storage, and native `WebGPU / WGSL` browser kernels.

👉 **[Become a GitHub Sponsor (`github.com/sponsors/byte-od`)](https://github.com/sponsors/byte-od)**

---

## 📄 License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
Developed by **Julio Cesar Reis Filho** (*Byte-OD Research & System One Architecture Lab*, 2026).

```bibtex
@techreport{reisfilho2026mos_scache,
  author      = {Reis Filho, Julio Cesar},
  title       = {Mixture of Scars (MoS) and SCache O(1): Breaking the Dual Memory Wall Across 0.5B -> 7.62B -> 72.71B Transformers via Fused Activation-Aware Scar Experts and 32k Widrow-Hoff Associative Memory},
  institution = {Byte-OD Research \& System One Architecture Lab},
  year        = {2026},
  month       = nov,
  version     = {1.3.0},
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
