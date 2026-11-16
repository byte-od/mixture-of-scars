# Scar Architecture Ecosystem (`v6.0` Release & `ScarField v7.0` Frontier): `ScarNativeLM v6.0`, `PointScarFlow` 3D, Mixture of Scars (`MoS` up to `72.71B`), `SCache` $\mathcal{O}(1)$ & Physical `ScarCodec`

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24172784.svg)](https://doi.org/10.5281/zenodo.24172784)
[![Release: v6.0.0](https://img.shields.io/badge/Release-v6.0.0%20(12L%20Shared--Stage%20MoS%20%7C%20Chunkwise%20WY%2012.63x%20%7C%20PointScarFlow%203D)-brightgreen.svg)](zenodo_deposit_v6_0/ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf)
[![Physics Paper: v1.0.0](https://img.shields.io/badge/Physics%20Preprint-NASA%20Spitzer%20%7C%20Shuttle%20%7C%20PhysioNet%20SHA--256-0284c7.svg)](zenodo_deposit_scarcodec_public_data_v1_0/ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf)
[![ScarField v7.0 Frontier](https://img.shields.io/badge/Frontier%20R%26D-ScarField%20v7.0%20(E8%201bpw%20%7C%20Zero%20ReLU%2FSoftmax%2FRoPE)-f59e0b.svg)](scar_field/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![38 WebGPU Shaders](https://img.shields.io/badge/WebGPU%20WGSL%201.0-26%20LLM%20%2B%2012%203D%20Compute%20Shaders%20(cos%3E0.999999)-blueviolet.svg)](scar_cache/webgpu/mos_scache_kernels.wgsl)
[![72.7B Sub-2-Bit](https://img.shields.io/badge/72.7B%20Sub--2--Bit-20.65%20GiB%20Shards%20(%3C24%20GB%20VRAM)-9cf.svg)](#4-horizon-1-v140-ghost-jump-speculative-reflex--sub-2-bit-mos--24-gb-7271b)
[![Physical ScarCodec](https://img.shields.io/badge/scar__codec-Standalone%20FITS%20%7C%20Telemetry%20%7C%20PDE%20Library-0d9488.svg)](scar_codec/README.md)
[![Sponsor Research](https://img.shields.io/badge/Sponsor-Byte--OD%20Research-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/byte-od)

This repository houses the complete **Scar Operator** research and engineering ecosystem across **four standalone Python packages** — spanning post-training `72.71B` LLM compression, domain-independent spacecraft/astronomy signal processing, *Tabula-Rasa* `12`-layer `ScarNativeLM v6.0` + `PointScarFlow` 3D generation (`26 + 12` `WebGPU WGSL 1.0` shaders), and the experimental **post-Transformer `ScarField v7.0`** engine:

| Package | Scope & Zero-Facade Guarantees | Key Empirical Result |
| :--- | :--- | :--- |
| **[`scar_native/`](scar_native/)** (`v6.0.0`, `mos-scar-native`) | *Tabula-Rasa* `12L` Shared-Stage MoS (`0` `nn.Linear`), `Chunkwise WY`, `Dual-Tier SCache O(1)`, `mos-digest` (`0x00` GDPR + `<abstain>`), & `PointScarFlow` 3D (`26 + 12` WGSL shaders) | **`4.13 MiB`** (`1.90 MiB` stream, `15.0×` < FP16, `0.0` PTQ shock), **`12.63×` A100 WY speedup**, **`98.96%` `PT/EN/ES/FR` `mos-digest`**, **`0.0317` 3D face Chamfer (`cos = 0.99999988` WebGPU)** |
| **[`scar_codec/`](scar_codec/README.md)** (`v6.0.0`, `mos-scar-codec`) | Zero-PyTorch `NumPy` + C99/SIMD (`0.389 µs` $\mathcal{O}(1)$ `mmap`), `.scar` + `Zarr`, $L_\infty$ `abs_tolerance`, 3D Tucker & `1.000 bpw` $E_8$ (`0` `transformers`, `0` trained weights) | **`0.389 µs` C/SIMD $\mathcal{O}(1)$ read (`344×` < PyTorch, `11,061×` < `zlib`)**, **`21.0×` $E_8$ Spitzer** (`0.956` peak err), **`1,636×` < `ZFP` peak err**, **`6/6 = 100%`** Shuttle faults in `3,120 B` |
| **[`scar_cache/`](scar_cache/)** (`v4.0`, `mos-run` / `mos-store` / `mos-webgpu`) | Drop-in HuggingFace `SCache O(1)`, `0.5B → 72.71B` `MoS` (`Block-64 INT4`, `Sub-2-Bit`, `ProtectedInt4Scar`), `140 KB` `HTTP 206` cartridges & Sinkhorn Re-Basin | **`72.71B` in `21.89 GiB` VRAM (`< 24 GB` GPU)**, **`100%` Passkey @ `32.9k` (`50.9×` less KV, `5.84×` faster)**, **`164/164` polyglot `3B` `HTTP 206` routing** |
| **[`scar_field/`](scar_field/)** (`v7.0.0` *Frontier R&D*) | **Post-Transformer Zero-Legacy Engine** (`0` `ReLU/SwiGLU`, `0` `Softmax`, `0` integer `RoPE`, `0` `BPE`, `0` `AdamW`): Gosset $E_8$ `1.000 bpw` + Reversible Cayley $\mathcal{O}(1)$ + `1`-Hop Causal `Kalman-Scar` | **`306.87×` A100 WY speedup**, **`11.65×` Cayley VRAM savings**, **`1.87 MiB` $E_8$ (`9.22×` < FP16)**, **`100%` Unseen PIN @ `T=96,160` & `75%` @ `T=512` OOD (`0%` Dense)**, **`16/16` `0x00` abstention** |

![ScarNativeLM v6.0 & PointScarFlow v1.0 Empirical Audit — Chunkwise WY (12.63x), Shared-Stage MoS (4.13 MiB / 1.90 MiB Stream), Dual-Tier SCache O(1), mos-digest (98.96% PT/EN/ES/FR + 0x00 GDPR <abstain>) & 3D Flow](zenodo_deposit_v6_0/mos_scarnative_v6_0_hero.png)

---

## 🔥 Top-of-README Empirical Audits

### 0A. `ScarNativeLM v6.0` (`12L` Shared-Stage MoS on `NVIDIA A100`) & `PointScarFlow v1.0` (`3D` Rectified Flow)
Pretrained **100% from scratch (`seed = 202606`, zero `torch.nn.Linear` modules, zero $\mathcal{O}(T)$ KV caches)** on `NVIDIA A100-SXM4-40GB` and verified on Apple Silicon Metal WebGPU ([`results/scar_native_v6_0_audit.json`](results/scar_native_v6_0_audit.json), [`results/point_scar_flow_v1_0_audit.json`](results/point_scar_flow_v1_0_audit.json), Preprint [`zenodo_deposit_v6_0/ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf`](zenodo_deposit_v6_0/ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf)):

| `v6.0` Subsystem | Baseline (`12L` FP16 Dense / Sphere) | `ScarNativeLM v6.0` / `PointScarFlow` | Audited Guarantee |
| :--- | :---: | :---: | :--- |
| **Parallel `Chunkwise WY` Widrow-Hoff Scan** | Sequential `1,178.5 ms` | **`93.3 ms` (`C = 64`)** | **`12.63×` A100 speedup** (`cos_fwd = 0.99999988`, `cos_grad = 0.99999982`) |
| **`12L` Shared-Stage `MoS` (`3` Universal Stages)** | `62.02 MiB` (`PPL = 1.583`) | **`4.13 MiB` (`1.90 MiB` *Stream*, `PPL = 1.395`)** | **`15.00×` smaller** (`32.6×` streaming, `0.00000000` KL PTQ shock, `Δlogits = 0.0`) |
| **`Dual-Tier SCache` $\mathcal{O}(1)$ (`26` WGSL Shaders)** | $\mathcal{O}(T)$ KV growth (`0%` windowed) | **`1,329,408 B` (`1,298.3 KB`, `Δ = +0 B`)** | **`100%` Passkey @ `T = 567`** (`23.6×` overflow), WebGPU `min_cos = 0.99999988` |
| **`mos-digest` Document Compiler (`24 × 12 KB`)** | Entangled in dense weights | **`95/96 = 98.96%` (`PT / EN / ES / FR`)** | Minted from `EN` Markdown (`0` SGD); document `0x00` GDPR wipe $\to$ **`24/24` `<abstain>`** |
| **`v6-Frontier` Architectural Scaling (`96L, d=8192`)** | `205.01 GiB` (`128k` ctx) | **`4.12 GiB` LRU / `0.40–0.91 GiB` *Stream*** | Sub-`1 GiB` active stage streaming with `0.0` logit drift |
| **`PointScarFlow` Continuous 3D Scar Generator** | Sphere `Chamfer = 0.4663` | **`Chamfer = 0.0621` / Face `0.0317`** | **`3.87 MiB`**, `12` WGSL 1.0 shaders (`cos = 0.99999988`), `> 0.997` `PT/ES/FR ↔ EN` 3D cosine |

```bash
# Verify ScarNative v6.0 + PointScarFlow WebGPU parity & cross-lingual cartridges:
pytest tests/test_scar_native_v6.py tests/test_point_scar_flow.py -v
```

### 0B. Post-Transformer Frontier R&D: `ScarField v7.0` on `NVIDIA A100-SXM4-40GB` ([`scar_field/`](scar_field/) — Zero `ReLU`, Zero `Softmax`, Zero `RoPE(t)`, `1.000 bpw` $E_8$)
Pretrained **100% *Tabula Rasa* on `NVIDIA A100-SXM4-40GB`** across a 4-model controlled ablation without a single pre-scar Transformer relic (`0` `nn.Linear`, `0` `RMSNorm/LayerNorm`, `0` `ReLU/GELU/SiLU/SwiGLU`, `0` `Softmax/exp/log`, `0` integer `RoPE(t)`, `0` static `BPE`, `0` `AdamW`, audited in [`results/scar_field_v7_0_audit.json`](results/scar_field_v7_0_audit.json)):

| Controlled `A100` Model (`680` Steps, `UTF-8` Bytes) | Packed Weights | Dynamic State @ `T = 2048` | PTQ Shock | Unseen 3-Digit PIN (`T = 96 / 160 / 288 / 512` OOD) | Out-of-Cone `0x00` Abstention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `DenseByteTransformer FP16` (`6L, d=384, O(T) KV`)** | `17.26 MiB` | `18.00 MiB` (`+9.2 KB/step`) | `0.0` *(FP16)* | `0.0% / 0.0% / 0.0% / 0.0%` | `0/16` (`0.0%`) |
| **2. `Dense + Post-Hoc E8 (1bpw) + Window (R=16)`** | `1.73 MiB` | `144.0 KB` (`Δ = +0 B`) | `0.19154` | `0.0% / 0.0% / 0.0% / 0.0%` | `0/16` (`0.0%`) |
| **3. `ScarField v7.0 w/o Kalman-Bus` (`R=16` Ring Only)** | `1.87 MiB` | `129.0 KB` (`Δ = +0 B`) | `1.2e-7` | `0.0% / 0.0% / 0.0% / 0.0%` | **`16/16` (`100%`)** |
| **4. `ScarField v7.0 Full` (`E8` + `Cayley` + `1-Hop Kalman-Scar`)** | **`1.87 MiB` (`9.22×↓`)** | **`129.0 KB` (`132,122 B`, `Δ = +0 B`)** | **`1.2e-7`** | **`100% / 100% / 75.0% / 75.0%` (`32×` ring!)** | **`16/16` (`100%`)** |

- **Silicon Hardware Verification (`NVIDIA A100-SXM4-40GB`):** `Chunkwise WY (C=64)` reaches **`306.87×` wall-clock speedup** (`25.09 ms` vs `7,698.57 ms` sequential, `cos_fwd = 0.99999994`, `cos_grad = 1.00000000`); `ReversibleSymplecticCayleyFlow (K=16)` achieves **`11.65×` CUDA VRAM activation savings** (`689.22 MiB` vs `8,026.43 MiB`), plus **`32.98%` adaptive `S1/S2` Cayley step savings**.
- **Polyglot & Epistemic Physics:** **`142.86×` smaller workspace** at `T=2048` (**`2,285.69×` smaller** at `T=32,768`), **`0.0` passive forgetting** across `1,000` vacuum steps, **`98.25%` exact `0.0` Tsallis-1.5 simplex sparsity**, **`0.9863–0.9875` native `PT/ES/FR ↔ EN` `S_world` alignment (`0` Procrustes)**, and **`86/96 = 89.58%` Zero-SGD `.uscar` polyglot fact routing** (`0.0` residual after `0x00` GDPR wipe).

```bash
# Run the ScarField v7.0 forensic mathematical & head-to-head benchmark suite:
pytest tests/test_scar_field_v7.py -v
python scripts/benchmark_scar_field_v7_vs_dense.py
```

### 0C. Industrial `ScarCodec v6.0` on 5 Public Archives (`ZFP` / `FITS fpack` Audit + `0.389 µs` C/SIMD `mmap` + `Zarr` + $E_8$)
Zero-PyTorch standalone [`scar_codec/`](scar_codec/README.md) (`numpy` only + auto-compiled C99/SIMD `_scar_codec_c`, audited in [`results/scar_codec_public_datasets_v1_0_audit.json`](results/scar_codec_public_datasets_v1_0_audit.json), [`results/scar_codec_public_datasets_v1_1_audit.json`](results/scar_codec_public_datasets_v1_1_audit.json), and [`results/scar_codec_v6_0_consolidation_audit.json`](results/scar_codec_v6_0_consolidation_audit.json)):
- **Beats Official `ZFP` (`zfpy`) & Complements `CFITSIO FITS fpack`:** **`1,636.2×` (`Spitzer`), `204.4×` (`Bolocam`), and `4.4×` (`FCRAO`) lower bright-source peak error** and **`1.42×–4.05×` lower global RMSE** than byte-matched `ZFP` (`+4.50 dB` higher SNR on `64`-ch `PhysioNet`); beats `FITS fpack` on `Bolocam` in **both** RMSE and peak error (`15.3×`) and by `2.75×` on `Spitzer` stellar peaks.
- **4-Pillar `v6.0` Industrial Consolidation:** **(1)** Zero-PyTorch `NumPyVQ4ScarCodec` + **`0.389 µs` (`389 ns`)** C99/SIMD $\mathcal{O}(1)$ random 64-float block read (`344.2×` faster than PyTorch, **`11,060.8×` faster than `zlib -9`**); **(2)** `.scar` binary container (`CRC32` + zero-copy `ScarMmapFile`) & `ScarZarrCodec` (`codec_id="scar"`); **(3)** Provable deterministic $\|X - \widehat{X}\|_\infty \le \varepsilon$ `abs_tolerance` via Dekker `FP16+FP16` splitting (`11.9992 <= 12.0 MJy/sr` on `3,938` Spitzer peaks); **(4)** **`21.00×` Gosset $E_8$ (`1.000 bpw`)** Spitzer mode (`0.9556 MJy/sr` peak err) & native 3D `ScarCodecND` on `[53, 105, 105]` `FCRAO L1448` (`12.32×` compression, **`0.0566 K` peak err = `2.03×` better than `fpack`**).
- **`NASA Space Shuttle` & `PhysioNet` $\mathcal{O}(1)$ Telemetry:** Fixed `3,120 B` (`Δ = 0 B`, `40,400 FPS`) retains **`6/6 = 100%`** rare faults (`0/6` in ring buffer), while `W_scar` reconstructs orthogonal superposed cortical modes in `6.0 µs` at **`cos = 0.8826`** (vs `0.6917` $1/\sqrt{2}$ `Min-Heap` ceiling).

### 0D. `ScarNativeLM v5.0` (`4L` Foundation Proof — *Tabula Rasa* from `step = 0`)
Audited in [`results/scar_native_v5_0_audit.json`](results/scar_native_v5_0_audit.json) and [`zenodo_deposit_v5_0/Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf`](zenodo_deposit_v5_0/Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf):

| Controlled Tabula-Rasa Model (`step 0 → 520`) | Packed Binary | Dynamic State @ `T = 527` | Held-Out PPL | PTQ Export KL Shock | Exact 3-Slot Passkey (`T = 83 / 127 / 260 / 527`) | 4-Lang Detached Facts (`PT/EN/ES/FR`) & GDPR |
|---|---|---|---|---|---|---|
| **1. `Dense Transformer O(T)` (FP16)** | `5.50 MiB` | `1,054.0 KB` ($\mathcal{O}(T)$) | `1.5833` | `0.0` *(uncompressed)* | `8.3% / 4.2% / 4.2% / 0.0%` | `100.0%` *(entangled in trunk)* |
| **2. `Dense + Post-Hoc VQ4 + Window O(1)`** | `1.06 MiB` | `40.0 KB` (`Δ = 0 B`) | `2.3770` | **`0.394499 nats`** | `0.0% / 0.0% / 0.0% / 0.0%` | `100.0%` *(suffers PTQ shock)* |
| **3. `ScarNativeLM v5.0 Full` (Ours)** | **`1.02 MiB` (`5.39×↓`)** | **`124.25 KB` (`Δ = 0 B!`)** | **`1.0786` *(Best!)*** | **`0.000000 nats!`** | **`100% / 100% / 100% / 87.5%` (`26.4×` window!)** | **`90.62%` (`58/64`, `0 SGD`) · `0.0000` ΔPPL!** |

---

### 1. `16.6k` & `32.9k` Passkey / Needle-in-a-Haystack on Pretrained `Qwen2.5`: `SCache O(1)` vs. Normal KV Cache
Measured live on Apple Silicon M2 (`16 GB`, `MPS`, `FP16`) across non-repeating Python documentation (**zero fine-tuning**, `0.00 B` pointer discrepancy in [`results/passkey_16k_32k_llm_audit.json`](results/passkey_16k_32k_llm_audit.json)):

| Pretrained Model | Context (`L`) | Cache Engine | Physical KV Memory | Throughput (`tok/s`) | Passkey Recall | KV Reduction & Speed vs. Normal KV |
|---|---|---|---|---|---|---|
| **`Qwen2.5-3B-Instruct`** (`36L`) | `32,868` tok | Normal KV $\mathcal{O}(T)$ (`DynamicCache`) | `1,155.80 MiB` (`1.13 GiB`) | `51.5 tok/s` (`638.3s`) | `100%` (`3/3`) | `1.0×` RAM · Quadratic $\mathcal{O}(T^2)$ stall |
| **`Qwen2.5-3B-Instruct`** (`36L`) | `32,868` tok | `StreamingLLM` $\mathcal{O}(1)$ ($W=256$) | `9.00 MiB` *(constant)* | `391.4 tok/s` (`84.0s`) | **`0%` (Evicted)** | `128.4×` less RAM · Amnesia outside $W$ |
| **`Qwen2.5-3B-Instruct`** (`36L`) | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`22.71 MiB` (`Δ = +0 B` vs `16.6k`!)** | **`300.8 tok/s` (`109.3s`)** | **`100%` (`3/3`)** | **`50.9×` less KV RAM · `5.84×` faster!** |
| **`Qwen2.5-0.5B-Instruct`** (`24L`) | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`6.86 MiB` (`Δ = +0 B` vs `16.6k`!)** | **`1,176.9 tok/s` (`27.9s`)** | **`100%` (`5/5`)** | **`56.2×` less KV RAM · `1.60×` faster!** |

### 2. Colossal `72.71B` (`Qwen2.5-72B-Instruct`, `80L`): `< 24 GB` GPU (`Sub-2-Bit`) & `16 GB` Laptop Streaming
Measured with fused Apple Metal `_weight_int4pack_mm` (`50.38×` faster 4L compute) and CUDA Triton `W4A16 / W2A16 + ASVD` ([`scar_cache/triton_mos_gemm.py`](scar_cache/triton_mos_gemm.py), [`scar_cache/sub2bit_mos.py`](scar_cache/sub2bit_mos.py)):
- **`Sub-2-Bit + ASVD` (`2.25 bpw`, `mos-transcode-sub2bit`):** Shrinks `72.71B` across `20` shards to **`20.65 GiB`** (**`21.89 GiB` GPU Resident VRAM** including `lm_head` + `SCache O(1)`), fitting **100% resident on a single `24 GB` RTX 3090 / 4090 / L4** and `< 32 GB` M2 Pro (`cos = 0.99911` under `195,720×` SwiGLU spikes).
- **Canonical `Block-64 INT4 + ASVD` (`38.12 GiB` public SHA-256 shards, `mos-run --download`):** Runs in **`1.92 GiB` peak MPS** on a `16 GB` MacBook (`--streaming`), **`38.56 GiB` resident** on `48 GB+` Macs (`3.46–4.87 tok/s`), and **`4.03–7.27+ tok/s`** on NVIDIA CUDA (`100.0%` Top-5 parity, `120.4–166.0 MiB` constant `SCache O(1)`).
- **Closed-Form `72.71B → 3B` `ScarStore` (`140.00 KB = 35 × 4 KB`, `mos-store`) & Browser `WebGPU` (`mos-webgpu`):** `98.61%` unseen `72.71B` Top-1 agreement in `0` SGD, `18 µs` RAM / `7.64 ms` SSD `0x00` GDPR wipe, Sinkhorn Monomial Re-Basin (`56.36×` lower shared-base KL), and `1,601.57 MiB` `ProtectedInt4Scar + P_⊥CJK` `Qwen2.5-3B` running 100% in-browser at `~21 tok/s` (`164/164` polyglot `PT/EN/ES/FR` `HTTP 206` cartridge routing).

---

## 🚀 Quick Start & Installation

```bash
git clone https://github.com/byte-od/mixture-of-scars.git
cd mixture-of-scars
pip install -e . pytest
```

### 1. Drop-in HuggingFace $\mathcal{O}(1)$ KV-Cache (`SCacheDynamicCache`)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache import SCacheDynamicCache

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

cache = SCacheDynamicCache(window_size=256, num_sink_tokens=4, num_slots=256)
inputs = tokenizer("Your 32k+ token context here...", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)
print(f"Constant SCache Memory: {cache.total_memory_bytes / (1024**2):.2f} MiB (+0.00 B drift!)")
```

### 2. Post-Transformer `ScarField v7.0` (`scar_field` — Zero `ReLU`, Zero `Softmax`, `1.000 bpw` $E_8$)

```python
import torch
from scar_field import ScarFieldConfig, ScarFieldEngine, RiemannianMuon

engine = ScarFieldEngine(ScarFieldConfig(d_form=64, d_world=64, s2_steps=4))
opt = RiemannianMuon(engine.parameters(), lr=2.0e-2)

byte_seq = torch.tensor([[b for b in "SCAR: pure O(1) geodesic flow!".encode("utf-8")]])
losses = engine.compute_loss(byte_seq)
losses["loss"].backward()
opt.step()
```

### 3. Command-Line Tools (`8` Installed CLIs)

```bash
mos-scar-codec --demo                                         # Standalone physical signal/FITS/SCache verification
mos-scar-native verify-parity                                 # Native <-> WebGPU WGSL parity check
mos-store ui --model Qwen/Qwen2.5-3B-Instruct --remote auto   # Local HTTP 206 Cartridge Store Web UI
mos-run --download --chat --model-dir ./models/qwen25_72b     # Download & chat with 72.71B MoS locally
```

---

## 📄 Preprints (Zenodo), Replication Bundles & Repository Map

- **Main Architecture Series (Zenodo Concept DOI [`10.5281/zenodo.24144512`](https://doi.org/10.5281/zenodo.24144512) / Version DOI [`10.5281/zenodo.24172784`](https://doi.org/10.5281/zenodo.24172784)):**
  - **`v6.0.0` (`ScarNativeLM v6.0` & `PointScarFlow v1.0`):** [`zenodo_deposit_v6_0/ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf`](zenodo_deposit_v6_0/ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf)
  - **`v5.0.0` (`ScarNativeLM v5.0` Foundation):** [`zenodo_deposit_v5_0/Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf`](zenodo_deposit_v5_0/Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf)
  - **`v4.0.0` (`Unified WebGPU + ScarCodec`):** [`zenodo_deposit_v4_0/Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf`](zenodo_deposit_v4_0/Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf)
  - **`v3.0.0` (`Online HTTP 206 Library & Re-Basin`):** [`zenodo_deposit_v3_0/Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf`](zenodo_deposit_v3_0/Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf)
- **Standalone Observational Physics & Telemetry Paper (`v1.0.0`):**
  - [`zenodo_deposit_scarcodec_public_data_v1_0/ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf`](zenodo_deposit_scarcodec_public_data_v1_0/ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf) · Package docs in [`scar_codec/README.md`](scar_codec/README.md)
- **Top-Level Packages & Verification:**
  - [`scar_field/`](scar_field/) (`v7.0` Post-Transformer Zero-Legacy Engine) · [`scar_native/`](scar_native/) (`v5.0/v6.0` + `PointScarFlow`) · [`scar_codec/`](scar_codec/README.md) (Physical Operators) · [`scar_cache/`](scar_cache/) (`0.5B–72.71B` LLM Operators & `26` WGSL Shaders) · [`docs/`](docs/) · [`tests/`](tests/) · [`scripts/`](scripts/) · [`results/`](results/).

---

## 📄 License & Citation

Released under the **MIT License** ([LICENSE](LICENSE)). Developed by **Julio Cesar Reis Filho** (*Byte-OD Research & System One Architecture Lab*, 2026).

```bibtex
@techreport{reisfilho2026scarnative_v6,
  author      = {Reis Filho, Julio Cesar da Costa},
  title       = {Shared-Stage Scar-Native Neural Architecture (ScarNativeLM v6.0) and PointScarFlow v1.0: Chunkwise WY Widrow-Hoff Pretraining, Dual-Tier SCache O(1), Procrustes mos-digest Cartridges with 0x00 Document GDPR Unlearning, and Continuous 3D Optimal-Transport Flow},
  institution = {Byte-OD Research \& System One Architecture Lab},
  year        = {2026},
  month       = nov,
  version     = {6.0.0},
  doi         = {10.5281/zenodo.24172784},
  url         = {https://doi.org/10.5281/zenodo.24172784}
}

@techreport{reisfilho2026scarcodec_public,
  author      = {Reis Filho, Julio Cesar da Costa},
  title       = {Empirical Evaluation of Domain-Independent ScarCodec Operators on Public Observational Astronomy, Spacecraft Telemetry, and 64-Channel Biomedical Sensor Archives},
  institution = {Byte-OD Research \& System One Architecture Lab},
  year        = {2026},
  month       = nov,
  version     = {1.0.0},
  url         = {https://github.com/byte-od/mixture-of-scars/tree/main/scar_codec}
}
```
