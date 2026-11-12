---
title: Mixture of Scars & SCache O(1) Red-Team Lab
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: true
license: mit
short_description: Live O(1) Widrow-Hoff KV Cache (32.9k NIAH) + MoS ASVD INT4 Lab
tags:
  - kv-cache
  - fast-weights
  - widrow-hoff
  - quantization
  - needle-in-a-haystack
  - transformers
  - pytorch
---

# Mixture of Scars (`MoS`) & `SCache` $\mathcal{O}(1)$ — Live Adversarial Red-Team Laboratory (`v1.3.0`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24172784.svg)](https://doi.org/10.5281/zenodo.24172784)
[![GitHub Release](https://img.shields.io/badge/GitHub-v1.3.0-181717.svg)](https://github.com/byte-od/mixture-of-scars)

Interactive verification laboratory for **[Mixture of Scars (MoS) & SCache $\mathcal{O}(1)$](https://doi.org/10.5281/zenodo.24172784)** by **Julio Cesar Reis Filho (*Byte-OD Research & System One Architecture Lab*)**.

## Highlights Verified in `v1.3.0` (`44/44` `pytest` Suite)
- **`16.6k` & `32.9k` Passkey / Needle-in-a-Haystack (`100%` Recall, `+0.00 B` Drift):** On `Qwen/Qwen2.5-3B-Instruct` (`36L`) across non-repeating Python documentation, `SCacheDynamicCache` ($W=256, S=256$) achieves **`100%` recall** at both `16,568` (`5/5`) and `32,868` tokens (`3/3`) inside a strictly constant **`22.71 MiB`** (`50.9x` smaller than `1,155.80 MiB` Normal KV) while running **`5.84x` faster** (`300.8` vs. `51.5 tok/s` at `32.9k`), whereas `StreamingLLM` scores **`0%`**.
- **Colossal `72.71B` (`Qwen2.5-72B-Instruct`, `80L`, `195,720x` SwiGLU Outliers):** Compresses `135.43 GiB → 38.12 GiB` (`-96.37%` / `27.59x` across 8 specialists) with **`100.0%` Top-5 token agreement**, accelerated by fused Apple Metal `_weight_int4pack_mm` (**`50.38x` faster compute**, `cos = 1.000000`) and NVIDIA CUDA Triton `Block-64 W4A16 + ASVD Scar` kernels (`sm_75+`).

## What You Can Test Live in This Space
1. **Live Pretrained LLM Evicted-Passcode & Byte-Pointer Audit (`Qwen/Qwen2.5-0.5B-Instruct`):**
   - Buries your custom passcode near token ~20, floods the context past `window_size=64` so the passcode is **physically evicted ~470 tokens prior to query**, walks `tensor.untyped_storage().data_ptr()` recursively to verify **`0.00 KB` hidden byte discrepancy**, and demonstrates **`22.58x` higher exact target probability** than **StreamingLLM** with zero fine-tuning.
2. **Widrow-Hoff Delta vs. Additive Hebbian Anti-Ghosting Arena (`34.8 KB` Canonical Buffer):**
   - Mutates state variables repeatedly on the same key alongside orthogonal distractors and `500x` antipodal DoS noise to demonstrate exact orthogonal state overwrite (`cos = 1.0000`) vs. additive Hebbian ghosting.
3. **Live 24-Layer `MoS` (`INT4` + `ASVD` Scar Experts) Outlier & Causality Audit:**
   - Converts all `72` SwiGLU MLP projections of `Qwen/Qwen2.5-0.5B-Instruct` into `INT4 + ASVD Scar Experts` calibrated on a single 24-token generic English sentence and tests unseen Python, JSON, Quantum Physics, or your own custom prompt (`>3x` lower logit KL error + `-92%+` 8-specialist multi-tenant RAM).

## Quick Drop-in Usage (`transformers`)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache.hf_cache import SCacheDynamicCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

cache = SCacheDynamicCache(
    window_size=256,
    num_slots=256,
    num_sink_tokens=4,
    project_slots=True,
)
```

## 📦 Download `70B`-Class Model (`Qwen2.5-72B-Instruct MoS v2`) & Practical Guide
All `22` **SHA-256 verified** `.safetensors` shards (`38.12 GiB` total, `20 × 1.79 GiB` 4-layer groups) are publicly available with **zero login required**:
- **Direct Model Download (`HTTP 206 Range`):** [github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
- **Step-by-Step Practical Local Deployment Guide (Apple Silicon, CUDA & CPU):** [docs/GUIA_PRATICO_MOS_SCACHE_72B.md](https://github.com/byte-od/mixture-of-scars/blob/main/docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
```bash
mos-run --download --chat --model-dir ./models/qwen25_72b_mos_v2
```

## Official Citation
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
