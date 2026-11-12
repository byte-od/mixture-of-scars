---
title: Mixture of Scars (MoS) & SCache O(1) Interactive Demo
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
tags:
- mixture-of-scars
- scache
- kv-cache
- fast-weights
- efficiency
- long-context
---

# 🧬 Mixture of Scars (MoS) & SCache Interactive Demo (`v1.3.0`)

Interactive demonstration of:
1. **SCache DynamicCache $O(1)$:** Constant physical memory KV-cache replacement (`22.71 MiB`, `+0.00 B` drift from `16.6k` to `32.9k` tokens) achieving `100%` Passkey recall across `5%–95%` depths (`50.9x` less KV RAM & `5.84x` faster than `DynamicCache` on `Qwen2.5-3B-Instruct`).
2. **Mixture of Scars (MoS):** Symmetric `Block-64 INT4` bases + Activation-Aware SVD (`ASVD`) low-rank scars compressing `Qwen2.5-72B-Instruct` (`135.43 GiB → 38.12 GiB` standalone, `27.59x` across 8 specialists) with `100%` Top-5 preservation (`195,720x` SwiGLU outlier rescue) and `50.38x` Fused Apple Metal + NVIDIA CUDA Triton `W4A16+ASVD` kernels.
3. **Needle In A Haystack (NIAH):** Audited on real non-repeating Python documentation up to `32,868` tokens and synthetic stress sweeps up to `64k` tokens.

* **GitHub Repository:** [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars)
* **Zenodo Preprint & Replication Deposit (`v1.3.0`):** [10.5281/zenodo.24143839](https://doi.org/10.5281/zenodo.24143839)
