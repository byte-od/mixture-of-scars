---
title: Mixture of Scars (MoS) - Running OOM Models on Free 16GB CPU
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Mixture of Scars (MoS) & SCache: Breaking the Dual Memory Wall

Official interactive demonstration of **Mixture of Scars (MoS)** and **SCache**.

See how long contexts (up to 50,000+ tokens) and deep 70B-scale architectures that traditionally cause **Out-Of-Memory (OOM)** errors run smoothly inside a **Free 16GB CPU Space**!

* **GitHub Repository:** [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars)
* **Zenodo Preprint & Replication Deposit (`v1.3.0`):** [10.5281/zenodo.24143839](https://doi.org/10.5281/zenodo.24143839)
* **Audited Highlights (`v1.3.0`):** `100%` Passkey recall at `16.6k` & `32.9k` tokens (`22.71 MiB` constant `O(1)` KV RAM, `+0.00 B` drift, `50.9x` less KV RAM & `5.84x` faster than Normal KV), `72.71B` `Qwen2.5-72B-Instruct` `100%` Top-5 retention (`195,720x` SwiGLU outlier neutralization), and `50.38x` Fused Apple Metal + NVIDIA CUDA Triton `W4A16+ASVD` kernels.
* **License:** MIT
* **Research Team:** Julio Cesar Reis Filho & Byte-OD Lab (2026)
