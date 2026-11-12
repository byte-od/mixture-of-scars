---
language:
- en
- pt
license: mit
tags:
- mixture-of-scars
- mos
- scache
- fast-weights
- kv-cache
- needle-in-a-haystack
- transformers
- efficiency
- long-context
base_model: Qwen/Qwen2.5-72B-Instruct
pipeline_tag: text-generation
---

# 🧬 Mixture of Scars (`MoS v2`) & `SCache` $\mathcal{O}(1)$ (`v1.3.0`)

**Developed by:** Julio Cesar Reis Filho — Byte-OD Research & System One Architecture Lab (2026)  
**Preprint (CERN / Zenodo DOI):** [https://doi.org/10.5281/zenodo.24143839](https://doi.org/10.5281/zenodo.24143839)  
**Code & Public `72.71B` Shards:** [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars)  
**License:** MIT  

---

## 💡 Overview

**Mixture of Scars (MoS)** and **SCache $\mathcal{O}(1)$** eliminate both halves of the Transformer memory wall without retraining from scratch:
1. **Mixture of Scars (`MoS v2`):** Pairs canonical `Block-64 INT4` bases with Activation-Aware SVD (`ASVD`) low-rank scar experts (`135.43 GiB → 38.12 GiB` standalone at `72.71B`; **`-96.37%` / `27.59x`** across an 8-specialist bank) and `20 × 4`-layer streaming shards (`1.79 GiB`/shard), accelerated by fused Apple Silicon Metal (`_weight_int4pack_mm`, **`50.38x` faster compute**) and NVIDIA CUDA Triton `W4A16 + ASVD` kernels (`sm_75+`).
2. **SCache (`SCacheDynamicCache` $\mathcal{O}(1)$):** Drop-in HuggingFace `Cache` combining attention sinks, a rolling FIFO window, synchronized deep-layer `32`-token micro-page routing, and per-KV-head orthogonal **Widrow-Hoff Delta Rule** fast weights (`0.00 B` marginal memory growth at any sequence length).

---

## 🏆 Audited Benchmark Highlights (`v1.3.0`)

### 1. `16.6k` & `32.9k` Passkey / Needle-in-a-Haystack (Non-Repeating Python Docs, Zero Fine-Tuning)
| Model | Context (`L`) | Cache Engine | Physical KV Memory (`Δ = 0 B`) | Throughput (`tok/s`) | Passkey Recall | Advantage vs. Normal KV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-3B-Instruct`** | `16,568` tok | Normal KV $\mathcal{O}(T)$ | `582.75 MiB` | `245.3 tok/s` (`67.5s`) | `100%` (`5/5`) | `1.0x` |
| **`Qwen2.5-3B-Instruct`** | `16,568` tok | `StreamingLLM` ($W=256$) | `9.00 MiB` | `394.0 tok/s` (`42.1s`) | **`0%` (Evicted)** | Amnesia outside $W$ |
| **`Qwen2.5-3B-Instruct`** | **`16,568` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`22.71 MiB` (`+0 B`)** | **`299.0 tok/s` (`55.4s`)** | **`100%` (`5/5`)** | **`25.7x` smaller · `1.22x` faster** |
| **`Qwen2.5-3B-Instruct`** | `32,868` tok | Normal KV $\mathcal{O}(T)$ | `1,155.80 MiB` | `51.5 tok/s` (`638.3s`) | `100%` (`3/3`) | $\mathcal{O}(T^2)$ stall |
| **`Qwen2.5-3B-Instruct`** | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`22.71 MiB` (`+0 B`)** | **`300.8 tok/s` (`109.3s`)** | **`100%` (`3/3`)** | **`50.9x` smaller · `5.84x` faster!** |
| **`Qwen2.5-0.5B-Instruct`** | **`32,868` tok** | **`SCache` $\mathcal{O}(1)$ ($W=256, S=256$)** | **`6.86 MiB` (`+0 B`)** | **`1,176.9 tok/s` (`27.9s`)** | **`100%` (`5/5`)** | **`56.2x` smaller · `1.60x` faster!** |

### 2. Colossal `72.71B` (`Qwen2.5-72B-Instruct`, `80` Layers) Multi-Hardware Execution
- **Unseen-Domain Fidelity under `195,720x` SwiGLU Outliers:** **`100.0%` Top-5 token agreement** across Python, Quantum Physics, Tech Portuguese, and Strict JSON (`3.08x–3.29x` lower KL error than uniform `INT4`; non-repeating PPL **`4.75`** vs. **`4.35`** Dense FP16).
- **Apple Silicon (`MPS`) & NVIDIA (`CUDA`):** `1.92 GiB` peak active MPS on `16 GB` M2 (`4.23 tok/s` prefill); `~3.46–4.87 tok/s` resident on `48 GB` M4 Pro / `64 GB+` M2 Max (`38.56 GiB` MPS + `128.51 MiB` `SCache`); `4.03–7.27+ tok/s` on NVIDIA CUDA (`155x–280x` faster than `AirLLM` FP16 NVMe).

---

## 🚀 Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache import SCacheDynamicCache

model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

cache = SCacheDynamicCache(window_size=256, num_slots=256, num_sink_tokens=4)
inputs = tokenizer("Explain the Widrow-Hoff Delta Rule in three concise sentences.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Constant O(1) SCache Memory: {cache.total_memory_bytes / (1024**2):.2f} MiB")
```
