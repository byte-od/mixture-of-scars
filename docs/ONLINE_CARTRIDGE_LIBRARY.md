# 🌐 Official Online Cartridge Library (`v3.0.0` — Live `HTTP/1.1 206` Cloud Streaming)

The **Mixture of Scars (`MoS v3.0`) Online Cartridge Library** hosts real closed-form knowledge cartridges extracted with **ZERO backpropagation / ZERO SGD** from **`Qwen2.5-72B-Instruct MoS v2` (`72.71B`, `80` layers)** for both:
- **`Qwen/Qwen2.5-3B-Instruct`** (`36L`, $d = 2048$) — exact **`143,360 Bytes` (`140.00 KB = 35 × 4,096 B` OS `mmap` pages, `0 B` internal padding)** per cartridge.
- **`Qwen/Qwen2.5-0.5B-Instruct`** (`24L`, $d = 896$) — exact **`62,720 Bytes` (`61.25 KB`)** per cartridge.

All assets are served publicly 24/7 via **RFC 7233 `HTTP/1.1 206 Partial Content` (`Range: bytes=...`)** directly from the GitHub Release CDN (`releases/tag/v3.0.0`) — **zero custom server required**.

---

## ⚡ Quick Start (Visual Web UI + CLI)

### 0. Launch the Zero-Dependency Visual Cartridge Store & Live Chat (`mos-store ui`)
Open an App-Store-style local web interface (`http://127.0.0.1:7860`) with **zero extra Python packages** — browse human-named cartridges, stream `140 KB` / `61.25 KB` slices live via `HTTP 206`, chat with `SCache O(1)` telemetry, mint new Open-Weights cartridges, and trigger 1-click `0x00` LGPD/GDPR erasure:
```bash
mos-store ui --model Qwen/Qwen2.5-3B-Instruct --remote auto
```

### 1. Browse the Live Online Catalog (No GPU or Model Required)
```bash
mos-store catalog
```

### 2. Ask Any Question with Live On-Demand `140 KB` Streaming (`--remote auto`)
On first run, `mos-store` downloads a lightweight **Zero-Payload Cloud Shell** (`~/.cache/mixture_of_scars/scarstore/qwen3b_v3_0.shell.scarstore`), routes your prompt semantically on the local Kronecker grid, and fetches **ONLY the exact `140.00 KB` (`35 × 4 KB`) byte slice** needed for your topic over `HTTP 206`:

```bash
# Run on Qwen2.5-3B-Instruct (streams 140.00 KB slice on demand):
mos-store query --remote auto --model Qwen/Qwen2.5-3B-Instruct \
  --prompt "Explain concisely what the Fischer-Lynch-Paterson impossibility theorem proves about asynchronous consensus."

# Or run on ultra-light Qwen2.5-0.5B-Instruct (streams 61.25 KB slice on demand):
mos-store query --remote auto --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain concisely what Landauer's principle states for the minimum energy dissipated when erasing one bit at temperature T."
```

Add `--persist-fetched` to save streamed cartridges permanently into your local `.scarstore` (`0 ms` `mmap` on future queries), or `--chat` for an interactive `SCache O(1)` terminal conversation!

### 3. Mass-Produce Custom Cartridges from Open-Weight Models (`Zero SGD`)
Mint brand-new `140 KB` / `61.25 KB` cartridges from **any** Hugging Face open-weight specialist (`same_vocab_exact` for `Qwen2.5-7B/32B/72B`, `Qwen2.5-Coder`, `DeepSeek-R1-Distill-Qwen`; `cross_tokenizer_bridge` for `Mistral`, `Llama-3`, `Gemma-2`; or `curated_trajectory` JSON exports from `405B`/`671B` clusters):
```bash
# Live extraction from any Open-Weight Teacher (appends into existing store):
mos-store build --student Qwen/Qwen2.5-3B-Instruct --teacher Qwen/Qwen2.5-7B-Instruct \
  --capsule-id ow_postgres_mvcc --title "PostgreSQL Internals — MVCC Snapshot Isolation" \
  --category "Databases" --prompt "Explain how PostgreSQL MVCC visibility checks xmin and xmax." \
  --out ./my_3b_library.scarstore --append

# Batch-mint multiple cartridges from a JSON spec file (< 1 s per cartridge):
mos-store build --student Qwen/Qwen2.5-3B-Instruct \
  --spec-json examples/open_weights_cartridge_specs_example.json \
  --out ./my_3b_library.scarstore --append
```

### 4. Download Offline Pack & Test Surgical `0x00` GDPR Unlearning
```bash
# Download the complete 9-cartridge store locally:
mos-store pull --model 3b --mode full --out ./my_3b_library.scarstore

# Inspect all human-readable titles, Kronecker coordinates & 4 KB OS page offsets:
mos-store inspect --store ./my_3b_library.scarstore

# Surgically wipe ONE cartridge (0x00 overwrite + fsync in < 10 ms) without touching the other 8:
mos-store unlearn --store ./my_3b_library.scarstore --id titan72b_fischer_lynch_paterson --persist
```

---

## 📚 Available `v3.0.0` Online Cartridges (`9` Slots)

| # | Human Title (EN / PT-BR) | Category | Cartridge ID (`--id`) | `3B` Size | `0.5B` Size |
|---|---|---|---|---|---|
| `01` | **High-Dimensional Sphere Packing — Kabatiansky-Levenshtein Bound**<br>*Empacotamento Esférico — Cota de Kabatiansky-Levenshtein* | Geometry & Coding Theory | `titan72b_kabatiansky_levenshtein` | `140.00 KB` | `61.25 KB` |
| `02` | **Thermodynamics of Computation — Landauer's Principle ($k_B T \ln 2$)**<br>*Termodinâmica da Computação — Princípio de Landauer* | Physics of Information | `titan72b_landauer_principle` | `140.00 KB` | `61.25 KB` |
| `03` | **Space Complexity — Immerman-Szelepcsényi Theorem ($\text{NSPACE}=\text{co-NSPACE}$)**<br>*Complexidade Espacial — Teorema de Immerman-Szelepcsényi* | Theoretical Computer Science | `titan72b_immerman_szelepcsenyi` | `140.00 KB` | `61.25 KB` |
| `04` | **Distributed Consensus — FLP Impossibility Theorem**<br>*Sistemas Distribuídos — Teorema da Impossibilidade FLP* | Distributed Systems | `titan72b_fischer_lynch_paterson` | `140.00 KB` | `61.25 KB` |
| `05` | **Algorithmic Information Theory — Chaitin's Incompleteness Theorem**<br>*Informação Algorítmica — Incompletude de Chaitin (Kolmogorov)* | Logic & Information Theory | `titan72b_chaitin_incompleteness` | `140.00 KB` | `61.25 KB` |
| `06` | **Post-Quantum Cryptography — Regev's LWE Worst-Case Lattice Reduction**<br>*Criptografia Pós-Quântica — Redução Quântica LWE de Regev* | Cryptography & Security | `titan72b_regev_lwe_lattice` | `140.00 KB` | `61.25 KB` |
| `07` | **Random Matrix Theory — Marchenko-Pastur Spectral Distribution**<br>*Matrizes Aleatórias em IA — Lei Espectral de Marchenko-Pastur* | Spectral Math & Deep Learning | `titan72b_marchenko_pastur_law` | `140.00 KB` | `61.25 KB` |
| `08` | **Communications & Signal Theory — Shannon-Hartley Channel Capacity**<br>*Engenharia de Comunicações — Lei de Capacidade de Shannon-Hartley* | Communications & Information | `titan72b_shannon_hartley_capacity` | `140.00 KB` | `61.25 KB` |
| `09` | **Interdisciplinary Sleep Bridge — Thermodynamics × NSPACE Complexity**<br>*Ponte Interdisciplinar — Termodinâmica de Landauer × Complexidade NSPACE* | Interdisciplinary Synthesis | `bridge_landauer_immerman` | `140.00 KB` | `61.25 KB` |

---

## 🔗 Direct CDN Endpoints (`HTTP/1.1 206 Range` Ready)

- **Catalog JSON:** `https://github.com/byte-od/mixture-of-scars/releases/download/v3.0.0/scarstore_catalog_v3_0.json`
- **`Qwen2.5-3B` Master Store (`140 KB`/slot):** `https://github.com/byte-od/mixture-of-scars/releases/download/v3.0.0/qwen3b_v3_0.scarstore`
- **`Qwen2.5-3B` Manifest & Zero-Payload Shell:** `qwen3b_v3_0.manifest.json` · `qwen3b_v3_0.shell.scarstore.gz`
- **`Qwen2.5-0.5B` Master Store (`61.25 KB`/slot):** `https://github.com/byte-od/mixture-of-scars/releases/download/v3.0.0/qwen0p5b_v3_0.scarstore`
- **`Qwen2.5-0.5B` Manifest & Zero-Payload Shell:** `qwen0p5b_v3_0.manifest.json` · `qwen0p5b_v3_0.shell.scarstore.gz`
