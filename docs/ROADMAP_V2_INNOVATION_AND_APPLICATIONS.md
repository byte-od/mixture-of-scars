# 🗺️ Official Roadmap (`v1.4` → `v2.x`): Next-Generation `MoS`, `SCache O(1)` & Cross-Domain Applications

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24143839.svg)](https://doi.org/10.5281/zenodo.24143839)
[![Baseline Release](https://img.shields.io/badge/Baseline-v1.3.0%20Audited-brightgreen.svg)](../README.md)
[![72B MoS Checkpoint](https://img.shields.io/badge/72B%20MoS%20Model-38.12%20GiB%20Public-purple.svg)](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos)
[![Architectural Integrity](https://img.shields.io/badge/Mandate-Zero%20Facades%20%2F%20Strict%20O(1)-critical.svg)]()

> **Core Thesis — The Unified Scar Principle:**  
> Do not spend high-precision memory storing predictable routine. Compress the common substrate to its physical hardware floor (`INT4` / `2-bit` bases in space; fixed `O(1)` windows in time) and allocate capacity **strictly where error, outlier amplification, or prediction surprise occurs** — gated by one-sided directional cones with exact `0.000000` out-of-hemisphere interference.

---

## 📐 1. Audited `v1.3.0` Foundation (Where We Stand Today)

Every milestone in this roadmap builds directly upon verified, reproducible `v1.3.0` telemetry (`44/44` `pytest` suites passing, `0.00 B` recursive storage-pointer discrepancy):

$$\mathbf{y}(\mathbf{x}) = F_{\text{base}}(\mathbf{x}) + \sum_{k} \operatorname{ReLU}\!\left(\frac{\mathbf{x}^\top \mathbf{z}_k}{\|\mathbf{x}\|_2 \|\mathbf{z}_k\|_2 + \epsilon}\right) \Delta_{\text{scar}, k}(\mathbf{x})$$

| Pillar | `v1.3.0` Audited Achievement | Open Physical Boundary Targeted in `v2.x` |
| :--- | :--- | :--- |
| **Static Weights (`MoS`)** | `Qwen2.5-72B` (`72.71B`, `195,720×` SwiGLU outliers): `135.43 GiB → 38.12 GiB` (`100%` Top-5 agreement; `-96.37%` / `27.59×` across 8 specialists) | `38.12 GiB` fits `48 GB+` RAM, but spills `~6–14 GiB` to NVMe on `24 GB` GPUs (RTX 3090/4090) and `32 GB` Macs (`M2 Pro`) |
| **Fused GPU Kernels** | Apple Metal `_weight_int4pack_mm` (`50.38×` group speedup, `cos = 1.000000`) + CUDA Triton `Block-64 W4A16+ASVD` (`sm_75+` / `sm_80+`) | Decode is still `1`-token-per-sweep bandwidth-bound without `System 1` speculative verification; browser `WebGPU/WGSL` pending |
| **Dynamic Memory (`SCache`)** | `100%` Passkey recall at `16,568` & `32,868` tokens (`+0.00 B` drift, `22.71 MiB` vs. `1,155.8 MiB` Normal KV, **`5.84×` faster** @ `300.8 tok/s`) | Fixed `S_slots` holds single/few needles cleanly, but needs hierarchical paging for `50+` simultaneous needles at `128k–1M` tokens and offline consolidation into permanent weights |
| **Cross-Layer Base Sharing** | Validated on multi-specialist banks anchored to a shared backbone (`-96.37%` VRAM) | Direct cross-layer sharing on pretrained models incurs `KL ≈ 0.78` without neuron permutation alignment |

---

## 🧭 2. Strategic Execution Architecture (`v1.4` → `v2.x`)

```mermaid
flowchart TD
    V13["✅ Baseline v1.3.0<br>72.71B MoS (38.12 GiB) + SCache O(1) (32.9k, 5.84x faster)<br>Fused Metal (50.38x) & CUDA Triton Kernels"]

    subgraph H1["⚡ Horizon 1: Immediate Speed & 24 GB Hardware Barrier (v1.4)"]
        P11["Track 1.1: System 1 + 72B MoS Fusion<br>Ghost-Jump Speculative Reflex (3x–5x tok/s)"]
        P12["Track 1.2: Sub-2-Bit / Ternary MoS + Rank-96 ASVD<br>72.71B in ~19–21 GiB (100% Resident on 24 GB GPU / 32 GB Mac)"]
    end

    subgraph H2["🧠 Horizon 2: Core Algorithmic Breakthroughs (v2.0)"]
        P21["Track 2.1: Artificial Sleep Cycle (SCache → MoS)<br>Lifelong Continual Consolidation (Zero Forgetting)"]
        P22["Track 2.2: Sinkhorn Re-Basin Permutation Alignment<br>Cross-Layer Base Sharing on Pretrained 70B–405B"]
    end

    subgraph H3["🗄️ Horizon 3: Data Storage, Search Speed & WebGPU (v2.1+)"]
        P31["Track 3.1: Scar-Search & .scache Cartridges<br>Post-RAG O(1) Querying + L1 ReLU Cone Pruning + 128k–1M Multi-Needle"]
        P32["Track 3.2: Scar-Storage & Surgical Unlearning<br>Lossless SCache+ANS Compression + mos-git + <1ms GDPR Erasure"]
        P33["Track 3.3: Native WebGPU/WGSL & Flash-SCache<br>Zero-Install Browser Inference + Single-Pass Attention Kernel"]
    end

    subgraph H4["🌍 Horizon 4: Transdisciplinary Systems"]
        P41["Embodied VLA Robotics · 24/7 AR Video World Models · 34.8 KB Medical Wearables · DoS-Immune Security · Genomics"]
    end

    V13 --> H1 --> H2 --> H3 --> H4
```

---

## ⚡ 3. Horizon 1 — Immediate Speed & Breaking the `24 GB` Hardware Barrier (`v1.4`)

### Track 1.1: `System 1` + `System 2` Speculative Reflex Coupling (*Ghost-Jump Decoding*)
* **Problem:** At batch-1 decode (`M = 1`), reading `38.12 GiB` from Unified Memory or GDDR6X takes virtually the same wall-clock time as verifying `M = 8` candidate tokens simultaneously (already supported by our `M <= 8` register-coalesced path in `scar_cache/triton_mos_gemm.py`).
* **Engineering Plan:**
  1. Pair the `Universal System 1` / `Qwen2.5-0.5B-Instruct` `MoS + SCache` scout (`1,176.9 tok/s`, `6.86 MiB` `SCache`) with the `72.71B MoS` verifier (`scar_cache/cli_72b.py`).
  2. Use `SCache` resonance confidence (`margin = P_1 - P_2`) to dynamically adapt draft horizon length (`K = 4..8` tokens).
  3. Execute exact speculative rejection sampling over a single `M = K + 1` forward pass on `72.71B`, rolling back both the draft and target `SCache` windows/fast-weights atomically on rejection.
* **Acceptance KPIs:**
  * **`3.0× – 5.0×` end-to-end `tok/s` decode speedup** on `48 GB+` Apple Silicon (`~12–20 tok/s`) and `24 GB+` CUDA (`~20–35+ tok/s`).
  * **Exact mathematical equivalence** (`KL = 0.0000` under greedy/rejection sampling) to standalone `72.71B MoS`.

### Track 1.2: `MoS Sub-2-Bit` (`2-bit` / `1.58-bit` Base + `ASVD` Outlier Shield)
* **Problem:** Standard `2-bit` quantization collapses on `70B+` models due to `195,720×` SwiGLU intermediate activation spikes. Yet `MoS` *already* isolates those spikes into activation-weighted `ASVD` low-rank scars!
* **Engineering Plan:**
  1. Compress healthy non-salient base blocks from `Block-64 INT4` (`4.25 bits/weight` effective) to symmetric/codebook **`2-bit` (`~2.25 bits/weight`)** across all `240` SwiGLU MLPs of `Qwen2.5-72B`, while scaling `ASVD` rank from `48 → 96/128` (`~1.6 GiB` total scar budget).
  2. Extend `scar_cache/triton_mos_gemm.py` (4-values-per-byte nibble unpacking in registers) and Apple Metal custom bit-unpacking kernels.
* **Acceptance KPIs:**
  * Total `72.71B` checkpoint footprint reduced from `38.12 GiB` to **`~19.5 – 21.5 GiB`**.
  * **`100%` RAM-resident execution** (`0 MB/tok` SSD paging!) on a single **`24 GB` NVIDIA GPU (RTX 3090 / 4090 / L4)** and **`32 GB` Apple M2 Pro**.
  * Unseen-domain Top-5 agreement `>= 98.5%` and non-repeating PPL delta `<= +0.25` vs. `v1.2` `INT4+ASVD`.

---

## 🧠 4. Horizon 2 — Core Algorithmic & Scientific Innovations (`v2.0`)

### Track 2.1: "The Artificial Sleep Cycle" — Autonomous `SCache O(1) → MoS` Consolidation
* **Scientific Breakthrough:** Biological brains capture high-surprise daytime shocks in the **Hippocampus** (fast, bounded associative memory = `SCache O(1)`) and consolidate recurring/critical patterns into the **Neocortex** during sleep (permanent synaptic low-rank scars = `MoS`), clearing hippocampal slots for the next day.
* **Mathematical Formulation:**
  During idle/consolidation phases, extract the top singular vectors of accumulated Widrow-Hoff fast-weight deltas and high-surprise micro-page key-value pairs, solving a constrained rank-$r$ distillation into a new directional `ScarExpert` $(\mathbf{B}_{\text{new}}, \mathbf{A}_{\text{new}}, \mathbf{z}_{\text{new}})$:

$$\min_{\mathbf{B}_{\text{new}}, \mathbf{A}_{\text{new}}} \sum_{(\mathbf{x}, \mathbf{y}^\star) \in \mathcal{M}_{\text{surprise}}} \left\| \mathbf{y}^\star - \widehat{\mathbf{W}}_{\text{base}}\mathbf{x} - \operatorname{ReLU}(\cos(\mathbf{x}, \mathbf{z}_{\text{new}}))\,\mathbf{B}_{\text{new}}(\mathbf{A}_{\text{new}}\mathbf{x}) \right\|_2^2$$

  followed by orthogonal slot release in `SCache`.
* **Acceptance KPIs:**
  * Lifelong continual adaptation across `1,000+` multi-session facts at strictly constant `O(1)` runtime RAM.
  * **`0.000000` out-of-cone catastrophic forgetting** on unrelated domains (guaranteed by `ReLU` spherical-cap gating).

### Track 2.2: *Sinkhorn Re-Basin* Cross-Layer Permutation Alignment
* **Scientific Breakthrough:** Resolve Section 5 Limitation #1 of the `v1.3` preprint (`KL ≈ 0.78` when sharing dense bases across distinct pretrained layers).
* **Mathematical Formulation:**
  Before computing `ASVD` residuals between adjacent Transformer layers $\ell$ and $\ell+1$, align their hidden/intermediate permutation symmetries via entropic Soft Sinkhorn bistochastic optimization:

$$\mathbf{P}^\star = \arg\min_{\mathbf{P} \in \mathcal{B}_n} \left\| \mathbf{W}_{\ell+1}\operatorname{diag}(\mathbf{s}_{\ell+1}) - \mathbf{P}\,\mathbf{W}_{\ell}\operatorname{diag}(\mathbf{s}_{\ell})\,\mathbf{Q}^\top \right\|_F^2 - \tau H(\mathbf{P})$$

  followed by Hungarian projection to an exact permutation matrix (zero runtime overhead once folded into indices!).
* **Acceptance KPIs:**
  * Cut cross-layer shared-base KL divergence from `0.78` to **`< 0.08`** on pretrained `7B–72B` models without pretraining from scratch.

---

## 🗄️ 5. Horizon 3 — Data Storage, Search Speed & Browser Infrastructure (`v2.1+`)

### Track 3.1: Search Speed — *Scar-Search*, `.scache` Cartridges & `128k–1M` Multi-Needle Hierarchy
1. **L1-Cache `ReLU` Geometric Pre-Filter (`< 5 ns` Shard Pruning):**
   * Pack up to `256` normalized semantic centroids $\mathbf{z}_k$ into a **`34.8 KB` L1-resident index** (`scar_memory.py` layout).
   * Execute SIMD/GPU one-sided spherical-cap gating $\operatorname{ReLU}(\cos(\mathbf{q}, \mathbf{z}_k) - \tau)$ to prune `50%–95%` of cold vector shards *before* touching DDR5 RAM or NVMe HNSW graphs.
2. **Surprise-Guided Zero-Bloat Indexing (`20×–50×` Smaller Vector DBs):**
   * Filter enterprise corpora through the Widrow-Hoff Surprise Oracle $\|\mathbf{v}_t - \mathbf{W}_{\text{scar}}\hat{\mathbf{k}}_t\|_2$ during ingestion: absorb repetitive boilerplate (`90%+` of legal/log corpora) into `O(1)` topic matrices and index *only* high-surprise micro-pages.
3. **Portable `.scache` Knowledge Cartridges ("Post-RAG `O(1)` Engine") & Multi-Tier Paging:**
   * Ship `mos-digest <repo_or_docs> -o cartridge.scache`: pre-digest entire repositories or technical manuals once into a fixed-size `.scache` file (`~22–64 MiB`) mountable in milliseconds with constant `O(1)` query speed, paired with hierarchical `L1/L2` micro-page slots to retain `50+` simultaneous needles at `128k–1M` context lengths.

### Track 3.2: Data Storage — *Scar-Storage*, `mos-git` & Sub-Millisecond *Machine Unlearning*
1. **Lossless Long-Range `SCache + ANS` Entropy Compressor (`scar-zip`):**
   * Couple a lightweight deterministic `SCache O(1)` byte/token predictor with Asymmetric Numeral Systems (`ANS`) arithmetic coding: `W_scar` tracks long-range structure across multi-gigabyte dumps/genomes in constant RAM while `ANS` encodes the exact prediction residual **100% losslessly**, overcoming `gzip`/`zstd` window amnesia without `O(T^2)` neural compression stalls.
2. **`mos-git` — Delta-Scar Version Control for Giant Tensors & Datasets:**
   * Store `1` canonical quantized base + `KB/MB`-scale activation-weighted `ASVD` diffs per fine-tune, checkpoint epoch, or scientific simulation snapshot (`-96%+` storage footprint).
3. **Surgical `< 1 ms` *Machine Unlearning* (GDPR / LGPD Right-to-be-Forgotten):**
   * Expose `scache.unlearn(key)` executing exact Widrow-Hoff orthogonal subspace projection $\mathbf{W}_{\text{scar}} \leftarrow \mathbf{W}_{\text{scar}}(\mathbf{I} - \hat{\mathbf{k}}\hat{\mathbf{k}}^\top)$, mathematically erasing targeted user associations in `< 1 ms` without retraining or corrupting orthogonal memories.

### Track 3.3: Native `WebGPU / WGSL` Shaders & Fused `Flash-SCache` Kernel
* **Strict Zero-Facade Mandate:** Implement genuine `WGSL` compute shaders for `Block-64 / 2-bit + ASVD` register fusion and `SCache` Widrow-Hoff Delta updates so `MoS + SCache` executes natively inside Chrome/Safari (`WebGPU`) with zero server dependency.
* Fuse `Attention Sinks + Rolling Window + Micro-Page Slots + W_scar Associative Readout` into a single `Flash-SCache` Triton/Metal kernel to eliminate Python wrapper overhead at `100k+` token prefill.

---

## 🌍 6. Horizon 4 — High-Impact Transdisciplinary Applications

| Domain | Core Bottleneck Today | `MoS + SCache` Architectural Solution | Target Impact |
| :--- | :--- | :--- | :--- |
| **1. Embodied Robotics & Autonomous Vehicles** | `7B–70B` VLA models exceed onboard power/VRAM; local floor/joint friction changes require cloud retraining | Onboard fused `MoS` trunk + `34.8 KB` `SCache` proprioceptive shock scars (`0.000000` leak on normal terrain) | `< 1 ms` real-time physical adaptation without cloud round-trips |
| **2. 24/7 AR Smart Glasses & Video World Models** | `2` minutes of video = `32k+` tokens (`O(T^2)` collapse or `StreamingLLM` amnesia) | Widrow-Hoff Surprise Oracle absorbs static scenes (`+0 B/hour`) and locks only high-surprise state-change frames | All-day `O(1)` visual episodic memory with ghost-free object relocation |
| **3. Multi-Tenant Enterprise Cloud SaaS** | Hosting `1,000` customized `72B` client models requires `~135 TB` VRAM | `1` frozen `MoS` base + `1,000` hot-swappable `ScarExpert` cartridges + fixed-ceiling `SCache` per session | **`-96%+` VRAM cost** and zero OOM tail-latency risk |
| **4. Medical Wearables & ICU Biosignals (ECG/EEG)** | Tiny MCU SRAM (`< 256 KB`) vs. months of continuous physiological streams | `34.8 KB` `GradientScarMemory` in SRAM learning patient baseline and locking rare arrhythmia micro-windows | Ultra-low-power personalized anomaly capture on edge silicon |
| **5. Cybersecurity & High-Frequency Fraud** | Attackers use `500×` noise flooding (*DoS*) to wash out online detectors | `ReLU` spherical-cap immunity (`Vector 4: 0.000000` antipodal leak) + instant Delta signature updates | Inline sub-millisecond drift adaptation immune to noise dilution |
| **6. Computational Genomics & Proteomics** | Chromosome-scale sequences (`10M+` bp) + extreme structural outlier sensitivity in folding MLPs | `MoS` outlier-shielded weights on lab GPUs + `SCache` scanning massive sequences at `O(1)` RAM | Democratized `70B`-scale bio-models on single university workstations |
| **7. Satellite / Deep-Sea / Swarm Telemetry** | Expensive/contested radio links saturated by predictable sensor streams | Synchronized sender/receiver `W_scar`: transmit `0` bytes when $\|\mathbf{e}_t\|_2 < \tau$, sending only surprise scar packets | Orders-of-magnitude bandwidth reduction on constrained links |

---

## ⚖️ 7. Physical Boundary & Scientific Honesty Covenant

In strict compliance with our **Zero-Facade Engineering Mandate**:
1. **Shannon Entropy Floor:** No associative matrix or low-rank scar can losslessly compress pure cryptographic white noise (e.g., `1,000,000` uncorrelated random keys with zero structural redundancy) below Shannon's information-theoretic bit floor. `MoS` and `SCache` achieve `25×–280×` gains precisely because real-world weights, code, language, video, and physical sensors exhibit extreme power-law structure (`95%–99.9%` predictable manifold + `0.1%–5%` high-surprise outliers/needles).
2. **Mandatory Byte-Pointer & Causality Audits:** Every milestone above (`v1.4` through `v2.x`) must pass recursive `untyped_storage().data_ptr()` byte verification (`0.00 B` unreported allocation), mid-flight causality ablations, and public reproducible benchmarks before release.
