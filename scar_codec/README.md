# `scar_codec` (`v6.0.0`) — Zero-PyTorch C/SIMD Physical Signal, FITS/Zarr Cube, $E_8$ & Constant-Memory Telemetry Library

[![License: MIT](https://img.shields.io/badge/License-MIT%20(Code)%20%7C%20CC--BY--4.0%20(Report)-blue.svg)](../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero PyTorch Required](https://img.shields.io/badge/Dependencies-NumPy%20%2B%20Auto%20C99%2FSIMD%20(PyTorch%20Optional)-0d9488.svg)](numpy_engine.py)
[![C/SIMD Speed](https://img.shields.io/badge/O(1)%20Block%20Read-0.388%20%C2%B5s%20C99%2FSIMD%20mmap%20(11%2C088x%20%3E%20zlib)-16a34a.svg)](c_kernels.c)
[![Zenodo Preprint v2.0](https://img.shields.io/badge/Zenodo%20Preprint%20v2.0-NASA%20%7C%20ZFP%20%7C%20fpack%20%7C%20Gosset%20E8-0284c7.svg)](../zenodo_deposit_scarcodec_v2_0/ScarCodec_v6_Public_Observational_Datasets_Preprint_v2.0.pdf)
[![JOSS Paper](https://img.shields.io/badge/JOSS%20Submission%20Ready-paper%2Fpaper.md-7c3aed.svg)](../paper/paper.md)

`scar_codec` extracts the five pure mathematical operators of **Mixture of Scars (`MoS`)** and **`SCache` $\mathcal{O}(1)$** into a standalone, **neural-network-free** signal, scientific array, storage, and black-box telemetry library (`0` dependency on `transformers`, `0` trained weights):

1. **Multiplierless Orthonormal `FWHT-64` + Channel Equilibration (`d_eq_f16`):** Spreads localized shock energy isotropically across 64-element blocks using addition/subtraction butterflies only.
2. **`2 KB` SRAM $D_4/D_4^*$-Seeded `LUT256` Vector Quantizer with Hard-Pinned Exact Vacuum (`0x00`):** Seeded with `121` algebraic shells (`1` origin + `16` $\mathbb{Z}^4$ axes + `72` $D_4$ roots + `32` $D_4^*$ dual vertices) plus `135` data-adapted vectors refined via weighted Lloyd-Max while hard-pinning entry `0` to `(0.0, 0.0, 0.0, 0.0)`, guaranteeing **exact `0.0` reconstruction error** on quiet sky/vacuum blocks without spectral Gibbs ringing.
3. **`2-Pass INT4 Residual Scar Healing` + Sparse Point-Shock Capture ($\mathcal{O}(1)$ Block Read):** Absorbs correlated coherent structures via low-rank `INT4` factors plus top-$K$ point shocks, supporting random single-block reads in `130.2–133.8 µs` (`24.2×–36.8×` faster than streaming `zlib -9`, with `0.0` full-vs-block discrepancy).
4. **Blind Covariance `OrthogonalSubspaceProjector` ($P_\perp = \mathbf{I} - \mathbf{U}\mathbf{U}^\top$):** Cancels narrowband/spatial interference calibrated solely from noisy baseline runs (`0` clean-signal oracle).
5. **Constant-Memory `SurpriseGatedAnomalySCache` (`34,816 B` = `34.8 KB`, $\Delta = 0\text{ B}$):** Fixed-footprint Widrow-Hoff associative black-box recorder (`> 40,000 frames/s`) that ignores stationary background and retains ultra-rare mission faults across arbitrarily long streams, with masked partial-cue hetero-associative sensor reconstruction.

![ScarCodec Empirical Evaluation on 5 Public Observational Archives](../zenodo_deposit_scarcodec_public_data_v1_0/scarcodec_public_data_v1_0_hero.png)

---

## 1. Audited Results on 5 Official Public Scientific Archives (`SHA-256` Verified, Zero Synthetic Data)

Evaluated with **unmodified [`core.py`](core.py)** (`synthetic_data_used = false`, `oracle_steering_used = false` in [`../results/scar_codec_public_datasets_v1_0_audit.json`](../results/scar_codec_public_datasets_v1_0_audit.json); full paper in [`ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf`](../zenodo_deposit_scarcodec_public_data_v1_0/ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf)):

| Public Archive (`SHA-256` Verified) | Physical Domain | `ScarCodec` vs. `zlib -9` | Key Empirical Result vs. Byte-Matched Baseline |
| :--- | :--- | :---: | :--- |
| **`PhysioNet EEGMMIDB`** (`S001R01 → S001R04`) | `64`-ch cortical EEG (`160 Hz` Ag/AgCl) | **`10.47×`** vs. `3.17×` | Blind cross-run $P_\perp$: **`+14.63 dB` SNR** (`cos = 0.98267`; **`+8.40 dB`** on physical `C3` Hjorth Laplacian vs. `-4.85 dB` `2-bit` ADC) + `32 → 32` unseen electrode reconstruction at **`cos = 0.99667`** in fixed `34,816 B` ($\Delta = 0\text{ B}$). |
| **`NASA Space Shuttle`** (`UCI Statlog ID 148`) | Radiator telemetry (`14,500 × 8` sensors) | Fixed **`3,120 B`** ($\Delta = 0\text{ B}$) | Retains **`6/6 = 100.0%`** of ultra-rare critical bypass-valve faults (`0.0414%` prevalence, `40,400 FPS`) vs. **`0/6 = 0.0%`** in an equal-byte (`3,120 B`) circular ring buffer. |
| **`FCRAO 14-m L1448 13CO`** (`FITS`) | Radio spectral line cube (`[371, 1575]`, HDR `11.7×`) | **`12.14×`** vs. `1.10×` | **`3.83×` lower RMSE** & **`4.6×` lower peak error** vs. byte-matched 2D `FP16 rFFT`; **`36.8×` faster** $\mathcal{O}(1)$ block read (`130.2 µs` vs. `4,789.9 µs` `zlib`). |
| **`CSO Bolocam Sgr A* 1.1 mm`** (`FITS`) | Submillimeter Galactic Center (`[638, 640]`, HDR `217×`) | **`12.27×`** vs. `1.12×` | **`45.9×` lower bright-core error**, **`6.06×` lower `NaN`-edge ringing**, **`0.0` exact vacuum error** (vs. `0.146` `rFFT` ringing) & **`24.2×` faster** $\mathcal{O}(1)$ read. |
| **`NASA Spitzer IRAC`** (`FITS`) | Space infrared mosaic (`[513, 1025]`, HDR `968.9×`) | **`11.77×`** vs. `1.14×` | **`2,093.6×` lower top-32 stellar peak error** (`0.999` vs. `2,091.0 MJy/sr` Gibbs ringing in byte-matched `FP16 rFFT`), **`6.00×` lower global RMSE** & **`32.1×` faster** $\mathcal{O}(1)$ read. |

### Monotonic HDR Contrast Scaling Behavior & Comparison vs. Domain-Standard `ZFP` and `CFITSIO FITS fpack` (`v1.1` Audit)
Across the three public FITS archives, while lossless `zlib -9` stalls at `1.10×–1.14×` on floating-point sky noise, `ScarCodec`'s photometric advantage over global 2D `FP16 rFFT` and local `4×4` block-transform **`ZFP`** (`zfpy` fixed-rate byte-matched) scales monotonically with observational dynamic range ([`../results/scar_codec_public_datasets_v1_1_audit.json`](../results/scar_codec_public_datasets_v1_1_audit.json)):

| Public Archive (Byte-Matched Budget) | `ScarCodec` RMSE / Peak Err | `ZFP` (`zfpy`) RMSE / Peak Err | `FITS fpack` (`RICE_1`) RMSE / Peak Err | Key Structural Finding |
| :--- | :---: | :---: | :---: | :--- |
| **`NASA Spitzer IRAC`** (`~178 KB`, `11.8×`, HDR `968.9×`) | `5.135` / **`0.999`** | `20.801` / `1,634.25` | **`1.560`** / `2.743` | **`1,636.2×` lower stellar peak error** & **`4.05×` lower RMSE** than `ZFP`; **`2.75×` lower peak error** than `fpack` (while `fpack` leads diffuse sky RMSE). |
| **`CSO Bolocam Sgr A*`** (`~133 KB`, `12.3×`, HDR `217.0×`) | **`0.0365`** / **`0.0039`** | `0.0518` / `0.7973` | `0.0407` / `0.0597` | **`ScarCodec` wins BOTH global RMSE and bright-core photometry** against `ZFP` (`204.4×` peak, `0.0` vs `0.422` vacuum leak) and `fpack` (`15.3×` peak). |
| **`FCRAO 14-m L1448`** (`~193 KB`, `12.1×`, HDR `11.7×`) | `0.0652` / `0.1902` | `0.1864` / `0.8417` | **`0.0618`** / **`0.1148`** | **`2.86×` lower RMSE** & **`4.43×` lower peak error** than `ZFP`; ties `fpack` within `5%` RMSE on smooth low-HDR gas while preserving fixed-stride SIMT GPU access. |
| **`PhysioNet 64-Ch S001R04`** (`~488 KB`, `10.5×`) | **`+14.63 dB`** (`+8.40 dB` C3) | `+10.13 dB` (`+6.59 dB` C3) | N/A (`1D/2D` sensor stream) | **`+4.50 dB` higher array SNR** than byte-matched `ZFP` (`+7.63 dB` vs raw `ZFP`), capturing non-local cross-scalp spatial correlations invisible to `4×4` blocks. |

### `Min-Heap` Trigger Slots vs. `Widrow-Hoff W_scar` Fast-Weight Ablation
Within the fixed `34,816 B` ($\Delta = 0\text{ B}$) footprint of `SurpriseGatedAnomalySCache`, the two internal structures serve distinct, experimentally verified roles:
- **`48`-Slot Priority Trigger Table (`slot_ts_i64` / `slot_vals_f16`):** Retains exact discrete fault timestamps (`6/6 = 100%` on NASA Space Shuttle; identical `3/3` early bursts as a standalone `48`-slot `Min-Heap`).
- **`Widrow-Hoff` Associative Matrix `W_scar` (`[64, 64]`):** Uniquely reconstructs **simultaneous superposed multi-region anomalies** from partial sensor cues in a single branchless `6.0 µs` GEMV (`3.85×` faster than `23.1 µs` slot scanning), achieving **`cos = 0.8826`** on `32` blind odd cortical electrodes when two orthogonal real `PhysioNet` cortical modes (`|cos| = 0.0012`) co-activate — where discrete single-slot `Min-Heap` nearest-neighbor lookup hits the algebraic $1/\sqrt{2}$ ceiling (**`cos = 0.6917`**).

---

## 2. Controlled Physical Testbeds (`../results/scar_codec_physical_v4_0_audit.json`)

- **Radio Array Blind RFI Nulling (`-60 dB` SIR / `1,000×` jammer):** Blind SVD covariance projector + `VQ4-Scar` (`8.67×` compression) recovers weak dispersed pulses at **`cos = 0.9837`** vs. `cos = 0.0249` for `2-bit` VLBI quantization.
- **Partial-Cue Hetero-Associative Telemetry (`34,816 B`, $\Delta = 0\text{ B}$):** Reconstructs `4` unmeasured sensor groups from `2` observed cue channels at **`masked cos = 0.9999`**.
- **2D Nonlinear Conservative Burgers Shockwave PDE (`Rusanov` finite-volume solver):** Achieves **`188.2×` lower RMSE** (`[512, 512]` native) / **`194.5×` lower RMSE** (browser WebGPU testbed) than byte-matched 2D `FP16 rFFT` / `DCT-II`, with exact `0.0` error in undisturbed vacuum regions.

---

## 3. Quick Start & Python API

```python
import torch
from scar_codec import (
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    SurpriseGatedAnomalySCache,
)

# 1. Compress a 2D scientific field (vacuum + sharp HDR cores) with O(1) random block access
codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=64)
pkg = codec.encode(field_2d_fp32)                                # ~12x smaller than FP32
block_64 = codec.decode_block_o1(pkg, row_idx=128, block_idx=2)  # ~130 us O(1) random read!
recon_full = codec.decode_full(pkg)

# 2. Blind Interference Nulling (calibrated ONLY from noisy baseline samples — zero oracle)
proj = OrthogonalSubspaceProjector.fit_from_interference_calibration(noisy_baseline, subspace_rank=4)
clean_signal = proj.project_clean(heldout_recording)

# 3. Constant-Space O(1) Black-Box Telemetry Recorder (34,816 B fixed forever, Delta = 0 B)
#    `surprise_threshold` gates RMS channel z-score (~1.0 on nominal Gaussian noise, > 4.2 on physical shocks)
scache = SurpriseGatedAnomalySCache(channels=64, surprise_threshold=4.2)
for t, sensor_frame in enumerate(mission_stream):
    scache.ingest(sensor_frame, timestamp=t)                     # Physical RAM never grows by a single byte

# Reconstruct missing/failed sensors from observed sensors (or probe raw unit cue via `query_associative`):
reconstructed = scache.recall_from_partial_cue(partial_frame, observed_mask=known_channels_bool)
delta_shock = scache.query_associative(masked_deviation_cue)
```

### Command-Line & Full SHA-256 Public Dataset Reproduction

```bash
# Quick self-contained CLI verification:
mos-scar-codec --demo

# Download & SHA-256 verify all 5 public archives (PhysioNet, NASA Shuttle, Spitzer, Bolocam, FCRAO)
# and reproduce every number in `results/scar_codec_public_datasets_v1_0_audit.json`:
python scripts/benchmark_scar_codec_public_datasets_v1_0.py
pytest tests/test_scar_codec_physical.py tests/test_scar_codec_public_datasets.py -v
```

---

## 4. Honest Epistemic Scope & Boundary Conditions

- **Smooth Bandlimited vs. Shock/HDR Regimes:** On purely smooth, low-contrast sinusoidal fields without discontinuities, vacuum, or `NaN` mask boundaries, spectral transforms (`rFFT` / `DCT`) remain asymptotically optimal. `ScarCodec` is specifically engineered for **high-dynamic-range (`HDR`), shock-dominated, masked, or sparse-vacuum physical fields** where global Fourier bases suffer severe Gibbs ringing (`2,093.6×` peak degradation on Spitzer IRAC) and lossless `zlib` fails to compress floating-point thermal noise (`1.10×–1.14×`).
- **Rank Saturation Under 1-Sensor Scalar Cueing:** When querying `SurpriseGatedAnomalySCache` with only `1` scalar sensor out of $D$ after `> D` high-surprise writes (`rank = D` saturation), unmasked recall is algebraically bounded by $O(1/\sqrt{D})$ (`|cos| = 0.3237` on 8-sensor NASA Shuttle); multi-channel partial cueing (`32/64` channels on PhysioNet) achieves `cos = 0.99667`.

---

## 5. `ScarCodec v6.0` Industrial Consolidation (4 Production Pillars — [`../results/scar_codec_v6_0_consolidation_audit.json`](../results/scar_codec_v6_0_consolidation_audit.json))

1. **Zero-PyTorch Pure `NumPy` + Auto-Compiled Native C99/SIMD (`< 0.4 µs` $\mathcal{O}(1)$ Block Reads):**
   - [`numpy_engine.py`](numpy_engine.py) & [`pyproject.standalone.toml`](pyproject.standalone.toml) run with **`numpy` ONLY** (`0` `torch` dependency, verified in isolated subprocess with `sys.modules["torch"] = None`).
   - [`c_kernels.c`](c_kernels.c) auto-compiles `_scar_codec_c` (`-O3` SIMD), decoding random 64-float blocks in **`0.389 µs` (`389 ns`)** — **`344.2×` faster than PyTorch (`133.8 µs`)** and **`11,060.8×` faster than `zlib -9` (`4,299.0 µs`)**.
2. **Canonical `.scar` On-Disk Container (`CRC32` + Zero-Copy OS `mmap`) & Official `Zarr` / `numcodecs` Plugin:**
   - [`binary_format.py`](binary_format.py) defines the 64-byte SIMD-aligned `SCARv6\x00\x00` format with `CRC32` bit-rot verification and `ScarMmapFile` (`read_block_o1` directly off SSD via OS `mmap` without loading the file into RAM).
   - [`zarr_codec.py`](zarr_codec.py) provides `ScarZarrCodec` (`codec_id = "scar"`), drop-in compatible with `Zarr`, `numcodecs`, `xarray`, and `Dask`.
3. **Provable Deterministic $L_\infty$ Error-Bounded Mode (`abs_tolerance`) Across 29 Orders of Magnitude:**
   - Passing `abs_tolerance=eps` to `NumPyVQ4ScarCodec`, `PhysicalVQ4ScarCodec`, `ScarCodecND`, or `ScarCodecE8` guarantees $\max |X - \widehat{X}| \le \varepsilon$ via lossless IEEE-754 power-of-two `global_scale` ($2^E$) + 6-pass clamped Dekker `FP16 + FP16` residual splitting (forensically verified from **`10⁻²¹` LIGO gravitational-wave strain** to **`1.5×10⁸` Chandra X-ray photon counts** with `0.0` vacuum error and zero `FP16` overflow/underflow in [`../results/scar_codec_forensic_and_synergy_audit.json`](../results/scar_codec_forensic_and_synergy_audit.json)).
4. **Native 3D / $N$-D Spectral-Tucker Cubes (`ScarCodecND`), `1.000 bpw` Gosset $E_8$ Mode (`ScarCodecE8`) & `ScarField v7.0` Synergy:**
   - [`nd_e8.py`](nd_e8.py): **`ScarCodecE8`** (`240` minimal $E_8$ roots + `0x00` vacuum + `15` Hadamard outer anchors, `0.0` codebook drift vs `scar_field`) compresses `NASA Spitzer IRAC` by **`21.00×`** (`100,160 B`) with **`0.9556 MJy/sr` top-32 stellar peak error** (`1,710×` lower than `ZFP` at nearly `2×` higher compression).
   - **`ScarCodecND`** encodes 3D radio cubes natively (`[53, 105, 105]` `FCRAO L1448 13CO` at **`12.32×` compression** and **`0.0566 K` top-32 peak error** — `3.36×` lower than 2D unfold and **`2.03×` lower than `CFITSIO FITS fpack`**) with true $\mathcal{O}(1)$ 64-voxel spatial-strip reads (`decode_voxel_strip_o1`).
   - **Direct `ScarField v7.0` Synergy:** `c_kernels.c` provides `scar_e8_spherical_ternary_gemm_f32` (`0 B` weight-unpack heap) and `binary_format.py` provides the zero-pickle `.sfield.scar` `CRC32` `mmap` container (`1.88 MiB` vs `18.36 MiB` `.pt`), cutting `ScarField v7.0` live `E8` RAM by **`31.68×`** and speeding up full-engine CPU `S1` reflex by **`58.76×`** (`cos = 1.0`).

```python
import numpy as np
from scar_codec import NumPyVQ4ScarCodec, ScarCodecE8, ScarCodecND, ScarMmapFile, ScarZarrCodec, save_scar_file

# Zero-PyTorch C/SIMD + deterministic L_inf error bound + OS mmap O(1) read:
pkg = NumPyVQ4ScarCodec(rank=16, abs_tolerance=0.05).encode(field_2d_np)
save_scar_file(pkg, "archive.scar")
with ScarMmapFile("archive.scar") as mf:
    strip64 = mf.read_block_o1(row_idx=64, block_idx=2)  # Zero-copy SSD mmap -> C99/SIMD!

# 21.0x Gosset E8 (1.000 bit/sample) & 3D Spectral-Tucker Cubes:
pkg_e8 = ScarCodecE8(rank=8).encode(field_2d_np)
pkg_3d = ScarCodecND(spectral_rank=18, spatial_scar_rank=16).encode(cube_3d_np)
voxel_strip = ScarCodecND.decode_voxel_strip_o1(pkg_3d, v_chan=24, spatial_block_idx=10)
```
