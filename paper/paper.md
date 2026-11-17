---
title: 'ScarCodec: Zero-PyTorch Lattice Vector Quantization, Sub-Microsecond O(1) Block Reads, and Constant-Memory Telemetry Operators for Scientific Archives'
tags:
  - Python
  - C
  - scientific compression
  - FITS astronomy
  - Zarr
  - lattice quantization
  - telemetry
  - biomedical signal processing
authors:
  - name: Julio Cesar Reis Filho
    affiliation: 1
affiliations:
  - name: Byte-OD Research & System One Architecture Lab, Independent Researcher, Brazil
    index: 1
date: 16 November 2026
bibliography: paper.bib
---

# Summary

`ScarCodec` (`scar-codec` v6.0.0) is an open-source, neural-network-free Python and C99/SIMD library for floating-point array compression, $\mathcal{O}(1)$ random block access, multi-sensor interference nulling, and constant-memory anomaly recording across observational astronomy, spacecraft telemetry, and biomedical sensor arrays. Requiring only `numpy` at runtime, `ScarCodec` provides deterministic vector-lattice codecs (`NumPyVQ4ScarCodec`, `ScarCodecE8`, `ScarCodecND`), an orthogonal baseline interference projector (`OrthogonalSubspaceProjectorNumPy`), a fixed-byte associative anomaly flight recorder (`SurpriseGatedAnomalySCacheNumPy`), a `CRC32`-verified memory-mapped binary format (`.scar` via `ScarMmapFile`), and a drop-in `numcodecs` plugin (`ScarZarrCodec`) for `Zarr`, `Xarray`, and `Dask` scientific pipelines. Optional zero-copy `PyTorch` bindings remain fully supported when `torch` is installed.

# Statement of Need

Modern scientific instruments—including infrared space telescopes [@fazio2004irac], submillimeter bolometer arrays [@ginsburg2013bolocam], spectral-line radio cubes [@ridge2006complete], spacecraft engineering buses, and high-density electroencephalography (EEG) helmets [@schalk2004bci2000; @goldberger2000physionet]—produce IEEE-754 floating-point tensors [@pence2010fits; @astropy2022project; @kemp1992edf] where smooth correlated backgrounds and unmapped zero-valued sky masks coexist with localized, extreme high-dynamic-range (HDR) spikes. Existing storage and telemetry tools leave three practical engineering gaps:

1. **Lossless byte codecs versus random slice access:** General-purpose lossless compressors (`zlib`/`DEFLATE`) achieve only $1.10\times$–$1.14\times$ compression on noisy astronomical float mantissas and require multi-millisecond full-stream inflation to inspect a local detector slice.
2. **HDR point-source ringing and variable-length tiles:** Global spectral truncation introduces Gibbs oscillations around bright sources, while fixed-rate $4^d$ block transform codecs such as `ZFP` [@lindstrom2014zfp] share bitplane budgets across smooth pixels and extreme point shocks inside a block, degrading peak photometry on high-contrast fields. Conversely, astronomical tile-Rice codecs (`CFITSIO fpack`, `RICE_1`) [@pence2009fpack] excel on smooth backgrounds via subtractive dithering, but emit variable-length compressed tiles rather than fixed-stride records addressable in $\mathcal{O}(1)$ pointer arithmetic by CPU `mmap` or SIMT GPU kernels.
3. **Heavy ML runtimes and $\mathcal{O}(T)$ circular telemetry:** Learned neural codecs require multi-gigabyte deep-learning frameworks unsuitable for lightweight observatory pipelines or embedded flight recorders, while fixed-size circular ring buffers overwrite early low-prevalence hardware faults during subsequent nominal operation.

`ScarCodec` addresses these needs in a dependency-light (`pip install scar-codec`, `numpy` + portable C99) package designed for researchers and data engineers who need $10\times$–$21\times$ compression, exact `0.0` vacuum preservation, sub-microsecond random $64$-sample reads, optional strict $L_\infty$ error bounds, and $\mathcal{O}(1)$ anomaly recall.

# Software Architecture and Mathematical Operators

`ScarCodec` adapts training-free lattice and fast-weight mechanics originally developed in `Mixture of Scars` [@reisfilho2026mos_zenodo] into five standalone physical operators:

- **Shock-Peeled Lattice Block Quantization (`NumPyVQ4ScarCodec`, `ScarCodecE8`, `ScarCodecND`):** Outliers exceeding $4.2\sigma_{\text{MAD}}$ are peeled into a sorted sparse coordinate table (`int32` index + `FP16` amplitude) searchable via binary search in $\mathcal{O}(\log M_{\text{spikes}})$. The smooth base field undergoes channel equilibration and an orthonormal $64$-point Fast Walsh–Hadamard Transform (`FWHT-64`) [@walsh1923closed] with deterministic Rademacher sign flips, followed by either a $2.0$-bit-per-weight $D_4/D_4^*$-seeded `LUT256` vector quantizer [@conway1982fast] or an optimal 8D Gosset $E_8$ sphere-packing quantizer (`240` roots + `16` $\pm\sqrt{2}e_i$ poles = `256` states = `1.000 bpw`) [@viazovska2017sphere], plus two-pass `INT4` low-rank residual scars. Codeword `0` and zero-variance block/row scales are algebraically pinned to `0.0`, guaranteeing exact zero reconstruction on unmapped sky masks. Native 3D cubes (`ScarCodecND`) add spectral-axis Tucker compression prior to spatial block coding.
- **Compiled C99/SIMD `mmap` Container and `ScarZarrCodec`:** `c_kernels.c` compiles automatically with `-O3` SIMD vectorization (ARM NEON / x86 AVX2) and falls back bit-identically to pure `NumPy` when no C compiler is available. `.scar` files include a $64$-byte header with header-inclusive `CRC32` verification and support zero-copy OS `mmap` random block reads (`0.388 µs` per $64$-float slice; zero tail-padding leakage on non-multiple-of-64 widths). `ScarZarrCodec` implements the `numcodecs.abc.Codec` interface (`1D`–`4D`, `float16`/`float32`/`float64`, `vq4` or `e8`).
- **29-Decade Deterministic $L_\infty$ Error Bounds:** When `abs_tolerance` ($\varepsilon > 0$) is specified, `ScarCodec` factors out a lossless IEEE-754 power-of-two exponent $2^E$ ($E=0$ on $[10^{-2}, 4096]$) to prevent `FP16` subnormal underflow or overflow without mantissa rounding, then applies a 6-pass clamped Dekker `FP16` residual split [@dekker1971floating] guaranteeing $\|X - \widehat{X}\|_\infty \le \varepsilon$ from $10^{-21}$ gravitational-wave strain to $1.5\times 10^8$ X-ray photon counts.
- **Subspace Interference Nulling & Constant-Memory $\mathcal{O}(1)$ Telemetry:** `OrthogonalSubspaceProjectorNumPy` fits $P_\perp = I_C - U_{\text{cal}}U_{\text{cal}}^\top$ [@leshem2000multichannel] on baseline interference recordings prior to compression. `SurpriseGatedAnomalySCacheNumPy` maintains an exact constant byte footprint (`34,816 B`, $\Delta = 0\text{ B}$ for $C=64$), combining discrete high-surprise event slots with a Widrow–Hoff associative matrix $W_{\text{scar}} \in \mathbb{R}^{C \times C}$ [@widrow1960adaptive] supporting partial-sensor masked recall (`recall_from_partial_cue`).

![Empirical and systems evaluation of ScarCodec v6.0 across five SHA-256-verified public scientific archives: (A) cross-run 64-channel PhysioNet interference cancellation vs. byte-matched ZFP; (B) constant-space O(1) anomaly retention and associative recall vs. circular ring and Min-Heap; (C) archive compression ratios including 21.0x Gosset E8; (D) NASA Spitzer protostar profile; (E) HDR peak photometry scaling; (F) 0.388 µs C99/SIMD mmap O(1) random block read latency.](fig1_scarcodec_v6_hero.png)

# Validation on Public Scientific Archives and Trade-Offs

A dedicated empirical companion study evaluates unmodified `ScarCodec v6.0` across seven SHA-256-verified files from five public archives (Figure 1):

- **64-Channel Biomedical Array (`PhysioNet EEGMMIDB`):** Calibrating $P_\perp$ ($r=4$) strictly on eyes-open baseline `S001R01.edf` and compressing held-out motor imagery `S001R04.edf` at $10.47\times$ yields $+14.63\text{ dB}$ cortical SNR ($+4.50\text{ dB}$ over byte-matched `ZFP` + $P_\perp$) and $+8.40\text{ dB}$ on the model-free Hjorth surface Laplacian [@hjorth1975onlinesource] over motor cortex `C3`.
- **Constant-Space Telemetry (`PhysioNet` & `NASA Space Shuttle`):** Streaming $39{,}520 \times 64$ EEG frames in exact `34,816 B` ($\Delta=0\text{ B}$) reconstructs $32$ masked odd electrodes from $32$ even electrodes at $\cos = 0.99667$ and outperforms a $48$-slot `Min-Heap` on superposed orthogonal cortical modes ($\cos = 0.8826$ vs. $0.6917$ in $5.23\text{ }\mu\text{s}$). On `NASA Space Shuttle` radiator telemetry ($14{,}500 \times 8$), default `SCache` (`3,120 B`, $\Delta=0\text{ B}$) retains `6/6` (`100%`) of $0.0414\%$-prevalence Bypass Valve faults (`0/6` in an equal-byte circular ring), while transparently exhibiting rank-$8$ associative saturation ($|\cos| = 0.3237$) under $99$ secondary excursions.
- **Astronomical FITS (`NASA Spitzer`, `CSO Bolocam`, `FCRAO 14-m`):** At $11.8\times$–$12.3\times$ (`VQ4`) and $21.0\times$ (`Gosset E8`), `ScarCodec` reduces top-32 peak photometry error by $4.4\times$–$1{,}710\times$ relative to fixed-rate `ZFP` and by $2.19\times$–$16.51\times$ relative to `CFITSIO fpack` (while `fpack` retains lower RMSE on smooth diffuse backgrounds), and accelerates random $64$-float block reads to `0.388 µs` (`11,088x` faster than `zlib -9`).

# Availability and Reproducibility

`ScarCodec` is released under the MIT License at [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars) (`scar_codec/`), with automated `pytest` suites running under GitHub Actions across Linux and macOS in both Zero-PyTorch (`NumPy + C99/SIMD`) and `PyTorch` environments, alongside deterministic SHA-256 public-archive benchmark scripts.

# References
