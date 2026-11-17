---
title: 'ScarCodec: Zero-PyTorch Biorthogonal Lattice Compression, Sub-Microsecond O(1) Block Reads, and Constant-Memory Dual Telemetry for Scientific Archives'
tags:
  - Python
  - C
  - scientific compression
  - FITS astronomy
  - Zarr
  - lattice quantization
  - biorthogonal projection
  - telemetry
  - biomedical signal processing
authors:
  - name: Julio Cesar Reis Filho
    affiliation: 1
affiliations:
  - name: Byte-OD Research & System One Architecture Lab, Independent Researcher, Brazil
    index: 1
date: 17 November 2026
bibliography: paper.bib
---

# Summary

`ScarCodec` (`scar-codec` v6.0 / v6.3-Biorthogonal) is an open-source, neural-network-free Python and multithreaded C99/SIMD library for IEEE-754 floating-point compression, sub-microsecond $\mathcal{O}(1)$ random block access, oblique interference nulling, and constant-memory anomaly recording across observational astronomy, gravitational-wave and seismic observatories, climate oceanography, spacecraft telemetry, and clinical biopotential arrays. Requiring only `numpy` at runtime, `ScarCodec` provides biorthogonal smooth–shock–lattice codecs (`NumPyVQ4ScarCodec`, `ScarCodecE8`, `ScarCodecND`), orthogonal and Behrens–Scharf oblique interference projectors (`OrthogonalSubspaceProjectorNumPy`, `BiorthogonalObliqueProjectorNumPy`), a fixed-byte associative anomaly recorder with dual Gram readout (`SurpriseGatedAnomalySCacheNumPy`), a `CRC32`-verified memory-mapped binary format (`.scar` via `ScarMmapFile`), and a drop-in `numcodecs` plugin (`ScarZarrCodec`) for `Zarr`, `Xarray`, and `Dask`. Optional zero-copy `PyTorch` and browser TypeScript/WebGPU engines preserve bit-exact codebook parity.

# Statement of Need

Modern scientific instruments—including space infrared cameras [@fazio2004irac], submillimeter bolometer surveys [@ginsburg2013bolocam], soft X-ray sky maps [@snowden1997rosat], spectral-line radio cubes [@ridge2006complete], laser gravitational-wave interferometers [@abbott2016observation], broadband seismographs [@asl1988gsn], global sea-surface temperature grids [@reynolds2002improved], spacecraft buses [@michie1994statlog], and high-density electroencephalography (EEG) or Holter electrocardiography (ECG) monitors [@schalk2004bci2000; @moody2001impact; @goldberger2000physionet]—emit floating-point tensors [@pence2010fits; @astropy2022project; @kemp1992edf] spanning 29 orders of magnitude ($10^{-21}$ strain to $10^8$ photon counts). Three engineering tensions motivate `ScarCodec`:

1. **Variable-length smooth-field codecs versus $\mathcal{O}(1)$ random slice reads and zero masks:** Lossless compressors (`zlib`, `Blosc2`) stall near $1.1\times$–$1.3\times$ on noisy float mantissas. Modern error-bounded compressors such as `SZ3` [@liang2022sz3] (Lorenzo prediction + entropy coding), `SPERR` [@li2023sperr] (Cohen–Daubechies–Feauveau $9/7$ biorthogonal wavelets [@cohen1992biorthogonal] + SPECK), and astronomical tile-Rice `CFITSIO fpack` [@pence2009fpack] achieve high compression on smooth fields, yet emit variable-length bitstreams requiring $0.4$–$21.8\text{ ms}$ chunk inflation for random $64$-sample seeks and leaking wavelet energy into zero-valued vacuum or continent masks. Fixed-rate $4^d$ block coders (`ZFP`) [@lindstrom2014zfp] smear high-contrast stellar peaks across bitplanes.
2. **Non-orthogonal smooth–shock–lattice and oblique jammer crosstalk:** Greedy sequential peeling leaks localized shocks into low-rank factors, while orthogonal subspace projection $P_\perp$ [@leshem2000multichannel] attenuates desired signals when interference and target beams overlap obliquely ($\cos\theta \neq 0$), where moment-based radio-frequency interference (RFI) detectors such as generalized spectral kurtosis [@nita2010generalized] also fail on Gaussian broadband jammers.
3. **Constant-space $\mathcal{O}(1)$ telemetry without mode crosstalk:** Circular ring buffers erase early low-prevalence hardware faults, streaming SVD sketches such as `Frequent Directions` [@ghashami2016frequent] discard discrete fault timestamps and require $\mathcal{O}(\ell C)$ linear solves, and symmetric Widrow–Hoff fast weights [@widrow1960adaptive] suffer interference when correlated anomalies co-activate.

# Software Architecture and Mathematical Operators

`ScarCodec` adapts training-free lattice and fast-weight mechanics from `Mixture of Scars` [@reisfilho2026mos_zenodo] into five standalone operators:

- **Biorthogonal Smooth–Shock–Lattice Quantization (`biorthogonal=True`):** Outliers above $4.2\sigma_{\text{MAD}}$ are stored in a binary-searchable sparse table, while the base field undergoes channel equilibration and orthonormal $64$-point Fast Walsh–Hadamard rotation (`FWHT-64`) [@walsh1923closed] followed by $2.0\text{-bpw}$ $D_4/D_4^*$ `LUT256` quantization [@conway1982fast] or $1.000\text{-bpw}$ 8D Gosset $E_8$ sphere-packing quantization [@viazovska2017sphere] plus `INT4` low-rank scars. Enabling `biorthogonal=True` alternates coupled projections with exact dual least-squares block scaling $\alpha_b^* = \langle r_b, q_b\rangle / \|q_b\|_2^2$ at **zero extra `.scar` header bytes** ($\Delta = +0\text{ B}$). In `v6.3`, multithreaded C99 Conway–Sloane $\mathcal{O}(8)$ $D_8$ coset decoding and active-only `Gate G0` block compaction encode real archives at $44.4$–$72.5\text{ MB/s}$ on a single CPU, while codeword `0` locks unmapped sky and land masks to exact `0.0`.
- **Sub-Microsecond C99/SIMD `mmap` Container and `ScarZarrCodec`:** `c_kernels.c` compiles with `-O3` SIMD vectorization (ARM NEON / x86 AVX2) and falls back bit-identically to `NumPy`. `.scar` containers verify integrity via header-inclusive `CRC32` and read random $64$-float OS `mmap` slices in `0.376 µs` (`8`-thread safe, zero file-descriptor leaks).
- **29-Decade Deterministic $L_\infty$ Bounds:** Lossless IEEE-754 power-of-two pre-scaling $2^E$ plus 6-pass Dekker `FP16` residual splitting [@dekker1971floating] enforces $\|X - \widehat{X}\|_\infty \le \varepsilon$ from $10^{-21}$ strain to $1.5\times 10^8$ counts, with $0$-ULP lossless `float32` recovery across $200\text{ dB}$ intra-block contrast.
- **Oblique Interference Nulling & Dual $\mathcal{O}(1)$ Telemetry:** `BiorthogonalObliqueProjectorNumPy` implements $P_{\text{biorth}} = I_C - U_J \widetilde{U}_J^\top$ ($\widetilde{U}_J^\top U_S = 0$) [@behrens1994signal] to null $-60\text{ dB}$ oblique jammers while preserving $100\%$ target amplitude. `SurpriseGatedAnomalySCacheNumPy` (`34,816 B` exact, $\Delta = 0\text{ B}$) resolves the active-slot dual Gram system $(K_{\mathcal{O}}K_{\mathcal{O}}^\top + \lambda I)^{-1}$ alongside $W_{\text{scar}}$ for crosstalk-free partial-sensor recall.

![Forensic and real-data evaluation of ScarCodec v6.3 across ten SHA-256-verified public scientific archives and 2026 SOTA baselines: (A) biorthogonal smooth-shock-lattice and oblique P_biorth gains; (B) constant-space 34.8 KB Dual SCache vs. Frequent Directions SVD and Orthogonal SCache; (C) single-CPU C99 Conway-Sloane O(8) encode throughput and compression ratios; (D) NASA Spitzer protostar cut and Bolocam/ROSAT peak photometry vs. SZ3, SPERR, and ZFP; (E) rate-access Pareto frontier; (F) 0.376 µs C99/SIMD mmap O(1) read latency and 37/37 forensic audit.](fig1_scarcodec_v6_hero.png)

# Validation Across Ten Public Archives and Candid Trade-Offs

A companion empirical study evaluates `ScarCodec` across $16$ SHA-256-verified files from $10$ public archives (Figure 1):

- **Biorthogonal Gains & Bright-Source Photometry vs. `SZ3` / `SPERR` / `ZFP`:** Without changing `.scar` size, `biorthogonal=True` yields $+8.56\text{ dB}$ SNR gain on the $M_w\text{ }9.1$ `IRIS Tohoku` seismogram, $-27.06\%$ bytes on `MIT-BIH 119` Holter ECG under IEC $\pm 50\text{ }\mu\text{V}$ ($0$ violations), and $0.8821\text{ }^\circ\text{C}$ ocean RMSE at $23.60\times$ on 3D `NOAA OISST v2` with exact $0.0000\text{ }^\circ\text{C}$ Antarctica land reconstruction (vs. $0.079\text{ }^\circ\text{C}$ `SZ3` and $0.159\text{ }^\circ\text{C}$ `SPERR` bleed). Byte-matched bright-source peak error is $7.88\times$–$10.81\times$ lower than `SZ3`/`SPERR` on `CSO Bolocam` (`0.0031 Jy`, `0.0` vacuum) and $2.19\times$–$3.16\times$ lower at $17.1\times$ Gosset $E_8$ on `ROSAT` (`7.80` vs. `17.06` `SZ3` and `24.68` `SPERR`), while on smooth uniform-$L_\infty$ fields `SZ3` and `SPERR` achieve higher compression ratios ($10\times$–$18\times$ vs. $2.1\times$–$2.6\times$) at the cost of $888\times$–$25{,}991\times$ slower random $64$-sample reads (`0.376 µs` C99 `mmap`).
- **Biomedical Arrays & Constant-Memory Dual `SCache`:** Cross-run $P_\perp$ on $64$-channel `PhysioNet` yields $+14.63\text{ dB}$ SNR ($+8.40\text{ dB}$ `C3` Hjorth Laplacian [@hjorth1975onlinesource]), while `34,816 B` Dual `SCache` ($\Delta=0\text{ B}$) reaches $\cos=0.9822$ on oblique $32\to 32$ EEG superposition (vs. $0.9570$ `Frequent Directions` and $0.7433$ orthogonal), $\cos=0.9858$ on `MIT-BIH` fusion beats, and $\cos=0.99994$ ($159\times$ lower $L_2$ error; $100\%$ rare bypass-valve capture) on `NASA Space Shuttle`.

# Availability and Reproducibility

`ScarCodec` is released under the MIT License at [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars) (`scar_codec/`), with multi-OS GitHub Actions CI (`NumPy + C99/SIMD` and `PyTorch`), $42$ `pytest` suites, and deterministic SHA-256 benchmark scripts.

# References
