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

Scientific instruments—from space telescopes and gravitational-wave detectors to seismic networks, ocean climate satellites, spacecraft sensors, and clinical heart or brain monitors—continuously record large streams of floating-point numbers. Researchers working with these archives face a recurring practical dilemma: standard compression tools either shrink smooth data well while making it slow to read small slices at random, or allow fast random access while blurring sharp physical events such as bright stars, earthquakes, or cardiac arrhythmias.

`ScarCodec` (`scar-codec` v6.3.0) is an open-source, neural-network-free Python and multithreaded C99 library designed to resolve this tension. Requiring only `NumPy` at runtime, `ScarCodec` separates smooth background structure, sharp localized events, and structured lattice residuals so that archives compress compactly while any individual $64$-number block can still be read directly from disk in less than half a microsecond via operating-system memory mapping (`mmap`). Alongside array compression, `ScarCodec` provides mathematical filters that cancel overlapping radio or sensor interference without weakening the target signal, a fixed-size ($34.8\text{ KB}$) streaming anomaly memory for long telemetry runs, a `CRC32`-verified `.scar` container, and a drop-in `numcodecs` plugin (`ScarZarrCodec`) for `Zarr`, `Xarray`, and `Dask`.

# Statement of need

Observational and experimental workflows across infrared and submillimeter astronomy [@fazio2004irac; @ginsburg2013bolocam], X-ray and radio spectral surveys [@snowden1997rosat; @ridge2006complete], laser gravitational-wave interferometry [@abbott2016observation], broadband seismology [@asl1988gsn], global sea-surface temperature grids [@reynolds2002improved], spacecraft bus monitoring [@michie1994statlog], and high-density electroencephalography (EEG) or Holter electrocardiography (ECG) [@schalk2004bci2000; @moody2001impact; @goldberger2000physionet] store IEEE-754 floating-point arrays [@pence2010fits; @astropy2022project; @kemp1992edf] spanning 29 orders of magnitude ($10^{-21}$ strain to $10^8$ photon counts).

Target users include observational astronomers, geophysicists, biomedical signal researchers, and data engineers building interactive visualization portals, out-of-core training loaders, or edge telemetry recorders. These workflows share three coupled requirements that existing single-paradigm tools struggle to satisfy simultaneously: (1) sub-microsecond $\mathcal{O}(1)$ random slice access without full-chunk decompression; (2) exact preservation of high-contrast point sources, strict clinical error bounds, and zero-valued sky or continent masks; and (3) constant-memory $\mathcal{O}(1)$ anomaly capture and oblique interference rejection on correlated multichannel arrays.

# State of the field

General-purpose lossless codecs (`zlib`, `Blosc2`) stall near $1.1\times$–$1.3\times$ on noisy floating-point mantissas. Error-bounded scientific compressors such as `SZ3` [@liang2022sz3] (Lorenzo spline prediction plus entropy coding), `SPERR` [@li2023sperr] (Cohen–Daubechies–Feauveau $9/7$ biorthogonal wavelets [@cohen1992biorthogonal] plus SPECK), and astronomical tile-Rice `CFITSIO fpack` [@pence2009fpack] achieve strong compression on smooth fields, but emit variable-length bitstreams requiring $0.4$–$21.8\text{ ms}$ chunk inflation for random $64$-sample reads and leaking wavelet ringing into zero-valued vacuum or land masks. Conversely, fixed-rate $4^d$ block transform coders such as `ZFP` [@lindstrom2014zfp] support random access but smear high-contrast stellar peaks across shared block bitplanes. In multichannel interference rejection and streaming telemetry, orthogonal subspace projection $P_\perp$ [@leshem2000multichannel] attenuates desired signals whenever jammer and target beams overlap obliquely, generalized spectral kurtosis [@nita2010generalized] degrades under stationary Gaussian broadband jammers, and streaming SVD sketches such as `Frequent Directions` [@ghashami2016frequent] discard discrete fault timestamps while symmetric Widrow–Hoff fast weights [@widrow1960adaptive] suffer crosstalk under correlated modes.

**Build versus contribute justification:** `ScarCodec` could not be implemented as a patch or plugin inside `SZ3`, `SPERR`, or `ZFP` because its core guarantees—fixed-stride $8\text{D}$ Gosset $E_8$ [@viazovska2017sphere] and $4\text{D}$ $D_4$ [@conway1982fast] lattice addressing with dedicated codeword `0` for exact physical vacuum, coupled smooth–shock–lattice biorthogonal projections, and fixed-$34{,}816\text{-byte}$ associative telemetry—require a fundamentally different binary layout from variable-length entropy streams or $4^d$ orthogonal block transforms. Instead of duplicating array ecosystem infrastructure, `ScarCodec` integrates directly into the scientific Python stack by implementing the standard `numcodecs.abc.Codec` interface (`ScarZarrCodec`) for seamless use inside `Zarr`, `Xarray`, and `Dask`, while remaining dependency-light (`numpy` only).

# Software design

`ScarCodec` translates lattice quantization and associative fast-weight primitives originally explored in `Mixture of Scars` [@reisfilho2026mos_zenodo] into five modular, neural-network-free operators built around explicit engineering trade-offs:

- **Fixed-stride lattice blocks + sparse shock table vs. global entropy coding:** We traded away global Huffman/Zstd bitstream coupling on the base grid in exchange for deterministic $\mathcal{O}(1)$ byte offsets. Outliers exceeding $4.2\sigma_{\text{MAD}}$ are separated into a binary-searchable sparse shock table, while the base array undergoes channel equilibration and orthonormal $64$-point Fast Walsh–Hadamard rotation (`FWHT-64`) [@walsh1923closed] followed by $2.0\text{-bpw}$ $D_4/D_4^*$ `LUT256` or $1.000\text{-bpw}$ $8\text{D}$ Gosset $E_8$ quantization (`NumPyVQ4ScarCodec`, `ScarCodecE8`, `ScarCodecND`) plus `INT4` low-rank residuals. Codeword `0` maps strictly to the zero vector, guaranteeing zero mask bleed.
- **Zero-byte-overhead biorthogonal decoupling (`biorthogonal=True`):** Greedy sequential peeling leaks sharp transients into smooth low-rank factors. Alternating coupled projections with exact dual least-squares block scaling $\alpha_b^* = \langle r_b, q_b\rangle / \|q_b\|_2^2$ improves reconstruction fidelity without adding a single header byte ($\Delta = +0\text{ B}$) to `.scar` files.
- **Zero-PyTorch C99/SIMD engine with bit-identical `NumPy` fallback:** `c_kernels.c` implements multithreaded Conway–Sloane $\mathcal{O}(8)$ $D_8$ parity-coset decoding, `Gate G0` active-block compaction ($44.4$–$72.5\text{ MB/s}$ single-CPU encode), and `-O3` ARM NEON / x86 AVX2 `mmap` reads (`0.376 µs` per $64$-float block, `CRC32` verified, `8`-thread safe), while falling back transparently to pure `NumPy` on restricted platforms.
- **29-decade deterministic $L_\infty$ bounds:** Lossless IEEE-754 power-of-two scaling $2^E$ combined with 6-pass Dekker `FP16` residual splitting [@dekker1971floating] enforces $\|X - \widehat{X}\|_\infty \le \varepsilon$ from $10^{-21}$ strain to $1.5\times 10^8$ counts ($0$-ULP exact across $200\text{ dB}$ intra-block contrast).
- **Oblique interference nulling & $34{,}816\text{-byte}$ Dual `SCache`:** `BiorthogonalObliqueProjectorNumPy` implements Behrens–Scharf oblique projection $P_{\text{biorth}} = I_C - U_J \widetilde{U}_J^\top$ ($\widetilde{U}_J^\top U_S = 0$) [@behrens1994signal] to cancel $-60\text{ dB}$ jammers with $100\%$ target amplitude retention, while `SurpriseGatedAnomalySCacheNumPy` resolves the active-slot dual Gram system $(K_{\mathcal{O}}K_{\mathcal{O}}^\top + \lambda I)^{-1}$ in strictly constant $34{,}816\text{ B}$ ($\Delta = 0\text{ B}$).

![Forensic and real-data evaluation of ScarCodec v6.3 across ten SHA-256-verified public scientific archives and 2026 SOTA baselines: (A) biorthogonal smooth-shock-lattice and oblique P_biorth gains; (B) constant-space 34.8 KB Dual SCache vs. Frequent Directions SVD and Orthogonal SCache; (C) single-CPU C99 Conway-Sloane O(8) encode throughput and compression ratios; (D) NASA Spitzer protostar cut and Bolocam/ROSAT peak photometry vs. SZ3, SPERR, and ZFP; (E) rate-access Pareto frontier; (F) 0.376 µs C99/SIMD mmap O(1) read latency and 37/37 forensic audit.](fig1_scarcodec_v6_hero.png)

# Research impact statement

`ScarCodec` already underpins two active research workflows and comes with reproducible SHA-256 empirical benchmarks across $16$ files from $10$ public scientific archives spanning $8$ physical sciences (Figure 1):

1. **Multi-disciplinary archive benchmarks and candid Pareto mapping:** Enabling `biorthogonal=True` ($\Delta = +0\text{ B}$) yields $+8.56\text{ dB}$ SNR gain on the $M_w\text{ }9.1$ `IRIS Tohoku` seismogram, $-27.06\%$ byte footprint on `MIT-BIH 119` Holter ECG under clinical IEC 60601-2-47 $\pm 50\text{ }\mu\text{V}$ ($0$ violations across $1.30\text{M}$ samples), and $0.8821\text{ }^\circ\text{C}$ ocean RMSE at $23.60\times$ on 3D `NOAA OISST v2` with exact $0.0000\text{ }^\circ\text{C}$ Antarctica land reconstruction (vs. $0.079\text{ }^\circ\text{C}$ `SZ3` and $0.159\text{ }^\circ\text{C}$ `SPERR` bleed). Byte-matched bright-source peak error is $7.88\times$–$10.81\times$ lower than `SZ3`/`SPERR` on `CSO Bolocam` (`0.0031 Jy`) and $2.19\times$–$3.16\times$ lower at $17.1\times$ Gosset $E_8$ on `ROSAT`. On smooth uniform-$L_\infty$ fields, `SZ3`/`SPERR` compress tighter ($10\times$–$18\times$ vs. $2.1\times$–$2.6\times$) at the cost of $888\times$–$25{,}991\times$ slower random $64$-sample reads. On $64$-channel `PhysioNet` EEG [@hjorth1975onlinesource] and `NASA Space Shuttle` telemetry, $34{,}816\text{ B}$ Dual `SCache` reaches $\cos=0.9822$ (vs. $0.9570$ `Frequent Directions`) and $\cos=0.99994$ ($159\times$ lower $L_2$ error; $100\%$ rare fault capture).
2. **Ecosystem and cross-stack adoption:** Beyond standalone `Zarr`/`FITS`/`EDF` scientific pipelines, `ScarCodec`'s `CRC32` `mmap` container and C99 Conway–Sloane $E_8$ kernels serve as the binary storage and zero-copy cold-load substrate (`.sfield.scar`) for the `ScarField` geodesic dynamical intelligence framework and interactive browser WebGPU scientific laboratories. All $48$ automated `pytest` suites, standalone Zero-PyTorch wheel builds, and SHA-256 replication scripts run via multi-OS GitHub Actions CI (`ubuntu-latest` and `macos-latest`).

# AI usage disclosure

Generative AI tools (Google Antigravity agentic development environment powered by Gemini models) were used during software development and manuscript preparation to assist with C99/Python implementation refactoring, unit and adversarial `pytest` scaffolding, benchmark automation scripts, documentation formatting, and English copy-editing of the manuscript. All core problem framing, mathematical formulations (biorthogonal smooth–shock–lattice decoupling, Conway–Sloane Gosset $E_8$ and $D_4$ `LUT256` layout, 29-decade Dekker $L_\infty$ residual cascades, and dual Gram `SCache` readout), architectural trade-offs, and experimental designs were conceived and directed by the human author. Every line of source code, C99 kernel, unit test, SHA-256 dataset checksum, numerical benchmark output, and manuscript claim was personally reviewed, executed, edited, and forensically verified by the author.

# Acknowledgements

This research was conducted independently by the author at Byte-OD Research & System One Architecture Lab without external grant funding; no external sponsor had any involvement in study design, software implementation, or manuscript preparation. The author declares no financial, personal, or professional conflicts of interest. Public scientific datasets were provided by NASA/IPAC (`Spitzer`), Caltech/CSO (`Bolocam`), MPE/NASA (`ROSAT`), FCRAO (`COMPLETE`), LIGO/GWOSC, IRIS/USGS, NOAA PSL, UCI (`Statlog`), and MIT-LCP (`PhysioNet`).

# References
