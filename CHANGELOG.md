# Changelog

All notable changes to `ScarCodec`, `ScarCache`, `ScarNativeLM`, and `ScarField` are documented in this file.

## [6.3.0] — 2026-11-17 (`ScarCodec v6.3-Biorthogonal` & `ScarField v8.0`)

### Added
- **`ScarCodec v6.3.0` (`scar_codec/`):**
  - Multithreaded C99 Conway–Sloane $\mathcal{O}(8)$ Gosset $E_8$ parity-coset encoder (`scar_conway_sloane_e8_encode_f32`), `FWHT-64` Rademacher batch kernel, and thread-local `float64` `VQ4` accumulator in `scar_codec/c_kernels.c` (`44.4–72.5 MB/s` single-CPU encode; `0.3765 µs` `mmap` random 64-float slice read).
  - Active-only `System 1 Gate G0` block compaction across C99, NumPy, PyTorch, and TypeScript/WebGPU surfaces with `0.0` RMSE drift and exact `0.0` vacuum/land mask preservation.
  - Zero-byte-overhead (`Δ = +0 B` `.scar`) biorthogonal smooth–shock–lattice decoupling (`biorthogonal=True`), Behrens–Scharf oblique interference projector (`BiorthogonalObliqueProjectorNumPy`), and `34,816 B` active-slot Dual Gram `SurpriseGatedAnomalySCacheNumPy`.
  - Standalone Zero-PyTorch wheel/sdist builder (`scripts/build_standalone_scar_codec_pypi.py`) verified in read-only (`0o555`) `site-packages`, 37-check adversarial IEEE-754/CRC32/concurrency forensic suite (`tests/test_physics_lab_production_forensic.py`), and Preprint `v3.0` SHA-256 replication kit (`zenodo_deposit_scarcodec_v3_0/`).
- **`ScarField v7.5-Fused` & `v8.0 Macro-Scout Geodesic Fleet` (`scar_field/`):**
  - Native `@triton.jit` GPU kernels (`triton_kernels.py`), zero-autograd analytical symplectic midpoint adjoints, bounded block-`2×2` recursive `Chunkwise WY` triangular solver, and 1-step causal shock-closure FIR (`87.5%` unseen 3-turn `O(1)` PIN recall with `0/24` stutters).
  - 3-tier `MacroScoutGeodesicFleet` + `SharedCrystalMacroSwarm` (`macro_scout_fleet.py`) with multi-stream CUDA concurrency, `Log`/`Exp` biorthogonal conflict-free tangent merge, and read-only shared `1.000 bpw` Gosset $E_8$ crystal distillation.

## [6.0.0] — 2026-11-16 (`ScarNativeLM v6.0`, `PointScarFlow v1.0` & `ScarCodec` 10-Archive Public Data Suite)
- Added `Chunkwise WY` parallel Widrow–Hoff scan, `SharedStageUniversalBase` + `ScarMoSExpertBank`, `HierarchicalSCacheAttentionV6` (`26` WGSL 1.0 compute shaders), `mos-digest` `.scarstore` CLI, and `PointScarFlow` 3D generator (`12` WGSL shaders).
- Added `ScarCodec` standalone `NumPy` + C99/SIMD engine, `.scar` `CRC32` `mmap` container, `ScarZarrCodec` (`numcodecs`), and 10-archive SHA-256 physical science benchmark (`LIGO`, `NASA Spitzer`, `CSO Bolocam`, `ROSAT`, `FCRAO`, `IRIS Tohoku`, `NOAA OISST`, `NASA Space Shuttle`, `MIT-BIH`, `PhysioNet`).

## [5.0.0] — 2026-11-15 (`ScarNativeLM v5.0` Tabula-Rasa Foundation)
- Introduced pure scar-native training from `step = 0` (`ScarNativeLinear` `D4-LUT256 VQ4-STE`, `DifferentiableSCacheAttention`, `LanguageNullspaceProjector`, and `DetachedScarStore` with cryptographic `0x00` GDPR erasure) and WGSL Shaders `#21`–`#22`.

## [4.0.0] — 2026-11-14 (`VQ4-Scar` 2-Pass Residual Healing & 20-Shader WebGPU Engine)
- Replaced Gaussian scalar quantization across native and WebGPU engines with 4D `D4-LUT256` vector micro-scars, `FWHT-64` Rademacher rotation, and 2-pass `INT4` residual scar healing; added `SurpriseGatedAnomalySCache` (`34,816 B`).

## [3.0.0] — 2026-11-12 (Online `.scarstore` HTTP 206 Library, Sinkhorn Re-Basin & Sub-2-Bit MoS)
- Added `mos-store` HTTP 206 Range-streaming cartridge library, `mos-rebasin` shared-grid SwiGLU compression, `mos-graft` zero-data SVD scar grafting, and 20-shader WebGPU browser inference (`Qwen2.5-3B`).

## [1.2.0] — 2026-11-09 (`Mixture of Scars` 72B Streaming & `SCache` $\mathcal{O}(1)$ Memory)
- Added sequential INT4 + rank-64 SVD scar streaming (`Qwen2.5-72B` under `< 2.2 GB` active group buffer) and `SCacheDynamicCache` constant-memory associative replacement for linear $\mathcal{O}(T)$ KV caches (`v0.4.0`–`v1.2.0`).
