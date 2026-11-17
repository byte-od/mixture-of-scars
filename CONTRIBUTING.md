# Contributing to Mixture of Scars & ScarCodec

Thank you for your interest in contributing to `ScarCodec` (`scar_codec/`) and the broader `Mixture of Scars` (`scar_cache/`, `scar_native/`, `scar_field/`) open-source research ecosystem!

## Governance and Support Expectations

This repository is actively maintained by **Julio Cesar Reis Filho** (*Byte-OD Research & System One Architecture Lab*).
- **Bug reports & reproducible issues:** Reviewed via [GitHub Issues](https://github.com/byte-od/mixture-of-scars/issues). Please include your OS, Python/`numpy` version, compiler (`clang`/`gcc`), and a minimal script or SHA-256 input checksum reproducing the issue.
- **Support scope:** Support is provided on an open-source, best-effort basis through public GitHub Issues for installation questions, C99/SIMD fallback diagnostics, `ScarZarrCodec` integration, and `.scar` binary specification questions.
- **Pull requests:** External contributions (bug fixes, new scientific format adapters, SIMD architecture backends, documentation, and benchmark datasets) are welcome and reviewed openly via GitHub Pull Requests.

## Local Development & Test Setup

### 1. Standalone Zero-PyTorch `ScarCodec` (`numpy` + C99/SIMD only)

`ScarCodec` is designed to operate with **zero PyTorch or deep-learning dependencies**:

```bash
git clone https://github.com/byte-od/mixture-of-scars.git
cd mixture-of-scars

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip numpy scipy astropy zarr numcodecs pytest

# Build and verify the standalone Zero-PyTorch wheel & sdist in read-only site-packages
python scripts/build_standalone_scar_codec_pypi.py --verify

# Run the complete 48-test ScarCodec + Biorthogonal + 37-check Forensic pytest suite
pytest -q \
  tests/test_scar_codec_v6_consolidation.py \
  tests/test_scar_codec_forensic_and_synergy.py \
  tests/test_scar_codec_physical.py \
  tests/test_physics_lab_production_forensic.py \
  tests/test_biorthogonal_v7_3_and_codec.py
```

### 2. Full Ecosystem (`ScarCache`, `ScarNative`, `ScarField`)

```bash
pip install -e ".[dev]"
pytest -q
```

## Architectural & Scientific Integrity Standards

1. **Zero-PyTorch Purity in `scar_codec/`:** Files inside `scar_codec/` (`numpy_engine.py`, `nd_e8.py`, `binary_format.py`, `zarr_plugin.py`, `c_kernels.c`) must never require `torch` at import or runtime, and must preserve bit-identical fallback when a C99 compiler is unavailable.
2. **Backward-Compatible `.scar` Binary Layout:** Any algorithmic enhancement to quantization or biorthogonal projection must preserve header-inclusive `CRC32` validation and existing `.scar` container compatibility (or increment the format version explicitly with migration tests).
3. **Deterministic Benchmarks & Candid Reporting:** All benchmark additions must verify raw public archive inputs via `SHA-256`, use zero oracle tuning on held-out slices, and report both strengths and Pareto trade-offs transparently.
4. **AI Assistance Disclosure:** In accordance with JOSS policy, contributors using generative AI tools for code, tests, or documentation should note the tool and confirm human review and verification in their Pull Request description.

## License

By contributing to this repository, you agree that your contributions will be licensed under the [MIT License](LICENSE).
