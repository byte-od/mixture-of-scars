"""Command-line interface (`scar-codec` / `mos-scar-codec`) for Zero-PyTorch physical ScarCodec v6.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from scar_codec import (
    HAS_TORCH,
    NumPyVQ4ScarCodec,
    OrthogonalSubspaceProjectorNumPy,
    ScarCodecE8,
    ScarMmapFile,
    SurpriseGatedAnomalySCacheNumPy,
    __version__,
    get_compiled_c_lib,
    save_scar_file,
)


def run_quick_demo() -> dict:
    rng = np.random.default_rng(2026)
    scache = SurpriseGatedAnomalySCacheNumPy(channels=64)
    b0 = scache.physical_bytes()
    for t in range(1024):
        frame = (0.05 * rng.standard_normal(64)).astype(np.float32)
        if t in (120, 480):
            frame[8:16] += 2.5
        scache.ingest(frame, timestamp=t)
    b1 = scache.physical_bytes()

    t_axis = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    field = np.outer(np.sin(2.0 * np.pi * t_axis), np.cos(4.0 * np.pi * t_axis)).astype(np.float32)
    field[:64, :] = 0.0
    field[128, 128] = 50.0

    codec = NumPyVQ4ScarCodec(rank=16, max_point_spikes=64)
    pkg = codec.encode(field)
    blk = codec.decode_block_o1(pkg, row_idx=128, block_idx=2)
    full = codec.decode_full(pkg)

    codec_e8 = ScarCodecE8(rank=12, max_point_spikes=64)
    pkg_e8 = codec_e8.encode(field)

    with tempfile.TemporaryDirectory() as td:
        scar_p = Path(td) / "demo.scar"
        save_scar_file(pkg, scar_p)
        with ScarMmapFile(scar_p) as mm:
            _ = mm.read_block_o1(row_idx=128, block_idx=2)
            t0 = time.perf_counter()
            for _ in range(200):
                blk_mm = mm.read_block_o1(row_idx=128, block_idx=2)
            mmap_block_us = ((time.perf_counter() - t0) / 200.0) * 1e6
            mmap_diff = float(np.max(np.abs(blk_mm - full[128, 128:192])))

    proj = OrthogonalSubspaceProjectorNumPy.fit_from_interference_calibration(
        rng.standard_normal((16, 128)).astype(np.float32), subspace_rank=2
    )
    clean_test = proj.project_clean(rng.standard_normal((16, 64)).astype(np.float32))

    return {
        "version": __version__,
        "zero_pytorch_native": True,
        "c_simd_compiled": get_compiled_c_lib() is not None,
        "optional_torch_installed": HAS_TORCH,
        "scache_physical_bytes_initial": b0,
        "scache_physical_bytes_final": b1,
        "scache_delta_bytes": b1 - b0,
        "scache_recorded_shocks": scache.recorded_timestamps(),
        "vq4_compression_ratio_vs_fp32": round(pkg.compression_ratio_vs_fp32(), 3),
        "gosset_e8_1bpw_compression_ratio_vs_fp32": round(pkg_e8.compression_ratio_vs_fp32(), 3),
        "vacuum_rows_max_abs_err": float(np.max(np.abs(full[:64, :]))),
        "block_o1_max_diff": float(np.max(np.abs(blk - full[128, 128:192]))),
        "mmap_o1_max_diff": mmap_diff,
        "mmap_o1_single_block_us": round(mmap_block_us, 3),
        "projector_output_shape": list(clean_test.shape),
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="scar-codec",
        description=f"ScarCodec v{__version__} — Zero-PyTorch Physical Signal, FITS/Zarr Storage & O(1) Telemetry Scar Operators",
    )
    parser.add_argument("--version", action="version", version=f"scar-codec {__version__}")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a self-contained Zero-PyTorch verification of VQ4/E8 O(1) mmap decode, P_perp RFI nulling & 34.8 KB SCache",
    )
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(run_quick_demo(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
