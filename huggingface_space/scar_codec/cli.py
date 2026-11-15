"""Command-line interface (`mos-scar-codec`) for domain-independent physical ScarCodec audits."""

from __future__ import annotations

import argparse
import json
import torch

from scar_codec.core import (
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    SurpriseGatedAnomalySCache,
    __version__,
)


def run_quick_demo() -> dict:
    torch.manual_seed(2026)
    scache = SurpriseGatedAnomalySCache()
    b0 = scache.physical_bytes()
    for t in range(1024):
        frame = 0.05 * torch.randn(64)
        if t in (120, 480):
            frame[8:16] += 2.5
        scache.ingest(frame, timestamp=t)
    b1 = scache.physical_bytes()

    t_axis = torch.linspace(0.0, 1.0, 256)
    field = torch.outer(torch.sin(2.0 * 3.14159 * t_axis), torch.cos(4.0 * 3.14159 * t_axis))
    field[:64, :] = 0.0
    field[128, 128] = 50.0
    codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=64)
    pkg = codec.encode(field)
    blk = codec.decode_block_o1(pkg, 128, 2)
    full = codec.decode_full(pkg)

    proj = OrthogonalSubspaceProjector.fit_from_interference_calibration(torch.randn(16, 128), subspace_rank=2)
    clean_test = proj.project_clean(torch.randn(16, 64))

    return {
        "version": __version__,
        "scache_physical_bytes_initial": b0,
        "scache_physical_bytes_final": b1,
        "scache_delta_bytes": b1 - b0,
        "scache_recorded_shocks": scache.recorded_timestamps(),
        "field_compression_ratio_vs_fp32": round(pkg.compression_ratio_vs_fp32(), 3),
        "block_o1_max_diff": float((blk - full[128, 128:192]).abs().max().item()),
        "projector_output_shape": list(clean_test.shape),
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="mos-scar-codec",
        description="ScarCodec v4.0.0 — Domain-Independent Physical Signal, Storage & Telemetry Scar Operators",
    )
    parser.add_argument("--version", action="version", version=f"scar-codec {__version__}")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick self-contained verification of VQ4-Scar O(1) decode, P_perp RFI nulling & 34.8 KB SCache",
    )
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(run_quick_demo(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
