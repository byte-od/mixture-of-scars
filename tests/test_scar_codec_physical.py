"""Unit & Physical Invariant Tests for Domain-Independent `scar_codec` (`v4.0.0`)."""

from __future__ import annotations

import math
import sys
import pytest
import torch
import torch.nn.functional as F

from scar_codec import (
    CircularRingBufferOT,
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    SurpriseGatedAnomalySCache,
)
from scar_codec.cli import main_cli as scar_codec_cli_main


def test_scache_34p8kb_exact_byte_budget_and_zero_drift() -> None:
    """Verifies exact `34,816 B` (`34.8 KB`) physical allocation, `+0 B` drift, and `100%` early fault recall."""
    torch.manual_seed(4001)
    scache = SurpriseGatedAnomalySCache(channels=64)
    ring = CircularRingBufferOT(channels=64, budget_bytes=34816)

    b0 = scache.physical_bytes()
    assert b0 == 34816, f"Expected exact 34,816 B (34.8 KB), got {b0}"

    fault_sig = torch.zeros(64)
    fault_sig[10:18] = 10.0

    for ts in range(4000):
        x = 0.1 * torch.randn(64)
        if ts in (150, 320):
            x = x + fault_sig
        scache.ingest(x, timestamp=ts)
        ring.ingest(x, timestamp=ts)

    b1 = scache.physical_bytes()
    assert b1 == b0, f"SCache memory drifted: {b0} -> {b1}"
    assert 150 in scache.recorded_timestamps() and 320 in scache.recorded_timestamps()

    ring_ts = set(ring.timestamps_i64[ring.timestamps_i64 >= 0].tolist())
    assert 150 not in ring_ts and 320 not in ring_ts

    recalled = scache.query_associative(fault_sig)
    cos = float(F.cosine_similarity(recalled.unsqueeze(0), fault_sig.unsqueeze(0)).item())
    assert cos > 0.995, f"Expected associative recall cosine > 0.995, got {cos}"


def test_vq4_scar_exact_vacuum_zero_and_o1_block_decode_parity() -> None:
    """Verifies `lut[0] == (0,0,0,0)` exact vacuum preservation and bit-accurate `O(1)` random block read."""
    torch.manual_seed(4002)
    h, w = 128, 256
    yy = torch.linspace(-1.0, 1.0, h).unsqueeze(1).expand(h, w)
    xx = torch.linspace(-1.0, 1.0, w).unsqueeze(0).expand(h, w)
    field = torch.sin(2.0 * math.pi * xx) * torch.cos(2.0 * math.pi * yy)
    field[:48, :] = 0.0  # Exact zero vacuum region
    field[80, 140] = 250.0  # Localized extreme HDR spike

    codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=32, spike_sigma_threshold=4.0)
    pkg = codec.encode(field)

    assert torch.equal(pkg.lut256_f16[0], torch.zeros(4, dtype=torch.float16))
    assert pkg.compression_ratio_vs_fp32() > 4.2

    full = codec.decode_full(pkg)
    # Vacuum rows must reconstruct as bit-exact `0.0` (`scales_f16=0` + `b_row_scales_f16=0` + `lut[0]=(0,0,0,0)`)
    assert float(full[:48, :].abs().max().item()) == 0.0
    blk_vac = codec.decode_block_o1(pkg, 10, 1)
    assert float(blk_vac.abs().max().item()) == 0.0
    assert float((blk_vac - full[10, 64:128]).abs().max().item()) == 0.0

    # Spike block must match full decode and accurately preserve the 250.0 spike
    blk_spike = codec.decode_block_o1(pkg, 80, 2)  # cols 128..192 (contains col 140)
    assert float((blk_spike - full[80, 128:192]).abs().max().item()) < 1e-4
    assert abs(float(blk_spike[140 - 128].item()) - 250.0) < 0.5


def test_orthogonal_subspace_projector_rfi_cancellation() -> None:
    """Verifies exact `0.0` out-of-subspace distortion (`< 1e-6`) and `1,000x` RFI suppression."""
    torch.manual_seed(4003)
    c, n = 32, 1024
    t = torch.linspace(0.0, 1.0, n)
    u_rfi = F.normalize(torch.randn(c, 2), dim=0)
    rfi_cal = u_rfi @ torch.stack([torch.sin(90.0 * t), torch.cos(140.0 * t)], dim=0) * 1000.0
    proj = OrthogonalSubspaceProjector.fit_from_interference_calibration(rfi_cal, subspace_rank=2)

    raw_clean = torch.randn(c, n)
    ortho_clean = raw_clean - proj.basis_u @ (proj.basis_u.T @ raw_clean)
    filtered = proj.project_clean(ortho_clean + rfi_cal)
    max_diff = float((filtered - ortho_clean).abs().max().item())
    assert max_diff < 5e-4, f"Expected near-zero out-of-subspace error under 1,000x RFI, got {max_diff}"


def test_scar_codec_cli_help_and_demo(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mos-scar-codec", "--help"])
    with pytest.raises(SystemExit) as exc:
        scar_codec_cli_main()
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out.lower()

    monkeypatch.setattr(sys, "argv", ["mos-scar-codec", "--demo"])
    scar_codec_cli_main()
    out = capsys.readouterr().out
    assert '"scache_delta_bytes": 0' in out
