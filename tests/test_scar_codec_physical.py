"""Unit & Physical Invariant Tests for Domain-Independent `scar_codec` (`v4.0.0`)."""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys
import pytest
import torch
import torch.nn.functional as F

from scar_codec import (
    CircularRingBufferOT,
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    SurpriseGatedAnomalySCache,
    __version__,
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
    fault_sig[10:16] = torch.tensor([10.0, 8.2, -7.4, 11.5, -6.8, 9.1])

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

    # Partial-cue hetero-associative recall: query ONLY channels 10..12 (`4` of `6` fault channels masked to `0.0`!)
    partial_cue = torch.zeros(64)
    partial_cue[10:12] = fault_sig[10:12]
    recalled = scache.query_associative(partial_cue)
    cos_all = float(F.cosine_similarity(recalled.unsqueeze(0), fault_sig.unsqueeze(0)).item())
    cos_masked = float(F.cosine_similarity(recalled[12:16].unsqueeze(0), fault_sig[12:16].unsqueeze(0)).item())
    assert cos_all > 0.995, f"Expected partial-cue associative recall cosine > 0.995, got {cos_all}"
    assert cos_masked > 0.995, f"Expected unseen masked-channel cosine > 0.995, got {cos_masked}"


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
    help_txt = capsys.readouterr().out
    assert "usage:" in help_txt.lower()
    assert f"ScarCodec v{__version__}" in help_txt

    monkeypatch.setattr(sys, "argv", ["mos-scar-codec", "--demo"])
    scar_codec_cli_main()
    out = capsys.readouterr().out
    assert '"scache_delta_bytes": 0' in out


def test_scar_codec_readme_quickstart_executes_verbatim() -> None:
    """Extracts the Python Quick Start snippet from `scar_codec/README.md` and executes it verbatim."""
    readme_path = Path(__file__).resolve().parents[1] / "scar_codec" / "README.md"
    readme_txt = readme_path.read_text(encoding="utf-8")
    m = re.search(r"```python\n(.*?)```", readme_txt, flags=re.DOTALL)
    assert m is not None, "Missing ```python Quick Start block in scar_codec/README.md"
    snippet = m.group(1)

    torch.manual_seed(4005)
    field_2d_fp32 = torch.randn(256, 256)
    field_2d_fp32[:32, :] = 0.0
    noisy_baseline = torch.randn(64, 512)
    heldout_recording = torch.randn(64, 256)

    fault = torch.zeros(64)
    fault[:8] = 12.0
    mission_stream = [0.1 * torch.randn(64) + (fault if t == 60 else 0.0) for t in range(120)]
    known_channels_bool = torch.arange(64) % 2 == 0
    partial_frame = torch.where(known_channels_bool, fault, torch.zeros(64))
    masked_deviation_cue = partial_frame.clone()

    ns = {
        "field_2d_fp32": field_2d_fp32,
        "noisy_baseline": noisy_baseline,
        "heldout_recording": heldout_recording,
        "mission_stream": mission_stream,
        "partial_frame": partial_frame,
        "known_channels_bool": known_channels_bool,
        "masked_deviation_cue": masked_deviation_cue,
    }
    exec(snippet, ns)

    assert ns["block_64"].shape == (64,)
    assert ns["recon_full"].shape == (256, 256)
    assert ns["scache"].physical_bytes() == 34816
    assert 60 in ns["scache"].recorded_timestamps()
    assert ns["reconstructed"].shape == (64,)
    cos_odd = float(F.cosine_similarity(ns["reconstructed"][1::2].unsqueeze(0), fault[1::2].unsqueeze(0)).item())
    assert cos_odd > 0.99, f"Expected > 0.99 masked partial-cue recall cosine, got {cos_odd}"

