"""Automated Forensic Verification Suite for `ScarCodec` on Real Public Scientific Archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "public_physical_benchmarks" / "manifest_sha256.json"
REPORT_PATH = REPO_ROOT / "results" / "scar_codec_public_datasets_v1_0_audit.json"


def test_public_archives_sha256_provenance() -> None:
    assert MANIFEST_PATH.exists(), "Missing manifest_sha256.json"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["total_files"] == 7
    for entry in manifest["files"]:
        assert entry["url"].startswith("https://")
        assert len(entry["sha256"]) == 64
        local_f = MANIFEST_PATH.parent / entry["filename"]
        if local_f.exists():
            assert local_f.stat().st_size == entry["bytes"]
            assert hashlib.sha256(local_f.read_bytes()).hexdigest() == entry["sha256"]


def test_zero_oracle_and_physionet_cross_run_p_perp() -> None:
    assert REPORT_PATH.exists(), "Missing scar_codec_public_datasets_v1_0_audit.json"
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert rep["scar_codec_modified"] is False
    assert rep["synthetic_data_used"] is False
    assert rep["oracle_steering_used"] is False

    e1 = rep["experiment_1_physionet_64ch_cross_run_p_perp_vq4"]
    assert "S001R01.edf" in e1["calibration_run"]
    assert "S001R04.edf" in e1["heldout_run"]
    assert e1["calibration_run"] != e1["heldout_run"]
    assert e1["scar_codec_compression_x"] > 10.0
    assert e1["clean_array_p_perp_vq4_scar_snr_db"] > e1["clean_array_unshielded_vq4_snr_db"] + 2.0
    assert e1["raw_c3_laplacian_p_perp_vq4_scar_snr_db"] > e1["raw_c3_laplacian_2bit_adc_snr_db"] + 12.0
    assert e1["o1_block_vs_full_decode_max_diff"] == 0.0


def test_scache_o1_constant_memory_and_shuttle_recall() -> None:
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    e2 = rep["experiment_2_scache_o1_public_telemetry"]

    p64 = e2["physionet_64ch_stream"]
    assert p64["scache_bytes_initial"] == 34816
    assert p64["scache_bytes_final"] == 34816
    assert p64["scache_delta_bytes"] == 0
    assert p64["early_r01_r02_transients_retained_scache"] > 0
    assert p64["early_r01_r02_transients_retained_ring"] == 0
    assert p64["blind_32even_to_32odd_masked_cosine"] > 0.99

    sh = e2["nasa_shuttle_radiator_stream"]
    assert sh["scache_delta_bytes"] == 0
    assert sh["scache_bypass_valve_recall_pct"] == 100.0
    assert sh["circular_ring_bypass_valve_recall_pct"] == 0.0


def test_astronomy_fits_byte_matched_fp16_rfft_and_o1_block_parity() -> None:
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    e3 = rep["experiment_3_public_astronomy_fits_vq4_vs_fp16_rfft"]

    for key in ("nasa_spitzer_irac", "cso_bolocam_sgr_a", "fcrao_l1448_13co_cube"):
        ds = e3[key]
        assert ds["scar_codec_compression_x"] > 10.0
        assert ds["zlib_compression_x"] < 2.0
        # Byte-matched spectral fairness check (`fft_bytes` within 5% of `scar_bytes`)
        ratio = ds["byte_matched_fp16_rfft_bytes"] / ds["scar_codec_bytes"]
        assert 0.90 <= ratio <= 1.05, f"Spectral baseline not byte-matched for {key}: {ratio}"
        assert ds["rmse_gain_vs_fp16_rfft_x"] > 1.0
        assert ds["peak_error_reduction_vs_fp16_rfft_x"] > 4.0
        assert ds["o1_random_read_speedup_vs_zlib_x"] > 5.0
        assert ds["o1_block_vs_full_decode_max_diff"] == 0.0

    # Verify empirical HDR contrast scaling law across the 3 independent observatories
    assert (
        e3["nasa_spitzer_irac"]["peak_error_reduction_vs_fp16_rfft_x"]
        > e3["cso_bolocam_sgr_a"]["peak_error_reduction_vs_fp16_rfft_x"]
        > e3["fcrao_l1448_13co_cube"]["peak_error_reduction_vs_fp16_rfft_x"]
    )
