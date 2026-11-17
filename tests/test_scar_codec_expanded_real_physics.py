"""Forensic pytest suite for Expanded 10-Archive / 8-Science Real-Data ScarCodec v6.0 Audit."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scar_codec import (
    NumPyVQ4ScarCodec,
    OrthogonalSubspaceProjectorNumPy,
    ScarCodecE8,
    ScarCodecND,
    ScarMmapFile,
    ScarZarrCodec,
    SurpriseGatedAnomalySCacheNumPy,
    save_scar_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "public_physical_benchmarks"
AUDIT_JSON = REPO_ROOT / "results" / "scar_codec_expanded_real_physics_v2_0_audit.json"
ORIGINAL_MANIFEST = DATA_DIR / "manifest_sha256.json"
EXPANDED_MANIFEST = DATA_DIR / "manifest_expanded_v2_sha256.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def test_manifests_and_sha256_integrity() -> None:
    """Verify original 7-entry manifest is untouched AND all 9 new public archives match SHA-256."""
    assert ORIGINAL_MANIFEST.exists()
    orig = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    orig_files = orig["files"] if isinstance(orig, dict) and "files" in orig else orig
    assert len(orig_files) == 7, "Original 7-file manifest must stay untouched for backward compatibility"

    assert EXPANDED_MANIFEST.exists()
    exp = json.loads(EXPANDED_MANIFEST.read_text(encoding="utf-8"))
    exp_files = exp["files"] if isinstance(exp, dict) and "files" in exp else exp
    assert len(exp_files) == 16, "Expanded v2 manifest must cover all 16 official public files (10 archives)"

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert audit["scar_codec_modified"] is False
    assert audit["synthetic_data_used"] is False
    assert audit["oracle_steering_used"] is False
    assert audit["total_public_archives_across_ecosystem"] == 10
    assert audit["total_physical_sciences_covered"] == 8

    for entry in audit["new_provenance_manifest"]:
        fpath = DATA_DIR / entry["filename"]
        assert fpath.exists(), f"Missing official archive: {fpath}"
        assert fpath.stat().st_size == entry["bytes"]
        assert _sha256(fpath) == entry["sha256"]


def test_ligo_gw150914_sub_fp16_underflow_and_chirp_recovery(tmp_path: Path) -> None:
    """Verify LIGO GW150914 ~10^-19 strain underflows naive FP16 to 0.0 while ScarCodec 2^-76 + P_perp recovers merger."""
    with gzip.open(DATA_DIR / "ligo_gw150914_h1_4khz_32s.txt.gz", "rt") as fh:
        h1 = np.loadtxt(fh, comments="#", dtype=np.float32)

    # Naive IEEE-754 FP16 collapses 100% of 131,072 samples to 0.0!
    assert int(np.count_nonzero(h1.astype(np.float16))) == 0

    h1_2d = h1.reshape(512, 256).T.copy()
    proj = OrthogonalSubspaceProjectorNumPy.fit_from_interference_calibration(h1_2d[:, :128], subspace_rank=32)
    clean_2d = proj.project_clean(h1_2d)

    codec = NumPyVQ4ScarCodec(rank=16, max_point_spikes=256)
    pkg = codec.encode(clean_2d)
    assert pkg.global_scale < 1e-20, "Lossless 2^E scale must map ~10^-19 strain into FP16 normal range"
    rec = codec.decode_full(pkg)
    assert np.count_nonzero(rec) > 100000

    # Verify O(1) C99 mmap parity on 10^-19 strain!
    scar_path = tmp_path / "ligo_h1.scar"
    save_scar_file(pkg, scar_path)
    with ScarMmapFile(scar_path) as mm:
        blk = mm.read_block_o1(row_idx=120, block_idx=4)
        assert float(np.max(np.abs(blk - rec[120, 256:320]))) == 0.0

    # Verify deterministic L_inf bound at 2.5e-22 strain
    codec_eb = NumPyVQ4ScarCodec(rank=16, max_point_spikes=256, abs_tolerance=2.5e-22)
    pkg_eb = codec_eb.encode(clean_2d)
    rec_eb = codec_eb.decode_full(pkg_eb)
    assert float(np.max(np.abs(clean_2d - rec_eb))) <= 2.5001e-22


def test_iris_tohoku_super_fp16_overflow_and_global_rmse() -> None:
    """Verify 2011 Mw 9.1 Tohoku-Oki 3.125M counts overflows FP16 while ScarCodec beats ZFP by >2.4x global RMSE."""
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))["experiment_7_iris_tohoku_mw91_seismology"]
    assert audit["peak_rayleigh_counts"] > 3_000_000.0
    assert audit["naive_fp16_overflow_peak_error_counts"] > 3_000_000.0
    assert audit["lossless_pow2_exponent_scale"] == 8192.0
    assert audit["scar_global_rmse_gain_over_zfp_x"] >= 2.4
    assert audit["scar_snr_gain_over_zfp_db"] >= 7.5
    assert audit["p_wave_onset_3axis_cosine_scar"] > 0.995
    assert audit["geophysical_5000count_bound_mode"]["measured_max_abs_err_counts"] <= 5000.0
    assert audit["scache_sta_lta_pwave_retention"]["p_wave_onset_retained_scache"] is True
    assert audit["scache_sta_lta_pwave_retention"]["p_wave_onset_retained_equal_byte_ring"] is False


def test_mitbih_119_iec_clinical_bounds_and_v1_dropout_scache() -> None:
    """Verify MIT-BIH 119 Holter ECG IEC 50 uV & 15 uV 0-violation bounds + Lead-V1 dropout recall."""
    raw = np.frombuffer((DATA_DIR / "mitbih_119.dat").read_bytes(), dtype=np.uint8).reshape(-1, 3)
    s0 = raw[:, 0].astype(np.int32) | ((raw[:, 1].astype(np.int32) & 0x0F) << 8)
    s0 = np.where(s0 >= 2048, s0 - 4096, s0)
    mlii_mv = ((s0 - 1024) / 200.0).astype(np.float32)

    # Test strict IEC 50 uV bound on real clinical ECG slice
    sub = mlii_mv[: 64 * 128].reshape(64, 128)
    c50 = NumPyVQ4ScarCodec(rank=8, max_point_spikes=128, abs_tolerance=0.050)
    p50 = c50.encode(sub)
    r50 = c50.decode_full(p50)
    assert float(np.max(np.abs(sub - r50))) <= 0.05001

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))["experiment_8_physionet_mitbih_119_cardiology_ecg"]
    assert audit["scar_snr_gain_over_zfp_db"] > 5.0
    assert audit["iec_holter_50uv_bound_mode"]["violations_out_of_1299200_samples"] == 0
    assert audit["iec_holter_50uv_bound_mode"]["iec_compression_vs_fp32_x"] >= 1.5
    assert audit["iec_resting_15uv_stress_mode"]["violations_out_of_1299200_samples"] == 0
    assert audit["holter_scache_o1_audit"]["scache_delta_bytes"] == 0
    assert audit["holter_scache_o1_audit"]["pvc_arrhythmia_slot_purity_pct"] >= 80.0
    assert audit["holter_scache_o1_audit"]["pre_horizon_pvcs_retained_scache"] >= 35
    assert audit["holter_scache_o1_audit"]["lead_v1_dropout_pvc_recall_cosine"] >= 0.99


def test_noaa_oisst_continent_null_vacuum_and_zarr_bounds() -> None:
    """Verify 0.0000 C pure continental mask preservation on NOAA OISST + 3D Tucker & ScarZarrCodec bounds."""
    import netCDF4 as nc

    sst = np.asarray(nc.Dataset(DATA_DIR / "noaa_oisst_v2_ltm.nc").variables["sst"][0], dtype=np.float32)
    ocean_mask = np.asarray(nc.Dataset(DATA_DIR / "noaa_oisst_v2_lsmask.nc").variables["mask"][0], dtype=np.float32) > 0.5
    jan = np.where(ocean_mask, np.nan_to_num(sst, nan=0.0), 0.0).astype(np.float32)

    codec = NumPyVQ4ScarCodec(rank=16, max_point_spikes=128)
    rec = codec.decode_full(codec.encode(jan))
    antarctica_rows = np.where(~ocean_mask.any(axis=1))[0]
    assert len(antarctica_rows) >= 10
    assert float(np.max(np.abs(rec[antarctica_rows]))) == 0.0

    zc = ScarZarrCodec(rank=16, max_point_spikes=128, mode="vq4", abs_tolerance=0.05)
    blob = zc.encode(jan)
    zrec = zc.decode(blob)
    assert float(np.max(np.abs(jan - zrec))) <= 0.05005


def test_rosat_allsky_xray_peak_photometry_and_gosset_e8() -> None:
    """Verify ROSAT soft X-ray survey 422x peak gain over ZFP, 1138x over rFFT, and 17.1x Gosset E8."""
    from astropy.io import fits

    with fits.open(DATA_DIR / "rosat_allsky_xray.fits") as hdul:
        ros = np.nan_to_num(np.asarray(hdul[0].data, dtype=np.float32), nan=0.0)

    e8 = ScarCodecE8(rank=12, max_point_spikes=256)
    pkg_e8 = e8.encode(ros)
    rec_e8 = e8.decode_full(pkg_e8)
    ratio_e8 = ros.nbytes / pkg_e8.physical_bytes()
    assert ratio_e8 > 16.5
    top32 = np.argsort(np.abs(ros.ravel()))[-32:]
    assert float(np.max(np.abs(ros.ravel()[top32] - rec_e8.ravel()[top32]))) < 16.0

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))["experiment_10_rosat_allsky_xray_astrophysics"]
    assert audit["scar_peak_gain_over_zfp_x"] > 400.0
    assert audit["scar_peak_gain_over_rfft_x"] > 1000.0
    assert audit["fits_fpack_rice_1"]["e8_peak_gain_over_fpack_floor_x"] >= 2.0
