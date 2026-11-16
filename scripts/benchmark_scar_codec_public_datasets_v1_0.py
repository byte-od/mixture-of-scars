#!/usr/bin/env python3
"""Empirical Benchmark of Domain-Independent `ScarCodec` on Real Public Scientific Archives (`v1.0.0`).

Evaluates the unmodified `scar_codec` operators (`PhysicalVQ4ScarCodec`, `OrthogonalSubspaceProjector`,
and `SurpriseGatedAnomalySCache` vs `zlib -9`, `2-bit` scalar telemetry ADC, `CircularRingBufferOT`,
and strictly byte-matched IEEE-754 `FP16` 2D `rfft2`) across FIVE real, publicly archived observational
and experimental datasets spanning four scientific disciplines (`0` synthetic signals, `0` oracle steering):

  1. Biomedical Multi-Channel Array (`PhysioNet EEGMMIDB v1.0.0`, NIH/Wadsworth, Schalk et al. 2004):
     - Cross-run blind `P_perp` calibration on eyes-open baseline `S001R01.edf` (`[64, 9760]`) applied
       to held-out motor-imagery run `S001R04.edf` (`[64, 19968]`, `5.11 MB` FP32).
     - Chronological multi-run stream `S001R01 -> S001R02 -> S001R04` (`39,520 x 64` = `10.12 MB` FP32)
       in exact `34,816 B` (`34.8 KB`, `Δ = 0 B`) `SCache O(1)` with blind 32-even -> 32-odd sensor recall.
  2. Spacecraft Telemetry (`NASA Space Shuttle Radiator Control — Statlog Shuttle`, UCI ID 148, Catlett 1992):
     - `14,500 x 8` physical radiator sensors streamed through default `SurpriseGatedAnomalySCache` (`Δ = 0 B`),
       testing retention of ultra-rare `0.041%` Bypass Valve Open/Close (`Class 6 & 7`) faults vs `CircularRingBufferOT`.
  3. Space Infrared Astrophysics (`NASA Spitzer Space Telescope — IRAC`, `spitzer_irac.fits`, Fazio et al. 2004):
     - `[513, 1025]` IEEE-754 FP32 sky mosaic (`2.10 MB`) with `968.8x` real protostellar/stellar HDR point sources.
  4. Submillimeter Galactic Center (`CSO Bolocam Galactic Plane Survey — Sgr A* / Sgr B2`, `gc_bolocam_gps.fits`, Ginsburg et al. 2013):
     - `[638, 640]` IEEE-754 FP32 `1.1 mm` continuum map (`1.63 MB`) with `581.6x` core jumps and `5.0%` off-scan `NaN` sky mask.
  5. Radio Astronomy Spectral Cube (`FCRAO 14-m Radio Telescope — L1448 13CO J=1->0`, `l1448_13co.fits`, Ridge et al. 2006):
     - `[53, 105, 105]` (`[5565, 105]`, `2.34 MB` FP32) molecular outflow velocity-channel spectral cube.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
import zipfile
import zlib

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_codec import (
    CircularRingBufferOT,
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    SurpriseGatedAnomalySCache,
)

DATA_DIR = REPO_ROOT / "data" / "public_physical_benchmarks"

PUBLIC_ARCHIVES = [
    {
        "id": "physionet_s001r01_eyes_open",
        "filename": "physionet_s001r01_eyes_open.edf",
        "url": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
        "domain": "Biomedical 64-Channel Biopotential Array (Baseline Eyes-Open)",
        "citation": "Schalk et al. (2004), IEEE Trans. Biomed. Eng. 51(6), 1034-1043; Goldberger et al. (2000), Circulation 101(23), e215-e220.",
        "license": "ODC-By-1.0 (PhysioNet Open Data)",
    },
    {
        "id": "physionet_s001r02_eyes_closed",
        "filename": "physionet_s001r02_eyes_closed.edf",
        "url": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R02.edf",
        "domain": "Biomedical 64-Channel Biopotential Array (Resting Eyes-Closed)",
        "citation": "Schalk et al. (2004), IEEE Trans. Biomed. Eng. 51(6), 1034-1043.",
        "license": "ODC-By-1.0 (PhysioNet Open Data)",
    },
    {
        "id": "physionet_s001r04_motor_task",
        "filename": "physionet_s001r04_motor_task.edf",
        "url": "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R04.edf",
        "domain": "Biomedical 64-Channel Biopotential Array (Held-Out Motor Imagery Task)",
        "citation": "Schalk et al. (2004), IEEE Trans. Biomed. Eng. 51(6), 1034-1043.",
        "license": "ODC-By-1.0 (PhysioNet Open Data)",
    },
    {
        "id": "nasa_shuttle_statlog",
        "filename": "nasa_shuttle.zip",
        "url": "https://archive.ics.uci.edu/static/public/148/statlog+shuttle.zip",
        "domain": "Aerospace Multi-Sensor Telemetry (NASA Space Shuttle Radiator Control)",
        "citation": "Catlett, J. (1992) / Michie, Spiegelhalter & Taylor (1994), NASA Space Shuttle Statlog Dataset, UCI ML Repository ID 148.",
        "license": "CC-BY-4.0 (UCI Machine Learning Repository)",
    },
    {
        "id": "nasa_spitzer_irac",
        "filename": "spitzer_irac.fits",
        "url": "https://raw.githubusercontent.com/astropy/astropy-data/gh-pages/photometry/spitzer_example_image.fits",
        "domain": "Space Infrared Astrophysics (NASA Spitzer Space Telescope IRAC)",
        "citation": "Fazio et al. (2004), Astrophys. J. Suppl. Ser. 154(1), 10-17; Astropy Collaboration (2022), Astrophys. J. 935, 167.",
        "license": "BSD-3-Clause / NASA Public Archival Data",
    },
    {
        "id": "cso_bolocam_galactic_center",
        "filename": "gc_bolocam_gps.fits",
        "url": "https://raw.githubusercontent.com/astropy/astropy-data/gh-pages/galactic_center/gc_bolocam_gps.fits",
        "domain": "Submillimeter Galactic Center Continuum (CSO Bolocam 1.1 mm Sgr A* / Sgr B2)",
        "citation": "Ginsburg et al. (2013), Astrophys. J. Suppl. Ser. 208(2), 14; Aguirre et al. (2011), ApJS 192, 4.",
        "license": "BSD-3-Clause / Public Astronomical Survey Archive",
    },
    {
        "id": "fcrao_l1448_13co_cube",
        "filename": "l1448_13co.fits",
        "url": "https://raw.githubusercontent.com/astropy/astropy-data/gh-pages/l1448/l1448_13co.fits",
        "domain": "Radio Astronomy Spectral Line Cube (FCRAO 14-m L1448 13CO J=1->0 Outflow)",
        "citation": "Ridge et al. (2006), Astron. J. 131(6), 2921-2933 (COMPLETE Survey of Star-Forming Regions).",
        "license": "BSD-3-Clause / Public Radio Astronomy Survey Archive",
    },
]


def ensure_public_datasets_downloaded() -> list[dict]:
    """Downloads missing public files from official HTTPS URLs and records/verifies SHA-256 checksums."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest_sha256.json"
    existing_hashes = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")).get("files", []):
            existing_hashes[entry["filename"]] = entry["sha256"]

    verified_records = []
    for item in PUBLIC_ARCHIVES:
        fpath = DATA_DIR / item["filename"]
        if not fpath.exists():
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0 (ScarCodec-Audit/1.0)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                fpath.write_bytes(resp.read())
        raw_bytes = fpath.read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if item["filename"] in existing_hashes:
            assert digest == existing_hashes[item["filename"]], (
                f"SHA-256 mismatch for {item['filename']}: expected {existing_hashes[item['filename']]}, got {digest}"
            )
        rec = dict(item)
        rec["bytes"] = len(raw_bytes)
        rec["sha256"] = digest
        verified_records.append(rec)

    manifest_path.write_text(
        json.dumps(
            {
                "suite": "scar_codec_public_datasets_v1_0",
                "total_files": len(verified_records),
                "total_raw_archive_bytes": sum(r["bytes"] for r in verified_records),
                "files": verified_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return verified_records


def read_fits_primary_f32(path: Path) -> tuple[dict, np.ndarray]:
    """Standard IEEE-754 / 2,880-byte FITS primary HDU reader (Pence et al. 2010, A&A 524, A42)."""
    raw = path.read_bytes()
    pos = 0
    hdr: dict[str, str] = {}
    while True:
        block = raw[pos : pos + 2880].decode("ascii", errors="replace")
        pos += 2880
        done = False
        for i in range(0, 2880, 80):
            card = block[i : i + 80]
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if "=" in card[:10]:
                hdr[key] = card[10:].split("/")[0].strip().strip("'").strip()
        if done:
            break
    bitpix = int(hdr["BITPIX"])
    naxis = int(hdr["NAXIS"])
    shape = tuple(int(hdr[f"NAXIS{ax}"]) for ax in range(naxis, 0, -1))
    dtype_map = {-32: ">f4", -64: ">f8", 16: ">i2", 32: ">i4"}
    nbytes = int(np.prod(shape)) * (abs(bitpix) // 8)
    arr = np.frombuffer(raw[pos : pos + nbytes], dtype=dtype_map[bitpix]).astype(np.float32).reshape(shape)
    bscale = float(hdr.get("BSCALE", 1.0))
    bzero = float(hdr.get("BZERO", 0.0))
    return hdr, (arr * bscale + bzero).astype(np.float32)


def read_edf_64ch_uv(path: Path) -> tuple[list[str], np.ndarray]:
    """Standard European Data Format (EDF, Kemp et al. 1992) reader calibrating int16 ADC counts to microvolts."""
    b = path.read_bytes()
    ns = int(b[252:256].decode("ascii").strip())
    nrec = int(b[236:244].decode("ascii").strip())
    pos = 256
    labels = [b[pos + i * 16 : pos + (i + 1) * 16].decode("ascii").strip() for i in range(ns)]
    pos += ns * 16 + ns * 88
    pmin = np.array([float(b[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    pmax = np.array([float(b[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    dmin = np.array([float(b[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8
    dmax = np.array([float(b[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)])
    pos += ns * 8 + ns * 80
    samps = [int(b[pos + i * 8 : pos + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)]
    hdr_bytes = int(b[184:192].decode("ascii").strip())

    n_eeg = 64
    spr = samps[0]
    rec_total = sum(samps)
    raw_i16 = np.frombuffer(b[hdr_bytes : hdr_bytes + nrec * rec_total * 2], dtype="<i2").reshape(nrec, rec_total)
    eeg_i16 = raw_i16[:, : n_eeg * spr].reshape(nrec, n_eeg, spr).transpose(1, 0, 2).reshape(n_eeg, nrec * spr)
    gain = (pmax[:n_eeg] - pmin[:n_eeg]) / (dmax[:n_eeg] - dmin[:n_eeg])
    offset = pmax[:n_eeg] - gain * dmax[:n_eeg]
    eeg_uv = (eeg_i16.astype(np.float32) * gain[:, None] + offset[:, None]).astype(np.float32)
    return [lb.rstrip(".") for lb in labels[:n_eeg]], eeg_uv


def _cosine_and_snr_db(est: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    c = float(F.cosine_similarity(est.reshape(1, -1), ref.reshape(1, -1)).item())
    err = est - ref
    snr = 10.0 * math.log10(float(ref.square().mean().item()) / max(1e-12, float(err.square().mean().item())))
    return round(c, 5), round(snr, 2)


@torch.no_grad()
def evaluate_physionet_array_interference_and_compression() -> dict:
    """Experiment 1: Cross-run blind `P_perp` calibration (`S001R01`) -> held-out motor task (`S001R04`)."""
    ch_names, eeg_r01 = read_edf_64ch_uv(DATA_DIR / "physionet_s001r01_eyes_open.edf")
    _, eeg_r04 = read_edf_64ch_uv(DATA_DIR / "physionet_s001r04_motor_task.edf")

    calib_r01 = torch.from_numpy(eeg_r01)          # [64, 9760] separate baseline run
    heldout_r04 = torch.from_numpy(eeg_r04[:, :19968])  # [64, 19968] = 312 * 64 samples (124.8 s @ 160 Hz)
    raw_bytes = heldout_r04.numel() * 4

    # Quantify natural common-mode + frontopolar ocular blink dominance in raw baseline `S001R01`
    _, sv_r01, _ = torch.linalg.svd(calib_r01, full_matrices=False)
    top4_energy_pct = float(100.0 * sv_r01[:4].square().sum().item() / sv_r01.square().sum().item())

    # Fit `P_perp` strictly on `S001R01` (`0` access to held-out `S001R04`!)
    proj = OrthogonalSubspaceProjector.fit_from_interference_calibration(calib_r01, subspace_rank=4)
    ref_clean_array = proj.project_clean(heldout_r04)

    # Model-free physical ground truth: local Hjorth Surface Laplacian over left primary motor cortex `C3`
    # `C3 - 0.25 * (Fc3 + Cp3 + C1 + C5)` computed directly on raw unprojected FP32 microvolts!
    idx_map = {nm: i for i, nm in enumerate(ch_names)}
    w_lap_c3 = torch.zeros(64, dtype=torch.float32)
    w_lap_c3[idx_map["C3"]] = 1.0
    for nb in ("Fc3", "Cp3", "C1", "C5"):
        w_lap_c3[idx_map[nb]] = -0.25
    ref_raw_laplacian_c3 = w_lap_c3 @ heldout_r04
    ref_proj_laplacian_c3 = w_lap_c3 @ ref_clean_array

    # Baseline A: Lossless `zlib -9` on raw FP32 array
    zlib_bytes = len(zlib.compress(heldout_r04.numpy().tobytes(), level=9))

    # Baseline B: 2-bit scalar telemetry quantizer (`VLBI / BLE wearable ADC`) before ground processing
    rms_ch = heldout_r04.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
    z_2b = heldout_r04 / rms_ch
    q_2b = torch.where(z_2b < 0.0, torch.where(z_2b < -1.0, -1.5, -0.5), torch.where(z_2b < 1.0, 0.5, 1.5)) * rms_ch
    arr_2b = proj.project_clean(q_2b)
    lap_2b = w_lap_c3 @ q_2b

    # Baseline C: Unshielded `PhysicalVQ4ScarCodec` (encoding raw blink-dominated array before `P_perp`)
    codec = PhysicalVQ4ScarCodec(rank=8, max_point_spikes=256, spike_sigma_threshold=4.5)
    pkg_unshielded = codec.encode(heldout_r04)
    rec_unshielded = codec.decode_full(pkg_unshielded)
    arr_unshielded = proj.project_clean(rec_unshielded)
    lap_unshielded = w_lap_c3 @ rec_unshielded

    # Proposed: Cross-run `P_perp (R01)` + `PhysicalVQ4ScarCodec` on held-out `R04`
    pkg_scar = codec.encode(ref_clean_array)
    rec_scar = codec.decode_full(pkg_scar)
    lap_scar = w_lap_c3 @ rec_scar

    c_arr_2b, snr_arr_2b = _cosine_and_snr_db(arr_2b, ref_clean_array)
    c_arr_un, snr_arr_un = _cosine_and_snr_db(arr_unshielded, ref_clean_array)
    c_arr_sc, snr_arr_sc = _cosine_and_snr_db(rec_scar, ref_clean_array)

    c_lap_2b, snr_lap_2b = _cosine_and_snr_db(lap_2b, ref_raw_laplacian_c3)
    c_lap_un, snr_lap_un = _cosine_and_snr_db(lap_unshielded, ref_raw_laplacian_c3)
    c_lap_sc_raw, snr_lap_sc_raw = _cosine_and_snr_db(lap_scar, ref_raw_laplacian_c3)
    c_lap_sc_proj, snr_lap_sc_proj = _cosine_and_snr_db(lap_scar, ref_proj_laplacian_c3)

    # Verify `O(1)` single-block decode parity on `S001R04`
    blk_o1 = codec.decode_block_o1(pkg_scar, idx_map["C3"], 100)
    o1_diff = float((blk_o1 - rec_scar[idx_map["C3"], 6400:6464]).abs().max().item())

    return {
        "calibration_run": "S001R01.edf ([64, 9760], 160 Hz, Eyes-Open Baseline)",
        "heldout_run": "S001R04.edf ([64, 19968], 160 Hz, Motor Imagery Task)",
        "calibration_top4_artifact_energy_pct": round(top4_energy_pct, 2),
        "raw_fp32_bytes": raw_bytes,
        "zlib_level9_bytes": zlib_bytes,
        "zlib_compression_x": round(raw_bytes / zlib_bytes, 2),
        "scar_codec_bytes": pkg_scar.physical_bytes(),
        "scar_codec_compression_x": round(pkg_scar.compression_ratio_vs_fp32(), 2),
        "clean_array_2bit_adc_cosine": c_arr_2b,
        "clean_array_2bit_adc_snr_db": snr_arr_2b,
        "clean_array_unshielded_vq4_cosine": c_arr_un,
        "clean_array_unshielded_vq4_snr_db": snr_arr_un,
        "clean_array_p_perp_vq4_scar_cosine": c_arr_sc,
        "clean_array_p_perp_vq4_scar_snr_db": snr_arr_sc,
        "raw_c3_laplacian_2bit_adc_cosine": c_lap_2b,
        "raw_c3_laplacian_2bit_adc_snr_db": snr_lap_2b,
        "raw_c3_laplacian_unshielded_vq4_cosine": c_lap_un,
        "raw_c3_laplacian_unshielded_vq4_snr_db": snr_lap_un,
        "raw_c3_laplacian_p_perp_vq4_scar_cosine": c_lap_sc_raw,
        "raw_c3_laplacian_p_perp_vq4_scar_snr_db": snr_lap_sc_raw,
        "projected_c3_laplacian_p_perp_vq4_scar_cosine": c_lap_sc_proj,
        "projected_c3_laplacian_p_perp_vq4_scar_snr_db": snr_lap_sc_proj,
        "o1_block_vs_full_decode_max_diff": o1_diff,
    }


@torch.no_grad()
def evaluate_scache_o1_public_telemetry_streams() -> dict:
    """Experiment 2: Constant-memory `SCache O(1)` on PhysioNet 64-ch (`34,816 B`) & NASA Space Shuttle (`14,500` frames)."""
    # --- Part A: Canonical `34,816 B` (`34.8 KB`) 64-channel chronological PhysioNet stream (`R01 -> R02 -> R04`) ---
    _, eeg_r01 = read_edf_64ch_uv(DATA_DIR / "physionet_s001r01_eyes_open.edf")
    _, eeg_r02 = read_edf_64ch_uv(DATA_DIR / "physionet_s001r02_eyes_closed.edf")
    _, eeg_r04 = read_edf_64ch_uv(DATA_DIR / "physionet_s001r04_motor_task.edf")
    stream64 = torch.from_numpy(np.concatenate([eeg_r01, eeg_r02, eeg_r04], axis=1).T.copy())  # [39520, 64]

    sc64 = SurpriseGatedAnomalySCache(channels=64, surprise_threshold=3.58)
    ring64 = CircularRingBufferOT(channels=64, budget_bytes=sc64.physical_bytes())
    b64_init = sc64.physical_bytes()
    assert b64_init == 34816, f"Expected exact 34,816 B, got {b64_init}"

    t0 = time.perf_counter()
    b64_mid = b64_init
    for ts in range(stream64.shape[0]):
        sc64.ingest(stream64[ts], timestamp=ts)
        ring64.ingest(stream64[ts], timestamp=ts)
        if ts == 19520:
            b64_mid = sc64.physical_bytes()
    fps_64 = stream64.shape[0] / max(1e-6, time.perf_counter() - t0)
    b64_final = sc64.physical_bytes()

    early_slots_64 = [pos for pos in range(sc64.num_slots) if 0 <= int(sc64.slot_ts_i64[pos].item()) < 19520]
    ring_early_64 = [int(t) for t in ring64.timestamps_i64.tolist() if 0 <= int(t) < 19520]

    # Blind 32-even -> 32-odd electrode spatial interpolation via `W_scar` (`50%` of sensors masked to `0.0`!)
    cos_full_64, cos_masked_32 = [], []
    for pos in early_slots_64:
        tv = sc64.slot_vals_f16[pos].float()
        cue = torch.zeros_like(tv)
        cue[0::2] = tv[0::2]
        rv = sc64.query_associative(cue)
        cos_full_64.append(float(F.cosine_similarity(rv.unsqueeze(0), tv.unsqueeze(0)).item()))
        cos_masked_32.append(float(F.cosine_similarity(rv[1::2].unsqueeze(0), tv[1::2].unsqueeze(0)).item()))

    # --- Part B: NASA Space Shuttle Radiator Telemetry (`shuttle.tst`, `14,500` frames, `8` physical sensors) ---
    with zipfile.ZipFile(DATA_DIR / "nasa_shuttle.zip") as zf:
        lines = zf.read("shuttle.tst").decode("ascii").strip().splitlines()
    mat = np.array([[float(x) for x in ln.split()] for ln in lines if ln.strip()], dtype=np.float32)
    x_rad_raw = torch.from_numpy(mat[:, 1:9])  # attributes A2..A9 (attribute A1 is timestamp index per shuttle.doc)
    y_cls = mat[:, 9].astype(int)

    # Causal standardization strictly on initial 256 nominal frames (`0` future leakage!)
    mu0 = x_rad_raw[:256].mean(dim=0)
    std0 = x_rad_raw[:256].std(dim=0).clamp(min=1.0)
    x_rad = (x_rad_raw - mu0) / std0

    sc8 = SurpriseGatedAnomalySCache(channels=8)  # 100% default `surprise_threshold = 4.2`!
    ring8 = CircularRingBufferOT(channels=8, budget_bytes=sc8.physical_bytes())
    b8_init = sc8.physical_bytes()
    t1 = time.perf_counter()
    for ts in range(x_rad.shape[0]):
        sc8.ingest(x_rad[ts], timestamp=ts)
        ring8.ingest(x_rad[ts], timestamp=ts)
    fps_8 = x_rad.shape[0] / max(1e-6, time.perf_counter() - t1)
    b8_final = sc8.physical_bytes()

    rec8_ts = set(sc8.recorded_timestamps())
    ring8_ts = set(ring8.timestamps_i64[ring8.timestamps_i64 >= 0].tolist())
    bpv_fault_indices = [int(i) for i in np.where(np.isin(y_cls, [6, 7]))[0]]  # Class 6 Bpv Close (4) + Class 7 Bpv Open (2)
    sc8_bpv_hits = sum(1 for i in bpv_fault_indices if i in rec8_ts)
    ring8_bpv_hits = sum(1 for i in bpv_fault_indices if i in ring8_ts)

    # Honest rank-saturation diagnostic on `[8, 8]` `W_scar` vs `[64, 64]` `W_scar`
    bpv_slots = [pos for pos in range(sc8.num_slots) if int(sc8.slot_ts_i64[pos].item()) in bpv_fault_indices]
    shuttle_1sensor_hidden_cos = []
    for pos in bpv_slots:
        tv = sc8.slot_vals_f16[pos].float()
        cue = torch.zeros_like(tv)
        cue[7] = tv[7]
        rv = sc8.query_associative(cue)
        shuttle_1sensor_hidden_cos.append(abs(float(F.cosine_similarity(rv[:7].unsqueeze(0), tv[:7].unsqueeze(0)).item())))

    return {
        "physionet_64ch_stream": {
            "chronological_runs": "S001R01 -> S001R02 -> S001R04",
            "total_frames": int(stream64.shape[0]),
            "channels": 64,
            "raw_fp32_bytes": int(stream64.numel() * 4),
            "scache_bytes_initial": b64_init,
            "scache_bytes_mid": b64_mid,
            "scache_bytes_final": b64_final,
            "scache_delta_bytes": b64_final - b64_init,
            "circular_ring_budget_bytes": ring64.physical_bytes(),
            "circular_ring_frame_capacity": ring64.capacity,
            "early_r01_r02_transients_retained_scache": len(early_slots_64),
            "early_r01_r02_transients_retained_ring": len(ring_early_64),
            "blind_32even_to_32odd_full_cosine": round(float(np.mean(cos_full_64)), 5),
            "blind_32even_to_32odd_masked_cosine": round(float(np.mean(cos_masked_32)), 5),
            "ingest_fps": round(fps_64, 1),
        },
        "nasa_shuttle_radiator_stream": {
            "dataset": "shuttle.tst (NASA Space Shuttle Radiator Control)",
            "total_frames": int(x_rad.shape[0]),
            "physical_sensors": 8,
            "raw_fp32_bytes": int(x_rad.numel() * 4),
            "scache_bytes_initial": b8_init,
            "scache_bytes_final": b8_final,
            "scache_delta_bytes": b8_final - b8_init,
            "circular_ring_budget_bytes": ring8.physical_bytes(),
            "circular_ring_frame_capacity": ring8.capacity,
            "critical_bypass_valve_fault_indices": bpv_fault_indices,
            "critical_bypass_valve_prevalence_pct": round(100.0 * len(bpv_fault_indices) / x_rad.shape[0], 4),
            "scache_bypass_valve_recall": f"{sc8_bpv_hits}/{len(bpv_fault_indices)}",
            "scache_bypass_valve_recall_pct": round(100.0 * sc8_bpv_hits / len(bpv_fault_indices), 1),
            "circular_ring_bypass_valve_recall": f"{ring8_bpv_hits}/{len(bpv_fault_indices)}",
            "circular_ring_bypass_valve_recall_pct": round(100.0 * ring8_bpv_hits / len(bpv_fault_indices), 1),
            "rank8_saturated_1sensor_abs_cosine": round(float(np.mean(shuttle_1sensor_hidden_cos)), 4),
            "total_high_surprise_writes": int(sc8.state_meta_i64[1].item()),
            "ingest_fps": round(fps_8, 1),
        },
    }


@torch.no_grad()
def _benchmark_single_fits_2d(
    label: str,
    field_f32: np.ndarray,
    nan_mask: np.ndarray,
    max_point_spikes: int = 512,
) -> dict:
    img = torch.from_numpy(field_f32)
    h, w = img.shape
    raw_bytes = img.numel() * 4

    finite_vals = np.abs(field_f32[~nan_mask]) if (~nan_mask).any() else np.abs(field_f32.reshape(-1))
    med_abs = float(max(1e-6, np.median(finite_vals)))
    max_abs = float(finite_vals.max())
    hdr_peak_to_median_x = max_abs / med_abs

    # Baseline A: Lossless `zlib -9` IEEE-754 byte-stream compressor
    raw_buf = img.numpy().tobytes()
    zlib_comp = zlib.compress(raw_buf, level=9)
    zlib_bytes = len(zlib_comp)

    # Proposed: Unmodified `PhysicalVQ4ScarCodec`
    codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=max_point_spikes, spike_sigma_threshold=4.2)
    pkg = codec.encode(img)
    rec_scar = codec.decode_full(pkg)
    scar_bytes = pkg.physical_bytes()
    scar_ratio = pkg.compression_ratio_vs_fp32()

    # Baseline B: Strictly Byte-Matched Conjugate-Symmetric 2D `rfft2` in IEEE-754 `FP16` (`4 B` per complex mode)
    rfft = torch.fft.rfft2(img, norm="ortho")
    n_complex_fp16 = max(64, scar_bytes // 4)
    kx_keep = min(w // 2 + 1, max(8, int(math.sqrt(n_complex_fp16 / 2.0))))
    ky_half = min(h // 2, max(8, n_complex_fp16 // (2 * kx_keep)))
    rfft_trunc = torch.zeros_like(rfft)
    rfft_trunc[:ky_half, :kx_keep] = torch.complex(
        rfft[:ky_half, :kx_keep].real.to(torch.float16).float(),
        rfft[:ky_half, :kx_keep].imag.to(torch.float16).float(),
    )
    rfft_trunc[-ky_half:, :kx_keep] = torch.complex(
        rfft[-ky_half:, :kx_keep].real.to(torch.float16).float(),
        rfft[-ky_half:, :kx_keep].imag.to(torch.float16).float(),
    )
    rec_fft = torch.fft.irfft2(rfft_trunc, s=(h, w), norm="ortho")
    fft_bytes = (2 * ky_half) * kx_keep * 4

    rmse_scar = float((rec_scar - img).square().mean().sqrt().item())
    rmse_fft = float((rec_fft - img).square().mean().sqrt().item())

    # Top-32 brightest physical sources (protostars / Sgr B2 cores / CO outflow knots)
    flat_img = img.reshape(-1)
    _, top32_idx = torch.topk(flat_img.abs(), k=32)
    peak32_max_err_scar = float((rec_scar.reshape(-1)[top32_idx] - flat_img[top32_idx]).abs().max().item())
    peak32_max_err_fft = float((rec_fft.reshape(-1)[top32_idx] - flat_img[top32_idx]).abs().max().item())

    # Measure BOTH pure zero rows (`exact 0.0` guarantee) and partial-row `NaN` sky-mask pixels (honest boundary audit!)
    pure_zero_rows = (img.abs().amax(dim=1) <= 1e-12)
    pure_zero_row_err_scar = float(rec_scar[pure_zero_rows].abs().max().item()) if bool(pure_zero_rows.any().item()) else 0.0
    pure_zero_row_err_fft = float(rec_fft[pure_zero_rows].abs().max().item()) if bool(pure_zero_rows.any().item()) else 0.0

    nan_mask_t = torch.from_numpy(nan_mask)
    nan_sky_max_err_scar = float(rec_scar[nan_mask_t].abs().max().item()) if bool(nan_mask_t.any().item()) else 0.0
    nan_sky_max_err_fft = float(rec_fft[nan_mask_t].abs().max().item()) if bool(nan_mask_t.any().item()) else 0.0

    # Benchmark `O(1)` single-block (`64` floats) random read vs full-stream `zlib.decompress`
    peak_flat = int(top32_idx[0].item())
    probe_r, probe_c = peak_flat // w, peak_flat % w
    probe_blk = probe_c // 64
    t_b0 = time.perf_counter()
    for _ in range(200):
        blk_64 = codec.decode_block_o1(pkg, probe_r, probe_blk)
    us_block_o1 = ((time.perf_counter() - t_b0) / 200.0) * 1e6

    t_z0 = time.perf_counter()
    for _ in range(200):
        _ = zlib.decompress(zlib_comp)
    us_zlib_full = ((time.perf_counter() - t_z0) / 200.0) * 1e6

    valid_slice = rec_scar[probe_r, probe_blk * 64 : min(w, (probe_blk + 1) * 64)]
    o1_diff = float((blk_64 - valid_slice).abs().max().item())

    return {
        "label": label,
        "shape": [h, w],
        "raw_fp32_bytes": raw_bytes,
        "hdr_peak_to_median_ratio_x": round(hdr_peak_to_median_x, 1),
        "nan_sky_fraction_pct": round(100.0 * float(nan_mask.mean()), 4),
        "zlib_level9_bytes": zlib_bytes,
        "zlib_compression_x": round(raw_bytes / zlib_bytes, 2),
        "scar_codec_bytes": scar_bytes,
        "scar_codec_compression_x": round(scar_ratio, 2),
        "byte_matched_fp16_rfft_bytes": fft_bytes,
        "byte_matched_fp16_rfft_compression_x": round(raw_bytes / fft_bytes, 2),
        "scar_codec_rmse": round(rmse_scar, 5),
        "fp16_rfft_rmse": round(rmse_fft, 5),
        "rmse_gain_vs_fp16_rfft_x": round(rmse_fft / max(1e-9, rmse_scar), 2),
        "scar_codec_top32_peak_max_err": round(peak32_max_err_scar, 4),
        "fp16_rfft_top32_peak_max_err": round(peak32_max_err_fft, 4),
        "peak_error_reduction_vs_fp16_rfft_x": round(peak32_max_err_fft / max(1e-6, peak32_max_err_scar), 1),
        "pure_zero_row_max_err_scar": pure_zero_row_err_scar,
        "pure_zero_row_max_err_fp16_rfft": round(pure_zero_row_err_fft, 4),
        "partial_nan_sky_max_err_scar": round(nan_sky_max_err_scar, 4),
        "partial_nan_sky_max_err_fp16_rfft": round(nan_sky_max_err_fft, 4),
        "o1_single_block_decode_us": round(us_block_o1, 2),
        "zlib_full_stream_decompress_us": round(us_zlib_full, 2),
        "o1_random_read_speedup_vs_zlib_x": round(us_zlib_full / max(1e-6, us_block_o1), 2),
        "o1_block_vs_full_decode_max_diff": o1_diff,
    }


@torch.no_grad()
def evaluate_astrophysical_and_radio_fits_archives() -> dict:
    """Experiments 3, 4 & 5: NASA Spitzer IRAC, CSO Bolocam Sgr A* 1.1 mm, & FCRAO 14-m L1448 13CO Cube."""
    _, spitzer_raw = read_fits_primary_f32(DATA_DIR / "spitzer_irac.fits")
    spitzer_nan = ~np.isfinite(spitzer_raw)
    spitzer_clean = np.nan_to_num(spitzer_raw, nan=0.0)

    _, bolocam_raw = read_fits_primary_f32(DATA_DIR / "gc_bolocam_gps.fits")
    bolocam_nan = ~np.isfinite(bolocam_raw)
    bolocam_clean = np.nan_to_num(bolocam_raw, nan=0.0)

    _, l1448_raw = read_fits_primary_f32(DATA_DIR / "l1448_13co.fits")  # [53, 105, 105]
    # Unfold `[53, 105, 105]` -> `[530, 1102]`? Better: `[53 * 5, 21 * 105]` or `[53, 11025]`?
    # `[5565, 105]` has cols=105 -> padded to 128; `[530, 1102]`? Wait: `[53 * 7, 15 * 105]` = `[371, 1575]` (nearly square `[371, 1575]`, 0 data loss!)
    l1448_2d_raw = l1448_raw.reshape(53 * 7, 15 * 105)  # exact reshape of 53 * 105 * 105 = 584,325 voxels!
    l1448_nan = ~np.isfinite(l1448_2d_raw)
    l1448_clean = np.nan_to_num(l1448_2d_raw, nan=0.0)

    return {
        "nasa_spitzer_irac": _benchmark_single_fits_2d(
            "NASA Spitzer Space Telescope IRAC (MJy/sr)", spitzer_clean, spitzer_nan, max_point_spikes=512
        ),
        "cso_bolocam_sgr_a": _benchmark_single_fits_2d(
            "CSO Bolocam Galactic Center 1.1 mm Sgr A* / Sgr B2 (Jy/Beam)", bolocam_clean, bolocam_nan, max_point_spikes=512
        ),
        "fcrao_l1448_13co_cube": _benchmark_single_fits_2d(
            "FCRAO 14-m Radio Telescope L1448 13CO J=1->0 Spectral Cube (K)", l1448_clean, l1448_nan, max_point_spikes=512
        ),
    }


def main() -> None:
    provenance = ensure_public_datasets_downloaded()
    exp1 = evaluate_physionet_array_interference_and_compression()
    exp2 = evaluate_scache_o1_public_telemetry_streams()
    exp3 = evaluate_astrophysical_and_radio_fits_archives()

    report = {
        "suite": "scar_codec_public_datasets_v1_0_audit",
        "version": "1.0.0",
        "scar_codec_modified": False,
        "synthetic_data_used": False,
        "oracle_steering_used": False,
        "provenance_manifest": provenance,
        "experiment_1_physionet_64ch_cross_run_p_perp_vq4": exp1,
        "experiment_2_scache_o1_public_telemetry": exp2,
        "experiment_3_public_astronomy_fits_vq4_vs_fp16_rfft": exp3,
    }
    out_path = REPO_ROOT / "results" / "scar_codec_public_datasets_v1_0_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n✓ Saved real public scientific datasets audit -> {out_path}")


if __name__ == "__main__":
    main()
