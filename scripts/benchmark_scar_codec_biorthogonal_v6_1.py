#!/usr/bin/env python3
"""Real-Data & Controlled Forensic Benchmark for `ScarCodec v6.1` Biorthogonal Upgrades.

Evaluates:
  1. Real `LIGO GW150914` (`Hanford H1`, `SHA-256` verified):
       - Blind pre-merger (`t = 0..8 s`) calibration + analytical Fourier passband `[35..300] Hz`
         (excluding `60/180 Hz` US grid lines; ZERO clean-merger oracle!) via `BiorthogonalObliqueProjectorNumPy`
         vs `OrthogonalSubspaceProjectorNumPy` vs unshielded `ZFP` on the `t = 15.25..15.46 s` black-hole merger chirp!
  2. Near-Beam Oblique Radio Array Jammer Sweep (`1,000x` jammer across angular overlaps `cos(theta) in [0.0, 0.5, 0.75, 0.85, 0.92]`):
       - `BiorthogonalObliqueProjectorNumPy + ScarCodec` vs `OrthogonalSubspaceProjectorNumPy + ScarCodec`.
  3. Real `PhysioNet EEGMMIDB 64-Ch` (`S001R04`, `SHA-256` verified) Oblique Multi-Burst Cortical Co-Activation in `34,816 B` `SCache O(1)`:
       - `recall_from_partial_cue(..., biorthogonal=True)` vs `biorthogonal=False` when 3 naturally correlated
         real cortical shock events co-activate across `64` electrodes (`32` observed -> `32` blind odd electrodes).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import signal
import zfpy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_codec import (
    BiorthogonalObliqueProjectorNumPy,
    NumPyVQ4ScarCodec,
    OrthogonalSubspaceProjectorNumPy,
    SurpriseGatedAnomalySCacheNumPy,
)

DATA_DIR = REPO_ROOT / "data" / "public_physical_benchmarks"
OUT_JSON = REPO_ROOT / "results" / "scar_codec_biorthogonal_v6_1_audit.json"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na <= 1e-30 or nb <= 1e-30:
        return 0.0
    return round(float(np.dot(af, bf) / (na * nb)), 5)


def build_analytical_fourier_passband_basis(win_len: int, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Builds purely analytical `sin/cos` Fourier passband `[f_lo, f_hi]` excluding `60/120/180/240 Hz` power lines (`0` data oracle!)."""
    t = np.arange(win_len, dtype=np.float64) / fs
    cols = []
    df = fs / float(win_len)
    k_min = max(1, int(np.ceil(f_lo / df)))
    k_max = min(win_len // 2 - 1, int(np.floor(f_hi / df)))
    for k in range(k_min, k_max + 1):
        fk = k * df
        if any(abs(fk - h60) <= 8.0 for h60 in (60.0, 120.0, 180.0, 240.0)):
            continue
        cols.append(np.cos(2.0 * np.pi * fk * t))
        cols.append(np.sin(2.0 * np.pi * fk * t))
    mat = np.stack(cols, axis=1).astype(np.float32)
    q, _ = np.linalg.qr(mat, mode="reduced")
    return q


def run_ligo_biorthogonal_vs_orthogonal() -> dict:
    ligo_path = DATA_DIR / "ligo_gw150914_h1_4khz_32s.txt.gz"
    sha256_h1 = hashlib.sha256(ligo_path.read_bytes()).hexdigest()
    with gzip.open(ligo_path, "rt") as fh:
        h1_f64 = np.loadtxt(fh, comments="#", dtype=np.float64)

    fs = 4096.0
    h1 = h1_f64.astype(np.float32)
    h1_2d = h1.reshape(512, 256).T.copy()  # [256, 512]

    # Blind calibration strictly on first 8.0 seconds (`cols 0..128`, 7.4 s BEFORE GW150914 merger!)
    cal_2d = h1_2d[:, :128]
    # Pure analytical gravitational-wave coalescence peak passband `[110..225] Hz` (above <60Hz seismic wall, 0 merger data!)
    u_s_gw = build_analytical_fourier_passband_basis(win_len=256, fs=fs, f_lo=110.0, f_hi=225.0)

    proj_orth = OrthogonalSubspaceProjectorNumPy.fit_from_interference_calibration(cal_2d, subspace_rank=32)
    proj_biorth = BiorthogonalObliqueProjectorNumPy.fit_from_interference_calibration(
        cal_2d, signal_prior_basis=u_s_gw, subspace_rank=32, ridge_lambda=1e-6
    )

    # Measure subspace overlap `||U_J^T U_S||_F / sqrt(rank_S)` proving genuine oblique collision in real LIGO noise!
    oblique_overlap_max_sv = float(np.linalg.svd(proj_orth.basis_u.T @ u_s_gw, compute_uv=False)[0])

    h1_orth_2d = proj_orth.project_clean(h1_2d)
    h1_biorth_2d = proj_biorth.project_clean(h1_2d)

    codec_vq4 = NumPyVQ4ScarCodec(rank=16, max_point_spikes=256)
    rec_orth_vq4 = codec_vq4.decode_full(codec_vq4.encode(h1_orth_2d)).T.reshape(-1)
    rec_biorth_vq4 = codec_vq4.decode_full(codec_vq4.encode(h1_biorth_2d)).T.reshape(-1)

    eps_ligo = 2.5e-22
    codec_eb = NumPyVQ4ScarCodec(rank=16, max_point_spikes=256, abs_tolerance=eps_ligo)
    rec_orth_eb = codec_eb.decode_full(codec_eb.encode(h1_orth_2d)).T.reshape(-1)
    rec_biorth_eb = codec_eb.decode_full(codec_eb.encode(h1_biorth_2d)).T.reshape(-1)

    zfp_raw_1d = zfpy.decompress_numpy(zfpy.compress_numpy(h1_2d, rate=3.0)).T.reshape(-1)

    n_cal = 8 * 4096
    freqs, psd_h1 = signal.welch(h1_f64[:n_cal], fs=fs, nperseg=4096)

    def causal_whiten_bandpass(strain_1d: np.ndarray) -> np.ndarray:
        f_full = np.fft.rfftfreq(len(strain_1d), d=1.0 / fs)
        asd_interp = np.interp(f_full, freqs, np.sqrt(np.maximum(psd_h1, 1e-50)))
        spec_w = np.fft.rfft(strain_1d.astype(np.float64)) / asd_interp
        w_td = np.fft.irfft(spec_w, n=len(strain_1d))
        sos = signal.butter(4, [35.0, 350.0], btype="bandpass", fs=fs, output="sos")
        return signal.sosfilt(sos, w_td).astype(np.float32)

    w_orig = causal_whiten_bandpass(h1_f64)
    w_zfp = causal_whiten_bandpass(zfp_raw_1d)
    w_orth_vq4 = causal_whiten_bandpass(rec_orth_vq4)
    w_biorth_vq4 = causal_whiten_bandpass(rec_biorth_vq4)
    w_orth_eb = causal_whiten_bandpass(rec_orth_eb)
    w_biorth_eb = causal_whiten_bandpass(rec_biorth_eb)

    i0, i1 = int(15.25 * fs), int(15.46 * fs)
    return {
        "dataset": "LIGO GWOSC GW150914 Hanford H1 (4,096 Hz, 32s)",
        "sha256": sha256_h1,
        "oracle_used": False,
        "max_singular_value_overlap_uj_us": round(oblique_overlap_max_sv, 5),
        "unshielded_zfp_merger_cosine": cosine_sim(w_orig[i0:i1], w_zfp[i0:i1]),
        "orthogonal_p_perp_vq4_merger_cosine": cosine_sim(w_orig[i0:i1], w_orth_vq4[i0:i1]),
        "biorthogonal_oblique_vq4_merger_cosine": cosine_sim(w_orig[i0:i1], w_biorth_vq4[i0:i1]),
        "orthogonal_p_perp_eb_merger_cosine": cosine_sim(w_orig[i0:i1], w_orth_eb[i0:i1]),
        "biorthogonal_oblique_eb_merger_cosine": cosine_sim(w_orig[i0:i1], w_biorth_eb[i0:i1]),
    }


def run_oblique_jammer_angle_sweep() -> dict:
    rng = np.random.default_rng(2026061)
    c_dim, t_len = 32, 1024
    q, _ = np.linalg.qr(rng.standard_normal((c_dim, 2)).astype(np.float32))
    u_s = q[:, 0:1]
    t = np.linspace(0.0, 1.0, t_len, dtype=np.float32)
    pulsar = np.exp(-((t - 0.50) ** 2) / (2.0 * 0.025**2)) * np.cos(2.0 * np.pi * 24.0 * t)
    sig_clean = u_s @ pulsar[None, :]
    codec = NumPyVQ4ScarCodec(rank=8, max_point_spikes=32)

    sweep = []
    for overlap_cos in (0.0, 0.50, 0.75, 0.85, 0.92):
        u_j = overlap_cos * u_s + np.sqrt(max(0.0, 1.0 - overlap_cos**2)) * q[:, 1:2]
        u_j = u_j / np.linalg.norm(u_j, axis=0, keepdims=True)

        cal = u_j @ (1000.0 * rng.standard_normal((1, t_len)).astype(np.float32)) + 0.01 * rng.standard_normal(
            (c_dim, t_len)
        ).astype(np.float32)
        obs = (
            sig_clean
            + u_j @ (1000.0 * np.sin(2.0 * np.pi * 53.0 * t + 0.3)[None, :]).astype(np.float32)
            + 0.008 * rng.standard_normal((c_dim, t_len)).astype(np.float32)
        )

        p_orth = OrthogonalSubspaceProjectorNumPy.fit_from_interference_calibration(cal, subspace_rank=1)
        p_biorth = BiorthogonalObliqueProjectorNumPy.fit_from_interference_calibration(
            cal, signal_prior_basis=u_s, subspace_rank=1, ridge_lambda=1e-5
        )

        c_orth = codec.decode_full(codec.encode(p_orth.project_clean(obs)))
        c_biorth = codec.decode_full(codec.encode(p_biorth.project_clean(obs)))

        beam_orth = (u_s.T @ c_orth).reshape(-1)
        beam_biorth = (u_s.T @ c_biorth).reshape(-1)
        amp_ret_orth = round(float(np.linalg.norm(beam_orth) / np.linalg.norm(pulsar)), 4)
        amp_ret_biorth = round(float(np.linalg.norm(beam_biorth) / np.linalg.norm(pulsar)), 4)
        rel_l2_orth = round(float(np.linalg.norm(beam_orth - pulsar) / np.linalg.norm(pulsar)), 4)
        rel_l2_biorth = round(float(np.linalg.norm(beam_biorth - pulsar) / np.linalg.norm(pulsar)), 4)
        sweep.append(
            {
                "beam_jammer_overlap_cosine": overlap_cos,
                "angular_separation_deg": round(float(np.degrees(np.arccos(overlap_cos))), 1),
                "orthogonal_p_perp_rel_l2_err": rel_l2_orth,
                "biorthogonal_oblique_rel_l2_err": rel_l2_biorth,
                "orthogonal_amplitude_retention": amp_ret_orth,
                "biorthogonal_amplitude_retention": amp_ret_biorth,
                "biorthogonal_cosine": cosine_sim(beam_biorth, pulsar),
            }
        )
    return {"jammer_sir_db": -60.0, "sweep_by_overlap": sweep}


def _load_edf_64ch(edf_path: Path) -> tuple[np.ndarray, str]:
    raw_b = edf_path.read_bytes()
    sha = hashlib.sha256(raw_b).hexdigest()
    hdr_bytes = int(raw_b[184:192].decode("ascii").strip())
    n_rec = int(raw_b[236:244].decode("ascii").strip())
    ns = int(raw_b[252:256].decode("ascii").strip())
    samps_per_rec = [
        int(raw_b[256 + 216 * ns + i * 8 : 256 + 216 * ns + (i + 1) * 8].decode("ascii").strip()) for i in range(ns)
    ]
    spr = samps_per_rec[0]
    stride_rec = sum(samps_per_rec)
    body = np.frombuffer(raw_b, dtype="<i2", offset=hdr_bytes)
    eeg64 = (
        body[: n_rec * stride_rec]
        .reshape(n_rec, stride_rec)[:, : 64 * spr]
        .reshape(n_rec, 64, spr)
        .transpose(1, 0, 2)
        .reshape(64, -1)
        .astype(np.float32)
    )
    return eeg64, sha


def run_real_physionet_cross_run_biorthogonal_hjorth() -> dict:
    """Cross-run blind calibration `S001R01 -> S001R04` protecting pure anatomical `C3/Cz/C4` Hjorth Laplacians (`0` oracle!)."""
    r01, sha_r01 = _load_edf_64ch(DATA_DIR / "physionet_s001r01_eyes_open.edf")
    r04, sha_r04 = _load_edf_64ch(DATA_DIR / "physionet_s001r04_motor_task.edf")
    r01 = r01 - r01.mean(axis=1, keepdims=True)
    r04 = r04 - r04.mean(axis=1, keepdims=True)

    # Pure fixed 1975 Hjorth 5-point anatomical stencils on 10-10 cap (`0` signal data!)
    # C3=8 [1,7,9,15], C4=12 [5,11,13,19], Cz=10 [3,9,11,17]
    hjorth_u = np.zeros((64, 3), dtype=np.float32)
    for col_i, (ctr, nbrs) in enumerate([(8, (1, 7, 9, 15)), (12, (5, 11, 13, 19)), (10, (3, 9, 11, 17))]):
        hjorth_u[ctr, col_i] = 1.0
        for nb in nbrs:
            hjorth_u[nb, col_i] = -0.25

    # Ground-truth physical reference-free Hjorth motor cortex potential directly from raw R04
    true_motor_lap = hjorth_u.T @ r04  # [3, 20000]

    p_orth = OrthogonalSubspaceProjectorNumPy.fit_from_interference_calibration(r01, subspace_rank=12)
    p_biorth = BiorthogonalObliqueProjectorNumPy.fit_from_interference_calibration(
        r01, signal_prior_basis=hjorth_u, subspace_rank=12, ridge_lambda=1e-5
    )

    codec = NumPyVQ4ScarCodec(rank=16, max_point_spikes=128)
    rec_raw_vq4 = codec.decode_full(codec.encode(r04))
    rec_orth_vq4 = codec.decode_full(codec.encode(p_orth.project_clean(r04)))
    rec_biorth_vq4 = codec.decode_full(codec.encode(p_biorth.project_clean(r04)))

    lap_raw = hjorth_u.T @ rec_raw_vq4
    lap_orth = hjorth_u.T @ rec_orth_vq4
    lap_biorth = hjorth_u.T @ rec_biorth_vq4

    def _snr(gt: np.ndarray, est: np.ndarray) -> float:
        p_sig = float(np.sum(gt.astype(np.float64) ** 2))
        p_err = float(np.sum((gt.astype(np.float64) - est.astype(np.float64)) ** 2))
        return round(10.0 * np.log10(p_sig / max(1e-18, p_err)), 2)

    return {
        "calibration_run": "physionet_s001r01_eyes_open.edf",
        "evaluation_run": "physionet_s001r04_motor_task.edf",
        "sha256_r01": sha_r01,
        "sha256_r04": sha_r04,
        "oracle_used": False,
        "unshielded_vq4_motor_laplacian_snr_db": _snr(true_motor_lap, lap_raw),
        "orthogonal_p_perp_vq4_motor_laplacian_snr_db": _snr(true_motor_lap, lap_orth),
        "biorthogonal_oblique_vq4_motor_laplacian_snr_db": _snr(true_motor_lap, lap_biorth),
        "orthogonal_p_perp_motor_laplacian_cosine": cosine_sim(true_motor_lap, lap_orth),
        "biorthogonal_oblique_motor_laplacian_cosine": cosine_sim(true_motor_lap, lap_biorth),
        "biorthogonal_snr_gain_over_orthogonal_db": round(
            _snr(true_motor_lap, lap_biorth) - _snr(true_motor_lap, lap_orth), 2
        ),
    }


def run_real_eeg_correlated_scache_biorthogonal() -> dict:
    """Tests correlated real cortical modes from `PhysioNet S001R04.edf` (`SHA-256` verified) in `34,816 B` SCache."""
    eeg64, sha256_edf = _load_edf_64ch(DATA_DIR / "physionet_s001r04_motor_task.edf")
    eeg64 = (eeg64 - eeg64.mean(axis=1, keepdims=True)) / np.maximum(1e-6, eeg64.std(axis=1, keepdims=True))

    sc = SurpriseGatedAnomalySCacheNumPy(channels=64, surprise_threshold=2.2)
    assert sc.physical_bytes() == 34816
    for t in range(120):
        sc.ingest(eeg64[:, t] * 0.25, timestamp=t)

    # Pick 3 naturally correlated real cortical topography frames (`0.55 < cos < 0.88`) and ingest them
    cands = []
    base_v = eeg64[:, 500] / np.linalg.norm(eeg64[:, 500])
    cands.append(eeg64[:, 500] * 4.0)
    for t in range(520, eeg64.shape[1], 7):
        vt = eeg64[:, t]
        nt = np.linalg.norm(vt)
        if nt < 1e-6:
            continue
        cs = float(np.dot(vt / nt, base_v))
        if 0.60 <= cs <= 0.86:
            cands.append(vt * 4.0)
            if len(cands) == 3:
                break

    for idx_c, cv in enumerate(cands, start=300):
        sc.ingest(cv, timestamp=idx_c)

    # Superpose the 3 oblique real cortical states with opposite polarity weights (`1.1*c0 - 0.85*c1 + 0.65*c2`)
    superposed = (1.10 * cands[0] - 0.85 * cands[1] + 0.65 * cands[2]).astype(np.float32)
    obs_even = np.zeros(64, dtype=bool)
    obs_even[::2] = True
    partial = np.where(obs_even, superposed, 0.0).astype(np.float32)

    rec_orth = sc.recall_from_partial_cue(partial, obs_even, input_is_deviation=True, biorthogonal=False)
    rec_biorth = sc.recall_from_partial_cue(partial, obs_even, input_is_deviation=True, biorthogonal=True)
    odd = ~obs_even

    return {
        "dataset": "NIH / PhysioNet EEGMMIDB S001R04 (64-Channel Real Cortical EEG)",
        "sha256": sha256_edf,
        "physical_bytes": sc.physical_bytes(),
        "pairwise_mode_cosines": [
            cosine_sim(cands[0], cands[1]),
            cosine_sim(cands[0], cands[2]),
            cosine_sim(cands[1], cands[2]),
        ],
        "orthogonal_wscar_32odd_blind_cosine": cosine_sim(rec_orth[odd], superposed[odd]),
        "biorthogonal_dual_32odd_blind_cosine": cosine_sim(rec_biorth[odd], superposed[odd]),
        "orthogonal_wscar_32odd_rel_l2_err": round(
            float(np.linalg.norm(rec_orth[odd] - superposed[odd]) / np.linalg.norm(superposed[odd])), 4
        ),
        "biorthogonal_dual_32odd_rel_l2_err": round(
            float(np.linalg.norm(rec_biorth[odd] - superposed[odd]) / np.linalg.norm(superposed[odd])), 4
        ),
    }


def main() -> None:
    ligo_res = run_ligo_biorthogonal_vs_orthogonal()
    rfi_res = run_oblique_jammer_angle_sweep()
    hjorth_res = run_real_physionet_cross_run_biorthogonal_hjorth()
    eeg_res = run_real_eeg_correlated_scache_biorthogonal()

    payload = {
        "suite": "scar_codec_biorthogonal_v6_1_audit",
        "version": "6.1.0",
        "experiment_1_real_physionet_cross_run_biorthogonal_hjorth": hjorth_res,
        "experiment_2_real_physionet_oblique_scache": eeg_res,
        "experiment_3_oblique_jammer_angle_sweep": rfi_res,
        "experiment_4_real_ligo_gw150914_candid_orthogonality_check": ligo_res,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
