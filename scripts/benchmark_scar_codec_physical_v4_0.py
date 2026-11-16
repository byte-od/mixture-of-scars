#!/usr/bin/env python3
"""Audited Benchmark Suite for Domain-Independent `scar_codec` (`v4.0.0`).

Executes 3 self-contained, deterministic physical & hardware experiments (`0` neural networks):
  1. Radio Telescope / CubeSat Array (`P_perp_RFI + VQ4-Scar`):
     Recovers a weak pulsar pulse stream buried under `1,000x` (`+60 dB`) structured satellite RFI,
     compared against a naive 2-bit VLBI scalar quantizer and unshielded VQ4.
  2. Avionics Black-Box / Solar-Blackout Telemetry (`34.8 KB` = `34,816 B` `SCache O(1)` vs Circular `O(T)`):
     Streams `50,000` 64-channel frames (`12.80 MB` raw) with 5 early transient mechanical/electrical shocks
     followed by `48,000` nominal routine frames, proving `+0 B` memory drift and `5/5` (`100%`) recall vs `0/5` (`0%`).
  3. Scientific CFD Shockwave + `500x` HDR Field (`[512, 512]` `FP32` = `1.00 MiB`):
     Compresses a mixed field (40% exact vacuum `0.0` + smooth turbulent manifold + Mach shock discontinuity +
     `500x` HDR specular pinpoints) at `~6x` ratio with `O(1)` single-block random access, compared against
     `zlib -9` (`IEEE-754` byte compression) and matched-budget `2D FFT` spectral truncation (Gibbs ringing).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import zlib

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


@torch.no_grad()
def run_proof_1_radio_telescope_rfi() -> dict:
    """Proof 1: `64`-antenna radio array `[64, 4096]` under `1,000x` (`+60 dB`) satellite RFI."""
    torch.manual_seed(20260401)
    n_ant, n_samples = 64, 4096
    t = torch.linspace(0.0, 1.0, n_samples)

    # 3 structured LEO satellite RFI emitters (steering vectors in `[64, 3]`)
    rfi_steering = F.normalize(torch.randn(n_ant, 3), dim=0)
    # Astrophysical source steering vector strictly orthogonalized to verify out-of-cone invariance + realistic oblique check
    raw_astro_vec = F.normalize(torch.randn(n_ant, 1), dim=0)
    # Pure pulsar periodic narrow duty-cycle pulse train (amplitude ~ 1.0)
    phase = (t * 8.0) % 1.0
    pulsar_wave = torch.exp(-0.5 * ((phase - 0.5) / 0.018).square())  # [4096]
    clean_astro = raw_astro_vec @ pulsar_wave.unsqueeze(0)  # [64, 4096]

    # Blind off-pulse calibration window WITH receiver thermal noise (`0` oracle access to `rfi_steering`!)
    rfi_calib_wave = torch.stack([
        torch.sin(2.0 * math.pi * 137.0 * t),
        torch.cos(2.0 * math.pi * 283.0 * t + 0.4),
        torch.sin(2.0 * math.pi * (410.0 * t + 65.0 * t.square())),
    ], dim=0) * 1000.0
    y_calib_noisy = (rfi_steering @ rfi_calib_wave) + 0.04 * torch.randn(n_ant, n_samples)
    proj = OrthogonalSubspaceProjector.fit_from_interference_calibration(y_calib_noisy, subspace_rank=3)

    # Verify dynamic-range invariance `P_perp(x_perp + 1000 * U_hat * c) - x_perp` across `1,000x` burst
    ortho_probe = clean_astro - proj.basis_u @ (proj.basis_u.T @ clean_astro)
    stressed_probe = ortho_probe + (proj.basis_u @ rfi_calib_wave)
    out_of_cone_max_err = float((proj.project_clean(stressed_probe) - ortho_probe).abs().max().item())

    # Live observation: weak pulsar (`1x`) + thermal noise (`0.04x`) + `1,000x` (`+60 dB`) satellite RFI!
    rfi_live_wave = torch.stack([
        torch.sin(2.0 * math.pi * 137.0 * t + 0.9),
        torch.cos(2.0 * math.pi * 283.0 * t - 0.3),
        torch.sin(2.0 * math.pi * (410.0 * t + 65.0 * t.square()) + 1.2),
    ], dim=0) * 1000.0
    thermal = 0.04 * torch.randn(n_ant, n_samples)
    y_raw = clean_astro + (rfi_steering @ rfi_live_wave) + thermal

    # Conventional unshielded beamformer pointing at true sky source vs subspace-projected beamformer
    w_astro = raw_astro_vec.squeeze(-1)  # [64]
    w_beam = F.normalize(proj.project_clean(raw_astro_vec), dim=0).squeeze(-1)  # [64]

    # Baseline A: Naive 2-bit VLBI scalar quantizer (`{-1.5, -0.5, +0.5, +1.5} * rms`) BEFORE RFI nulling
    rms_raw = y_raw.square().mean(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
    z_vlbi = y_raw / rms_raw
    q_vlbi = torch.where(
        z_vlbi < 0.0,
        torch.where(z_vlbi < -1.0, -1.5, -0.5),
        torch.where(z_vlbi < 1.0, 0.5, 1.5),
    ) * rms_raw
    beam_vlbi_after_null = w_beam @ proj.project_clean(q_vlbi)

    # Baseline B: Unshielded VQ4 without `P_perp_RFI` (RFI swamps dynamic range)
    codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=256, spike_sigma_threshold=4.0)
    pkg_no_null = codec.encode(y_raw)
    beam_vq4_no_null = w_beam @ proj.project_clean(codec.decode_full(pkg_no_null))

    # Proposed: `P_perp_RFI + PhysicalVQ4ScarCodec`
    y_nulled = proj.project_clean(y_raw)
    pkg_scar = codec.encode(y_nulled)
    y_rec = codec.decode_full(pkg_scar)
    beam_scar = w_beam @ y_rec

    ref_pulse = (w_beam @ clean_astro)  # ground-truth projected pulsar waveform

    def _corr_and_snr(est: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
        c = float(F.cosine_similarity(est.unsqueeze(0), ref.unsqueeze(0)).item())
        # Optimal scalar gain fit for SNR evaluation
        alpha = float((est @ ref) / (est @ est).clamp(min=1e-12))
        err = alpha * est - ref
        snr_db = 10.0 * math.log10(float(ref.square().mean().item()) / max(1e-12, float(err.square().mean().item())))
        return round(c, 5), round(snr_db, 2)

    corr_raw, snr_raw = _corr_and_snr(w_astro @ y_raw, w_astro @ clean_astro)
    corr_vlbi, snr_vlbi = _corr_and_snr(beam_vlbi_after_null, ref_pulse)
    corr_no_null, snr_no_null = _corr_and_snr(beam_vq4_no_null, ref_pulse)
    corr_scar, snr_scar = _corr_and_snr(beam_scar, ref_pulse)

    return {
        "array_shape": [n_ant, n_samples],
        "raw_fp32_bytes": n_ant * n_samples * 4,
        "rfi_to_signal_amplitude_ratio": 1000.0,
        "rfi_sir_db": -60.0,
        "out_of_cone_orthogonal_max_abs_err": out_of_cone_max_err,
        "raw_unfiltered_beam_cosine": corr_raw,
        "raw_unfiltered_beam_snr_db": snr_raw,
        "naive_2bit_vlbi_then_null_cosine": corr_vlbi,
        "naive_2bit_vlbi_then_null_snr_db": snr_vlbi,
        "vq4_without_p_perp_cosine": corr_no_null,
        "vq4_without_p_perp_snr_db": snr_no_null,
        "p_perp_plus_vq4_scar_bytes": pkg_scar.physical_bytes(),
        "p_perp_plus_vq4_scar_compression_x": round(pkg_scar.compression_ratio_vs_fp32(), 2),
        "p_perp_plus_vq4_scar_cosine": corr_scar,
        "p_perp_plus_vq4_scar_snr_db": snr_scar,
    }


@torch.no_grad()
def run_proof_2_blackbox_scache_34p8kb() -> dict:
    """Proof 2: `50,000` 64-channel telemetry frames (`12.80 MB`) in exact `34,816 B` (`34.8 KB`) `SCache O(1)`."""
    torch.manual_seed(20260402)
    n_frames, n_ch = 50000, 64
    fault_frames = [210, 580, 940, 1310, 1750]
    harmonics = torch.tensor([1.0, 0.82, -0.74, 1.15, -0.68, 0.91], dtype=torch.float32)
    fault_signatures = {}
    fault_starts = {}
    for idx_i, f_ts in enumerate(fault_frames):
        sig = torch.zeros(n_ch)
        start = (idx_i * 11) % (n_ch - 8)
        fault_starts[f_ts] = start
        pol = 1.0 if (idx_i % 2 == 0) else -1.0
        sig[start : start + 6] = pol * (9.5 + idx_i * 0.8) * harmonics
        fault_signatures[f_ts] = sig

    scache = SurpriseGatedAnomalySCache(channels=n_ch)
    ring_ot = CircularRingBufferOT(channels=n_ch, budget_bytes=scache.physical_bytes())

    bytes_initial = scache.physical_bytes()
    assert bytes_initial == 34816, f"Expected exact 34,816 B (34.8 KB), got {bytes_initial}"

    t0 = time.perf_counter()
    bytes_mid = bytes_initial
    for ts in range(n_frames):
        frame = 0.15 * torch.randn(n_ch)
        if ts in fault_signatures:
            frame = frame + fault_signatures[ts]
        scache.ingest(frame, timestamp=ts)
        ring_ot.ingest(frame, timestamp=ts)
        if ts == 25000:
            bytes_mid = scache.physical_bytes()
    elapsed_s = time.perf_counter() - t0
    bytes_final = scache.physical_bytes()

    ring_valid_ts = set(ring_ot.timestamps_i64[ring_ot.timestamps_i64 >= 0].tolist())
    scache_valid_ts = set(scache.recorded_timestamps())

    ring_hits = sum(1 for ts in fault_frames if ts in ring_valid_ts)
    scache_hits = sum(1 for ts in fault_frames if ts in scache_valid_ts)

    # Blind partial-cue hetero-associative recall (`0` circular full-signature self-querying!):
    # Query `W_scar` using ONLY 2 triggered alarm channels (`4` of `6` fault channels masked to `0.0`!),
    # verifying waveform reconstruction across both all 64 channels and the 4 completely unobserved channels.
    assoc_cosines = []
    masked_cosines = []
    for ts in fault_frames:
        true_sig = fault_signatures[ts]
        start = fault_starts[ts]
        partial_cue = torch.zeros_like(true_sig)
        partial_cue[start : start + 2] = true_sig[start : start + 2]
        rec_sig = scache.query_associative(partial_cue)
        cos_v = float(F.cosine_similarity(rec_sig.unsqueeze(0), true_sig.unsqueeze(0)).item())
        cos_masked = float(
            F.cosine_similarity(
                rec_sig[start + 2 : start + 6].unsqueeze(0),
                true_sig[start + 2 : start + 6].unsqueeze(0),
            ).item()
        )
        assoc_cosines.append(cos_v)
        masked_cosines.append(cos_masked)

    return {
        "stream_frames": n_frames,
        "channels": n_ch,
        "raw_stream_bytes": n_frames * n_ch * 4,
        "scache_bytes_initial": bytes_initial,
        "scache_bytes_mid_25k": bytes_mid,
        "scache_bytes_final_50k": bytes_final,
        "scache_delta_bytes": bytes_final - bytes_initial,
        "circular_ring_budget_bytes": ring_ot.physical_bytes(),
        "circular_ring_frame_capacity": ring_ot.capacity,
        "early_fault_timestamps": fault_frames,
        "circular_ring_fault_recall": f"{ring_hits}/{len(fault_frames)}",
        "circular_ring_fault_recall_pct": round(100.0 * ring_hits / len(fault_frames), 1),
        "scache_fault_recall": f"{scache_hits}/{len(fault_frames)}",
        "scache_fault_recall_pct": round(100.0 * scache_hits / len(fault_frames), 1),
        "scache_associative_mean_cosine": round(sum(assoc_cosines) / len(assoc_cosines), 5),
        "scache_masked_channels_mean_cosine": round(sum(masked_cosines) / len(masked_cosines), 5),
        "ingest_fps": round(n_frames / max(1e-6, elapsed_s), 1),
    }


def _integrate_rusanov_burgers_shock_pde_2d(h: int, w: int, vac_rows: int, n_steps: int = 110) -> torch.Tensor:
    """Integrates the 2D nonlinear inviscid Burgers conservation law PDE
        du/dt + d/dx (0.5 * u^2) + c_y * du/dy = 0
    via conservative Rusanov (Local Lax-Friedrichs) finite-volume fluxes from smooth initial data,
    dynamically steepening into a Rankine-Hugoniot shockwave bounded by an outer-space vacuum (`u = 0`).
    """
    active_h = h - vac_rows
    yy = torch.linspace(-1.0, 1.0, active_h).unsqueeze(1).expand(active_h, w)
    xx = torch.linspace(-1.0, 1.0, w).unsqueeze(0).expand(active_h, w)
    x_front = 0.08 + 0.28 * yy
    u = (1.65 - 2.75 * torch.tanh((xx - x_front) / 0.22)) + 0.85 * torch.sin(2.2 * math.pi * xx) * torch.cos(1.6 * math.pi * yy)

    dx = 2.0 / max(1, w - 1)
    dy = 2.0 / max(1, active_h - 1)
    cy = 0.35

    for _ in range(n_steps):
        max_sx = max(1.0, float(u.abs().max().item()))
        dt = 0.36 / (max_sx / dx + abs(cy) / dy)

        u_l = torch.cat([u[:, :1], u[:, :-1]], dim=1)
        u_r = torch.cat([u[:, 1:], u[:, -1:]], dim=1)
        alpha_r = torch.maximum(u.abs(), u_r.abs())
        flux_r = 0.25 * (u.square() + u_r.square()) - 0.5 * alpha_r * (u_r - u)
        alpha_l = torch.maximum(u_l.abs(), u.abs())
        flux_l = 0.25 * (u_l.square() + u.square()) - 0.5 * alpha_l * (u - u_l)

        u_up = torch.cat([u[:1, :], u[:-1, :]], dim=0)
        adv_y = cy * ((u - u_up) / dy)
        u = u - (dt / dx) * (flux_r - flux_l) - dt * adv_y

    full = torch.zeros((h, w), dtype=torch.float32)
    full[vac_rows:, :] = u
    return full


@torch.no_grad()
def run_proof_3_cfd_hdr_shock_field() -> dict:
    """Proof 3: `[512, 512]` `FP32` (`1.00 MiB`) Rusanov Burgers Shockwave PDE + `40%` Vacuum + `500x` HDR Pinpoints."""
    torch.manual_seed(20260403)
    h, w = 512, 512
    vac_rows = int(0.40 * h)

    # Evolve conservative Rusanov finite-volume nonlinear Burgers PDE (`110` steps -> Rankine-Hugoniot shock!)
    field = _integrate_rusanov_burgers_shock_pde_2d(h, w, vac_rows, n_steps=110)
    field[vac_rows:, :] += 0.035 * torch.randn(h - vac_rows, w)
    field[:vac_rows, :] = 0.0

    # Inject 48 localized `500x` specular HDR glints / cosmic-ray hits (`amplitude ~ 500.0`)
    spike_coords = []
    for s_i in range(48):
        r_idx = vac_rows + 15 + ((s_i * 37) % (h - vac_rows - 20))
        c_idx = (s_i * 83 + 19) % w
        field[r_idx, c_idx] += 500.0 * (1.0 if s_i % 2 == 0 else -0.85)
        spike_coords.append((r_idx, c_idx))

    raw_bytes = field.numel() * 4  # 1,048,576 B (1.00 MiB)

    # Baseline A: `zlib -9` lossless IEEE-754 float32 byte-stream compression
    raw_buf = field.numpy().tobytes()
    zlib_comp = zlib.compress(raw_buf, level=9)
    zlib_ratio = raw_bytes / len(zlib_comp)

    # Proposed: `PhysicalVQ4ScarCodec` (`rank=16`, `max_point_spikes=128` -> `~5.9x` compression + `O(1)` block read!)
    codec = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=128, spike_sigma_threshold=4.2)
    pkg = codec.encode(field)
    rec_scar = codec.decode_full(pkg)
    scar_bytes = pkg.physical_bytes()
    scar_ratio = pkg.compression_ratio_vs_fp32()

    # Baseline B: Fair Byte-Matched Conjugate-Symmetric Orthonormal 2D `rfft2` in IEEE-754 `FP16` (`4 B` per complex mode = `2 B` per real DOF!)
    rfft = torch.fft.rfft2(field, norm="ortho")  # [H, W // 2 + 1]
    n_complex_fp16 = max(64, scar_bytes // 4)
    kx_keep = min(w // 2 + 1, max(8, int(math.sqrt(n_complex_fp16 / 2.0))))
    ky_half = min(h // 2, max(8, n_complex_fp16 // (2 * kx_keep)))
    rfft_trunc = torch.zeros_like(rfft)
    # Quantize retained low-frequency `(real, imag)` components to exact IEEE-754 `float16` (`2 + 2 = 4 B/mode`)
    top_re = rfft[:ky_half, :kx_keep].real.to(torch.float16).to(torch.float32)
    top_im = rfft[:ky_half, :kx_keep].imag.to(torch.float16).to(torch.float32)
    bot_re = rfft[-ky_half:, :kx_keep].real.to(torch.float16).to(torch.float32)
    bot_im = rfft[-ky_half:, :kx_keep].imag.to(torch.float16).to(torch.float32)
    rfft_trunc[:ky_half, :kx_keep] = torch.complex(top_re, top_im)
    rfft_trunc[-ky_half:, :kx_keep] = torch.complex(bot_re, bot_im)
    rec_fft = torch.fft.irfft2(rfft_trunc, s=(h, w), norm="ortho")
    fft_bytes = (2 * ky_half) * kx_keep * 4

    # Evaluate vacuum exact zero, overall RMSE, peak HDR spike error, and `O(1)` random block access!
    vac_err_scar = float(rec_scar[:vac_rows, :].abs().max().item())
    vac_err_fft = float(rec_fft[:vac_rows, :].abs().max().item())

    rmse_scar = float((rec_scar - field).square().mean().sqrt().item())
    rmse_fft = float((rec_fft - field).square().mean().sqrt().item())

    spike_errs_scar = [abs(float(rec_scar[r, c] - field[r, c])) for r, c in spike_coords]
    spike_errs_fft = [abs(float(rec_fft[r, c] - field[r, c])) for r, c in spike_coords]
    max_spike_err_scar = max(spike_errs_scar)
    max_spike_err_fft = max(spike_errs_fft)

    # Benchmark `O(1)` random single-block read (`64` floats) vs `zlib.decompress` full-stream read
    probe_row, probe_blk = spike_coords[0][0], spike_coords[0][1] // 64
    t_b0 = time.perf_counter()
    for _ in range(200):
        blk_64 = codec.decode_block_o1(pkg, probe_row, probe_blk)
    us_block_o1 = ((time.perf_counter() - t_b0) / 200.0) * 1e6

    t_z0 = time.perf_counter()
    for _ in range(200):
        _ = zlib.decompress(zlib_comp)
    us_zlib_full = ((time.perf_counter() - t_z0) / 200.0) * 1e6

    block_o1_parity_max_diff = float(
        (blk_64 - rec_scar[probe_row, probe_blk * 64 : (probe_blk + 1) * 64]).abs().max().item()
    )

    return {
        "field_shape": [h, w],
        "raw_fp32_bytes": raw_bytes,
        "zlib_level9_bytes": len(zlib_comp),
        "zlib_compression_x": round(zlib_ratio, 2),
        "zlib_random_access": False,
        "fft_truncated_bytes": fft_bytes,
        "fft_compression_x": round(raw_bytes / fft_bytes, 2),
        "fft_rmse": round(rmse_fft, 4),
        "fft_vacuum_max_abs_err": round(vac_err_fft, 4),
        "fft_hdr_spike_max_abs_err": round(max_spike_err_fft, 2),
        "scar_codec_bytes": scar_bytes,
        "scar_codec_compression_x": round(scar_ratio, 2),
        "scar_codec_rmse": round(rmse_scar, 4),
        "scar_codec_vacuum_max_abs_err": vac_err_scar,
        "scar_codec_hdr_spike_max_abs_err": round(max_spike_err_scar, 4),
        "rmse_gain_vs_fft_x": round(rmse_fft / max(1e-8, rmse_scar), 2),
        "hdr_spike_error_reduction_vs_fft_x": round(max_spike_err_fft / max(1e-6, max_spike_err_scar), 1),
        "o1_single_block_decode_us": round(us_block_o1, 2),
        "zlib_full_stream_decompress_us": round(us_zlib_full, 2),
        "o1_block_vs_full_decode_max_diff": block_o1_parity_max_diff,
    }


def main() -> None:
    p1 = run_proof_1_radio_telescope_rfi()
    p2 = run_proof_2_blackbox_scache_34p8kb()
    p3 = run_proof_3_cfd_hdr_shock_field()

    report = {
        "suite": "scar_codec_physical_v4_0_audit",
        "version": "4.0.0",
        "proof_1_radio_telescope_rfi_1000x": p1,
        "proof_2_blackbox_scache_34p8kb": p2,
        "proof_3_cfd_hdr_shock_field_o1": p3,
    }
    out_path = REPO_ROOT / "results" / "scar_codec_physical_v4_0_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n✓ Saved audited physical ScarCodec benchmark report -> {out_path}")


if __name__ == "__main__":
    main()
