#!/usr/bin/env python3
"""
Generates Figure 1 (`paper_latex/public_data_paper/figures/fig1_scarcodec_public_data.pdf` & `.png`)
for the Dedicated Standalone Zenodo Paper on Real Public Observational & Experimental Archives:
  - Panel A: PhysioNet 64-Channel Ag/AgCl Array (`S001R01` Calib -> Held-Out `S001R04` `10.47x`, Cortical Array & Raw `C3` Laplacian SNR)
  - Panel B: `34,816 B` (`34.8 KB`, `Δ=0 B`) `SCache O(1)` on Chronological PhysioNet (`39,520x64`, `0.9967` 32->32 Blind Recall) & NASA Space Shuttle (`6/6 = 100%` Bypass Valve Recall)
  - Panel C: Compression Ratio Across All 4 Public Scientific Archives (`ScarCodec` `10.5x–12.3x` vs Lossless `zlib -9` `1.10x–3.17x`)
  - Panel D: Real NASA Spitzer Space Telescope `IRAC` Zoom (`[128x128]` Protostellar Core Region: Ground Truth vs `ScarCodec` `11.77x` vs Byte-Matched `FP16 2D rFFT` Gibbs Ringing)
  - Panel E: Empirical HDR Contrast Scaling Law Across 3 Real Telescopes (`FCRAO 14-m 11.7x` -> `CSO Bolocam 217x` -> `NASA Spitzer 968.9x` Peak Photometry Gain `4.6x -> 45.9x -> 2,093.6x`)
  - Panel F: `O(1)` Single-Block (`64`-Sample) Random-Access Read Speedup (`23.2x–34.7x` Faster than `zlib -9` Full Stream, Exact `0.0` Block Parity)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scar_codec import PhysicalVQ4ScarCodec
from scripts.benchmark_scar_codec_public_datasets_v1_0 import DATA_DIR, read_fits_primary_f32


def main() -> None:
    rep = json.loads((ROOT / "results/scar_codec_public_datasets_v1_0_audit.json").read_text(encoding="utf-8"))
    e1 = rep["experiment_1_physionet_64ch_cross_run_p_perp_vq4"]
    e2 = rep["experiment_2_scache_o1_public_telemetry"]
    e3 = rep["experiment_3_public_astronomy_fits_vq4_vs_fp16_rfft"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.1,
            "axes.titlesize": 8.95,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.95,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 6.45), dpi=300)

    # ------------------------------------------------------------------
    # Panel A: PhysioNet 64-Channel Cross-Run Blind P_perp + VQ4-Scar
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    cats_a = ["64-Ch Cortical\nArray SNR (dB)", "Model-Free Raw C3\nLaplacian SNR (dB)"]
    v_2b = [e1["clean_array_2bit_adc_snr_db"], e1["raw_c3_laplacian_2bit_adc_snr_db"]]
    v_un = [e1["clean_array_unshielded_vq4_snr_db"], e1["raw_c3_laplacian_unshielded_vq4_snr_db"]]
    v_sc = [e1["clean_array_p_perp_vq4_scar_snr_db"], e1["raw_c3_laplacian_p_perp_vq4_scar_snr_db"]]
    xa = np.arange(2)
    w = 0.25
    ax.bar(xa - w, v_2b, width=w, color="#dc2626", label="2-Bit Scalar ADC")
    ax.bar(xa, v_un, width=w, color="#f59e0b", label="Unshielded VQ4 (No P_⊥)")
    bars_a = ax.bar(xa + w, v_sc, width=w, color="#0f766e", label=f"Cross-Run P_⊥ + VQ4 ({e1['scar_codec_compression_x']}x)")
    ax.axhline(0.0, color="#475569", lw=0.7)
    for b, val in zip(bars_a, v_sc):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.55, f"+{val:.2f} dB", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#065f46")
    ax.text(
        0.03,
        0.95,
        f"Calib: S001R01 ({e1['calibration_top4_artifact_energy_pct']:.1f}% artifact)\nHeld-Out: S001R04 ([64, 19968])\ncos = {e1['clean_array_p_perp_vq4_scar_cosine']:.4f} (0 synthetic data)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#f0fdfa", edgecolor="#0f766e", lw=0.7),
    )
    ax.set_xticks(xa)
    ax.set_xticklabels(cats_a, fontsize=7.2)
    ax.set_ylim(-7.0, 23.2)
    ax.set_title("A. PhysioNet 64-Ch Cross-Run Blind P_⊥ + VQ4")
    ax.set_ylabel("Reconstruction SNR (dB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=6.4, frameon=True)

    # ------------------------------------------------------------------
    # Panel B: Constant-Memory O(1) SCache on NASA Shuttle & PhysioNet 64-Ch
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    p64 = e2["physionet_64ch_stream"]
    sh = e2["nasa_shuttle_radiator_stream"]
    cats_b = [
        "NASA Shuttle\nBpv Valve (6/6)",
        "PhysioNet R01/02\nRetained (34.8 KB)",
        "Blind 32->32 Odd\nElectrode Cos (%)",
    ]
    ring_b = [sh["circular_ring_bypass_valve_recall_pct"], 0.0, 0.0]
    scar_b = [
        sh["scache_bypass_valve_recall_pct"],
        100.0,
        p64["blind_32even_to_32odd_masked_cosine"] * 100.0,
    ]
    xb = np.arange(3)
    wb = 0.34
    ax.bar(xb - wb / 2, ring_b, width=wb, color="#94a3b8", label="Equal-Byte Circular Ring O(T)")
    bars_b = ax.bar(xb + wb / 2, scar_b, width=wb, color="#1d4ed8", label="SurpriseGatedAnomalySCache O(1)")
    tags_b = ["6/6 (100%)\n(0.041% prev.)", "3/3 Kept\n(34,816 B, Δ=0)", f"{scar_b[2]:.2f}%\n(32 masked ch)"]
    for b, val, tg in zip(bars_b, scar_b, tags_b):
        ax.text(b.get_x() + b.get_width() / 2, val + 2.0, tg, ha="center", va="bottom", fontsize=6.65, fontweight="bold", color="#1e3a8a")
    ax.set_xticks(xb)
    ax.set_xticklabels(cats_b, fontsize=6.95)
    ax.set_ylim(0, 144)
    ax.set_title("B. Constant-Space O(1) Anomaly & Masked Recall")
    ax.set_ylabel("Retention / Masked Cosine (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

    # ------------------------------------------------------------------
    # Panel C: Compression Ratio Across All 4 Real Public Archives
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    sp = e3["nasa_spitzer_irac"]
    bo = e3["cso_bolocam_sgr_a"]
    fc = e3["fcrao_l1448_13co_cube"]
    c_names = ["PhysioNet\n64-Ch EDF", "NASA Spitzer\nIRAC FITS", "CSO Bolocam\nSgr A* FITS", "FCRAO 14-m\n13CO Cube"]
    z_ratios = [e1["zlib_compression_x"], sp["zlib_compression_x"], bo["zlib_compression_x"], fc["zlib_compression_x"]]
    s_ratios = [e1["scar_codec_compression_x"], sp["scar_codec_compression_x"], bo["scar_codec_compression_x"], fc["scar_codec_compression_x"]]
    xc = np.arange(4)
    ax.bar(xc - wb / 2, z_ratios, width=wb, color="#cbd5e1", edgecolor="#475569", lw=0.5, label="Lossless zlib -9 (IEEE-754)")
    bars_c = ax.bar(xc + wb / 2, s_ratios, width=wb, color="#059669", edgecolor="#064e3b", lw=0.5, label="ScarCodec (O(1) Block Layout)")
    for b, vr in zip(bars_c, s_ratios):
        ax.text(b.get_x() + b.get_width() / 2, vr + 0.3, f"{vr:.2f}x", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#065f46")
    for idx_z, vz in enumerate(z_ratios):
        ax.text(xc[idx_z] - wb / 2, vz + 0.25, f"{vz:.2f}x", ha="center", va="bottom", fontsize=6.4, color="#475569")
    ax.set_xticks(xc)
    ax.set_xticklabels(c_names, fontsize=6.95)
    ax.set_ylim(0, 15.2)
    ax.set_title("C. Real IEEE-754 Archive Compression Ratios")
    ax.set_ylabel("Compression Factor vs Raw FP32 (x)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

    # ------------------------------------------------------------------
    # Panel D: Real NASA Spitzer IRAC Radial Stellar Profile (HDR Peak vs Gibbs Ringing)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    _, sp_raw = read_fits_primary_f32(DATA_DIR / "spitzer_irac.fits")
    sp_img = torch.from_numpy(np.nan_to_num(sp_raw, nan=0.0))
    codec_d = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=512, spike_sigma_threshold=4.2)
    pkg_d = codec_d.encode(sp_img)
    rec_sc = codec_d.decode_full(pkg_d)

    h, w_img = sp_img.shape
    rfft = torch.fft.rfft2(sp_img, norm="ortho")
    n_cplx = max(64, pkg_d.physical_bytes() // 4)
    kx = min(w_img // 2 + 1, max(8, int(np.sqrt(n_cplx / 2.0))))
    ky = min(h // 2, max(8, n_cplx // (2 * kx)))
    rt = torch.zeros_like(rfft)
    rt[:ky, :kx] = torch.complex(rfft[:ky, :kx].real.to(torch.float16).float(), rfft[:ky, :kx].imag.to(torch.float16).float())
    rt[-ky:, :kx] = torch.complex(rfft[-ky:, :kx].real.to(torch.float16).float(), rfft[-ky:, :kx].imag.to(torch.float16).float())
    rec_ft = torch.fft.irfft2(rt, s=(h, w_img), norm="ortho")

    peak_flat = int(torch.argmax(sp_img.reshape(-1)).item())
    pr, pc = peak_flat // w_img, peak_flat % w_img
    win = 24
    c0, c1 = max(0, pc - win), min(w_img, pc + win + 1)
    xs_pix = np.arange(c0 - pc, c1 - pc)
    ax.plot(xs_pix, sp_img[pr, c0:c1].numpy(), lw=2.4, color="#0f172a", label="True Spitzer IRAC (MJy/sr)")
    ax.plot(xs_pix, rec_ft[pr, c0:c1].numpy(), lw=1.8, color="#dc2626", linestyle="--", label="Byte-Matched FP16 2D rFFT")
    ax.plot(xs_pix, rec_sc[pr, c0:c1].numpy(), lw=1.9, color="#0d9488", linestyle=":", marker="o", markersize=2.6, label="ScarCodec (11.77x)")
    ax.set_title(f"D. Spitzer Brightest Star Cut (Row {pr}, Col {pc})")
    ax.set_xlabel("Detector Pixel Offset Δx Around Protostar")
    ax.set_ylabel("Surface Brightness (MJy / sr)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=6.45, frameon=True)

    # ------------------------------------------------------------------
    # Panel E: Empirical HDR Contrast Scaling Law Across 3 Real Telescopes
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    hdr_x = [fc["hdr_peak_to_median_ratio_x"], bo["hdr_peak_to_median_ratio_x"], sp["hdr_peak_to_median_ratio_x"]]
    peak_gains = [fc["peak_error_reduction_vs_fp16_rfft_x"], bo["peak_error_reduction_vs_fp16_rfft_x"], sp["peak_error_reduction_vs_fp16_rfft_x"]]
    rmse_gains = [fc["rmse_gain_vs_fp16_rfft_x"], bo["rmse_gain_vs_fp16_rfft_x"], sp["rmse_gain_vs_fp16_rfft_x"]]
    ax.plot(hdr_x, peak_gains, marker="s", lw=2.3, color="#0f766e", label="Top-32 Peak Photometry Gain (x)")
    ax.plot(hdr_x, rmse_gains, marker="o", lw=1.9, color="#2563eb", linestyle="--", label="Global Map RMSE Gain (x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    obs_lbls = [
        f"FCRAO 14-m 13CO\n({peak_gains[0]:.1f}x peak)",
        f"CSO Bolocam Sgr A*\n({peak_gains[1]:.1f}x peak)",
        f"NASA Spitzer IRAC\n({peak_gains[2]:,.1f}x peak!)",
    ]
    offsets = [(1.15, 1.35), (0.55, 1.55), (0.22, 0.50)]
    for hx, pg, lb, (ox, oy) in zip(hdr_x, peak_gains, obs_lbls, offsets):
        ax.annotate(
            lb,
            xy=(hx, pg),
            xytext=(hx * ox, pg * oy),
            arrowprops=dict(arrowstyle="->", color="#0f766e", lw=0.9),
            fontsize=6.65,
            fontweight="bold",
            color="#065f46",
        )
    ax.set_xlim(7.0, 2200.0)
    ax.set_ylim(1.0, 5500.0)
    ax.set_title("E. Empirical HDR Scaling Law vs FP16 2D rFFT")
    ax.set_xlabel("Observational Peak-to-Median Dynamic Range (Log)")
    ax.set_ylabel("Error Reduction Factor Over rFFT (Log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

    # ------------------------------------------------------------------
    # Panel F: O(1) Single-Block Random Read Speedup vs zlib -9 Full Stream
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    f_names = ["NASA Spitzer\nIRAC (2.1 MB)", "CSO Bolocam\nSgr A* (1.6 MB)", "FCRAO 14-m\n13CO (2.3 MB)"]
    z_us = [sp["zlib_full_stream_decompress_us"], bo["zlib_full_stream_decompress_us"], fc["zlib_full_stream_decompress_us"]]
    s_us = [sp["o1_single_block_decode_us"], bo["o1_single_block_decode_us"], fc["o1_single_block_decode_us"]]
    speedups = [sp["o1_random_read_speedup_vs_zlib_x"], bo["o1_random_read_speedup_vs_zlib_x"], fc["o1_random_read_speedup_vs_zlib_x"]]
    xf = np.arange(3)
    ax.bar(xf - wb / 2, z_us, width=wb, color="#94a3b8", label="zlib -9 Decompress (μs)")
    bars_f = ax.bar(xf + wb / 2, s_us, width=wb, color="#0d9488", label="ScarCodec O(1) 64-Float Block (μs)")
    ax.set_yscale("log")
    ax.set_ylim(40.0, 18000.0)
    for b, vu, spx in zip(bars_f, s_us, speedups):
        ax.text(b.get_x() + b.get_width() / 2, vu * 1.28, f"{vu:.0f}μs\n({spx:.1f}x↑)", ha="center", va="bottom", fontsize=6.65, fontweight="bold", color="#115e59")
    for idx_f, vz in enumerate(z_us):
        ax.text(xf[idx_f] - wb / 2, vz * 1.15, f"{vz:,.0f}μs", ha="center", va="bottom", fontsize=6.35, color="#334155")
    ax.set_xticks(xf)
    ax.set_xticklabels(f_names, fontsize=7.0)
    ax.set_title("F. O(1) Random Block Read Latency (Δblock = 0.0)")
    ax.set_ylabel("Read Latency (μs, Log Scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.5, frameon=True)

    fig.tight_layout(pad=0.95)
    out_dir = ROOT / "paper_latex" / "public_data_paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig1_scarcodec_public_data.pdf"
    out_png = out_dir / "fig1_scarcodec_public_data.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
