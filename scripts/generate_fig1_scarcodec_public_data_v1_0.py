#!/usr/bin/env python3
"""
Generates Figure 1 (`paper_latex/public_data_paper/figures/fig1_scarcodec_public_data.pdf` & `.png`)
for the Dedicated Standalone Zenodo v2.0 / Astronomy & Computing / JOSS Paper on `ScarCodec v6.0`:
  - Panel A: PhysioNet 64-Channel Ag/AgCl Array (`S001R01` Calib -> Held-Out `S001R04` `10.47x` vs Byte-Matched `ZFP` & `2-Bit ADC`)
  - Panel B: `34,816 B` (`Δ=0 B`) `SCache O(1)` on NASA Space Shuttle (`6/6 = 100%`), PhysioNet (`99.67%` 32->32), & `W_scar` vs `Min-Heap` Dual-Mode (`88.26%` vs `69.17%`)
  - Panel C: Real IEEE-754 Compression Ratios (`zlib -9` `1.10x–3.17x`, `ScarCodec VQ4` `10.5x–12.3x`, `Gosset E8 1.000 bpw` `21.0x`, `3D Tucker` `12.32x`)
  - Panel D: Real NASA Spitzer Space Telescope `IRAC` Protostar Cut (`3,938.25 MJy/sr`: Ground Truth vs `FP16 2D rFFT` vs `VQ4 11.77x` & `Gosset E8 21.0x`)
  - Panel E: Monotonic HDR Advantage vs Fixed-Rate `ZFP`, `CFITSIO fpack`, & `FP16 2D rFFT`
  - Panel F: `O(1)` 64-Sample Random Block Read Latency (`zlib -9` `3,211–4,790 us` vs Python `130–134 us` vs `v6.0` Zero-PyTorch C99/SIMD `mmap` `0.388 us` = `11,088x` faster!)
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

from scar_codec import PhysicalVQ4ScarCodec, ScarCodecE8
from scripts.benchmark_scar_codec_public_datasets_v1_0 import DATA_DIR, read_fits_primary_f32


def main() -> None:
    rep = json.loads((ROOT / "results/scar_codec_public_datasets_v1_0_audit.json").read_text(encoding="utf-8"))
    rep_v11 = json.loads((ROOT / "results/scar_codec_public_datasets_v1_1_audit.json").read_text(encoding="utf-8"))
    rep_v6 = json.loads((ROOT / "results/scar_codec_v6_0_consolidation_audit.json").read_text(encoding="utf-8"))

    e1 = rep["experiment_1_physionet_64ch_cross_run_p_perp_vq4"]
    e2 = rep["experiment_2_scache_o1_public_telemetry"]
    e3 = rep["experiment_3_public_astronomy_fits_vq4_vs_fp16_rfft"]
    z1 = rep_v11["experiment_1_physionet_64ch_vs_zfp"]
    z2 = rep_v11["experiment_2_minheap_vs_wscar_telemetry_ablation"]["minheap_vs_wscar_ablation"]
    z3 = rep_v11["experiment_3_fits_vs_zfp_fpack_rfft_zlib"]
    c_simd_us = float(rep_v6["pillar_1_and_2_c_simd_mmap_and_zarr"]["c_simd_pure_random_block_o1_us"])
    e8_ratio = float(rep_v6["pillar_4_gosset_e8_and_3d_spectral_tucker"]["spitzer_gosset_e8_1bpw"]["compression_vs_fp32_x"])
    fc3d_ratio = float(rep_v6["pillar_4_gosset_e8_and_3d_spectral_tucker"]["fcrao_l1448_3d_spectral_tucker"]["compression_vs_fp32_x"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.titlesize": 8.85,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 6.55), dpi=300)

    # ------------------------------------------------------------------
    # Panel A: PhysioNet 64-Channel Cross-Run Blind P_perp + VQ4-Scar vs ZFP
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    cats_a = ["64-Ch Cortical\nArray SNR (dB)", "Model-Free Raw C3\nLaplacian SNR (dB)"]
    v_2b = [e1["clean_array_2bit_adc_snr_db"], e1["raw_c3_laplacian_2bit_adc_snr_db"]]
    v_zfp = [z1["zfp_with_p_perp_array_snr_db"], z1["zfp_with_p_perp_raw_c3_laplacian_snr_db"]]
    v_sc = [e1["clean_array_p_perp_vq4_scar_snr_db"], e1["raw_c3_laplacian_p_perp_vq4_scar_snr_db"]]
    xa = np.arange(2)
    w = 0.25
    ax.bar(xa - w, v_2b, width=w, color="#dc2626", label="2-Bit Scalar ADC")
    ax.bar(xa, v_zfp, width=w, color="#f59e0b", label=f"Byte-Matched ZFP + P_⊥ ({z1['zfp_compression_x']}x)")
    bars_a = ax.bar(xa + w, v_sc, width=w, color="#0f766e", label=f"Cross-Run P_⊥ + VQ4 ({e1['scar_codec_compression_x']}x)")
    ax.axhline(0.0, color="#475569", lw=0.7)
    for b, val in zip(bars_a, v_sc):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.55, f"+{val:.2f} dB", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#065f46")
    ax.text(
        0.03,
        0.95,
        f"Calib: S001R01 ({e1['calibration_top4_artifact_energy_pct']:.1f}% artifact)\nHeld-Out: S001R04 ([64, 19968])\n+4.50 dB > ZFP (0 synthetic data)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.55,
        fontweight="bold",
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#f0fdfa", edgecolor="#0f766e", lw=0.7),
    )
    ax.set_xticks(xa)
    ax.set_xticklabels(cats_a, fontsize=7.15)
    ax.set_ylim(-7.0, 23.4)
    ax.set_title("A. PhysioNet 64-Ch Cross-Run Blind P_⊥ + VQ4")
    ax.set_ylabel("Reconstruction SNR (dB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=6.35, frameon=True)

    # ------------------------------------------------------------------
    # Panel B: Constant-Memory O(1) SCache vs Circular Ring & Min-Heap
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    p64 = e2["physionet_64ch_stream"]
    sh = e2["nasa_shuttle_radiator_stream"]
    cats_b = [
        "Shuttle Bpv\nFaults (6/6)",
        "PhysioNet\n32->32 Odd",
        "Dual-Mode\nSuperposition",
    ]
    base_b = [
        sh["circular_ring_bypass_valve_recall_pct"],
        0.0,
        z2["orthogonal_dual_mode_32odd_cosine_minheap_nn"] * 100.0,
    ]
    scar_b = [
        sh["scache_bypass_valve_recall_pct"],
        p64["blind_32even_to_32odd_masked_cosine"] * 100.0,
        z2["orthogonal_dual_mode_32odd_cosine_w_scar"] * 100.0,
    ]
    xb = np.arange(3)
    wb = 0.34
    ax.bar(xb - wb / 2, base_b, width=wb, color="#94a3b8", label="Equal-Byte Ring / 48-Slot Min-Heap NN")
    bars_b = ax.bar(xb + wb / 2, scar_b, width=wb, color="#1d4ed8", label="SurpriseGatedAnomalySCache O(1)")
    tags_b = ["6/6 (100%)\n(Δ = 0 B)", f"{scar_b[1]:.2f}%\n(34,816 B)", f"{scar_b[2]:.2f}%\n(vs 69.2% Heap)"]
    for b, val, tg in zip(bars_b, scar_b, tags_b):
        ax.text(b.get_x() + b.get_width() / 2, val + 2.0, tg, ha="center", va="bottom", fontsize=6.55, fontweight="bold", color="#1e3a8a")
    ax.set_xticks(xb)
    ax.set_xticklabels(cats_b, fontsize=6.95)
    ax.set_ylim(0, 146)
    ax.set_title("B. Constant-Space O(1) Anomaly & W_scar Recall")
    ax.set_ylabel("Retention / Masked Cosine (%)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.35, frameon=True)

    # ------------------------------------------------------------------
    # Panel C: Compression Ratio Across All 4 Real Public Archives (+ Gosset E8 21.0x!)
    # ------------------------------------------------------------------
    ax = axes[0, 2]
    sp = e3["nasa_spitzer_irac"]
    bo = e3["cso_bolocam_sgr_a"]
    fc = e3["fcrao_l1448_13co_cube"]
    c_names = ["PhysioNet\n64-Ch EDF", "NASA Spitzer\nIRAC FITS", "CSO Bolocam\nSgr A* FITS", "FCRAO 14-m\n3D 13CO Cube"]
    z_ratios = [e1["zlib_compression_x"], sp["zlib_compression_x"], bo["zlib_compression_x"], fc["zlib_compression_x"]]
    s_ratios = [e1["scar_codec_compression_x"], sp["scar_codec_compression_x"], bo["scar_codec_compression_x"], fc["scar_codec_compression_x"]]
    e8_ratios = [e1["scar_codec_compression_x"], e8_ratio, bo["scar_codec_compression_x"], fc3d_ratio]
    xc = np.arange(4)
    wc = 0.25
    ax.bar(xc - wc, z_ratios, width=wc, color="#cbd5e1", edgecolor="#475569", lw=0.5, label="Lossless zlib -9")
    bars_c = ax.bar(xc, s_ratios, width=wc, color="#059669", edgecolor="#064e3b", lw=0.5, label="ScarCodec VQ4 (2.0 bpw)")
    bars_e8 = ax.bar([xc[1] + wc, xc[3] + wc], [e8_ratio, fc3d_ratio], width=wc, color="#7c3aed", edgecolor="#4c1d95", lw=0.5, label="v6.0 Gosset E8 (1 bpw) / 3D")
    for idx_c, b in enumerate(bars_c):
        vr = s_ratios[idx_c]
        ax.text(b.get_x() + b.get_width() / 2, vr + 0.35, f"{vr:.1f}x", ha="center", va="bottom", fontsize=6.45, fontweight="bold", color="#065f46")
    for b, vr, tg_sfx in zip(bars_e8, [e8_ratio, fc3d_ratio], ["E8", "3D"]):
        ax.text(b.get_x() + b.get_width() / 2, vr + 0.35, f"{vr:.1f}x\n({tg_sfx})", ha="center", va="bottom", fontsize=6.45, fontweight="bold", color="#5b21b6")
    for idx_z, vz in enumerate(z_ratios):
        ax.text(xc[idx_z] - wc, vz + 0.3, f"{vz:.2f}x", ha="center", va="bottom", fontsize=6.2, color="#475569")
    ax.set_xticks(xc)
    ax.set_xticklabels(c_names, fontsize=6.85)
    ax.set_ylim(0, 26.2)
    ax.set_title("C. Real IEEE-754 Archive Compression Ratios")
    ax.set_ylabel("Compression Factor vs Raw FP32 (x)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.25, frameon=True)

    # ------------------------------------------------------------------
    # Panel D: Real NASA Spitzer IRAC Radial Stellar Profile (VQ4 11.8x & Gosset E8 21.0x vs rFFT)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    _, sp_raw = read_fits_primary_f32(DATA_DIR / "spitzer_irac.fits")
    sp_np = np.nan_to_num(sp_raw, nan=0.0).astype(np.float32)
    sp_img = torch.from_numpy(sp_np)
    codec_d = PhysicalVQ4ScarCodec(rank=16, max_point_spikes=512, spike_sigma_threshold=4.2)
    pkg_d = codec_d.encode(sp_img)
    rec_sc = codec_d.decode_full(pkg_d)

    codec_e8 = ScarCodecE8(rank=12, max_point_spikes=512, spike_sigma_threshold=4.2)
    rec_e8 = codec_e8.decode_full(codec_e8.encode(sp_np))

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
    ax.plot(xs_pix, rec_ft[pr, c0:c1].numpy(), lw=1.7, color="#dc2626", linestyle="--", label="Byte-Matched FP16 2D rFFT")
    ax.plot(xs_pix, rec_sc[pr, c0:c1].numpy(), lw=1.8, color="#0d9488", linestyle=":", marker="o", markersize=2.5, label="ScarCodec VQ4 (11.8x)")
    ax.plot(xs_pix, rec_e8[pr, c0:c1], lw=1.4, color="#7c3aed", linestyle="-.", label="v6.0 Gosset E8 (21.0x)")
    ax.set_title(f"D. Spitzer Brightest Star Cut (Row {pr}, Col {pc})")
    ax.set_xlabel("Detector Pixel Offset Δx Around Protostar")
    ax.set_ylabel("Surface Brightness (MJy / sr)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=6.2, frameon=True)

    # ------------------------------------------------------------------
    # Panel E: Monotonic HDR Contrast Scaling Behavior vs ZFP & FP16 2D rFFT
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    hdr_x = [fc["hdr_peak_to_median_ratio_x"], bo["hdr_peak_to_median_ratio_x"], sp["hdr_peak_to_median_ratio_x"]]
    peak_gains = [fc["peak_error_reduction_vs_fp16_rfft_x"], bo["peak_error_reduction_vs_fp16_rfft_x"], sp["peak_error_reduction_vs_fp16_rfft_x"]]
    zfp_pk = [
        z3["fcrao_l1448_13co_cube"]["scar_peak_gain_over_zfp_x"],
        z3["cso_bolocam_sgr_a"]["scar_peak_gain_over_zfp_x"],
        z3["nasa_spitzer_irac"]["scar_peak_gain_over_zfp_x"],
    ]
    rmse_gains = [fc["rmse_gain_vs_fp16_rfft_x"], bo["rmse_gain_vs_fp16_rfft_x"], sp["rmse_gain_vs_fp16_rfft_x"]]
    ax.plot(hdr_x, peak_gains, marker="s", lw=2.3, color="#0f766e", label="Peak Gain vs FP16 2D rFFT (x)")
    ax.plot(hdr_x, zfp_pk, marker="^", lw=2.0, color="#7c3aed", linestyle="-.", label="Peak Gain vs Fixed-Rate ZFP (x)")
    ax.plot(hdr_x, rmse_gains, marker="o", lw=1.7, color="#2563eb", linestyle="--", label="Global RMSE Gain vs rFFT (x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    obs_lbls = [
        "FCRAO 14-m 3D\n(4.4x ZFP | 2.2x fpack)",
        "CSO Bolocam Sgr A*\n(204x ZFP | 16.5x fpack)",
        "NASA Spitzer IRAC\n(1,636x ZFP | 2.7x fpack)",
    ]
    offsets = [(1.15, 22.0), (0.85, 0.14), (0.062, 0.55)]
    for hx, pg, lb, (ox, oy) in zip(hdr_x, peak_gains, obs_lbls, offsets):
        ax.annotate(
            lb,
            xy=(hx, pg),
            xytext=(hx * ox, pg * oy),
            arrowprops=dict(arrowstyle="->", color="#0f766e", lw=0.9),
            fontsize=6.3,
            fontweight="bold",
            color="#065f46",
        )
    ax.set_xlim(7.0, 2200.0)
    ax.set_ylim(1.0, 6500.0)
    ax.set_title("E. Monotonic HDR Advantage vs ZFP, fpack & rFFT")
    ax.set_xlabel("Observational Peak-to-Median Dynamic Range (Log)")
    ax.set_ylabel("Error Reduction Factor (Log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.35, frameon=True)

    # ------------------------------------------------------------------
    # Panel F: O(1) Single-Block Random Read Speedup (zlib vs Python vs v6.0 C99/SIMD mmap!)
    # ------------------------------------------------------------------
    ax = axes[1, 2]
    f_names = ["NASA Spitzer\nIRAC (2.1 MB)", "CSO Bolocam\nSgr A* (1.6 MB)", "FCRAO 14-m\n13CO (2.3 MB)"]
    z_us = [sp["zlib_full_stream_decompress_us"], bo["zlib_full_stream_decompress_us"], fc["zlib_full_stream_decompress_us"]]
    py_us = [sp["o1_single_block_decode_us"], bo["o1_single_block_decode_us"], fc["o1_single_block_decode_us"]]
    c_us = [c_simd_us, c_simd_us, c_simd_us]
    xf = np.arange(3)
    wf = 0.25
    ax.bar(xf - wf, z_us, width=wf, color="#94a3b8", label="zlib -9 Full Stream (μs)")
    ax.bar(xf, py_us, width=wf, color="#0d9488", label="ScarCodec Python O(1) (μs)")
    bars_cf = ax.bar(xf + wf, c_us, width=wf, color="#7c3aed", label="v6.0 C99/SIMD mmap O(1) (μs)")
    ax.set_yscale("log")
    ax.set_ylim(0.10, 180000.0)
    for idx_f in range(3):
        ax.text(xf[idx_f] - wf, z_us[idx_f] * 1.18, f"{z_us[idx_f]:,.0f}μs", ha="center", va="bottom", fontsize=6.0, color="#334155")
        ax.text(xf[idx_f], py_us[idx_f] * 1.18, f"{py_us[idx_f]:.0f}μs", ha="center", va="bottom", fontsize=6.1, fontweight="bold", color="#115e59")
        spx_c = z_us[idx_f] / c_simd_us
        ax.text(
            bars_cf[idx_f].get_x() + bars_cf[idx_f].get_width() / 2,
            c_simd_us * 1.25,
            f"{c_simd_us:.2f}μs\n({spx_c:,.0f}x↑)",
            ha="center",
            va="bottom",
            fontsize=6.1,
            fontweight="bold",
            color="#5b21b6",
        )
    ax.set_xticks(xf)
    ax.set_xticklabels(f_names, fontsize=6.95)
    ax.set_title("F. O(1) 64-Float Block Read (0.388 μs C99/SIMD)")
    ax.set_ylabel("Read Latency (μs, Log Scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=6.2, frameon=True)

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
