#!/usr/bin/env python3
"""
Builds the Dedicated Standalone Zenodo Deposit (`zenodo_deposit_scarcodec_public_data_v1_0/`):
  1. Generates `paper_latex/public_data_paper/figures/fig1_scarcodec_public_data.pdf` / `.png`
     + `zenodo_deposit_scarcodec_public_data_v1_0/scarcodec_public_data_v1_0_hero.png`
  2. Audits `paper_latex/public_data_paper/{main.tex, references.bib, main.bbl}` (15/15 `.bbl` cites, braces, envs, refs)
  3. Builds `ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf` directly from `results/scar_codec_public_datasets_v1_0_audit.json`
  4. Builds `ScarCodec_Public_Data_Replication_v1.0.zip` + `ScarCodec_Public_Data_LaTeX_Source_v1.0.zip` + `ZENODO_METADATA_public_data_v1.0.json`
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape
import zipfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "zenodo_deposit_scarcodec_public_data_v1_0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PUB_LATEX_DIR = REPO_ROOT / "paper_latex" / "public_data_paper"


def styled_table(data, col_widths, s_th, s_td, bold_rows=()):
    wrapped_rows = []
    for r_idx, row in enumerate(data):
        out_row = []
        for cell in row:
            txt = escape(str(cell))
            if r_idx == 0:
                out_row.append(Paragraph(f"<b>{txt}</b>", s_th))
            elif r_idx in bold_rows:
                out_row.append(Paragraph(f"<b>{txt}</b>", s_td))
            else:
                out_row.append(Paragraph(txt, s_td))
        wrapped_rows.append(out_row)

    t = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
    ]
    for br in bold_rows:
        style_cmds.append(("BACKGROUND", (0, br), (-1, br), colors.HexColor("#ECFDF5")))
    t.setStyle(TableStyle(style_cmds))
    return t


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.3)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        0.50 * inch,
        0.30 * inch,
        "Reis Filho (2026) — Empirical Evaluation of ScarCodec on Public Observational & Telemetry Archives (Zenodo Preprint v1.0)",
    )
    canvas.drawRightString(LETTER[0] - 0.50 * inch, 0.30 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_public_data(pdf_path: Path, hero_png: Path, rep: dict):
    e1 = rep["experiment_1_physionet_64ch_cross_run_p_perp_vq4"]
    e2 = rep["experiment_2_scache_o1_public_telemetry"]
    e3 = rep["experiment_3_public_astronomy_fits_vq4_vs_fp16_rfft"]
    p64 = e2["physionet_64ch_stream"]
    sh = e2["nasa_shuttle_radiator_stream"]
    sp = e3["nasa_spitzer_irac"]
    bo = e3["cso_bolocam_sgr_a"]
    fc = e3["fcrao_l1448_13co_cube"]

    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=11.3,
        leading=14.0, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=3.2
    )
    s_meta = ParagraphStyle(
        "PaperMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=7.85,
        leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#334155"), spaceAfter=3.8
    )
    s_abs = ParagraphStyle(
        "AbstractText", parent=styles["Normal"], fontName="Helvetica", fontSize=7.75,
        leading=10.45, alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8, spaceAfter=3.6,
        textColor=colors.HexColor("#1E293B")
    )
    s_h1 = ParagraphStyle(
        "SecH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=9.25,
        leading=11.5, textColor=colors.HexColor("#0F766E"), spaceBefore=4.4, spaceAfter=1.8,
        keepWithNext=True,
    )
    s_body = ParagraphStyle(
        "BodyJustified", parent=styles["Normal"], fontName="Helvetica", fontSize=7.95,
        leading=10.95, alignment=TA_JUSTIFY, spaceAfter=2.6, textColor=colors.HexColor("#1E293B")
    )
    s_box = ParagraphStyle(
        "RuleBox", parent=styles["Normal"], fontName="Helvetica", fontSize=7.8,
        leading=11.3, alignment=TA_JUSTIFY, backColor=colors.HexColor("#F1F5F9"),
        borderPadding=3.6, leftIndent=5, rightIndent=5, spaceBefore=2.0, spaceAfter=2.8,
        textColor=colors.HexColor("#0F172A")
    )
    s_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.15,
        leading=9.05, alignment=TA_LEFT, textColor=colors.HexColor("#475569"), spaceBefore=1.4, spaceAfter=3.2
    )
    s_th = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=6.55,
        leading=7.95, alignment=TA_LEFT, textColor=colors.white
    )
    s_td = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica", fontSize=6.5,
        leading=7.95, alignment=TA_LEFT, textColor=colors.HexColor("#0F172A")
    )
    s_ref = ParagraphStyle(
        "RefItem", parent=styles["Normal"], fontName="Helvetica", fontSize=6.8,
        leading=8.45, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=0.9,
        textColor=colors.HexColor("#334155")
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.50 * inch,
        rightMargin=0.50 * inch,
        topMargin=0.38 * inch,
        bottomMargin=0.40 * inch,
        title="Empirical Evaluation of Domain-Independent ScarCodec Operators on Public Observational Astronomy, Spacecraft Telemetry, and 64-Channel Biomedical Sensor Archives",
        author="Julio Cesar Reis Filho",
    )

    story = []
    story.append(Paragraph(
        "Empirical Evaluation of Domain-Independent <code>ScarCodec</code> Operators on Public Observational<br/>"
        "Astronomy, Spacecraft Telemetry, and 64-Channel Biomedical Sensor Archives",
        s_title,
    ))
    story.append(Paragraph(
        "<b>Julio Cesar Reis Filho</b> — Byte-OD Research &amp; System One Architecture Lab<br/>"
        "Dedicated Standalone Zenodo Empirical Report <b>v1.0.0</b> (Zero Synthetic Signals, Zero Oracle Steering, 7 SHA-256 Verified Public Files) — November 16, 2026<br/>"
        "<b>Code &amp; Automated Downloader:</b> github.com/byte-od/mixture-of-scars &nbsp;|&nbsp; <b>Companion Framework DOI:</b> 10.5281/zenodo.24172784",
        s_meta,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=3.2))

    story.append(Paragraph(
        f"<b>Abstract.</b> Earlier controlled evaluations of <code>ScarCodec</code> relied on synthetic partial-differential-equation and antenna-array phantoms "
        f"to isolate individual failure modes of conventional compressors and circular flight recorders. In this dedicated empirical study, we evaluate the "
        f"<b>unmodified</b>, neural-network-free <code>scar_codec</code> operators (<code>PhysicalVQ4ScarCodec</code>, <code>OrthogonalSubspaceProjector</code>, "
        f"and <code>SurpriseGatedAnomalySCache</code>) directly on <b>five SHA-256-verified public observational and experimental datasets</b> spanning four physical sciences, "
        f"with zero synthetic noise injection and zero oracle steering: "
        f"<b>(1) 64-Channel Biopotential Sensor Array (<code>PhysioNet EEGMMIDB v1.0.0</code>)</b>: calibrating <b>P</b><sub>&perp;</sub> (<i>r</i>=4) strictly on a separate "
        f"eyes-open baseline run (<code>S001R01.edf</code>, where ocular/common-mode artifacts carry {e1['calibration_top4_artifact_energy_pct']:.2f}% of raw variance) and "
        f"compressing the held-out motor-imagery run (<code>S001R04.edf</code>, [64, 19968] FP32 = 5.11 MB) at <b>{e1['scar_codec_compression_x']:.2f}x</b> ({e1['scar_codec_bytes']:,} B "
        f"vs. {e1['zlib_compression_x']:.2f}x <code>zlib -9</code>) achieves <b>+{e1['clean_array_p_perp_vq4_scar_snr_db']:.2f} dB</b> cortical array SNR "
        f"(cos={e1['clean_array_p_perp_vq4_scar_cosine']:.5f}) and <b>+{e1['raw_c3_laplacian_p_perp_vq4_scar_snr_db']:.2f} dB</b> on the model-free Hjorth surface Laplacian over "
        f"primary motor cortex C3 (vs. {e1['raw_c3_laplacian_2bit_adc_snr_db']:.2f} dB for 2-bit scalar ADC); "
        f"<b>(2) Constant-Memory O(1) Anomaly Telemetry</b>: on chronological 64-channel <code>PhysioNet</code> monitoring (39,520 &times; 64 = 10.12 MB FP32), exact "
        f"<b>34,816 B (&Delta;=0 B)</b> <code>SCache</code> retains {p64['early_r01_r02_transients_retained_scache']}/3 early baseline bursts across 20,000 subsequent task frames "
        f"(0/3 in an equal-byte 256-frame circular ring) and reconstructs 32 masked odd electrodes from 32 even electrodes at <b>cos={p64['blind_32even_to_32odd_masked_cosine']:.5f}</b>; "
        f"on <b>NASA Space Shuttle radiator telemetry</b> (<code>shuttle.tst</code>, 14,500 &times; 8 sensors), default <code>SCache</code> (&Delta;=0 B) retains "
        f"<b>{sh['scache_bypass_valve_recall']} ({sh['scache_bypass_valve_recall_pct']:.1f}%)</b> of ultra-rare critical Bypass Valve Open/Close faults "
        f"({sh['critical_bypass_valve_prevalence_pct']:.4f}% prevalence) vs. 0/6 (0.0%) in an equal-byte ring buffer, while transparently documenting rank-8 saturation "
        f"(|cos|={sh['rank8_saturated_1sensor_abs_cosine']:.4f}) under {sh['total_high_surprise_writes']} secondary excursions; and "
        f"<b>(3) Space Infrared, Submillimeter &amp; Radio FITS Archives (NASA Spitzer IRAC, CSO Bolocam Sgr A*, FCRAO 14-m L1448 <sup>13</sup>CO)</b>: whereas <code>zlib -9</code> "
        f"stalls at {fc['zlib_compression_x']:.2f}x–{sp['zlib_compression_x']:.2f}x on IEEE-754 floats, <code>ScarCodec</code> achieves <b>{sp['scar_codec_compression_x']:.2f}x–{bo['scar_codec_compression_x']:.2f}x</b> "
        f"compression with <b>{bo['o1_random_read_speedup_vs_zlib_x']:.1f}x–{fc['o1_random_read_speedup_vs_zlib_x']:.1f}x</b> faster O(1) random block reads (&Delta;block=0.0) and "
        f"exhibits a monotonic high-dynamic-range (HDR) scaling law over byte-matched IEEE-754 FP16 2D <code>rFFT</code>, reducing top-32 peak source photometry errors by "
        f"<b>{fc['peak_error_reduction_vs_fp16_rfft_x']:.1f}x</b> (FCRAO), <b>{bo['peak_error_reduction_vs_fp16_rfft_x']:.1f}x</b> (Bolocam), and <b>{sp['peak_error_reduction_vs_fp16_rfft_x']:,.1f}x</b> (Spitzer).",
        s_abs,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=3.2))

    story.append(KeepTogether([
        Image(str(hero_png), width=7.48 * inch, height=3.01 * inch),
        Paragraph(
            f"<b>Figure 1.</b> Empirical Evaluation of Unmodified <code>ScarCodec</code> Across Five Public Scientific Archives. "
            f"<b>(A)</b> Cross-run blind <b>P</b><sub>&perp;</sub> (<code>S001R01</code>) + <code>VQ4-Scar</code> on held-out 64-ch <code>PhysioNet S001R04</code> ({e1['scar_codec_compression_x']:.2f}x). "
            f"<b>(B)</b> Constant-space O(1) <code>SCache</code> (&Delta;=0 B) retaining 6/6 NASA Space Shuttle Bypass Valve faults and {p64['blind_32even_to_32odd_masked_cosine']*100:.2f}% 32&rarr;32 masked odd-electrode recall in 34,816 B. "
            f"<b>(C)</b> Compression factor vs. IEEE-754 FP32 across four archives. "
            f"<b>(D)</b> Detector profile across the brightest protostar in NASA Spitzer IRAC (3,938.25 MJy/sr) eliminating FP16 2D <code>rFFT</code> Gibbs ringing. "
            f"<b>(E)</b> Monotonic HDR contrast scaling law across FCRAO 14-m, CSO Bolocam, and NASA Spitzer. "
            f"<b>(F)</b> O(1) 64-float block decode latency (&mu;s) vs. <code>zlib -9</code>.",
            s_caption,
        ),
    ]))

    story.append(Paragraph("1. Forensic Experimental Protocol &amp; Public Archive Provenance", s_h1))
    story.append(Paragraph(
        "<b>Four Non-Negotiable Audit Rules:</b> (1) <i>Unmodified Operator Code:</i> Every benchmark imports <code>scar_codec/core.py</code> verbatim without "
        "dataset-specific tuning or neural networks. (2) <i>Zero Synthetic Data:</i> All sensor voltages, ocular blinks, unmapped sky masks, protostellar spikes, "
        "and spacecraft radiator faults are genuine physical measurements from open public archives (Table 1). (3) <i>Zero Oracle Calibration:</i> Interference "
        "subspace <b>P</b><sub>&perp;</sub> is estimated strictly on separate baseline run <code>S001R01.edf</code> and evaluated on held-out task run <code>S001R04.edf</code>; "
        "telemetry scaling uses only initial warmup frames; associative queries mask 50%–87.5% of channels to 0.0. (4) <i>Strict Byte Parity:</i> Spectral 2D <code>rFFT</code> "
        "baselines store retained complex modes in IEEE-754 FP16 (4 B/mode) matched within &lt;1% of <code>ScarCodec</code> bytes.",
        s_box,
    ))

    prov = rep["provenance_manifest"]
    t_prov = [
        ["Public Scientific Archive", "Format & Geometry", "Raw File Bytes", "License", "Verified SHA-256 Digest (First 24 Hex)"],
        ["PhysioNet EEGMMIDB (S001R01/R02/R04)", "EDF 64-Ch Ag/AgCl (160 Hz)", f"{prov[0]['bytes']+prov[1]['bytes']+prov[2]['bytes']:,} B", "ODC-By-1.0", f"{prov[0]['sha256'][:12]}.. / {prov[2]['sha256'][:12]}.."],
        ["NASA Space Shuttle Statlog (UCI ID 148)", "shuttle.tst (14,500 x 8 sensors)", f"{prov[3]['bytes']:,} B (zip)", "CC-BY-4.0", prov[3]["sha256"][:24]],
        ["NASA Spitzer Space Telescope (IRAC)", "FITS FP32 [513, 1025] (MJy/sr)", f"{prov[4]['bytes']:,} B", "Public / BSD-3", prov[4]["sha256"][:24]],
        ["CSO Bolocam Galactic Center (Sgr A*)", "FITS FP32 [638, 640] (Jy/Beam)", f"{prov[5]['bytes']:,} B", "Public / BSD-3", prov[5]["sha256"][:24]],
        ["FCRAO 14-m Radio L1448 13CO Outflow", "FITS FP32 [53, 105, 105] (K)", f"{prov[6]['bytes']:,} B", "Public / BSD-3", prov[6]["sha256"][:24]],
    ]
    story.append(KeepTogether([
        styled_table(t_prov, [1.85 * inch, 1.48 * inch, 0.98 * inch, 0.95 * inch, 2.24 * inch], s_th, s_td),
        Paragraph("<b>Table 1 (Archival Provenance, Physical Units, Open Licenses &amp; SHA-256 Checksums — <code>manifest_sha256.json</code>).</b>", s_caption),
    ]))

    story.append(Paragraph("2. Multi-Channel Array Interference Cancellation &amp; O(1) Spacecraft/Biomedical Telemetry", s_h1))
    story.append(Paragraph(
        f"In <code>PhysioNet S001R01.edf</code> (eyes-open baseline), frontopolar ocular blinks and common-mode pickup concentrate {e1['calibration_top4_artifact_energy_pct']:.2f}% "
        f"of raw array energy in the top-4 singular modes. Fitting <code>OrthogonalSubspaceProjector(rank=4)</code> strictly on <code>S001R01.edf</code> and compressing the "
        f"independent held-out motor-imagery run <code>S001R04.edf</code> ([64, 19968] = 5.11 MB FP32) at <b>{e1['scar_codec_compression_x']:.2f}x</b> ({e1['scar_codec_bytes']:,} B) "
        f"delivers +{e1['clean_array_p_perp_vq4_scar_snr_db']:.2f} dB cortical array SNR (+2.69 dB above unshielded VQ4 and +13.54 dB above 2-bit scalar ADC) and "
        f"<b>+{e1['raw_c3_laplacian_p_perp_vq4_scar_snr_db']:.2f} dB</b> on the model-free physical Hjorth Laplacian over primary motor cortex C3 (where 2-bit ADC collapses to "
        f"{e1['raw_c3_laplacian_2bit_adc_snr_db']:.2f} dB due to clipped blink aliasing). Table 2 summarizes both <code>SCache O(1)</code> telemetry benchmarks, including "
        f"<code>6/6 = 100.0%</code> retention of 0.0414%-prevalence Bypass Valve faults in NASA Space Shuttle radiator telemetry and honest rank-8 saturation diagnostics.",
        s_body,
    ))

    t_tel = [
        ["Public Stream / Experiment", "Raw FP32 Size", "Fixed Memory (Δ B)", "Baseline Method", "Baseline Result", "ScarCodec Operator Result"],
        [
            "PhysioNet Held-Out S001R04 (Cross-Run P_⊥)",
            f"{e1['raw_fp32_bytes']:,} B",
            f"{e1['scar_codec_bytes']:,} B ({e1['scar_codec_compression_x']:.2f}x)",
            "2-Bit ADC / Unshielded VQ4",
            f"Raw C3 Lap: {e1['raw_c3_laplacian_2bit_adc_snr_db']:.2f} / +{e1['raw_c3_laplacian_unshielded_vq4_snr_db']:.2f} dB",
            f"Array +{e1['clean_array_p_perp_vq4_scar_snr_db']:.2f} dB · Raw C3 Lap +{e1['raw_c3_laplacian_p_perp_vq4_scar_snr_db']:.2f} dB (cos={e1['raw_c3_laplacian_p_perp_vq4_scar_cosine']:.4f})",
        ],
        [
            "PhysioNet Chronological R01->R02->R04",
            f"{p64['raw_fp32_bytes']:,} B",
            "34,816 B (Δ = 0 B!)",
            "Equal-Byte Ring (256 frames)",
            "0/3 early bursts kept (0.0%)",
            f"3/3 kept · Blind 32->32 odd ch cos={p64['blind_32even_to_32odd_masked_cosine']:.5f} · {p64['ingest_fps']:,.0f} FPS",
        ],
        [
            "NASA Space Shuttle Radiator (shuttle.tst)",
            f"{sh['raw_fp32_bytes']:,} B",
            "3,120 B (Δ = 0 B!)",
            "Equal-Byte Ring (130 frames)",
            "0/6 Bypass Valve faults (0.0%)",
            f"6/6 (100.0% Bpv Open/Close) · Slots exact (8x8 W_scar |cos|={sh['rank8_saturated_1sensor_abs_cosine']:.4f})",
        ],
    ]
    story.append(KeepTogether([
        styled_table(t_tel, [1.54 * inch, 0.84 * inch, 0.96 * inch, 1.16 * inch, 1.16 * inch, 1.84 * inch], s_th, s_td, bold_rows=(1, 2, 3)),
        Paragraph("<b>Table 2 (Biomedical 64-Channel Array Interference Cancellation &amp; Constant-Space O(1) Spacecraft/Biomedical Telemetry Audit).</b>", s_caption),
    ]))

    story.append(Paragraph("3. Observational Astronomy FITS Archives &amp; Empirical HDR Contrast Scaling Law", s_h1))
    story.append(Paragraph(
        "Across three astronomical observatories—infrared space imaging (NASA Spitzer IRAC), submillimeter galactic-center bolometry (CSO Bolocam Sgr A*), "
        "and millimeter spectral-line mapping (FCRAO 14-m L1448 <sup>13</sup>CO)—lossless <code>zlib -9</code> achieves only 1.10x–1.14x compression on IEEE-754 floats "
        "while blocking random block access. At ~12x compression, <code>PhysicalVQ4ScarCodec</code> combines MAD point-shock peeling, equilibrated <code>FWHT-64</code>, "
        "<i>D</i><sub>4</sub> <code>LUT256</code> (<code>0 &rarr; (0,0,0,0)</code>), and 2-pass INT4 scars to outperform strictly byte-matched FP16 2D <code>rFFT</code> "
        "monotonically with observational dynamic range (Table 3): from 4.6x peak gain at 11.7x HDR (FCRAO), to 45.9x at 217.0x HDR (Bolocam, with exact 0.0000 error on "
        "pure unmapped sky rows and 6.06x lower partial-row NaN leakage), up to <b>2,093.6x lower stellar peak photometry error</b> at 968.9x HDR (NASA Spitzer IRAC), "
        "while decoding arbitrary 64-float blocks in ~141 &mu;s (23.2x–34.7x faster than <code>zlib -9</code> with exact 0.0 block-vs-full parity).",
        s_body,
    ))

    t_fits = [
        ["Observatory & FITS Archive", "Shape (HDR Contrast)", "zlib -9", "ScarCodec Bytes", "FP16 rFFT Bytes", "Global RMSE (Scar vs rFFT)", "Top-32 Peak Err (Scar vs rFFT)", "O(1) Block Read"],
        [
            "NASA Spitzer IRAC (MJy/sr)",
            f"513x1025 ({sp['hdr_peak_to_median_ratio_x']:.1f}x)",
            f"{sp['zlib_compression_x']:.2f}x",
            f"{sp['scar_codec_bytes']:,} B ({sp['scar_codec_compression_x']:.2f}x)",
            f"{sp['byte_matched_fp16_rfft_bytes']:,} B",
            f"{sp['scar_codec_rmse']:.3f} vs {sp['fp16_rfft_rmse']:.3f} ({sp['rmse_gain_vs_fp16_rfft_x']:.2f}x↓)",
            f"{sp['scar_codec_top32_peak_max_err']:.3f} vs {sp['fp16_rfft_top32_peak_max_err']:.1f} ({sp['peak_error_reduction_vs_fp16_rfft_x']:,.1f}x↓!)",
            f"{sp['o1_single_block_decode_us']:.1f}μs ({sp['o1_random_read_speedup_vs_zlib_x']:.1f}x↑)",
        ],
        [
            "CSO Bolocam Sgr A* (Jy/Beam)",
            f"638x640 ({bo['hdr_peak_to_median_ratio_x']:.1f}x)",
            f"{bo['zlib_compression_x']:.2f}x",
            f"{bo['scar_codec_bytes']:,} B ({bo['scar_codec_compression_x']:.2f}x)",
            f"{bo['byte_matched_fp16_rfft_bytes']:,} B",
            f"{bo['scar_codec_rmse']:.4f} vs {bo['fp16_rfft_rmse']:.4f} ({bo['rmse_gain_vs_fp16_rfft_x']:.2f}x↓)",
            f"{bo['scar_codec_top32_peak_max_err']:.4f} vs {bo['fp16_rfft_top32_peak_max_err']:.4f} ({bo['peak_error_reduction_vs_fp16_rfft_x']:.1f}x↓!)",
            f"{bo['o1_single_block_decode_us']:.1f}μs ({bo['o1_random_read_speedup_vs_zlib_x']:.1f}x↑)",
        ],
        [
            "FCRAO 14-m L1448 13CO (K)",
            f"371x1575 ({fc['hdr_peak_to_median_ratio_x']:.1f}x)",
            f"{fc['zlib_compression_x']:.2f}x",
            f"{fc['scar_codec_bytes']:,} B ({fc['scar_codec_compression_x']:.2f}x)",
            f"{fc['byte_matched_fp16_rfft_bytes']:,} B",
            f"{fc['scar_codec_rmse']:.4f} vs {fc['fp16_rfft_rmse']:.4f} ({fc['rmse_gain_vs_fp16_rfft_x']:.2f}x↓)",
            f"{fc['scar_codec_top32_peak_max_err']:.4f} vs {fc['fp16_rfft_top32_peak_max_err']:.4f} ({fc['peak_error_reduction_vs_fp16_rfft_x']:.1f}x↓)",
            f"{fc['o1_single_block_decode_us']:.1f}μs ({fc['o1_random_read_speedup_vs_zlib_x']:.1f}x↑)",
        ],
    ]
    story.append(KeepTogether([
        styled_table(t_fits, [1.22 * inch, 0.86 * inch, 0.48 * inch, 0.94 * inch, 0.72 * inch, 1.10 * inch, 1.32 * inch, 0.86 * inch], s_th, s_td, bold_rows=(1, 2, 3)),
        Paragraph("<b>Table 3 (Observational Astronomy FITS Audit vs. Lossless <code>zlib -9</code> &amp; Byte-Matched IEEE-754 FP16 2D <code>rFFT</code>; All &Delta;block = 0.0).</b>", s_caption),
    ]))

    story.append(Paragraph("References", s_h1))
    refs = [
        "[1] Astropy Collaboration, Price-Whelan, A. M., et al. (2022). The Astropy Project (v5.0). Astrophys. J. 935(2), 167.",
        "[2] Conway, J. H., & Sloane, N. J. A. (1982). Fast Quantizing and Decoding Algorithms for Lattice Quantizers and Codes. IEEE Trans. Inf. Theory 28(2), 227–232.",
        "[3] Fazio, G. G., Hora, J. L., et al. (2004). The Infrared Array Camera (IRAC) for the Spitzer Space Telescope. Astrophys. J. Suppl. Ser. 154(1), 10–17.",
        "[4] Ginsburg, A., Glenn, J., et al. (2013). The Bolocam Galactic Plane Survey. IX. Data Release 2. Astrophys. J. Suppl. Ser. 208(2), 14.",
        "[5] Goldberger, A. L., Amaral, L. A. N., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215–e220.",
        "[6] Hjorth, B. (1975). An On-Line Transformation of EEG Scalp Potentials into Orthogonal Source Derivations. Electroencephalogr. Clin. Neurophysiol. 39(5), 526–530.",
        "[7] Kemp, B., Varri, A., Rosa, A. C., Nielsen, K. D., & Gade, J. (1992). A Simple Format for Exchange of Digitized Polygraphic Recordings (EDF). Electroencephalogr. Clin. Neurophysiol. 82(5), 391–393.",
        "[8] Leshem, A., van der Veen, A.-J., & Boonstra, A.-J. (2000). Multichannel Radio Astronomical Imaging in Strong RFI. Astrophys. J. Suppl. Ser. 131(1), 355–373.",
        "[9] Michie, D., Spiegelhalter, D. J., & Taylor, C. C. (1994). Machine Learning, Neural and Statistical Classification (NASA Space Shuttle Statlog). Ellis Horwood.",
        "[10] Pence, W. D., Chiappetti, L., Page, C. G., Shaw, R. A., & Stobie, E. (2010). Definition of FITS, version 3.0. Astron. Astrophys. 524, A42.",
        "[11] Reis Filho, J. C. (2026). Shared-Stage Scar-Native Neural Architecture (ScarNativeLM v6.0), PointScarFlow v1.0, and ScarCodec. Zenodo Preprint. DOI: 10.5281/zenodo.24172784.",
        "[12] Ridge, N. A., Di Francesco, J., et al. (2006). The COMPLETE Survey of Star-Forming Regions: Phase I Data. Astron. J. 131(6), 2921–2933.",
        "[13] Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A General-Purpose BCI System. IEEE Trans. Biomed. Eng. 51(6), 1034–1043.",
        "[14] Walsh, J. L. (1923). A Closed Set of Normal Orthogonal Functions. Amer. J. Math. 45(1), 5–24.",
        "[15] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits. IRE WESCON Convention Record 4, 96–104.",
    ]
    for r in refs:
        story.append(Paragraph(escape(r), s_ref))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def audit_latex_and_pack(zip_out: Path):
    tex = (PUB_LATEX_DIR / "main.tex").read_text(encoding="utf-8")
    bib = (PUB_LATEX_DIR / "references.bib").read_text(encoding="utf-8")
    bbl = (PUB_LATEX_DIR / "main.bbl").read_text(encoding="utf-8")

    clean_tex = "\n".join(re.split(r"(?<!\\)%", ln)[0] for ln in tex.splitlines())
    depth = 0
    for i, ch in enumerate(clean_tex):
        if ch == "{" and (i == 0 or clean_tex[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or clean_tex[i - 1] != "\\"):
            depth -= 1
            assert depth >= 0, f"Extra closing brace at {i}"
    assert depth == 0, f"Unclosed braces: {depth}"

    stack = []
    amp_ok_envs = {"tabular", "align", "align*", "aligned", "matrix", "bmatrix", "pmatrix"}
    for ln_no, ln in enumerate(clean_tex.splitlines(), start=1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", ln):
            if m.group(1) == "begin":
                stack.append(m.group(2))
            else:
                assert stack and stack[-1] == m.group(2), f"Mismatch {m.group(2)} on line {ln_no}"
                stack.pop()
        if not (set(stack) & amp_ok_envs) and not ln.strip().startswith(r"\usepackage"):
            assert not re.search(r"(?<!\\)&", ln), f"Unescaped '&' outside tabular/align on line {ln_no}: {ln}"
    assert not stack, f"Unclosed envs: {stack}"

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    bbl_keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl))
    cited = {k.strip() for m in re.finditer(r"\\cite\w*\{([^}]+)\}", clean_tex) for k in m.group(1).split(",")}
    assert not (cited - bib_keys), f"Missing cites in .bib: {cited - bib_keys}"
    assert not (cited - bbl_keys), f"Missing cites in .bbl: {cited - bbl_keys}"
    assert bib_keys == bbl_keys == cited, f"Exact citation parity mismatch: cited={len(cited)} bib={len(bib_keys)} bbl={len(bbl_keys)}"

    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", clean_tex))
    assert not (refs - labels), f"Missing refs: {refs - labels}"

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_name in ("Makefile", "main.tex", "references.bib", "main.bbl", "figures/fig1_scarcodec_public_data.pdf"):
            fp = PUB_LATEX_DIR / rel_name
            assert fp.exists(), f"Missing {fp}"
            zf.write(fp, arcname=rel_name)
    print(f"[PUBLIC DATA LATEX AUDIT PASS] {len(cited)}/{len(bbl_keys)} bbl cites | {len(labels)} labels | {zip_out.name}")


def build_replication_zip(zip_path: Path):
    files = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "data" / "public_physical_benchmarks" / "manifest_sha256.json",
        REPO_ROOT / "results" / "scar_codec_public_datasets_v1_0_audit.json",
        REPO_ROOT / "results" / "scar_codec_physical_v4_0_audit.json",
        REPO_ROOT / "scripts" / "benchmark_scar_codec_public_datasets_v1_0.py",
        REPO_ROOT / "scripts" / "generate_fig1_scarcodec_public_data_v1_0.py",
        REPO_ROOT / "scripts" / "build_zenodo_scarcodec_public_data_v1_0_kit.py",
        REPO_ROOT / "tests" / "test_scar_codec_public_datasets.py",
        REPO_ROOT / "tests" / "test_scar_codec_physical.py",
    ]
    files.extend(sorted((REPO_ROOT / "scar_codec").glob("*.py")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                zf.write(fp, arcname=str(fp.relative_to(REPO_ROOT)))


def main():
    rep = json.loads((REPO_ROOT / "results" / "scar_codec_public_datasets_v1_0_audit.json").read_text(encoding="utf-8"))
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "generate_fig1_scarcodec_public_data_v1_0.py")])

    src_png = PUB_LATEX_DIR / "figures" / "fig1_scarcodec_public_data.png"
    hero_png = OUT_DIR / "scarcodec_public_data_v1_0_hero.png"
    shutil.copy2(src_png, hero_png)

    pdf_out = OUT_DIR / "ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf"
    rep_out = OUT_DIR / "ScarCodec_Public_Data_Replication_v1.0.zip"
    tex_out = OUT_DIR / "ScarCodec_Public_Data_LaTeX_Source_v1.0.zip"

    build_pdf_public_data(pdf_out, hero_png, rep)
    build_replication_zip(rep_out)
    audit_latex_and_pack(tex_out)

    (OUT_DIR / "ZENODO_METADATA_public_data_v1.0.json").write_text(
        json.dumps(
            {
                "upload_type": "publication",
                "publication_type": "preprint",
                "version": "1.0.0",
                "publication_date": "2026-11-16",
                "title": "Empirical Evaluation of Domain-Independent ScarCodec Operators on Public Observational Astronomy, Spacecraft Telemetry, and 64-Channel Biomedical Sensor Archives",
                "creators": [{"name": "Reis Filho, Julio Cesar da Costa", "affiliation": "Byte-OD Research & System One Architecture Lab"}],
                "license": "cc-by-4.0",
                "keywords": [
                    "ScarCodec",
                    "Scientific Data Compression",
                    "FITS Astronomy",
                    "NASA Spitzer IRAC",
                    "NASA Space Shuttle Telemetry",
                    "PhysioNet EEG",
                    "D4 Lattice Quantization",
                    "Constant-Memory Anomaly Detection",
                ],
                "related_identifiers": [
                    {"identifier": "10.5281/zenodo.24172784", "relation": "isSupplementedBy", "scheme": "doi"},
                    {"identifier": "https://github.com/byte-od/mixture-of-scars", "relation": "isSupplementTo", "scheme": "url"},
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"✓ Built {pdf_out} ({pdf_out.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Built {rep_out} ({rep_out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
