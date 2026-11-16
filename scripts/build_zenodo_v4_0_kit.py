#!/usr/bin/env python3
"""
Builds the Official v4.0.0 Publication Deposit Kit (`zenodo_deposit_v4_0/`):
  1. Generates `paper_latex/figures/fig1_mos_scache_v4_audit.pdf` / `.png` + `zenodo_deposit_v4_0/mos_scache_v4_0_hero.png`
  2. Syncs `huggingface_space/scar_cache/*.py` & `huggingface_space/scar_codec/*.py`
  3. Audits `paper_latex/main.tex`, `references.bib` & `main.bbl` (braces, envs, ampersands, cites, refs & v4 scope boundary)
  4. Builds `zenodo_deposit_v4_0/Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf`
  5. Builds `zenodo_deposit_v4_0/MoS_SCache_Adversarial_Replication_v4.0.zip`
  6. Builds `zenodo_deposit_v4_0/MoS_SCache_LaTeX_arXiv_TMLR_v4.0.zip` & `arXiv_Direct_Upload_v4.0.zip`
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

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
OUT_DIR = REPO_ROOT / "zenodo_deposit_v4_0"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATEX_DIR = REPO_ROOT / "paper_latex"


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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.6),
        ("TOPPADDING", (0, 0), (-1, -1), 2.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
    ]
    for br in bold_rows:
        style_cmds.append(("BACKGROUND", (0, br), (-1, br), colors.HexColor("#EBF3FF")))
    t.setStyle(TableStyle(style_cmds))
    return t


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        0.52 * inch,
        0.31 * inch,
        "Reis Filho (2026) — Mixture of Scars (MoS) v4.0, SCache O(1) & Physical ScarCodec Preprint | DOI: 10.5281/zenodo.24172784",
    )
    canvas.drawRightString(LETTER[0] - 0.52 * inch, 0.31 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_v4_0(pdf_path: Path, hero_png: Path):
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=11.6,
        leading=14.4, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=3.5
    )
    s_meta = ParagraphStyle(
        "PaperMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.0,
        leading=10.8, alignment=TA_CENTER, textColor=colors.HexColor("#334155"), spaceAfter=4.5
    )
    s_abs_box = ParagraphStyle(
        "AbstractText", parent=styles["Normal"], fontName="Helvetica", fontSize=7.9,
        leading=10.7, alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8, spaceAfter=4.0,
        textColor=colors.HexColor("#1E293B")
    )
    s_h1 = ParagraphStyle(
        "SecH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=9.5,
        leading=11.8, textColor=colors.HexColor("#1B365D"), spaceBefore=4.6, spaceAfter=2.0,
        keepWithNext=True,
    )
    s_body = ParagraphStyle(
        "BodyJustified", parent=styles["Normal"], fontName="Helvetica", fontSize=8.0,
        leading=11.2, alignment=TA_JUSTIFY, spaceAfter=2.8, textColor=colors.HexColor("#1E293B")
    )
    s_prop = ParagraphStyle(
        "PropBox", parent=styles["Normal"], fontName="Helvetica", fontSize=7.85,
        leading=12.0, alignment=TA_JUSTIFY, backColor=colors.HexColor("#F1F5F9"),
        borderPadding=3.8, leftIndent=5, rightIndent=5, spaceBefore=2.2, spaceAfter=3.0,
        textColor=colors.HexColor("#0F172A")
    )
    s_eq = ParagraphStyle(
        "Equation", parent=styles["Normal"], fontName="Helvetica", fontSize=8.0,
        leading=12.0, alignment=TA_CENTER, backColor=colors.HexColor("#F8FAFC"),
        borderPadding=3.4, spaceBefore=2.4, spaceAfter=3.0, textColor=colors.HexColor("#0F172A")
    )
    s_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.3,
        leading=9.3, alignment=TA_LEFT, textColor=colors.HexColor("#475569"), spaceBefore=1.6, spaceAfter=3.6
    )
    s_th = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=6.7,
        leading=8.2, alignment=TA_LEFT, textColor=colors.white
    )
    s_td = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica", fontSize=6.65,
        leading=8.1, alignment=TA_LEFT, textColor=colors.HexColor("#0F172A")
    )
    s_ref = ParagraphStyle(
        "RefItem", parent=styles["Normal"], fontName="Helvetica", fontSize=6.9,
        leading=8.7, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=1.0,
        textColor=colors.HexColor("#334155")
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.52 * inch,
        rightMargin=0.52 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.44 * inch,
        title="Mixture of Scars (MoS) v4.0, SCache O(1), and Domain-Independent ScarCodec",
        author="Julio Cesar Reis Filho",
    )

    story = []
    story.append(Paragraph(
        "Mixture of Scars (MoS) v4.0, SCache O(1), and Domain-Independent <code>ScarCodec</code>:<br/>"
        "Unified Native &amp; 20-Shader WebGPU Scar Operators, &lt;24 GB 72.7B Compression,<br/>"
        "Language-Invariant 140 KB HTTP 206 Cartridges, and Hardware-to-Science Signal Proofs",
        s_title,
    ))
    story.append(Paragraph(
        "<b>Julio Cesar Reis Filho</b> — Byte-OD Research &amp; System One Architecture Lab<br/>"
        "Preprint Technical Report <b>v4.0.0</b> (Unified Native + 20-Shader WebGPU <code>VQ4-Scar</code>/<code>ProtectedInt4Scar</code>, <i>P</i><sub>&perp;lang</sub> Polyglot HTTP 206, &lt;24 GB 72.7B &amp; Physical <code>ScarCodec</code>) — November 15, 2026<br/>"
        "<b>DOI:</b> 10.5281/zenodo.24172784 &nbsp;|&nbsp; <b>Code, 7 CLIs &amp; 114/114 Tests:</b> github.com/byte-od/mixture-of-scars &nbsp;|&nbsp; <b>72B Shards:</b> <code>releases/tag/v1.2.0-qwen72b-mos</code>",
        s_meta,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=3.6))

    story.append(Paragraph(
        "<b>Abstract.</b> Pretrained autoregressive Transformers and physical high-dynamic-range signals share a universal structural triad: "
        "a smooth correlated substrate, exact zero-energy vacuum spans, and sparse extreme shocks (607x–195,720x above RMS). Uniform scalar quantization, "
        "linear O(<i>T</i>) buffers, and global Fourier/SGD updates violate this geometry across six hardware walls: O(<i>N</i>) parameter bloat (135.43 GiB at 72.7B), "
        "O(<i>T</i>) KV-cache stalls, memory-bound decode, backpropagation-bound specialization, multilingual script corruption in browser/edge quantizers, and Gibbs ringing "
        "or overwrite amnesia in physical telemetry. In <b>Revision v4.0</b>, we unify native server/desktop execution (PyTorch/Metal/Triton), zero-Python browser inference "
        "(<b>20 pure WGSL 1.0 compute shaders</b>), and domain-independent physical signal processing (<code>scar_codec</code>) under one directional <b>Scar Operator</b>: "
        "<b>(1) Unified Spatial Scars (<code>MoS</code>, <code>Sub-2-Bit</code>, <code>VQ4-Scar</code> &amp; <code>ProtectedInt4Scar</code>)</b> shrink 72.71B into <b>20.65 GiB shards / 21.89 GiB VRAM</b> "
        "(&lt;24 GB GPU, cos=0.99911 under 195,720x spikes), break &lt;1.00 GiB on Qwen2.5-3B (<b>0.992 GiB <code>VQ4-Scar</code></b> via multiplier-free <code>FWHT-64</code>, <i>D</i><sub>4</sub> <code>LUT256</code> "
        "with exact (0,0,0,0) vacuum, and 2-pass INT4-ASVD healing), and reach <b>KL=0.6938</b> (74.5% Top-1 across PT/EN/ES/FR, <b>0% CJK leak</b> via 18.55 KB <i>P</i><sub>&perp;CJK</sub>, "
        "<b>cos=1.00000012</b> Native&harr;WGSL parity, ~21 tok/s WebGPU) in 1,601.57 MiB <code>ProtectedInt4Scar</code>; "
        "<b>(2) SCache O(1) &amp; Ghost-Jump Reflex</b> deliver <b>100% Passkey recall at 16.6k and 32.9k tokens</b> in strictly constant <b>22.71 MiB (&Delta;=+0 B, 50.9x smaller, 5.84x faster)</b> "
        "plus lossless speculative rollback (KL=0.0000, 1.89x–1.92x speedup); "
        "<b>(3) Closed-Form 140 KB <code>HeadExpandedScarStore</code> &amp; Language-Nullspace <code>GoldenSliceScarStore v4.0</code></b> achieve 98.61% unseen 72.7B parity (0 SGD), "
        "18 &mu;s RAM / 7.64 ms <code>0x00</code> SSD GDPR unlearning, and <b>164/164 (100.0%)</b> cross-lingual HTTP 206 routing across PT/EN/ES/FR via <i>P</i><sub>&perp;lang</sub> = <b>I</b> &minus; <b>U</b><sub>lang</sub><b>U</b><sub>lang</sub><sup>T</sup> (<i>k</i><sub>lang</sub>=14 at Layer 23/36); "
        "<b>(4) Sinkhorn Monomial–Procrustes Re-Basin</b> cuts shared-base KL by <b>56.36x</b> on specialists and <b>14.68x</b> across 12 layers at 0 extra FLOPs; and "
        "<b>(5) Domain-Independent Physical <code>ScarCodec</code> (<code>mos-scar-codec</code>)</b> spans 12 hardware, storage, aerospace, and scientific domains across 3 controlled benchmark-phantom proofs: "
        "blind noisy-covariance &minus;60 dB (1,000x) radio-telescope RFI recovery (<b>cos=0.9753</b> vs 0.0252 VLBI 2-bit at 8.67x compression, zero oracle steering), <b>34.8 KB (34,816 B, &Delta;=0 B)</b> avionics black-box SCache "
        "(<b>100% 5/5</b> early fault recall vs 0% circular buffer, reconstructing 4 unseen masked channels from a 2-channel cue at <b>cos=0.9999</b>), and [512,512] conservative Rusanov 2D Burgers shockwave PDE/500x-HDR compression (<b>12.05x</b>, <b>188.2x lower RMSE</b> &amp; <b>3,445x lower spike error</b> "
        "than byte-matched FP16 2D <code>rFFT</code>, exact 0.0 vacuum error, 10.2x faster O(1) block read than <code>zlib -9</code>). All <b>114/114 pytest checks</b> across 29 suites pass with <b>0.00 B</b> storage discrepancy.",
        s_abs_box,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=3.6))

    story.append(KeepTogether([
        Image(str(hero_png), width=7.44 * inch, height=2.93 * inch),
        Paragraph(
            "<b>Figure 1.</b> Audited MoS v4.0 + SCache O(1) + Physical ScarCodec Unified Results (0.5B &rarr; 3B &rarr; 7.62B &rarr; 72.71B &amp; Hardware/Science Proofs; 0.00 B discrepancy). "
            "<b>(A)</b> 32,868-token Passkey: SCache O(1) holds at 22.71 MiB (&Delta;=+0 B, 50.9x smaller, 100% recall, 5.84x faster). "
            "<b>(B)</b> Sub-2-Bit MoS shrinks 72.71B to 21.89 GiB VRAM (&lt;24 GB GPU / &lt;32 GB laptop, cos=0.99911). "
            "<b>(C)</b> Unified Native + 20-Shader WebGPU 3B polyglot Pareto (PT/EN/ES/FR): VQ4-Scar breaks &lt;1.00 GiB (0.992 GiB) while ProtectedInt4Scar + P_&perp;CJK reaches KL=0.6938 (0% CJK leak, cos=1.00000012, ~21 tok/s). "
            "<b>(D)</b> 140 KB HTTP 206 cartridges slash 72.7B&rarr;3B KL up to 98x (0 SGD) and hit 164/164 (100%) polyglot routing via P_&perp;lang. "
            "<b>(E)</b> Sinkhorn Re-Basin cuts shared-base KL by 56.4x (specialists), 3.4x (3B pair), and 14.7x (12 layers). "
            "<b>(F)</b> Standalone ScarCodec controlled benchmark proofs: blind &minus;60 dB RFI recovery (97.5% cos), 34.8 KB partial-cue black-box SCache (5/5 recall, masked cos=0.9999), and [512,512] Rusanov Burgers PDE/HDR (188.2x lower RMSE than byte-matched FP16 2D rFFT, 0.0 vacuum error).",
            s_caption,
        ),
    ]))

    story.append(Paragraph("1. Unified Scar Law, &lt;24 GB 72.71B Sub-2-Bit MoS &amp; 20-Shader WebGPU Operators", s_h1))
    story.append(Paragraph(
        "Across pretrained neural weights, long-horizon KV states, and physical sensor arrays, states decompose into a predictable correlated manifold, "
        "exact zero-energy spans, and sparse high-surprise shocks [1, 2, 11]. Revision v4.0 governs all regimes under one operator coupling orthogonal subspace "
        "shielding <b>P</b><sub>&perp;</sub>, equilibrated lattice/block quantization, and one-sided ReLU spherical-cap scars:",
        s_body,
    ))
    story.append(Paragraph(
        "<b>Unified Scar Law (v4.0):</b> &nbsp; <i>F</i>(<b>x</b>) = <i>F</i><sub>base</sub>(<b>P</b><sub>&perp;</sub><b>x</b>) + &Sigma;<sub><i>k</i></sub> "
        "ReLU( cos(<b>P</b><sub>&perp;</sub><b>x</b>, <b>z</b><sub><i>k</i></sub>) &minus; &tau;<sub><i>k</i></sub> ) &middot; &Delta;<sub>scar,<i>k</i></sub>(<b>x</b>) &nbsp;&nbsp; (Eq. 1)",
        s_eq,
    ))
    story.append(Paragraph(
        "<b>Eight Structural Replacements on Existing Pretrained Models:</b> Eq. 1 replaces (1) scalar Gaussian quantization with dual-engine <code>VQ4-Scar</code> / "
        "<code>ProtectedInt4Scar</code>, (2) O(<i>T</i>) KV caches with <code>SCache O(1)</code>, (3) heavy text RAG with 140 KB HTTP 206 latent cartridges, (4) SGD LoRA with "
        "closed-form Woodbury synthesis + <code>0x00</code> GDPR unlearning, (5) resident MoE bloat with streaming <code>MoS</code> (20.65 GiB 72.7B Sub-2-Bit shards, 21.89 GiB VRAM &lt;24 GB GPU), "
        "(6) multilingual model duplication with <i>P</i><sub>&perp;lang</sub> + <i>P</i><sub>&perp;CJK</sub>, (7) verbose test-time chains with lossless <code>Ghost-Jump</code> speculative reflex, and "
        "(8) domain-locked AI codecs with standalone physical <code>ScarCodec</code>.<br/>"
        "<b>Unified Native + 20-Shader WGSL 1.0 Operators (<code>vq4_scar_mos.py</code> &amp; <code>mos_scache_kernels.wgsl</code>):</b> Forensic profiling on polyglot Qwen2.5-3B reveals "
        "extreme weight-vs-activation kurtosis asymmetry (<i>W</i><sub>kurt</sub>=3.4–5.4 vs <i>X</i><sub>kurt</sub>&le;80,252), sink collapse when compressing L0..L4 (KL=11.29), and noise concentration "
        "in {<code>gate_proj</code>, <code>down_proj</code>, <code>o_proj</code>}. Channel equilibration <b>d</b><sub>eq</sub>=(RMS(<b>X</b>)/med)<sup>&beta;</sup> neutralizes outlier columns once per input vector "
        "before 6-stage multiplier-free <code>FWHT-64</code> [20]. <code>LUT256</code> (4 KB in <code>var&lt;workgroup&gt;</code>) locks entry 0&rarr;(0,0,0,0) alongside Conway–Sloane <i>D</i><sub>4</sub> shells [2] "
        "and &alpha;<sub><i>b</i></sub><sup>2</sup>-weighted Lloyd-Max centroids, while 2-pass INT4-ASVD healing compresses residual scars 3.76x smaller than FP16. Shader #18 (<code>ScriptSubspaceGate</code>, "
        "18.55 KB bitmask over 31,112 CJK tokens) eliminates quantization-induced CJK script drift across PT/EN/ES/FR while keeping native ZH 100% intact.",
        s_body,
    ))

    story.append(Paragraph("2. SCache O(1), Ghost-Jump Reflex, P_&perp;lang HTTP 206 Cartridges &amp; Sinkhorn Re-Basin", s_h1))
    story.append(Paragraph(
        "<b>Proposition 1 (SCache O(1) &amp; Lossless Speculative Rollback).</b> Widrow-Hoff fast weights <b>W</b><sub>scar</sub>(<i>t</i>)=&gamma;<b>W</b><sub>scar</sub>(<i>t</i>&minus;1)+&eta;<i>g</i>(<b>v</b><sub>ev</sub>&minus;<b>W</b><sub>scar</sub><b>k</b>)<b>k</b><sup>T</sup> "
        "[17, 21, 23] + synchronized 32-token micro-page routing enforce &Delta;<i>B</i>/&Delta;<i>T</i>=0 (22.71 MiB on 3B at 32.9k; 14.99 MB in WebGPU across infinite turns) and exact speculative parity KL=0.0000 [8].<br/>"
        "<b>Proposition 2 (Closed-Form 140 KB Cartridges &amp; Language-Nullspace Invariance <i>P</i><sub>&perp;lang</sub>).</b> Active-Set Woodbury Rosetta inversion + AlphaEdit nullspace [5] mints exact "
        "143,360 B (140.00 KB = 35&times;4 KB) slots in 0 SGD over 4,194,304 Kronecker caps [6]. In <code>GoldenSliceScarStore v4.0</code> (<code>golden_slice_v4.py</code>), projecting Layer-23 hidden states onto "
        "<b>P</b><sub>&perp;lang</sub> = <b>I</b> &minus; <b>U</b><sub>lang</sub><b>U</b><sub>lang</sub><sup>T</sup> (<i>k</i><sub>lang</sub>=14) makes routing translation-invariant (<b>P</b><sub>&perp;lang</sub><b>h</b><sub>PT</sub> = <b>P</b><sub>&perp;lang</sub><b>h</b><sub>EN</sub>), "
        "achieving <b>164/164 (100.0%)</b> HTTP 206 accuracy across PT/EN/ES/FR with 0 out-of-cone bleed.<br/>"
        "<b>Proposition 3 (SwiGLU Monomial Gauge &amp; Tiled Log-Sinkhorn).</b> Folding &alpha;<sub><i>i</i></sub><sup>&minus;1</sup> into <code>transposed_grid=True</code> FP16 scales aligns SwiGLU specialists "
        "and cross-layer pairs [1, 17] in &lt;350 MiB tile VRAM at 72.7B with 0 extra runtime FLOPs.",
        s_prop,
    ))

    story.append(Paragraph("3. Domain-Independent Physical ScarCodec &amp; 12 Hardware/Space/Science Applications", s_h1))
    story.append(Paragraph(
        "<b>Proposition 4 (Exact Vacuum Preservation, O(1) Block Read &amp; Subspace RFI Immunity).</b> Standalone package <code>scar_codec</code> (<code>mos-scar-codec</code>, zero <code>transformers</code> dependency) "
        "applies <b>P</b><sub>&perp;int</sub> = <b>I</b> &minus; <b>U</b><sub>int</sub><b>U</b><sub>int</sub><sup>T</sup> [7], MAD point-shock peeling, <code>VQ4-Scar</code>, and <code>SurpriseGatedAnomalySCache</code> to physical tensors: "
        "(i) pure-vacuum 64-sample blocks map to code 0&rarr;(0,0,0,0) with zeroed scar row-scales for <b>exact 0.0 vacuum error</b> (zero Gibbs ringing); (ii) any block (<i>r</i>,<i>b</i>) decodes in <b>O(1)</b> time "
        "without full-stream decompression; and (iii) signals orthogonal to colspan(<b>U</b><sub>int</sub>) suffer zero attenuation while preventing quantizer saturation under &minus;60 dB SIR (1,000x RFI).",
        s_prop,
    ))
    story.append(Paragraph(
        "Because <code>FWHT-64</code> requires zero hardware multipliers (1–2 adder cycles in FPGA/ASIC) and <code>LUT256</code> fits in 2 KB SRAM, <code>ScarCodec</code> directly targets <b>12 physical domains</b>: "
        "(1) QLC/PLC NAND Flash <i>D</i><sub>4</sub> modulation &amp; wear scars, (2) NVMe O(1) random-read float compression &amp; HDD ITI cancellation, (3) inline IMC/CXL DRAM cache-line compression, "
        "(4) GPU 3D Gaussian Splatting &amp; HDR lightmaps, (5) CubeSat/SAR adder-only onboard compression &amp; scar-first downlink, (6) SKA/VLBI radio-telescope <i>P</i><sub>&perp;RFI</sub> correlators &amp; ELT &lt;100 &mu;s SRAM adaptive optics, "
        "(7) wireless ultrasound RF &amp; diagnostic-scar CT/MRI, (8) subsea DAS fiber seismology &amp; smart-grid phasors, (9) 5G/6G Massive MIMO CSI feedback, (10) 34.8 KB perpetual avionics/industrial black-boxes, "
        "(11) Cryo-EM &amp; single-cell RNA-seq rare-marker preservation, and (12) Gibbs-free supersonic CFD checkpoints &amp; H-matrix solvers.",
        s_body,
    ))

    story.append(Paragraph("4. Multi-Scale AI &amp; Physical Empirical Audit (114/114 Pytest Checks Across 29 Suites)", s_h1))

    t1 = [
        ["Model / Pair", "Workload & Context", "Engine", "Audited KV Memory (0 B Drift)", "Throughput", "Recall / Parity", "Gain vs. Baseline"],
        ["Qwen2.5-3B (36L)", "16,568 tok Passkey", "Normal KV / SCache O(1)", "582.75 -> 22.71 MiB (+0 B)", "245.3 -> 299.0 tok/s", "100% (5/5)", "25.7x smaller · 1.22x faster"],
        ["Qwen2.5-3B (36L)", "32,868 tok Passkey", "Normal KV / StreamingLLM", "1,155.8 / 9.0 MiB", "51.5 / 391.4 tok/s", "100% / 0%", "O(T^2) stall / Total amnesia"],
        ["Qwen2.5-3B (36L)", "32,868 tok Passkey", "SCache O(1) (W=256, S=256)", "22.71 MiB (+0 B vs 16.6k!)", "300.8 tok/s (109.3s)", "100% (3/3)", "50.9x smaller · 5.84x faster!"],
        ["Qwen2.5-0.5B (24L)", "32,868 tok Passkey", "SCache O(1) (W=256, S=256)", "6.86 MiB (+0 B vs 16.6k!)", "1,176.9 tok/s (27.9s)", "100% (5/5)", "56.2x smaller · 1.60x faster"],
        ["0.5B -> 3B Ghost-Jump", "Python / Math (96 tok)", "Speculative + SCache O(1)", "3.12 + 11.61 MiB (0 B leak)", "23.65 / 23.30 tok/s", "96/96 (KL=0.0000)", "1.89x–1.92x (4.8–5.05 tok/swp)"],
    ]
    story.append(KeepTogether([
        styled_table(t1, [1.14 * inch, 1.08 * inch, 1.30 * inch, 1.26 * inch, 0.94 * inch, 0.80 * inch, 0.96 * inch], s_th, s_td, bold_rows=(1, 3, 4, 5)),
        Paragraph("<b>Table 1 (32.9k Passkey Audit &amp; Ghost-Jump Speculative Reflex):</b> Zero fine-tuning, 0.00 B storage-pointer discrepancy across all runs.", s_caption),
    ]))

    t2 = [
        ["Scale & Compression Regime", "Hardware / Kernel Path", "Shards / Resident Footprint", "SCache KV", "Fidelity / PPL", "Decode Speed", "Key Architectural Result"],
        ["72.71B Dense BF16 + Full KV", "Multi-A100/H100 Cluster", "135.43 GiB (866 GiB 8x)", "10.7 GiB @ 32k", "PPL 4.35 (Ref)", "Cluster Ref", "Requires 2x80 GB datacenter GPUs"],
        ["72.71B MoS v1.2 (INT4+ASVD)", "Apple Metal / CUDA Triton", "38.12 GiB (1.79 GiB/shard)", "128.5 MiB O(1)", "100% Top-5, PPL 4.75", "3.46–7.27+ tok/s", "50.38x Metal int4pack (1.92 GiB @ 16GB)"],
        ["72.71B MoS Sub-2-Bit", "CUDA W2A16+ASVD / Metal", "20.65 GiB / 21.89 GiB VRAM", "128.5 MiB O(1)", "L79 cos=0.99911 (2.49b)", "Single 24 GB GPU!", "Fits <24 GB RTX 3090/4090 & <32 GB Mac!"],
        ["3B Sub-2-Bit (108 MLPs)", "Naive 2-Bit vs MoS Sub-2-Bit", "4,644 -> 997.3 MiB (4.66x)", "O(1)", "1.35e6 -> 109.99 PPL", "0-Data Salience", "12,290x PPL reduction vs naive 2-bit"],
        ["3B / 0.5B Zero-Spike Student", "Metal int4pack + ASVD (r=128)", "3.49 / 0.78 GiB VRAM (0.55G RSS)", "10.0 MiB O(1)", "cos=0.9926 vs BF16", "17.1 tok/s (4.1s)", "Eliminates 12.4 GiB 2x CPU->GPU spike!"],
    ]
    story.append(KeepTogether([
        styled_table(t2, [1.34 * inch, 1.18 * inch, 1.18 * inch, 0.72 * inch, 0.96 * inch, 0.78 * inch, 1.32 * inch], s_th, s_td, bold_rows=(2, 3, 4, 5)),
        Paragraph("<b>Table 2 (72.71B Multi-Hardware &amp; Sub-2-Bit Residency Audit):</b> Neutralizes 195,720x SwiGLU outliers while breaking 24 GB GPU &amp; 32 GB Mac walls.", s_caption),
    ]))

    t3 = [
        ["Audit Dimension", "Domain / Polyglot Suite", "Raw 3B Student", "0.5B + 61.25 KB Scar", "3B + 140.00 KB Scar (35x4KB)", "Verified System Guarantee"],
        ["Unseen Paraphrase vs 72B", "Sphere / Landauer / Immerman", "87.5% mean (KL 0.49)", "94.4% mean (KL 0.17)", "98.61% (71/72; Landauer 97.7x down)", "0 autograd steps (<4.4s; k_B T ln 2)"],
        ["Out-of-Cone & Haystack", "12 Controls/Traps + 1,560t Rumor", "Hallucinates k_B T log m", "24/24 torch.equal (0 B)", "36/36 torch.equal · exact kT ln 2", "0 false triggers · 0.0 B KV drift"],
        ["Multi-Scar & Cloud HTTP", "3-Way Synthesis + HTTP 206", "Collapses / Uninstalled", "Spread 0.918 (188 KB)", "Spread 0.970 · HTTP 206: 58.3 ms", "GDPR Wipe: 18 us RAM / 7.64 ms 0x00 SSD"],
        ["GoldenSlice v4.0 (P_perp_lang)", "41 Cartridges x PT/EN/ES/FR", "Hallucinates / Missing", "61.25 KB Dual-Depth", "164/164 (100.0%) Polyglot Match", "12/12 Open Controls (0 bleed) WebGPU!"],
    ]
    story.append(KeepTogether([
        styled_table(t3, [1.14 * inch, 1.28 * inch, 0.96 * inch, 1.00 * inch, 1.44 * inch, 1.66 * inch], s_th, s_td, bold_rows=(1, 2, 3, 4)),
        Paragraph("<b>Table 3 (Closed-Form 140 KB HeadExpandedScarStore &amp; Language-Nullspace GoldenSliceScarStore v4.0 Across PT/EN/ES/FR).</b>", s_caption),
    ]))

    t4 = [
        ["Scale & Shared Regime", "Shape & Compression", "Naive Unaligned", "Sinkhorn Monomial (0 FLOP)", "Sinkhorn–Procrustes", "KL / Error Gain", "Spectral / Geometric Finding"],
        ["0.5B Permuted Specialist", "[4864, 896] (Git Re-Basin)", "KL=0.1578 (87.5% Top1)", "—", "KL=0.0028 (100% Top5)", "56.36x lower KL", "Alignment cos: 0.0678 -> 0.9817"],
        ["3B Interior pair_L22_L23", "258.0 -> 50.65 MB (5.09x)", "KL=0.1190 (88.4% Top1)", "KL=0.0395 < 0.08 (99.8%)", "KL=0.0353 < 0.08 (92.6%)", "3.37x lower KL", "Surpasses <0.08 target at 0 FLOPs"],
        ["3B 12-Layer L18..L29 (1/3)", "1,548 -> 303.9 MB (5.09x)", "KL=6.4634 (21.3% Top1)", "KL=0.4403 (95.02% Top5)", "—", "14.68x lower KL", "Eliminates multi-layer collapse"],
        ["72.71B Shard Pair L78_L79", "2,772 -> 445.2 MiB (6.23x)", "Out cos=0.1626 (err 1.19)", "Tiled Log-Sinkhorn (<350MB)", "Out cos=0.8835 (err 0.44)", "109.2x inner gain", "MP best cos=0.1228 vs floor 0.0501"],
    ]
    story.append(KeepTogether([
        styled_table(t4, [1.14 * inch, 1.02 * inch, 1.02 * inch, 1.12 * inch, 1.12 * inch, 0.78 * inch, 1.28 * inch], s_th, s_td, bold_rows=(1, 2, 3, 4)),
        Paragraph("<b>Table 4 (Sinkhorn Monomial–Procrustes Re-Basin &amp; Residual Stream Marchenko-Pastur Orthogonality Law).</b>", s_caption),
    ]))

    t5 = [
        ["Unified Regime (Native PyTorch <-> 20-Shader WebGPU WGSL 1.0)", "Footprint", "Comp.", "4-Lang Mean KL", "Top-1", "WGSL Parity", "Polyglot Audit (PT / EN / ES / FR)"],
        ["Legacy Scalar Gaussian 2/3-Bit / NF3 + WHT", "1,027 / 1,196 MiB", "5.9x / 5.0x", "3.7905 / 2.9290", "27.6–38.7%", "—", "PT loops; FR drifts into English"],
        ["Unshielded Block-64 INT4 (Zero ASVD Scars)", "1,566.7 MiB", "3.84x", "1.0583", "72.8%", "cos > 0.999999", "Leaks Chinese CJK at FR tok ~90"],
        ["Sub-1GB VQ4-Scar (LUT256 + FWHT-64 + 2-Pass INT4)", "1,016.0 MiB (0.992G!)", "5.92x", "1.5561", "59.3% (PT 64%)", "cos=1.0000001", "Breaks <1.00 GiB; all 4 langs stable!"],
        ["Track 2.4-C Shielded VQ4-Scar + Shader #18 P_perp_CJK", "1,190.5 MiB (1.163G)", "5.18x", "1.5240 (PT 1.39)", "59.3% (PT 64.5%)", "cos=1.00000012", "0% CJK leak (8/8), 100% LangPure + ZH!"],
        ["Track 2.5 ProtectedInt4Scar (40 <=44 MiB WebGPU Shards)", "1,601.6 MiB (1.564G)", "3.76x", "0.6938", "74.5% (ES 80.4%)", "cos=1.0000000", "~21 tok/s WebGPU · 14.99 MB SCache (+0 B)!"],
    ]
    story.append(KeepTogether([
        styled_table(t5, [1.86 * inch, 0.98 * inch, 0.56 * inch, 0.84 * inch, 0.72 * inch, 0.80 * inch, 1.72 * inch], s_th, s_td, bold_rows=(3, 4, 5)),
        Paragraph("<b>Table 5 (Unified Native + 20-Shader WebGPU WGSL 1.0 Polyglot Audit on Qwen2.5-3B-Instruct Across PT, EN, ES, FR).</b>", s_caption),
    ]))

    t6 = [
        ["Physical Benchmark & Geometry", "Conventional Baseline", "Baseline Failure Mode", "ScarCodec v4.0 Stack", "Compressed Size", "Audited Physical Result"],
        ["1. Radio Telescope [64, 4096] (-60 dB SIR, 1,000x RFI)", "Raw Beam / 2-Bit VLBI + Null", "cos=0.0044 / 0.0252 (0.0 dB: clipped RFI aliases)", "Blind P_perp_RFI (r=2) + VQ4-Scar (FWHT-64 + 2-Pass INT4)", "120,960 B (8.67x vs 1.00 MiB)", "cos=0.97527 (+13.11 dB SNR) · Out-of-cone err 2.75e-4!"],
        ["2. Avionics / CubeSat Black-Box (50,000 x 64, 12.8 MB)", "Equal-Byte Circular Ring O(T) (256 frames)", "0/5 (0.0% recall: overwritten 195x by cruise)", "SurpriseGatedAnomalySCache (32 slots + W_scar)", "34,816 B (34.8 KB, Delta=0 B!)", "5/5 (100.0% recall) · Masked 4-ch cos=0.9999 · 37,501 FPS!"],
        ["3. Rusanov 2D Burgers PDE + 40% Vac + 500x HDR [512,512]", "zlib -9 (1.87x, no O(1)) / Byte-Matched FP16 2D rFFT", "rFFT RMSE=5.742, vac ring 5.44, HDR err 418.34", "MAD Spike Peel + VQ4-Scar (Block-64 O(1) Layout)", "87,040 B (12.05x vs 1.00 MiB)", "RMSE=0.0305 (188.2x down), vac=0.0, HDR 3,445x down, 10.2x Rd!"],
    ]
    story.append(KeepTogether([
        styled_table(t6, [1.26 * inch, 1.06 * inch, 1.18 * inch, 1.24 * inch, 0.96 * inch, 1.78 * inch], s_th, s_td, bold_rows=(1, 2, 3)),
        Paragraph("<b>Table 6 (Domain-Independent Physical ScarCodec Audit — <code>results/scar_codec_physical_v4_0_audit.json</code>).</b>", s_caption),
    ]))

    story.append(Paragraph("5. Honest Limitations, Open Directions &amp; Reproducibility", s_h1))
    story.append(Paragraph(
        "(1) <b>Pretrained Gaussian Backbone Floor:</b> Applying post-training scar operators to existing dense checkpoints still inherits a pretrained backbone whose syntax, "
        "reasoning, and world facts remain entangled across dense weights (~0.99–1.56 GiB on 3B), bounding multi-step deduction by the frozen base model's depth. "
        "(2) <b>Shannon Slot Budget &amp; Automated Corpus Segmentation (<code>mos-digest</code>):</b> Fixed <i>S</i><sub>slots</sub> achieves 100% single/few-needle recall at 32.9k tokens (+0 B drift), "
        "whereas verbatim recall of hundreds of uncorrelated random hashes at 128k–1M tokens requires hierarchical disk-backed micro-page paging, and ingesting 500-page manuals requires automated topic segmentation. "
        "(3) <b>RTL Silicon Tape-Out &amp; Cross-Family Tokenizer Bridges:</b> Synthesizing physical Verilog/VHDL FPGA bitstreams for NAND/CubeSat controllers and scaling cross-tokenizer bridges across distant vocabularies remain active engineering frontiers. "
        "Full source code, <b>114/114 pytest checks across 29 suites</b>, 7 CLIs (<code>mos-run</code>, <code>mos-store</code>, <code>mos-graft</code>, <code>mos-rebasin</code>, <code>mos-transcode-sub2bit</code>, <code>mos-webgpu</code>, <code>mos-scar-codec</code>), "
        "20 WGSL 1.0 shaders, interactive <i>Physical Scar Lab</i>, and JSON telemetry are archived under MIT/CC-BY-4.0 at <b>DOI: 10.5281/zenodo.24172784</b> [13, 14].",
        s_body,
    ))

    story.append(Paragraph("References", s_h1))
    refs = [
        "[1] Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. ICLR 2023.",
        "[2] Conway, J. H., & Sloane, N. J. A. (1982). Fast Quantizing and Decoding Algorithms for Lattice Quantizers and Codes. IEEE Trans. Inf. Theory 28(2), 227–232.",
        "[3] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 35, 30318–30332.",
        "[4] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 36.",
        "[5] Fang, J., Jiang, H., Wang, K., Ma, Y., et al. (2025). AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. ICLR 2025.",
        "[6] Kabatiansky, G. A., & Levenshtein, V. I. (1978). Bounds for Packings on a Sphere and in Space. Problems of Information Transmission 14(1), 1–17.",
        "[7] Leshem, A., van der Veen, A.-J., & Boonstra, A.-J. (2000). Multichannel Radio Astronomical Imaging in the Presence of Strong Radio-Frequency Interference. Astrophys. J. Suppl. Ser. 131(1), 355–373.",
        "[8] Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML, 19274–19286.",
        "[9] Li, Y., Yu, Y., Liang, C., He, P., et al. (2024). LoftQ: LoRA-Fine-Tuning-Aware Quantization for LLMs. ICLR 2024.",
        "[10] Lin, J., Tang, J., Tang, H., Yang, S., et al. (2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression. MLSys 6.",
        "[11] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of Eigenvalues for Some Sets of Random Matrices. Math. USSR-Sbornik 1(4), 457–483.",
        "[12] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 35, 17359–17372.",
        "[13] Reis Filho, J. C. (2026a). Mixture of Scars (MoS) v4.0, SCache O(1), and Domain-Independent ScarCodec. Zenodo Preprint. DOI: 10.5281/zenodo.24172784.",
        "[14] Reis Filho, J. C. (2026b). Qwen2.5-72B-Instruct MoS v2 Checkpoint & Practical Deployment Guide. URL: https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos.",
        "[15] Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML, PMLR 139, 9355–9366.",
        "[16] Schmidhuber, J. (1992). Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation 4(1), 131–139.",
        "[17] Sinkhorn, R., & Knopp, P. (1967). Concerning Nonnegative Matrices and Doubly Stochastic Matrices. Pacific J. Math. 21(2), 343–348.",
        "[18] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS 30.",
        "[19] W3C GPU for the Web Working Group (2024). WebGPU and WebGPU Shading Language (WGSL) Specification. World Wide Web Consortium (W3C).",
        "[20] Walsh, J. L. (1923). A Closed Set of Normal Orthogonal Functions. American Journal of Mathematics 45(1), 5–24.",
        "[21] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits. IRE WESCON Convention Record, Part 4, 96–104.",
        "[22] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024.",
        "[23] Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 37.",
        "[24] Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., & Sun, G. (2023). ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs. arXiv:2312.05821.",
        "[25] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 36.",
    ]
    for r in refs:
        story.append(Paragraph(escape(r), s_ref))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def sync_huggingface_space():
    hf_sc = REPO_ROOT / "huggingface_space" / "scar_cache"
    if hf_sc.exists():
        for src_py in sorted((REPO_ROOT / "scar_cache").glob("*.py")):
            shutil.copy2(src_py, hf_sc / src_py.name)
    hf_codec = REPO_ROOT / "huggingface_space" / "scar_codec"
    if (REPO_ROOT / "scar_codec").exists():
        hf_codec.mkdir(parents=True, exist_ok=True)
        for src_py in sorted((REPO_ROOT / "scar_codec").glob("*.py")):
            shutil.copy2(src_py, hf_codec / src_py.name)
    print(f"[HF SPACE SYNC] Synced scar_cache & scar_codec -> {REPO_ROOT / 'huggingface_space'}")


def audit_latex_and_pack(zip_out: Path):
    v4_dir = LATEX_DIR / "revisions" / "v4_0"
    tex = (v4_dir / "main.tex").read_text(encoding="utf-8")
    bib = (v4_dir / "references.bib").read_text(encoding="utf-8")
    bbl = (v4_dir / "main.bbl").read_text(encoding="utf-8")

    # Enforce strict v4.0 scope boundary (zero mention of v5.0 Pure Scar-Native)
    for name_check, content_check in (("main.tex", tex), ("references.bib", bib), ("main.bbl", bbl)):
        assert not re.search(r"scar[-_\s]native", content_check, flags=re.IGNORECASE), (
            f"Forbidden v5.0 scope term found in {name_check}"
        )

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
    assert bib_keys == bbl_keys, f".bib / .bbl mismatch: {bib_keys ^ bbl_keys}"

    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", clean_tex))
    assert not (refs - labels), f"Missing refs: {refs - labels}"

    fig_pdf = LATEX_DIR / "figures" / "fig1_mos_scache_v4_audit.pdf"
    (v4_dir / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_pdf, v4_dir / "figures" / "fig1_mos_scache_v4_audit.pdf")

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_name in ("Makefile", "main.tex", "references.bib", "main.bbl", "figures/fig1_mos_scache_v4_audit.pdf"):
            fp = v4_dir / rel_name
            if fp.exists():
                zf.write(fp, arcname=rel_name)
    print(f"[ZENODO v4.0 LATEX AUDIT PASS] {len(cited)}/{len(bbl_keys)} bbl cites | {len(labels)} labels | {zip_out.name}")


def build_replication_zip_v4_0(zip_path: Path):
    files = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "docs" / "AUDIT_AND_VERIFICATION_GUIDE.md",
        REPO_ROOT / "docs" / "BEYOND_AI_SCAR_ARCHITECTURE_APPLICATIONS.md",
        REPO_ROOT / "docs" / "GUIA_PRATICO_MOS_SCACHE_72B.md",
        REPO_ROOT / "docs" / "ONLINE_CARTRIDGE_LIBRARY.md",
        REPO_ROOT / "docs" / "ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md",
        REPO_ROOT / "docs" / "ROADMAP_V4_PORTFOLIO_3B_MOS_AND_VECTOR_SCARS.md",
        REPO_ROOT / "docs" / "SCAR_STORE_V2_ARCHITECTURE.md",
        REPO_ROOT / "docs" / "UNIFIED_SCAR_OPERATORS_AND_8_SOLUTIONS.md",
        REPO_ROOT / "mixture_of_scars" / "__init__.py",
        REPO_ROOT / "scripts" / "adversarial_mos_scache_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_7b_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_72b_suite.py",
        REPO_ROOT / "scripts" / "adversarial_core_0p5b_vs_3b_suite.py",
        REPO_ROOT / "scripts" / "benchmark_16k_passkey_llm.py",
        REPO_ROOT / "scripts" / "benchmark_72b_apple_silicon.py",
        REPO_ROOT / "scripts" / "benchmark_horizon1_v1_4.py",
        REPO_ROOT / "scripts" / "benchmark_scar_grafting_v2.py",
        REPO_ROOT / "scripts" / "benchmark_scar_codec_physical_v4_0.py",
        REPO_ROOT / "scripts" / "audit_scar_store_v2_stage5_3b_vs_0p5b.py",
        REPO_ROOT / "scripts" / "audit_mos_store_cli_v2_3.py",
        REPO_ROOT / "scripts" / "audit_sinkhorn_rebasin_v2_4.py",
        REPO_ROOT / "scripts" / "build_online_cartridge_library_v3_0.py",
        REPO_ROOT / "scripts" / "build_universal_scarstore_v4_0.py",
        REPO_ROOT / "scripts" / "test_track24_full_3b_native_and_webgpu.py",
        REPO_ROOT / "scripts" / "verify_track24c_shielded_3b_polyglot.py",
        REPO_ROOT / "scripts" / "generate_fig1_multiscale_v4_0.py",
        REPO_ROOT / "scripts" / "build_zenodo_v4_0_kit.py",
        REPO_ROOT / "benchmark_results" / "adversarial_mos_scache_report.json",
        REPO_ROOT / "results" / "passkey_16k_32k_llm_audit.json",
        REPO_ROOT / "results" / "horizon1_v1_4_audit.json",
        REPO_ROOT / "results" / "horizon2_scar_grafting_v2_audit.json",
        REPO_ROOT / "results" / "scar_store_v2_stage5_3b_vs_0p5b_audit.json",
        REPO_ROOT / "results" / "adversarial_0p5b_vs_3b_audit.json",
        REPO_ROOT / "results" / "mos_store_v2_3_cli_http206_audit.json",
        REPO_ROOT / "results" / "online_cartridge_library_v3_0_audit.json",
        REPO_ROOT / "results" / "online_library_v3_0" / "scarstore_catalog_v3_0.json",
        REPO_ROOT / "results" / "online_library_v3_0" / "qwen3b_portfolio_v4_3_prot_int4.manifest.json",
        REPO_ROOT / "results" / "sinkhorn_rebasin_v2_4_audit.json",
        REPO_ROOT / "results" / "webgpu_wgsl_v3_0_audit.json",
        REPO_ROOT / "results" / "scar_codec_physical_v4_0_audit.json",
        REPO_ROOT / "results" / "scale_7b_audit" / "adversarial_scale_7b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "adversarial_scale_72b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_live_72b_generation.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_72b_speed_memory_audit.json",
    ]
    files.extend(sorted((REPO_ROOT / "scar_cache").glob("*.py")))
    files.extend(sorted(p for p in (REPO_ROOT / "scar_cache" / "webgpu").rglob("*") if p.is_file() and not p.name.startswith(".")))
    files.extend(sorted((REPO_ROOT / "scar_codec").glob("*.py")))
    files.extend(sorted((REPO_ROOT / "tests").glob("test_*.py")))
    seen = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                arc = str(fp.relative_to(REPO_ROOT))
                if arc not in seen:
                    seen.add(arc)
                    zf.write(fp, arcname=arc)


def main():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "generate_fig1_multiscale_v4_0.py")])
    sync_huggingface_space()

    src_png = LATEX_DIR / "figures" / "fig1_mos_scache_v4_audit.png"
    hero_png = OUT_DIR / "mos_scache_v4_0_hero.png"
    shutil.copy2(src_png, hero_png)

    for stale_f in ("arXiv_Direct_Upload_v4.0.zip", "MoS_SCache_LaTeX_arXiv_TMLR_v4.0.zip"):
        if (OUT_DIR / stale_f).exists():
            (OUT_DIR / stale_f).unlink()

    pdf_v40 = OUT_DIR / "Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf"
    rep_v40 = OUT_DIR / "MoS_SCache_Adversarial_Replication_v4.0.zip"
    tex_v40 = OUT_DIR / "MoS_SCache_LaTeX_Source_v4.0.zip"

    build_pdf_v40_ok = build_pdf_v4_0(pdf_v40, hero_png)
    _ = build_pdf_v40_ok
    build_replication_zip_v4_0(rep_v40)
    audit_latex_and_pack(tex_v40)
    import json as _json
    (OUT_DIR / "ZENODO_METADATA_v4.0.json").write_text(
        _json.dumps(
            {
                "upload_type": "publication",
                "publication_type": "preprint",
                "version": "4.0.0",
                "publication_date": "2026-11-16",
                "title": "Mixture of Scars (MoS) v4.0, SCache O(1), and Domain-Independent ScarCodec: Unified Native & 20-Shader WebGPU Scar Operators, <24 GB 72.7B Compression, Language-Invariant 140 KB HTTP 206 Cartridges, and Forensic Signal/PDE Proofs",
                "creators": [{"name": "Reis Filho, Julio Cesar da Costa", "affiliation": "Byte-OD Research & System One Architecture Lab"}],
                "license": "cc-by-4.0",
                "keywords": ["Mixture of Scars", "SCache O(1)", "VQ4-Scar", "WebGPU WGSL", "ScarCodec", "Widrow-Hoff Fast Weights", "Nullspace Projection"],
                "related_identifiers": [{"identifier": "https://github.com/byte-od/mixture-of-scars", "relation": "isSupplementTo", "scheme": "url"}],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"✓ Built {pdf_v40} ({pdf_v40.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Built {rep_v40} ({rep_v40.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
