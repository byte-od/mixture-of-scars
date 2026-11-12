#!/usr/bin/env python3
"""
Builds the Official v1.3.0 Publication Deposit Kit (`zenodo_deposit_v1_3/`):
  1. Generates `paper_latex/figures/fig1_mos_scache_audit.pdf` / `.png` + `zenodo_deposit_v1_3/mos_scache_v1_3_hero.png`
  2. Syncs `huggingface_space/scar_cache/*.py` with `scar_cache/*.py`
  3. Audits `paper_latex/main.tex` & `references.bib` (zero syntax/brace/citation/ref issues)
  4. Builds `zenodo_deposit_v1_3/Mixture_of_Scars_and_SCache_Preprint_v1.3.pdf`
  5. Builds `zenodo_deposit_v1_3/MoS_SCache_Adversarial_Replication_v1.3.zip`
  6. Builds `zenodo_deposit_v1_3/MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip`
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
OUT_DIR = REPO_ROOT / "zenodo_deposit_v1_3"
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
        ("LEFTPADDING", (0, 0), (-1, -1), 4.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4.0),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    for br in bold_rows:
        style_cmds.append(("BACKGROUND", (0, br), (-1, br), colors.HexColor("#EBF3FF")))
    t.setStyle(TableStyle(style_cmds))
    return t


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        0.58 * inch,
        0.34 * inch,
        "Reis Filho (2026) — Mixture of Scars (MoS) & SCache O(1) v1.3.0 (32.9k NIAH & 72.71B Metal/CUDA) | DOI: 10.5281/zenodo.24143839",
    )
    canvas.drawRightString(LETTER[0] - 0.58 * inch, 0.34 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_v1_3(pdf_path: Path, hero_png: Path):
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=12.6,
        leading=15.8, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=4
    )
    s_meta = ParagraphStyle(
        "PaperMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.6,
        leading=11.6, alignment=TA_CENTER, textColor=colors.HexColor("#334155"), spaceAfter=6
    )
    s_abs_box = ParagraphStyle(
        "AbstractText", parent=styles["Normal"], fontName="Helvetica", fontSize=8.25,
        leading=11.3, alignment=TA_JUSTIFY, leftIndent=10, rightIndent=10, spaceAfter=5,
        textColor=colors.HexColor("#1E293B")
    )
    s_h1 = ParagraphStyle(
        "SecH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=10.0,
        leading=12.4, textColor=colors.HexColor("#1B365D"), spaceBefore=5.2, spaceAfter=2.4,
        keepWithNext=True,
    )
    s_body = ParagraphStyle(
        "BodyJustified", parent=styles["Normal"], fontName="Helvetica", fontSize=8.35,
        leading=12.0, alignment=TA_JUSTIFY, spaceAfter=3.4, textColor=colors.HexColor("#1E293B")
    )
    s_eq = ParagraphStyle(
        "Equation", parent=styles["Normal"], fontName="Helvetica", fontSize=8.25,
        leading=13.2, alignment=TA_CENTER, backColor=colors.HexColor("#F8FAFC"),
        borderPadding=4.2, spaceBefore=3.5, spaceAfter=4.2, textColor=colors.HexColor("#0F172A")
    )
    s_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.55,
        leading=9.8, alignment=TA_LEFT, textColor=colors.HexColor("#475569"), spaceBefore=1.8, spaceAfter=4.2
    )
    s_th = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=7.05,
        leading=8.7, alignment=TA_LEFT, textColor=colors.white
    )
    s_td = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica", fontSize=6.95,
        leading=8.6, alignment=TA_LEFT, textColor=colors.HexColor("#0F172A")
    )
    s_ref = ParagraphStyle(
        "RefItem", parent=styles["Normal"], fontName="Helvetica", fontSize=7.25,
        leading=9.2, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=1.3,
        textColor=colors.HexColor("#334155")
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.46 * inch,
        bottomMargin=0.48 * inch,
        title="Mixture of Scars (MoS) and SCache O(1) v1.3.0: 32.9k Passkey Recall, Fused Metal/CUDA Kernels & 72.71B Pareto",
        author="Julio Cesar Reis Filho",
    )

    story = []
    story.append(Paragraph(
        "Mixture of Scars (MoS) and SCache O(1):<br/>"
        "Breaking the Dual Memory Wall Across 0.5B &rarr; 7.62B &rarr; 72.71B Transformers via<br/>"
        "Fused Activation-Aware Scar Experts and 32k Widrow-Hoff Associative Memory",
        s_title,
    ))
    story.append(Paragraph(
        "<b>Julio Cesar Reis Filho</b> — Byte-OD Research &amp; System One Architecture Lab<br/>"
        "Preprint Technical Report <b>v1.3.0</b> (16.6k–32.9k Passkey Recall, Fused Metal/CUDA W4A16+ASVD &amp; 72.71B Pareto) — November 12, 2026<br/>"
        "<b>DOI:</b> 10.5281/zenodo.24143839 &nbsp;|&nbsp; <b>72B Download &amp; Practical Guide:</b> github.com/byte-od/mixture-of-scars (<code>releases/tag/v1.2.0-qwen72b-mos</code>)",
        s_meta,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=4.2))

    story.append(Paragraph(
        "<b>Abstract.</b> Autoregressive Large Language Models (LLMs) face a compounding <i>Dual Memory Wall</i>: "
        "(1) a static parameter wall O(<i>N</i>) requiring up to 145.4 GB (135.43 GiB) of VRAM at 72.7B scale, and (2) a dynamic context wall "
        "O(<i>T</i>) where KV caches grow linearly with sequence length <i>T</i> while incurring quadratic O(<i>T</i><sup>2</sup>) attention stalls. "
        "Existing post-training methods either degrade under emergent SwiGLU activation outliers (surging from 3,336x at 7.62B to <b>195,720x at 72.71B</b>), "
        "suffer permanent eviction amnesia (StreamingLLM), or stream dense weights per token (AirLLM/mmap) at sub-token-per-second I/O ceilings. "
        "We introduce two training-free primitives: <b>Mixture of Scars (MoS)</b> (canonical Block-64 INT4 bases + Activation-Aware SVD 'scar' experts + "
        "20&times;4-layer shards + fused Apple Metal <code>_weight_int4pack_mm</code> [<b>50.38x faster</b> compute at cos=1.000000] and NVIDIA CUDA Triton "
        "<code>W4A16+ASVD</code> kernels) and <b>SCache</b> (drop-in O(1) cache combining attention sinks, a rolling FIFO window, synchronized deep-layer "
        "32-token micro-page routing, and per-KV-head Widrow-Hoff Delta fast weights). Audited with recursive storage-pointer verification (0.00 B discrepancy) "
        "across <b>0.5B, 3B, 7.62B, and 72.71B</b> models, SCache achieves <b>100% Passkey recall at 16,568 and 32,868 tokens</b> on non-repeating Python "
        "documentation (Qwen2.5-3B &amp; 0.5B, vs. 0% StreamingLLM) in a strictly constant <b>22.71 MiB (&Delta;=+0 B vs. 16.6k; 50.9x smaller than 1,155.8 MiB "
        "Normal KV)</b> while running <b>5.84x faster (300.8 vs. 51.5 tok/s)</b>. At 72.71B (80L), MoS shrinks weights 135.43 &rarr; 38.12 GiB (-96.37% / 27.59x "
        "across 8 specialists) with <b>100.0% Top-5 agreement</b>, cuts tail PPL by 5.00x vs. StreamingLLM, and reaches non-repeating PPL <b>4.75 vs. 4.35 Dense FP16</b> "
        "at 4.03–7.27+ tok/s on CUDA and 3.46–4.87 tok/s resident on 48–64 GB Apple Silicon (155x–280x faster than NVMe FP16 offloading).",
        s_abs_box,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=4))

    story.append(Paragraph("1. Mathematical Architecture &amp; Fused Multi-Hardware Kernels", s_h1))
    story.append(Paragraph(
        "Layer-wise profiling across Qwen2.5-0.5B (24L), 7B (28L), and 72B (80L, 240 SwiGLU matrices [29568, 8192]) reveals superlinear outlier amplification "
        "[1, 2]: SwiGLU intermediate RMS ratios surge from 3,336x at 7.62B to <b>195,720.1x at 72.71B</b>. Meanwhile, StreamingLLM [3] suffers 0% recall "
        "once needles leave window <i>W</i>, additive Hebbian fast weights [4] ghost under state updates, and uncompressed O(<i>T</i>) KV caches stall "
        "quadratically at 32.9k tokens. MoS and SCache eliminate both bottlenecks without retraining:",
        s_body,
    ))
    story.append(Paragraph(
        "<b>MoS Static Decomposition:</b> &nbsp; <b>y</b>(<b>x</b>) = dequant<sub>Block64-INT4</sub>(<b>W</b><sub><i>q</i></sub>)<b>x</b> + "
        "&Sigma;<sub><i>k</i></sub> ReLU(cos(<b>x</b>, <b>z</b><sub><i>k</i></sub>)) <b>B</b><sub><i>k</i></sub>(<b>A</b><sub><i>k</i></sub><b>x</b>), "
        "&nbsp;&nbsp; where (<b>W</b> &minus; <b>W</b><sub>4</sub>) diag(<b>s</b>) &approx; <b>U</b><sub><i>r</i></sub><b>&Sigma;</b><sub><i>r</i></sub><b>V</b><sub><i>r</i></sub><sup>T</sup> &nbsp; (Eq. 1)",
        s_eq,
    ))
    story.append(Paragraph(
        "<b>SCache O(1) Delta Rule:</b> &nbsp; <b>W</b><sub>scar</sub><sup>(<i>t</i>)</sup> = &gamma;<b>W</b><sub>scar</sub><sup>(<i>t</i>&minus;1)</sup> + "
        "&eta; &middot; ReLU(<b>k</b><sub><i>u</i></sub><sup>T</sup><b>c</b><sub><i>m*</i></sub>) &middot; "
        "(<b>v</b><sub>ev</sub> &minus; <b>W</b><sub>scar</sub><sup>(<i>t</i>&minus;1)</sup><b>k</b><sub><i>u</i></sub>)<b>k</b><sub><i>u</i></sub><sup>T</sup> &nbsp; (Eq. 2)",
        s_eq,
    ))
    story.append(Paragraph(
        "<b>Fused Metal &amp; CUDA Kernels:</b> Dynamic per-row scaling <i>c<sub>m</sub></i> = max(||<b>x</b><sub><i>m</i></sub>||<sub>&infin;</sub>/512, 10<sup>&minus;4</sup>) "
        "prevents &gt;65,504 FP16 overflow under 195,720x spikes. On Apple Silicon, native Metal <code>_weight_int4pack_mm</code> achieves <b>cos = 1.000000</b> "
        "across 4 full 72B layers while accelerating compute by <b>50.38x (1,495.3 &rarr; 29.7 ms)</b>. On NVIDIA CUDA (<code>sm_75+</code> Turing T4 through Hopper), "
        "our fused Triton <code>W4A16+ASVD</code> kernels unpack canonical Block-64 <code>uint8</code> nibbles in registers (128-byte coalesced FP32 FMA GEMV for "
        "<i>M</i>&le;8 decode; split even/odd Tensor-Core <code>tl.dot</code> for <i>M</i>&gt;8 prefill) and fuse (<b>h</b><sub>scar</sub><b>B</b><sub>scar</sub><sup>T</sup>), "
        "<i>c<sub>m</sub></i> rescaling, and bias in a single global write with zero weight swizzling. In SCache, evicted tokens are scored synchronously across "
        "deep layers [<i>L</i>/2 : <i>L</i>) by Widrow-Hoff prediction surprise [5, 6] to retain top 32-token micro-pages alongside orthogonal fast-weight updates.",
        s_body,
    ))

    story.append(KeepTogether([
        Image(str(hero_png), width=7.32 * inch, height=1.88 * inch),
        Paragraph(
            "<b>Figure 1.</b> Multi-Scale Empirical Audit (0.5B &rarr; 3B &rarr; 7.62B &rarr; 72.71B). "
            "<b>(A)</b> On 32,868-token non-repeating Python docs (Qwen2.5-3B), SCache O(1) stays locked at 22.71 MiB (&Delta;=+0 B vs. 16.6k, 50.9x smaller) with "
            "100% Passkey recall and 5.84x higher speed (300.8 vs. 51.5 tok/s). "
            "<b>(B)</b> Under 195,720x 72.7B SwiGLU outliers, MoS cuts unseen KL by 3.08x–3.29x (100.0% Top-5). "
            "<b>(C)</b> Fused Apple Metal &amp; CUDA decode speed vs. AirLLM FP16 NVMe ceiling.",
            s_caption,
        ),
    ]))

    story.append(Paragraph("2. 16.6k &amp; 32.9k Passkey / Needle-in-a-Haystack &amp; 195,720x Outlier Audit", s_h1))
    t_niah = [
        ["Pretrained Model", "Context L", "Cache Engine", "Audited KV RAM (0 B Drift)", "Throughput (Wall)", "Passkey Recall", "vs. Normal KV O(T)"],
        ["Qwen2.5-3B-Instruct (36L)", "16,568 tok", "Normal KV O(T)", "582.75 MiB (linear)", "245.3 tok/s (67.5s)", "100% (5/5)", "1.0x RAM · 1.00x speed"],
        ["Qwen2.5-3B-Instruct (36L)", "16,568 tok", "StreamingLLM (W=256)", "9.00 MiB O(1)", "394.0 tok/s (42.1s)", "0% (Evicted)", "64.8x smaller · Amnesia"],
        ["Qwen2.5-3B-Instruct (36L)", "16,568 tok", "SCache O(1) (W=256, S=256)", "22.71 MiB (+0.00 B)", "299.0 tok/s (55.4s)", "100% (5/5)", "25.7x smaller · 1.22x faster"],
        ["Qwen2.5-3B-Instruct (36L)", "32,868 tok", "Normal KV O(T)", "1,155.80 MiB (1.13 GiB)", "51.5 tok/s (638.3s)", "100% (3/3)", "Quadratic O(T^2) stall"],
        ["Qwen2.5-3B-Instruct (36L)", "32,868 tok", "StreamingLLM (W=256)", "9.00 MiB O(1)", "391.4 tok/s (84.0s)", "0% (Evicted)", "128.4x smaller · Amnesia"],
        ["Qwen2.5-3B-Instruct (36L)", "32,868 tok", "SCache O(1) (W=256, S=256)", "22.71 MiB (+0 B vs 16.6k!)", "300.8 tok/s (109.3s)", "100% (3/3)", "50.9x smaller · 5.84x faster!"],
        ["Qwen2.5-0.5B-Instruct (24L)", "16,568 tok", "SCache O(1) (S=256 / S=128)", "6.86 / 5.32 MiB (+0 B)", "1,159.4 tok/s (14.3s)", "80% / 100% (5/5)", "28.3x–36.5x smaller · 1.1x faster"],
        ["Qwen2.5-0.5B-Instruct (24L)", "32,868 tok", "SCache O(1) (W=256, S=256)", "6.86 MiB (+0 B vs 16.6k!)", "1,176.9 tok/s (27.9s)", "100% (5/5)", "56.2x smaller · 1.60x faster!"],
    ]
    story.append(KeepTogether([
        styled_table(t_niah, [1.36 * inch, 0.68 * inch, 1.30 * inch, 1.24 * inch, 0.94 * inch, 0.74 * inch, 1.08 * inch], s_th, s_td, bold_rows=(3, 6, 8)),
        Paragraph(
            "<b>Table 1 (16.6k &amp; 32.9k Passkey / NIAH Audit on Non-Repeating Python Docs):</b> Zero fine-tuning, recursive (device, data_ptr) byte verification "
            "(0.00 B discrepancy across all runs). SCache O(1) eliminates both linear KV memory growth and O(T^2) attention latency collapse.",
            s_caption,
        ),
    ]))

    t_v1 = [
        ["Scale (Outliers)", "Unseen Domain", "Dense FP16 PPL", "MoS (r=48) PPL", "Hidden Cos", "Top-5 Hit", "KL Reduction vs. INT4"],
        ["7.62B (3,336x)", "Python / Physics", "1.648 / 6.253", "1.641 / 6.257", "0.991 / 0.982", "100.0% / 100.0%", "2.39x / 1.68x lower KL"],
        ["7.62B (3,336x)", "Portug. / JSON", "19.152 / 25.356", "21.717 / 25.352", "0.989 / 0.993", "100.0% / 100.0%", "2.27x / 3.33x lower KL"],
        ["72.71B (195,720x)", "Python (Dijkstra)", "1.705", "1.674 (Blk64: 1.850)", "0.967 / 0.981", "100.0% (99.3%)", "3.29x lower (0.0474 -> 0.0144)"],
        ["72.71B (195,720x)", "Quantum Physics", "6.187", "6.257 (Blk64: 6.786)", "0.977 / 0.981", "100.0% (98.9%)", "3.19x lower (0.1749 -> 0.0549)"],
        ["72.71B (195,720x)", "Tech Portuguese", "19.478", "18.446 (Blk64: 20.940)", "0.979 / 0.980", "100.0% (98.7%)", "3.16x lower (0.1431 -> 0.0453)"],
        ["72.71B (195,720x)", "Strict Nested JSON", "21.285", "19.058 (Blk64: 23.787)", "0.976 / 0.982", "100.0% (99.0%)", "3.08x lower (0.3717 -> 0.1206)"],
    ]
    story.append(KeepTogether([
        styled_table(t_v1, [1.06 * inch, 1.06 * inch, 0.90 * inch, 1.24 * inch, 0.80 * inch, 0.88 * inch, 1.40 * inch], s_th, s_td, bold_rows=(3, 4, 5, 6)),
        Paragraph(
            "<b>Table 2 (Vector 1 Across 7.62B &amp; 72.71B — Dominance over 195,720x SwiGLU Outliers):</b> Full 72.71B shrinks 135.43 &rarr; 38.12 GiB (-71.85%); "
            "8-specialist bank shrinks 866.3 &rarr; 31.4 GiB (-96.37% / 27.59x). Vector 0 80-layer KV stays locked at 64.73 MiB (-94.94% at 4k, 0.00 KB diff).",
            s_caption,
        ),
    ]))

    story.append(Paragraph("3. 72.71B Multi-Hardware Speed/Memory Audit (Apple Metal &amp; NVIDIA CUDA) &amp; 3D Pareto", s_h1))
    t_pareto = [
        ["Scale & Hardware Mode", "Non-Repeat PPL", "Peak Active GPU / MPS", "SCache / KV", "Prefill", "Decode", "Notes / Speedup vs. AirLLM FP16"],
        ["7.62B AirLLM FP16 SSD (1.57 GB/s)", "6.05", "<2 GiB (SSD-stalled)", "224 MiB O(T)", "—", "0.11 tok/s", "1.0x (9.1 s/tok, 14.2 GiB/tok)"],
        ["7.62B MoS + SCache (CUDA, W=256)", "6.67", "2.99 GiB MLP (3.72 GiB 8x)", "22.0 MiB O(1)", "60+ tok/s", "24.7 tok/s", "224.5x (62.7x vs INT4 offload)"],
        ["72.71B Dense FP16 + Full KV", "4.35", "135.4 GiB (866 GiB 8x)", "1,280 MiB O(T)", "Cluster", "Cluster", "Multi-A100 Only (10.7 GiB KV @ 32k)"],
        ["72.71B AirLLM FP16 (NVMe* / Cloud)", "4.35", "<2 GiB (NVMe-stalled)", "1,280 MiB O(T)", "—", "0.026 / 0.0006", "1.0x (38.7 s/tok 3.5 GiB/s NVMe)"],
        ["72.71B Apple M2 (16 GB) --streaming", "4.86", "1.92 GiB MPS (0.52 GiB RSS)", "128.5 MiB O(1)", "4.23 tok/s", "0.023 tok/s", "Full 72.7B on 16 GB laptop (0 OOM)"],
        ["72.71B Apple M2 / M2 Pro Unpaged*", "4.86", "1.90 GiB/grp (38.56 GiB eq.)", "128.5 MiB O(1)", "2.5–5.1 tok/s", "1.28 / 2.57*", "50.38x faster Metal int4pack (*bus)"],
        ["72.71B M4 Pro (48 GB) / M2 Max (64 GB+)", "4.86", "38.56 GiB Unified MPS", "128.5 MiB O(1)", "6.8–9.6 tok/s", "3.46–4.87 tok/s", "0 MB/tok SSD; saves >10.5 GiB @ 32k!"],
        ["72.71B NVIDIA CUDA (Triton W4A16, W=384)", "4.75", "38.12 GiB (31.4 GiB 8x)", "166.0 MiB O(1)", "18+ tok/s", "4.03–7.27+ tok/s", "155x–280x NVMe (6,717x Cloud)"],
    ]
    story.append(KeepTogether([
        styled_table(t_pareto, [1.62 * inch, 0.68 * inch, 1.24 * inch, 0.80 * inch, 0.68 * inch, 0.78 * inch, 1.54 * inch], s_th, s_td, bold_rows=(2, 7, 8)),
        Paragraph(
            "<b>Table 3 (Vector 7 &amp; Multi-Hardware 72.71B Speed/Memory Audit):</b> *M2 Pro maxes out at 32 GB RAM (&lt;38.12 GiB shards), yielding ~0.40–0.55 tok/s "
            "under hybrid NVMe paging vs. 2.57 tok/s pure 200 GB/s Metal bus ceiling; &ge;48 GB Macs hold all 80 layers + 32k SCache 100% resident in RAM.",
            s_caption,
        ),
    ]))

    t_v36 = [
        ["Adversarial Vector", "Conventional Baseline", "MoS / SCache O(1) Result", "Audited 0.5B -> 7.62B -> 72.71B Finding"],
        ["V2: Periodic Callback (W=64)", "72.7B Naive: 127.91; StreamLLM: 97.46", "72.7B SCache Tail PPL: 19.51 (-86.8% KV)", "5.00x lower tail PPL than StreamingLLM at 72.7B"],
        ["V3A: Contradictory Overwrite", "Hebbian: cos=0.467, Ghost +0.501", "Widrow-Hoff: cos=1.0000, Ghost -0.138", "Zero ghosting in constant 34.8 KB bank"],
        ["V3B: Passcode (445–1,745 tok)", "72B StreamLLM NLL: 3.241 / 2.961", "72B SCache O(1) NLL: 0.576 / 1.279", "14.38x (445t) to 5.38x (1,225t) prob gain at 72.7B"],
        ["V4: 500x DoS + Antipodal", "Linear subspace drift / corruption", "Target cos: 1.0000; Leak: 0.00000000", "Exact ReLU cone immunity to hostile noise"],
        ["V5: Full-Stack 72.7B QA/Code", "72.7B Dense+Full KV: 3/6, NLL 2.486", "72.7B MoS+SCache: 3/6, Ans NLL 1.858", "-72.7% MLP & -86.8% KV, lower answer NLL"],
        ["V6: Canonical 72.7B / 123B", "72.7B: 135.4 GiB; 123B: 246 GB", "72.7B: 20x1.79 GiB (1.92 GiB MPS)", "Bit-exact 4L vs 8L; 123B at 0.04 GiB post-group"],
    ]
    story.append(KeepTogether([
        styled_table(t_v36, [1.50 * inch, 1.78 * inch, 1.86 * inch, 2.20 * inch], s_th, s_td, bold_rows=(1, 3, 6)),
        Paragraph("<b>Table 4 (Vectors 2–6 — Colossal 72.71B Long-Horizon Recall, Anti-Ghosting, DoS Immunity, and Group Streaming).</b>", s_caption),
    ]))

    story.append(Paragraph("4. Boundary Conditions, Public 70B/72.71B Download, Practical Guide &amp; Reproducibility", s_h1))
    story.append(Paragraph(
        "(1) <b>Cross-Layer SwiGLU Permutation Sensitivity:</b> Sharing one dense base across <i>distinct</i> pretrained layers without permutation "
        "matching incurs distortion (KL &approx; 0.78); off-the-shelf models pair per-layer INT4 bases with ASVD scars (38.12 GiB at 72.71B), "
        "reserving single-base sharing for multi-tenant banks (-96.37% VRAM across 8 specialists at 72.71B). "
        "(2) <b>Associative Slot Budget:</b> Synchronized 32-token micro-page routing + Widrow-Hoff fast weights achieve 100% passkey recall at 32.9k "
        "tokens (<i>S</i><sub>slots</sub>=256); copying dozens of uncorrelated random hashes simultaneously scales with <i>S</i><sub>slots</sub>. "
        "(3) <b>Public 70B-Class Model Download &amp; Practical Guide (<code>mos-run --download</code>):</b> All 22 SHA-256 verified 20&times;4-layer "
        "<b>Qwen2.5-72B-Instruct MoS v2</b> shards (38.12 GiB, zero-login HTTP 206 Range at <code>github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos</code>) "
        "and the step-by-step <b>Practical Local Deployment Guide</b> (<code>docs/GUIA_PRATICO_MOS_SCACHE_72B.md</code>) [14], alongside fused CUDA "
        "Triton &amp; Apple Metal kernels, <b>44/44 pytest suite</b>, and raw JSON audits, are archived at <b>DOI: 10.5281/zenodo.24143839</b> [13].",
        s_body,
    ))

    story.append(Paragraph("References", s_h1))
    refs = [
        "[1] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 35, 30318–30332.",
        "[2] Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., et al. (2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression. MLSys 6.",
        "[3] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024.",
        "[4] Schmidhuber, J. (1992). Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation 4(1), 131–139.",
        "[5] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits. IRE WESCON Convention Record, Part 4, 96–104.",
        "[6] Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML, PMLR 139, 9355–9366.",
        "[7] Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 37.",
        "[8] Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., & Sun, G. (2023). ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs. arXiv:2312.05821.",
        "[9] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 36.",
        "[10] Li, Y., Yu, Y., Liang, C., He, P., Karampatziakis, N., Chen, W., & Zhao, T. (2024). LoftQ: LoRA-Fine-Tuning-Aware Quantization for LLMs. ICLR 2024.",
        "[11] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 36.",
        "[12] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., et al. (2017). Attention Is All You Need. NeurIPS 30.",
        "[13] Reis Filho, J. C. (2026). Mixture of Scars (MoS) and SCache O(1): Multi-Scale 0.5B–72.71B Replication Kit & 32.9k Passkey Telemetry (v1.3.0). Zenodo. DOI: 10.5281/zenodo.24143839.",
        "[14] Reis Filho, J. C. (2026). Qwen2.5-72B-Instruct MoS v2 (72.71B, 38.12 GiB Canonical INT4+ASVD Checkpoint) & Practical Local Deployment Guide (docs/GUIA_PRATICO_MOS_SCACHE_72B.md). URL: https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos.",
    ]
    for r in refs:
        story.append(Paragraph(escape(r), s_ref))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def sync_huggingface_space():
    hf_sc = REPO_ROOT / "huggingface_space" / "scar_cache"
    if hf_sc.exists():
        for src_py in sorted((REPO_ROOT / "scar_cache").glob("*.py")):
            shutil.copy2(src_py, hf_sc / src_py.name)
        print(f"[HF SPACE SYNC] Synced {len(list(hf_sc.glob('*.py')))} modules -> {hf_sc}")


def audit_latex_and_pack(zip_out: Path):
    tex = (LATEX_DIR / "main.tex").read_text(encoding="utf-8")
    bib = (LATEX_DIR / "references.bib").read_text(encoding="utf-8")
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
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean_tex):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        else:
            assert stack and stack[-1] == m.group(2), f"Mismatch {m.group(2)}"
            stack.pop()
    assert not stack, f"Unclosed envs: {stack}"

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
    cited = {k.strip() for m in re.finditer(r"\\cite\w*\{([^}]+)\}", clean_tex) for k in m.group(1).split(",")}
    assert not (cited - bib_keys), f"Missing cites: {cited - bib_keys}"

    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", clean_tex))
    assert not (refs - labels), f"Missing refs: {refs - labels}"

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(LATEX_DIR.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.suffix != ".png":
                zf.write(p, arcname=str(p.relative_to(LATEX_DIR)))
    print(f"[LATEX v1.3 AUDIT PASS] {len(cited)}/{len(bib_keys)} cites | {len(labels)} labels | {zip_out.name} ({zip_out.stat().st_size} B)")


def build_replication_zip_v1_3(zip_path: Path):
    files = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "docs" / "GUIA_PRATICO_MOS_SCACHE_72B.md",
        REPO_ROOT / "scripts" / "adversarial_mos_scache_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_7b_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_72b_suite.py",
        REPO_ROOT / "scripts" / "benchmark_16k_passkey_llm.py",
        REPO_ROOT / "scripts" / "benchmark_72b_apple_silicon.py",
        REPO_ROOT / "scripts" / "export_canonical_72b_mos.py",
        REPO_ROOT / "scripts" / "run_live_72b_apple_m2_proof.py",
        REPO_ROOT / "scripts" / "generate_fig1_multiscale_v1_3.py",
        REPO_ROOT / "benchmark_results" / "adversarial_mos_scache_report.json",
        REPO_ROOT / "results" / "passkey_16k_32k_llm_audit.json",
        REPO_ROOT / "results" / "scale_7b_audit" / "adversarial_scale_7b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "adversarial_scale_72b_report.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_live_72b_generation.json",
        REPO_ROOT / "results" / "scale_72b_audit" / "apple_m2_72b_speed_memory_audit.json",
    ]
    files.extend(sorted((REPO_ROOT / "scar_cache").glob("*.py")))
    files.extend(sorted((REPO_ROOT / "tests").glob("test_*.py")))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if fp.exists():
                zf.write(fp, arcname=str(fp.relative_to(REPO_ROOT)))


def main():
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "generate_fig1_multiscale_v1_3.py")])
    sync_huggingface_space()

    src_png = LATEX_DIR / "figures" / "fig1_mos_scache_audit.png"
    hero_png = OUT_DIR / "mos_scache_v1_3_hero.png"
    shutil.copy2(src_png, hero_png)

    pdf_v13 = OUT_DIR / "Mixture_of_Scars_and_SCache_Preprint_v1.3.pdf"
    rep_v13 = OUT_DIR / "MoS_SCache_Adversarial_Replication_v1.3.zip"
    tex_v13 = OUT_DIR / "MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip"

    build_pdf_v1_3(pdf_v13, hero_png)
    build_replication_zip_v1_3(rep_v13)
    audit_latex_and_pack(tex_v13)
    print(f"✓ Built {pdf_v13} ({pdf_v13.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Built {rep_v13} ({rep_v13.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
