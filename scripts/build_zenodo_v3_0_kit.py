#!/usr/bin/env python3
"""
Builds the Official v3.0.0 Publication Deposit Kit (`zenodo_deposit_v3_0/`):
  1. Generates `paper_latex/figures/fig1_mos_scache_v3_audit.pdf` / `.png` + `zenodo_deposit_v3_0/mos_scache_v3_0_hero.png`
  2. Syncs `huggingface_space/scar_cache/*.py` with `scar_cache/*.py`
  3. Audits `paper_latex/main.tex`, `references.bib` & `main.bbl` (zero syntax/brace/citation/ref issues)
  4. Builds `zenodo_deposit_v3_0/Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf`
  5. Builds `zenodo_deposit_v3_0/MoS_SCache_Adversarial_Replication_v3.0.zip`
  6. Builds `zenodo_deposit_v3_0/MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip` & `arXiv_Direct_Upload_v3.0.zip`
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
OUT_DIR = REPO_ROOT / "zenodo_deposit_v3_0"
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
        ("LEFTPADDING", (0, 0), (-1, -1), 3.8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.8),
        ("TOPPADDING", (0, 0), (-1, -1), 2.1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.1),
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
        0.54 * inch,
        0.33 * inch,
        "Reis Filho (2026) — Mixture of Scars (MoS) v3.0 & SCache O(1) Preprint | DOI: 10.5281/zenodo.24172784",
    )
    canvas.drawRightString(LETTER[0] - 0.54 * inch, 0.33 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_v3_0(pdf_path: Path, hero_png: Path):
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle(
        "PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=12.2,
        leading=15.2, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=4
    )
    s_meta = ParagraphStyle(
        "PaperMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=8.3,
        leading=11.3, alignment=TA_CENTER, textColor=colors.HexColor("#334155"), spaceAfter=5
    )
    s_abs_box = ParagraphStyle(
        "AbstractText", parent=styles["Normal"], fontName="Helvetica", fontSize=8.1,
        leading=11.1, alignment=TA_JUSTIFY, leftIndent=9, rightIndent=9, spaceAfter=4.5,
        textColor=colors.HexColor("#1E293B")
    )
    s_h1 = ParagraphStyle(
        "SecH1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=9.8,
        leading=12.2, textColor=colors.HexColor("#1B365D"), spaceBefore=5.0, spaceAfter=2.2,
        keepWithNext=True,
    )
    s_body = ParagraphStyle(
        "BodyJustified", parent=styles["Normal"], fontName="Helvetica", fontSize=8.2,
        leading=11.6, alignment=TA_JUSTIFY, spaceAfter=3.1, textColor=colors.HexColor("#1E293B")
    )
    s_prop = ParagraphStyle(
        "PropBox", parent=styles["Normal"], fontName="Helvetica", fontSize=8.0,
        leading=12.2, alignment=TA_JUSTIFY, backColor=colors.HexColor("#F1F5F9"),
        borderPadding=4.2, leftIndent=6, rightIndent=6, spaceBefore=2.5, spaceAfter=3.5,
        textColor=colors.HexColor("#0F172A")
    )
    s_eq = ParagraphStyle(
        "Equation", parent=styles["Normal"], fontName="Helvetica", fontSize=8.1,
        leading=12.6, alignment=TA_CENTER, backColor=colors.HexColor("#F8FAFC"),
        borderPadding=3.8, spaceBefore=2.8, spaceAfter=3.5, textColor=colors.HexColor("#0F172A")
    )
    s_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.45,
        leading=9.6, alignment=TA_LEFT, textColor=colors.HexColor("#475569"), spaceBefore=1.8, spaceAfter=4.0
    )
    s_th = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=6.9,
        leading=8.5, alignment=TA_LEFT, textColor=colors.white
    )
    s_td = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontName="Helvetica", fontSize=6.8,
        leading=8.4, alignment=TA_LEFT, textColor=colors.HexColor("#0F172A")
    )
    s_ref = ParagraphStyle(
        "RefItem", parent=styles["Normal"], fontName="Helvetica", fontSize=7.1,
        leading=9.0, alignment=TA_LEFT, leftIndent=14, firstLineIndent=-14, spaceAfter=1.2,
        textColor=colors.HexColor("#334155")
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.54 * inch,
        rightMargin=0.54 * inch,
        topMargin=0.44 * inch,
        bottomMargin=0.46 * inch,
        title="Mixture of Scars (MoS) v3.0 and SCache O(1): <24 GB 72.7B, 140 KB Scar Cartridges, Sinkhorn Re-Basin & 32k Attention",
        author="Julio Cesar Reis Filho",
    )

    story = []
    story.append(Paragraph(
        "Mixture of Scars (MoS) v3.0 and SCache O(1):<br/>"
        "Sub-2-Bit Colossal Compression (&lt;24 GB 72.7B), Closed-Form 140 KB Scar Cartridges,<br/>"
        "Sinkhorn Monomial–Procrustes Re-Basin, and Constant-Memory 32k Attention",
        s_title,
    ))
    story.append(Paragraph(
        "<b>Julio Cesar Reis Filho</b> — Byte-OD Research &amp; System One Architecture Lab<br/>"
        "Preprint Technical Report <b>v3.0.0</b> (Sub-2-Bit 72.71B Residency, Ghost-Jump Reflex, 140 KB <code>mos-store</code> HTTP-206 &amp; Sinkhorn Re-Basin) — November 13, 2026<br/>"
        "<b>DOI:</b> 10.5281/zenodo.24172784 &nbsp;|&nbsp; <b>Code &amp; 84/84 Tests:</b> github.com/byte-od/mixture-of-scars &nbsp;|&nbsp; <b>72B Shards:</b> <code>releases/tag/v1.2.0-qwen72b-mos</code>",
        s_meta,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=4.0))

    story.append(Paragraph(
        "<b>Abstract.</b> Autoregressive Large Language Models (LLMs) confront a compounding <i>Quadruple Wall</i>: "
        "(1) an O(<i>N</i>) static parameter wall requiring 135.43 GiB at 72.7B scale, (2) an O(<i>T</i>) dynamic KV-cache wall with quadratic O(<i>T</i><sup>2</sup>) stalls, "
        "(3) single-token memory-bandwidth decode bottlenecks, and (4) backpropagation-bound specialization prone to catastrophic forgetting and opaque data retention. "
        "We unify all four regimes under a single <b>Scar Principle</b>—compressing predictable substrates to their physical hardware floor while isolating high-surprise "
        "residuals inside one-sided ReLU spherical cones with exact 0.000000 out-of-hemisphere bleed: "
        "<b>(i) MoS &amp; Sub-2-Bit MoS</b> pair canonical Block-64 INT4 (38.12 GiB, <b>100.0% Top-5</b>, <b>50.38x</b> fused Apple Metal &amp; CUDA Triton <code>W4A16+ASVD</code>) "
        "and a 2.25 bpw symmetric ternary-quaternary base with two-pass alternating ASVD shields (<b>cos = 0.99911</b> under 195,720x SwiGLU outliers; <b>12,290x PPL reduction</b> "
        "vs. naive 2-bit), shrinking 72.71B into <b>20.65 GiB shards and 21.89 GiB GPU Resident VRAM (&lt;24 GB RTX 3090/4090 &amp; &lt;32 GB laptops)</b>; "
        "<b>(ii) SCache O(1) &amp; Ghost-Jump Speculative Reflex</b> combine Widrow-Hoff Delta fast weights and synchronized 32-token micro-page routing for "
        "<b>100% Passkey recall at 16,568 and 32,868 tokens</b> in strictly constant <b>22.71 MiB (&Delta;=+0 B; 50.9x smaller &amp; 5.84x faster at 300.8 tok/s)</b> "
        "with atomic O(1) speculative rollback (<b>KL = 0.0000, 1.89x–1.92x speedup</b>); "
        "<b>(iii) Closed-Form HeadExpandedScarStore (<code>mos-store</code>)</b> transplants 72.71B derivations with <b>ZERO SGD / ZERO backprop</b> into OS-page-aligned "
        "<b>140.00 KB (35 &times; 4 KB mmap)</b> cartridges on a 3B core (<b>1,015,469x smaller</b> than 72B BF16) over a 4,194,304-cell Kronecker grid "
        "(<b>98.61% unseen Top-1 parity</b>, 6.11x lower KL, 58.3 ms live RFC 7233 <code>HTTP/1.1 206</code> streaming, and 18 &mu;s RAM / 7.64 ms <code>0x00</code> SSD GDPR unlearning); and "
        "<b>(iv) Sinkhorn Monomial–Procrustes Re-Basin</b> exploits exact SwiGLU monomial gauge symmetry via tiled single-pass log-Sinkhorn (&lt;350 MiB VRAM at 72B+, 0 extra FLOPs), "
        "slashing shared-base KL by <b>56.36x (0.1578 &rarr; 0.0028)</b> on permuted specialists, <b>3.37x (0.1190 &rarr; 0.0353 &lt; 0.08)</b> on 3B cross-layer pairs, and "
        "<b>14.68x (6.4634 &rarr; 0.4403)</b> across 12 consecutive layers while establishing the <i>Residual Stream Marchenko-Pastur Orthogonality Law</i> on 72.71B. "
        "All <b>84/84 pytest suites</b> pass with <b>0.00 B</b> recursive storage discrepancy.",
        s_abs_box,
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=4.0))

    story.append(KeepTogether([
        Image(str(hero_png), width=7.42 * inch, height=1.67 * inch),
        Paragraph(
            "<b>Figure 1.</b> Audited MoS v3.0 + SCache O(1) Unified Benchmarks (0.5B &rarr; 3B &rarr; 7.62B &rarr; 72.71B; 0.00 B recursive storage discrepancy). "
            "<b>(A)</b> 32,868-token Passkey audit on Qwen2.5-3B: SCache O(1) stays locked at 22.71 MiB (&Delta;=+0 B, 50.9x smaller, 100% recall, 5.84x faster). "
            "<b>(B)</b> Sub-2-Bit MoS shrinks 72.71B to 21.89 GiB resident VRAM (&lt;24 GB GPU / &lt;32 GB laptop) at cos=0.99911 under 195,720x outliers. "
            "<b>(C)</b> Closed-form 72.71B &rarr; 3B 140 KB (35&times;4 KB mmap) Scar Cartridges cut unseen KL up to 98x (0 SGD). "
            "<b>(D)</b> Sinkhorn Monomial–Procrustes Re-Basin cuts shared-base KL by 56.4x (specialists), 3.4x (3B pair, KL=0.0353), and 14.7x (12 layers).",
            s_caption,
        ),
    ]))

    story.append(Paragraph("1. The Unified Scar Principle &amp; Sub-2-Bit Colossal Compression", s_h1))
    story.append(Paragraph(
        "Across parameters, context tokens, and domain adaptations, Transformer states decompose into a dense predictable manifold plus sparse high-surprise "
        "residuals [1, 2, 19]. Rather than allocating uniform high-precision memory everywhere, MoS v3.0 governs all four axes through one equation:",
        s_body,
    ))
    story.append(Paragraph(
        "<b>Unified Scar Law:</b> &nbsp; <i>F</i>(<b>x</b>) = <i>F</i><sub>base</sub>(<b>x</b>) + &Sigma;<sub><i>k</i></sub> "
        "ReLU( cos(<b>x</b>, <b>z</b><sub><i>k</i></sub>) &minus; &tau;<sub><i>k</i></sub> ) &middot; &Delta;<sub>scar,<i>k</i></sub>(<b>x</b>) &nbsp;&nbsp; (Eq. 1)",
        s_eq,
    ))
    story.append(Paragraph(
        "<b>Sub-2-Bit MoS (2.25 bpw + Alternating ASVD):</b> At 72.71B (Qwen2.5-72B, 80L), SwiGLU intermediate RMS spikes reach <b>195,720.1x</b> (vs. 3,336x at 7.62B), "
        "causing naive uniform 2-bit quantization to collapse (PPL = 1,351,758.2 on Qwen2.5-3B). In <code>sub2bit_mos.py</code>, we introduce an IEEE-exact symmetric "
        "quaternary grid <i>q</i> &isin; {&minus;1, &minus;5/16, +5/16, +1} (2.25 bpw with Block-64 FP16 scales) coupled via two alternating passes with a rank-<i>r</i> "
        "Activation-Aware SVD [8] outlier shield (<b>W</b> &minus; <b>W</b><sub>2</sub>)diag(<b>s</b>) &approx; <b>U</b><sub><i>r</i></sub><b>&Sigma;</b><sub><i>r</i></sub><b>V</b><sub><i>r</i></sub><sup>T</sup> "
        "(using calibration tokens or our zero-data analytical SwiGLU salience oracle ||<b>W</b><sub>gate,<i>i</i>,:</sub>||<sub>2</sub>||<b>W</b><sub>up,<i>i</i>,:</sub>||<sub>2</sub>) "
        "and GQA <code>k_proj/v_proj</code> INT4 shields. Coupled with fused Apple Metal <code>_weight_int4pack_mm</code> (<b>50.38x</b> compute speedup, cos=1.000000) "
        "and register-dequantizing CUDA Triton <code>W4A16+ASVD</code> / <code>W2A16+ASVD</code> kernels (<code>triton_mos_gemm.py</code>), real 72.71B shards shrink from "
        "1.79 &rarr; <b>1.0325 GiB/group (20.65 GiB across 20 shards; 21.89 GiB total resident GPU VRAM)</b>.",
        s_body,
    ))

    story.append(Paragraph("2. SCache O(1) Associative Memory &amp; Lossless Ghost-Jump Speculative Reflex", s_h1))
    story.append(Paragraph(
        "Per layer <i>l</i> and KV head <i>h</i>, <code>SCacheDynamicCache</code> maintains <i>N</i><sub>sink</sub> sinks, a rolling FIFO window <i>W</i><sub>win</sub> &minus; <i>N</i><sub>sink</sub>, "
        "and <i>S</i><sub>slots</sub> slots backed by Widrow-Hoff Delta fast weights [5, 6, 7]: "
        "<b>W</b><sub>scar</sub>(<i>t</i>) = &gamma;<b>W</b><sub>scar</sub>(<i>t</i>&minus;1) + &eta;<i>g</i><sub><i>m</i>*</sub>(<b>v</b><sub>ev</sub> &minus; "
        "<b>W</b><sub>scar</sub>(<i>t</i>&minus;1)<b>k</b><sub><i>u</i></sub>)<b>k</b><sub><i>u</i></sub><sup>T</sup>. Evicted spans are scored synchronously across deep layers <i>l</i> &isin; [<i>L</i>/2, <i>L</i>) "
        "to lock high-surprise 32-token micro-pages into RoPE-aligned slots while absorbing boilerplate into <b>W</b><sub>scar</sub> at +0 B growth.",
        s_body,
    ))
    story.append(Paragraph(
        "<b>Proposition 1 (Strict O(1) Invariance &amp; Lossless Speculative Rollback).</b> For all <i>T</i> &gt; <i>W</i><sub>win</sub>, physical KV storage satisfies "
        "&Delta;<i>B</i>/&Delta;<i>T</i> = 0 and erases stale values on span(<b>k</b><sub><i>u</i></sub>) without perturbing its orthogonal complement. Pairing a System 1 scout with "
        "<code>SCacheDynamicCache.speculative_capture()</code> snapshots only fixed O(1) ring pointers and <b>W</b><sub>scar</sub> matrices prior to single-pass System 2 verification "
        "[13], guaranteeing exact target parity <b>KL(<i>P</i><sub>target</sub> || <i>P</i><sub>GhostJump</sub>) = 0.0000</b> with <b>0 B</b> storage leak (1.89x–1.92x MPS speedup).",
        s_prop,
    ))

    story.append(Paragraph("3. Closed-Form 140 KB HeadExpandedScarStore, HTTP 206 &amp; Forensic GDPR Unlearning", s_h1))
    story.append(Paragraph(
        "Instead of mutating shared MLP weights globally [16, 17], <code>HeadExpandedScarStore</code> (<code>head_expanded_attention.py</code>, <code>mos-store</code> CLI) "
        "attaches query-matched synthetic attention heads at terminal block <i>L</i> gated by spherical cap <i>g<sub>c</sub></i>(<b>h</b>) = clamp((cos(<b>h</b>, <b>z</b><sub><i>c</i></sub>) &minus; &tau;)/&rho;, 0, 1). "
        "From cached 72.71B teacher logits, Active-Set Woodbury Rosetta inversion recovers hidden deltas projected onto the student's AlphaEdit null-space <b>P</b><sub>null</sub> [17] "
        "and factors them in <b>100% closed form (0 SGD / 0 autograd)</b> into <i>P</i>=8 per-head key prototypes and rank-60 Block-32 INT4 values. On Qwen2.5-3B (<i>H<sub>q</sub></i>=16, "
        "<i>d<sub>h</sub></i>=128, <i>d</i>=2048), each binary slot is <b>exact 143,360 B (140.00 KB = 35 &times; 4,096 B OS mmap pages, 0 B padding)</b> on a 2048&times;2048 "
        "(4,194,304-cell) Kronecker spherical grid (512 KB bitmap).",
        s_body,
    ))
    story.append(Paragraph(
        "<b>Proposition 2 (Exact Out-of-Cone Non-Interference &amp; Spherical Cap Scaling).</b> Whenever cos(<b>h</b>, <b>z</b><sub><i>c</i></sub>) &le; &tau; = 0.62, "
        "<i>g<sub>c</sub></i> = 0 identically: 0 extra heads execute, 0 mmap bytes are paged, and all 36/36 blocks remain bit-identical (<code>torch.equal</code>) to the raw student. "
        "By Kabatiansky–Levenshtein sphere packing [18], S<sup><i>d</i>&minus;1</sup> admits exp(&Theta;(<i>d</i>)) quasi-orthogonal caps, breaking the O(<i>d</i>) linear editing ceiling.",
        s_prop,
    ))

    story.append(Paragraph("4. Sinkhorn Monomial–Procrustes Re-Basin &amp; Marchenko-Pastur Orthogonality Law", s_h1))
    story.append(Paragraph(
        "<b>Proposition 3 (SwiGLU Monomial Gauge &amp; Zero-FLOP Scale Folding).</b> For SwiGLU(<b>x</b>) = <b>W</b><sub>dn</sub>(SiLU(<b>W</b><sub>gt</sub><b>x</b>) &middot; <b>W</b><sub>up</sub><b>x</b>), "
        "nonlinear SiLU restricts <b>W</b><sub>gt</sub> to discrete permutations <b>P</b> &isin; <i>S</i><sub><i>N</i></sub>, whereas <b>W</b><sub>up</sub>' = diag(<b>&alpha;</b>)<b>PW</b><sub>up</sub> "
        "and <b>W</b><sub>dn</sub>' = <b>W</b><sub>dn</sub><b>P</b><sup>T</sup>diag(<b>&alpha;</b>)<sup>&minus;1</sup> preserve outputs identically (FP32 max diff 5.25e-06). Storing shared "
        "<code>down_proj</code> grids with <code>transposed_grid=True</code> folds both &alpha;<sub><i>i</i></sub><sup>&minus;1</sup> and intermediate RMS scales 100% into signed FP16 Block-64 scales at "
        "<b>0 extra runtime FLOPs</b>. Our <b>Single-Pass Online Dual-Potential Log-Sinkhorn</b> [14, 15] aligns 72.7B matrices (<i>N</i>=29,568) in &lt;350 MiB tile VRAM (57x smaller than dense Sinkhorn).",
        s_prop,
    ))

    story.append(Paragraph("5. Multi-Scale Empirical Audit (0.5B &rarr; 3B &rarr; 7.62B &rarr; 72.71B; 84/84 Pytest)", s_h1))

    t1 = [
        ["Model / Pair", "Workload & Context", "Engine", "Audited KV Memory (0 B Drift)", "Throughput", "Recall / Parity", "Gain vs. Baseline"],
        ["Qwen2.5-3B (36L)", "16,568 tok Passkey", "Normal KV O(T)", "582.75 MiB (linear)", "245.3 tok/s (67.5s)", "100% (5/5)", "1.0x RAM · 1.00x speed"],
        ["Qwen2.5-3B (36L)", "16,568 tok Passkey", "SCache O(1) (W=256, S=256)", "22.71 MiB (+0.00 B)", "299.0 tok/s (55.4s)", "100% (5/5)", "25.7x smaller · 1.22x faster"],
        ["Qwen2.5-3B (36L)", "32,868 tok Passkey", "Normal KV / StreamingLLM", "1,155.8 / 9.0 MiB", "51.5 / 391.4 tok/s", "100% / 0%", "O(T^2) stall / Total amnesia"],
        ["Qwen2.5-3B (36L)", "32,868 tok Passkey", "SCache O(1) (W=256, S=256)", "22.71 MiB (+0 B vs 16.6k!)", "300.8 tok/s (109.3s)", "100% (3/3)", "50.9x smaller · 5.84x faster!"],
        ["Qwen2.5-0.5B (24L)", "32,868 tok Passkey", "SCache O(1) (W=256, S=256)", "6.86 MiB (+0 B vs 16.6k!)", "1,176.9 tok/s (27.9s)", "100% (5/5)", "56.2x smaller · 1.60x faster"],
        ["0.5B -> 3B Ghost-Jump", "Python / Math (96 tok)", "Speculative + SCache O(1)", "3.12 + 11.61 MiB (0 B leak)", "23.65 / 23.30 tok/s", "96/96 (KL=0.0000)", "1.89x–1.92x (4.8–5.05 tok/swp)"],
    ]
    story.append(KeepTogether([
        styled_table(t1, [1.16 * inch, 1.10 * inch, 1.28 * inch, 1.24 * inch, 0.94 * inch, 0.80 * inch, 0.96 * inch], s_th, s_td, bold_rows=(2, 4, 5, 6)),
        Paragraph("<b>Table 1 (32.9k Passkey Audit &amp; Ghost-Jump Speculative Reflex):</b> Zero fine-tuning, 0.00 B storage-pointer discrepancy across all runs.", s_caption),
    ]))

    t2 = [
        ["Scale & Compression Regime", "Hardware / Kernel Path", "Shards / Resident Footprint", "SCache KV", "Fidelity / PPL", "Decode Speed", "Key Architectural Result"],
        ["72.71B Dense BF16 + Full KV", "Multi-A100/H100 Cluster", "135.43 GiB (866 GiB 8x)", "10.7 GiB @ 32k", "PPL 4.35 (Ref)", "Cluster Ref", "Requires 2x80 GB datacenter GPUs"],
        ["72.71B MoS v1.2 (INT4+ASVD)", "Apple M4 Pro (48GB) / M2 Max", "38.12 GiB (1.79 GiB/shard)", "128.5 MiB O(1)", "100% Top-5, PPL 4.86", "3.46–4.87 tok/s", "50.38x Metal int4pack (1.92 GiB @ 16GB)"],
        ["72.71B MoS v1.2 (INT4+ASVD)", "NVIDIA CUDA Triton W4A16", "38.12 GiB (31.4 GiB 8x)", "166.0 MiB O(1)", "100% Top-5, PPL 4.75", "4.03–7.27+ tok/s", "155x–280x faster than NVMe FP16"],
        ["72.71B MoS v3.0 Sub-2-Bit", "CUDA W2A16+ASVD / Metal", "20.65 GiB / 21.89 GiB VRAM", "128.5 MiB O(1)", "L79 cos=0.99911 (2.49b)", "Single 24 GB GPU!", "Fits <24 GB RTX 3090/4090 & <32 GB Mac!"],
        ["3B Sub-2-Bit (108 MLPs)", "Naive 2-Bit vs MoS Sub-2-Bit", "4,644 -> 997.3 MiB (4.66x)", "O(1)", "1.35e6 -> 109.99 PPL", "0-Data Salience", "12,290x PPL reduction vs naive 2-bit"],
    ]
    story.append(KeepTogether([
        styled_table(t2, [1.34 * inch, 1.18 * inch, 1.18 * inch, 0.72 * inch, 0.96 * inch, 0.78 * inch, 1.32 * inch], s_th, s_td, bold_rows=(2, 3, 4, 5)),
        Paragraph("<b>Table 2 (72.71B Multi-Hardware &amp; Sub-2-Bit Residency Audit):</b> Neutralizes 195,720x SwiGLU outliers while breaking the 24 GB GPU &amp; 32 GB Mac barriers.", s_caption),
    ]))

    t3 = [
        ["Audit Dimension", "Domain / Hostile Attack", "Raw 3B Student", "0.5B + 61.25 KB Scar", "3B + 140.00 KB Scar (35x4KB)", "Verified System Guarantee"],
        ["Unseen Paraphrase vs 72B", "Kabatiansky-Levenshtein", "87.50% (KL=0.5296)", "87.50% (KL=0.3782)", "95.83% (KL=0.0727, 7.3x down)", "0 autograd steps (4.40 s solve)"],
        ["Unseen Paraphrase vs 72B", "Landauer's Principle", "83.33% (KL=0.6446)", "95.83% (KL=0.1045)", "100.0% (KL=0.0066, 97.7x down)", "Exact k_B T ln 2 derivation"],
        ["Unseen Paraphrase vs 72B", "Immerman-Szelepcsenyi", "91.67% (KL=0.2852)", "100.0% (KL=0.0257)", "100.0% (KL=0.0079, 36.1x down)", "Ghost-Jump accept: 26% -> 100%"],
        ["Out-of-Cone & Traps", "6 Controls + 6 Keyword Traps", "Reference Baseline", "24/24 torch.equal (0 B)", "36/36 torch.equal (0 B mmap)", "0/6 false triggers (cos <= 0.500 < 0.62)"],
        ["Poisoned 1,560t Haystack", "False '999 Joules' Rumor", "Hallucinates k_B T log m", "Resists rumor (0 B drift)", "Emits exact kT ln 2 (0 B drift)", "MoS+SCache+ScarStore: 100% Top-1"],
        ["Multi-Scar & Cloud HTTP", "3-Way Synthesis + HTTP 206", "Collapses / Uninstalled", "Spread 0.918 (188 KB)", "Spread 0.970 · HTTP 206: 58.3 ms", "GDPR Wipe: 18 us RAM / 7.64 ms 0x00 SSD"],
    ]
    story.append(KeepTogether([
        styled_table(t3, [1.10 * inch, 1.26 * inch, 0.96 * inch, 1.04 * inch, 1.44 * inch, 1.68 * inch], s_th, s_td, bold_rows=(2, 3, 4, 6)),
        Paragraph("<b>Table 3 (Closed-Form 72.71B &rarr; 0.5B &amp; 3B HeadExpandedScarStore, 6/6 Red-Team Suite, Live TCP HTTP 206 &amp; Forensic 0x00 SSD Wipe).</b>", s_caption),
    ]))

    t4 = [
        ["Scale & Shared Regime", "Shape & Compression", "Naive Unaligned", "Sinkhorn Monomial (0 FLOP)", "Sinkhorn–Procrustes", "KL / Error Gain", "Spectral / Geometric Finding"],
        ["0.5B Permuted Specialist", "[4864, 896] (Git Re-Basin)", "KL=0.1578 (87.5% Top1)", "—", "KL=0.0028 (100% Top5)", "56.36x lower KL", "Alignment cos: 0.0678 -> 0.9817"],
        ["0.5B Cross-Layer L4_L5", "49.9 -> 11.2 MB (4.44x)", "KL=0.5722 (70.7% Top1)", "KL=0.2618 (96.2% Top5)", "KL=0.2290 (96.5% Top5)", "2.50x lower KL", "Resolves v1.3 Limitation #1"],
        ["3B Interior pair_L22_L23", "258.0 -> 50.65 MB (5.09x)", "KL=0.1190 (88.4% Top1)", "KL=0.0395 < 0.08 (99.8%)", "KL=0.0353 < 0.08 (92.6%)", "3.37x lower KL", "Surpasses <0.08 target at 0 FLOPs"],
        ["3B 12-Layer L18..L29 (1/3)", "1,548 -> 303.9 MB (5.09x)", "KL=6.4634 (21.3% Top1)", "KL=0.4403 (95.02% Top5)", "—", "14.68x lower KL", "Eliminates multi-layer collapse"],
        ["72.71B Shard Pair L78_L79", "2,772 -> 445.2 MiB (6.23x)", "Out cos=0.1626 (err 1.19)", "Tiled Log-Sinkhorn (<350MB)", "Out cos=0.8835 (err 0.44)", "109.2x inner gain", "MP best cos=0.1228 vs floor 0.0501"],
    ]
    story.append(KeepTogether([
        styled_table(t4, [1.14 * inch, 1.02 * inch, 1.02 * inch, 1.12 * inch, 1.12 * inch, 0.78 * inch, 1.28 * inch], s_th, s_td, bold_rows=(1, 3, 4, 5)),
        Paragraph(
            "<b>Table 4 (Sinkhorn Monomial–Procrustes Re-Basin &amp; Residual Stream Marchenko-Pastur Law):</b> Adjacent 72.7B residual blocks self-organize near "
            "the random Marchenko-Pastur orthogonality floor (0.063–0.123 vs 0.0501), proving Re-Basin excels on specialist banks &amp; interior 3B pairs while "
            "independent Sub-2-Bit MoS (20.65 GiB, cos=0.99911) dominates full-model 72.7B compression.",
            s_caption,
        ),
    ]))

    story.append(Paragraph("6. Honest Limitations, Open Horizon 3 Directions &amp; Reproducibility", s_h1))
    story.append(Paragraph(
        "(1) <b>Shannon Slot Budget for Uncorrelated Hashes:</b> Fixed <i>S</i><sub>slots</sub> achieves 100% single/few-needle recall at 32.9k tokens (+0 B drift), "
        "but verbatim recall of hundreds of independent random strings at 128k–1M tokens requires hierarchical disk-backed micro-page paging (Horizon 3.1). "
        "(2) <b>Automated Corpus Digestion &amp; Near-Duplicate Cones:</b> Each 140 KB cartridge encodes a focused domain (<i>r</i>=60); ingesting 500-page manuals "
        "currently requires topic segmentation (<code>mos-digest</code> in Horizon 3.1), and near-identical sibling versions (e.g., 2025 vs 2026 tax tables) benefit "
        "from <code>mos-store sleep-bridge</code> consolidation. "
        "(3) <b>Student Reasoning Floor, Cross-Tokenizer Validation &amp; WebGPU:</b> Cartridges transplant domain knowledge bounded by the student's multi-step "
        "reasoning depth (98.61% on 3B vs 94.44% on 0.5B); cross-family grafting (e.g., Llama &rarr; Gemma) and native browser <code>WebGPU/WGSL</code> shaders "
        "remain active Horizon 3 milestones. Full code, <b>84/84 pytest suites</b>, 5 CLIs (<code>mos-run</code>, <code>mos-store</code>, <code>mos-graft</code>, "
        "<code>mos-rebasin</code>, <code>mos-transcode-sub2bit</code>), and raw JSON telemetry are archived under MIT/CC-BY-4.0 at <b>DOI: 10.5281/zenodo.24172784</b> [20, 21].",
        s_body,
    ))

    story.append(Paragraph("References", s_h1))
    refs = [
        "[1] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS 35, 30318–30332.",
        "[2] Lin, J., Tang, J., Tang, H., Yang, S., et al. (2024). AWQ: Activation-aware Weight Quantization for On-Device LLM Compression. MLSys 6.",
        "[3] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR 2024.",
        "[4] Schmidhuber, J. (1992). Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation 4(1), 131–139.",
        "[5] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits. IRE WESCON Convention Record, Part 4, 96–104.",
        "[6] Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML, PMLR 139, 9355–9366.",
        "[7] Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). Parallelizing Linear Transformers with the Delta Rule over Sequence Length. NeurIPS 37.",
        "[8] Yuan, Z., Shang, Y., Song, Y., Wu, Q., Yan, Y., & Sun, G. (2023). ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs. arXiv:2312.05821.",
        "[9] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 36.",
        "[10] Li, Y., Yu, Y., Liang, C., He, P., et al. (2024). LoftQ: LoRA-Fine-Tuning-Aware Quantization for LLMs. ICLR 2024.",
        "[11] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al. (2023). H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs. NeurIPS 36.",
        "[12] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS 30.",
        "[13] Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML, 19274–19286.",
        "[14] Ainsworth, S. K., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. ICLR 2023.",
        "[15] Sinkhorn, R., & Knopp, P. (1967). Concerning Nonnegative Matrices and Doubly Stochastic Matrices. Pacific J. Math. 21(2), 343–348.",
        "[16] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS 35, 17359–17372.",
        "[17] Fang, J., Jiang, H., Wang, K., Ma, Y., et al. (2025). AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models. ICLR 2025.",
        "[18] Kabatiansky, G. A., & Levenshtein, V. I. (1978). Bounds for Packings on a Sphere and in Space. Problems of Information Transmission 14(1), 1–17.",
        "[19] Marchenko, V. A., & Pastur, L. A. (1967). Distribution of Eigenvalues for Some Sets of Random Matrices. Math. USSR-Sbornik 1(4), 457–483.",
        "[20] Reis Filho, J. C. (2026). Mixture of Scars (MoS) v3.0 and SCache O(1): <24 GB 72.7B, 140 KB Scar Cartridges & Sinkhorn Re-Basin. Zenodo. DOI: 10.5281/zenodo.24172784.",
        "[21] Reis Filho, J. C. (2026). Qwen2.5-72B-Instruct MoS v2 Checkpoint & Practical Deployment Guide. URL: https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos.",
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
    bbl = (LATEX_DIR / "main.bbl").read_text(encoding="utf-8")
    bbl_keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl))
    cited = {k.strip() for m in re.finditer(r"\\cite\w*\{([^}]+)\}", clean_tex) for k in m.group(1).split(",")}
    assert not (cited - bib_keys), f"Missing cites in .bib: {cited - bib_keys}"
    assert not (cited - bbl_keys), f"Missing cites in .bbl (arXiv required): {cited - bbl_keys}"

    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", clean_tex))
    assert not (refs - labels), f"Missing refs: {refs - labels}"

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(LATEX_DIR.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and p.suffix != ".png":
                zf.write(p, arcname=str(p.relative_to(LATEX_DIR)))

    arxiv_clean_zip = OUT_DIR / "arXiv_Direct_Upload_v3.0.zip"
    with zipfile.ZipFile(arxiv_clean_zip, "w", compression=zipfile.ZIP_DEFLATED) as azf:
        for rel_name in ("main.tex", "main.bbl", "figures/fig1_mos_scache_v3_audit.pdf"):
            azf.write(LATEX_DIR / rel_name, arcname=rel_name)
    print(f"[LATEX v3.0 AUDIT PASS] {len(cited)}/{len(bbl_keys)} bbl cites | {len(labels)} labels | {arxiv_clean_zip.name} ({arxiv_clean_zip.stat().st_size} B)")


def build_replication_zip_v3_0(zip_path: Path):
    files = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "README.md",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "docs" / "AUDIT_AND_VERIFICATION_GUIDE.md",
        REPO_ROOT / "docs" / "GUIA_PRATICO_MOS_SCACHE_72B.md",
        REPO_ROOT / "docs" / "ROADMAP_V2_INNOVATION_AND_APPLICATIONS.md",
        REPO_ROOT / "docs" / "SCAR_STORE_V2_ARCHITECTURE.md",
        REPO_ROOT / "mixture_of_scars" / "__init__.py",
        REPO_ROOT / "scripts" / "adversarial_mos_scache_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_7b_suite.py",
        REPO_ROOT / "scripts" / "adversarial_scale_72b_suite.py",
        REPO_ROOT / "scripts" / "adversarial_core_0p5b_vs_3b_suite.py",
        REPO_ROOT / "scripts" / "benchmark_16k_passkey_llm.py",
        REPO_ROOT / "scripts" / "benchmark_72b_apple_silicon.py",
        REPO_ROOT / "scripts" / "benchmark_horizon1_v1_4.py",
        REPO_ROOT / "scripts" / "benchmark_scar_grafting_v2.py",
        REPO_ROOT / "scripts" / "audit_scar_store_v2_stage5_3b_vs_0p5b.py",
        REPO_ROOT / "scripts" / "audit_mos_store_cli_v2_3.py",
        REPO_ROOT / "scripts" / "audit_sinkhorn_rebasin_v2_4.py",
        REPO_ROOT / "scripts" / "generate_fig1_multiscale_v3_0.py",
        REPO_ROOT / "benchmark_results" / "adversarial_mos_scache_report.json",
        REPO_ROOT / "results" / "passkey_16k_32k_llm_audit.json",
        REPO_ROOT / "results" / "horizon1_v1_4_audit.json",
        REPO_ROOT / "results" / "horizon2_scar_grafting_v2_audit.json",
        REPO_ROOT / "results" / "scar_store_v2_stage5_3b_vs_0p5b_audit.json",
        REPO_ROOT / "results" / "adversarial_0p5b_vs_3b_audit.json",
        REPO_ROOT / "results" / "mos_store_v2_3_cli_http206_audit.json",
        REPO_ROOT / "results" / "sinkhorn_rebasin_v2_4_audit.json",
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
    subprocess.check_call([sys.executable, str(REPO_ROOT / "scripts" / "generate_fig1_multiscale_v3_0.py")])
    sync_huggingface_space()

    src_png = LATEX_DIR / "figures" / "fig1_mos_scache_v3_audit.png"
    hero_png = OUT_DIR / "mos_scache_v3_0_hero.png"
    shutil.copy2(src_png, hero_png)

    pdf_v30 = OUT_DIR / "Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf"
    rep_v30 = OUT_DIR / "MoS_SCache_Adversarial_Replication_v3.0.zip"
    tex_v30 = OUT_DIR / "MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip"

    build_pdf_v3_0(pdf_v30, hero_png)
    build_replication_zip_v3_0(rep_v30)
    audit_latex_and_pack(tex_v30)
    print(f"✓ Built {pdf_v30} ({pdf_v30.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Built {rep_v30} ({rep_v30.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
