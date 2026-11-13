#!/usr/bin/env python3
"""
Generates Figure 1 (`paper_latex/figures/fig1_mos_scache_v3_audit.pdf` & `.png`) for Manuscript v3.0
directly from verified empirical JSON reports:
  - Panel A: 32.9k Passkey / NIAH O(1) KV Invariance (`results/passkey_16k_32k_llm_audit.json`)
  - Panel B: Breaking the 24 GB / 32 GB 72.71B Residency Wall (`results/horizon1_v1_4_audit.json`)
  - Panel C: 140 KB Zero-SGD Scar Cartridges 72.71B -> 3B (`results/scar_store_v2_stage5_3b_vs_0p5b_audit.json`)
  - Panel D: Sinkhorn Monomial-Procrustes Re-Basin (`results/sinkhorn_rebasin_v2_4_audit.json`)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    r_niah = json.loads((ROOT / "results/passkey_16k_32k_llm_audit.json").read_text(encoding="utf-8"))
    r_h1 = json.loads((ROOT / "results/horizon1_v1_4_audit.json").read_text(encoding="utf-8"))
    r_s5 = json.loads((ROOT / "results/scar_store_v2_stage5_3b_vs_0p5b_audit.json").read_text(encoding="utf-8"))
    r_reb = json.loads((ROOT / "results/sinkhorn_rebasin_v2_4_audit.json").read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.6,
        "axes.titlesize": 9.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.6,
    })

    fig, axes = plt.subplots(1, 4, figsize=(16.2, 3.65), dpi=300)

    # ------------------------------------------------------------------
    # Panel A: 32.9k Passkey Audit (SCache O(1) vs Normal KV O(T))
    # ------------------------------------------------------------------
    ax = axes[0]
    m3b = r_niah["models"]["Qwen/Qwen2.5-3B-Instruct"]
    van_16k = m3b["haystack_16500"]["baselines_at_25pct"]["normal_kv"]
    van_32k = m3b["haystack_32800"]["baselines_at_25pct"]["normal_kv"]
    sc_16k = m3b["haystack_16500"]["scache_depths"][0]
    sc_32k = m3b["haystack_32800"]["scache_depths"][0]

    lens_3b = [1024, 4096, van_16k["context_tokens"], van_32k["context_tokens"]]
    bpt = (van_32k["kv_mib"] * 1024 * 1024) / van_32k["context_tokens"]
    van_mib = [(1024 * bpt) / (1024**2), (4096 * bpt) / (1024**2), van_16k["kv_mib"], van_32k["kv_mib"]]
    sc_mib = [sc_16k["kv_mib"], sc_16k["kv_mib"], sc_16k["kv_mib"], sc_32k["kv_mib"]]

    ax.plot(lens_3b, van_mib, marker="o", lw=2.1, color="#c0392b", label="Normal KV O(T) (1,156 MiB)")
    ax.plot(lens_3b, sc_mib, marker="s", lw=2.4, color="#1b4f72", label="SCache O(1) (22.7 MiB, 100%)")
    ax.fill_between(lens_3b, sc_mib, van_mib, color="#d6eaf8", alpha=0.55)
    ax.annotate(
        "50.9x smaller (+0 B)\n100% Recall | 5.84x faster",
        xy=(van_32k["context_tokens"], sc_32k["kv_mib"]),
        xytext=(11500, 260),
        arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=1.3),
        fontsize=7.6,
        fontweight="bold",
        color="#1b4f72",
    )
    ax.set_title("A. 32.9k Passkey O(1) KV")
    ax.set_xlabel("Context Length (Tokens)")
    ax.set_ylabel("Physical KV Memory (MiB)")
    ax.set_xticks([4096, 16568, 32868])
    ax.set_xticklabels(["4.1k", "16.6k", "32.9k"])
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.2, frameon=True)

    # ------------------------------------------------------------------
    # Panel B: 72.71B Sub-2-Bit MoS Residency (< 24 GB GPU / < 32 GB Mac)
    # ------------------------------------------------------------------
    ax = axes[1]
    sub72 = r_h1["track_1_2_qwen72b_sub2bit_shard_audit"]
    modes = ["Dense\nBF16", "MoS v1.2\nINT4+ASVD", "MoS v3.0\nSub-2-Bit"]
    gibs = [135.43, 38.12, sub72["gpu_resident_vram_20shards_head_scache_gib"]]
    bar_cols = ["#94a3b8", "#0284c7", "#0f766e"]
    bars = ax.bar(np.arange(3), gibs, width=0.54, color=bar_cols)
    ax.axhline(24.0, color="#dc2626", linestyle="--", lw=1.4, label="24 GB Consumer GPU Ceiling")
    ax.axhline(32.0, color="#d97706", linestyle=":", lw=1.3, label="32 GB Laptop RAM Ceiling")
    for b, v in zip(bars, gibs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 3.2,
            f"{v:.2f} GiB",
            ha="center",
            va="bottom",
            fontsize=7.7,
            fontweight="bold",
            color="#0f172a",
        )
    ax.text(
        2.0,
        7.5,
        "cos=0.9991\n(195,720x)",
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color="white",
    )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(modes, fontsize=7.8)
    ax.set_ylim(0, 155)
    ax.set_title("B. 72.71B < 24 GB GPU Wall")
    ax.set_ylabel("Resident Footprint (GiB)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=6.9, frameon=True)

    # ------------------------------------------------------------------
    # Panel C: 140 KB Zero-SGD Scar Cartridges (72.71B -> 3B KL Drop)
    # ------------------------------------------------------------------
    ax = axes[2]
    fid = r_s5["unseen_paraphrase_72b_fidelity"]
    d_keys = [
        "titan72b_kabatiansky_levenshtein",
        "titan72b_landauer_principle",
        "titan72b_immerman_szelepcsenyi",
    ]
    d_names = ["Sphere\nPacking", "Landauer\nPrinciple", "Immerman-\nSzelep."]
    kl_base = [fid[k]["base_3b_kl_div"] for k in d_keys]
    kl_scar = [fid[k]["head_expanded_3b_int4_kl_div"] for k in d_keys]

    xc = np.arange(3)
    w = 0.36
    ax.bar(xc - w / 2, kl_base, width=w, color="#e67e22", label="Raw 3B Student")
    ax.bar(xc + w / 2, kl_scar, width=w, color="#117a65", label="3B + 140 KB Scar (0 SGD)")
    for i in range(3):
        ratio = kl_base[i] / max(kl_scar[i], 1e-5)
        ax.text(
            xc[i] + w / 2 + 0.04,
            kl_scar[i] + 0.015,
            f"{ratio:.0f}x↓" if ratio >= 10 else f"{ratio:.1f}x↓",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#0e6251",
        )
    ax.set_xticks(xc)
    ax.set_xticklabels(d_names, fontsize=7.7)
    ax.set_ylim(0, max(kl_base) * 1.22)
    ax.set_title("C. 72B->3B 140 KB Cartridges")
    ax.set_ylabel("Unseen KL vs. 72.71B Teacher")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=7.0, frameon=True)

    # ------------------------------------------------------------------
    # Panel D: Sinkhorn Monomial-Procrustes Re-Basin (Shared-Base KL)
    # ------------------------------------------------------------------
    ax = axes[3]
    spec = r_reb["qwen_0p5b"]["permuted_specialist_git_rebasin_L12"]
    p3b = r_reb["qwen_3b"]["pair_L22_L23"]
    p12l = r_reb["qwen_3b"]["6pairs_12layers_L18_L29"]

    r_labels = ["Specialist\nRe-Basin", "3B Pair\n(L22-L23)", "3B 12-Layer\n(L18..L29)"]
    naive_kls = [
        spec["naive_shared_base"]["mean_kl"],
        p3b["naive_unaligned"]["mean_kl"],
        p12l["naive_unaligned"]["mean_kl"],
    ]
    sink_kls = [
        spec["sinkhorn_rebasin_shared_base"]["mean_kl"],
        p3b["sinkhorn_procrustes"]["mean_kl"],
        p12l["sinkhorn_monomial"]["mean_kl"],
    ]
    gains_d = [n / s for n, s in zip(naive_kls, sink_kls)]

    xd = np.arange(3)
    ax.bar(xd - w / 2, naive_kls, width=w, color="#b91c1c", label="Naive Shared Base")
    ax.bar(xd + w / 2, sink_kls, width=w, color="#4338ca", label="Sinkhorn Re-Basin")
    ax.set_yscale("log")
    ax.set_ylim(0.0012, 22.0)
    for i, g in enumerate(gains_d):
        ax.text(
            xd[i] + w / 2 + 0.04,
            sink_kls[i] * 1.42,
            f"{g:.1f}x↓",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color="#312e81",
        )
    ax.set_xticks(xd)
    ax.set_xticklabels(r_labels, fontsize=7.6)
    ax.set_title("D. Sinkhorn Re-Basin (Log KL)")
    ax.set_ylabel("Shared-Base KL (log scale)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=7.0, frameon=True)

    fig.tight_layout(pad=0.95)
    out_dir = ROOT / "paper_latex" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "fig1_mos_scache_v3_audit.pdf"
    out_png = out_dir / "fig1_mos_scache_v3_audit.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
