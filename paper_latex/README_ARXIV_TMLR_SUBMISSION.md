# LaTeX Manuscript Source (`v1.3` — arXiv / TMLR / Double-Blind Workshops)

## Quick Upload to Overleaf
1. Go to https://www.overleaf.com/project -> **New Project** -> **Upload Project**.
2. Drop `zenodo_deposit_v1_3/MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip`. It compiles immediately with `pdfLaTeX` + `BibTeX`.

## Switching Between arXiv Preprint and Double-Blind Peer Review
Open `main.tex` line 10:
- **arXiv / Preprint (Author, GitHub & Zenodo DOI shown):** `\anonymousfalse`
- **Double-Blind Peer Review (TMLR / NeurIPS / ICML / ICLR Workshop):** change line 10 to `\anonymoustrue`

## Submitting to arXiv (`cs.LG` primary, `cs.CL` secondary)
1. Upload `zenodo_deposit_v1_3/MoS_SCache_LaTeX_arXiv_TMLR_v1.3.zip` at https://arxiv.org/submit.
2. Set **Report-no / DOI** field to: `10.5281/zenodo.24143839`.
3. License: `CC BY 4.0`.

## Public `70B`-Class Model Download (`Qwen2.5-72B-Instruct MoS v2`) & Practical Guide Citation
Both `main.tex` and `references.bib` (`\citep{reisfilho2026qwen72b_mos_v2}`) explicitly cite:
- **Zero-Login `72.71B` Checkpoint Download (`38.12 GiB`, `22` SHA-256 `.safetensors` shards, `mos-run --download`):** https://github.com/byte-od/mixture-of-scars/releases/tag/v1.2.0-qwen72b-mos
- **Step-by-Step Practical Local Deployment Guide:** [`../docs/GUIA_PRATICO_MOS_SCACHE_72B.md`](../docs/GUIA_PRATICO_MOS_SCACHE_72B.md)
