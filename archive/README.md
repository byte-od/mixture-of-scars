# Historical Archive (`archive/`)

This directory preserves earlier milestone artifacts, cloud runner bundles, and prior Zenodo deposit revisions cleanly separated from the canonical **`v3.0.0`** root layout:

- **`zenodo_deposits/`**: Historical Zenodo deposit kits (`zenodo_deposit_v1`, `v1_1`, `v1_2`, `v1_3`). For the current **`v3.0.0`** Preprint PDF, LaTeX/arXiv package, and Adversarial Replication bundle, see [`../zenodo_deposit_v3_0/`](../zenodo_deposit_v3_0/).
- **`paper_v1/`**: Superseded `v1.2`/`v1.3` figure and kit generator scripts. Current `v3.0` generators live in [`../scripts/generate_fig1_multiscale_v3_0.py`](../scripts/generate_fig1_multiscale_v3_0.py) and [`../scripts/build_zenodo_v3_0_kit.py`](../scripts/build_zenodo_v3_0_kit.py).
- **`legacy_deployments/`**: Early Hugging Face Space / Leaderboard prototypes (`hf_space`, `hf_leaderboard`, `spaces`). The canonical, actively synchronized Gradio Red-Team Laboratory lives in [`../huggingface_space/`](../huggingface_space/).
- **`kaggle_bundles/`**: Cloud Kaggle batch export and red-team job wrappers used to produce the public `Qwen2.5-72B-Instruct MoS v2` shards and multi-scale reports.
- **`legacy_docs/`**: Initial `v0.1` (`Nov 7, 2026`) design notes (`AGENT_CONTEXT.md`, `ROADMAP_KV_CACHE_REPLACEMENT.md`). Current architecture and audit guides live in [`../docs/`](../docs/).
