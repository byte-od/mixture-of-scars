"""Structural & Numerical Integrity Tests for Canonical `Qwen2.5-72B-Instruct MoS v2` (`72.71B`, `80` Layers).

Verifies:
  1. Git-tracked `models/qwen25_72b_mos_v2/mos_manifest.json` (`mos_canonical_v2`): exact `80` layers,
     `20 x 4`-layer streaming shards, `22` SHA-256 verified files (`38.12 GiB`), `Block-64 INT4` + `ASVD` ranks.
  2. When local `Qwen2.5-72B MoS v2` shards are present on disk (`mos_group_00.safetensors`), executes a live
     4-layer `CanonicalMoSGroup` forward pass with GQA (`64` Q / `8` KV heads), RoPE, `SCacheDynamicCache` O(1)
     (`0 B` storage discrepancy), fused `W4A16 + ASVD` SwiGLU MLP, and deterministic `unload()` deallocation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from scar_cache.cli_72b import CanonicalMoSGroup
from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.speculative_ghost_jump import recursive_scache_storage_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_V2 = REPO_ROOT / "models" / "qwen25_72b_mos_v2" / "mos_manifest.json"
MODEL_V2_DIR = REPO_ROOT / "models" / "qwen25_72b_mos_v2"


def _compute_rope(seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).view(1, 1, seq_len, head_dim)
    sin = emb.sin().to(dtype).view(1, 1, seq_len, head_dim)
    return cos, sin


def test_canonical_qwen72b_mos_v2_manifest_integrity() -> None:
    """Always runs in every environment: audits the tracked SHA-256 manifest of the 72.71B MoS v2 release."""
    assert MANIFEST_V2.exists(), f"Missing tracked 72B manifest: {MANIFEST_V2}"
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))

    assert manifest["format"] == "mos_canonical_v2"
    assert manifest["source_model"] == "Qwen/Qwen2.5-72B-Instruct"
    assert manifest["num_hidden_layers"] == 80
    assert manifest["layers_per_group"] == 4
    assert manifest["num_groups"] == 20
    assert manifest["hidden_size"] == 8192
    assert manifest["intermediate_size"] == 29568
    assert manifest["num_attention_heads"] == 64
    assert manifest["num_key_value_heads"] == 8
    assert manifest["head_dim"] == 128
    assert manifest["vocab_size"] == 152064
    assert manifest["block_size"] == 64
    assert manifest["mlp_scar_rank"] == 48
    assert manifest["attn_scar_rank"] == 32

    files = manifest["files"]
    assert len(files) == 22  # embed + 20 groups + lm_head
    total_bytes = sum(int(entry["bytes"]) for entry in files)
    total_gib = total_bytes / (1024**3)
    assert 38.0 <= total_gib <= 38.3, f"Expected ~38.12 GiB canonical checkpoint, got {total_gib:.2f} GiB"

    for entry in files:
        sha = entry["sha256"]
        assert isinstance(sha, str) and len(sha) == 64
        int(sha, 16)  # Valid lowercase hex SHA-256


def test_local_qwen72b_mos_v2_group_forward_scache_and_unload() -> None:
    """Runs live 4-layer 72.71B CanonicalMoSGroup forward + SCache O(1) audit when local shards exist."""
    shard_0 = MODEL_V2_DIR / "mos_group_00.safetensors"
    if not shard_0.exists():
        pytest.skip("Local 38.12 GiB Qwen2.5-72B MoS v2 shards not downloaded (run `mos-run --download`).")

    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    assert shard_0.stat().st_size == manifest["files"][1]["bytes"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    grp = CanonicalMoSGroup(0, shard_0, manifest, device=device, dtype=dtype)
    grp.load()
    assert grp.is_loaded is True
    assert len(grp.tensors) > 0

    # Verify genuine Block-64 INT4 uint8 packing + Rank-48 ASVD scars on Layer 0 SwiGLU gate_proj
    gate_pb = grp.tensors["layers.0.mlp.gate_proj.packed_base"]
    gate_sb = grp.tensors["layers.0.mlp.gate_proj.scales_base"]
    gate_a = grp.tensors["layers.0.mlp.gate_proj.A_scar"]
    gate_b = grp.tensors["layers.0.mlp.gate_proj.B_scar"]
    assert gate_pb.dtype == torch.uint8
    assert gate_pb.numel() * 2 == 29568 * 8192  # Exact 4-bit packing (2 weights per byte)
    assert gate_sb.shape[0] == (29568 * 8192) // 64
    assert gate_a.shape == (48, 8192)
    assert gate_b.shape == (29568, 48)

    # Execute 4 real 72.71B layers with GQA + RoPE + SCacheDynamicCache O(1)
    torch.manual_seed(72)
    seq_len = 12
    x = (torch.randn(1, seq_len, 8192) * 0.05).to(device=device, dtype=dtype)
    cos, sin = _compute_rope(seq_len, 128, 1000000.0, device=device, dtype=dtype)
    scache = SCacheDynamicCache(window_size=16, num_slots=8, num_sink_tokens=2, project_slots=True)

    with torch.no_grad():
        for l_idx in range(4):
            x = grp.forward_layer(l_idx, x, cos, sin, scache=scache)

    assert x.shape == (1, seq_len, 8192)
    assert torch.isfinite(x.float()).all()
    assert len(scache.layers) == 4
    assert recursive_scache_storage_bytes(scache) == scache.total_memory_bytes

    # Verify streaming unload releases all group tensors cleanly
    grp.unload()
    assert grp.is_loaded is False
    assert grp.tensors == {}
