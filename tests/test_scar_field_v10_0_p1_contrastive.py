"""Forensic Unit Tests for `ScarField v10.0-P1` Direct Biorthogonal Contrastive Pull-Push Geometry."""

from __future__ import annotations

import torch

from scar_field.contrastive_biorth_p1 import (
    ContrastiveP1BatchMeta,
    compute_biorthogonal_contrastive_p1_loss,
    extract_bipartite_turn_poles,
)
from scar_field.e8_lattice import project_rows_to_unit_sphere


def test_p1_contrastive_pull_push_coalesces_paraphrases_and_repels_neighbors() -> None:
    """Proves pure `C^1` spherical-cap Pull-Push increases in-concept cosine & reduces neighbor cosine."""
    torch.manual_seed(1001)
    bsz, seq_len, d_w, d_f = 6, 16, 64, 32
    raw_w = torch.randn(bsz, seq_len, d_w, requires_grad=True)
    raw_f = torch.randn(bsz, seq_len, d_f, requires_grad=True)

    meta = ContrastiveP1BatchMeta(
        prompt_span_mask=torch.ones(bsz, seq_len),
        trigger_indices=torch.full((bsz,), seq_len - 1, dtype=torch.long),
        concept_ids=torch.tensor([0, 0, 1, 1, 2, -1], dtype=torch.long),
        lang_ids=torch.tensor([0, 1, 0, 1, 2, -1], dtype=torch.long),
    )

    opt = torch.optim.Adam([raw_w, raw_f], lr=0.15)
    initial_stats = None
    final_stats = None
    for step in range(25):
        opt.zero_grad()
        hw = project_rows_to_unit_sphere(raw_w)
        hf = project_rows_to_unit_sphere(raw_f)
        loss, stats = compute_biorthogonal_contrastive_p1_loss(hw, hf, meta)
        if step == 0:
            initial_stats = stats
        final_stats = stats
        loss.backward()
        opt.step()

    assert initial_stats is not None and final_stats is not None
    assert final_stats["world_pos_cos"] > initial_stats["world_pos_cos"] + 0.25
    assert final_stats["world_neg_max_cos"] <= 0.24
    u_w, u_f = extract_bipartite_turn_poles(
        project_rows_to_unit_sphere(raw_w.detach()),
        project_rows_to_unit_sphere(raw_f.detach()),
        meta,
    )
    assert torch.allclose(u_w.norm(dim=-1), torch.ones(bsz), atol=1e-5)
    assert torch.allclose(u_f.norm(dim=-1), torch.ones(bsz), atol=1e-5)
