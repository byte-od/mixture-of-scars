"""ScarField v10.0-P1 — Direct Biorthogonal Contrastive Pull-Push Geometry on `S_world x S_form`.

Implements Track `P1` from `SCARFIELD_V10_FRONTIER_RESEARCH_ROADMAP.md` (100% Pure Zero-Legacy,
Zero External Open-Weight Teachers, Zero `Softmax`/`exp`):
  1. **Concept-Coalescent Pull on `S^{D_w - 1}` (`L_world_pull`)**:
     Coalesces arbitrary surface paraphrases, telegraphic shorthands (`"dna 3d"`, `"nave 3d"`),
     and multilingual `(PT, EN, ES, FR)` phrasings of the same concept `c` onto a single tight
     spherical cap on `S_world` right at the prompt-to-generation boundary (`t_trig`).
  2. **Biorthogonal Spherical-Cap Repulsion on `S^{D_w - 1}` (`L_world_push`)**:
     Actively repels distinct concepts `(c_i != c_k)` whose prompt trajectories approach closer
     than `kappa_sep = 0.12` (`> 83.1 deg` geodesic separation), plus a hard-neighbor worst-case
     hinge eliminating adjacent-capsule collisions (e.g. `"dna 3d"` vs. `"espaçonave em 3D"`).
  3. **Bipartite Syntax/Language Separation on `S^{D_f - 1}` (`L_form_disentangle`)**:
     Clusters `h_form[t_trig]` by target language/mode (`PT, EN, ES, FR, Code`) while keeping
     `h_world[t_trig]` language-invariant, realizing true bipartite factorization `Form x World`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from scar_field.e8_lattice import project_rows_to_unit_sphere


LANG_TO_ID: Dict[str, int] = {
    "pt": 0,
    "en": 1,
    "es": 2,
    "fr": 3,
    "code": 4,
}


@dataclass
class ContrastiveP1BatchMeta:
    """Per-row semantic concept & language metadata aligned with `[B, T]` training batches."""

    prompt_span_mask: torch.Tensor   # [B, T] float32 (`1.0` over active turn's User prompt bytes)
    trigger_indices: torch.Tensor    # [B] int64 (index of last prompt prefix byte before Assistant token 0)
    concept_ids: torch.Tensor        # [B] int64 (`>= 0` for canonical concepts; `-1` for OOC / generic PIN)
    lang_ids: torch.Tensor           # [B] int64 (`0=pt, 1=en, 2=es, 3=fr, 4=code, -1=none`)


def extract_bipartite_turn_poles(
    hw_seq: torch.Tensor,
    hf_seq: torch.Tensor,
    meta: ContrastiveP1BatchMeta,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts `(u_world [B, D_w], u_form [B, D_f])` on unit spheres at the active turn boundary."""
    bsz, seq_len, d_w = hw_seq.shape
    d_f = hf_seq.shape[-1]
    dev = hw_seq.device

    pm = meta.prompt_span_mask.to(device=dev, dtype=hw_seq.dtype).unsqueeze(-1)  # [B, T, 1]
    denom = pm.sum(dim=1).clamp(min=1.0)                                         # [B, 1]
    hw_span_mean = project_rows_to_unit_sphere((hw_seq * pm).sum(dim=1) / denom)

    trig = meta.trigger_indices.to(device=dev, dtype=torch.long).clamp(0, seq_len - 1)
    hw_trig = hw_seq.gather(1, trig.view(bsz, 1, 1).expand(bsz, 1, d_w)).squeeze(1)
    hf_trig = hf_seq.gather(1, trig.view(bsz, 1, 1).expand(bsz, 1, d_f)).squeeze(1)

    u_world = project_rows_to_unit_sphere(0.5 * hw_span_mean + 0.5 * hw_trig)
    u_form = project_rows_to_unit_sphere(hf_trig)
    return u_world, u_form


def compute_biorthogonal_contrastive_p1_loss(
    hw_seq: torch.Tensor,
    hf_seq: torch.Tensor,
    meta: ContrastiveP1BatchMeta,
    kappa_world_sep: float = 0.12,
    kappa_hard_neg: float = 0.22,
    kappa_form_pull: float = 0.55,
    kappa_form_sep: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Computes pure `C^1` spherical-cap Pull-Push loss on `(S_world, S_form)` with zero `Softmax`."""
    u_world, u_form = extract_bipartite_turn_poles(hw_seq, hf_seq, meta)
    dev = hw_seq.device
    c_ids = meta.concept_ids.to(device=dev, dtype=torch.long)
    l_ids = meta.lang_ids.to(device=dev, dtype=torch.long)
    bsz = int(c_ids.shape[0])

    eye_mask = torch.eye(bsz, dtype=torch.bool, device=dev)
    valid_c = c_ids >= 0
    valid_pair_c = valid_c.unsqueeze(1) & valid_c.unsqueeze(0) & (~eye_mask)

    gram_w = u_world @ u_world.T  # [B, B] exact cosines on `S^{D_w - 1}`
    pos_c_mask = valid_pair_c & (c_ids.unsqueeze(1) == c_ids.unsqueeze(0))
    neg_c_mask = valid_pair_c & (c_ids.unsqueeze(1) != c_ids.unsqueeze(0))

    if bool(pos_c_mask.any().item()):
        pos_cos = gram_w[pos_c_mask]
        loss_w_pull = (1.0 - pos_cos).mean()
        mean_pos_cos = float(pos_cos.detach().mean().item())
    else:
        loss_w_pull = gram_w.new_zeros(())
        mean_pos_cos = 1.0

    if bool(neg_c_mask.any().item()):
        neg_cos = gram_w[neg_c_mask]
        exc_sep = (neg_cos - kappa_world_sep).clamp(min=0.0)
        exc_hard = (neg_cos.amax() - kappa_hard_neg).clamp(min=0.0)
        loss_w_push = (exc_sep * exc_sep).mean() + 0.50 * (exc_hard * exc_hard)
        max_neg_cos = float(neg_cos.detach().max().item())
        mean_neg_cos = float(neg_cos.detach().mean().item())
    else:
        loss_w_push = gram_w.new_zeros(())
        max_neg_cos = 0.0
        mean_neg_cos = 0.0

    # Bipartite `S_form` language/register separation (`PT, EN, ES, FR, Code`)
    valid_l = l_ids >= 0
    valid_pair_l = valid_l.unsqueeze(1) & valid_l.unsqueeze(0) & (~eye_mask)
    gram_f = u_form @ u_form.T
    same_l_mask = valid_pair_l & (l_ids.unsqueeze(1) == l_ids.unsqueeze(0))
    diff_l_mask = valid_pair_l & (l_ids.unsqueeze(1) != l_ids.unsqueeze(0))

    loss_f = gram_f.new_zeros(())
    if bool(same_l_mask.any().item()):
        def_f = (kappa_form_pull - gram_f[same_l_mask]).clamp(min=0.0)
        loss_f = loss_f + (def_f * def_f).mean()
    if bool(diff_l_mask.any().item()):
        exc_f = (gram_f[diff_l_mask] - kappa_form_sep).clamp(min=0.0)
        loss_f = loss_f + (exc_f * exc_f).mean()

    loss_p1_total = 0.65 * loss_w_pull + 1.15 * loss_w_push + 0.25 * loss_f
    metrics = {
        "p1_loss": round(float(loss_p1_total.detach().item()), 5),
        "world_pos_cos": round(mean_pos_cos, 4),
        "world_neg_mean_cos": round(mean_neg_cos, 4),
        "world_neg_max_cos": round(max_neg_cos, 4),
    }
    return loss_p1_total, metrics
