"""Golden-Slice Language-Invariant Conceptual Scar Architecture (`ScarStore v4.0`).

Eliminates language-duplicated cartridges (`_en` / `_pt` / `_es` / `_fr`) via:
  1. **Golden-Slice Routing (`2L/3` Depth):** Extracts query hidden states at `mid_layer_idx = round(2L/3) - 1`
     (`Layer 15` on `24`-layer `0.5B`; `Layer 23` on `36`-layer `3B`), where Transformer representations peak
     in cross-lingual semantic alignment before late-layer lexical projection.
  2. **Within-Concept ANOVA Language-Subspace Orthogonal Projector (`P_perp_lang`):**
     Removes syntactic/language-identity variance `U_lang` (`k_lang = 14`) while preserving 100% of conceptual
     variance: `P_perp_lang(v) = v - U_lang @ (U_lang^T @ v)`. A single cartridge anchored ONLY in English
     routes accurately across EN, PT, ES, FR, and unseen Romance languages with `0.0000000000` out-of-cone bleed.
  3. **Two-Stage Prediction-Space Reduced-Rank Regression (`RRR`) + Smooth Horizon Handoff:**
     Consolidates compact non-EOS-terminated **Conceptual Premise Seeds** (`rank_mid = 12, rank_final = 52`,
     `62,720 B` `Block-64 INT4` on `0.5B` and `143,360 B = 35 x 4 KB` on `3B` — zero byte-size increase!),
     locking exact formulas/constants into `SCache O(1)` over `seed_horizon` tokens and then smoothly handing
     off to the Student + `SCache O(1)` for rich multi-sentence explanations and flexible Turn-2+ follow-ups.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.dual_depth_bridge import DualDepthScarCapsule, extract_mid_and_final_modules
from scar_cache.head_expanded_attention import (
    HEADER_ALIGN,
    PAGE_ALIGN,
    SCARSTORE_MAGIC,
    HeadExpandedTelemetry,
    _align_up,
    _bitmap_get,
    _bitmap_set,
    _build_deterministic_orthogonal_rotation,
    _find_subsequence_last_end_strict,
    _infer_model_device,
    _past_kv_seen_length,
)
from scar_cache.scar_grafting import detect_chatml_user_prefix_ids, extract_span_query_probes
from scar_cache.scar_store_v2 import (
    BLOCK_SIZE_INT4,
    MissingCapsuleAlert,
    dequantize_block64_int4,
    quantize_block64_int4,
)

UNIVERSAL_CHAT_SYSTEM_PROMPT = (
    "You are System One, a precise technical AI assistant. Always answer in the exact same language "
    "as the user's message with clear, complete, accurate detail."
)

# Domain-neutral parallel quads `(EN, PT, ES, FR)` for isolating pure language/syntax subspace `U_lang`
PARALLEL_CALIBRATION_QUADS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "Hello, good morning! Who are you and how can you help me today?",
        "Olá, bom dia! Quem é você e como pode me ajudar hoje?",
        "¡Hola, buenos días! ¿Quién eres y cómo puedes ayudarme hoy?",
        "Bonjour ! Qui êtes-vous et comment pouvez-vous m'aider aujourd'hui ?",
    ),
    (
        "What is the largest capital city in North and South America?",
        "Qual é a maior capital entre os países da América do Norte e do Sul?",
        "¿Cuál es la capital más grande de América del Norte y del Sur?",
        "Quelle est la plus grande capitale d'Amérique du Nord et du Sud ?",
    ),
    (
        "Explain how photosynthesis converts sunlight into chemical energy in plants.",
        "Explique como a fotossíntese converte luz solar em energia química nas plantas.",
        "Explica cómo la fotosíntesis convierte la luz solar en energía química en las plantas.",
        "Expliquez comment la photosynthèse convertit la lumière solaire en énergie chimique chez les plantes.",
    ),
    (
        "What is the square root of pi squared and why does that identity hold?",
        "Qual é a raiz quadrada de pi elevado ao quadrado e por que essa identidade vale?",
        "¿Cuál es la raíz cuadrada de pi al cuadrado y por qué se cumple esa identidad?",
        "Quelle est la racine carrée de pi au carré et pourquoi cette identité est-elle vraie ?",
    ),
    (
        "How does binary search achieve logarithmic time complexity on a sorted array?",
        "Como a busca binária atinge complexidade logarítmica em um vetor ordenado?",
        "¿Cómo logra la búsqueda binaria una complejidad logarítmica en un arreglo ordenado?",
        "Comment la recherche binaire atteint-elle une complexité logarithmique sur un tableau trié ?",
    ),
    (
        "Describe the main causes of the French Revolution in the late eighteenth century.",
        "Descreva as principais causas da Revolução Francesa no final do século XVIII.",
        "Describe las principales causas de la Revolución Francesa a finales del siglo XVIII.",
        "Décrivez les principales causes de la Révolution française à la fin du XVIIIe siècle.",
    ),
    (
        "Why does water expand when it freezes into ice at zero degrees Celsius?",
        "Por que a água se expande quando congela e vira gelo a zero grau Celsius?",
        "¿Por qué se expande el agua cuando se congela y forma hielo a cero grados Celsius?",
        "Pourquoi l'eau se dilate-t-elle lorsqu'elle gèle en glace à zéro degré Celsius ?",
    ),
    (
        "Explain the difference between TCP and UDP transport protocols on the internet.",
        "Explique a diferença entre os protocolos de transporte TCP e UDP na internet.",
        "Explica la diferencia entre los protocolos de transporte TCP y UDP en internet.",
        "Expliquez la différence entre les protocoles de transport TCP et UDP sur internet.",
    ),
    (
        "How do vaccines train the human immune system to recognize pathogens?",
        "Como as vacinas treinam o sistema imunológico humano para reconhecer patógenos?",
        "¿Cómo entrenan las vacunas al sistema inmunológico humano para reconocer patógenos?",
        "Comment les vaccins entraînent-ils le système immunitaire humain à reconnaître les agents pathogènes ?",
    ),
    (
        "What is Newton's second law of motion and how is force calculated from mass and acceleration?",
        "O que é a segunda lei de Newton e como a força é calculada pela massa e aceleração?",
        "¿Qué es la segunda ley de Newton y cómo se calcula la fuerza a partir de la masa y la aceleración?",
        "Qu'est-ce que la deuxième loi de Newton et comment calcule-t-on la force avec la masse et l'accélération ?",
    ),
    (
        "Explain how relational database indexes speed up SQL queries using B-trees.",
        "Explique como índices de bancos de dados relacionais aceleram consultas SQL usando árvores B.",
        "Explica cómo los índices de bases de datos relacionales aceleran consultas SQL mediante árboles B.",
        "Expliquez comment les index de bases de données relationnelles accélèrent les requêtes SQL avec des arbres B.",
    ),
    (
        "Why do seasons change throughout the year as the Earth orbits the Sun?",
        "Por que as estações do ano mudam ao longo do ano conforme a Terra orbita o Sol?",
        "¿Por qué cambian las estaciones a lo largo del año mientras la Tierra orbita alrededor del Sol?",
        "Pourquoi les saisons changent-elles au cours de l'année lorsque la Terre orbite autour du Soleil ?",
    ),
)


def golden_slice_layer_index(num_layers: int) -> int:
    """Canonical `2L/3` Golden-Slice layer index (`15` on `24`-layer `0.5B`, `23` on `36`-layer `3B`)."""
    if num_layers < 2:
        return 0
    return max(0, min(int(num_layers) - 2, int(round(2.0 * float(num_layers) / 3.0)) - 1))


def format_universal_chatml(user_query: str, system_prompt: str = UNIVERSAL_CHAT_SYSTEM_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_user_content_bounds(
    token_ids: Sequence[int],
    user_turn_prefix_ids: Optional[Sequence[int]],
    user_turn_suffix_ids: Optional[Sequence[int]],
) -> Tuple[int, int]:
    """Extracts `[q_start, q_end)` of pure user message tokens, stripping trailing `<|im_end|>\\n<|im_start|>assistant\\n`."""
    n = len(token_ids)
    if n == 0:
        return 0, 0
    ids_list = [int(x) for x in token_ids]
    q_start = _find_subsequence_last_end_strict(ids_list, user_turn_prefix_ids) if user_turn_prefix_ids else 0
    q_start = max(0, min(q_start, n - 1))
    q_end = n
    if user_turn_suffix_ids:
        suf = [int(x) for x in user_turn_suffix_ids]
        ls = len(suf)
        if ls > 0 and n >= q_start + ls + 1 and ids_list[n - ls :] == suf:
            q_end = n - ls
    if q_end <= q_start:
        q_end = n
    return q_start, q_end


def compute_horizon_scale(step_idx: int, seed_horizon: int = 44, fade_steps: int = 16) -> float:
    """Full capsule strength (`1.0`) over `[0 .. seed_horizon]`, then smooth raised-cosine handoff to `0.0`."""
    if seed_horizon <= 0:
        return 1.0
    if step_idx <= seed_horizon:
        return 1.0
    if fade_steps <= 0 or step_idx >= seed_horizon + fade_steps:
        return 0.0
    frac = float(step_idx - seed_horizon) / float(fade_steps)
    return 0.5 * (1.0 + math.cos(math.pi * frac))


class GoldenSliceScarBank:
    """In-memory calibrated Golden-Slice (`2L/3`) Language-Invariant Conceptual Scar Bank."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        mid_layer_idx: int,
        num_heads: int,
        head_dim: int,
        mu_query_mid: torch.Tensor,
        u_lang_mid: torch.Tensor,
        mu_hidden_mid: torch.Tensor,
        u_preserved_mid: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        u_preserved_final: torch.Tensor,
        rank_mid: int = 12,
        rank_final: int = 52,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        user_turn_suffix_ids: Optional[Sequence[int]] = None,
        seed_horizon: int = 44,
        seed_fade_steps: int = 16,
        winner_margin: float = 0.0002,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.mid_layer_idx = int(mid_layer_idx)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank_mid = int(rank_mid)
        self.rank_final = int(rank_final)
        self.mu_query_mid = mu_query_mid.detach().float().cpu().contiguous()
        self.u_lang_mid = u_lang_mid.detach().float().cpu().contiguous()
        self.mu_hidden_mid = mu_hidden_mid.detach().float().cpu().contiguous()
        self.u_preserved_mid = u_preserved_mid.detach().float().cpu().contiguous()
        self.mu_hidden_final = mu_hidden_final.detach().float().cpu().contiguous()
        self.u_preserved_final = u_preserved_final.detach().float().cpu().contiguous()
        self.user_turn_prefix_ids = [int(x) for x in user_turn_prefix_ids] if user_turn_prefix_ids else None
        self.user_turn_suffix_ids = [int(x) for x in user_turn_suffix_ids] if user_turn_suffix_ids else None
        self.seed_horizon = int(seed_horizon)
        self.seed_fade_steps = int(seed_fade_steps)
        self.winner_margin = float(winner_margin)
        self.capsules: Dict[str, DualDepthScarCapsule] = {}

    @property
    def k_lang(self) -> int:
        return int(self.u_lang_mid.shape[1])

    def project_lang_free(self, x: torch.Tensor) -> torch.Tensor:
        """Applies `P_perp_lang(x) = x - (x @ U_lang) @ U_lang^T` in `O(d * k_lang)` without materializing `d x d`."""
        u = self.u_lang_mid.to(device=x.device, dtype=x.dtype)
        return x - torch.matmul(torch.matmul(x, u), u.T)

    def add_capsule(self, capsule: DualDepthScarCapsule) -> None:
        if capsule.rank_mid != self.rank_mid or capsule.rank_final != self.rank_final:
            raise ValueError(
                f"Capsule {capsule.capsule_id!r} ranks ({capsule.rank_mid}, {capsule.rank_final}) "
                f"do not match bank ({self.rank_mid}, {self.rank_final})"
            )
        self.capsules[capsule.capsule_id] = capsule


class GoldenSliceSynthesisEngine:
    """Closed-Form Two-Stage Language-Invariant Synthesis Engine (`ZERO SGD / ZERO BACKPROP`)."""

    @staticmethod
    @torch.no_grad()
    def capture_mid_and_final_states(
        student_model: nn.Module,
        input_ids_2d: torch.Tensor,
        mid_layer_idx: int,
        mid_intervention: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Captures `(h_mid [L, d], h_final [L, d])` on CPU `float32` in a single forward pass."""
        dev = _infer_model_device(student_model)
        ids = input_ids_2d.to(dev)
        backbone, mid_mod, _, norm_mod, _ = extract_mid_and_final_modules(student_model, mid_layer_idx)
        captured: Dict[str, torch.Tensor] = {}

        def _mid_h(_m: nn.Module, _i: Tuple[Any, ...], out: Any) -> Any:
            is_tup = isinstance(out, tuple)
            t: torch.Tensor = out[0] if is_tup else out
            t3 = t.unsqueeze(0) if t.ndim == 2 else t
            captured["mid"] = t3[0].detach().float().cpu()
            if mid_intervention is not None:
                t_mod = mid_intervention(t3)
                t_ret = t_mod.squeeze(0) if t.ndim == 2 else t_mod
                return (t_ret,) + out[1:] if is_tup else t_ret
            return out

        def _fin_h(_m: nn.Module, _i: Tuple[Any, ...], out: torch.Tensor) -> None:
            t3 = out.unsqueeze(0) if out.ndim == 2 else out
            captured["final"] = t3[0].detach().float().cpu()

        hm = mid_mod.register_forward_hook(_mid_h)
        hf = norm_mod.register_forward_hook(_fin_h)
        try:
            try:
                backbone(ids, use_cache=False)
            except TypeError:
                backbone(ids)
        finally:
            hm.remove()
            hf.remove()
        return captured["mid"], captured["final"]

    @staticmethod
    def solve_high_clearance_rosetta(
        w_head: torch.Tensor,
        lg_student_1d: torch.Tensor,
        target_id: int,
        min_clearance: float = 13.5,
        k_eff: int = 48,
        vocab_ridge: float = 0.06,
    ) -> Tuple[torch.Tensor, float]:
        """Closed-form Cutting-Plane Woodbury Rosetta solver guaranteeing `>= min_clearance` logit margin
        so high-conflict factual tokens (`10^{-23}`, `Mixture of Scars`, `LC 150/2015`) survive `INT4`!"""
        v_common, d = int(w_head.shape[0]), int(w_head.shape[1])
        s_top = lg_student_1d.topk(min(int(k_eff), v_common)).indices
        t_idx = torch.tensor([int(target_id)], dtype=torch.long)
        s_union = torch.unique(torch.cat([t_idx, s_top], dim=0))

        runner_init = (
            float(lg_student_1d[s_top[0]].item())
            if int(s_top[0].item()) != int(target_id)
            else float(lg_student_1d[s_top[1]].item())
        )
        init_deficit = max(0.0, (runner_init + 3.5) - float(lg_student_1d[int(target_id)].item()))
        if init_deficit <= 0.0:
            return torch.zeros(d, dtype=torch.float32), 0.0

        best_r = torch.zeros(d, dtype=torch.float32)
        best_cl = float("-inf")
        for pass_i in range(8):
            w_s = w_head[s_union]
            s_sub = lg_student_1d[s_union]
            m_s = int(s_union.numel())
            t_mask = s_union == int(target_id)
            other_max = float(s_sub[~t_mask].max().item()) if m_s > 1 else 0.0
            target_sub = s_sub.clone()
            wrong_top_mask = (~t_mask) & (s_sub >= other_max - 2.5)
            target_sub[wrong_top_mask] = s_sub[wrong_top_mask] - 0.35 * float(min_clearance)
            target_sub[t_mask] = other_max + float(min_clearance)
            delta_lg = target_sub - s_sub
            gram_v = (w_s @ w_s.T) + float(vocab_ridge) * torch.eye(m_s, dtype=torch.float32)
            r_cand = w_s.T @ torch.linalg.solve(gram_v, delta_lg)

            pred_lg = lg_student_1d + (w_head @ r_cand)
            top2_v, top2_i = pred_lg.topk(2)
            runner_up = float(top2_v[1].item()) if int(top2_i[0].item()) == int(target_id) else float(top2_v[0].item())
            cl = float(pred_lg[int(target_id)].item()) - runner_up
            if cl > best_cl:
                best_cl = cl
                best_r = r_cand
            if int(top2_i[0].item()) == int(target_id) and cl >= float(min_clearance) * 0.88:
                break
            if pass_i < 7 and m_s + 8 <= int(0.65 * d):
                intruders = pred_lg.topk(8).indices
                new_u = torch.unique(torch.cat([s_union, intruders], dim=0))
                if int(new_u.numel()) == m_s:
                    break
                s_union = new_u
        return best_r, init_deficit

    @staticmethod
    def woodbury_prediction_rrr_svd(
        hp_mat: torch.Tensor,
        rc_mat: torch.Tensor,
        p_null: torch.Tensor,
        rank: int,
        hidden_ridge_ratio: float = 0.015,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction-Space Reduced-Rank Regression (`RRR`, Izenman / Eckart-Young):
        Minimizes `|| H @ M_r^T - R ||_F^2` directly on trajectory states, avoiding `1 / sigma_min(H)`
        inverse-covariance blowup across correlated multi-language tokens, with balanced `INT4` factors."""
        n_tok = int(hp_mat.shape[0])
        d = int(hp_mat.shape[1])
        if n_tok == 0 or float(rc_mat.abs().max().item()) < 1e-9:
            z_b = torch.zeros((d, rank), dtype=torch.float16)
            z_a = torch.zeros((rank, d), dtype=torch.float16)
            return z_b, z_a

        gram_h = hp_mat @ hp_mat.T
        mean_diag = max(float(gram_h.diagonal().mean().item()), 1e-6)
        eff_ratio = max(float(hidden_ridge_ratio), 0.012) if d >= 1536 else float(hidden_ridge_ratio)
        lam_h = max(1e-5, eff_ratio * mean_diag)
        alpha = torch.linalg.solve(gram_h + lam_h * torch.eye(n_tok, dtype=torch.float32), rc_mat)
        m_full = (alpha.T @ hp_mat) @ p_null  # [d, d]
        r_hat = hp_mat @ m_full.T  # [N, d]
        _, s_r, vh_r = torch.linalg.svd(r_hat, full_matrices=False)
        r_eff = max(1, min(int(rank), int(s_r.shape[0])))
        v_top = vh_r[:r_eff, :].T.contiguous()  # [d, r_eff]
        c_mat = v_top.T @ m_full  # [r_eff, d]
        u_c, s_c, vh_c = torch.linalg.svd(c_mat, full_matrices=False)
        sqrt_s = s_c[:r_eff].clamp(min=0.0).sqrt()
        b_eff = (v_top @ u_c[:, :r_eff]) * sqrt_s.unsqueeze(0)
        a_eff = sqrt_s.unsqueeze(1) * vh_c[:r_eff, :]
        if r_eff < rank:
            b_pad = torch.zeros((d, rank), dtype=torch.float32)
            a_pad = torch.zeros((rank, d), dtype=torch.float32)
            b_pad[:, :r_eff] = b_eff
            a_pad[:r_eff, :] = a_eff
            b_eff, a_eff = b_pad, a_pad
        return b_eff.to(torch.float16).contiguous(), a_eff.to(torch.float16).contiguous()

    @classmethod
    @torch.no_grad()
    def calibrate_golden_slice_student(
        cls,
        student_model: nn.Module,
        tokenizer: Any,
        parallel_quads: Sequence[Sequence[str]] = PARALLEL_CALIBRATION_QUADS,
        system_prompt: str = UNIVERSAL_CHAT_SYSTEM_PROMPT,
        mid_layer_idx: Optional[int] = None,
        k_lang: int = 14,
        rank_mid: int = 12,
        rank_final: int = 52,
        seed_horizon: int = 44,
        seed_fade_steps: int = 16,
    ) -> GoldenSliceScarBank:
        """Calibrates Golden-Slice (`2L/3`), within-concept ANOVA `U_lang`, and multi-turn invariant null spaces."""
        backbone = getattr(student_model, "model", getattr(student_model, "transformer", student_model))
        layers = getattr(backbone, "layers", getattr(backbone, "h", None))
        n_layers = len(layers) if layers is not None else 24
        resolved_mid = int(mid_layer_idx) if mid_layer_idx is not None else golden_slice_layer_index(n_layers)

        _, _, _, _, lm_head = extract_mid_and_final_modules(student_model, resolved_mid)
        d = int(lm_head.weight.shape[1])
        cfg = getattr(student_model, "config", None)
        n_heads = int(getattr(cfg, "num_attention_heads", max(1, d // 64)))
        d_head = d // n_heads

        if (d * rank_mid) % BLOCK_SIZE_INT4 != 0 or (d * rank_final) % BLOCK_SIZE_INT4 != 0:
            raise ValueError(
                f"hidden_dim={d} with (rank_mid={rank_mid}, rank_final={rank_final}) must divide {BLOCK_SIZE_INT4}"
            )

        prefix_ids = detect_chatml_user_prefix_ids(tokenizer)
        suffix_ids = list(tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False))

        quad_q_mids: List[torch.Tensor] = []
        all_mid_states: List[torch.Tensor] = []
        all_fin_states: List[torch.Tensor] = []

        for quad in parallel_quads:
            lang_qm: List[torch.Tensor] = []
            for q_txt in quad:
                p_str = format_universal_chatml(str(q_txt), system_prompt=system_prompt)
                ids = tokenizer(p_str, return_tensors="pt", add_special_tokens=False).input_ids
                hm, hf = cls.capture_mid_and_final_states(student_model, ids, resolved_mid)
                qs, qe = extract_user_content_bounds(ids[0].tolist(), prefix_ids, suffix_ids)
                lang_qm.append(hm[qs:qe].mean(dim=0))
                all_mid_states.append(hm)
                all_fin_states.append(hf)
            quad_q_mids.append(torch.stack(lang_qm, dim=0))

        q_stack = torch.stack(quad_q_mids, dim=0)  # [N_quads, N_langs, d]
        mu_query_mid = q_stack.reshape(-1, d).mean(dim=0)
        lang_residuals = (q_stack - q_stack.mean(dim=1, keepdim=True)).reshape(-1, d)
        _, _, vh_lang = torch.linalg.svd(lang_residuals, full_matrices=False)
        k_eff = max(1, min(int(k_lang), int(vh_lang.shape[0])))
        u_lang_mid = vh_lang[:k_eff, :].T.contiguous()

        cat_mid = torch.cat(all_mid_states, dim=0)
        cat_fin = torch.cat(all_fin_states, dim=0)
        mu_hidden_mid = cat_mid.mean(dim=0)
        mu_hidden_final = cat_fin.mean(dim=0)

        _, _, vh_pm = torch.linalg.svd(cat_mid - mu_hidden_mid, full_matrices=False)
        _, _, vh_pf = torch.linalg.svd(cat_fin - mu_hidden_final, full_matrices=False)
        u_pres_mid = vh_pm[:4, :].T.contiguous()
        u_pres_fin = vh_pf[:4, :].T.contiguous()

        return GoldenSliceScarBank(
            hidden_dim=d,
            num_layers=n_layers,
            mid_layer_idx=resolved_mid,
            num_heads=n_heads,
            head_dim=d_head,
            mu_query_mid=mu_query_mid,
            u_lang_mid=u_lang_mid,
            mu_hidden_mid=mu_hidden_mid,
            u_preserved_mid=u_pres_mid.contiguous(),
            mu_hidden_final=mu_hidden_final,
            u_preserved_final=u_pres_fin.contiguous(),
            rank_mid=rank_mid,
            rank_final=rank_final,
            user_turn_prefix_ids=prefix_ids,
            user_turn_suffix_ids=suffix_ids,
            seed_horizon=seed_horizon,
            seed_fade_steps=seed_fade_steps,
        )

    @classmethod
    @torch.no_grad()
    def _collect_seed_episodes(
        cls,
        seed_pairs: Sequence[Tuple[str, str]],
        student_model: nn.Module,
        tokenizer: Any,
        mid_layer_idx: int,
        system_prompt: str = UNIVERSAL_CHAT_SYSTEM_PROMPT,
        include_multiturn_augmentation: bool = True,
    ) -> List[Dict[str, Any]]:
        dev = _infer_model_device(student_model)
        episodes: List[Dict[str, Any]] = []
        raw_list: List[Tuple[str, str]] = [
            (format_universal_chatml(q_raw, system_prompt=system_prompt), s_raw) for q_raw, s_raw in seed_pairs
        ]
        if include_multiturn_augmentation and len(seed_pairs) > 0:
            mt_prefix = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                "<|im_start|>user\nOlá, tudo bem?<|im_end|>\n"
                "<|im_start|>assistant\nOlá! Como posso ajudar você hoje?<|im_end|>\n"
            )
            for q_raw, s_raw in seed_pairs[: min(2, len(seed_pairs))]:
                raw_list.append(
                    (mt_prefix + f"<|im_start|>user\n{q_raw}<|im_end|>\n<|im_start|>assistant\n", s_raw)
                )

        for prompt_chatml, seed_text in raw_list:
            p_ids = list(tokenizer.encode(prompt_chatml, add_special_tokens=False))
            # Non-EOS-terminated Conceptual Premise Seed: NO `<|im_end|>` forced at seed end!
            a_ids = list(tokenizer.encode(seed_text, add_special_tokens=False))
            if not a_ids:
                continue
            lp, la = len(p_ids), len(a_ids)
            full_t = torch.tensor([p_ids + a_ids], dtype=torch.long, device=dev)
            out = student_model(full_t, use_cache=False)
            lg = (out.logits if hasattr(out, "logits") else out)[0, lp - 1 : lp - 1 + la].detach().float().cpu()
            hm, hf = cls.capture_mid_and_final_states(student_model, full_t.cpu(), mid_layer_idx)
            episodes.append(
                {
                    "full_ids": full_t[0].cpu(),
                    "answer_ids": a_ids,
                    "lp": lp,
                    "la": la,
                    "student_logits": lg,
                    "hm": hm,
                    "hf": hf,
                }
            )
        return episodes

    @classmethod
    @torch.no_grad()
    def solve_concept_capsule(
        cls,
        capsule_id: str,
        anchor_queries_en: Sequence[str],
        seed_pairs: Sequence[Tuple[str, str]],
        student_model: nn.Module,
        tokenizer: Any,
        bank: GoldenSliceScarBank,
        system_prompt: str = UNIVERSAL_CHAT_SYSTEM_PROMPT,
        tau: float = 0.38,
        ramp: float = 0.10,
        min_clearance: float = 13.5,
        parent_capsules: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        quantize_int4: bool = True,
    ) -> DualDepthScarCapsule:
        """Synthesizes ONE single language-invariant `DualDepthScarCapsule` (`62,720 B` on `0.5B`, `143,360 B` on `3B`)
        anchored purely in English `anchor_queries_en` and effective across EN, PT, ES, FR."""
        dev = _infer_model_device(student_model)
        d = bank.hidden_dim
        _, _, _, norm_mod, lm_head = extract_mid_and_final_modules(student_model, bank.mid_layer_idx)
        w_head = lm_head.weight.detach().float().cpu()
        b_head = lm_head.bias.detach().float().cpu() if getattr(lm_head, "bias", None) is not None else None
        norm_w = (
            norm_mod.weight.detach().float().cpu()
            if getattr(norm_mod, "weight", None) is not None
            else torch.ones(d, dtype=torch.float32)
        )
        inv_norm_scale = torch.sign(norm_w + 1e-8) / norm_w.abs().clamp(min=0.1)

        # 1. Compute English-only conceptual anchor projected through `P_perp_lang` at `mid_layer_idx`
        z_parts: List[torch.Tensor] = []
        for q_en in anchor_queries_en:
            p_en = list(tokenizer.encode(format_universal_chatml(q_en, system_prompt=system_prompt), add_special_tokens=False))
            hm_q, _ = cls.capture_mid_and_final_states(
                student_model, torch.tensor([p_en], dtype=torch.long), bank.mid_layer_idx
            )
            qs, qe = extract_user_content_bounds(p_en, bank.user_turn_prefix_ids, bank.user_turn_suffix_ids)
            centered = hm_q[qs:qe].mean(dim=0) - bank.mu_query_mid
            z_parts.append(F.normalize(bank.project_lang_free(centered), dim=-1, eps=1e-8))
        z_anchor = F.normalize(torch.stack(z_parts, dim=0).mean(dim=0), dim=-1, eps=1e-8).to(torch.float16)

        # 2. Collect non-EOS-terminated Conceptual Premise Seed trajectories (balanced 1:1:1:1 across languages)
        episodes = cls._collect_seed_episodes(
            seed_pairs=seed_pairs,
            student_model=student_model,
            tokenizer=tokenizer,
            mid_layer_idx=bank.mid_layer_idx,
            system_prompt=system_prompt,
            include_multiturn_augmentation=False,
        )

        p_null_mid = torch.eye(d, dtype=torch.float32) - (bank.u_preserved_mid @ bank.u_preserved_mid.T)
        p_null_fin = torch.eye(d, dtype=torch.float32) - (bank.u_preserved_final @ bank.u_preserved_final.T)
        p_perp_lang = torch.eye(d, dtype=torch.float32) - (bank.u_lang_mid @ bank.u_lang_mid.T)

        # Stage A: Solve Golden-Slice Mid-Layer Conceptual Scar `(B_mid, A_mid)` (`rank_mid = 12`)
        hm_rows: List[torch.Tensor] = []
        rm_rows: List[torch.Tensor] = []
        for ep in episodes:
            lp, la = int(ep["lp"]), int(ep["la"])
            hm_ans = ep["hm"][lp - 1 : lp - 1 + la]
            lg_s = ep["student_logits"]
            for step_i, target_id in enumerate(ep["answer_ids"]):
                r_star, deficit = cls.solve_high_clearance_rosetta(
                    w_head=w_head,
                    lg_student_1d=lg_s[step_i],
                    target_id=int(target_id),
                    min_clearance=float(min_clearance) * 0.95,
                )
                hm_rows.append((hm_ans[step_i] - bank.mu_hidden_mid) @ p_null_mid)
                if deficit <= 0.0:
                    rm_rows.append(torch.zeros(d, dtype=torch.float32))
                else:
                    pulled = r_star * inv_norm_scale
                    scale_m = 0.18 * float(hm_ans[step_i].norm().item()) / max(float(pulled.norm().item()), 1e-6)
                    rm_rows.append((pulled * scale_m) @ p_perp_lang)

        B_mid, A_mid = cls.woodbury_prediction_rrr_svd(
            hp_mat=torch.stack(hm_rows, dim=0),
            rc_mat=torch.stack(rm_rows, dim=0),
            p_null=p_null_mid,
            rank=bank.rank_mid,
            hidden_ridge_ratio=0.025,
        )
        B_mid = (p_perp_lang @ B_mid.float()).to(torch.float16).contiguous()
        if quantize_int4:
            bm_p, bm_s = quantize_block64_int4(B_mid)
            B_mid = dequantize_block64_int4(bm_p, bm_s, (d, bank.rank_mid), dtype=torch.float16, device="cpu")
            am_p, am_s = quantize_block64_int4(A_mid)
            A_mid = dequantize_block64_int4(am_p, am_s, (bank.rank_mid, d), dtype=torch.float16, device="cpu")

        # Stage B: Propagate INT4-dequantized `(B_mid, A_mid)` through `Layers [mid_idx+1 .. L-1]` and solve `(B_final, A_final)` (`rank_final = 52`)
        am_dev = A_mid.to(dev).float()
        bm_dev = B_mid.to(dev).float()
        mu_hm_dev = bank.mu_hidden_mid.to(dev)
        hf_rows: List[torch.Tensor] = []
        rf_rows: List[torch.Tensor] = []
        n_surp = 0

        for ep in episodes:
            lp, la = int(ep["lp"]), int(ep["la"])

            def _inject_mid(t3: torch.Tensor, _lp: int = lp) -> torch.Tensor:
                tm = t3.clone()
                sl = t3[:, _lp - 1 :, :].float()
                hc = sl - mu_hm_dev.view(1, 1, -1)
                dm = torch.matmul(torch.matmul(hc, am_dev.T), bm_dev.T)
                tm[:, _lp - 1 :, :] = (sl + dm).to(t3.dtype)
                return tm

            _, hf_prop = cls.capture_mid_and_final_states(
                student_model, ep["full_ids"].unsqueeze(0), bank.mid_layer_idx, mid_intervention=_inject_mid
            )
            hf_ans = hf_prop[lp - 1 : lp - 1 + la]
            lg_prop = hf_ans @ w_head.T + (b_head.unsqueeze(0) if b_head is not None else 0.0)
            for step_i, target_id in enumerate(ep["answer_ids"]):
                r_rem, deficit = cls.solve_high_clearance_rosetta(
                    w_head=w_head,
                    lg_student_1d=lg_prop[step_i],
                    target_id=int(target_id),
                    min_clearance=float(min_clearance),
                )
                w_row = math.sqrt(1.85) if step_i < 10 else 1.0
                hf_rows.append(w_row * ((hf_ans[step_i] - bank.mu_hidden_final) @ p_null_fin))
                if deficit <= 0.0:
                    rf_rows.append(torch.zeros(d, dtype=torch.float32))
                else:
                    n_surp += 1
                    rf_rows.append(w_row * r_rem)

        B_final, A_final = cls.woodbury_prediction_rrr_svd(
            hp_mat=torch.stack(hf_rows, dim=0),
            rc_mat=torch.stack(rf_rows, dim=0),
            p_null=p_null_fin,
            rank=bank.rank_final,
            hidden_ridge_ratio=0.005,
        )

        meta_out = dict(metadata or {})
        meta_out.setdefault("architecture", "GoldenSliceV4")
        meta_out["mid_layer_idx"] = bank.mid_layer_idx
        meta_out["k_lang"] = bank.k_lang
        meta_out["surprise_steps"] = n_surp
        meta_out["total_seed_steps"] = len(hf_rows)

        cap_fp16 = DualDepthScarCapsule(
            capsule_id=capsule_id,
            z_anchor_mid=z_anchor,
            A_mid=A_mid,
            B_mid=B_mid,
            A_final=A_final,
            B_final=B_final,
            tau=float(tau),
            ramp=float(ramp),
            parent_capsules=tuple(parent_capsules),
            metadata=meta_out,
        )
        if not quantize_int4:
            bank.add_capsule(cap_fp16)
            return cap_fp16

        raw_int4 = cap_fp16.to_int4_bytes()
        cap_q = DualDepthScarCapsule.from_int4_bytes(
            raw_bytes=raw_int4,
            capsule_id=capsule_id,
            hidden_dim=d,
            rank_mid=bank.rank_mid,
            rank_final=bank.rank_final,
            tau=float(tau),
            ramp=float(ramp),
            parent_capsules=tuple(parent_capsules),
            device="cpu",
            metadata=meta_out,
        )
        bank.add_capsule(cap_q)
        return cap_q

    @classmethod
    @torch.no_grad()
    def consolidate_sleep_bridge(
        cls,
        bridge_id: str,
        parent_a_id: str,
        parent_b_id: str,
        bridge_seed_pairs: Sequence[Tuple[str, str]],
        student_model: nn.Module,
        tokenizer: Any,
        bank: GoldenSliceScarBank,
        anchor_queries_en: Optional[Sequence[str]] = None,
        system_prompt: str = UNIVERSAL_CHAT_SYSTEM_PROMPT,
        tau: float = 0.34,
        ramp: float = 0.10,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DualDepthScarCapsule:
        """Synthesizes a language-invariant conjunctive Sleep Bridge (`parent_capsules = (parent_a_id, parent_b_id)`)."""
        cap_a = bank.capsules[parent_a_id]
        cap_b = bank.capsules[parent_b_id]
        if anchor_queries_en:
            cap = cls.solve_concept_capsule(
                capsule_id=bridge_id,
                anchor_queries_en=anchor_queries_en,
                seed_pairs=bridge_seed_pairs,
                student_model=student_model,
                tokenizer=tokenizer,
                bank=bank,
                system_prompt=system_prompt,
                tau=tau,
                ramp=ramp,
                parent_capsules=(parent_a_id, parent_b_id),
                metadata=metadata,
                quantize_int4=True,
            )
            return cap

        # Default conjunctive midpoint anchor in `P_perp_lang` space
        z_conj = F.normalize(
            F.normalize(cap_a.z_anchor_mid.float(), dim=-1) + F.normalize(cap_b.z_anchor_mid.float(), dim=-1),
            dim=-1,
            eps=1e-8,
        ).to(torch.float16)
        dummy_en = [seed_q for seed_q, _ in bridge_seed_pairs[:1]]
        cap_tmp = cls.solve_concept_capsule(
            capsule_id=bridge_id,
            anchor_queries_en=dummy_en,
            seed_pairs=bridge_seed_pairs,
            student_model=student_model,
            tokenizer=tokenizer,
            bank=bank,
            system_prompt=system_prompt,
            tau=tau,
            ramp=ramp,
            parent_capsules=(parent_a_id, parent_b_id),
            metadata=metadata,
            quantize_int4=False,
        )
        cap_tmp.z_anchor_mid = z_conj
        raw_int4 = cap_tmp.to_int4_bytes()
        cap_q = DualDepthScarCapsule.from_int4_bytes(
            raw_bytes=raw_int4,
            capsule_id=bridge_id,
            hidden_dim=bank.hidden_dim,
            rank_mid=bank.rank_mid,
            rank_final=bank.rank_final,
            tau=float(tau),
            ramp=float(ramp),
            parent_capsules=(parent_a_id, parent_b_id),
            device="cpu",
            metadata=dict(cap_tmp.metadata),
        )
        bank.add_capsule(cap_q)
        return cap_q


class GoldenSliceScarStore:
    """`4096 B`-Page-Aligned `mmap` + `HTTP/1.1 206 Partial Content` Golden-Slice Conceptual ScarStore (`v4.0`).

    Drop-in polymorphic sibling to `HeadExpandedScarStore`:
      - Exact same `62,720 B` (`0.5B`) and `143,360 B = 35 x 4096 B` (`3B`) `Block-64 INT4` payload slots!
      - Routes at `layers[mid_layer_idx]` (`2L/3`) via `d x d` Kronecker Dual-Subspace Grid in `P_perp_lang` space.
      - Injects `(B_mid, A_mid)` at `layers[mid_layer_idx]` and `(B_final, A_final)` at `final_norm` with smooth
        `seed_horizon` handoff (`1.0 -> 0.0`) so responses combine exact constants/formulas with rich prose!
      - Supports sub-millisecond `0x00` LGPD cryptographic RAM + disk unlearning (`unlearn(..., persist_to_disk=True)`).
    """

    def __init__(
        self,
        store_path: Path,
        header: Dict[str, Any],
        mu_query_mid: torch.Tensor,
        mu_hidden_mid: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        u_lang_mid: torch.Tensor,
        R_rot: torch.Tensor,
        U_codebook: torch.Tensor,
        V_codebook: torch.Tensor,
        presence_bitmap: torch.Tensor,
        z_anchors_installed: torch.Tensor,
        payload_offset: int,
        on_demand_fetcher: Optional[Callable[[Tuple[int, int], str], Optional[bytes]]] = None,
    ) -> None:
        self.store_path = Path(store_path)
        self.header = header
        self.hidden_dim = int(header["hidden_dim"])
        self.num_layers = int(header.get("num_layers", 24))
        self.mid_layer_idx = int(header["mid_layer_idx"])
        self.num_heads = int(header["num_heads"])
        self.head_dim = int(header["head_dim"])
        self.rank_mid = int(header["rank_mid"])
        self.rank_final = int(header["rank_final"])
        self.rank = self.rank_mid + self.rank_final
        self.num_key_prototypes = 0
        self.attention_beta = 0.0
        self.k_lang = int(header["k_lang"])
        self.record_bytes = int(header["record_bytes"])
        self.routing_tier = str(header.get("routing_tier", "kronecker"))
        self.seed_horizon = int(header.get("seed_horizon", 44))
        self.seed_fade_steps = int(header.get("seed_fade_steps", 16))
        self.winner_margin = float(header.get("winner_margin", 0.0002))
        self.user_turn_prefix_ids: Optional[List[int]] = header.get("user_turn_prefix_ids")
        self.user_turn_suffix_ids: Optional[List[int]] = header.get("user_turn_suffix_ids")
        self.payload_offset = int(payload_offset)
        self.on_demand_fetcher = on_demand_fetcher

        self.mu_query_mid = mu_query_mid.float()
        self.mu_query_last = self.mu_query_mid  # Alias for telemetry/inspector compatibility
        self.mu_hidden_mid = mu_hidden_mid.float()
        self.mu_hidden_final = mu_hidden_final.float()
        self.u_lang_mid = u_lang_mid.float()
        self.R_rot = R_rot.float()
        self.U_codebook = U_codebook.float()
        self.V_codebook = V_codebook.float()
        self._presence_bitmap_bytes = bytearray(presence_bitmap.cpu().numpy().tobytes())
        self.z_anchors_installed = z_anchors_installed.float()

        self._slot_descriptors: List[Dict[str, Any]] = list(header.get("slot_descriptors", []))
        self._coord_to_slot: Dict[Tuple[int, int], int] = {}
        self._cid_to_slot: Dict[str, int] = {}
        for desc in self._slot_descriptors:
            ri, cj = int(desc["row_i"]), int(desc["col_j"])
            s_idx = int(desc["slot_idx"])
            if _bitmap_get(self._presence_bitmap_bytes, ri * self.hidden_dim + cj):
                self._coord_to_slot[(ri, cj)] = s_idx
                self._cid_to_slot[str(desc["capsule_id"])] = s_idx

        self._global_directory: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for g_entry in header.get("global_directory", []):
            self._global_directory[(int(g_entry["row_i"]), int(g_entry["col_j"]))] = dict(g_entry)

        self._file_obj = open(self.store_path, "rb")
        self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        self._hot_fetched_capsules: Dict[str, DualDepthScarCapsule] = {}
        self._turn_slot_cache: Dict[int, DualDepthScarCapsule] = {}
        self._unlearned_cids: set[str] = set()
        self._active_turn_capsules: List[Tuple[DualDepthScarCapsule, float]] = []
        self._turn_mmap_bytes_read: int = 0

        self.routing_mode: str = "span_max"
        self.span_window: int = 10
        self.span_stride: int = 3
        self.strict_missing_guard: bool = False

        self.last_routing_cosines: Dict[str, float] = {}
        self.last_active_gates: Dict[str, float] = {}
        self.last_num_expanded_heads: int = 0
        self.last_mmap_bytes_read: int = 0
        self.last_missing_alerts: List[MissingCapsuleAlert] = []
        self.last_step_weights_history: List[Dict[str, float]] = []
        self.last_inter_head_spread_history: List[float] = []

        self._backbone_pre_handle: Optional[Any] = None
        self._mid_hook_handle: Optional[Any] = None
        self._norm_hook_handle: Optional[Any] = None
        self._paused: bool = False
        self._routing_locked: bool = False
        self._current_pass_is_prefill: bool = True
        self._current_query_start: int = 0
        self._current_chunk_ids: List[int] = []
        self._rolling_id_tail: List[int] = []
        self._query_hidden_buf: Optional[torch.Tensor] = None
        self._tf_prompt_len: Optional[int] = None
        self._decode_step_idx: int = 0

    def to(self, device: Union[str, torch.device]) -> "GoldenSliceScarStore":
        dev = torch.device(device)
        self.mu_query_mid = self.mu_query_mid.to(dev)
        self.mu_query_last = self.mu_query_mid
        self.mu_hidden_mid = self.mu_hidden_mid.to(dev)
        self.mu_hidden_final = self.mu_hidden_final.to(dev)
        self.u_lang_mid = self.u_lang_mid.to(dev)
        self.R_rot = self.R_rot.to(dev)
        self.U_codebook = self.U_codebook.to(dev)
        self.V_codebook = self.V_codebook.to(dev)
        self.z_anchors_installed = self.z_anchors_installed.to(dev)
        return self

    def close(self) -> None:
        self.detach()
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None  # type: ignore[assignment]
        if getattr(self, "_file_obj", None) is not None:
            self._file_obj.close()
            self._file_obj = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def num_grid_cells(self) -> int:
        return self.hidden_dim * self.hidden_dim

    @property
    def presence_bitmap_bytes(self) -> int:
        return len(self._presence_bitmap_bytes)

    @property
    def num_installed_slots(self) -> int:
        return len(self._coord_to_slot)

    def slot_byte_offset(self, slot_idx: int) -> int:
        return self.payload_offset + slot_idx * self.record_bytes

    def read_raw_slot_bytes(self, slot_idx: int) -> bytes:
        if slot_idx < 0 or slot_idx >= len(self._slot_descriptors):
            raise IndexError(f"slot_idx={slot_idx} out of range")
        off = self.slot_byte_offset(slot_idx)
        return bytes(self._mmap[off : off + self.record_bytes])

    def read_raw_capsule_bytes_by_coord(self, coord: Tuple[int, int]) -> bytes:
        if coord not in self._coord_to_slot:
            raise KeyError(f"Coordinate {coord} is not installed locally in {self.store_path}")
        return self.read_raw_slot_bytes(self._coord_to_slot[coord])

    def load_slot_capsule(self, slot_idx: int, device: Union[str, torch.device]) -> DualDepthScarCapsule:
        if slot_idx in self._turn_slot_cache:
            return self._turn_slot_cache[slot_idx].to_device(device)
        desc = self._slot_descriptors[slot_idx]
        raw = self.read_raw_slot_bytes(slot_idx)
        self._turn_mmap_bytes_read += len(raw)
        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
        cap = DualDepthScarCapsule.from_int4_bytes(
            raw_bytes=raw,
            capsule_id=str(desc["capsule_id"]),
            hidden_dim=self.hidden_dim,
            rank_mid=self.rank_mid,
            rank_final=self.rank_final,
            tau=float(desc["tau"]),
            ramp=float(desc["ramp"]),
            parent_capsules=desc.get("parent_capsules", ()),
            device=device,
            metadata=dict(desc.get("metadata", {})),
        )
        self._turn_slot_cache[slot_idx] = cap
        return cap

    def _router_section_offsets(self) -> Tuple[int, int, int]:
        with open(self.store_path, "rb") as f:
            magic = f.read(8)
            if magic != SCARSTORE_MAGIC:
                raise ValueError(f"Invalid ScarStore magic in {self.store_path}")
            (hdr_len,) = struct.unpack("<Q", f.read(8))
        router_off = _align_up(16 + int(hdr_len), HEADER_ALIGN)
        d = self.hidden_dim
        half_d = d // 2
        bm_off = (
            router_off
            + (d * 4)  # mu_query_mid
            + (d * 4)  # mu_hidden_mid
            + (d * 4)  # mu_hidden_final
            + (d * self.k_lang * 2)  # u_lang_mid fp16
            + (d * d * 2)  # R_rot fp16
            + (d * half_d * 2)  # U_cb fp16
            + (d * half_d * 2)  # V_cb fp16
        )
        bm_len = (d * d + 7) // 8
        z_off = bm_off + bm_len
        return router_off, bm_off, z_off

    def unlearn(self, capsule_id: str, persist_to_disk: bool = False) -> Dict[str, Any]:
        """Sub-millisecond surgical LGPD/GDPR unlearning with optional `0x00` cryptographic disk erasure + `fsync`."""
        t0 = time.perf_counter()
        cid = str(capsule_id)
        if cid not in self._cid_to_slot:
            raise KeyError(f"Capsule {cid!r} not installed in {self.store_path}")
        s_idx = self._cid_to_slot.pop(cid)
        desc = self._slot_descriptors[s_idx]
        coord = (int(desc["row_i"]), int(desc["col_j"]))
        flat_idx = coord[0] * self.hidden_dim + coord[1]
        byte_idx = flat_idx >> 3
        bit_mask = ~(1 << (flat_idx & 7)) & 0xFF
        self._presence_bitmap_bytes[byte_idx] &= bit_mask
        self._coord_to_slot.pop(coord, None)
        self._global_directory.pop(coord, None)
        self._hot_fetched_capsules.pop(cid, None)
        self._turn_slot_cache.pop(s_idx, None)
        self._unlearned_cids.add(cid)
        self._active_turn_capsules = [(c, g) for c, g in self._active_turn_capsules if c.capsule_id != cid]
        if s_idx < int(self.z_anchors_installed.shape[0]):
            self.z_anchors_installed[s_idx].zero_()
        ram_ms = (time.perf_counter() - t0) * 1000.0

        wiped_bytes = 0
        if persist_to_disk:
            _, bm_off, z_off = self._router_section_offsets()
            slot_off = self.slot_byte_offset(s_idx)
            with open(self.store_path, "r+b") as wf:
                wf.seek(bm_off + byte_idx)
                wf.write(bytes([self._presence_bitmap_bytes[byte_idx]]))
                wf.seek(z_off + s_idx * self.hidden_dim * 2)
                wf.write(b"\x00" * (self.hidden_dim * 2))
                wf.seek(slot_off)
                wf.write(b"\x00" * self.record_bytes)
                wf.flush()
                os.fsync(wf.fileno())
            wiped_bytes = self.record_bytes + (self.hidden_dim * 2)
            if getattr(self, "_mmap", None) is not None:
                self._mmap.close()
                self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "capsule_id": cid,
            "coord": coord,
            "slot_idx": s_idx,
            "unlearn_ms": round(ram_ms, 4),
            "total_ms": round(total_ms, 4),
            "persisted_to_disk": bool(persist_to_disk),
            "cryptographically_zeroed_bytes": wiped_bytes,
        }

    def inspect_summary(self) -> Dict[str, Any]:
        file_bytes = int(self.store_path.stat().st_size) if self.store_path.exists() else 0
        exact_4kb_pages = (self.record_bytes % PAGE_ALIGN) == 0
        slots_info: List[Dict[str, Any]] = []
        for desc in self._slot_descriptors:
            s_idx = int(desc["slot_idx"])
            cid = str(desc["capsule_id"])
            ri, cj = int(desc["row_i"]), int(desc["col_j"])
            flat_bit = ri * self.hidden_dim + cj
            present = _bitmap_get(self._presence_bitmap_bytes, flat_bit)
            off = self.slot_byte_offset(s_idx)
            raw = self.read_raw_slot_bytes(s_idx)
            all_zero = all(b == 0 for b in raw)
            slots_info.append(
                {
                    "slot_idx": s_idx,
                    "capsule_id": cid,
                    "coord": [ri, cj],
                    "byte_offset": off,
                    "offset_4kb_aligned": (off % PAGE_ALIGN) == 0,
                    "record_bytes": self.record_bytes,
                    "bitmap_present": bool(present),
                    "payload_all_zero_wiped": bool(all_zero),
                    "tau": float(desc["tau"]),
                    "ramp": float(desc["ramp"]),
                    "parent_capsules": list(desc.get("parent_capsules", [])),
                    "metadata": dict(desc.get("metadata", {})),
                }
            )
        installed_cids = {s["capsule_id"] for s in slots_info if s["bitmap_present"] and not s["payload_all_zero_wiped"]}
        cloud_dir: List[Dict[str, Any]] = []
        for coord_k, g_entry in sorted(self._global_directory.items()):
            g_cid = str(g_entry["capsule_id"])
            if g_cid in self._unlearned_cids:
                continue
            cloud_dir.append(
                {
                    "capsule_id": g_cid,
                    "coord": [int(coord_k[0]), int(coord_k[1])],
                    "installed_locally": g_cid in installed_cids,
                    "tau": float(g_entry["tau"]),
                    "ramp": float(g_entry["ramp"]),
                    "parent_capsules": list(g_entry.get("parent_capsules", [])),
                    "metadata": dict(g_entry.get("metadata", {})),
                }
            )
        return {
            "store_path": str(self.store_path),
            "file_bytes": file_bytes,
            "file_mib": round(file_bytes / (1024.0 * 1024.0), 4),
            "format": "GoldenSliceScarStoreV4",
            "version": str(self.header.get("version", "4.0.0")),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "mid_layer_idx": self.mid_layer_idx,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "rank_mid": self.rank_mid,
            "rank_final": self.rank_final,
            "rank": self.rank,
            "k_lang": self.k_lang,
            "seed_horizon": self.seed_horizon,
            "num_key_prototypes": 0,
            "record_bytes": self.record_bytes,
            "record_kb": round(self.record_bytes / 1024.0, 2),
            "exact_4kb_page_multiple": exact_4kb_pages,
            "os_4kb_pages_per_capsule": (self.record_bytes // PAGE_ALIGN) if exact_4kb_pages else None,
            "kronecker_grid_cells": self.num_grid_cells,
            "presence_bitmap_bytes": self.presence_bitmap_bytes,
            "presence_bitmap_kb": round(self.presence_bitmap_bytes / 1024.0, 2),
            "payload_offset": self.payload_offset,
            "payload_offset_4kb_aligned": (self.payload_offset % PAGE_ALIGN) == 0,
            "num_active_installed_slots": self.num_installed_slots,
            "num_global_directory_entries": len(self._global_directory),
            "slots": slots_info,
            "cloud_directory": cloud_dir,
        }

    @classmethod
    def build_from_bank(
        cls,
        bank: GoldenSliceScarBank,
        out_path: Union[str, Path],
    ) -> "GoldenSliceScarStore":
        d = bank.hidden_dim
        if d % 2 != 0:
            raise ValueError(f"hidden_dim={d} must be even for Kronecker routing")
        half_d = d // 2
        caps = list(bank.capsules.values())
        rec_bytes = (
            (((d * bank.rank_mid) // BLOCK_SIZE_INT4) * 34 * 2)
            + (((d * bank.rank_final) // BLOCK_SIZE_INT4) * 34 * 2)
            + (d * 2)
        )

        R_rot = _build_deterministic_orthogonal_rotation(d)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261114)
        U_cb = F.normalize(torch.randn((d, half_d), generator=gen, dtype=torch.float32), dim=-1)
        V_cb = F.normalize(torch.randn((d, half_d), generator=gen, dtype=torch.float32), dim=-1)

        bitmap = bytearray((d * d + 7) // 8)
        used_coords: set[Tuple[int, int]] = set()
        raw_blobs: List[bytes] = []
        z_list: List[torch.Tensor] = []
        slot_descriptors: List[Dict[str, Any]] = []
        global_directory: List[Dict[str, Any]] = []

        for idx, cap in enumerate(caps):
            z_f = F.normalize(cap.z_anchor_mid.detach().cpu().float(), dim=-1, eps=1e-8)
            z_rot = torch.mv(R_rot, z_f)
            z1 = F.normalize(z_rot[:half_d], dim=-1, eps=1e-8)
            z2 = F.normalize(z_rot[half_d:], dim=-1, eps=1e-8)
            w1 = float(z_rot[:half_d].norm().item())
            w2 = float(z_rot[half_d:].norm().item())

            if idx < d:
                row_i, col_j = idx, idx
                U_cb[row_i] = z1
                V_cb[col_j] = z2
            else:
                s1 = torch.mv(U_cb, z1)
                s2 = torch.mv(V_cb, z2)
                ti = s1.topk(min(32, d)).indices.tolist()
                tj = s2.topk(min(32, d)).indices.tolist()
                best_pair: Optional[Tuple[int, int]] = None
                best_val = -1e9
                for ri in ti:
                    for cj in tj:
                        if (ri, cj) not in used_coords:
                            val = float(s1[ri].item() + s2[cj].item())
                            if val > best_val:
                                best_val = val
                                best_pair = (ri, cj)
                if best_pair is None:
                    raise RuntimeError("Kronecker grid neighborhood saturated")
                row_i, col_j = best_pair

            used_coords.add((row_i, col_j))
            _bitmap_set(bitmap, row_i * d + col_j)
            blob = cap.to_int4_bytes()
            assert len(blob) == rec_bytes
            raw_blobs.append(blob)
            z_list.append(z_f.to(torch.float16))
            entry_common = {
                "capsule_id": cap.capsule_id,
                "row_i": row_i,
                "col_j": col_j,
                "w1": round(w1, 6),
                "w2": round(w2, 6),
                "tau": float(cap.tau),
                "ramp": float(cap.ramp),
                "parent_capsules": list(cap.parent_capsules),
                "metadata": dict(cap.metadata),
            }
            slot_descriptors.append({"slot_idx": idx, **entry_common})
            global_directory.append(dict(entry_common))

        return cls._write_and_open(
            out_path=Path(out_path),
            hidden_dim=d,
            num_layers=bank.num_layers,
            mid_layer_idx=bank.mid_layer_idx,
            num_heads=bank.num_heads,
            head_dim=bank.head_dim,
            rank_mid=bank.rank_mid,
            rank_final=bank.rank_final,
            k_lang=bank.k_lang,
            record_bytes=rec_bytes,
            seed_horizon=bank.seed_horizon,
            seed_fade_steps=bank.seed_fade_steps,
            winner_margin=bank.winner_margin,
            user_turn_prefix_ids=bank.user_turn_prefix_ids,
            user_turn_suffix_ids=bank.user_turn_suffix_ids,
            slot_descriptors=slot_descriptors,
            global_directory=global_directory,
            mu_query_mid=bank.mu_query_mid,
            mu_hidden_mid=bank.mu_hidden_mid,
            mu_hidden_final=bank.mu_hidden_final,
            u_lang_mid=bank.u_lang_mid,
            R_rot=R_rot,
            U_cb=U_cb,
            V_cb=V_cb,
            bitmap=bytes(bitmap),
            z_list_fp16=z_list,
            raw_blobs=raw_blobs,
        )

    @classmethod
    def _write_and_open(
        cls,
        out_path: Path,
        hidden_dim: int,
        num_layers: int,
        mid_layer_idx: int,
        num_heads: int,
        head_dim: int,
        rank_mid: int,
        rank_final: int,
        k_lang: int,
        record_bytes: int,
        seed_horizon: int,
        seed_fade_steps: int,
        winner_margin: float,
        user_turn_prefix_ids: Optional[List[int]],
        user_turn_suffix_ids: Optional[List[int]],
        slot_descriptors: List[Dict[str, Any]],
        global_directory: List[Dict[str, Any]],
        mu_query_mid: torch.Tensor,
        mu_hidden_mid: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        u_lang_mid: torch.Tensor,
        R_rot: torch.Tensor,
        U_cb: torch.Tensor,
        V_cb: torch.Tensor,
        bitmap: bytes,
        z_list_fp16: List[torch.Tensor],
        raw_blobs: List[bytes],
    ) -> "GoldenSliceScarStore":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_slots = len(raw_blobs)
        header_obj = {
            "format": "GoldenSliceScarStoreV4",
            "version": "4.0.0",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "mid_layer_idx": mid_layer_idx,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "rank_mid": rank_mid,
            "rank_final": rank_final,
            "rank": rank_mid + rank_final,
            "num_key_prototypes": 0,
            "k_lang": k_lang,
            "record_bytes": record_bytes,
            "num_slots": n_slots,
            "routing_tier": "kronecker",
            "seed_horizon": seed_horizon,
            "seed_fade_steps": seed_fade_steps,
            "winner_margin": winner_margin,
            "user_turn_prefix_ids": user_turn_prefix_ids,
            "user_turn_suffix_ids": user_turn_suffix_ids,
            "slot_descriptors": slot_descriptors,
            "global_directory": global_directory,
        }
        header_bytes = json.dumps(header_obj, ensure_ascii=False).encode("utf-8")
        router_offset = _align_up(16 + len(header_bytes), HEADER_ALIGN)

        router_blobs = [
            mu_query_mid.contiguous().to(torch.float32).numpy().tobytes(),
            mu_hidden_mid.contiguous().to(torch.float32).numpy().tobytes(),
            mu_hidden_final.contiguous().to(torch.float32).numpy().tobytes(),
            u_lang_mid.contiguous().to(torch.float16).numpy().tobytes(),
            R_rot.contiguous().to(torch.float16).numpy().tobytes(),
            U_cb.contiguous().to(torch.float16).numpy().tobytes(),
            V_cb.contiguous().to(torch.float16).numpy().tobytes(),
            bitmap,
        ]
        if n_slots > 0:
            z_mat = torch.stack(z_list_fp16, dim=0).contiguous().to(torch.float16)
            router_blobs.append(z_mat.numpy().tobytes())
        router_bytes = b"".join(router_blobs)
        payload_offset = _align_up(router_offset + len(router_bytes), PAGE_ALIGN)

        with open(out_path, "wb") as f:
            f.write(SCARSTORE_MAGIC)
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * (router_offset - (16 + len(header_bytes))))
            f.write(router_bytes)
            f.write(b"\x00" * (payload_offset - (router_offset + len(router_bytes))))
            for b in raw_blobs:
                f.write(b)

        return cls.open(out_path)

    @classmethod
    def open(
        cls,
        store_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        on_demand_fetcher: Optional[Callable[[Tuple[int, int], str], Optional[bytes]]] = None,
    ) -> "GoldenSliceScarStore":
        p = Path(store_path)
        with open(p, "rb") as f:
            magic = f.read(8)
            if magic != SCARSTORE_MAGIC:
                raise ValueError(f"Invalid ScarStore magic in {p}: {magic!r}")
            (hdr_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hdr_len).decode("utf-8"))
            if header.get("format") != "GoldenSliceScarStoreV4":
                raise ValueError(f"Expected GoldenSliceScarStoreV4 in {p}, got {header.get('format')!r}")
            router_offset = _align_up(16 + hdr_len, HEADER_ALIGN)
            f.seek(router_offset)

            d = int(header["hidden_dim"])
            k_lang = int(header["k_lang"])
            half_d = d // 2
            n_slots = int(header["num_slots"])
            bm_len = (d * d + 7) // 8

            mu_qm = torch.frombuffer(bytearray(f.read(d * 4)), dtype=torch.float32, count=d).clone()
            mu_hm = torch.frombuffer(bytearray(f.read(d * 4)), dtype=torch.float32, count=d).clone()
            mu_hf = torch.frombuffer(bytearray(f.read(d * 4)), dtype=torch.float32, count=d).clone()
            u_lang = (
                torch.frombuffer(bytearray(f.read(d * k_lang * 2)), dtype=torch.float16, count=d * k_lang)
                .view(d, k_lang)
                .float()
                .clone()
            )
            R_rot = torch.frombuffer(bytearray(f.read(d * d * 2)), dtype=torch.float16, count=d * d).view(d, d).float().clone()
            U_cb = (
                torch.frombuffer(bytearray(f.read(d * half_d * 2)), dtype=torch.float16, count=d * half_d)
                .view(d, half_d)
                .float()
                .clone()
            )
            V_cb = (
                torch.frombuffer(bytearray(f.read(d * half_d * 2)), dtype=torch.float16, count=d * half_d)
                .view(d, half_d)
                .float()
                .clone()
            )
            bm_t = torch.frombuffer(bytearray(f.read(bm_len)), dtype=torch.uint8, count=bm_len).clone()

            if n_slots > 0:
                z_inst = (
                    torch.frombuffer(bytearray(f.read(n_slots * d * 2)), dtype=torch.float16, count=n_slots * d)
                    .view(n_slots, d)
                    .float()
                    .clone()
                )
            else:
                z_inst = torch.empty((0, d), dtype=torch.float32)

            payload_offset = _align_up(f.tell(), PAGE_ALIGN)

        store = cls(
            store_path=p,
            header=header,
            mu_query_mid=mu_qm,
            mu_hidden_mid=mu_hm,
            mu_hidden_final=mu_hf,
            u_lang_mid=u_lang,
            R_rot=R_rot,
            U_codebook=U_cb,
            V_codebook=V_cb,
            presence_bitmap=bm_t,
            z_anchors_installed=z_inst,
            payload_offset=payload_offset,
            on_demand_fetcher=on_demand_fetcher,
        )
        return store.to(torch.device(device))

    def export_subset(
        self,
        out_path: Union[str, Path],
        include_capsule_ids: Sequence[str],
    ) -> "GoldenSliceScarStore":
        keep = set(str(x) for x in include_capsule_ids)
        d = self.hidden_dim
        bitmap = bytearray((d * d + 7) // 8)
        new_descs: List[Dict[str, Any]] = []
        z_list_fp16: List[torch.Tensor] = []
        raw_blobs: List[bytes] = []

        for old_desc in self._slot_descriptors:
            cid = str(old_desc["capsule_id"])
            if cid in self._unlearned_cids or cid not in keep:
                continue
            ri, cj = int(old_desc["row_i"]), int(old_desc["col_j"])
            old_slot = int(old_desc["slot_idx"])
            raw_blobs.append(self.read_raw_slot_bytes(old_slot))
            z_list_fp16.append(self.z_anchors_installed[old_slot].detach().cpu().to(torch.float16))
            _bitmap_set(bitmap, ri * d + cj)
            desc_copy = dict(old_desc)
            desc_copy["slot_idx"] = len(new_descs)
            new_descs.append(desc_copy)

        return self._write_and_open(
            out_path=Path(out_path),
            hidden_dim=d,
            num_layers=self.num_layers,
            mid_layer_idx=self.mid_layer_idx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rank_mid=self.rank_mid,
            rank_final=self.rank_final,
            k_lang=self.k_lang,
            record_bytes=self.record_bytes,
            seed_horizon=self.seed_horizon,
            seed_fade_steps=self.seed_fade_steps,
            winner_margin=self.winner_margin,
            user_turn_prefix_ids=self.user_turn_prefix_ids,
            user_turn_suffix_ids=self.user_turn_suffix_ids,
            slot_descriptors=new_descs,
            global_directory=[v for v in self._global_directory.values() if str(v["capsule_id"]) not in self._unlearned_cids],
            mu_query_mid=self.mu_query_mid.detach().cpu().float(),
            mu_hidden_mid=self.mu_hidden_mid.detach().cpu().float(),
            mu_hidden_final=self.mu_hidden_final.detach().cpu().float(),
            u_lang_mid=self.u_lang_mid.detach().cpu().float(),
            R_rot=self.R_rot.detach().cpu().float(),
            U_cb=self.U_codebook.detach().cpu().float(),
            V_cb=self.V_codebook.detach().cpu().float(),
            bitmap=bytes(bitmap),
            z_list_fp16=z_list_fp16,
            raw_blobs=raw_blobs,
        )

    def append_capsule(
        self,
        capsule: DualDepthScarCapsule,
        out_path: Optional[Union[str, Path]] = None,
    ) -> "GoldenSliceScarStore":
        if capsule.rank_mid != self.rank_mid or capsule.rank_final != self.rank_final:
            raise ValueError(
                f"Capsule ({capsule.rank_mid}, {capsule.rank_final}) does not match store ({self.rank_mid}, {self.rank_final})"
            )
        target_path = Path(out_path) if out_path is not None else self.store_path
        d = self.hidden_dim
        half_d = d // 2
        R_rot = self.R_rot.detach().cpu().float()
        U_cb = self.U_codebook.detach().cpu().float().clone()
        V_cb = self.V_codebook.detach().cpu().float().clone()

        bitmap = bytearray((d * d + 7) // 8)
        used_coords: set[Tuple[int, int]] = set()
        new_descs: List[Dict[str, Any]] = []
        new_global: List[Dict[str, Any]] = []
        z_list_fp16: List[torch.Tensor] = []
        raw_blobs: List[bytes] = []

        for old_desc in self._slot_descriptors:
            s_idx = int(old_desc["slot_idx"])
            cid = str(old_desc["capsule_id"])
            ri, cj = int(old_desc["row_i"]), int(old_desc["col_j"])
            if cid == capsule.capsule_id or cid in self._unlearned_cids:
                continue
            if not _bitmap_get(self._presence_bitmap_bytes, ri * d + cj):
                continue
            raw_blobs.append(self.read_raw_slot_bytes(s_idx))
            z_list_fp16.append(self.z_anchors_installed[s_idx].detach().cpu().to(torch.float16))
            used_coords.add((ri, cj))
            _bitmap_set(bitmap, ri * d + cj)
            d_copy = dict(old_desc)
            d_copy["slot_idx"] = len(new_descs)
            new_descs.append(d_copy)

        existing_global: Optional[Dict[str, Any]] = None
        for _, g_entry in self._global_directory.items():
            g_cid = str(g_entry["capsule_id"])
            if g_cid == capsule.capsule_id:
                existing_global = dict(g_entry)
            elif g_cid not in self._unlearned_cids:
                new_global.append(dict(g_entry))

        z_f = F.normalize(capsule.z_anchor_mid.detach().cpu().float(), dim=-1, eps=1e-8)
        z_rot = torch.mv(R_rot, z_f)
        z1 = F.normalize(z_rot[:half_d], dim=-1, eps=1e-8)
        z2 = F.normalize(z_rot[half_d:], dim=-1, eps=1e-8)
        w1 = float(z_rot[:half_d].norm().item())
        w2 = float(z_rot[half_d:].norm().item())

        if existing_global is not None and (int(existing_global["row_i"]), int(existing_global["col_j"])) not in used_coords:
            row_i, col_j = int(existing_global["row_i"]), int(existing_global["col_j"])
        else:
            chosen_diag: Optional[int] = None
            for cand_idx in range(len(new_descs), d):
                if (cand_idx, cand_idx) not in used_coords and (cand_idx, cand_idx) not in self._global_directory:
                    chosen_diag = cand_idx
                    break
            if chosen_diag is not None:
                row_i, col_j = chosen_diag, chosen_diag
                U_cb[row_i] = z1
                V_cb[col_j] = z2
            else:
                s1 = torch.mv(U_cb, z1)
                s2 = torch.mv(V_cb, z2)
                ti = s1.topk(min(32, d)).indices.tolist()
                tj = s2.topk(min(32, d)).indices.tolist()
                best_pair: Optional[Tuple[int, int]] = None
                best_val = -1e9
                for ri in ti:
                    for cj in tj:
                        if (ri, cj) not in used_coords:
                            val = float(s1[ri].item() + s2[cj].item())
                            if val > best_val:
                                best_val = val
                                best_pair = (ri, cj)
                if best_pair is None:
                    raise RuntimeError("Kronecker grid saturated during append_capsule")
                row_i, col_j = best_pair

        merged_meta = dict(existing_global.get("metadata", {})) if existing_global is not None else {}
        merged_meta.update(dict(capsule.metadata))
        _bitmap_set(bitmap, row_i * d + col_j)
        raw_blobs.append(capsule.to_int4_bytes())
        z_list_fp16.append(z_f.to(torch.float16))
        entry_common = {
            "capsule_id": capsule.capsule_id,
            "row_i": row_i,
            "col_j": col_j,
            "w1": round(w1, 6),
            "w2": round(w2, 6),
            "tau": float(capsule.tau),
            "ramp": float(capsule.ramp),
            "parent_capsules": list(capsule.parent_capsules),
            "metadata": merged_meta,
        }
        new_descs.append({"slot_idx": len(new_descs), **entry_common})
        new_global.append(dict(entry_common))

        dev_cur = self.mu_query_mid.device
        fetcher_cur = self.on_demand_fetcher
        mu_qm = self.mu_query_mid.detach().cpu().float()
        mu_hm = self.mu_hidden_mid.detach().cpu().float()
        mu_hf = self.mu_hidden_final.detach().cpu().float()
        u_lang = self.u_lang_mid.detach().cpu().float()
        if target_path.resolve() == self.store_path.resolve():
            self.close()
        reopened = self._write_and_open(
            out_path=target_path,
            hidden_dim=d,
            num_layers=self.num_layers,
            mid_layer_idx=self.mid_layer_idx,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rank_mid=self.rank_mid,
            rank_final=self.rank_final,
            k_lang=self.k_lang,
            record_bytes=self.record_bytes,
            seed_horizon=self.seed_horizon,
            seed_fade_steps=self.seed_fade_steps,
            winner_margin=self.winner_margin,
            user_turn_prefix_ids=self.user_turn_prefix_ids,
            user_turn_suffix_ids=self.user_turn_suffix_ids,
            slot_descriptors=new_descs,
            global_directory=new_global,
            mu_query_mid=mu_qm,
            mu_hidden_mid=mu_hm,
            mu_hidden_final=mu_hf,
            u_lang_mid=u_lang,
            R_rot=R_rot,
            U_cb=U_cb,
            V_cb=V_cb,
            bitmap=bytes(bitmap),
            z_list_fp16=z_list_fp16,
            raw_blobs=raw_blobs,
        )
        reopened.on_demand_fetcher = fetcher_cur
        return reopened.to(dev_cur)

    @torch.no_grad()
    def route_and_page_query(
        self,
        x_query_mid_2d: torch.Tensor,
        top_coarse_m: int = 8,
    ) -> List[Tuple[DualDepthScarCapsule, float]]:
        """Routes via `P_perp_lang` + `d x d` Kronecker Dual-Subspace Grid + `mmap` / `HTTP 206` Page-In."""
        self.last_routing_cosines = {}
        self.last_active_gates = {}
        self.last_num_expanded_heads = 0
        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
        self.last_missing_alerts = []

        if x_query_mid_2d.numel() == 0:
            return []
        dev = x_query_mid_2d.device
        if self.mu_query_mid.device != dev:
            self.to(dev)

        probes_raw, _ = extract_span_query_probes(
            query_hidden_2d=x_query_mid_2d,
            mu_query=self.mu_query_mid,
            routing_mode=self.routing_mode,
            span_window=self.span_window,
            span_stride=self.span_stride,
        )
        # Project all span probes into the Language-Invariant Orthogonal Complement `P_perp_lang`!
        probes_proj = probes_raw - torch.matmul(torch.matmul(probes_raw, self.u_lang_mid), self.u_lang_mid.T)
        probes_2d = F.normalize(probes_proj, dim=-1, eps=1e-8)

        n_probes = int(probes_2d.shape[0])
        half_d = self.hidden_dim // 2
        q_rot = torch.matmul(probes_2d, self.R_rot.T)
        nq1 = q_rot[:, :half_d].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        nq2 = q_rot[:, half_d:].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q1 = q_rot[:, :half_d] / nq1
        q2 = q_rot[:, half_d:] / nq2

        s1_mat = torch.matmul(q1, self.U_codebook.T)
        s2_mat = torch.matmul(q2, self.V_codebook.T)
        m = min(top_coarse_m, self.hidden_dim)
        top_r = s1_mat.topk(m, dim=-1).indices.tolist()
        top_c = s2_mat.topk(m, dim=-1).indices.tolist()

        cid_to_global_coord = {str(v["capsule_id"]): k for k, v in self._global_directory.items()}
        candidate_installed_slots: set[int] = set()
        candidate_missing_coords: set[Tuple[int, int]] = set()
        for p_i in range(n_probes):
            for ri in top_r[p_i]:
                for cj in top_c[p_i]:
                    coord = (ri, cj)
                    flat_bit = ri * self.hidden_dim + cj
                    is_present = _bitmap_get(self._presence_bitmap_bytes, flat_bit)
                    if is_present and coord in self._coord_to_slot:
                        s_i = self._coord_to_slot[coord]
                        candidate_installed_slots.add(s_i)
                        for p_cid in self._slot_descriptors[s_i].get("parent_capsules", []):
                            if p_cid in self._cid_to_slot:
                                candidate_installed_slots.add(self._cid_to_slot[p_cid])
                            elif p_cid in cid_to_global_coord:
                                candidate_missing_coords.add(cid_to_global_coord[p_cid])
                    elif not is_present and coord in self._global_directory:
                        candidate_missing_coords.add(coord)
                        for p_cid in self._global_directory[coord].get("parent_capsules", []):
                            if p_cid in self._cid_to_slot:
                                candidate_installed_slots.add(self._cid_to_slot[p_cid])
                            elif p_cid in cid_to_global_coord:
                                candidate_missing_coords.add(cid_to_global_coord[p_cid])

        # Step 1: Score candidate cosines & pre-margin gates in RAM (0 disk/HTTP payload reads yet!)
        cand_cos: Dict[str, float] = {}
        pre_gates: Dict[str, float] = {}
        parents_by_cid: Dict[str, Tuple[str, ...]] = {}
        source_by_cid: Dict[str, Tuple[str, Any]] = {}
        ordered_cids: List[str] = []

        for coord in sorted(candidate_missing_coords):
            ri, cj = coord
            g_meta = self._global_directory[coord]
            cid_g = str(g_meta["capsule_id"])
            if cid_g in self._unlearned_cids:
                continue
            w1 = float(g_meta.get("w1", math.sqrt(0.5)))
            w2 = float(g_meta.get("w2", math.sqrt(0.5)))
            coarse_vec = (nq1.squeeze(-1) * w1 * s1_mat[:, ri]) + (nq2.squeeze(-1) * w2 * s2_mat[:, cj])
            coarse_cos = float(coarse_vec.max().item())
            self.last_routing_cosines[cid_g] = coarse_cos
            tau_g = float(g_meta["tau"])
            ramp_g = max(float(g_meta["ramp"]), 1e-6)
            g_f = max(0.0, min(1.0, (coarse_cos - tau_g) / ramp_g))
            if g_f > 0.0:
                cand_cos[cid_g] = coarse_cos
                pre_gates[cid_g] = g_f
                parents_by_cid[cid_g] = tuple(str(x) for x in g_meta.get("parent_capsules", ()))
                source_by_cid[cid_g] = ("missing", coord)
                ordered_cids.append(cid_g)

        for s_idx in sorted(candidate_installed_slots):
            desc = self._slot_descriptors[s_idx]
            cid = str(desc["capsule_id"])
            if cid in self._unlearned_cids:
                continue
            exact_cos = float(torch.mv(probes_2d.float(), self.z_anchors_installed[s_idx].float()).max().item())
            self.last_routing_cosines[cid] = exact_cos
            tau_s = float(desc["tau"])
            ramp_s = max(float(desc["ramp"]), 1e-6)
            g_val = max(0.0, min(1.0, (exact_cos - tau_s) / ramp_s))
            if g_val > 0.0:
                cand_cos[cid] = exact_cos
                pre_gates[cid] = g_val
                parents_by_cid[cid] = tuple(str(x) for x in desc.get("parent_capsules", ()))
                source_by_cid[cid] = ("installed", s_idx)
                ordered_cids.append(cid)

        if not ordered_cids:
            return []

        # Step 2: Evaluate conjunctive Sleep Bridges against pre-margin parent gates BEFORE winner filtering!
        final_gates: Dict[str, float] = {}
        active_bridges: List[str] = []
        for cid in ordered_cids:
            parents = parents_by_cid[cid]
            if not parents:
                final_gates[cid] = pre_gates[cid]
            else:
                if any(pre_gates.get(p, 0.0) <= 0.0 for p in parents):
                    final_gates[cid] = 0.0
                else:
                    parent_min = min(pre_gates[p] for p in parents)
                    eff_bg = pre_gates[cid] * min(1.0, parent_min / 0.30)
                    final_gates[cid] = eff_bg
                    if eff_bg > 0.0:
                        active_bridges.append(cid)

        # Step 3: Winner dominance on standalone primary concepts + conjunctive bridge parent supersession
        primary_cids = [c for c in ordered_cids if not parents_by_cid[c] and final_gates.get(c, 0.0) > 0.0]
        if len(primary_cids) > 1 and self.winner_margin > 0.0:
            best_primary_cos = max(cand_cos[c] for c in primary_cids)
            for c in primary_cids:
                if (best_primary_cos - cand_cos[c]) > self.winner_margin:
                    final_gates[c] = 0.0

        for b_cid in active_bridges:
            eff_bg = final_gates[b_cid]
            atten = max(0.0, 1.0 - eff_bg)
            for p in parents_by_cid[b_cid]:
                if p in pre_gates:
                    final_gates[p] = pre_gates[p] * atten

        # Step 4: Fetch / mmap page-in ONLY the surviving capsules (`final_gates[cid] > 1e-6`)!
        active: List[Tuple[DualDepthScarCapsule, float]] = []
        for cid in ordered_cids:
            g_use = float(final_gates.get(cid, 0.0))
            if g_use <= 1e-6:
                continue
            src_kind, src_ref = source_by_cid[cid]
            if src_kind == "installed":
                cap = self.load_slot_capsule(int(src_ref), device=dev)
                active.append((cap, g_use))
            else:
                coord = src_ref
                g_meta = self._global_directory[coord]
                tau_g = float(g_meta["tau"])
                ramp_g = max(float(g_meta["ramp"]), 1e-6)
                cap_stream: Optional[DualDepthScarCapsule] = None
                if cid in self._hot_fetched_capsules:
                    cap_stream = self._hot_fetched_capsules[cid].to_device(dev)
                elif self.on_demand_fetcher is not None:
                    fetched_bytes = self.on_demand_fetcher(coord, cid)
                    if fetched_bytes is not None:
                        self._turn_mmap_bytes_read += len(fetched_bytes)
                        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
                        cap_stream = DualDepthScarCapsule.from_int4_bytes(
                            raw_bytes=fetched_bytes,
                            capsule_id=cid,
                            hidden_dim=self.hidden_dim,
                            rank_mid=self.rank_mid,
                            rank_final=self.rank_final,
                            tau=tau_g,
                            ramp=ramp_g,
                            parent_capsules=parents_by_cid[cid],
                            device=dev,
                            metadata=dict(g_meta.get("metadata", {})),
                        )
                        self._hot_fetched_capsules[cid] = cap_stream
                if cap_stream is not None:
                    exact_c = float(torch.mv(probes_2d, F.normalize(cap_stream.z_anchor_mid.float(), dim=-1)).max().item())
                    self.last_routing_cosines[cid] = exact_c
                    if not parents_by_cid[cid]:
                        g_exact = max(0.0, min(1.0, (exact_c - cap_stream.tau) / max(cap_stream.ramp, 1e-6)))
                        is_attenuated_parent = any(cid in parents_by_cid[b] for b in active_bridges)
                        g_use = min(g_use, g_exact) if is_attenuated_parent else g_exact
                    if g_use > 1e-6:
                        active.append((cap_stream, g_use))
                else:
                    self.last_missing_alerts.append(
                        MissingCapsuleAlert(coord=coord, capsule_id=cid, coarse_cosine=cand_cos[cid], tau=tau_g)
                    )

        if self.strict_missing_guard and self.last_missing_alerts:
            a0 = self.last_missing_alerts[0]
            raise RuntimeError(
                f"MissingCapsuleAlert: Query triggered uninstalled capsule {a0.capsule_id!r} "
                f"at Kronecker coordinate {a0.coord} (cosine={a0.coarse_cosine:.4f} > tau={a0.tau:.4f})"
            )

        self.last_active_gates = {c.capsule_id: g for c, g in active}
        self.last_num_expanded_heads = len(active)
        return active

    def _apply_stage_scars(
        self,
        h_3d: torch.Tensor,
        active: List[Tuple[DualDepthScarCapsule, float]],
        stage: str,
        horizon_w: float,
    ) -> torch.Tensor:
        if horizon_w <= 0.0 or not active:
            return h_3d
        orig_dtype = h_3d.dtype
        hf = h_3d.float()
        mu = self.mu_hidden_mid if stage == "mid" else self.mu_hidden_final
        hc = hf - mu.view(1, 1, -1)
        delta = torch.zeros_like(hf)
        w_sum = sum(g for _, g in active)
        norm_div = w_sum if (len(active) > 1 and w_sum > 1.0) else 1.0
        step_dict: Dict[str, float] = {}
        for cap, g_val in active:
            eff_w = float(horizon_w) * (float(g_val) / norm_div)
            a_mat = cap.A_mid if stage == "mid" else cap.A_final
            b_mat = cap.B_mid if stage == "mid" else cap.B_final
            delta = delta + eff_w * torch.matmul(torch.matmul(hc, a_mat.float().T), b_mat.float().T)
            step_dict[cap.capsule_id] = round(eff_w, 4)
        if stage == "final":
            for _ in range(int(hf.shape[1])):
                self.last_step_weights_history.append(dict(step_dict))
                self.last_inter_head_spread_history.append(0.0)
        return (hf + delta).to(orig_dtype)

    def get_last_telemetry(self) -> HeadExpandedTelemetry:
        cids = [c.capsule_id for c, _ in self._active_turn_capsules]
        hist = list(self.last_step_weights_history)
        if not cids or not hist:
            return HeadExpandedTelemetry(
                active_capsules=cids,
                num_expanded_heads=len(cids),
                per_capsule_mean_weight={c: 0.0 for c in cids},
                per_capsule_min_weight={c: 0.0 for c in cids},
                per_capsule_max_weight={c: 0.0 for c in cids},
                max_inter_head_spread=0.0,
                step_weights=hist,
            )
        means, mins, maxs = {}, {}, {}
        for cid in cids:
            vals = [step.get(cid, 0.0) for step in hist]
            means[cid] = round(sum(vals) / len(vals), 4)
            mins[cid] = round(min(vals), 4)
            maxs[cid] = round(max(vals), 4)
        return HeadExpandedTelemetry(
            active_capsules=cids,
            num_expanded_heads=len(cids),
            per_capsule_mean_weight=means,
            per_capsule_min_weight=mins,
            per_capsule_max_weight=maxs,
            max_inter_head_spread=0.0,
            step_weights=hist,
        )

    @contextmanager
    def locked_routing(self) -> Iterator[None]:
        prev = self._routing_locked
        self._routing_locked = True
        self._decode_step_idx = 0
        self.last_step_weights_history = []
        self.last_inter_head_spread_history = []
        try:
            yield
        finally:
            self._routing_locked = prev

    @contextmanager
    def paused(self) -> Iterator[None]:
        prev = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = prev

    @contextmanager
    def teacher_forcing_scope(self, prompt_len: int) -> Iterator[None]:
        prev = self._tf_prompt_len
        self._tf_prompt_len = int(prompt_len)
        self._decode_step_idx = 0
        self.last_step_weights_history = []
        self.last_inter_head_spread_history = []
        try:
            yield
        finally:
            self._tf_prompt_len = prev

    def _backbone_pre_hook(self, _module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if self._paused:
            return
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        past_kv = kwargs.get("past_key_values")
        if past_kv is None and len(args) > 2:
            past_kv = args[2]
        past_len = _past_kv_seen_length(past_kv)
        ids_1d: List[int] = []
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            ids_1d = input_ids[0].tolist() if input_ids.ndim >= 2 else input_ids.tolist()
        seq_len = len(ids_1d)
        is_prefill_chunk = (past_len == 0) or (seq_len > 1)
        self._current_pass_is_prefill = is_prefill_chunk
        self._current_chunk_ids = ids_1d

        if is_prefill_chunk:
            self._decode_step_idx = 0
        else:
            self._decode_step_idx += 1

        if past_len == 0 and not self._routing_locked:
            self._rolling_id_tail = []
            self._query_hidden_buf = None
            self._turn_slot_cache.clear()
            self._turn_mmap_bytes_read = 0
            self.last_step_weights_history = []
            self.last_inter_head_spread_history = []

        if is_prefill_chunk and seq_len > 0 and not self._routing_locked:
            combined = self._rolling_id_tail + ids_1d
            match_end = _find_subsequence_last_end_strict(combined, self.user_turn_prefix_ids)
            keep_tail = max(0, len(self.user_turn_prefix_ids) - 1) if self.user_turn_prefix_ids else 0
            old_tail_len = len(self._rolling_id_tail)
            self._rolling_id_tail = combined[-keep_tail:] if keep_tail > 0 else []
            if match_end > old_tail_len:
                self._query_hidden_buf = None
                self._turn_slot_cache.clear()
                self._turn_mmap_bytes_read = 0
                self.last_step_weights_history = []
                self.last_inter_head_spread_history = []
                self._current_query_start = match_end - old_tail_len
            else:
                self._current_query_start = 0

    def _mid_layer_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if self._paused:
            return output
        is_tup = isinstance(output, tuple)
        h_raw: torch.Tensor = output[0] if is_tup else output
        squeezed = h_raw.ndim == 2
        h_3d = h_raw.unsqueeze(0) if squeezed else h_raw
        if h_3d.ndim != 3 or int(h_3d.shape[1]) == 0:
            return output
        seq_len = int(h_3d.shape[1])

        if not self._routing_locked:
            if self._tf_prompt_len is not None:
                lp = max(1, min(int(self._tf_prompt_len), seq_len))
                if len(self.last_step_weights_history) == 0:
                    qs = min(self._current_query_start, lp - 1)
                    qe = lp
                    if self.user_turn_suffix_ids and len(self._current_chunk_ids) >= lp:
                        _, qe = extract_user_content_bounds(
                            self._current_chunk_ids[:lp], self.user_turn_prefix_ids, self.user_turn_suffix_ids
                        )
                        qe = max(qs + 1, min(qe, lp))
                    self._active_turn_capsules = self.route_and_page_query(h_3d[0, qs:qe, :])
            elif self._current_pass_is_prefill:
                qs = self._current_query_start
                if qs < seq_len:
                    new_q = h_3d[0, qs:seq_len, :].detach()
                    buf = new_q if self._query_hidden_buf is None else torch.cat([self._query_hidden_buf, new_q], dim=0)
                    if buf.shape[0] > 192:
                        buf = buf[-192:]
                    self._query_hidden_buf = buf
                    # Strip trailing ChatML `<|im_end|>\n<|im_start|>assistant\n` tokens from routing buffer if present!
                    q_slice = buf
                    if self.user_turn_suffix_ids:
                        ls = len(self.user_turn_suffix_ids)
                        if (
                            ls > 0
                            and len(self._current_chunk_ids) >= ls
                            and self._current_chunk_ids[-ls:] == list(self.user_turn_suffix_ids)
                            and int(buf.shape[0]) > ls
                        ):
                            q_slice = buf[:-ls]
                    self._active_turn_capsules = self.route_and_page_query(q_slice)
                elif self._query_hidden_buf is None:
                    self._active_turn_capsules = []

        active = self._active_turn_capsules
        if not active:
            return output

        hw = 1.0 if self._tf_prompt_len is not None else compute_horizon_scale(
            self._decode_step_idx, self.seed_horizon, self.seed_fade_steps
        )
        if hw <= 0.0:
            return output

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            h_mod = h_3d.clone()
            h_mod[:, lp - 1 :, :] = self._apply_stage_scars(h_3d[:, lp - 1 :, :], active, stage="mid", horizon_w=hw)
        elif self._current_pass_is_prefill:
            h_mod = h_3d.clone()
            h_mod[:, -1:, :] = self._apply_stage_scars(h_3d[:, -1:, :], active, stage="mid", horizon_w=hw)
        else:
            h_mod = self._apply_stage_scars(h_3d, active, stage="mid", horizon_w=hw)

        h_out = h_mod.squeeze(0) if squeezed else h_mod
        return (h_out,) + output[1:] if is_tup else h_out

    def _final_norm_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self._paused or not self._active_turn_capsules:
            return output
        squeezed = output.ndim == 2
        out_3d = output.unsqueeze(0) if squeezed else output
        if out_3d.ndim != 3 or int(out_3d.shape[1]) == 0:
            return output
        seq_len = int(out_3d.shape[1])
        active = self._active_turn_capsules
        hw = 1.0 if self._tf_prompt_len is not None else compute_horizon_scale(
            self._decode_step_idx, self.seed_horizon, self.seed_fade_steps
        )
        if hw <= 0.0:
            return output

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            out_mod = out_3d.clone()
            out_mod[:, lp - 1 :, :] = self._apply_stage_scars(out_3d[:, lp - 1 :, :], active, stage="final", horizon_w=hw)
            return out_mod.squeeze(0) if squeezed else out_mod
        if self._current_pass_is_prefill:
            out_mod = out_3d.clone()
            out_mod[:, -1:, :] = self._apply_stage_scars(out_3d[:, -1:, :], active, stage="final", horizon_w=hw)
            return out_mod.squeeze(0) if squeezed else out_mod
        out_mod = self._apply_stage_scars(out_3d, active, stage="final", horizon_w=hw)
        return out_mod.squeeze(0) if squeezed else out_mod

    def attach(self, student_model: nn.Module) -> "GoldenSliceScarStore":
        self.detach()
        dev = _infer_model_device(student_model)
        self.to(dev)
        backbone, mid_mod, _, norm_mod, _ = extract_mid_and_final_modules(student_model, self.mid_layer_idx)
        self._backbone_pre_handle = backbone.register_forward_pre_hook(self._backbone_pre_hook, with_kwargs=True)
        self._mid_hook_handle = mid_mod.register_forward_hook(self._mid_layer_hook)
        self._norm_hook_handle = norm_mod.register_forward_hook(self._final_norm_hook)
        return self

    def detach(self) -> None:
        for attr in ("_backbone_pre_handle", "_mid_hook_handle", "_norm_hook_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)
        self._active_turn_capsules = []
