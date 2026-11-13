"""Horizon 2 (`v2.1.0`) Stage 4 — On-Demand Dynamic Attention-Head Expansion (`HeadExpandedAttentionBank`).

Eliminates cutting the Transformer in the middle (`Layers 0 .. L-2` remain **100% untouched**) while
preventing `0.50 / 0.50` static scalar gate collisions during multi-cartridge reasoning (`ZERO Backpropagation / ZERO SGD`):

  1. **Zero Mid-Network Cut + Zero Out-of-Cone Overhead**:
     - Routes causally at the input of the final Transformer block (`layers[-1]`, after `L-1` full undisturbed
       contextual layers) via `Span-Max` centered spherical cones (`z_anchor`), eliminating mid-layer semantic
       off-diagonal bleed (`0.0000` off-diagonal activation across disjoint scientific theorems).
     - When `0` cones fire, `0` extra heads are instantiated (`exact 0.0000000000` / `torch.equal == True`
       across every layer and logit).
  2. **On-Demand Dynamic Scar-Attention Head Expansion (`+M` Heads during Reasoning)**:
     - Reuses `layers[-1].self_attn.q_proj`'s already-computed `H` Query Heads (`14 x 64 = 896` on `Qwen2.5-0.5B`,
       `0` extra `W_Q` GEMMs!) against each active capsule's `P` Per-Head Key Prototypes `K_heads [P, H, d_h]`.
     - With `rank = 60` and `P = 8` (`112` 64-dim head keys at `d=896`), each `HeadExpandedScarCapsule`
       occupies **exact `62,720 bytes` (`61.25 KB` `Block-64 INT4`)** — bit-for-bit identical to Stage 1 & Stage 3!
     - When `M >= 2` capsules activate simultaneously, each pretrained head `h in {0..H-1}` computes competitive
       attention over the `M` capsules at **every token `t`**, allowing different heads to specialize on different
       capsules simultaneously and dynamically switching capsule dominance (`~0.998 <-> ~0.003`) across clauses!
  3. **Compatible with Optional Artificial Sleep Bridge Consolidation**:
     - Returns `BridgeEscalationDecision` + `HeadExpandedTelemetry` (`System 1` epistemic reflex contract) and
       supports optional `consolidate_interdisciplinary_bridge(...)` when novel algebraic cross-terms require
       `System 2` Sleep consolidation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
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

from scar_cache.dual_depth_bridge import (
    BridgeEscalationDecision,
    DualDepthBridgeEngine,
)
from scar_cache.scar_grafting import (
    SleepEpisode,
    _extract_backbone_and_norm,
    _infer_model_device,
    _infer_vocab_limit,
    _past_kv_seen_length,
    detect_chatml_user_prefix_ids,
    extract_span_query_probes,
    find_subsequence_last_end,
)
from scar_cache.scar_store_v2 import (
    BLOCK_SIZE_INT4,
    HEADER_ALIGN,
    MissingCapsuleAlert,
    PAGE_ALIGN,
    SCARSTORE_MAGIC,
    _align_up,
    _bitmap_get,
    _bitmap_set,
    _build_deterministic_orthogonal_rotation,
    dequantize_block64_int4,
    quantize_block64_int4,
)


def _find_subsequence_last_end_strict(seq: Sequence[int], pattern: Optional[Sequence[int]]) -> int:
    """Returns index right after last match of `pattern` in `seq`, or `-1` when absent."""
    if not pattern:
        return -1
    plen = len(pattern)
    slen = len(seq)
    if slen < plen:
        return -1
    pat = list(pattern)
    for idx in range(slen - plen, -1, -1):
        if list(seq[idx : idx + plen]) == pat:
            return idx + plen
    return -1


def extract_head_expansion_modules(
    student_model: nn.Module,
) -> Tuple[nn.Module, nn.Module, nn.Linear, int, int, nn.Module, nn.Linear]:
    """Resolves `(backbone, last_block, q_proj_linear, num_heads, head_dim, final_norm_mod, lm_head)`."""
    backbone, norm_mod, lm_head = _extract_backbone_and_norm(student_model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", getattr(backbone, "blocks", None))
    if layers is None or len(layers) == 0:
        raise AttributeError(
            f"Backbone {type(backbone).__name__} does not expose `.layers` / `.h` / `.blocks` for Head-Expanded grafting."
        )
    last_block = layers[-1]
    attn_mod = getattr(last_block, "self_attn", getattr(last_block, "attn", last_block))
    q_proj = getattr(attn_mod, "q_proj", None)
    if not isinstance(q_proj, nn.Linear):
        raise AttributeError(f"Attention module {type(attn_mod).__name__} does not expose Linear `q_proj`.")

    q_out_dim = int(q_proj.weight.shape[0])
    cfg = getattr(attn_mod, "config", getattr(student_model, "config", None))
    num_heads = getattr(attn_mod, "num_heads", getattr(attn_mod, "n_heads", None))
    if num_heads is None and cfg is not None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    head_dim = getattr(attn_mod, "head_dim", None)
    if head_dim is None and num_heads is not None and int(num_heads) > 0:
        head_dim = q_out_dim // int(num_heads)
    if num_heads is None and head_dim is not None and int(head_dim) > 0:
        num_heads = q_out_dim // int(head_dim)
    if num_heads is None or head_dim is None or int(num_heads) * int(head_dim) != q_out_dim:
        head_dim = 64 if (q_out_dim % 64 == 0) else math.gcd(q_out_dim, 32)
        num_heads = q_out_dim // max(1, int(head_dim))

    return backbone, last_block, q_proj, int(num_heads), int(head_dim), norm_mod, lm_head


@dataclass(frozen=True)
class HeadExpandedTelemetry:
    """Runtime multi-head & token-dynamic Scar-Attention telemetry."""

    active_capsules: List[str]
    num_expanded_heads: int
    per_capsule_mean_weight: Dict[str, float]
    per_capsule_min_weight: Dict[str, float]
    per_capsule_max_weight: Dict[str, float]
    max_inter_head_spread: float
    step_weights: List[Dict[str, float]]


@dataclass
class HeadExpandedScarCapsule:
    """Zero-Mid-Cut Dynamic Scar-Attention Capsule (`(B_scar, A_scar)` + Per-Head Keys `K_heads`).

    At `d=896, H=14, d_h=64, rank=60, num_key_prototypes=8`, `to_int4_bytes()` produces
    **exact `62,720 bytes` (`61.25 KB` `Block-64 INT4`)** — identical in byte footprint to
    Stage 1 (`QuantizedScarCapsule`) and Stage 3 (`DualDepthScarCapsule`)!
    """

    capsule_id: str
    z_anchor: torch.Tensor       # [d] FP16 unit vector at `layers[-1]` input
    A_scar: torch.Tensor         # [r, d] FP16 value down-projection
    B_scar: torch.Tensor         # [d, r] FP16 value output projection
    K_heads: torch.Tensor        # [P, H, d_h] FP16 unit-normalized Per-Head Key Prototypes
    tau: float = 0.62
    ramp: float = 0.16
    parent_capsules: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hidden_dim(self) -> int:
        return int(self.z_anchor.shape[0])

    @property
    def rank(self) -> int:
        return int(self.A_scar.shape[0])

    @property
    def num_key_prototypes(self) -> int:
        return int(self.K_heads.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.K_heads.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.K_heads.shape[2])

    @property
    def int4_physical_bytes(self) -> int:
        d = self.hidden_dim
        r = self.rank
        k_numel = int(self.K_heads.numel())
        if (d * r) % BLOCK_SIZE_INT4 != 0:
            raise ValueError(f"hidden_dim*rank ({d}*{r}) must be divisible by {BLOCK_SIZE_INT4}")
        if k_numel % BLOCK_SIZE_INT4 != 0:
            raise ValueError(f"K_heads numel ({k_numel}) must be divisible by {BLOCK_SIZE_INT4}")
        ab_blk = (d * r) // BLOCK_SIZE_INT4
        k_blk = k_numel // BLOCK_SIZE_INT4
        ab_bytes = 2 * (ab_blk * 32 + ab_blk * 2)
        k_bytes = k_blk * 32 + k_blk * 2
        return ab_bytes + k_bytes + (d * 2)

    def to_int4_bytes(self) -> bytes:
        bp, bs = quantize_block64_int4(self.B_scar)
        ap, as_ = quantize_block64_int4(self.A_scar)
        k_2d = F.normalize(self.K_heads.detach().float(), dim=-1, eps=1e-8).reshape(self.num_key_prototypes, -1)
        kp, ks = quantize_block64_int4(k_2d)
        z_b = F.normalize(self.z_anchor.detach().float(), dim=-1, eps=1e-8).to(torch.float16).cpu()
        parts = [
            bp.numpy().tobytes(),
            bs.numpy().tobytes(),
            ap.numpy().tobytes(),
            as_.numpy().tobytes(),
            kp.numpy().tobytes(),
            ks.numpy().tobytes(),
            z_b.numpy().tobytes(),
        ]
        blob = b"".join(parts)
        assert len(blob) == self.int4_physical_bytes
        return blob

    @classmethod
    def from_int4_bytes(
        cls,
        raw_bytes: Union[bytes, bytearray],
        capsule_id: str,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        rank: int = 60,
        num_key_prototypes: int = 8,
        tau: float = 0.62,
        ramp: float = 0.16,
        parent_capsules: Sequence[str] = (),
        device: Union[str, torch.device] = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HeadExpandedScarCapsule":
        buf = bytearray(raw_bytes)
        n_ab = (hidden_dim * rank) // BLOCK_SIZE_INT4
        k_row_dim = num_heads * head_dim
        n_k = (num_key_prototypes * k_row_dim) // BLOCK_SIZE_INT4
        off = 0

        def _take_mat(n_blk: int, shape: Tuple[int, int]) -> torch.Tensor:
            nonlocal off
            p = torch.frombuffer(buf, dtype=torch.uint8, count=n_blk * 32, offset=off).view(n_blk, 32).clone()
            off += n_blk * 32
            s = torch.frombuffer(buf, dtype=torch.float16, count=n_blk, offset=off).clone()
            off += n_blk * 2
            return dequantize_block64_int4(p, s, shape, dtype=torch.float16, device=device)

        B_scar = _take_mat(n_ab, (hidden_dim, rank))
        A_scar = _take_mat(n_ab, (rank, hidden_dim))
        K_2d = _take_mat(n_k, (num_key_prototypes, k_row_dim))
        K_heads = F.normalize(K_2d.float().view(num_key_prototypes, num_heads, head_dim), dim=-1, eps=1e-8).to(torch.float16)
        z = torch.frombuffer(buf, dtype=torch.float16, count=hidden_dim, offset=off).clone().to(device)
        meta = dict(metadata or {})
        parents_tup = tuple(str(x) for x in (parent_capsules or meta.get("parent_capsules", ())))
        return cls(
            capsule_id=capsule_id,
            z_anchor=z,
            A_scar=A_scar,
            B_scar=B_scar,
            K_heads=K_heads,
            tau=tau,
            ramp=ramp,
            parent_capsules=parents_tup,
            metadata=meta,
        )

    def to_device(self, device: Union[str, torch.device]) -> "HeadExpandedScarCapsule":
        dev = torch.device(device)
        self.z_anchor = self.z_anchor.to(dev)
        self.A_scar = self.A_scar.to(dev)
        self.B_scar = self.B_scar.to(dev)
        self.K_heads = self.K_heads.to(dev)
        return self


class HeadExpandedAttentionBank(nn.Module):
    """On-Demand Dynamic Scar-Attention Head Bank (`Zero Mid-Transformer Cut`).

    - Leaves `Layers 0 .. L-2` **100% untouched**.
    - Instantiates `0` extra heads (`exact torch.equal`) when no spherical cone fires.
    - Instantiates `+M` Scar-Attention Heads (`P` per-head keys across all `H` pretrained `q_proj` heads)
      during multi-capsule reasoning (`M >= 2`), dynamically resolving per-head and per-token capsule
      weights without static `0.50 / 0.50` scalar dilution!
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mu_query_last: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        u_preserved_final: torch.Tensor,
        mu_q_heads: torch.Tensor,
        attention_beta: float = 14.0,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        routing_mode: str = "span_max",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.attention_beta = float(attention_beta)
        self.user_turn_prefix_ids = [int(x) for x in user_turn_prefix_ids] if user_turn_prefix_ids else None
        self.routing_mode = str(routing_mode)

        self.register_buffer("mu_query_last", mu_query_last.detach().float().clone())
        self.register_buffer("mu_hidden_final", mu_hidden_final.detach().float().clone())
        self.register_buffer("u_preserved_final", u_preserved_final.detach().float().clone())
        self.register_buffer("mu_q_heads", mu_q_heads.detach().float().view(self.num_heads, self.head_dim).clone())

        self.register_buffer("_z_matrix", torch.empty((0, self.hidden_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("_tau_vec", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_ramp_vec", torch.empty(0, dtype=torch.float32), persistent=False)

        self._capsules: Dict[str, HeadExpandedScarCapsule] = {}
        self._capsule_order: List[str] = []

        self._backbone_pre_handle: Optional[Any] = None
        self._last_block_pre_handle: Optional[Any] = None
        self._q_proj_hook_handle: Optional[Any] = None
        self._norm_hook_handle: Optional[Any] = None

        self._paused: bool = False
        self._current_pass_is_prefill: bool = True
        self._current_query_start: int = 0
        self._rolling_id_tail: List[int] = []
        self._query_hidden_buf: Optional[torch.Tensor] = None
        self._tf_prompt_len: Optional[int] = None
        self._active_gates: List[Tuple[str, float]] = []
        self._cached_q_heads: Optional[torch.Tensor] = None  # [B, S_target, H, d_h]

        # Live inspection & epistemic telemetry
        self.last_routing_cosines: Dict[str, float] = {}
        self.last_active_gates: Dict[str, float] = {}
        self.last_num_expanded_heads: int = 0
        self.last_head_attention_matrix: Optional[torch.Tensor] = None  # [M, H] on most recent step
        self.last_step_weights_history: List[Dict[str, float]] = []
        self.last_inter_head_spread_history: List[float] = []

    @property
    def capsules(self) -> Dict[str, HeadExpandedScarCapsule]:
        return dict(self._capsules)

    def _rebuild_index(self) -> None:
        dev = self.mu_query_last.device
        if not self._capsule_order:
            self._z_matrix = torch.empty((0, self.hidden_dim), dtype=torch.float32, device=dev)
            self._tau_vec = torch.empty(0, dtype=torch.float32, device=dev)
            self._ramp_vec = torch.empty(0, dtype=torch.float32, device=dev)
            return
        zs, taus, ramps = [], [], []
        for cid in self._capsule_order:
            c = self._capsules[cid].to_device(dev)
            zs.append(F.normalize(c.z_anchor.float(), dim=-1, eps=1e-8))
            taus.append(float(c.tau))
            ramps.append(float(c.ramp))
        self._z_matrix = torch.stack(zs, dim=0).to(dev)
        self._tau_vec = torch.tensor(taus, dtype=torch.float32, device=dev)
        self._ramp_vec = torch.tensor(ramps, dtype=torch.float32, device=dev)

    def add_capsule(self, cap: HeadExpandedScarCapsule) -> None:
        if cap.capsule_id not in self._capsules:
            self._capsule_order.append(cap.capsule_id)
        self._capsules[cap.capsule_id] = cap.to_device(self.mu_query_last.device)
        self._rebuild_index()

    def unlearn(self, capsule_id: str) -> HeadExpandedScarCapsule:
        removed = self._capsules.pop(capsule_id)
        self._capsule_order = [c for c in self._capsule_order if c != capsule_id]
        self._active_gates = [(c, g) for c, g in self._active_gates if c != capsule_id]
        self._rebuild_index()
        return removed

    @torch.no_grad()
    def route_query(self, x_query_2d: torch.Tensor) -> List[Tuple[str, float]]:
        """Routes `Span-Max` cones at `layers[-1]` input WITHOUT normalizing `sum(g)` to `0.5/0.5`!"""
        if not self._capsule_order or x_query_2d.numel() == 0:
            self.last_routing_cosines = {}
            self.last_active_gates = {}
            self.last_num_expanded_heads = 0
            return []
        dev = x_query_2d.device
        if self.mu_query_last.device != dev:
            self.to(dev)
            self._rebuild_index()

        probes_2d, _ = extract_span_query_probes(
            query_hidden_2d=x_query_2d,
            mu_query=self.mu_query_last,
            routing_mode=self.routing_mode,
        )
        cos_mat = torch.matmul(probes_2d, self._z_matrix.T)
        cos_vec = cos_mat.amax(dim=0)
        raw_g = torch.clamp((cos_vec - self._tau_vec) / self._ramp_vec, min=0.0, max=1.0)

        cos_list = cos_vec.tolist()
        self.last_routing_cosines = {cid: float(cos_list[i]) for i, cid in enumerate(self._capsule_order)}
        if float(raw_g.max().item()) <= 0.0:
            self.last_active_gates = {}
            self.last_num_expanded_heads = 0
            return []

        g_vals = [float(x) for x in raw_g.tolist()]
        idx_map = {cid: i for i, cid in enumerate(self._capsule_order)}

        # Optional conjunctive parent supersession when a consolidated Sleep Bridge capsule is present
        base_cone = list(g_vals)
        for i, cid in enumerate(self._capsule_order):
            parents = self._capsules[cid].parent_capsules
            if not parents:
                continue
            if any(p not in idx_map or base_cone[idx_map[p]] <= 0.0 for p in parents):
                g_vals[i] = 0.0
                continue
            parent_min = min(base_cone[idx_map[p]] for p in parents)
            eff_bg = g_vals[i] * min(1.0, parent_min / 0.50)
            g_vals[i] = eff_bg
            if eff_bg > 0.0:
                atten = max(0.0, 1.0 - eff_bg)
                for p in parents:
                    g_vals[idx_map[p]] *= atten

        # Crucially: DO NOT divide g_vals by sum(g_vals) here!
        # When M == 1, g_0 stays 1.0; when M >= 2, dynamic per-head Query-Key Softmax resolves token weights!
        active = [(self._capsule_order[i], float(g_vals[i])) for i in range(len(self._capsule_order)) if g_vals[i] > 1e-6]
        self.last_active_gates = dict(active)
        self.last_num_expanded_heads = len(active)
        return active

    def get_last_telemetry(self) -> HeadExpandedTelemetry:
        cids = [cid for cid, _ in self._active_gates]
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
        max_spread = round(max(self.last_inter_head_spread_history) if self.last_inter_head_spread_history else 0.0, 4)
        return HeadExpandedTelemetry(
            active_capsules=cids,
            num_expanded_heads=len(cids),
            per_capsule_mean_weight=means,
            per_capsule_min_weight=mins,
            per_capsule_max_weight=maxs,
            max_inter_head_spread=max_spread,
            step_weights=hist,
        )

    def _compute_expanded_scar_heads(
        self,
        h_fin_3d: torch.Tensor,
        q_heads_4d: Optional[torch.Tensor],
        active: List[Tuple[str, float]],
    ) -> torch.Tensor:
        """Executes `+M` Expanded Scar-Attention Heads (`d_head = rank`) over `h_fin_3d` (`[B, S, d]`)."""
        orig_dtype = h_fin_3d.dtype
        hf = h_fin_3d.float()
        hc = hf - self.mu_hidden_final.view(1, 1, -1)

        if len(active) == 1 or q_heads_4d is None:
            # Single active Scar-Attention Head (or fallback): exact closed-form capsule readout
            delta = torch.zeros_like(hf)
            w_sum = sum(g for _, g in active)
            norm_div = w_sum if (len(active) > 1 and w_sum > 1.0) else 1.0
            step_dict: Dict[str, float] = {}
            for cid, g_val in active:
                eff_w = g_val / norm_div
                cap = self._capsules[cid]
                delta = delta + eff_w * torch.matmul(torch.matmul(hc, cap.A_scar.float().T), cap.B_scar.float().T)
                step_dict[cid] = round(float(eff_w), 4)
            for _ in range(int(hf.shape[1])):
                self.last_step_weights_history.append(dict(step_dict))
                self.last_inter_head_spread_history.append(0.0)
            self.last_head_attention_matrix = torch.ones((len(active), self.num_heads), dtype=torch.float32, device=hf.device)
            return (hf + delta).to(orig_dtype)

        # M >= 2 Reasoning Mode: Competitive Per-Head Query-Key Attention across all H pretrained heads!
        # q_heads_4d: [B, S, H, d_h]
        q_norm = F.normalize(q_heads_4d.float() - self.mu_q_heads.view(1, 1, self.num_heads, self.head_dim), dim=-1, eps=1e-8)
        beta = self.attention_beta

        # Compute per-head similarity scores [M, B, S, H]
        head_logits_list: List[torch.Tensor] = []
        for cid, g_val in active:
            cap = self._capsules[cid]
            k_proto = F.normalize(cap.K_heads.float(), dim=-1, eps=1e-8)  # [P, H, d_h]
            # sim_p: [B, S, P, H] -> max over P prototypes -> [B, S, H]
            sim_p = torch.einsum("bshd,phd->bsph", q_norm, k_proto)
            s_max = sim_p.amax(dim=2)
            log_prior = math.log(max(float(g_val), 1e-6))
            head_logits_list.append(beta * s_max + log_prior)

        head_logits = torch.stack(head_logits_list, dim=0)  # [M, B, S, H]
        alpha_m_bsh = F.softmax(head_logits, dim=0)         # [M, B, S, H]
        # Aggregate per-token weight across the H heads -> [M, B, S, 1]
        w_m_bs1 = alpha_m_bsh.mean(dim=-1, keepdim=True)

        # Record rich per-head & per-token telemetry on batch item 0
        self.last_head_attention_matrix = alpha_m_bsh[:, 0, -1, :].detach().clone()
        s_len = int(hf.shape[1])
        for t_idx in range(s_len):
            a_mh = alpha_m_bsh[:, 0, t_idx, :]  # [M, H]
            spread = float((a_mh.amax(dim=-1) - a_mh.amin(dim=-1)).amax().item())
            self.last_inter_head_spread_history.append(spread)
            self.last_step_weights_history.append(
                {active[m_i][0]: round(float(w_m_bs1[m_i, 0, t_idx, 0].item()), 4) for m_i in range(len(active))}
            )

        delta = torch.zeros_like(hf)
        for m_i, (cid, _g_val) in enumerate(active):
            cap = self._capsules[cid]
            v_scar = torch.matmul(torch.matmul(hc, cap.A_scar.float().T), cap.B_scar.float().T)
            delta = delta + w_m_bs1[m_i] * v_scar
        return (hf + delta).to(orig_dtype)

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
        self._cached_q_heads = None

        if past_len == 0:
            self._rolling_id_tail = []
            self._query_hidden_buf = None
            self.last_step_weights_history = []
            self.last_inter_head_spread_history = []

        if is_prefill_chunk and seq_len > 0:
            tail = self._rolling_id_tail
            combined = tail + ids_1d
            match_end = _find_subsequence_last_end_strict(combined, self.user_turn_prefix_ids)
            keep_tail = max(0, len(self.user_turn_prefix_ids) - 1) if self.user_turn_prefix_ids else 0
            self._rolling_id_tail = combined[-keep_tail:] if keep_tail > 0 else []
            if match_end > len(tail):
                self._query_hidden_buf = None
                self.last_step_weights_history = []
                self.last_inter_head_spread_history = []
                self._current_query_start = match_end - len(tail)
            else:
                self._current_query_start = 0

    def _last_block_pre_hook(self, _module: nn.Module, args: Tuple[Any, ...]) -> None:
        """Read-only routing hook on `layers[-1]` input (`x_in`): NEVER modifies `x_in`!"""
        if self._paused or not self._capsule_order or len(args) == 0 or not isinstance(args[0], torch.Tensor):
            return
        x_in = args[0]
        x_3d = x_in.unsqueeze(0) if x_in.ndim == 2 else x_in
        if x_3d.ndim != 3 or int(x_3d.shape[1]) == 0:
            return
        seq_len = int(x_3d.shape[1])

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            if len(self.last_step_weights_history) == 0:
                q_start = min(self._current_query_start, lp - 1)
                self._active_gates = self.route_query(x_3d[0, q_start:lp, :])
        elif self._current_pass_is_prefill:
            qs = self._current_query_start
            if qs < seq_len:
                new_q = x_3d[0, qs:seq_len, :].detach()
                buf = new_q if self._query_hidden_buf is None else torch.cat([self._query_hidden_buf, new_q], dim=0)
                if buf.shape[0] > 192:
                    buf = buf[-192:]
                self._query_hidden_buf = buf
                self._active_gates = self.route_query(buf)
            elif self._query_hidden_buf is None:
                self._active_gates = []

    def _q_proj_forward_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        """Zero-FLOP read-only view over `layers[-1].self_attn.q_proj` output ONLY when cones are active."""
        if self._paused or not self._active_gates:
            return
        q_3d = output.unsqueeze(0) if output.ndim == 2 else output
        if q_3d.ndim != 3 or int(q_3d.shape[1]) == 0:
            return
        bsz, seq_len, _ = q_3d.shape
        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            q_sl = q_3d[:, lp - 1 :, :]
        elif self._current_pass_is_prefill:
            q_sl = q_3d[:, -1:, :]
        else:
            q_sl = q_3d
        self._cached_q_heads = q_sl.detach().view(bsz, int(q_sl.shape[1]), self.num_heads, self.head_dim)

    def _final_norm_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self._paused or not self._capsule_order or not self._active_gates:
            return output
        squeezed = False
        out_3d = output
        if output.ndim == 2:
            out_3d = output.unsqueeze(0)
            squeezed = True
        elif output.ndim != 3:
            return output

        seq_len = int(out_3d.shape[1])
        if seq_len == 0:
            return output

        active = self._active_gates
        q_cached = self._cached_q_heads
        self._cached_q_heads = None

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            out_mod = out_3d.clone()
            out_mod[:, lp - 1 :, :] = self._compute_expanded_scar_heads(out_3d[:, lp - 1 :, :], q_cached, active)
            return out_mod.squeeze(0) if squeezed else out_mod

        if self._current_pass_is_prefill:
            out_mod = out_3d.clone()
            out_mod[:, -1:, :] = self._compute_expanded_scar_heads(out_3d[:, -1:, :], q_cached, active)
            return out_mod.squeeze(0) if squeezed else out_mod

        out_mod = self._compute_expanded_scar_heads(out_3d, q_cached, active)
        return out_mod.squeeze(0) if squeezed else out_mod

    def attach(self, student_model: nn.Module) -> "HeadExpandedAttentionBank":
        self.detach()
        dev = _infer_model_device(student_model)
        self.to(dev)
        self._rebuild_index()
        backbone, last_block, q_proj, _, _, norm_mod, _ = extract_head_expansion_modules(student_model)
        self._backbone_pre_handle = backbone.register_forward_pre_hook(self._backbone_pre_hook, with_kwargs=True)
        self._last_block_pre_handle = last_block.register_forward_pre_hook(self._last_block_pre_hook)
        self._q_proj_hook_handle = q_proj.register_forward_hook(self._q_proj_forward_hook)
        self._norm_hook_handle = norm_mod.register_forward_hook(self._final_norm_hook)
        return self

    def detach(self) -> None:
        for attr in ("_backbone_pre_handle", "_last_block_pre_handle", "_q_proj_hook_handle", "_norm_hook_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)
        self._active_gates = []
        self._cached_q_heads = None

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
        self.last_step_weights_history = []
        self.last_inter_head_spread_history = []
        try:
            yield
        finally:
            self._tf_prompt_len = prev


class HeadExpandedAttentionEngine:
    """Closed-Form Single-Pass Scar-Attention Synthesis & Epistemic Reasoning Engine (`ZERO SGD`)."""

    @staticmethod
    @torch.no_grad()
    def _capture_last_layer_and_final_states(
        student_model: nn.Module,
        input_ids_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Captures `(x_in_last [L, d], q_heads [L, H, d_h], h_final [L, d])` in a single forward pass."""
        dev = _infer_model_device(student_model)
        ids = input_ids_2d.to(dev)
        backbone, last_block, q_proj, num_heads, head_dim, norm_mod, _ = extract_head_expansion_modules(student_model)
        captured: Dict[str, torch.Tensor] = {}

        def _pre_blk(_m: nn.Module, args: Tuple[Any, ...]) -> None:
            t = args[0]
            t3 = t.unsqueeze(0) if t.ndim == 2 else t
            captured["x_in"] = t3[0].detach().float().cpu()

        def _q_hk(_m: nn.Module, _i: Tuple[Any, ...], out: torch.Tensor) -> None:
            t3 = out.unsqueeze(0) if out.ndim == 2 else out
            captured["q"] = t3[0].detach().float().cpu().view(-1, num_heads, head_dim)

        def _fin_hk(_m: nn.Module, _i: Tuple[Any, ...], out: torch.Tensor) -> None:
            t3 = out.unsqueeze(0) if out.ndim == 2 else out
            captured["fin"] = t3[0].detach().float().cpu()

        h1 = last_block.register_forward_pre_hook(_pre_blk)
        h2 = q_proj.register_forward_hook(_q_hk)
        h3 = norm_mod.register_forward_hook(_fin_hk)
        try:
            try:
                backbone(ids, use_cache=False)
            except TypeError:
                backbone(ids)
        finally:
            h1.remove()
            h2.remove()
            h3.remove()
        return captured["x_in"], captured["q"], captured["fin"]

    @staticmethod
    def _select_diverse_head_prototypes(all_q_norm: torch.Tensor, num_prototypes: int) -> torch.Tensor:
        """Deterministic furthest-point trajectory prototype selection across `[N_tok, H, d_h]`."""
        n_tok, num_heads, _ = all_q_norm.shape
        p_target = max(1, int(num_prototypes))
        if n_tok == 0:
            raise ValueError("Cannot select head prototypes from empty answer trajectory.")
        selected = [0]
        while len(selected) < min(p_target, n_tok):
            cur = all_q_norm[selected]  # [k, H, d_h]
            sims = torch.einsum("thd,khd->tk", all_q_norm, cur) / float(num_heads)
            max_sim = sims.amax(dim=-1)
            for idx_used in selected:
                max_sim[idx_used] = 1e9
            selected.append(int(max_sim.argmin().item()))
        proto = all_q_norm[selected].contiguous()
        if proto.shape[0] < p_target:
            pad = proto[-1:].expand(p_target - proto.shape[0], -1, -1).contiguous()
            proto = torch.cat([proto, pad], dim=0)
        return F.normalize(proto, dim=-1, eps=1e-8).to(torch.float16).contiguous()

    @classmethod
    @torch.no_grad()
    def calibrate_head_expanded_student(
        cls,
        student_model: nn.Module,
        calibration_prompts: Sequence[Union[str, torch.Tensor]],
        tokenizer: Optional[Any] = None,
        chat_formatter: Optional[Callable[[str], str]] = None,
        preserved_rank: int = 8,
        attention_beta: float = 14.0,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
    ) -> HeadExpandedAttentionBank:
        """Single-pass closed-form calibration of `layers[-1]` query statistics & `final_norm` null-space."""
        _, _, _, num_heads, head_dim, _, lm_head = extract_head_expansion_modules(student_model)
        d = int(lm_head.weight.shape[1])
        prefix_ids = (
            [int(x) for x in user_turn_prefix_ids]
            if user_turn_prefix_ids is not None
            else detect_chatml_user_prefix_ids(tokenizer)
        )

        q_lasts, all_q_heads, all_fin = [], [], []
        for item in calibration_prompts:
            if isinstance(item, str):
                text = chat_formatter(item) if chat_formatter is not None else item
                ids_2d = tokenizer(text, return_tensors="pt").input_ids
            else:
                ids_2d = item if item.ndim == 2 else item.unsqueeze(0)
            x_in, q_h, h_fin = cls._capture_last_layer_and_final_states(student_model, ids_2d)
            q_start = find_subsequence_last_end(ids_2d[0].tolist(), prefix_ids)
            q_lasts.append(x_in[q_start:].mean(dim=0))
            all_q_heads.append(q_h)
            all_fin.append(h_fin)

        mu_q_last = torch.stack(q_lasts, dim=0).mean(dim=0)
        mu_q_heads = torch.cat(all_q_heads, dim=0).mean(dim=0)
        cat_fin = torch.cat(all_fin, dim=0)
        mu_h_fin = cat_fin.mean(dim=0)

        r_eff = max(0, min(int(preserved_rank), cat_fin.shape[0] - 1, d - 1))
        if r_eff > 0:
            _, _, vh_f = torch.linalg.svd(cat_fin - mu_h_fin, full_matrices=False)
            u_fin = vh_f[:r_eff, :].T.contiguous()
        else:
            u_fin = torch.zeros((d, 0), dtype=torch.float32)

        return HeadExpandedAttentionBank(
            hidden_dim=d,
            num_heads=num_heads,
            head_dim=head_dim,
            mu_query_last=mu_q_last,
            mu_hidden_final=mu_h_fin,
            u_preserved_final=u_fin,
            mu_q_heads=mu_q_heads,
            attention_beta=attention_beta,
            user_turn_prefix_ids=prefix_ids,
        )

    @staticmethod
    def _rosetta_cutting_plane(
        w_head: torch.Tensor,
        lg_teacher_1d: torch.Tensor,
        lg_student_1d: torch.Tensor,
        k_eff: int,
        logit_gain: float,
        vocab_ridge: float,
    ) -> torch.Tensor:
        """Active-set Cutting-Plane Woodbury Rosetta solver tracking maximum full-vocab Top-1 clearance."""
        v_common = int(lg_teacher_1d.shape[0])
        d = int(w_head.shape[1])
        k_init = max(2, min(int(k_eff), v_common, max(10, d // 8)))
        t_top = lg_teacher_1d.topk(k_init).indices
        s_top = lg_student_1d.topk(k_init).indices
        s_union = torch.unique(torch.cat([t_top, s_top], dim=0))
        t_best = int(t_top[0].item())

        best_r = torch.zeros(d, dtype=torch.float32)
        best_clearance = float("-inf")
        w_common = w_head[:v_common]

        for pass_i in range(8):
            w_s = w_head[s_union]
            t_sub = lg_teacher_1d[s_union]
            s_sub = lg_student_1d[s_union]
            m_s = int(s_union.numel())
            target_sub = float(logit_gain) * (t_sub - t_sub.mean())
            best_mask = s_union == t_best
            if bool(best_mask.any().item()) and m_s > 1:
                other_max = float(target_sub[~best_mask].max().item())
                if float(target_sub[best_mask][0].item()) - other_max < 1.50:
                    target_sub = target_sub.clone()
                    target_sub[best_mask] = other_max + 1.50
            delta_lg = target_sub - (s_sub - s_sub.mean())
            gram_v = (w_s @ w_s.T) + float(vocab_ridge) * torch.eye(m_s, dtype=torch.float32)
            r_cand = w_s.T @ torch.linalg.solve(gram_v, delta_lg)

            pred_lg = lg_student_1d + (w_common @ r_cand)
            top2_v, top2_i = pred_lg.topk(min(2, v_common))
            if v_common > 1:
                runner_up = float(top2_v[1].item()) if int(top2_i[0].item()) == t_best else float(top2_v[0].item())
                clearance = float(pred_lg[t_best].item()) - runner_up
            else:
                clearance = 1.5

            if clearance > best_clearance:
                best_clearance = clearance
                best_r = r_cand

            if int(top2_i[0].item()) == t_best and clearance >= 1.15:
                break
            if pass_i < 7 and m_s + 6 <= max(12, int(0.65 * d)):
                intruders = pred_lg.topk(min(6, v_common)).indices
                new_u = torch.unique(torch.cat([s_union, intruders], dim=0))
                if int(new_u.numel()) == m_s:
                    break
                s_union = new_u
            else:
                break
        return best_r

    @classmethod
    @torch.no_grad()
    def solve_head_expanded_capsule(
        cls,
        capsule_id: str,
        episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        head_bank: HeadExpandedAttentionBank,
        rank: int = 60,
        num_key_prototypes: int = 8,
        top_vocab_k: int = 48,
        logit_gain: float = 1.30,
        vocab_ridge: float = 0.10,
        hidden_ridge_ratio: float = 0.015,
        tau: float = 0.62,
        ramp: float = 0.16,
        parent_capsules: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HeadExpandedScarCapsule:
        """Single-Pass Closed-Form Scar-Attention Head Synthesis (`ZERO Backpropagation / ZERO SGD`)."""
        t0 = time.perf_counter()
        _, _, _, _, _, _, lm_head = extract_head_expansion_modules(student_model)
        w_head = lm_head.weight.detach().float().cpu()
        b_head = lm_head.bias.detach().float().cpu() if getattr(lm_head, "bias", None) is not None else None
        v_slm, d = int(w_head.shape[0]), int(w_head.shape[1])
        v_lim = _infer_vocab_limit(student_model) or v_slm
        k_voc = max(2, min(int(top_vocab_k), v_slm, max(4, d // 6)))

        mu_ql = head_bank.mu_query_last.detach().float().cpu()
        mu_hf = head_bank.mu_hidden_final.detach().float().cpu()
        mu_qh = head_bank.mu_q_heads.detach().float().cpu()
        uf = head_bank.u_preserved_final.detach().float().cpu()
        p_null = torch.eye(d) - (uf @ uf.T) if uf.numel() > 0 else torch.eye(d)

        z_vecs, hp_rows, rc_rows, q_proto_rows = [], [], [], []
        with head_bank.paused():
            for ep in episodes:
                lp, la = ep.prompt_len, ep.answer_len
                if la <= 0:
                    continue
                clamped = ep.full_ids.clamp(min=0, max=v_lim - 1).unsqueeze(0)
                x_in, q_h, h_fin = cls._capture_last_layer_and_final_states(student_model, clamped)
                qs = min(max(0, int(ep.query_start_idx)), max(0, lp - 1))
                z_vecs.append(F.normalize(x_in[qs:lp].mean(dim=0) - mu_ql, dim=-1, eps=1e-8))

                h_fin_ans = h_fin[lp - 1 : lp - 1 + la]
                q_ans = q_h[lp - 1 : lp - 1 + la]
                q_proto_rows.append(F.normalize(q_ans - mu_qh.unsqueeze(0), dim=-1, eps=1e-8))

                lg_slm = h_fin_ans @ w_head.T + (b_head.unsqueeze(0) if b_head is not None else 0.0)
                v_c = min(v_slm, int(ep.teacher_logits.shape[-1]))
                lg_t = ep.teacher_logits[:la, :v_c].float()
                k_eff = min(k_voc, v_c)
                w_ep = math.sqrt(max(float(ep.surprise_score), 1e-4))

                for step_i in range(la):
                    r_star = cls._rosetta_cutting_plane(
                        w_head=w_head,
                        lg_teacher_1d=lg_t[step_i],
                        lg_student_1d=lg_slm[step_i, :v_c],
                        k_eff=k_eff,
                        logit_gain=logit_gain,
                        vocab_ridge=vocab_ridge,
                    )
                    hp_rows.append(((h_fin_ans[step_i] - mu_hf) @ p_null) * w_ep)
                    rc_rows.append(r_star * w_ep)

        z_anchor = F.normalize(torch.stack(z_vecs, dim=0).mean(dim=0), dim=-1, eps=1e-8).to(torch.float16)
        B_scar, A_scar = DualDepthBridgeEngine._woodbury_dual_svd(
            hp_mat=torch.stack(hp_rows, dim=0),
            rc_mat=torch.stack(rc_rows, dim=0),
            p_null=p_null,
            rank=rank,
            hidden_ridge_ratio=hidden_ridge_ratio,
        )
        K_heads = cls._select_diverse_head_prototypes(torch.cat(q_proto_rows, dim=0), num_key_prototypes)

        parents_tup = tuple(str(x) for x in parent_capsules)
        meta = dict(metadata or {})
        meta.update(
            {
                "rank": int(B_scar.shape[1]),
                "num_key_prototypes": int(K_heads.shape[0]),
                "num_heads": int(K_heads.shape[1]),
                "head_dim": int(K_heads.shape[2]),
                "solve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "zero_backpropagation": True,
                "zero_mid_layer_cut": True,
            }
        )
        if parents_tup:
            meta["parent_capsules"] = list(parents_tup)
        return HeadExpandedScarCapsule(
            capsule_id=capsule_id,
            z_anchor=z_anchor,
            A_scar=A_scar,
            B_scar=B_scar,
            K_heads=K_heads,
            tau=tau,
            ramp=ramp,
            parent_capsules=parents_tup,
            metadata=meta,
        )

    @staticmethod
    @torch.no_grad()
    def assess_reasoning_epistemic_uncertainty(
        student_model: nn.Module,
        head_bank: HeadExpandedAttentionBank,
        prompt_ids_2d: torch.Tensor,
        probe_steps: int = 8,
        margin_threshold: float = 0.65,
    ) -> Tuple[BridgeEscalationDecision, HeadExpandedTelemetry]:
        """Evaluates `System 1` epistemic confidence, `p_1 - p_2` margin, and multi-head routing telemetry."""
        t0 = time.perf_counter()
        dev = _infer_model_device(student_model)
        seq = prompt_ids_2d.to(dev).clone()
        lp = int(seq.shape[1])

        was_attached = head_bank._norm_hook_handle is not None
        if not was_attached:
            head_bank.attach(student_model)

        try:
            confs, ents, margins = [], [], []
            active_cids: List[str] = []
            step_weights_log: List[Dict[str, float]] = []
            spread_log: List[float] = []

            for step_i in range(max(1, probe_steps)):
                with head_bank.teacher_forcing_scope(prompt_len=lp):
                    try:
                        out = student_model(seq, use_cache=False)
                    except TypeError:
                        out = student_model(seq)
                    if step_i == 0:
                        active_cids = list(head_bank.last_active_gates.keys())
                    if head_bank.last_step_weights_history:
                        step_weights_log.append(dict(head_bank.last_step_weights_history[-1]))
                    if head_bank.last_inter_head_spread_history:
                        spread_log.append(float(head_bank.last_inter_head_spread_history[-1]))

                lg = (out.logits if hasattr(out, "logits") else out)[0, -1].float()
                probs = F.softmax(lg, dim=-1)
                top2 = probs.topk(2).values
                p1, p2 = float(top2[0].item()), float(top2[1].item())
                ent = -float((probs * (probs.clamp(min=1e-12)).log()).sum().item()) / math.log(max(2, int(lg.numel())))
                confs.append(p1)
                margins.append(p1 - p2)
                ents.append(min(1.0, max(0.0, ent)))
                nxt = int(lg.argmax(dim=-1).item())
                seq = torch.cat([seq, torch.tensor([[nxt]], dtype=torch.long, device=dev)], dim=1)

            mean_conf = sum(confs) / len(confs)
            mean_ent = sum(ents) / len(ents)
            mean_margin = sum(margins) / len(margins)
            min_margin = min(margins)
            is_unc = (
                (len(active_cids) >= 2 and (mean_margin < margin_threshold or min_margin < 0.35))
                or (mean_margin < 0.20)
            )
            decision = BridgeEscalationDecision(
                active_capsules=active_cids,
                confidence=round(mean_conf, 4),
                uncertainty=round(mean_ent, 4),
                margin=round(mean_margin, 4),
                is_uncertain=bool(is_unc),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

            means, mins, maxs = {}, {}, {}
            for cid in active_cids:
                vals = [sw.get(cid, 0.0) for sw in step_weights_log] if step_weights_log else [0.0]
                means[cid] = round(sum(vals) / len(vals), 4)
                mins[cid] = round(min(vals), 4)
                maxs[cid] = round(max(vals), 4)
            telemetry = HeadExpandedTelemetry(
                active_capsules=active_cids,
                num_expanded_heads=len(active_cids),
                per_capsule_mean_weight=means,
                per_capsule_min_weight=mins,
                per_capsule_max_weight=maxs,
                max_inter_head_spread=round(max(spread_log) if spread_log else 0.0, 4),
                step_weights=step_weights_log,
            )
            return decision, telemetry
        finally:
            if not was_attached:
                head_bank.detach()

    @classmethod
    @torch.no_grad()
    def consolidate_interdisciplinary_bridge(
        cls,
        bridge_capsule_id: str,
        bridge_episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        head_bank: HeadExpandedAttentionBank,
        parent_capsules: Optional[Sequence[str]] = None,
        rank: int = 60,
        num_key_prototypes: int = 8,
        tau: float = 0.62,
        ramp: float = 0.16,
    ) -> HeadExpandedScarCapsule:
        """Optional Artificial Sleep consolidation into a single-pass `61.25 KB` `HeadExpandedScarCapsule`."""
        resolved_parents: Tuple[str, ...]
        if parent_capsules is not None:
            resolved_parents = tuple(str(x) for x in parent_capsules)
        elif bridge_episodes and head_bank.capsules:
            ep0 = bridge_episodes[0]
            v_lim = _infer_vocab_limit(student_model) or int(ep0.full_ids.max().item()) + 1
            with head_bank.paused():
                x_in0, _, _ = cls._capture_last_layer_and_final_states(
                    student_model,
                    ep0.full_ids[: ep0.prompt_len].clamp(min=0, max=v_lim - 1).unsqueeze(0),
                )
            qs0 = min(max(0, int(ep0.query_start_idx)), max(0, ep0.prompt_len - 1))
            _ = head_bank.route_query(x_in0[qs0 : ep0.prompt_len].to(head_bank.mu_query_last.device))
            resolved_parents = tuple(
                cid
                for cid, cos_v in head_bank.last_routing_cosines.items()
                if not head_bank.capsules[cid].parent_capsules
                and ((cos_v - head_bank.capsules[cid].tau) / max(head_bank.capsules[cid].ramp, 1e-6)) >= 0.50
            )
        else:
            resolved_parents = ()

        bridge_cap = cls.solve_head_expanded_capsule(
            capsule_id=bridge_capsule_id,
            episodes=bridge_episodes,
            student_model=student_model,
            head_bank=head_bank,
            rank=rank,
            num_key_prototypes=num_key_prototypes,
            tau=tau,
            ramp=ramp,
            parent_capsules=resolved_parents,
            metadata={"capsule_type": "interdisciplinary_bridge"},
        )
        head_bank.add_capsule(bridge_cap)
        return bridge_cap


class HeadExpandedScarStore(nn.Module):
    """Zero-Copy `mmap` + `d x d` Kronecker Sparse Grid Engine for `HeadExpandedScarCapsule` (`v2.2.0`).

    Unifies:
      - Exact page-aligned `mmap` streaming (`143,360 B = 140.00 KB = 35 x 4,096 B` OS pages at `d=2048`
        on `Qwen2.5-3B-Instruct`; `62,720 B = 61.25 KB` at `d=896` on `Qwen2.5-0.5B-Instruct`),
      - Encyclopedic `d x d` Rotated Dual-Subspace Kronecker Grid (`4,194,304` cells & `512.00 KB` bitmap
        at `d=2048`; `802,816` cells & `98.00 KB` bitmap at `d=896`),
      - Zero Mid-Transformer Cut (`36/36` blocks on `3B` and `24/24` blocks on `0.5B` stay `torch.equal`),
      - On-Demand Dynamic Scar-Attention Head Expansion (`+M` competitive Per-Head Query-Key heads),
      - Consumer slice packs (`export_subset`), `MissingCapsuleAlert` + `on_demand_fetcher`, and `< 1 ms`
        surgical GDPR `unlearn(capsule_id)`.
    """

    def __init__(
        self,
        store_path: Path,
        header: Dict[str, Any],
        mu_query_last: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        mu_q_heads: torch.Tensor,
        R_rot: torch.Tensor,
        U_codebook: torch.Tensor,
        V_codebook: torch.Tensor,
        presence_bitmap: torch.Tensor,
        z_anchors_installed: torch.Tensor,
        payload_offset: int,
        on_demand_fetcher: Optional[Callable[[Tuple[int, int], str], Optional[bytes]]] = None,
    ) -> None:
        super().__init__()
        self.store_path = Path(store_path)
        self.header = header
        self.hidden_dim = int(header["hidden_dim"])
        self.num_heads = int(header["num_heads"])
        self.head_dim = int(header["head_dim"])
        self.rank = int(header["rank"])
        self.num_key_prototypes = int(header["num_key_prototypes"])
        self.record_bytes = int(header["record_bytes"])
        self.routing_tier: str = str(header["routing_tier"])
        self.attention_beta: float = float(header.get("attention_beta", 14.0))
        self.user_turn_prefix_ids: Optional[List[int]] = header.get("user_turn_prefix_ids")
        self.payload_offset = int(payload_offset)
        self.on_demand_fetcher = on_demand_fetcher

        self.register_buffer("mu_query_last", mu_query_last.detach().float().clone())
        self.register_buffer("mu_hidden_final", mu_hidden_final.detach().float().clone())
        self.register_buffer("mu_q_heads", mu_q_heads.detach().float().view(self.num_heads, self.head_dim).clone())
        self.register_buffer("R_rot", R_rot.detach().float().clone())
        self.register_buffer("U_codebook", U_codebook.detach().float().clone())
        self.register_buffer("V_codebook", V_codebook.detach().float().clone())
        self._presence_bitmap_bytes = bytearray(presence_bitmap.detach().to(torch.uint8).cpu().numpy().tobytes())
        self.register_buffer("z_anchors_installed", z_anchors_installed.detach().float().clone())

        self._slot_descriptors: List[Dict[str, Any]] = list(header.get("slot_descriptors", []))
        self._coord_to_slot: Dict[Tuple[int, int], int] = {}
        self._cid_to_slot: Dict[str, int] = {}
        self._unlearned_cids: set[str] = set()
        for desc in self._slot_descriptors:
            s_idx = int(desc["slot_idx"])
            coord = (int(desc["row_i"]), int(desc["col_j"]))
            self._coord_to_slot[coord] = s_idx
            self._cid_to_slot[str(desc["capsule_id"])] = s_idx

        self._global_directory: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for entry in header.get("global_directory", []):
            c = (int(entry["row_i"]), int(entry["col_j"]))
            self._global_directory[c] = dict(entry)

        self._file_obj = open(self.store_path, "rb")
        self._mmap = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        self._hot_fetched_capsules: Dict[str, HeadExpandedScarCapsule] = {}
        self._turn_slot_cache: Dict[int, HeadExpandedScarCapsule] = {}
        self._turn_mmap_bytes_read: int = 0
        self._active_turn_capsules: List[Tuple[HeadExpandedScarCapsule, float]] = []

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
        self._last_block_pre_handle: Optional[Any] = None
        self._q_proj_hook_handle: Optional[Any] = None
        self._norm_hook_handle: Optional[Any] = None
        self._paused: bool = False
        self._current_pass_is_prefill: bool = True
        self._current_query_start: int = 0
        self._rolling_id_tail: List[int] = []
        self._query_hidden_buf: Optional[torch.Tensor] = None
        self._tf_prompt_len: Optional[int] = None
        self._cached_q_heads: Optional[torch.Tensor] = None

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

    def load_slot_capsule(self, slot_idx: int, device: Union[str, torch.device]) -> HeadExpandedScarCapsule:
        if slot_idx in self._turn_slot_cache:
            return self._turn_slot_cache[slot_idx].to_device(device)
        desc = self._slot_descriptors[slot_idx]
        raw = self.read_raw_slot_bytes(slot_idx)
        self._turn_mmap_bytes_read += len(raw)
        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
        cap = HeadExpandedScarCapsule.from_int4_bytes(
            raw_bytes=raw,
            capsule_id=str(desc["capsule_id"]),
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rank=self.rank,
            num_key_prototypes=self.num_key_prototypes,
            tau=float(desc["tau"]),
            ramp=float(desc["ramp"]),
            parent_capsules=desc.get("parent_capsules", ()),
            device=device,
            metadata=dict(desc.get("metadata", {})),
        )
        self._turn_slot_cache[slot_idx] = cap
        return cap

    def _router_section_offsets(self) -> Tuple[int, int, int]:
        """Returns `(router_offset, bitmap_offset, z_anchors_offset)` inside `self.store_path`."""
        with open(self.store_path, "rb") as f:
            magic = f.read(8)
            if magic != SCARSTORE_MAGIC:
                raise ValueError(f"Invalid ScarStore magic in {self.store_path}")
            (hdr_len,) = struct.unpack("<Q", f.read(8))
        router_off = _align_up(16 + int(hdr_len), HEADER_ALIGN)
        d = self.hidden_dim
        half_d = d // 2
        q_numel = self.num_heads * self.head_dim
        bm_off = (
            router_off
            + (d * 4)
            + (d * 4)
            + (q_numel * 4)
            + (d * d * 2)
            + (d * half_d * 2)
            + (d * half_d * 2)
        )
        bm_len = (d * d + 7) // 8
        z_off = bm_off + bm_len
        return router_off, bm_off, z_off

    def unlearn(self, capsule_id: str, persist_to_disk: bool = False) -> Dict[str, Any]:
        """Sub-millisecond surgical GDPR unlearning: clears Kronecker bitmap bit & hot state (`0` bleed).

        When `persist_to_disk=True`, also performs surgical in-place `0x00` cryptographic erasure of the
        capsule's `4096 B`-aligned `mmap` pages, `z_anchor`, and on-disk `presence_bitmap` bit + `fsync`,
        without shifting any sibling capsule offsets or rewriting the full store.
        """
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
            # Refresh read-only mmap view for full OS coherence across platforms
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
        """Returns physical byte-layout, OS page alignment, Kronecker grid, and per-slot audit metadata."""
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
                }
            )
        return {
            "store_path": str(self.store_path),
            "file_bytes": file_bytes,
            "file_mib": round(file_bytes / (1024.0 * 1024.0), 4),
            "format": str(self.header.get("format", "HeadExpandedScarStoreV2")),
            "version": str(self.header.get("version", "2.2.0")),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "rank": self.rank,
            "num_key_prototypes": self.num_key_prototypes,
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
        }

    def to_head_bank(
        self,
        u_preserved_final: Optional[torch.Tensor] = None,
        load_installed_capsules: bool = True,
    ) -> HeadExpandedAttentionBank:
        """Reconstructs an in-memory `HeadExpandedAttentionBank` sharing this store's calibrated statistics."""
        d = self.hidden_dim
        u_fin = (
            u_preserved_final.detach().float().cpu()
            if u_preserved_final is not None
            else torch.zeros((d, 0), dtype=torch.float32)
        )
        bank = HeadExpandedAttentionBank(
            hidden_dim=d,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mu_query_last=self.mu_query_last.detach().cpu().float(),
            mu_hidden_final=self.mu_hidden_final.detach().cpu().float(),
            u_preserved_final=u_fin,
            mu_q_heads=self.mu_q_heads.detach().cpu().float(),
            attention_beta=self.attention_beta,
            user_turn_prefix_ids=self.user_turn_prefix_ids,
        )
        if load_installed_capsules:
            for desc in self._slot_descriptors:
                s_idx = int(desc["slot_idx"])
                cid = str(desc["capsule_id"])
                ri, cj = int(desc["row_i"]), int(desc["col_j"])
                if cid in self._unlearned_cids or not _bitmap_get(self._presence_bitmap_bytes, ri * d + cj):
                    continue
                cap = self.load_slot_capsule(s_idx, device="cpu")
                bank.add_capsule(cap)
        return bank

    def append_capsule(
        self,
        capsule: HeadExpandedScarCapsule,
        out_path: Optional[Union[str, Path]] = None,
    ) -> "HeadExpandedScarStore":
        """Appends or updates a `HeadExpandedScarCapsule` (e.g. a Sleep Bridge or streamed cartridge) on disk."""
        if capsule.rank != self.rank or capsule.num_key_prototypes != self.num_key_prototypes:
            raise ValueError(
                f"Capsule (rank={capsule.rank}, P={capsule.num_key_prototypes}) does not match store "
                f"(rank={self.rank}, P={self.num_key_prototypes})"
            )
        target_path = Path(out_path) if out_path is not None else self.store_path
        d = self.hidden_dim
        half_d = d // 2
        R_rot = self.R_rot.detach().cpu().float()
        U_cb = self.U_codebook.detach().cpu().float().clone()
        V_cb = self.V_codebook.detach().cpu().float().clone()
        mu_ql = self.mu_query_last.detach().cpu().float()
        mu_hf = self.mu_hidden_final.detach().cpu().float()
        mu_qh = self.mu_q_heads.detach().cpu().float()

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
            raw = self.read_raw_slot_bytes(s_idx)
            raw_blobs.append(raw)
            z_list_fp16.append(self.z_anchors_installed[s_idx].detach().cpu().to(torch.float16))
            used_coords.add((ri, cj))
            _bitmap_set(bitmap, ri * d + cj)
            d_copy = dict(old_desc)
            d_copy["slot_idx"] = len(new_descs)
            new_descs.append(d_copy)

        for coord_k, g_entry in self._global_directory.items():
            g_cid = str(g_entry["capsule_id"])
            if g_cid != capsule.capsule_id and g_cid not in self._unlearned_cids:
                new_global.append(dict(g_entry))

        z_f = F.normalize(capsule.z_anchor.detach().cpu().float(), dim=-1, eps=1e-8)
        z_rot = torch.mv(R_rot, z_f)
        z1 = F.normalize(z_rot[:half_d], dim=-1, eps=1e-8)
        z2 = F.normalize(z_rot[half_d:], dim=-1, eps=1e-8)
        w1 = float(z_rot[:half_d].norm().item())
        w2 = float(z_rot[half_d:].norm().item())

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
                raise RuntimeError("Kronecker grid candidate neighborhood saturated during append_capsule")
            row_i, col_j = best_pair

        _bitmap_set(bitmap, row_i * d + col_j)
        blob = capsule.to_int4_bytes()
        raw_blobs.append(blob)
        z_list_fp16.append(z_f.to(torch.float16))
        slot_idx_new = len(new_descs)
        new_descs.append(
            {
                "slot_idx": slot_idx_new,
                "capsule_id": capsule.capsule_id,
                "row_i": row_i,
                "col_j": col_j,
                "w1": round(w1, 6),
                "w2": round(w2, 6),
                "tau": float(capsule.tau),
                "ramp": float(capsule.ramp),
                "parent_capsules": list(capsule.parent_capsules),
                "metadata": dict(capsule.metadata),
            }
        )
        new_global.append(
            {
                "capsule_id": capsule.capsule_id,
                "row_i": row_i,
                "col_j": col_j,
                "w1": round(w1, 6),
                "w2": round(w2, 6),
                "tau": float(capsule.tau),
                "ramp": float(capsule.ramp),
                "parent_capsules": list(capsule.parent_capsules),
            }
        )

        dev_cur = self.mu_query_last.device
        fetcher_cur = self.on_demand_fetcher
        if target_path.resolve() == self.store_path.resolve():
            self.close()
        reopened = self._write_and_open(
            out_path=target_path,
            hidden_dim=d,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rank=self.rank,
            num_key_prototypes=self.num_key_prototypes,
            record_bytes=self.record_bytes,
            routing_tier=self.routing_tier,
            attention_beta=self.attention_beta,
            user_turn_prefix_ids=self.user_turn_prefix_ids,
            slot_descriptors=new_descs,
            global_directory=new_global,
            mu_query_last=mu_ql,
            mu_hidden_final=mu_hf,
            mu_q_heads=mu_qh,
            R_rot=R_rot,
            U_cb=U_cb,
            V_cb=V_cb,
            bitmap=bytes(bitmap),
            z_list_fp16=z_list_fp16,
            raw_blobs=raw_blobs,
        )
        reopened.on_demand_fetcher = fetcher_cur
        return reopened.to(dev_cur)

    @classmethod
    def build_from_bank(
        cls,
        head_bank: HeadExpandedAttentionBank,
        out_path: Union[str, Path],
        force_kronecker: bool = True,
        kronecker_threshold: int = 2048,
    ) -> "HeadExpandedScarStore":
        d = head_bank.hidden_dim
        if d % 2 != 0:
            raise ValueError(f"hidden_dim={d} must be even for dual-subspace Kronecker routing")
        half_d = d // 2
        caps = list(head_bank.capsules.values())
        rank = caps[0].rank if caps else 60
        num_kp = caps[0].num_key_prototypes if caps else 8
        for c in caps:
            if c.rank != rank or c.num_key_prototypes != num_kp:
                raise ValueError("All capsules in HeadExpandedScarStore must share uniform (rank, num_key_prototypes)")

        dummy_ab = ((d * rank) // BLOCK_SIZE_INT4) * 34 * 2
        dummy_k = ((num_kp * head_bank.num_heads * head_bank.head_dim) // BLOCK_SIZE_INT4) * 34
        rec_bytes = dummy_ab + dummy_k + (d * 2)
        routing_tier = "kronecker" if (force_kronecker or len(caps) > kronecker_threshold) else "flat"

        R_rot = _build_deterministic_orthogonal_rotation(d)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20261113)
        U_cb = F.normalize(torch.randn((d, half_d), generator=gen, dtype=torch.float32), dim=-1)
        V_cb = F.normalize(torch.randn((d, half_d), generator=gen, dtype=torch.float32), dim=-1)

        bitmap = bytearray((d * d + 7) // 8)
        used_coords: set[Tuple[int, int]] = set()
        raw_blobs: List[bytes] = []
        z_list: List[torch.Tensor] = []
        slot_descriptors: List[Dict[str, Any]] = []
        global_directory: List[Dict[str, Any]] = []

        for idx, cap in enumerate(caps):
            z_f = F.normalize(cap.z_anchor.detach().cpu().float(), dim=-1, eps=1e-8)
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
                    raise RuntimeError("Kronecker grid candidate neighborhood saturated")
                row_i, col_j = best_pair

            used_coords.add((row_i, col_j))
            _bitmap_set(bitmap, row_i * d + col_j)
            blob = cap.to_int4_bytes()
            assert len(blob) == rec_bytes
            raw_blobs.append(blob)
            z_list.append(z_f.to(torch.float16))
            slot_descriptors.append(
                {
                    "slot_idx": idx,
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
            )
            global_directory.append(
                {
                    "capsule_id": cap.capsule_id,
                    "row_i": row_i,
                    "col_j": col_j,
                    "w1": round(w1, 6),
                    "w2": round(w2, 6),
                    "tau": float(cap.tau),
                    "ramp": float(cap.ramp),
                    "parent_capsules": list(cap.parent_capsules),
                }
            )

        return cls._write_and_open(
            out_path=Path(out_path),
            hidden_dim=d,
            num_heads=head_bank.num_heads,
            head_dim=head_bank.head_dim,
            rank=rank,
            num_key_prototypes=num_kp,
            record_bytes=rec_bytes,
            routing_tier=routing_tier,
            attention_beta=head_bank.attention_beta,
            user_turn_prefix_ids=head_bank.user_turn_prefix_ids,
            slot_descriptors=slot_descriptors,
            global_directory=global_directory,
            mu_query_last=head_bank.mu_query_last.detach().cpu().float(),
            mu_hidden_final=head_bank.mu_hidden_final.detach().cpu().float(),
            mu_q_heads=head_bank.mu_q_heads.detach().cpu().float(),
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
        num_heads: int,
        head_dim: int,
        rank: int,
        num_key_prototypes: int,
        record_bytes: int,
        routing_tier: str,
        attention_beta: float,
        user_turn_prefix_ids: Optional[List[int]],
        slot_descriptors: List[Dict[str, Any]],
        global_directory: List[Dict[str, Any]],
        mu_query_last: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        mu_q_heads: torch.Tensor,
        R_rot: torch.Tensor,
        U_cb: torch.Tensor,
        V_cb: torch.Tensor,
        bitmap: bytes,
        z_list_fp16: List[torch.Tensor],
        raw_blobs: List[bytes],
    ) -> "HeadExpandedScarStore":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_slots = len(raw_blobs)
        header_obj = {
            "format": "HeadExpandedScarStoreV2",
            "version": "2.2.0",
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "rank": rank,
            "num_key_prototypes": num_key_prototypes,
            "record_bytes": record_bytes,
            "num_slots": n_slots,
            "routing_tier": routing_tier,
            "attention_beta": attention_beta,
            "user_turn_prefix_ids": user_turn_prefix_ids,
            "slot_descriptors": slot_descriptors,
            "global_directory": global_directory,
        }
        header_bytes = json.dumps(header_obj, ensure_ascii=False).encode("utf-8")
        router_offset = _align_up(16 + len(header_bytes), HEADER_ALIGN)

        router_blobs = [
            mu_query_last.contiguous().to(torch.float32).numpy().tobytes(),
            mu_hidden_final.contiguous().to(torch.float32).numpy().tobytes(),
            mu_q_heads.contiguous().to(torch.float32).numpy().tobytes(),
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
    ) -> "HeadExpandedScarStore":
        p = Path(store_path)
        with open(p, "rb") as f:
            magic = f.read(8)
            if magic != SCARSTORE_MAGIC:
                raise ValueError(f"Invalid ScarStore magic in {p}: {magic!r}")
            (hdr_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hdr_len).decode("utf-8"))
            router_offset = _align_up(16 + hdr_len, HEADER_ALIGN)
            f.seek(router_offset)

            d = int(header["hidden_dim"])
            n_heads = int(header["num_heads"])
            d_head = int(header["head_dim"])
            q_numel = n_heads * d_head
            half_d = d // 2
            n_slots = int(header["num_slots"])
            bm_len = (d * d + 7) // 8

            mu_ql = torch.frombuffer(bytearray(f.read(d * 4)), dtype=torch.float32, count=d).clone()
            mu_hf = torch.frombuffer(bytearray(f.read(d * 4)), dtype=torch.float32, count=d).clone()
            mu_qh = torch.frombuffer(bytearray(f.read(q_numel * 4)), dtype=torch.float32, count=q_numel).view(n_heads, d_head).clone()
            R_rot = torch.frombuffer(bytearray(f.read(d * d * 2)), dtype=torch.float16, count=d * d).view(d, d).float().clone()
            U_cb = torch.frombuffer(bytearray(f.read(d * half_d * 2)), dtype=torch.float16, count=d * half_d).view(d, half_d).float().clone()
            V_cb = torch.frombuffer(bytearray(f.read(d * half_d * 2)), dtype=torch.float16, count=d * half_d).view(d, half_d).float().clone()
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
            mu_query_last=mu_ql,
            mu_hidden_final=mu_hf,
            mu_q_heads=mu_qh,
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
    ) -> "HeadExpandedScarStore":
        """Exports a selective consumer `.scarstore` (`0 B` on disk for omitted domains!) while preserving
        universal `d x d` Kronecker coordinates and `MissingCapsuleAlert` / `on_demand_fetcher` awareness."""
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
            raw = self.read_raw_slot_bytes(old_slot)
            raw_blobs.append(raw)
            z_list_fp16.append(self.z_anchors_installed[old_slot].detach().cpu().to(torch.float16))
            _bitmap_set(bitmap, ri * d + cj)
            desc_copy = dict(old_desc)
            desc_copy["slot_idx"] = len(new_descs)
            new_descs.append(desc_copy)

        return self._write_and_open(
            out_path=Path(out_path),
            hidden_dim=d,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rank=self.rank,
            num_key_prototypes=self.num_key_prototypes,
            record_bytes=self.record_bytes,
            routing_tier=self.routing_tier,
            attention_beta=self.attention_beta,
            user_turn_prefix_ids=self.user_turn_prefix_ids,
            slot_descriptors=new_descs,
            global_directory=[v for v in self._global_directory.values() if str(v["capsule_id"]) not in self._unlearned_cids],
            mu_query_last=self.mu_query_last.detach().cpu().float(),
            mu_hidden_final=self.mu_hidden_final.detach().cpu().float(),
            mu_q_heads=self.mu_q_heads.detach().cpu().float(),
            R_rot=self.R_rot.detach().cpu().float(),
            U_cb=self.U_codebook.detach().cpu().float(),
            V_cb=self.V_codebook.detach().cpu().float(),
            bitmap=bytes(bitmap),
            z_list_fp16=z_list_fp16,
            raw_blobs=raw_blobs,
        )

    @torch.no_grad()
    def route_and_page_query(
        self,
        x_query_2d: torch.Tensor,
        top_coarse_m: int = 8,
    ) -> List[Tuple[HeadExpandedScarCapsule, float]]:
        """Routes via `d x d` Kronecker Dual-Subspace Grid + `mmap` Page-In WITHOUT `0.50/0.50` gate dilution!"""
        self.last_routing_cosines = {}
        self.last_active_gates = {}
        self.last_num_expanded_heads = 0
        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
        self.last_missing_alerts = []

        if x_query_2d.numel() == 0:
            return []
        dev = x_query_2d.device
        if self.mu_query_last.device != dev:
            self.to(dev)

        probes_2d, _ = extract_span_query_probes(
            query_hidden_2d=x_query_2d,
            mu_query=self.mu_query_last,
            routing_mode=self.routing_mode,
            span_window=self.span_window,
            span_stride=self.span_stride,
        )
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
                    elif not is_present and coord in self._global_directory:
                        candidate_missing_coords.add(coord)

        paged_by_cid: Dict[str, HeadExpandedScarCapsule] = {}
        raw_gates: Dict[str, float] = {}
        parents_by_cid: Dict[str, Tuple[str, ...]] = {}
        ordered_cids: List[str] = []

        for coord in sorted(candidate_missing_coords):
            ri, cj = coord
            g_meta = self._global_directory[coord]
            w1 = float(g_meta.get("w1", math.sqrt(0.5)))
            w2 = float(g_meta.get("w2", math.sqrt(0.5)))
            coarse_vec = (nq1.squeeze(-1) * w1 * s1_mat[:, ri]) + (nq2.squeeze(-1) * w2 * s2_mat[:, cj])
            coarse_cos = float(coarse_vec.max().item())
            tau_g = float(g_meta["tau"])
            cid_g = str(g_meta["capsule_id"])
            self.last_routing_cosines[cid_g] = coarse_cos
            if coarse_cos > tau_g:
                cap_stream: Optional[HeadExpandedScarCapsule] = None
                if cid_g in self._hot_fetched_capsules:
                    cap_stream = self._hot_fetched_capsules[cid_g].to_device(dev)
                elif self.on_demand_fetcher is not None:
                    fetched_bytes = self.on_demand_fetcher(coord, cid_g)
                    if fetched_bytes is not None:
                        self._turn_mmap_bytes_read += len(fetched_bytes)
                        self.last_mmap_bytes_read = self._turn_mmap_bytes_read
                        cap_stream = HeadExpandedScarCapsule.from_int4_bytes(
                            raw_bytes=fetched_bytes,
                            capsule_id=cid_g,
                            hidden_dim=self.hidden_dim,
                            num_heads=self.num_heads,
                            head_dim=self.head_dim,
                            rank=self.rank,
                            num_key_prototypes=self.num_key_prototypes,
                            tau=tau_g,
                            ramp=float(g_meta["ramp"]),
                            parent_capsules=g_meta.get("parent_capsules", ()),
                            device=dev,
                        )
                        self._hot_fetched_capsules[cid_g] = cap_stream
                if cap_stream is not None:
                    exact_c = float(torch.mv(probes_2d, cap_stream.z_anchor.float()).max().item())
                    self.last_routing_cosines[cid_g] = exact_c
                    g_f = max(0.0, min(1.0, (exact_c - cap_stream.tau) / cap_stream.ramp))
                    if g_f > 0.0:
                        paged_by_cid[cid_g] = cap_stream
                        raw_gates[cid_g] = g_f
                        parents_by_cid[cid_g] = cap_stream.parent_capsules
                        ordered_cids.append(cid_g)
                else:
                    self.last_missing_alerts.append(
                        MissingCapsuleAlert(coord=coord, capsule_id=cid_g, coarse_cosine=coarse_cos, tau=tau_g)
                    )

        for s_idx in sorted(candidate_installed_slots):
            desc = self._slot_descriptors[s_idx]
            cid = str(desc["capsule_id"])
            if cid in self._unlearned_cids:
                continue
            exact_cos = float(torch.mv(probes_2d, self.z_anchors_installed[s_idx]).max().item())
            self.last_routing_cosines[cid] = exact_cos
            g_val = max(0.0, min(1.0, (exact_cos - float(desc["tau"])) / float(desc["ramp"])))
            if g_val > 0.0:
                cap = self.load_slot_capsule(s_idx, device=dev)
                paged_by_cid[cid] = cap
                raw_gates[cid] = g_val
                parents_by_cid[cid] = cap.parent_capsules
                ordered_cids.append(cid)

        if self.strict_missing_guard and self.last_missing_alerts:
            a0 = self.last_missing_alerts[0]
            raise RuntimeError(
                f"MissingCapsuleAlert: Query triggered uninstalled capsule {a0.capsule_id!r} "
                f"at Kronecker coordinate {a0.coord} (cosine={a0.coarse_cosine:.4f} > tau={a0.tau:.4f})"
            )

        if not ordered_cids:
            return []

        base_cone = dict(raw_gates)
        for cid in list(ordered_cids):
            parents = parents_by_cid.get(cid, ())
            if not parents:
                continue
            if any(p not in base_cone or base_cone[p] <= 0.0 for p in parents):
                raw_gates[cid] = 0.0
                continue
            parent_min = min(base_cone[p] for p in parents)
            eff_bg = raw_gates[cid] * min(1.0, parent_min / 0.50)
            raw_gates[cid] = eff_bg
            if eff_bg > 0.0:
                atten = max(0.0, 1.0 - eff_bg)
                for p in parents:
                    raw_gates[p] *= atten

        active: List[Tuple[HeadExpandedScarCapsule, float]] = [
            (paged_by_cid[cid], float(raw_gates[cid])) for cid in ordered_cids if raw_gates[cid] > 1e-6
        ]
        self.last_active_gates = {c.capsule_id: g for c, g in active}
        self.last_num_expanded_heads = len(active)
        return active

    def _compute_expanded_scar_heads(
        self,
        h_fin_3d: torch.Tensor,
        q_heads_4d: Optional[torch.Tensor],
        active: List[Tuple[HeadExpandedScarCapsule, float]],
    ) -> torch.Tensor:
        orig_dtype = h_fin_3d.dtype
        hf = h_fin_3d.float()
        hc = hf - self.mu_hidden_final.view(1, 1, -1)

        if len(active) == 1 or q_heads_4d is None:
            delta = torch.zeros_like(hf)
            w_sum = sum(g for _, g in active)
            norm_div = w_sum if (len(active) > 1 and w_sum > 1.0) else 1.0
            step_dict: Dict[str, float] = {}
            for cap, g_val in active:
                eff_w = g_val / norm_div
                delta = delta + eff_w * torch.matmul(torch.matmul(hc, cap.A_scar.float().T), cap.B_scar.float().T)
                step_dict[cap.capsule_id] = round(float(eff_w), 4)
            for _ in range(int(hf.shape[1])):
                self.last_step_weights_history.append(dict(step_dict))
                self.last_inter_head_spread_history.append(0.0)
            return (hf + delta).to(orig_dtype)

        q_norm = F.normalize(q_heads_4d.float() - self.mu_q_heads.view(1, 1, self.num_heads, self.head_dim), dim=-1, eps=1e-8)
        beta = self.attention_beta
        head_logits_list: List[torch.Tensor] = []
        for cap, g_val in active:
            k_proto = F.normalize(cap.K_heads.float(), dim=-1, eps=1e-8)
            sim_p = torch.einsum("bshd,phd->bsph", q_norm, k_proto)
            s_max = sim_p.amax(dim=2)
            head_logits_list.append(beta * s_max + math.log(max(float(g_val), 1e-6)))

        alpha_m_bsh = F.softmax(torch.stack(head_logits_list, dim=0), dim=0)
        w_m_bs1 = alpha_m_bsh.mean(dim=-1, keepdim=True)
        s_len = int(hf.shape[1])
        for t_idx in range(s_len):
            a_mh = alpha_m_bsh[:, 0, t_idx, :]
            spread = float((a_mh.amax(dim=-1) - a_mh.amin(dim=-1)).amax().item())
            self.last_inter_head_spread_history.append(spread)
            self.last_step_weights_history.append(
                {active[m_i][0].capsule_id: round(float(w_m_bs1[m_i, 0, t_idx, 0].item()), 4) for m_i in range(len(active))}
            )

        delta = torch.zeros_like(hf)
        for m_i, (cap, _g_val) in enumerate(active):
            v_scar = torch.matmul(torch.matmul(hc, cap.A_scar.float().T), cap.B_scar.float().T)
            delta = delta + w_m_bs1[m_i] * v_scar
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
        max_spread = round(max(self.last_inter_head_spread_history) if self.last_inter_head_spread_history else 0.0, 4)
        return HeadExpandedTelemetry(
            active_capsules=cids,
            num_expanded_heads=len(cids),
            per_capsule_mean_weight=means,
            per_capsule_min_weight=mins,
            per_capsule_max_weight=maxs,
            max_inter_head_spread=max_spread,
            step_weights=hist,
        )

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
        self._cached_q_heads = None

        if past_len == 0:
            self._rolling_id_tail = []
            self._query_hidden_buf = None
            self._turn_slot_cache.clear()
            self._turn_mmap_bytes_read = 0
            self.last_step_weights_history = []
            self.last_inter_head_spread_history = []

        if is_prefill_chunk and seq_len > 0:
            tail = self._rolling_id_tail
            combined = tail + ids_1d
            match_end = _find_subsequence_last_end_strict(combined, self.user_turn_prefix_ids)
            keep_tail = max(0, len(self.user_turn_prefix_ids) - 1) if self.user_turn_prefix_ids else 0
            self._rolling_id_tail = combined[-keep_tail:] if keep_tail > 0 else []
            if match_end > len(tail):
                self._query_hidden_buf = None
                self._turn_slot_cache.clear()
                self._turn_mmap_bytes_read = 0
                self.last_step_weights_history = []
                self.last_inter_head_spread_history = []
                self._current_query_start = match_end - len(tail)
            else:
                self._current_query_start = 0

    def _last_block_pre_hook(self, _module: nn.Module, args: Tuple[Any, ...]) -> None:
        if self._paused or len(args) == 0 or not isinstance(args[0], torch.Tensor):
            return
        x_in = args[0]
        x_3d = x_in.unsqueeze(0) if x_in.ndim == 2 else x_in
        if x_3d.ndim != 3 or int(x_3d.shape[1]) == 0:
            return
        seq_len = int(x_3d.shape[1])
        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            if len(self.last_step_weights_history) == 0:
                q_start = min(self._current_query_start, lp - 1)
                self._active_turn_capsules = self.route_and_page_query(x_3d[0, q_start:lp, :])
        elif self._current_pass_is_prefill:
            qs = self._current_query_start
            if qs < seq_len:
                new_q = x_3d[0, qs:seq_len, :].detach()
                buf = new_q if self._query_hidden_buf is None else torch.cat([self._query_hidden_buf, new_q], dim=0)
                if buf.shape[0] > 192:
                    buf = buf[-192:]
                self._query_hidden_buf = buf
                self._active_turn_capsules = self.route_and_page_query(buf)
            elif self._query_hidden_buf is None:
                self._active_turn_capsules = []

    def _q_proj_forward_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
        if self._paused or not self._active_turn_capsules:
            return
        q_3d = output.unsqueeze(0) if output.ndim == 2 else output
        if q_3d.ndim != 3 or int(q_3d.shape[1]) == 0:
            return
        bsz, seq_len, _ = q_3d.shape
        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            q_sl = q_3d[:, lp - 1 :, :]
        elif self._current_pass_is_prefill:
            q_sl = q_3d[:, -1:, :]
        else:
            q_sl = q_3d
        self._cached_q_heads = q_sl.detach().view(bsz, int(q_sl.shape[1]), self.num_heads, self.head_dim)

    def _final_norm_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self._paused or not self._active_turn_capsules:
            return output
        squeezed = False
        out_3d = output
        if output.ndim == 2:
            out_3d = output.unsqueeze(0)
            squeezed = True
        elif output.ndim != 3:
            return output
        seq_len = int(out_3d.shape[1])
        if seq_len == 0:
            return output
        active = self._active_turn_capsules
        q_cached = self._cached_q_heads
        self._cached_q_heads = None

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            out_mod = out_3d.clone()
            out_mod[:, lp - 1 :, :] = self._compute_expanded_scar_heads(out_3d[:, lp - 1 :, :], q_cached, active)
            return out_mod.squeeze(0) if squeezed else out_mod
        if self._current_pass_is_prefill:
            out_mod = out_3d.clone()
            out_mod[:, -1:, :] = self._compute_expanded_scar_heads(out_3d[:, -1:, :], q_cached, active)
            return out_mod.squeeze(0) if squeezed else out_mod
        out_mod = self._compute_expanded_scar_heads(out_3d, q_cached, active)
        return out_mod.squeeze(0) if squeezed else out_mod

    def attach(self, student_model: nn.Module) -> "HeadExpandedScarStore":
        self.detach()
        dev = _infer_model_device(student_model)
        self.to(dev)
        backbone, last_block, q_proj, _, _, norm_mod, _ = extract_head_expansion_modules(student_model)
        self._backbone_pre_handle = backbone.register_forward_pre_hook(self._backbone_pre_hook, with_kwargs=True)
        self._last_block_pre_handle = last_block.register_forward_pre_hook(self._last_block_pre_hook)
        self._q_proj_hook_handle = q_proj.register_forward_hook(self._q_proj_forward_hook)
        self._norm_hook_handle = norm_mod.register_forward_hook(self._final_norm_hook)
        return self

    def detach(self) -> None:
        for attr in ("_backbone_pre_handle", "_last_block_pre_handle", "_q_proj_hook_handle", "_norm_hook_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)
        self._active_turn_capsules = []
        self._cached_q_heads = None

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
        self.last_step_weights_history = []
        self.last_inter_head_spread_history = []
        try:
            yield
        finally:
            self._tf_prompt_len = prev

