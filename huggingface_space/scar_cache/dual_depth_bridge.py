"""Horizon 2 (`v2.0.0`) Stage 3 — Dual-Depth Golden-Slice Scar Grafting (`L_{mid} + L_{final}`) & Sleep Bridge Consolidation.

Completes the 3-stage `ScarStore v2` innovation architecture (`ZERO Backpropagation / ZERO SGD`):
  1. **Golden-Slice Dual-Depth Capsules (`DualDepthScarCapsule`, `61.25 KB` INT4 total at `d=896, r_mid=32, r_final=32`)**:
     - Routes causally at the `50%` Golden-Slice midpoint (`Layer 12` on `24`-layer `Qwen2.5-0.5B`) via
       `Span-Max` centered spherical cones (`z_anchor_mid`), preserving **exact `0.0000000000` (`torch.equal`)**
       out-of-cone invariance across all layers.
     - Injects null-space semantic steering `(B_mid, A_mid)` at `Layer L/2` so the upper half of the
       Transformer (`Layers L/2 .. L-1` multi-head self-attention + SwiGLU MLPs) **genuinely cross-attends
       and reasons across multiple simultaneous knowledge capsules** before `final_norm`!
     - Applies exact Cutting-Plane Woodbury residual cleanup `(B_final, A_final)` at `final_norm` (`Layer L`)
       solved *after* non-linear propagation through `Layers L/2 .. L-1`.
  2. **Epistemic Uncertainty Reflex & Interdisciplinary Bridge Consolidation (`BridgeEscalationDecision`)**:
     - Computes `System 1` epistemic telemetry (`confidence`, `uncertainty`, `margin = p_1 - p_2`,
       `is_uncertain`) when `M >= 2` capsules intersect on a novel cross-domain synthesis query.
     - Consolidates high-surprise Teacher / `Ghost-Jump` trajectories into a permanent `61.25 KB` INT4
       **Interdisciplinary Bridge Capsule** (`bridge_<A>__<B>`) during the Artificial Sleep Cycle!
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.scar_grafting import (
    ScarBank,
    ScarCapsule,
    ScarGraftEngine,
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
    dequantize_block64_int4,
    quantize_block64_int4,
)


def extract_mid_and_final_modules(
    student_model: nn.Module,
    mid_layer_idx: Optional[int] = None,
) -> Tuple[nn.Module, nn.Module, int, nn.Module, nn.Linear]:
    """Resolves `(backbone, mid_layer_module, resolved_mid_idx, final_norm_module, lm_head)`."""
    backbone, norm_mod, lm_head = _extract_backbone_and_norm(student_model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", getattr(backbone, "blocks", None))
    if layers is None or len(layers) == 0:
        raise AttributeError(
            f"Backbone {type(backbone).__name__} does not expose `.layers` / `.h` / `.blocks` for Dual-Depth grafting."
        )
    n_layers = len(layers)
    idx = (max(0, (n_layers // 2) - 1)) if mid_layer_idx is None else int(mid_layer_idx)
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"mid_layer_idx={idx} out of range for {n_layers}-layer backbone")
    return backbone, layers[idx], idx, norm_mod, lm_head


@dataclass(frozen=True)
class BridgeEscalationDecision:
    """`System 1` epistemic telemetry (`SKILL.md` `ReflexDecision` contract) for interdisciplinary queries."""

    active_capsules: List[str]
    confidence: float        # [0.0, 1.0] mean max token probability
    uncertainty: float       # [0.0, 1.0] normalized predictive entropy
    margin: float            # [0.0, 1.0] mean (p_1 - p_2) probability margin
    is_uncertain: bool       # True when System 1 should escalate to System 2 & mint a Bridge Scar
    latency_ms: float


@dataclass
class DualDepthScarCapsule:
    """Dual-Depth (`L_{mid}` Golden-Slice + `L_{final}`) directional knowledge capsule.

    When quantized via `to_int4_bytes()` at `d=896, rank_mid=32, rank_final=32`, occupies
    **exact `62,720 bytes` (`61.25 KB`)** — identical to a single-layer `r=64` INT4 capsule!
    """

    capsule_id: str
    z_anchor_mid: torch.Tensor   # [d] FP16 unit vector at Golden-Slice mid-layer
    A_mid: torch.Tensor          # [r_mid, d] FP16
    B_mid: torch.Tensor          # [d, r_mid] FP16
    A_final: torch.Tensor        # [r_final, d] FP16
    B_final: torch.Tensor        # [d, r_final] FP16
    tau: float = 0.60
    ramp: float = 0.16
    parent_capsules: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hidden_dim(self) -> int:
        return int(self.z_anchor_mid.shape[0])

    @property
    def rank_mid(self) -> int:
        return int(self.A_mid.shape[0])

    @property
    def rank_final(self) -> int:
        return int(self.A_final.shape[0])

    @property
    def total_rank(self) -> int:
        return self.rank_mid + self.rank_final

    @property
    def int4_physical_bytes(self) -> int:
        d = self.hidden_dim
        b_mid_blk = (d * self.rank_mid) // BLOCK_SIZE_INT4
        b_fin_blk = (d * self.rank_final) // BLOCK_SIZE_INT4
        mid_bytes = 2 * (b_mid_blk * 32 + b_mid_blk * 2)
        fin_bytes = 2 * (b_fin_blk * 32 + b_fin_blk * 2)
        return mid_bytes + fin_bytes + (d * 2)

    def to_int4_bytes(self) -> bytes:
        bm_p, bm_s = quantize_block64_int4(self.B_mid)
        am_p, am_s = quantize_block64_int4(self.A_mid)
        bf_p, bf_s = quantize_block64_int4(self.B_final)
        af_p, af_s = quantize_block64_int4(self.A_final)
        z_b = F.normalize(self.z_anchor_mid.detach().float(), dim=-1, eps=1e-8).to(torch.float16).cpu()
        parts = [
            bm_p.numpy().tobytes(),
            bm_s.numpy().tobytes(),
            am_p.numpy().tobytes(),
            am_s.numpy().tobytes(),
            bf_p.numpy().tobytes(),
            bf_s.numpy().tobytes(),
            af_p.numpy().tobytes(),
            af_s.numpy().tobytes(),
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
        rank_mid: int = 32,
        rank_final: int = 32,
        tau: float = 0.60,
        ramp: float = 0.16,
        parent_capsules: Sequence[str] = (),
        device: Union[str, torch.device] = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DualDepthScarCapsule":
        buf = bytearray(raw_bytes)
        n_m = (hidden_dim * rank_mid) // BLOCK_SIZE_INT4
        n_f = (hidden_dim * rank_final) // BLOCK_SIZE_INT4
        off = 0

        def _take_mat(n_blk: int, shape: Tuple[int, int]) -> torch.Tensor:
            nonlocal off
            p = torch.frombuffer(buf, dtype=torch.uint8, count=n_blk * 32, offset=off).view(n_blk, 32).clone()
            off += n_blk * 32
            s = torch.frombuffer(buf, dtype=torch.float16, count=n_blk, offset=off).clone()
            off += n_blk * 2
            return dequantize_block64_int4(p, s, shape, dtype=torch.float16, device=device)

        B_mid = _take_mat(n_m, (hidden_dim, rank_mid))
        A_mid = _take_mat(n_m, (rank_mid, hidden_dim))
        B_final = _take_mat(n_f, (hidden_dim, rank_final))
        A_final = _take_mat(n_f, (rank_final, hidden_dim))
        z = torch.frombuffer(buf, dtype=torch.float16, count=hidden_dim, offset=off).clone().to(device)
        meta = dict(metadata or {})
        parents_tup = tuple(str(x) for x in (parent_capsules or meta.get("parent_capsules", ())))
        return cls(
            capsule_id=capsule_id,
            z_anchor_mid=z,
            A_mid=A_mid,
            B_mid=B_mid,
            A_final=A_final,
            B_final=B_final,
            tau=tau,
            ramp=ramp,
            parent_capsules=parents_tup,
            metadata=meta,
        )

    def to_device(self, device: Union[str, torch.device]) -> "DualDepthScarCapsule":
        dev = torch.device(device)
        self.z_anchor_mid = self.z_anchor_mid.to(dev)
        self.A_mid = self.A_mid.to(dev)
        self.B_mid = self.B_mid.to(dev)
        self.A_final = self.A_final.to(dev)
        self.B_final = self.B_final.to(dev)
        return self


class DualDepthScarBank(nn.Module):
    """Causal Single-Pass Dual-Depth (`Layer L/2` + `Layer L`) Cone-Gated Scar Bank.

    Routes once at `Layer L/2` (`Golden-Slice`) via `Span-Max` sliding windows:
      - Out-of-cone prompts (`gate == 0`) bypass BOTH `Layer L/2` and `Layer L` with
        **exact `0.0000000000` (`torch.equal`)** invariance!
      - In-cone multi-domain prompts inject `(B_mid, A_mid)` into `Layer L/2`, enabling
        the upper Transformer self-attention blocks (`Layers L/2 .. L-1`) to cross-attend
        across active capsules before `(B_final, A_final)` refines `final_norm`!
    """

    def __init__(
        self,
        hidden_dim: int,
        mid_layer_idx: int,
        mu_query_mid: torch.Tensor,
        mu_hidden_mid: torch.Tensor,
        u_preserved_mid: torch.Tensor,
        mu_hidden_final: torch.Tensor,
        u_preserved_final: torch.Tensor,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        routing_mode: str = "span_max",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mid_layer_idx = int(mid_layer_idx)
        self.user_turn_prefix_ids = [int(x) for x in user_turn_prefix_ids] if user_turn_prefix_ids else None
        self.routing_mode = str(routing_mode)

        self.register_buffer("mu_query_mid", mu_query_mid.detach().float().clone())
        self.register_buffer("mu_hidden_mid", mu_hidden_mid.detach().float().clone())
        self.register_buffer("u_preserved_mid", u_preserved_mid.detach().float().clone())
        self.register_buffer("mu_hidden_final", mu_hidden_final.detach().float().clone())
        self.register_buffer("u_preserved_final", u_preserved_final.detach().float().clone())

        self.register_buffer("_z_mid_matrix", torch.empty((0, self.hidden_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("_tau_vec", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_ramp_vec", torch.empty(0, dtype=torch.float32), persistent=False)

        self._capsules: Dict[str, DualDepthScarCapsule] = {}
        self._capsule_order: List[str] = []

        self._pre_hook_handle: Optional[Any] = None
        self._mid_hook_handle: Optional[Any] = None
        self._norm_hook_handle: Optional[Any] = None
        self._paused: bool = False
        self._current_pass_is_prefill: bool = True
        self._current_query_start: int = 0
        self._tf_prompt_len: Optional[int] = None
        self._active_gates: List[Tuple[str, float]] = []

        self.last_routing_cosines: Dict[str, float] = {}
        self.last_active_gates: Dict[str, float] = {}

    @property
    def capsules(self) -> Dict[str, DualDepthScarCapsule]:
        return dict(self._capsules)

    def _rebuild_index(self) -> None:
        dev = self.mu_query_mid.device
        if not self._capsule_order:
            self._z_mid_matrix = torch.empty((0, self.hidden_dim), dtype=torch.float32, device=dev)
            self._tau_vec = torch.empty(0, dtype=torch.float32, device=dev)
            self._ramp_vec = torch.empty(0, dtype=torch.float32, device=dev)
            return
        zs, taus, ramps = [], [], []
        for cid in self._capsule_order:
            c = self._capsules[cid].to_device(dev)
            zs.append(F.normalize(c.z_anchor_mid.float(), dim=-1, eps=1e-8))
            taus.append(float(c.tau))
            ramps.append(float(c.ramp))
        self._z_mid_matrix = torch.stack(zs, dim=0).to(dev)
        self._tau_vec = torch.tensor(taus, dtype=torch.float32, device=dev)
        self._ramp_vec = torch.tensor(ramps, dtype=torch.float32, device=dev)

    def add_capsule(self, cap: DualDepthScarCapsule) -> None:
        if cap.capsule_id not in self._capsules:
            self._capsule_order.append(cap.capsule_id)
        self._capsules[cap.capsule_id] = cap.to_device(self.mu_query_mid.device)
        self._rebuild_index()

    def unlearn(self, capsule_id: str) -> DualDepthScarCapsule:
        removed = self._capsules.pop(capsule_id)
        self._capsule_order = [c for c in self._capsule_order if c != capsule_id]
        self._active_gates = [(c, g) for c, g in self._active_gates if c != capsule_id]
        self._rebuild_index()
        return removed

    @torch.no_grad()
    def route_mid_query(self, h_mid_query_2d: torch.Tensor) -> List[Tuple[str, float]]:
        if not self._capsule_order or h_mid_query_2d.numel() == 0:
            self.last_routing_cosines = {}
            self.last_active_gates = {}
            return []
        dev = h_mid_query_2d.device
        if self.mu_query_mid.device != dev:
            self.to(dev)
            self._rebuild_index()

        probes_2d, _ = extract_span_query_probes(
            query_hidden_2d=h_mid_query_2d,
            mu_query=self.mu_query_mid,
            routing_mode=self.routing_mode,
        )
        cos_mat = torch.matmul(probes_2d, self._z_mid_matrix.T)
        cos_vec = cos_mat.amax(dim=0)
        raw_g = torch.clamp((cos_vec - self._tau_vec) / self._ramp_vec, min=0.0, max=1.0)

        cos_list = cos_vec.tolist()
        self.last_routing_cosines = {cid: float(cos_list[i]) for i, cid in enumerate(self._capsule_order)}
        if float(raw_g.max().item()) <= 0.0:
            self.last_active_gates = {}
            return []

        g_vals = [float(x) for x in raw_g.tolist()]
        idx_map = {cid: i for i, cid in enumerate(self._capsule_order)}

        # Conjunctive intersection cone & smooth parent supersession for Interdisciplinary Bridge Capsules
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

        g_sum = sum(g_vals)
        if g_sum <= 1e-8:
            self.last_active_gates = {}
            return []
        if g_sum > 1.0:
            g_vals = [g / g_sum for g in g_vals]
        active = [(self._capsule_order[i], float(g_vals[i])) for i in range(len(self._capsule_order)) if g_vals[i] > 1e-6]
        self.last_active_gates = dict(active)
        return active

    def _apply_mid_scars(self, h_3d: torch.Tensor, active: List[Tuple[str, float]]) -> torch.Tensor:
        orig_dtype = h_3d.dtype
        hf = h_3d.float()
        hc = hf - self.mu_hidden_mid.view(1, 1, -1)
        delta = torch.zeros_like(hf)
        for cid, g_val in active:
            cap = self._capsules[cid]
            delta = delta + g_val * torch.matmul(torch.matmul(hc, cap.A_mid.float().T), cap.B_mid.float().T)
        return (hf + delta).to(orig_dtype)

    def _apply_final_scars(self, h_3d: torch.Tensor, active: List[Tuple[str, float]]) -> torch.Tensor:
        orig_dtype = h_3d.dtype
        hf = h_3d.float()
        hc = hf - self.mu_hidden_final.view(1, 1, -1)
        delta = torch.zeros_like(hf)
        for cid, g_val in active:
            cap = self._capsules[cid]
            delta = delta + g_val * torch.matmul(torch.matmul(hc, cap.A_final.float().T), cap.B_final.float().T)
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
        self._current_pass_is_prefill = (_past_kv_seen_length(past_kv) == 0)
        if self._current_pass_is_prefill and isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            ids_1d = input_ids[0].tolist() if input_ids.ndim >= 2 else input_ids.tolist()
            self._current_query_start = find_subsequence_last_end(ids_1d, self.user_turn_prefix_ids)

    def _mid_layer_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> Any:
        if self._paused or not self._capsule_order:
            return output
        is_tuple = isinstance(output, tuple)
        h_tensor: torch.Tensor = output[0] if is_tuple else output
        squeezed = False
        h_3d = h_tensor
        if h_tensor.ndim == 2:
            h_3d = h_tensor.unsqueeze(0)
            squeezed = True
        elif h_tensor.ndim != 3:
            return output

        seq_len = int(h_3d.shape[1])
        if seq_len == 0:
            return output

        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            q_start = min(self._current_query_start, lp - 1)
            active = self.route_mid_query(h_3d[0, q_start:lp, :])
            self._active_gates = active
            if not active:
                return output
            h_mod = h_3d.clone()
            h_mod[:, lp - 1 :, :] = self._apply_mid_scars(h_3d[:, lp - 1 :, :], active)
        elif self._current_pass_is_prefill:
            q_start = min(self._current_query_start, seq_len - 1)
            active = self.route_mid_query(h_3d[0, q_start:seq_len, :])
            self._active_gates = active
            if not active:
                return output
            h_mod = h_3d.clone()
            # Inject at `lp - 1` (matching Stage B training trajectory) so Layers L/2..L-1 self-attention cross-attends across all generated steps!
            h_mod[:, -1:, :] = self._apply_mid_scars(h_3d[:, -1:, :], active)
        else:
            active = self._active_gates
            if not active:
                return output
            h_mod = self._apply_mid_scars(h_3d, active)

        h_out = h_mod.squeeze(0) if squeezed else h_mod
        return (h_out,) + output[1:] if is_tuple else h_out

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
        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            out_mod = out_3d.clone()
            out_mod[:, lp - 1 :, :] = self._apply_final_scars(out_3d[:, lp - 1 :, :], active)
            return out_mod.squeeze(0) if squeezed else out_mod

        if self._current_pass_is_prefill:
            out_mod = out_3d.clone()
            out_mod[:, -1:, :] = self._apply_final_scars(out_3d[:, -1:, :], active)
            return out_mod.squeeze(0) if squeezed else out_mod

        out_mod = self._apply_final_scars(out_3d, active)
        return out_mod.squeeze(0) if squeezed else out_mod

    def attach(self, student_model: nn.Module) -> "DualDepthScarBank":
        self.detach()
        dev = _infer_model_device(student_model)
        self.to(dev)
        self._rebuild_index()
        backbone, mid_mod, _, norm_mod, _ = extract_mid_and_final_modules(student_model, self.mid_layer_idx)
        self._pre_hook_handle = backbone.register_forward_pre_hook(self._backbone_pre_hook, with_kwargs=True)
        self._mid_hook_handle = mid_mod.register_forward_hook(self._mid_layer_hook)
        self._norm_hook_handle = norm_mod.register_forward_hook(self._final_norm_hook)
        return self

    def detach(self) -> None:
        for attr in ("_pre_hook_handle", "_mid_hook_handle", "_norm_hook_handle"):
            h = getattr(self, attr, None)
            if h is not None:
                h.remove()
                setattr(self, attr, None)
        self._active_gates = []

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
        try:
            yield
        finally:
            self._tf_prompt_len = prev


class DualDepthBridgeEngine:
    """Closed-Form Two-Stage Dual-Depth Synthesis & Interdisciplinary Sleep Bridge Engine (`ZERO SGD`)."""

    @staticmethod
    @torch.no_grad()
    def _capture_mid_and_final_states(
        student_model: nn.Module,
        input_ids_2d: torch.Tensor,
        mid_layer_idx: Optional[int] = None,
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
    def _rosetta_cutting_plane(
        w_head: torch.Tensor,
        lg_teacher_1d: torch.Tensor,
        lg_student_1d: torch.Tensor,
        k_eff: int,
        logit_gain: float,
        vocab_ridge: float,
    ) -> torch.Tensor:
        v_common = int(lg_teacher_1d.shape[0])
        d = int(w_head.shape[1])
        k_init = max(2, min(int(k_eff), v_common, max(10, d // 10)))
        t_top = lg_teacher_1d.topk(k_init).indices
        s_top = lg_student_1d.topk(k_init).indices
        s_union = torch.unique(torch.cat([t_top, s_top], dim=0))
        t_best = int(t_top[0].item())
        r_star = torch.zeros(d, dtype=torch.float32)

        for pass_i in range(10):
            w_s = w_head[s_union]
            t_sub = lg_teacher_1d[s_union]
            s_sub = lg_student_1d[s_union]
            m_s = int(s_union.numel())
            target_sub = float(logit_gain) * (t_sub - t_sub.mean())
            best_mask = s_union == t_best
            if bool(best_mask.any().item()) and m_s > 1:
                other_max = float(target_sub[~best_mask].max().item())
                if float(target_sub[best_mask][0].item()) - other_max < 1.35:
                    target_sub = target_sub.clone()
                    target_sub[best_mask] = other_max + 1.35
            delta_lg = target_sub - (s_sub - s_sub.mean())
            gram_v = (w_s @ w_s.T) + float(vocab_ridge) * torch.eye(m_s, dtype=torch.float32)
            r_star = w_s.T @ torch.linalg.solve(gram_v, delta_lg)
            if pass_i < 9 and m_s + 6 <= max(12, int(0.62 * d)):
                pred_lg = lg_student_1d + (w_head[:v_common] @ r_star)
                top2_v, top2_i = pred_lg.topk(min(2, v_common))
                margin_ok = top2_v.numel() < 2 or float((top2_v[0] - top2_v[1]).item()) >= 1.00
                if int(top2_i[0].item()) == t_best and margin_ok:
                    break
                intruders = pred_lg.topk(min(6, v_common)).indices
                new_u = torch.unique(torch.cat([s_union, intruders], dim=0))
                if int(new_u.numel()) == m_s:
                    break
                s_union = new_u
            else:
                break
        return r_star

    @staticmethod
    def _woodbury_dual_svd(
        hp_mat: torch.Tensor,
        rc_mat: torch.Tensor,
        p_null: torch.Tensor,
        rank: int,
        hidden_ridge_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_tok = int(hp_mat.shape[0])
        gram_h = hp_mat @ hp_mat.T
        mean_diag = max(float(gram_h.diagonal().mean().item()), 1e-6)
        lam_h = max(1e-5, float(hidden_ridge_ratio) * mean_diag)
        alpha = torch.linalg.solve(gram_h + lam_h * torch.eye(n_tok, dtype=torch.float32), rc_mat)
        m_scar = (alpha.T @ hp_mat) @ p_null
        u_m, s_m, vh_m = torch.linalg.svd(m_scar, full_matrices=False)
        r_eff = max(1, min(int(rank), int(s_m.shape[0])))
        sqrt_s = s_m[:r_eff].clamp(min=0.0).sqrt()
        b_scar = (u_m[:, :r_eff] * sqrt_s.unsqueeze(0)).to(torch.float16).contiguous()
        a_scar = (sqrt_s.unsqueeze(1) * vh_m[:r_eff, :]).to(torch.float16).contiguous()
        return b_scar, a_scar

    @classmethod
    @torch.no_grad()
    def calibrate_dual_depth_student(
        cls,
        student_model: nn.Module,
        calibration_prompts: Sequence[Union[str, torch.Tensor]],
        tokenizer: Optional[Any] = None,
        chat_formatter: Optional[Callable[[str], str]] = None,
        mid_layer_idx: Optional[int] = None,
        preserved_rank: int = 8,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
    ) -> DualDepthScarBank:
        """Closed-form calibration at BOTH `Layer L/2` (`Golden-Slice`) and `Layer L` (`final_norm`)."""
        _, _, resolved_mid_idx, _, lm_head = extract_mid_and_final_modules(student_model, mid_layer_idx)
        d = int(lm_head.weight.shape[1])
        prefix_ids = (
            [int(x) for x in user_turn_prefix_ids]
            if user_turn_prefix_ids is not None
            else detect_chatml_user_prefix_ids(tokenizer)
        )

        q_mids, all_mid, all_fin = [], [], []
        for item in calibration_prompts:
            if isinstance(item, str):
                text = chat_formatter(item) if chat_formatter is not None else item
                ids_2d = tokenizer(text, return_tensors="pt").input_ids
            else:
                ids_2d = item if item.ndim == 2 else item.unsqueeze(0)
            h_mid, h_fin = cls._capture_mid_and_final_states(student_model, ids_2d, resolved_mid_idx)
            q_start = find_subsequence_last_end(ids_2d[0].tolist(), prefix_ids)
            q_mids.append(h_mid[q_start:].mean(dim=0))
            all_mid.append(h_mid)
            all_fin.append(h_fin)

        mu_q_mid = torch.stack(q_mids, dim=0).mean(dim=0)
        cat_mid = torch.cat(all_mid, dim=0)
        cat_fin = torch.cat(all_fin, dim=0)
        mu_h_mid = cat_mid.mean(dim=0)
        mu_h_fin = cat_fin.mean(dim=0)

        r_eff = max(0, min(int(preserved_rank), cat_mid.shape[0] - 1, d - 1))
        if r_eff > 0:
            _, _, vh_m = torch.linalg.svd(cat_mid - mu_h_mid, full_matrices=False)
            _, _, vh_f = torch.linalg.svd(cat_fin - mu_h_fin, full_matrices=False)
            u_mid = vh_m[:r_eff, :].T.contiguous()
            u_fin = vh_f[:r_eff, :].T.contiguous()
        else:
            u_mid = torch.zeros((d, 0), dtype=torch.float32)
            u_fin = torch.zeros((d, 0), dtype=torch.float32)

        return DualDepthScarBank(
            hidden_dim=d,
            mid_layer_idx=resolved_mid_idx,
            mu_query_mid=mu_q_mid,
            mu_hidden_mid=mu_h_mid,
            u_preserved_mid=u_mid,
            mu_hidden_final=mu_h_fin,
            u_preserved_final=u_fin,
            user_turn_prefix_ids=prefix_ids,
        )

    @classmethod
    @torch.no_grad()
    def solve_dual_depth_capsule(
        cls,
        capsule_id: str,
        episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        dual_bank: DualDepthScarBank,
        rank_mid: int = 32,
        rank_final: int = 32,
        mid_steering_alpha: float = 0.18,
        top_vocab_k: int = 48,
        logit_gain: float = 1.30,
        vocab_ridge: float = 0.10,
        hidden_ridge_ratio: float = 0.015,
        tau: float = 0.60,
        ramp: float = 0.16,
        parent_capsules: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DualDepthScarCapsule:
        """Two-Stage Closed-Form Dual-Depth Synthesis (`ZERO Backpropagation / ZERO SGD`):
        1. Solves `(B_mid, A_mid)` (`rank_mid=32`) at `Layer L/2` via norm-pulled semantic Rosetta targets.
        2. Propagates `(B_mid, A_mid)` forward through `Layers L/2 .. L-1` (real multi-head attention + SwiGLU!)
           and solves `(B_final, A_final)` (`rank_final=32`) at `Layer L` for exact Teacher logit parity!
        """
        t0 = time.perf_counter()
        _, _, mid_idx, norm_mod, lm_head = extract_mid_and_final_modules(student_model, dual_bank.mid_layer_idx)
        w_head = lm_head.weight.detach().float().cpu()
        b_head = lm_head.bias.detach().float().cpu() if getattr(lm_head, "bias", None) is not None else None
        v_slm, d = int(w_head.shape[0]), int(w_head.shape[1])
        v_lim = _infer_vocab_limit(student_model) or v_slm
        k_voc = max(2, min(int(top_vocab_k), v_slm, max(4, d // 6)))

        norm_w = getattr(norm_mod, "weight", None)
        if norm_w is not None:
            nw = norm_w.detach().float().cpu()
            inv_norm_scale = torch.sign(nw + 1e-8) / nw.abs().clamp(min=0.1)
        else:
            inv_norm_scale = torch.ones(d, dtype=torch.float32)

        mu_qm = dual_bank.mu_query_mid.detach().float().cpu()
        mu_hm = dual_bank.mu_hidden_mid.detach().float().cpu()
        mu_hf = dual_bank.mu_hidden_final.detach().float().cpu()
        um = dual_bank.u_preserved_mid.detach().float().cpu()
        uf = dual_bank.u_preserved_final.detach().float().cpu()
        p_null_m = torch.eye(d) - (um @ um.T) if um.numel() > 0 else torch.eye(d)
        p_null_f = torch.eye(d) - (uf @ uf.T) if uf.numel() > 0 else torch.eye(d)

        # --- Stage A: Capture clean mid + final states & solve `(B_mid, A_mid)` at Golden-Slice `L/2` ---
        q_mid_vecs, hm_rows, rm_rows = [], [], []
        with dual_bank.paused():
            for ep in episodes:
                lp, la = ep.prompt_len, ep.answer_len
                if la <= 0:
                    continue
                clamped = ep.full_ids.clamp(min=0, max=v_lim - 1).unsqueeze(0)
                h_mid, h_fin = cls._capture_mid_and_final_states(student_model, clamped, mid_idx)
                qs = min(max(0, int(ep.query_start_idx)), max(0, lp - 1))
                q_mid_vecs.append(F.normalize(h_mid[qs:lp].mean(dim=0) - mu_qm, dim=-1, eps=1e-8))

                h_fin_ans = h_fin[lp - 1 : lp - 1 + la]
                h_mid_ans = h_mid[lp - 1 : lp - 1 + la]
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
                    pulled = r_star * inv_norm_scale
                    scale_m = float(mid_steering_alpha) * float(h_mid_ans[step_i].norm().item()) / max(float(pulled.norm().item()), 1e-6)
                    r_mid = pulled * scale_m
                    hm_rows.append(((h_mid_ans[step_i] - mu_hm) @ p_null_m) * w_ep)
                    rm_rows.append(r_mid * w_ep)

        z_anchor_mid = F.normalize(torch.stack(q_mid_vecs, dim=0).mean(dim=0), dim=-1, eps=1e-8).to(torch.float16)
        B_mid, A_mid = cls._woodbury_dual_svd(
            hp_mat=torch.stack(hm_rows, dim=0),
            rc_mat=torch.stack(rm_rows, dim=0),
            p_null=p_null_m,
            rank=rank_mid,
            hidden_ridge_ratio=hidden_ridge_ratio,
        )

        # --- Stage B: Propagate `(B_mid, A_mid)` through Layers `L/2 .. L-1` & solve `(B_final, A_final)` ---
        dev = _infer_model_device(student_model)
        am_dev = A_mid.to(dev).float()
        bm_dev = B_mid.to(dev).float()
        mu_hm_dev = mu_hm.to(dev)

        hf_rows, rf_rows = [], []
        with dual_bank.paused():
            for ep in episodes:
                lp, la = ep.prompt_len, ep.answer_len
                if la <= 0:
                    continue
                clamped = ep.full_ids.clamp(min=0, max=v_lim - 1).unsqueeze(0)

                def _inject_mid(t3: torch.Tensor, _lp: int = lp) -> torch.Tensor:
                    t_mod = t3.clone()
                    sl = t3[:, _lp - 1 :, :].float()
                    hc = sl - mu_hm_dev.view(1, 1, -1)
                    d_m = torch.matmul(torch.matmul(hc, am_dev.T), bm_dev.T)
                    t_mod[:, _lp - 1 :, :] = (sl + d_m).to(t3.dtype)
                    return t_mod

                _, h_fin_prop = cls._capture_mid_and_final_states(student_model, clamped, mid_idx, mid_intervention=_inject_mid)
                h_fin_ans = h_fin_prop[lp - 1 : lp - 1 + la]
                lg_slm_prop = h_fin_ans @ w_head.T + (b_head.unsqueeze(0) if b_head is not None else 0.0)
                v_c = min(v_slm, int(ep.teacher_logits.shape[-1]))
                lg_t = ep.teacher_logits[:la, :v_c].float()
                k_eff = min(k_voc, v_c)
                w_ep = math.sqrt(max(float(ep.surprise_score), 1e-4))

                for step_i in range(la):
                    r_rem = cls._rosetta_cutting_plane(
                        w_head=w_head,
                        lg_teacher_1d=lg_t[step_i],
                        lg_student_1d=lg_slm_prop[step_i, :v_c],
                        k_eff=k_eff,
                        logit_gain=logit_gain,
                        vocab_ridge=vocab_ridge,
                    )
                    hf_rows.append(((h_fin_ans[step_i] - mu_hf) @ p_null_f) * w_ep)
                    rf_rows.append(r_rem * w_ep)

        B_final, A_final = cls._woodbury_dual_svd(
            hp_mat=torch.stack(hf_rows, dim=0),
            rc_mat=torch.stack(rf_rows, dim=0),
            p_null=p_null_f,
            rank=rank_final,
            hidden_ridge_ratio=hidden_ridge_ratio,
        )

        parents_tup = tuple(str(x) for x in parent_capsules)
        meta = dict(metadata or {})
        meta.update(
            {
                "mid_layer_idx": mid_idx,
                "rank_mid": int(B_mid.shape[1]),
                "rank_final": int(B_final.shape[1]),
                "solve_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "zero_backpropagation": True,
            }
        )
        if parents_tup:
            meta["parent_capsules"] = list(parents_tup)
        return DualDepthScarCapsule(
            capsule_id=capsule_id,
            z_anchor_mid=z_anchor_mid,
            A_mid=A_mid,
            B_mid=B_mid,
            A_final=A_final,
            B_final=B_final,
            tau=tau,
            ramp=ramp,
            parent_capsules=parents_tup,
            metadata=meta,
        )

    @staticmethod
    @torch.no_grad()
    def assess_interdisciplinary_epistemic_uncertainty(
        student_model: nn.Module,
        dual_bank: DualDepthScarBank,
        prompt_ids_2d: torch.Tensor,
        probe_steps: int = 6,
        margin_threshold: float = 0.55,
    ) -> BridgeEscalationDecision:
        """Evaluates `System 1` epistemic confidence & `p_1 - p_2` margin on multi-capsule synthesis queries."""
        t0 = time.perf_counter()
        dev = _infer_model_device(student_model)
        seq = prompt_ids_2d.to(dev).clone()
        lp = int(seq.shape[1])

        was_attached = dual_bank._norm_hook_handle is not None
        if not was_attached:
            dual_bank.attach(student_model)

        try:
            confs, ents, margins = [], [], []
            active_cids: List[str] = []
            with dual_bank.teacher_forcing_scope(prompt_len=lp):
                for step_i in range(max(1, probe_steps)):
                    try:
                        out = student_model(seq, use_cache=False)
                    except TypeError:
                        out = student_model(seq)
                    lg = (out.logits if hasattr(out, "logits") else out)[0, -1].float()
                    if step_i == 0:
                        active_cids = list(dual_bank.last_active_gates.keys())
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
            # Escalate when multiple unbridged capsules intersect under epistemic ambiguity (or zero confidence)
            is_unc = (
                (len(active_cids) >= 2 and (mean_margin < margin_threshold or min_margin < 0.35))
                or (mean_margin < 0.20)
            )
            return BridgeEscalationDecision(
                active_capsules=active_cids,
                confidence=round(mean_conf, 4),
                uncertainty=round(mean_ent, 4),
                margin=round(mean_margin, 4),
                is_uncertain=bool(is_unc),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
        finally:
            if not was_attached:
                dual_bank.detach()

    @classmethod
    @torch.no_grad()
    def consolidate_interdisciplinary_bridge(
        cls,
        bridge_capsule_id: str,
        bridge_episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        dual_bank: DualDepthScarBank,
        parent_capsules: Optional[Sequence[str]] = None,
        rank_mid: int = 32,
        rank_final: int = 32,
        tau: float = 0.60,
        ramp: float = 0.16,
    ) -> DualDepthScarCapsule:
        """Artificial Sleep Cycle consolidation of an interdisciplinary `System 2` synthesis trajectory
        into a permanent `61.25 KB` INT4 `DualDepthScarCapsule` mounted in `dual_bank`!
        """
        resolved_parents: Tuple[str, ...]
        if parent_capsules is not None:
            resolved_parents = tuple(str(x) for x in parent_capsules)
        elif bridge_episodes and dual_bank.capsules:
            ep0 = bridge_episodes[0]
            v_lim = _infer_vocab_limit(student_model) or int(ep0.full_ids.max().item()) + 1
            with dual_bank.paused():
                h_mid0, _ = cls._capture_mid_and_final_states(
                    student_model,
                    ep0.full_ids[: ep0.prompt_len].clamp(min=0, max=v_lim - 1).unsqueeze(0),
                    dual_bank.mid_layer_idx,
                )
            qs0 = min(max(0, int(ep0.query_start_idx)), max(0, ep0.prompt_len - 1))
            _ = dual_bank.route_mid_query(h_mid0[qs0 : ep0.prompt_len].to(dual_bank.mu_query_mid.device))
            resolved_parents = tuple(
                cid
                for cid, cos_v in dual_bank.last_routing_cosines.items()
                if not dual_bank.capsules[cid].parent_capsules
                and ((cos_v - dual_bank.capsules[cid].tau) / max(dual_bank.capsules[cid].ramp, 1e-6)) >= 0.50
            )
        else:
            resolved_parents = ()

        bridge_cap = cls.solve_dual_depth_capsule(
            capsule_id=bridge_capsule_id,
            episodes=bridge_episodes,
            student_model=student_model,
            dual_bank=dual_bank,
            rank_mid=rank_mid,
            rank_final=rank_final,
            tau=tau,
            ramp=ramp,
            parent_capsules=resolved_parents,
            metadata={"capsule_type": "interdisciplinary_bridge"},
        )
        dual_bank.add_capsule(bridge_cap)
        return bridge_cap
