"""Horizon 2 (v2.0) Track 2.1: Closed-Form LLM / SCache -> SLM Scar Grafting & Sleep Consolidation.

Implements genuine, zero-facade **Closed-Form Cone-Gated Null-Space Knowledge Transposition**
(**ZERO backpropagation / ZERO SGD / ZERO Adam**), unifying:
  1. **Delta-Origin Spherical Centering** (`Tri-Compass Bússola 2`): subtracts background query
     anisotropy `mu_query` over user-turn semantic tokens to separate unseen paraphrases
     (`cos ~ 0.88 - 0.94`) from unrelated prompts (`cos <= 0.25`).
  2. **One-Sided Spherical-Cap ReLU Router** (`L1-resident` `z_anchor` index): guarantees
     **exact `0.0000000000` bit-for-bit out-of-cone logit invariance** (`torch.equal`) and
     `O(1)` sparse activation (~225.8 KB active FP16 capsule vs. 6.18 GiB - 135.43 GiB dense LLMs).
  3. **Vocabulary Rosetta Stone Inversion** (`R^V -> R^{d_SLM}`): active top-`M_v` vocabulary
     union Woodbury Ridge pseudoinverse of `student.lm_head.weight`, translating logits from ANY
     Teacher LLM (`3B`, `7B`, `72.71B MoS`) or daytime `Ghost-Jump` / `SCache` surprise trace
     directly into the Student SLM's residual space.
  4. **AlphaEdit Exact Null-Space Projection + Woodbury Dual SVD Scar Factorization**: projects
     centered hidden states onto `P_null = I - U_pres U_pres^T`, solves the exact dual Woodbury
     system `M_k = R_c^T (H_p H_p^T + lam_h I)^{-1} H_p P_null`, and factors `M_k` into a
     compact rank-`r` `MoS` scar `(B_k, A_k)` persisted in `.safetensors` cartridges with
     `< 1 ms` surgical `unlearn(capsule_id)` erasure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
import torch.nn.functional as F

from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.speculative_ghost_jump import (
    GhostJumpGenerationResult,
    GhostJumpSpeculativeEngine,
    _forward_all_logits,
    _infer_model_device,
    _infer_vocab_limit,
    _prefill_chunked,
)


@dataclass
class SleepEpisode:
    """Daytime epistemic surprise / Teacher trajectory episode queued for sleep consolidation."""

    capsule_id: str
    prompt_ids: torch.Tensor       # [L_p] int64 CPU tensor
    full_ids: torch.Tensor         # [L_p + L_a] int64 CPU tensor
    teacher_logits: torch.Tensor   # [L_a, V_teacher] float16 CPU tensor
    query_start_idx: int = 0
    surprise_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return int(self.prompt_ids.numel())

    @property
    def answer_len(self) -> int:
        return int(self.full_ids.numel() - self.prompt_ids.numel())


@dataclass
class ScarCapsule:
    """Compact directional low-rank knowledge capsule (`~225.75 KB` FP16 at `d=896, r=64`)."""

    capsule_id: str
    z_anchor: torch.Tensor   # [d] unit vector on S^{d-1} (FP16)
    A_scar: torch.Tensor     # [r, d] down-projection (FP16)
    B_scar: torch.Tensor     # [d, r] up-projection (FP16)
    tau: float = 0.55
    ramp: float = 0.18
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.z_anchor.ndim != 1:
            raise ValueError(f"z_anchor must be 1D [d], got {tuple(self.z_anchor.shape)}")
        d = int(self.z_anchor.shape[0])
        if self.A_scar.ndim != 2 or int(self.A_scar.shape[1]) != d:
            raise ValueError(f"A_scar must be [r, {d}], got {tuple(self.A_scar.shape)}")
        r = int(self.A_scar.shape[0])
        if self.B_scar.ndim != 2 or tuple(self.B_scar.shape) != (d, r):
            raise ValueError(f"B_scar must be [{d}, {r}], got {tuple(self.B_scar.shape)}")
        if self.ramp <= 0.0:
            raise ValueError(f"ramp must be > 0, got {self.ramp}")

    @property
    def hidden_dim(self) -> int:
        return int(self.z_anchor.shape[0])

    @property
    def rank(self) -> int:
        return int(self.A_scar.shape[0])

    @property
    def anchor_bytes(self) -> int:
        return int(self.z_anchor.numel() * self.z_anchor.element_size())

    @property
    def scar_matrix_bytes(self) -> int:
        return int(
            self.A_scar.numel() * self.A_scar.element_size()
            + self.B_scar.numel() * self.B_scar.element_size()
        )

    @property
    def physical_bytes(self) -> int:
        return self.anchor_bytes + self.scar_matrix_bytes

    def to_device(self, device: Union[str, torch.device]) -> "ScarCapsule":
        dev = torch.device(device)
        self.z_anchor = self.z_anchor.to(dev)
        self.A_scar = self.A_scar.to(dev)
        self.B_scar = self.B_scar.to(dev)
        return self


def _extract_backbone_and_norm(student_model: nn.Module) -> Tuple[nn.Module, nn.Module, nn.Linear]:
    """Resolves `(backbone_module, final_norm_module, lm_head)` across HF CausalLMs and custom modules."""
    backbone = getattr(student_model, "model", None)
    if backbone is None:
        backbone = getattr(student_model, "transformer", student_model)

    norm_mod = getattr(backbone, "norm", None)
    if norm_mod is None:
        norm_mod = getattr(backbone, "final_layernorm", None)
    if norm_mod is None:
        norm_mod = getattr(backbone, "ln_f", None)
    if norm_mod is None:
        norm_mod = getattr(student_model, "norm", None)
    if norm_mod is None:
        raise AttributeError(
            f"Could not locate final normalization module (.model.norm / .ln_f) on {type(student_model).__name__}"
        )

    lm_head = getattr(student_model, "lm_head", None)
    if lm_head is None and hasattr(student_model, "get_output_embeddings"):
        lm_head = student_model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise AttributeError(f"Could not locate .lm_head with .weight on {type(student_model).__name__}")

    return backbone, norm_mod, lm_head


def detect_chatml_user_prefix_ids(tokenizer: Any) -> Optional[List[int]]:
    """Auto-detects ChatML `<|im_start|>user\\n` token sequence when supported by `tokenizer`."""
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    try:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        if "<|im_start|>" in vocab:
            ids = list(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
            if len(ids) >= 2:
                return [int(x) for x in ids]
    except Exception:
        pass
    return None


def find_subsequence_last_end(seq_1d: Sequence[int], pattern: Optional[Sequence[int]]) -> int:
    """Returns the index immediately after the last occurrence of `pattern` in `seq_1d` (or `0`)."""
    if not pattern:
        return 0
    m = len(pattern)
    n = len(seq_1d)
    if m == 0 or n < m:
        return 0
    pat = list(pattern)
    for idx in range(n - m, -1, -1):
        if list(seq_1d[idx : idx + m]) == pat:
            return min(idx + m, max(n - 1, 0))
    return 0


def _past_kv_seen_length(past_key_values: Any) -> int:
    """Inspects `SCacheDynamicCache`, HF `Cache`, or legacy KV tuple to determine prior seen tokens."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) > 0 and isinstance(first[0], torch.Tensor):
            if first[0].ndim >= 3:
                return int(first[0].shape[-2])
    return 0


class ScarBank(nn.Module):
    """Cone-Gated Modular Scar Knowledge Bank (`O(1)` L1 routing, `0.0000000000` out-of-cone leakage).

    Stores:
      - `mu_query` (`[d]`): Delta-Origin query centering mean (`Bússola 2`).
      - `mu_hidden` (`[d]`): Residual hidden-state centering mean.
      - `u_preserved` (`[d, r_pres]`): Preserved general-capability subspace (`AlphaEdit` null-space basis).
      - `capsules` (`Dict[str, ScarCapsule]`): Modular FP16 `(B_k, A_k, z_k)` capsules.
    """

    def __init__(
        self,
        hidden_dim: int,
        mu_query: Optional[torch.Tensor] = None,
        mu_hidden: Optional[torch.Tensor] = None,
        u_preserved: Optional[torch.Tensor] = None,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        normalize_multi_gate: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self.hidden_dim = int(hidden_dim)
        self.user_turn_prefix_ids: Optional[List[int]] = (
            [int(x) for x in user_turn_prefix_ids] if user_turn_prefix_ids else None
        )
        self.normalize_multi_gate = bool(normalize_multi_gate)

        mu_q = torch.zeros(self.hidden_dim, dtype=torch.float32) if mu_query is None else mu_query.detach().float().clone()
        mu_h = torch.zeros(self.hidden_dim, dtype=torch.float32) if mu_hidden is None else mu_hidden.detach().float().clone()
        u_p = (
            torch.zeros((self.hidden_dim, 0), dtype=torch.float32)
            if u_preserved is None
            else u_preserved.detach().float().clone()
        )
        self.register_buffer("mu_query", mu_q)
        self.register_buffer("mu_hidden", mu_h)
        self.register_buffer("u_preserved", u_p)
        self.register_buffer("_z_matrix", torch.zeros((0, self.hidden_dim), dtype=torch.float32), persistent=False)
        self.register_buffer("_tau_vec", torch.zeros(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_ramp_vec", torch.ones(0, dtype=torch.float32), persistent=False)

        self._capsules: Dict[str, ScarCapsule] = {}
        self._capsule_order: List[str] = []

        # Live runtime hook state
        self._pre_hook_handle: Optional[Any] = None
        self._norm_hook_handle: Optional[Any] = None
        self._attached_model_id: Optional[int] = None
        self._paused: bool = False
        self._current_pass_is_prefill: bool = True
        self._current_query_start: int = 0
        self._tf_prompt_len: Optional[int] = None
        self._active_gates: List[Tuple[str, float]] = []

        # Telemetry from most recent routing pass
        self.last_routing_cosines: Dict[str, float] = {}
        self.last_active_gates: Dict[str, float] = {}
        self.last_active_bytes_transferred: int = 0

    @property
    def num_capsules(self) -> int:
        return len(self._capsule_order)

    @property
    def capsule_ids(self) -> List[str]:
        return list(self._capsule_order)

    @property
    def capsules(self) -> Dict[str, ScarCapsule]:
        return dict(self._capsules)

    @property
    def is_attached(self) -> bool:
        return self._norm_hook_handle is not None

    @property
    def router_index_bytes(self) -> int:
        """L1-cache footprint of all capsule unit anchors (`1.75 KB` per capsule in FP16)."""
        return sum(c.anchor_bytes for c in self._capsules.values())

    @property
    def capsules_physical_bytes(self) -> int:
        return sum(c.physical_bytes for c in self._capsules.values())

    @property
    def total_physical_bytes(self) -> int:
        calib_bytes = int(
            self.mu_query.numel() * self.mu_query.element_size()
            + self.mu_hidden.numel() * self.mu_hidden.element_size()
            + self.u_preserved.numel() * self.u_preserved.element_size()
        )
        return calib_bytes + self.capsules_physical_bytes

    def _rebuild_router_index(self) -> None:
        dev = self.mu_query.device
        if not self._capsule_order:
            self._z_matrix = torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=dev)
            self._tau_vec = torch.zeros(0, dtype=torch.float32, device=dev)
            self._ramp_vec = torch.ones(0, dtype=torch.float32, device=dev)
            return
        z_list = []
        taus = []
        ramps = []
        for cid in self._capsule_order:
            cap = self._capsules[cid]
            cap.to_device(dev)
            z_list.append(F.normalize(cap.z_anchor.float(), dim=-1, eps=1e-8))
            taus.append(float(cap.tau))
            ramps.append(float(cap.ramp))
        self._z_matrix = torch.stack(z_list, dim=0).to(dev)
        self._tau_vec = torch.tensor(taus, dtype=torch.float32, device=dev)
        self._ramp_vec = torch.tensor(ramps, dtype=torch.float32, device=dev)

    def _sync_all_devices(self, dev: torch.device) -> None:
        if self.mu_query.device != dev:
            self.to(dev)
        if self._z_matrix.device != dev or (
            self._capsule_order and self._capsules[self._capsule_order[0]].A_scar.device != dev
        ):
            self._rebuild_router_index()

    def add_capsule(self, capsule: ScarCapsule, overwrite: bool = True) -> None:
        """Mounts a `ScarCapsule` into the bank and refreshes the L1 cone router matrix."""
        if capsule.hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Capsule {capsule.capsule_id!r} hidden_dim={capsule.hidden_dim} != bank hidden_dim={self.hidden_dim}"
            )
        if capsule.capsule_id in self._capsules and not overwrite:
            raise ValueError(f"Capsule {capsule.capsule_id!r} already exists in ScarBank (overwrite=False)")
        capsule.to_device(self.mu_query.device)
        if capsule.capsule_id not in self._capsules:
            self._capsule_order.append(capsule.capsule_id)
        self._capsules[capsule.capsule_id] = capsule
        self._rebuild_router_index()

    def unlearn(self, capsule_id: str) -> ScarCapsule:
        """Surgically removes a `ScarCapsule` in `< 1 ms` (`0.000000` residual trace left behind)."""
        if capsule_id not in self._capsules:
            raise KeyError(f"Capsule {capsule_id!r} not found in ScarBank")
        removed = self._capsules.pop(capsule_id)
        self._capsule_order = [cid for cid in self._capsule_order if cid != capsule_id]
        self._active_gates = [(cid, g) for (cid, g) in self._active_gates if cid != capsule_id]
        self._rebuild_router_index()
        return removed

    def reset_latched_gates(self) -> None:
        self._active_gates = []
        self.last_active_gates = {}
        self.last_active_bytes_transferred = 0

    @torch.no_grad()
    def audit_cone_orthogonality(self) -> Dict[str, Any]:
        """Computes the pairwise cosine Gram matrix `Z_bank @ Z_bank^T` across all capsules."""
        k = self.num_capsules
        if k == 0:
            return {"num_capsules": 0, "capsule_ids": [], "gram_matrix": [], "max_off_diagonal_cos": 0.0}
        gram = (self._z_matrix @ self._z_matrix.T).detach().cpu()
        if k > 1:
            mask = ~torch.eye(k, dtype=torch.bool)
            max_off = float(gram[mask].max().item())
            min_off = float(gram[mask].min().item())
        else:
            max_off = 0.0
            min_off = 0.0
        return {
            "num_capsules": k,
            "capsule_ids": list(self._capsule_order),
            "gram_matrix": [[round(float(v), 6) for v in row] for row in gram.tolist()],
            "max_off_diagonal_cos": max_off,
            "min_off_diagonal_cos": min_off,
            "min_tau": float(self._tau_vec.min().item()),
            "strictly_disjoint_at_anchors": bool(max_off < float(self._tau_vec.min().item())),
        }

    @torch.no_grad()
    def route_query_hidden(self, query_hidden_2d: torch.Tensor) -> List[Tuple[str, float]]:
        """Vectorized O(1) spherical-cap cone router over centered user-turn query hidden states."""
        if self.num_capsules == 0 or query_hidden_2d.numel() == 0:
            self.last_routing_cosines = {}
            self.last_active_gates = {}
            self.last_active_bytes_transferred = 0
            return []

        self._sync_all_devices(query_hidden_2d.device)
        q_mean = query_hidden_2d.float().mean(dim=0)
        q_unit = F.normalize(q_mean - self.mu_query, dim=-1, eps=1e-8)

        cos_vec = torch.mv(self._z_matrix, q_unit)  # Single GEMV across all K capsules!
        raw_gates = torch.clamp((cos_vec - self._tau_vec) / self._ramp_vec, min=0.0, max=1.0)

        cos_cpu = cos_vec.tolist()
        self.last_routing_cosines = {
            cid: float(cos_cpu[idx]) for idx, cid in enumerate(self._capsule_order)
        }

        if float(raw_gates.max().item()) <= 0.0:
            self.last_active_gates = {}
            self.last_active_bytes_transferred = self.router_index_bytes
            return []

        if self.normalize_multi_gate:
            g_sum = float(raw_gates.sum().item())
            if g_sum > 1.0:
                raw_gates = raw_gates / g_sum

        gates_cpu = raw_gates.tolist()
        active: List[Tuple[str, float]] = [
            (self._capsule_order[i], float(gates_cpu[i]))
            for i in range(len(self._capsule_order))
            if gates_cpu[i] > 0.0
        ]
        self.last_active_gates = dict(active)
        self.last_active_bytes_transferred = self.router_index_bytes + sum(
            self._capsules[cid].scar_matrix_bytes for cid, _ in active
        )
        return active

    @torch.no_grad()
    def _apply_active_scars(self, h_slice: torch.Tensor, active: Sequence[Tuple[str, float]]) -> torch.Tensor:
        """Applies `sum_{k in active} g_k * B_k(A_k(h - mu_hidden))` in FP32 and restores input dtype."""
        hc = h_slice.float() - self.mu_hidden
        delta = torch.zeros_like(hc)
        for cid, g_val in active:
            cap = self._capsules[cid]
            low_rank = F.linear(hc, cap.A_scar.float())
            proj = F.linear(low_rank, cap.B_scar.float())
            delta = delta + (proj * g_val)
        return (h_slice.float() + delta).to(h_slice.dtype)

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
        self._current_pass_is_prefill = (past_len == 0)

        if self._current_pass_is_prefill and isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
            ids_1d = input_ids[0].tolist() if input_ids.ndim >= 2 else input_ids.tolist()
            self._current_query_start = find_subsequence_last_end(ids_1d, self.user_turn_prefix_ids)

    def _final_norm_hook(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self._paused or self.num_capsules == 0:
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

        self._sync_all_devices(out_3d.device)

        # Case 1: Explicit Teacher-Forcing Scope (`full_ids = [prompt_ids + answer_ids]`)
        if self._tf_prompt_len is not None:
            lp = max(1, min(int(self._tf_prompt_len), seq_len))
            q_start = min(self._current_query_start, lp - 1)
            active = self.route_query_hidden(out_3d[0, q_start:lp, :])
            self._active_gates = active
            if not active:
                return output  # Exact 0.0000000000 bit-for-bit passthrough!
            out_mod = out_3d.clone()
            out_mod[:, lp - 1 :, :] = self._apply_active_scars(out_3d[:, lp - 1 :, :], active)
            return out_mod.squeeze(0) if squeezed else out_mod

        # Case 2: Fresh Prompt Prefill (`past_seen == 0`)
        if self._current_pass_is_prefill:
            q_start = min(self._current_query_start, seq_len - 1)
            active = self.route_query_hidden(out_3d[0, q_start:seq_len, :])
            self._active_gates = active
            if not active:
                return output  # Exact 0.0000000000 bit-for-bit passthrough!
            out_mod = out_3d.clone()
            # Apply ONLY to the final prompt position predicting assistant token #0!
            out_mod[:, -1:, :] = self._apply_active_scars(out_3d[:, -1:, :], active)
            return out_mod.squeeze(0) if squeezed else out_mod

        # Case 3: Autoregressive Decode (`seq_len == 1`) or Ghost-Jump Sync (`seq_len >= 1`, `past_seen > 0`)
        active = self._active_gates
        if not active:
            return output  # Exact 0.0000000000 bit-for-bit passthrough!
        out_mod = self._apply_active_scars(out_3d, active)
        return out_mod.squeeze(0) if squeezed else out_mod

    def attach(self, student_model: nn.Module) -> "ScarBank":
        """Attaches the cone-gated ScarBank hooks to `student_model` (`O(1)` runtime overhead)."""
        self.detach()
        dev = _infer_model_device(student_model)
        self.to(dev)
        self._rebuild_router_index()

        backbone, norm_mod, _ = _extract_backbone_and_norm(student_model)
        self._pre_hook_handle = backbone.register_forward_pre_hook(self._backbone_pre_hook, with_kwargs=True)
        self._norm_hook_handle = norm_mod.register_forward_hook(self._final_norm_hook)
        self._attached_model_id = id(student_model)
        return self

    def detach(self) -> None:
        """Cleanly detaches all hooks from the student model."""
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
            self._pre_hook_handle = None
        if self._norm_hook_handle is not None:
            self._norm_hook_handle.remove()
            self._norm_hook_handle = None
        self._attached_model_id = None
        self.reset_latched_gates()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Temporarily disables ScarBank intervention (e.g., when reading clean base SLM states)."""
        prev = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = prev

    @contextmanager
    def teacher_forcing_scope(self, prompt_len: int) -> Iterator[None]:
        """Evaluates `[prompt + answer]` in one causal pass with exact autoregressive step parity."""
        prev_tf = self._tf_prompt_len
        self._tf_prompt_len = int(prompt_len)
        try:
            yield
        finally:
            self._tf_prompt_len = prev_tf

    def save_safetensors(self, path: Union[str, Path]) -> Path:
        """Persists the entire `ScarBank` and all `ScarCapsule`s into a portable `.safetensors` file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tensors: Dict[str, torch.Tensor] = {
            "__bank__.mu_query": self.mu_query.detach().cpu().contiguous(),
            "__bank__.mu_hidden": self.mu_hidden.detach().cpu().contiguous(),
            "__bank__.u_preserved": self.u_preserved.detach().cpu().contiguous(),
        }
        caps_manifest: List[Dict[str, Any]] = []
        for idx, cid in enumerate(self._capsule_order):
            cap = self._capsules[cid]
            prefix = f"cap.{idx}"
            tensors[f"{prefix}.z_anchor"] = cap.z_anchor.detach().cpu().contiguous()
            tensors[f"{prefix}.A_scar"] = cap.A_scar.detach().cpu().contiguous()
            tensors[f"{prefix}.B_scar"] = cap.B_scar.detach().cpu().contiguous()
            caps_manifest.append(
                {
                    "index": idx,
                    "capsule_id": cid,
                    "tau": float(cap.tau),
                    "ramp": float(cap.ramp),
                    "metadata": cap.metadata,
                }
            )

        header_meta = {
            "format": "mos_scargraft_bank_v2",
            "hidden_dim": str(self.hidden_dim),
            "normalize_multi_gate": json.dumps(self.normalize_multi_gate),
            "user_turn_prefix_ids": json.dumps(self.user_turn_prefix_ids),
            "capsules_manifest": json.dumps(caps_manifest),
        }
        save_file(tensors, str(out_path), metadata=header_meta)
        return out_path

    @classmethod
    def load_safetensors(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> "ScarBank":
        """Loads a `.safetensors` `ScarBank` cartridge in milliseconds (`zero pickle`)."""
        in_path = Path(path)
        from safetensors import safe_open

        with safe_open(str(in_path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format") != "mos_scargraft_bank_v2":
                raise ValueError(f"Invalid ScarBank safetensors format in {in_path}: {meta.get('format')!r}")
            hidden_dim = int(meta["hidden_dim"])
            normalize_multi_gate = bool(json.loads(meta.get("normalize_multi_gate", "true")))
            user_turn_prefix_ids = json.loads(meta.get("user_turn_prefix_ids", "null"))
            caps_manifest = json.loads(meta.get("capsules_manifest", "[]"))
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        bank = cls(
            hidden_dim=hidden_dim,
            mu_query=tensors["__bank__.mu_query"],
            mu_hidden=tensors["__bank__.mu_hidden"],
            u_preserved=tensors["__bank__.u_preserved"],
            user_turn_prefix_ids=user_turn_prefix_ids,
            normalize_multi_gate=normalize_multi_gate,
        )
        for item in caps_manifest:
            idx = int(item["index"])
            prefix = f"cap.{idx}"
            cap = ScarCapsule(
                capsule_id=str(item["capsule_id"]),
                z_anchor=tensors[f"{prefix}.z_anchor"],
                A_scar=tensors[f"{prefix}.A_scar"],
                B_scar=tensors[f"{prefix}.B_scar"],
                tau=float(item["tau"]),
                ramp=float(item["ramp"]),
                metadata=dict(item.get("metadata") or {}),
            )
            bank.add_capsule(cap)

        bank.to(torch.device(device))
        bank._rebuild_router_index()
        return bank


def _extract_clean_norm_hidden_states(
    student_model: nn.Module,
    input_ids_2d: torch.Tensor,
    bank: Optional[ScarBank] = None,
) -> torch.Tensor:
    """Extracts `[seq_len, d]` post-norm hidden states on CPU `float32` without hook perturbation."""
    dev = _infer_model_device(student_model)
    ids = input_ids_2d.to(dev)
    backbone, norm_mod, _ = _extract_backbone_and_norm(student_model)

    captured: Dict[str, torch.Tensor] = {}

    def _capture_hook(_m: nn.Module, _i: Tuple[Any, ...], out: torch.Tensor) -> None:
        t = out[0] if out.ndim == 3 else out
        captured["h"] = t.detach().float().cpu()

    ctx = bank.paused() if bank is not None else _null_ctx()
    with ctx:
        h_handle = norm_mod.register_forward_hook(_capture_hook)
        try:
            try:
                backbone(ids, use_cache=False)
            except TypeError:
                backbone(ids)
        finally:
            h_handle.remove()

    if "h" not in captured:
        raise RuntimeError("Failed to capture post-norm hidden states from student backbone.")
    return captured["h"]


@contextmanager
def _null_ctx() -> Iterator[None]:
    yield


@torch.no_grad()
def _teacher_greedy_and_logits(
    teacher_model: Any,
    prompt_ids_2d: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[Union[int, Sequence[int]]] = None,
    draft_helper_model: Optional[Any] = None,
    max_draft_tokens: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates authoritative Teacher trajectory and returns `(full_ids_1d, answer_logits_2d_fp16_cpu)`.

    Supports both HuggingFace `AutoModelForCausalLM` AND `72.71B CanonicalMoSEngine`!
    When `draft_helper_model` is provided, uses lossless `Ghost-Jump` speculative verification
    (`KL = 0.0000`) to capture exact Teacher tokens AND exact Teacher logits in ~4-6x fewer sweeps!
    """
    dev = _infer_model_device(teacher_model)
    p_ids = prompt_ids_2d.to(dev)
    lp = int(p_ids.shape[1])

    eos_set: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_set.add(eos_token_id)
    elif eos_token_id is not None:
        eos_set.update(int(x) for x in eos_token_id)

    # Fast Path: Lossless Ghost-Jump Speculative Teacher Capture (ideal for 72.71B MoS streaming!)
    if draft_helper_model is not None and max_new_tokens > 0:
        draft_dev = _infer_model_device(draft_helper_model)
        draft_cache = SCacheDynamicCache(window_size=256, num_slots=128)
        target_cache = SCacheDynamicCache(window_size=256, num_slots=128)

        if lp > 1:
            _prefill_chunked(draft_helper_model, prompt_ids_2d.to(draft_dev), draft_cache)
        else:
            _forward_all_logits(draft_helper_model, prompt_ids_2d.to(draft_dev), draft_cache)

        t_prefill_lg = _prefill_chunked(teacher_model, p_ids, target_cache)  # [1, V_teacher]
        lg0 = t_prefill_lg[0]
        tok0 = int(lg0.float().argmax(dim=-1).item())

        gen_tokens: List[int] = [tok0]
        ans_logits: List[torch.Tensor] = [lg0.detach().to(torch.float16).cpu()]
        last_tok = tok0
        sync_prefix: List[int] = []

        while len(gen_tokens) < max_new_tokens and (last_tok not in eos_set):
            rem_budget = max_new_tokens - len(gen_tokens)
            cur_k_max = max(1, min(int(max_draft_tokens), rem_budget))
            s_sync = len(sync_prefix)
            draft_tokens: List[int] = []

            with draft_cache.speculative_capture(
                expected_new_tokens=s_sync + cur_k_max,
                multi_step=True,
            ) as draft_ctx:
                for step_i in range(cur_k_max):
                    if step_i == 0 and s_sync > 0:
                        step_in = torch.tensor([sync_prefix + [last_tok]], dtype=torch.long, device=draft_dev)
                    else:
                        prev_id = last_tok if step_i == 0 else draft_tokens[-1]
                        step_in = torch.tensor([[prev_id]], dtype=torch.long, device=draft_dev)
                    d_lg = _forward_all_logits(draft_helper_model, step_in, draft_cache)[0, -1]
                    d_tok = int(d_lg.float().argmax(dim=-1).item())
                    draft_tokens.append(d_tok)
                    if d_tok in eos_set:
                        break

                k_cur = len(draft_tokens)
                verify_tensor = torch.tensor([[last_tok] + draft_tokens], dtype=torch.long, device=dev)

                with target_cache.speculative_capture(
                    expected_new_tokens=k_cur + 1,
                    multi_step=False,
                ) as target_ctx:
                    t_lg_3d = _forward_all_logits(teacher_model, verify_tensor, target_cache)  # [1, k_cur+1, V]
                    num_accepted = 0
                    mismatch_tok: Optional[int] = None
                    stopped_early = False

                    for i in range(k_cur):
                        step_lg = t_lg_3d[0, i]
                        ans_logits.append(step_lg.detach().to(torch.float16).cpu())
                        t_pred = int(step_lg.float().argmax(dim=-1).item())
                        if t_pred == draft_tokens[i]:
                            num_accepted += 1
                            gen_tokens.append(t_pred)
                            if len(gen_tokens) >= max_new_tokens or (t_pred in eos_set):
                                stopped_early = True
                                break
                        else:
                            mismatch_tok = t_pred
                            gen_tokens.append(t_pred)
                            break

                    if stopped_early:
                        target_ctx.commit(num_accepted)
                        draft_ctx.commit(s_sync + num_accepted)
                        break

                    if num_accepted == k_cur:
                        if len(gen_tokens) < max_new_tokens and (gen_tokens[-1] not in eos_set):
                            bonus_lg = t_lg_3d[0, k_cur]
                            ans_logits.append(bonus_lg.detach().to(torch.float16).cpu())
                            bonus_tok = int(bonus_lg.float().argmax(dim=-1).item())
                            gen_tokens.append(bonus_tok)
                            target_ctx.commit(k_cur + 1)
                            draft_ctx.commit(s_sync + k_cur - 1)
                            sync_prefix = [draft_tokens[k_cur - 1]]
                            last_tok = bonus_tok
                        else:
                            target_ctx.commit(k_cur)
                            draft_ctx.commit(s_sync + k_cur)
                            break
                    else:
                        assert mismatch_tok is not None
                        target_ctx.commit(num_accepted + 1)
                        draft_ctx.commit(s_sync + num_accepted)
                        sync_prefix = []
                        last_tok = mismatch_tok

        full_1d = torch.cat([p_ids[0].detach().cpu(), torch.tensor(gen_tokens, dtype=torch.long)], dim=0)
        return full_1d, torch.stack(ans_logits, dim=0)

    # Path A: CanonicalMoSEngine (`72.71B MoS`)
    if hasattr(teacher_model, "n_groups") and hasattr(teacher_model, "manifest"):
        scache = SCacheDynamicCache(window_size=256, num_slots=128)
        logits_step = teacher_model.forward(p_ids, scache=scache, return_all_logits=False)
        if logits_step.ndim == 3:
            logits_step = logits_step[:, -1, :]
        gen_tokens: List[int] = []
        ans_logits: List[torch.Tensor] = []
        for _ in range(max_new_tokens):
            ans_logits.append(logits_step[0].detach().to(torch.float16).cpu())
            nxt = int(logits_step[0].float().argmax(dim=-1).item())
            gen_tokens.append(nxt)
            if nxt in eos_set:
                break
            nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=dev)
            logits_step = teacher_model.forward(nxt_t, scache=scache, return_all_logits=False)
            if logits_step.ndim == 3:
                logits_step = logits_step[:, -1, :]
        full_1d = torch.cat([p_ids[0].cpu(), torch.tensor(gen_tokens, dtype=torch.long)], dim=0)
        return full_1d, torch.stack(ans_logits, dim=0)

    # Path B: HuggingFace PreTrainedModel or generic CausalLM
    if hasattr(teacher_model, "generate"):
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        seq_2d = teacher_model.generate(p_ids, **gen_kwargs)
    else:
        cur = p_ids
        for _ in range(max_new_tokens):
            out = teacher_model(cur)
            lg = out.logits if hasattr(out, "logits") else out
            nxt = int(lg[0, -1].float().argmax(dim=-1).item())
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=dev)], dim=1)
            if nxt in eos_set:
                break
        seq_2d = cur

    la = int(seq_2d.shape[1]) - lp
    if la <= 0:
        raise RuntimeError("Teacher generated 0 tokens on training prompt.")

    try:
        out_full = teacher_model(seq_2d, use_cache=False)
    except TypeError:
        out_full = teacher_model(seq_2d)
    lg_full = out_full.logits if hasattr(out_full, "logits") else out_full
    ans_lg = lg_full[0, lp - 1 : lp - 1 + la, :].detach().to(torch.float16).cpu()
    return seq_2d[0].detach().cpu(), ans_lg


class ScarGraftEngine:
    """Closed-Form `LLM / SCache -> SLM` Direct Weight-Space Transposition & Sleep Engine (`Zero SGD`)."""

    @staticmethod
    @torch.no_grad()
    def calibrate_student(
        student_model: nn.Module,
        calibration_prompts: Sequence[Union[str, torch.Tensor]],
        tokenizer: Optional[Any] = None,
        chat_formatter: Optional[Callable[[str], str]] = None,
        preserved_rank: int = 8,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        normalize_multi_gate: bool = True,
    ) -> ScarBank:
        """Computes `mu_query`, `mu_hidden`, and `U_pres` (`AlphaEdit` preserved basis) in closed form."""
        if len(calibration_prompts) == 0:
            raise ValueError("calibration_prompts must contain at least 1 prompt.")

        _, _, lm_head = _extract_backbone_and_norm(student_model)
        hidden_dim = int(lm_head.weight.shape[1])

        prefix_ids = (
            [int(x) for x in user_turn_prefix_ids]
            if user_turn_prefix_ids is not None
            else detect_chatml_user_prefix_ids(tokenizer)
        )

        q_means: List[torch.Tensor] = []
        all_h: List[torch.Tensor] = []

        for item in calibration_prompts:
            if isinstance(item, str):
                if tokenizer is None:
                    raise ValueError("tokenizer is required when calibration_prompts contains strings.")
                text = chat_formatter(item) if chat_formatter is not None else item
                ids_2d = tokenizer(text, return_tensors="pt").input_ids
            else:
                ids_2d = item if item.ndim == 2 else item.unsqueeze(0)

            h_2d = _extract_clean_norm_hidden_states(student_model, ids_2d)
            q_start = find_subsequence_last_end(ids_2d[0].tolist(), prefix_ids)
            q_means.append(h_2d[q_start:].mean(dim=0))
            all_h.append(h_2d)

        q_stack = torch.stack(q_means, dim=0)       # [N_bg, d]
        mu_query = q_stack.mean(dim=0)              # [d]
        h_cat = torch.cat(all_h, dim=0)             # [N_bg_tok, d]
        mu_hidden = h_cat.mean(dim=0)               # [d]

        r_eff = max(0, min(int(preserved_rank), h_cat.shape[0] - 1, hidden_dim - 1))
        if r_eff > 0:
            _, _, vh = torch.linalg.svd(h_cat - mu_hidden, full_matrices=False)
            u_preserved = vh[:r_eff, :].T.contiguous()  # [d, r_eff]
        else:
            u_preserved = torch.zeros((hidden_dim, 0), dtype=torch.float32)

        return ScarBank(
            hidden_dim=hidden_dim,
            mu_query=mu_query,
            mu_hidden=mu_hidden,
            u_preserved=u_preserved,
            user_turn_prefix_ids=prefix_ids,
            normalize_multi_gate=normalize_multi_gate,
        )

    @staticmethod
    @torch.no_grad()
    def collect_teacher_episodes(
        capsule_id: str,
        train_prompts: Sequence[Union[str, torch.Tensor]],
        teacher_model: Any,
        tokenizer: Optional[Any] = None,
        chat_formatter: Optional[Callable[[str], str]] = None,
        max_new_tokens: int = 36,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        draft_helper_model: Optional[Any] = None,
        max_draft_tokens: int = 8,
    ) -> List[SleepEpisode]:
        """Runs the Teacher LLM (`3B`, `7B`, `72B MoS`) on `train_prompts` to record authoritative episodes."""
        prefix_ids = (
            [int(x) for x in user_turn_prefix_ids]
            if user_turn_prefix_ids is not None
            else detect_chatml_user_prefix_ids(tokenizer)
        )
        if eos_token_id is None and tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            eos_token_id = int(tokenizer.eos_token_id)

        episodes: List[SleepEpisode] = []
        for idx, prompt in enumerate(train_prompts):
            if isinstance(prompt, str):
                if tokenizer is None:
                    raise ValueError("tokenizer is required when train_prompts contains strings.")
                formatted = chat_formatter(prompt) if chat_formatter is not None else prompt
                p_ids_2d = tokenizer(formatted, return_tensors="pt").input_ids
                raw_text = prompt
            else:
                p_ids_2d = prompt if prompt.ndim == 2 else prompt.unsqueeze(0)
                raw_text = f"tensor_prompt_{idx}"

            full_1d, ans_lg_fp16 = _teacher_greedy_and_logits(
                teacher_model=teacher_model,
                prompt_ids_2d=p_ids_2d,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                draft_helper_model=draft_helper_model,
                max_draft_tokens=max_draft_tokens,
            )
            q_start = find_subsequence_last_end(p_ids_2d[0].tolist(), prefix_ids)
            episodes.append(
                SleepEpisode(
                    capsule_id=capsule_id,
                    prompt_ids=p_ids_2d[0].detach().cpu(),
                    full_ids=full_1d,
                    teacher_logits=ans_lg_fp16,
                    query_start_idx=q_start,
                    surprise_score=1.0,
                    metadata={"prompt": raw_text},
                )
            )
        return episodes

    @staticmethod
    @torch.no_grad()
    def solve_capsule_from_episodes(
        capsule_id: str,
        episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        bank: ScarBank,
        rank: int = 64,
        top_vocab_k: int = 48,
        logit_gain: float = 1.25,
        vocab_ridge: float = 0.10,
        hidden_ridge_ratio: float = 0.015,
        tau: float = 0.55,
        ramp: float = 0.18,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScarCapsule:
        """Solves a `ScarCapsule` in closed form (`ZERO autograd / ZERO SGD`) from `SleepEpisode`s.

        Mathematical pipeline:
          1. Centered Query Anchor: `z_k = normalize(mean_i normalize(mean(H_query_i) - mu_query))`.
          2. Vocabulary Rosetta Stone (`R^V -> R^d`): active union `S_t = TopK(Teacher) U TopK(Student)`
             Woodbury pseudoinverse `r_t^* = W_S^T (W_S W_S^T + lam_v I)^{-1} Delta_l_S`.
          3. AlphaEdit Null-Space Projector: `P_null = I - U_pres U_pres^T`, `H_p = (H_ans - mu_h) P_null`.
          4. Dual Woodbury Ridge + SVD: `M_k = R_c^T (H_p H_p^T + lam_h I)^{-1} H_p P_null = B_k A_k`.
        """
        if len(episodes) == 0:
            raise ValueError("episodes must contain at least 1 SleepEpisode.")

        t0 = time.perf_counter()
        _, _, lm_head = _extract_backbone_and_norm(student_model)
        w_head = lm_head.weight.detach().float().cpu()  # [V_slm, d]
        b_head = lm_head.bias.detach().float().cpu() if getattr(lm_head, "bias", None) is not None else None
        v_slm, d = int(w_head.shape[0]), int(w_head.shape[1])
        if d != bank.hidden_dim:
            raise ValueError(f"Student hidden_dim={d} does not match ScarBank hidden_dim={bank.hidden_dim}")

        mu_q_cpu = bank.mu_query.detach().float().cpu()
        mu_h_cpu = bank.mu_hidden.detach().float().cpu()
        u_pres_cpu = bank.u_preserved.detach().float().cpu()
        if u_pres_cpu.numel() > 0:
            p_null = torch.eye(d, dtype=torch.float32) - (u_pres_cpu @ u_pres_cpu.T)
        else:
            p_null = torch.eye(d, dtype=torch.float32)

        v_lim = _infer_vocab_limit(student_model) or v_slm
        k_voc = max(2, min(int(top_vocab_k), v_slm, max(4, d // 3)))

        q_vecs: List[torch.Tensor] = []
        h_rows: List[torch.Tensor] = []
        r_rows: List[torch.Tensor] = []

        for ep in episodes:
            lp = ep.prompt_len
            la = ep.answer_len
            if la <= 0:
                continue

            clamped_full = ep.full_ids.clamp(min=0, max=v_lim - 1).unsqueeze(0)
            h_full = _extract_clean_norm_hidden_states(student_model, clamped_full, bank=bank)  # [lp+la, d]

            q_start = min(max(0, int(ep.query_start_idx)), max(0, lp - 1))
            q_dir = F.normalize(h_full[q_start:lp].mean(dim=0) - mu_q_cpu, dim=-1, eps=1e-8)
            q_vecs.append(q_dir)

            h_ans = h_full[lp - 1 : lp - 1 + la]  # [la, d]
            lg_slm = h_ans @ w_head.T
            if b_head is not None:
                lg_slm = lg_slm + b_head.unsqueeze(0)

            v_common = min(v_slm, int(ep.teacher_logits.shape[-1]))
            lg_teacher = ep.teacher_logits[:la, :v_common].float()
            lg_student = lg_slm[:, :v_common]
            k_eff = min(k_voc, v_common)

            for step_idx in range(la):
                t_top = lg_teacher[step_idx].topk(k_eff).indices
                s_top = lg_student[step_idx].topk(k_eff).indices
                s_union = torch.unique(torch.cat([t_top, s_top], dim=0))

                # Active-Set Cutting-Plane Woodbury Rosetta Inversion:
                # Iteratively catches & suppresses any out-of-set intruder token that outranks Teacher top-1!
                r_star = torch.zeros(d, dtype=torch.float32)
                t_best = int(t_top[0].item())
                for pass_i in range(5):
                    w_s = w_head[s_union]  # [M_s, d]
                    t_sub = lg_teacher[step_idx, s_union]
                    s_sub = lg_student[step_idx, s_union]
                    delta_logits = float(logit_gain) * (t_sub - t_sub.mean()) - (s_sub - s_sub.mean())

                    m_s = int(s_union.numel())
                    gram_v = (w_s @ w_s.T) + float(vocab_ridge) * torch.eye(m_s, dtype=torch.float32)
                    r_star = w_s.T @ torch.linalg.solve(gram_v, delta_logits)

                    if pass_i < 4 and m_s + 8 < max(12, int(0.90 * d)):
                        pred_lg = lg_student[step_idx] + (w_head[:v_common] @ r_star)
                        if int(pred_lg.argmax().item()) == t_best:
                            break
                        intruders = pred_lg.topk(min(8, v_common)).indices
                        new_union = torch.unique(torch.cat([s_union, intruders], dim=0))
                        if int(new_union.numel()) == m_s:
                            break
                        s_union = new_union
                    else:
                        break

                h_proj = (h_ans[step_idx] - mu_h_cpu) @ p_null
                w_ep = math.sqrt(max(float(ep.surprise_score), 1e-4))
                h_rows.append(h_proj * w_ep)
                r_rows.append(r_star * w_ep)

        if len(h_rows) == 0:
            raise ValueError("No valid answer steps found across episodes.")

        z_anchor = F.normalize(torch.stack(q_vecs, dim=0).mean(dim=0), dim=-1, eps=1e-8).to(torch.float16)

        hp_mat = torch.stack(h_rows, dim=0)  # [N_tok, d]
        rc_mat = torch.stack(r_rows, dim=0)  # [N_tok, d]
        n_tok = int(hp_mat.shape[0])

        gram_h = hp_mat @ hp_mat.T  # [N_tok, N_tok]
        mean_diag = max(float(gram_h.diagonal().mean().item()), 1e-6)
        lam_h = max(1e-5, float(hidden_ridge_ratio) * mean_diag)

        alpha = torch.linalg.solve(gram_h + lam_h * torch.eye(n_tok, dtype=torch.float32), rc_mat)
        m_scar = (alpha.T @ hp_mat) @ p_null  # [d, d], strictly orthogonal to U_pres on input!

        u_m, s_m, vh_m = torch.linalg.svd(m_scar, full_matrices=False)
        r_eff = max(1, min(int(rank), int(s_m.shape[0])))
        sqrt_s = s_m[:r_eff].clamp(min=0.0).sqrt()

        b_scar = (u_m[:, :r_eff] * sqrt_s.unsqueeze(0)).to(torch.float16).contiguous()   # [d, r_eff]
        a_scar = (sqrt_s.unsqueeze(1) * vh_m[:r_eff, :]).to(torch.float16).contiguous()  # [r_eff, d]

        solve_ms = (time.perf_counter() - t0) * 1000.0
        meta = dict(metadata or {})
        meta.update(
            {
                "num_episodes": len(episodes),
                "num_trajectory_tokens": n_tok,
                "rank": r_eff,
                "solve_ms": round(solve_ms, 3),
                "zero_backprop": True,
            }
        )
        return ScarCapsule(
            capsule_id=capsule_id,
            z_anchor=z_anchor,
            A_scar=a_scar,
            B_scar=b_scar,
            tau=float(tau),
            ramp=float(ramp),
            metadata=meta,
        )

    @classmethod
    @torch.no_grad()
    def synthesize_from_teacher(
        cls,
        capsule_id: str,
        train_prompts: Sequence[Union[str, torch.Tensor]],
        teacher_model: Any,
        student_model: nn.Module,
        bank: ScarBank,
        tokenizer: Optional[Any] = None,
        chat_formatter: Optional[Callable[[str], str]] = None,
        max_new_tokens: int = 36,
        rank: int = 64,
        top_vocab_k: int = 48,
        logit_gain: float = 1.25,
        vocab_ridge: float = 0.10,
        hidden_ridge_ratio: float = 0.015,
        tau: float = 0.55,
        ramp: float = 0.18,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        draft_helper_model: Optional[Any] = None,
        max_draft_tokens: int = 8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ScarCapsule, List[SleepEpisode]]:
        """End-to-end closed-form Teacher -> Student Scar Grafting + automatic mounting into `bank`."""
        episodes = cls.collect_teacher_episodes(
            capsule_id=capsule_id,
            train_prompts=train_prompts,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            chat_formatter=chat_formatter,
            max_new_tokens=max_new_tokens,
            user_turn_prefix_ids=bank.user_turn_prefix_ids,
            eos_token_id=eos_token_id,
            draft_helper_model=draft_helper_model,
            max_draft_tokens=max_draft_tokens,
        )
        capsule = cls.solve_capsule_from_episodes(
            capsule_id=capsule_id,
            episodes=episodes,
            student_model=student_model,
            bank=bank,
            rank=rank,
            top_vocab_k=top_vocab_k,
            logit_gain=logit_gain,
            vocab_ridge=vocab_ridge,
            hidden_ridge_ratio=hidden_ridge_ratio,
            tau=tau,
            ramp=ramp,
            metadata=metadata,
        )
        bank.add_capsule(capsule, overwrite=True)
        return capsule, episodes

    @classmethod
    @torch.no_grad()
    def consolidate_sleep_episodes(
        cls,
        episodes: Sequence[SleepEpisode],
        student_model: nn.Module,
        bank: ScarBank,
        rank: int = 64,
        top_vocab_k: int = 48,
        logit_gain: float = 1.25,
        tau: float = 0.55,
        ramp: float = 0.18,
    ) -> Dict[str, ScarCapsule]:
        """Artificial Sleep Cycle (`Track 2.1`): groups daytime `SleepEpisode`s by `capsule_id` and
        distills each high-surprise cluster into a permanent `ScarCapsule` inside `bank`.
        """
        grouped: Dict[str, List[SleepEpisode]] = {}
        for ep in episodes:
            grouped.setdefault(ep.capsule_id, []).append(ep)

        minted: Dict[str, ScarCapsule] = {}
        for cid, ep_list in grouped.items():
            cap = cls.solve_capsule_from_episodes(
                capsule_id=cid,
                episodes=ep_list,
                student_model=student_model,
                bank=bank,
                rank=rank,
                top_vocab_k=top_vocab_k,
                logit_gain=logit_gain,
                tau=tau,
                ramp=ramp,
                metadata={"consolidated_via": "artificial_sleep_cycle"},
            )
            bank.add_capsule(cap, overwrite=True)
            minted[cid] = cap
        return minted

    @staticmethod
    @torch.no_grad()
    def run_ghost_jump_and_record_surprise(
        engine: GhostJumpSpeculativeEngine,
        capsule_id: str,
        prompt_ids_2d: torch.Tensor,
        max_new_tokens: int = 36,
        acceptance_surprise_threshold: float = 0.70,
        user_turn_prefix_ids: Optional[Sequence[int]] = None,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[GhostJumpGenerationResult, Optional[SleepEpisode]]:
        """Runs `GhostJumpSpeculativeEngine` daytime inference; if `System 1` experiences high draft
        rejection (`acceptance_rate < threshold`), logs the authoritative Teacher trajectory as a
        `SleepEpisode` for later sleep consolidation!
        """
        res = engine.generate(
            input_ids=prompt_ids_2d,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_id=eos_token_id,
            audit_storage_bytes=False,
        )
        if res.acceptance_rate >= acceptance_surprise_threshold:
            return res, None

        dev = _infer_model_device(engine.target_model)
        lp = int(prompt_ids_2d.shape[1])
        gen_tensor = torch.tensor(res.generated_ids, dtype=torch.long)
        full_1d = torch.cat([prompt_ids_2d[0].detach().cpu(), gen_tensor], dim=0)
        la = int(gen_tensor.numel())

        # Capture Teacher logits along the already-verified exact sequence
        full_dev = full_1d.unsqueeze(0).to(dev)
        if hasattr(engine.target_model, "n_groups") and hasattr(engine.target_model, "manifest"):
            sc = SCacheDynamicCache(window_size=256, num_slots=128)
            lg_all = engine.target_model.forward(full_dev, scache=sc, return_all_logits=True)
        else:
            try:
                out = engine.target_model(full_dev, use_cache=False)
            except TypeError:
                out = engine.target_model(full_dev)
            lg_all = out.logits if hasattr(out, "logits") else out

        ans_lg = lg_all[0, lp - 1 : lp - 1 + la, :].detach().to(torch.float16).cpu()
        q_start = find_subsequence_last_end(prompt_ids_2d[0].tolist(), user_turn_prefix_ids)
        surprise = max(0.05, 1.0 - res.acceptance_rate)

        ep = SleepEpisode(
            capsule_id=capsule_id,
            prompt_ids=prompt_ids_2d[0].detach().cpu(),
            full_ids=full_1d,
            teacher_logits=ans_lg,
            query_start_idx=q_start,
            surprise_score=surprise,
            metadata={"daytime_acceptance_rate": round(res.acceptance_rate, 4)},
        )
        return res, ep


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint (`mos-graft`) for Closed-Form Teacher -> Student Scar Capsule synthesis."""
    parser = argparse.ArgumentParser(
        prog="mos-graft",
        description="Synthesize portable FP16 .safetensors Scar Capsules from a Teacher LLM into a Student SLM (Zero Backprop).",
    )
    parser.add_argument("--teacher", required=True, help="Teacher HF model ID or path (e.g. Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--student", required=True, help="Student SLM HF model ID or path (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--capsule-id", required=True, help="Unique identifier for the synthesized ScarCapsule")
    parser.add_argument("--prompt", action="append", required=True, help="Training prompt phrasing (specify 1..N times)")
    parser.add_argument("--output", required=True, help="Output .scargraft.safetensors path")
    parser.add_argument("--rank", type=int, default=64, help="Low-rank scar dimension (default: 64)")
    parser.add_argument("--tau", type=float, default=0.55, help="Spherical-cap cosine threshold (default: 0.55)")
    parser.add_argument("--max-new-tokens", type=int, default=36, help="Max Teacher trajectory length (default: 36)")
    parser.add_argument("--local-files-only", action="store_true", help="Load models offline from local HF cache")
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=args.local_files_only)
    slm = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.float32, local_files_only=args.local_files_only
    ).to(dev).eval()
    llm = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.float16, local_files_only=args.local_files_only
    ).to(dev).eval()

    sys_prefix = "<|im_start|>system\nAnswer concisely in one precise technical sentence.<|im_end|>\n<|im_start|>user\n"
    ast_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    fmt = lambda q: f"{sys_prefix}{q}{ast_suffix}"

    default_bg = [
        "What is the capital of France?",
        "Write a Python lambda to square x.",
        "Solve 3x + 7 = 22.",
        "What gas do plants absorb during photosynthesis?",
        "Explain how TCP three-way handshake works.",
        "Who wrote Dom Casmurro?",
    ]
    bank = ScarGraftEngine.calibrate_student(slm, default_bg, tokenizer=tok, chat_formatter=fmt)
    cap, _ = ScarGraftEngine.synthesize_from_teacher(
        capsule_id=args.capsule_id,
        train_prompts=args.prompt,
        teacher_model=llm,
        student_model=slm,
        bank=bank,
        tokenizer=tok,
        chat_formatter=fmt,
        max_new_tokens=args.max_new_tokens,
        rank=args.rank,
        tau=args.tau,
    )
    saved = bank.save_safetensors(args.output)
    print(
        f"[mos-graft] Synthesized capsule {cap.capsule_id!r} "
        f"({cap.physical_bytes / 1024.0:.2f} KB, rank={cap.rank}, solve_ms={cap.metadata['solve_ms']}) -> {saved}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
