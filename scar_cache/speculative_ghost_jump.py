"""Horizon 1 (v1.4) Track 1.1: System 1 + System 2 Ghost-Jump Speculative Reflex Coupling.

Pairs a high-speed `System 1` Scout (`0.5B MoS + SCache O(1)`) with a `System 2` Verifier
(`3B` / `7B` / `72.71B CanonicalMoSEngine` + `SCache O(1)`):
  1. Adaptive Ghost-Jump Horizon (`K = 1..max_draft_tokens`) governed by `System One`
     `ReflexDecision` telemetry (`confidence`, `margin = P_1 - P_2`, normalized entropy).
  2. Single-Sweep Batched Verification (`M = K + 1` tokens per `System 2` weight pass).
  3. Mathematically Lossless Output (`KL = 0.0000`):
     - Exact Greedy Token Parity (`temperature <= 0.0`)
     - Exact Leviathan-Kalman-Matias Stochastic Rejection Sampling (`temperature > 0.0`)
       with automatic vocabulary-padding alignment (`151,936` <-> `152,064`).
  4. Atomic O(1) `SCacheDynamicCache` Rollback (`0.00 Bytes` storage discrepancy, zero
     recompute of accepted or rejected Transformer weights).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from scar_cache.hf_cache import SCacheDynamicCache


@dataclass(frozen=True)
class GhostJumpReflexDecision:
    """System One `ReflexDecision` contract for a single Ghost-Jump draft step."""

    token_id: int
    confidence: float       # [0.0, 1.0] top-1 probability
    uncertainty: float      # [0.0, 1.0] normalized Shannon entropy
    margin: float           # [0.0, 1.0] P_(1) - P_(2)
    is_uncertain: bool      # True when System 1 hits an epistemic bifurcation -> escalate to S2
    latency_ms: float


@dataclass
class GhostJumpGenerationResult:
    """Complete telemetry + generated sequence from `GhostJumpSpeculativeEngine`."""

    generated_ids: List[int]
    accepted_draft_tokens: int
    total_drafted_tokens: int
    target_sweeps: int
    prefill_elapsed_s: float
    decode_elapsed_s: float
    total_elapsed_s: float
    decode_tok_per_s: float
    draft_scache_bytes: int
    target_scache_bytes: int
    draft_storage_discrepancy_bytes: int
    target_storage_discrepancy_bytes: int
    reflex_trace: List[GhostJumpReflexDecision] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted_draft_tokens) / float(max(self.total_drafted_tokens, 1))

    @property
    def mean_tokens_per_target_sweep(self) -> float:
        return float(len(self.generated_ids)) / float(max(self.target_sweeps, 1))


def recursive_scache_storage_bytes(obj: Any, seen: Optional[set[int]] = None) -> int:
    """Walks all attributes/containers and sums unique underlying `untyped_storage` bytes."""
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, torch.Tensor):
        st = obj.untyped_storage()
        sid = int(st.data_ptr())
        if sid in seen:
            return 0
        seen.add(sid)
        return int(st.nbytes())
    if isinstance(obj, dict):
        return sum(recursive_scache_storage_bytes(k, seen) + recursive_scache_storage_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set)):
        return sum(recursive_scache_storage_bytes(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        return recursive_scache_storage_bytes(vars(obj), seen)
    return 0


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _infer_model_device(model: Any) -> torch.device:
    if hasattr(model, "device") and isinstance(model.device, torch.device):
        return model.device
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _infer_vocab_limit(model: Any) -> Optional[int]:
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "num_embeddings"):
                return int(emb.num_embeddings)
        except Exception:
            pass
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        return int(model.config.vocab_size)
    if hasattr(model, "vocab_size"):
        return int(model.vocab_size)
    if hasattr(model, "emb_q") and isinstance(model.emb_q, torch.Tensor):
        return int(model.emb_q.shape[0])
    return None


@torch.no_grad()
def _forward_all_logits(model: Any, input_ids: torch.Tensor, scache: SCacheDynamicCache) -> torch.Tensor:
    """Unified causal LM forward returning `[B, Q, V]` logits for HF models, CanonicalMoSEngine, or callables."""
    dev = _infer_model_device(model)
    ids = input_ids.to(dev) if input_ids.device != dev else input_ids
    v_lim = _infer_vocab_limit(model)
    if v_lim is not None and v_lim > 0:
        ids = ids.clamp(min=0, max=v_lim - 1)

    # 1. CanonicalMoSEngine (`scar_cache/cli_72b.py`)
    if hasattr(model, "n_groups") and hasattr(model, "forward") and hasattr(model, "manifest"):
        logits = model.forward(ids, scache=scache, return_all_logits=True)
        return logits if logits.ndim == 3 else logits.unsqueeze(1)

    # 2. HuggingFace PreTrainedModel (`AutoModelForCausalLM`)
    out = model(ids, past_key_values=scache, use_cache=True)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, torch.Tensor):
        return out if out.ndim == 3 else out.unsqueeze(1)
    raise TypeError(f"Unsupported model output type: {type(out)}")


@torch.no_grad()
def _prefill_chunked(
    model: Any,
    input_ids: torch.Tensor,
    scache: SCacheDynamicCache,
) -> torch.Tensor:
    """Chunked O(1) prefill returning last-position logits `[B, V]` while honoring `SCache` window geometry."""
    step = scache.prefill_chunk_size
    seq_len = int(input_ids.shape[1])
    last_logits: Optional[torch.Tensor] = None
    for s0 in range(0, seq_len, step):
        chunk = input_ids[:, s0 : s0 + step]
        if hasattr(model, "n_groups") and hasattr(model, "manifest"):
            dev = _infer_model_device(model)
            last_logits = model.forward(chunk.to(dev), scache=scache, return_all_logits=False)
            if last_logits.ndim == 3:
                last_logits = last_logits[:, -1, :]
        else:
            logits_3d = _forward_all_logits(model, chunk, scache)
            last_logits = logits_3d[:, -1, :]
    assert last_logits is not None
    return last_logits


def _compute_reflex_telemetry(
    logits_1d: torch.Tensor,
    temperature: float,
    min_confidence: float,
    min_margin: float,
    elapsed_ms: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, Optional[torch.Tensor], GhostJumpReflexDecision]:
    """Computes `ReflexDecision` epistemic metrics + drafts next token (`greedy` or `stochastic`)."""
    lf = logits_1d.float()
    vocab_size = int(lf.shape[-1])

    # Always compute calibrated epistemic telemetry on temperature-scaled (or T=1.0 for greedy) softmax
    eff_temp = max(temperature, 1.0) if temperature <= 0.0 else max(temperature, 1e-5)
    scaled = lf / eff_temp
    log_z = torch.logsumexp(scaled, dim=-1)
    log_probs = scaled - log_z
    probs = torch.exp(log_probs)

    top2_vals, top2_idx = torch.topk(probs, k=min(2, vocab_size), dim=-1)
    p1 = float(top2_vals[0].item())
    p2 = float(top2_vals[1].item()) if vocab_size > 1 else 0.0
    margin = max(0.0, p1 - p2)

    # Exact Shannon entropy in `[0, 1]`
    ent = -float((probs * log_probs.clamp(min=-80.0)).sum().item())
    max_ent = math.log(max(vocab_size, 2))
    uncertainty = min(1.0, max(0.0, ent / max_ent))

    if temperature <= 0.0:
        tok_id = int(lf.argmax(dim=-1).item())
        q_dist: Optional[torch.Tensor] = None
    else:
        q_dist = probs
        if generator is not None and probs.device.type == "cpu":
            tok_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        else:
            tok_id = int(torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item())

    is_uncertain = (p1 < min_confidence) or (margin < min_margin)
    decision = GhostJumpReflexDecision(
        token_id=tok_id,
        confidence=p1,
        uncertainty=uncertainty,
        margin=margin,
        is_uncertain=is_uncertain,
        latency_ms=elapsed_ms,
    )
    return tok_id, q_dist, decision


def _align_vocab_probs(q_probs: torch.Tensor, target_vocab: int) -> torch.Tensor:
    """Aligns `System 1` probability vector (`151,936`) to `System 2` vocabulary (`152,064`) losslessly."""
    q_cpu = q_probs.detach().float().cpu()
    v_q = int(q_cpu.shape[-1])
    if v_q == target_vocab:
        return q_cpu
    if v_q < target_vocab:
        return F.pad(q_cpu, (0, target_vocab - v_q), value=0.0)
    cropped = q_cpu[:target_vocab].clone()
    s = float(cropped.sum().item())
    return cropped / max(s, 1e-12)


class GhostJumpSpeculativeEngine:
    """System 1 + System 2 Speculative Reflex Decoding Engine with Atomic O(1) SCache Rollback."""

    def __init__(
        self,
        draft_model: Any,
        target_model: Any,
        max_draft_tokens: int = 6,
        min_draft_confidence: float = 0.35,
        min_draft_margin: float = 0.15,
        adaptive_horizon: bool = True,
    ) -> None:
        if max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be >= 1")
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_draft_tokens = int(max_draft_tokens)
        self.min_draft_confidence = float(min_draft_confidence)
        self.min_draft_margin = float(min_draft_margin)
        self.adaptive_horizon = bool(adaptive_horizon)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        draft_cache: Optional[SCacheDynamicCache] = None,
        target_cache: Optional[SCacheDynamicCache] = None,
        window_size: int = 256,
        num_slots: int = 128,
        num_sink_tokens: int = 4,
        project_slots: bool = True,
        temperature: float = 0.0,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        seed: Optional[int] = None,
        record_reflex_trace: bool = False,
        audit_storage_bytes: bool = True,
    ) -> GhostJumpGenerationResult:
        """Generates up to `max_new_tokens` with exact `KL = 0.0000` target fidelity and O(1) memory."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"GhostJumpSpeculativeEngine expects input_ids [1, seq_len], got {tuple(input_ids.shape)}")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be >= 1")

        eos_set: set[int] = set()
        if isinstance(eos_token_id, int):
            eos_set.add(eos_token_id)
        elif eos_token_id is not None:
            eos_set.update(int(x) for x in eos_token_id)

        rng: Optional[torch.Generator] = None
        if seed is not None:
            rng = torch.Generator(device="cpu").manual_seed(int(seed))

        if draft_cache is None:
            draft_cache = SCacheDynamicCache(
                window_size=window_size,
                num_slots=num_slots,
                num_sink_tokens=num_sink_tokens,
                project_slots=project_slots,
            )
        if target_cache is None:
            target_cache = SCacheDynamicCache(
                window_size=window_size,
                num_slots=num_slots,
                num_sink_tokens=num_sink_tokens,
                project_slots=project_slots,
            )

        draft_dev = _infer_model_device(self.draft_model)
        target_dev = _infer_model_device(self.target_model)

        _sync_device(draft_dev)
        _sync_device(target_dev)
        t_start = time.perf_counter()

        # 1. Chunked O(1) Prefill on Prompt Prefix `input_ids[:, :-1]` if len > 1 so `last_tok = input_ids[:, -1]`
        #    enters the unified Ghost-Jump loop cleanly — OR prefill full `input_ids` on target & `input_ids[:, :-1]`
        #    No: prefilling `input_ids` on BOTH caches gives `first_tok` directly from `target_model`!
        prompt_len = int(input_ids.shape[1])
        if prompt_len > 1:
            _prefill_chunked(self.draft_model, input_ids, draft_cache)
        else:
            _forward_all_logits(self.draft_model, input_ids, draft_cache)

        target_prefill_logits = _prefill_chunked(self.target_model, input_ids, target_cache)
        _sync_device(draft_dev)
        _sync_device(target_dev)
        t_after_prefill = time.perf_counter()

        # Sample / pick Token #1 directly from `target_prefill_logits[0]`
        if temperature <= 0.0:
            first_tok = int(target_prefill_logits[0].argmax(dim=-1).item())
        else:
            p0 = F.softmax(target_prefill_logits[0].float() / max(temperature, 1e-5), dim=-1).cpu()
            first_tok = int(torch.multinomial(p0, num_samples=1, generator=rng).item())

        generated_ids: List[int] = [first_tok]
        last_tok = first_tok
        sync_prefix: List[int] = []  # Holds `[d_{K-1}]` after 100%-accepted sweeps so zero extra S1 calls occur!

        accepted_draft_total = 0
        drafted_total = 0
        target_sweeps = 1  # Prefill sweep produced token #1
        reflex_trace: List[GhostJumpReflexDecision] = []

        # 2. Ghost-Jump Speculative Loop for remaining `max_new_tokens - 1` tokens
        while len(generated_ids) < max_new_tokens and (last_tok not in eos_set):
            rem_budget = max_new_tokens - len(generated_ids)
            cur_k_max = min(self.max_draft_tokens, rem_budget)
            s_sync = len(sync_prefix)

            draft_tokens: List[int] = []
            draft_dists: List[Optional[torch.Tensor]] = []

            # Step 2a: System 1 Scout Ghost-Jump Lookahead (`1 .. cur_k_max` tokens)
            with draft_cache.speculative_capture(
                expected_new_tokens=s_sync + cur_k_max,
                multi_step=True,
            ) as draft_ctx:
                for step_i in range(cur_k_max):
                    t_d0 = time.perf_counter()
                    if step_i == 0 and s_sync > 0:
                        step_in = torch.tensor([sync_prefix + [last_tok]], dtype=torch.long, device=draft_dev)
                    else:
                        prev_id = last_tok if step_i == 0 else draft_tokens[-1]
                        step_in = torch.tensor([[prev_id]], dtype=torch.long, device=draft_dev)

                    d_logits_3d = _forward_all_logits(self.draft_model, step_in, draft_cache)
                    d_logits = d_logits_3d[0, -1]
                    elapsed_ms = (time.perf_counter() - t_d0) * 1000.0

                    d_tok, q_dist, decision = _compute_reflex_telemetry(
                        d_logits,
                        temperature=temperature,
                        min_confidence=self.min_draft_confidence,
                        min_margin=self.min_draft_margin,
                        elapsed_ms=elapsed_ms,
                        generator=rng,
                    )
                    draft_tokens.append(d_tok)
                    draft_dists.append(q_dist)
                    if record_reflex_trace:
                        reflex_trace.append(decision)

                    if d_tok in eos_set:
                        break
                    if self.adaptive_horizon and step_i >= 1 and decision.is_uncertain:
                        break

                K_cur = len(draft_tokens)
                drafted_total += K_cur

                # Step 2b: Single-Sweep System 2 Verification (`M = K_cur + 1` tokens in ONE pass!)
                verify_seq = [last_tok] + draft_tokens
                verify_tensor = torch.tensor([verify_seq], dtype=torch.long, device=target_dev)

                with target_cache.speculative_capture(
                    expected_new_tokens=K_cur + 1,
                    multi_step=False,
                ) as target_ctx:
                    target_logits_3d = _forward_all_logits(self.target_model, verify_tensor, target_cache)
                    target_sweeps += 1

                    num_accepted = 0
                    next_tok: Optional[int] = None

                    if temperature <= 0.0:
                        # Exact Greedy Verification (`KL = 0.0000`)
                        for i in range(K_cur):
                            t_pred = int(target_logits_3d[0, i].argmax(dim=-1).item())
                            if t_pred == draft_tokens[i]:
                                num_accepted += 1
                            else:
                                next_tok = t_pred
                                break
                        if num_accepted == K_cur:
                            next_tok = int(target_logits_3d[0, K_cur].argmax(dim=-1).item())
                    else:
                        # Exact Leviathan-Kalman-Matias Stochastic Rejection Sampling (`KL = 0.0000`)
                        inv_t = 1.0 / max(temperature, 1e-5)
                        for i in range(K_cur):
                            p_vec = F.softmax(target_logits_3d[0, i].float() * inv_t, dim=-1).cpu()
                            q_raw = draft_dists[i]
                            assert q_raw is not None
                            q_vec = _align_vocab_probs(q_raw, int(p_vec.shape[-1]))

                            cand = draft_tokens[i]
                            p_c = float(p_vec[cand].item()) if cand < p_vec.shape[-1] else 0.0
                            q_c = float(q_vec[cand].item())
                            accept_prob = 1.0 if p_c >= q_c else (p_c / max(q_c, 1e-12))
                            u = float(torch.rand(1, generator=rng).item())

                            if u <= accept_prob:
                                num_accepted += 1
                            else:
                                diff = (p_vec - q_vec).clamp(min=0.0)
                                diff_sum = float(diff.sum().item())
                                if diff_sum > 1e-12:
                                    diff = diff / diff_sum
                                else:
                                    diff = p_vec
                                next_tok = int(torch.multinomial(diff, num_samples=1, generator=rng).item())
                                break

                        if num_accepted == K_cur:
                            p_bonus = F.softmax(target_logits_3d[0, K_cur].float() * inv_t, dim=-1).cpu()
                            next_tok = int(torch.multinomial(p_bonus, num_samples=1, generator=rng).item())

                    assert next_tok is not None

                    # Respect exact `max_new_tokens` & `eos_token_id` truncation cleanly
                    emitted_step: List[int] = []
                    hit_eos = False
                    for idx_acc in range(num_accepted):
                        tok_acc = draft_tokens[idx_acc]
                        emitted_step.append(tok_acc)
                        if tok_acc in eos_set or (len(generated_ids) + len(emitted_step)) >= max_new_tokens:
                            # Truncate valid cache commit right before the stopping token (`last_tok` holds it)
                            num_accepted_eff = idx_acc
                            target_ctx.commit(num_accepted_eff + 1)
                            draft_ctx.commit(s_sync + num_accepted_eff + 1)
                            accepted_draft_total += idx_acc + 1
                            last_tok = tok_acc
                            sync_prefix = []
                            hit_eos = True
                            break

                    if not hit_eos:
                        accepted_draft_total += num_accepted
                        emitted_step.append(next_tok)
                        if num_accepted < K_cur:
                            target_ctx.commit(num_accepted + 1)
                            draft_ctx.commit(s_sync + num_accepted + 1)
                            sync_prefix = []
                        else:
                            target_ctx.commit(K_cur + 1)
                            draft_ctx.commit(s_sync + K_cur)
                            sync_prefix = [draft_tokens[K_cur - 1]]
                        last_tok = next_tok

            generated_ids.extend(emitted_step)
            if last_tok in eos_set:
                break

        _sync_device(draft_dev)
        _sync_device(target_dev)
        t_end = time.perf_counter()

        prefill_s = t_after_prefill - t_start
        decode_s = max(t_end - t_after_prefill, 1e-6)
        total_s = t_end - t_start
        decode_gen = max(len(generated_ids) - 1, 1)

        d_rep = int(draft_cache.total_memory_bytes)
        t_rep = int(target_cache.total_memory_bytes)
        if audit_storage_bytes:
            d_rec = int(recursive_scache_storage_bytes(draft_cache))
            t_rec = int(recursive_scache_storage_bytes(target_cache))
            d_disc = abs(d_rec - d_rep)
            t_disc = abs(t_rec - t_rep)
            assert d_disc == 0, f"Draft SCache storage discrepancy: {d_rec} != {d_rep}"
            assert t_disc == 0, f"Target SCache storage discrepancy: {t_rec} != {t_rep}"
        else:
            d_disc = 0
            t_disc = 0

        return GhostJumpGenerationResult(
            generated_ids=generated_ids,
            accepted_draft_tokens=accepted_draft_total,
            total_drafted_tokens=drafted_total,
            target_sweeps=target_sweeps,
            prefill_elapsed_s=round(prefill_s, 4),
            decode_elapsed_s=round(decode_s, 4),
            total_elapsed_s=round(total_s, 4),
            decode_tok_per_s=round(decode_gen / decode_s, 2),
            draft_scache_bytes=d_rep,
            target_scache_bytes=t_rep,
            draft_storage_discrepancy_bytes=d_disc,
            target_storage_discrepancy_bytes=t_disc,
            reflex_trace=reflex_trace,
        )
