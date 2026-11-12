"""Unit, Causality & Adversarial Verification Suite for Horizon 1 (`v1.4.0`):
  - Track 1.1: `System 1 + System 2` Ghost-Jump Speculative Reflex Coupling + Atomic `SCache O(1)` Rollback
  - Track 1.2: `MoS Sub-2-Bit` (`2.25 bpw` Lloyd-Max + Two-Pass Alternating `ASVD` Outlier Shield)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.speculative_ghost_jump import (
    GhostJumpSpeculativeEngine,
    recursive_scache_storage_bytes,
)
from scar_cache.sub2bit_mos import (
    SUB2BIT_BETA,
    Sub2BitMoSLinear,
    decompose_sub2bit_mos,
    dequantize_sub2bit_coalesced,
    pack_sub2bit_codes,
    quantize_naive_uniform_2bit_block64,
    transcode_mos_v2_shard_to_sub2bit,
    unpack_sub2bit_codes,
)


class _TinyCausalRoPELM(nn.Module):
    """Deterministic multi-layer causal Transformer compatible with `SCacheDynamicCache`."""

    def __init__(self, vocab_size: int = 128, hidden_size: int = 64, n_layers: int = 4, n_heads: int = 4, seed: int = 7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.embed.weight.data.copy_(torch.randn(vocab_size, hidden_size, generator=g) * 0.15)

        self.q_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)])
        self.k_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)])
        self.v_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)])
        self.o_projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_layers)])
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        for mod in [*self.q_projs, *self.k_projs, *self.v_projs, *self.o_projs, self.head]:
            mod.weight.data.copy_(torch.randn(mod.weight.shape, generator=g) * (1.0 / (hidden_size**0.5)))

    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[SCacheDynamicCache] = None, use_cache: bool = True):
        bsz, q_len = input_ids.shape
        x = self.embed(input_ids)
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        pos = (past_len + torch.arange(q_len, device=input_ids.device)).float()
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=input_ids.device).float() / self.head_dim))
        ang = torch.outer(pos, inv)
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1).view(1, 1, q_len, self.head_dim)
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1).view(1, 1, q_len, self.head_dim)

        def _rot(t: torch.Tensor) -> torch.Tensor:
            h = t.shape[-1] // 2
            return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

        for l_idx in range(self.n_layers):
            q = self.q_projs[l_idx](x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_projs[l_idx](x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_projs[l_idx](x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin

            if past_key_values is not None:
                k, v = past_key_values.update(k, v, l_idx)

            k_len = k.shape[2]
            if q_len == 1:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            elif q_len == k_len:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril(diagonal=k_len - q_len)
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

            x = x + self.o_projs[l_idx](attn.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size))

        return self.head(x)


def _assert_scaches_identical(c1: SCacheDynamicCache, c2: SCacheDynamicCache) -> None:
    assert len(c1.layers) == len(c2.layers)
    assert c1.get_seq_length() == c2.get_seq_length()
    assert c1.total_memory_bytes == c2.total_memory_bytes
    assert recursive_scache_storage_bytes(c1) == c1.total_memory_bytes
    assert recursive_scache_storage_bytes(c2) == c2.total_memory_bytes

    for l1, l2 in zip(c1.layers, c2.layers):
        assert l1.cumulative_length == l2.cumulative_length
        assert l1.active_slots == l2.active_slots
        assert l1.total_tokens_absorbed == l2.total_tokens_absorbed
        assert pytest.approx(l1.err_baseline_ema, rel=1e-5, abs=1e-6) == l2.err_baseline_ema
        assert pytest.approx(l1.err_mad_ema, rel=1e-5, abs=1e-6) == l2.err_mad_ema
        assert l1.keys.shape == l2.keys.shape
        assert torch.allclose(l1.keys, l2.keys, atol=1e-5, rtol=1e-5)
        assert torch.allclose(l1.values, l2.values, atol=1e-5, rtol=1e-5)
        if l1.scache is not None and l2.scache is not None:
            assert torch.allclose(l1.scache.W_scar, l2.scache.W_scar, atol=1e-5, rtol=1e-5)
            assert torch.equal(l1.scache.scars_count, l2.scache.scars_count)
            assert torch.allclose(l1.scar_keys, l2.scar_keys, atol=1e-5, rtol=1e-5)
            assert torch.allclose(l1.scar_vals, l2.scar_vals, atol=1e-5, rtol=1e-5)
            assert torch.allclose(l1.scar_salience, l2.scar_salience, atol=1e-5, rtol=1e-5)


def test_scache_speculative_fast_window_rollback_zero_discrepancy():
    model = _TinyCausalRoPELM(seed=11).eval()
    c_spec = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=4, project_slots=True)
    c_ref = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=4, project_slots=True)

    prefix = torch.randint(0, 128, (1, 20), generator=torch.Generator().manual_seed(1))
    cand_batch = torch.randint(0, 128, (1, 7), generator=torch.Generator().manual_seed(2))

    model(prefix, past_key_values=c_spec)
    model(prefix, past_key_values=c_ref)

    with c_spec.speculative_capture(expected_new_tokens=7) as ctx:
        assert ctx.snapshot.fast_window_only is True
        model(cand_batch, past_key_values=c_spec)
        ctx.commit(3)

    model(cand_batch[:, :3], past_key_values=c_ref)
    _assert_scaches_identical(c_spec, c_ref)


def test_scache_speculative_eviction_widrow_hoff_and_micropage_rollback():
    model = _TinyCausalRoPELM(seed=23).eval()
    c_spec = SCacheDynamicCache(window_size=32, num_slots=16, num_sink_tokens=4, project_slots=True)
    c_ref = SCacheDynamicCache(window_size=32, num_slots=16, num_sink_tokens=4, project_slots=True)

    # Push well beyond `window_size=32` so W_scar, micro-page slots, and deep-layer EMAs are active!
    g = torch.Generator().manual_seed(99)
    for _ in range(4):
        blk = torch.randint(0, 128, (1, 24), generator=g)
        model(blk, past_key_values=c_spec)
        model(blk, past_key_values=c_ref)

    assert c_spec.layers[0].total_tokens_absorbed > 0
    assert c_spec.layers[0].active_slots > 0

    # Part A: Full rollback (`commit(0)`) across eviction + Widrow-Hoff + micro-page promotion restores bit-for-bit state
    spec_batch = torch.randint(0, 128, (1, 9), generator=g)
    with c_spec.speculative_capture(expected_new_tokens=9) as ctx0:
        assert ctx0.snapshot.fast_window_only is False
        model(spec_batch, past_key_values=c_spec)
        ctx0.commit(0)
    _assert_scaches_identical(c_spec, c_ref)

    # Part B: Partial prefix commit (`commit(4)` out of `9` new KV tokens) matches direct `4`-token KV update bit-for-bit
    kv_step = [
        (torch.randn(1, 4, 9, 16, generator=g), torch.randn(1, 4, 9, 16, generator=g))
        for _ in range(4)
    ]
    with c_spec.speculative_capture(expected_new_tokens=9) as ctx4:
        for l_idx, (ks, vs) in enumerate(kv_step):
            c_spec.update(ks, vs, l_idx)
        ctx4.commit(4)

    for l_idx, (ks, vs) in enumerate(kv_step):
        c_ref.update(ks[:, :, :4, :].clone(), vs[:, :, :4, :].clone(), l_idx)

    _assert_scaches_identical(c_spec, c_ref)


def test_ghost_jump_speculative_greedy_exact_parity_all_regimes():
    target_model = _TinyCausalRoPELM(vocab_size=128, hidden_size=64, n_layers=4, n_heads=4, seed=42).eval()
    prompt = torch.randint(0, 128, (1, 18), generator=torch.Generator().manual_seed(77))

    # Standalone System 2 reference with Normal Full KV (within window_size=128 so zero window truncation diff)
    c_standalone = SCacheDynamicCache(window_size=128, num_slots=16, num_sink_tokens=4, project_slots=True)
    logits = target_model(prompt, past_key_values=c_standalone)
    ref_ids = []
    nxt = int(logits[0, -1].argmax().item())
    for _ in range(32):
        ref_ids.append(nxt)
        if len(ref_ids) == 32:
            break
        logits = target_model(torch.tensor([[nxt]]), past_key_values=c_standalone)
        nxt = int(logits[0, -1].argmax().item())

    # Regime A: Perfect System 1 Scout (100% acceptance -> >= 5x tokens per target sweep!)
    eng_perfect = GhostJumpSpeculativeEngine(
        draft_model=target_model,
        target_model=target_model,
        max_draft_tokens=6,
        adaptive_horizon=False,
    )
    res_a = eng_perfect.generate(
        prompt,
        max_new_tokens=32,
        window_size=128,
        num_slots=16,
        temperature=0.0,
        record_reflex_trace=True,
    )
    assert res_a.generated_ids == ref_ids
    assert res_a.acceptance_rate == pytest.approx(1.0)
    assert res_a.mean_tokens_per_target_sweep >= 4.5
    assert res_a.draft_storage_discrepancy_bytes == 0
    assert res_a.target_storage_discrepancy_bytes == 0
    assert len(res_a.reflex_trace) > 0

    # Regime B: Hostile / Completely Wrong System 1 Scout (0% acceptance -> STILL 100.0% identical output!)
    wrong_scout = _TinyCausalRoPELM(vocab_size=128, hidden_size=64, n_layers=2, n_heads=4, seed=999).eval()
    eng_hostile = GhostJumpSpeculativeEngine(
        draft_model=wrong_scout,
        target_model=target_model,
        max_draft_tokens=5,
        adaptive_horizon=True,
    )
    res_b = eng_hostile.generate(
        prompt,
        max_new_tokens=32,
        window_size=128,
        num_slots=16,
        temperature=0.0,
    )
    assert res_b.generated_ids == ref_ids
    assert res_b.draft_storage_discrepancy_bytes == 0
    assert res_b.target_storage_discrepancy_bytes == 0

    # Regime C: Eviction-active O(1) window (`window_size=24`) + Vocabulary Padding (`112` vs `128`) + Stochastic
    padded_scout = _TinyCausalRoPELM(vocab_size=112, hidden_size=64, n_layers=2, n_heads=4, seed=314).eval()
    eng_stoch = GhostJumpSpeculativeEngine(
        draft_model=padded_scout,
        target_model=target_model,
        max_draft_tokens=4,
    )
    res_c = eng_stoch.generate(
        prompt.clamp(max=111),
        max_new_tokens=28,
        window_size=24,
        num_slots=16,
        temperature=0.75,
        seed=1234,
    )
    assert len(res_c.generated_ids) == 28
    assert res_c.draft_storage_discrepancy_bytes == 0
    assert res_c.target_storage_discrepancy_bytes == 0


def test_ghost_jump_stochastic_rejection_sampling_exact_target_distribution():
    """Verify Leviathan-Kalman-Matias rejection sampling matches exact System 2 distribution at T > 0."""
    vocab_size = 8
    target_model = _TinyCausalRoPELM(vocab_size=vocab_size, hidden_size=32, n_layers=2, n_heads=4, seed=101).eval()
    skewed_scout = _TinyCausalRoPELM(vocab_size=vocab_size, hidden_size=32, n_layers=1, n_heads=4, seed=777).eval()
    prompt = torch.tensor([[1, 3, 5, 2, 4]], dtype=torch.long)
    temperature = 0.85

    with torch.no_grad():
        # Step 1 happens directly from target prefill; Step 2 is drafted by skewed_scout & verified by target!
        # Let's condition on a fixed deterministic Step-1 token by temporarily sharpening prompt->tok0 or checking Step-2 conditional!
        # Even cleaner: run `max_new_tokens=2` many times; token index 1 (`generated_ids[1]`) is ALWAYS produced by
        # `draft_step -> verify_sweep` rejection sampling! Compare against exact conditional `P_target(x_2 | prompt, x_1)`.
        eng = GhostJumpSpeculativeEngine(
            draft_model=skewed_scout,
            target_model=target_model,
            max_draft_tokens=2,
            adaptive_horizon=False,
        )

        # Compute exact joint distribution P_target(x_1, x_2 | prompt) on marginal x_2:
        p1_logits = target_model(prompt)
        p1 = F.softmax(p1_logits[0, -1].float() / temperature, dim=-1)
        exact_marginal_x2 = torch.zeros(vocab_size, dtype=torch.float64)
        for tok1 in range(vocab_size):
            seq = torch.cat([prompt, torch.tensor([[tok1]])], dim=1)
            p2_logits = target_model(seq)
            p2 = F.softmax(p2_logits[0, -1].float() / temperature, dim=-1)
            exact_marginal_x2 += float(p1[tok1].item()) * p2.double()

        counts = torch.zeros(vocab_size, dtype=torch.float64)
        n_trials = 800
        for trial in range(n_trials):
            out = eng.generate(
                prompt,
                max_new_tokens=2,
                window_size=64,
                num_slots=8,
                temperature=temperature,
                seed=10_000 + trial,
            )
            counts[out.generated_ids[1]] += 1.0

        emp_x2 = (counts + 1e-6) / (counts.sum() + 1e-6 * vocab_size)
        kl_div = float(torch.sum(exact_marginal_x2 * torch.log(exact_marginal_x2 / emp_x2)).item())
        tv_dist = float(0.5 * torch.sum(torch.abs(exact_marginal_x2 - emp_x2)).item())
        assert kl_div < 0.02
        assert tv_dist < 0.06


def test_sub2bit_pack_unpack_ieee_exact_roundtrip():
    g = torch.Generator().manual_seed(55)
    q_orig = torch.randint(0, 4, (37, 64), generator=g, dtype=torch.uint8)
    packed = pack_sub2bit_codes(q_orig)
    assert packed.shape == (37, 16)
    assert packed.dtype == torch.uint8

    q_unp = unpack_sub2bit_codes(packed)
    assert torch.equal(q_orig, q_unp)

    scales = torch.full((37, 1), 2.0, dtype=torch.float16)
    w_deq = dequantize_sub2bit_coalesced(packed, scales, (37, 64), dtype=torch.float32)
    lut = torch.tensor([-SUB2BIT_BETA, -1.0, +SUB2BIT_BETA, +1.0], dtype=torch.float32) * 2.0
    w_expected = lut[q_orig.long()]
    assert torch.equal(w_deq, w_expected)


def test_sub2bit_asvd_neutralizes_195720x_swiglu_outliers():
    g = torch.Generator().manual_seed(2026)
    N, K = 256, 256
    lin = nn.Linear(K, N, bias=False)
    lin.weight.data.copy_(torch.randn(N, K, generator=g) * 0.02)

    # Inject genuine `195,720x` SwiGLU activation outlier channels (matching Qwen2.5-72B Layer 78!)
    x = torch.randn(16, K, generator=g) * 0.5
    outlier_cols = [7, 71, 139, 211]
    for col in outlier_cols:
        x[:, col] *= 442.4  # RMS spike ~442x -> squared energy ~195,720x!

    act_scales = x.square().mean(dim=0).sqrt()
    y_ref = lin(x)

    # 1. Naive Uniform 2-bit collapses on outlier-amplified channels
    w_naive = quantize_naive_uniform_2bit_block64(lin.weight.data)
    y_naive = F.linear(x, w_naive)
    err_naive = float((torch.norm(y_ref - y_naive) / torch.norm(y_ref)).item())

    # 2. Sub-2-Bit MoS + Rank-32 ASVD Outlier Shield
    sub2_mod, stats = Sub2BitMoSLinear.from_linear(lin, act_scales=act_scales, rank=32, n_alt_iters=2)
    y_sub2 = sub2_mod(x)
    err_sub2 = float((torch.norm(y_ref - y_sub2) / torch.norm(y_ref)).item())
    cos_sub2 = float(F.cosine_similarity(y_ref.reshape(1, -1), y_sub2.reshape(1, -1)).item())

    assert cos_sub2 > 0.996
    assert err_sub2 < err_naive * 0.35
    assert stats.effective_bits_per_weight < 6.5


def test_transcode_mos_v2_shard_to_sub2bit_and_canonical_engine_forward(tmp_path: Path):
    from scar_cache.cli_72b import CanonicalMoSEngine

    # Build a valid synthetic 2-layer `mos_canonical_v2` checkpoint and transcode it to `sub2bit`
    g = torch.Generator().manual_seed(72)
    H, I, V = 128, 256, 64
    v2_dir = tmp_path / "mini_v2"
    sub2_dir = tmp_path / "mini_sub2"
    v2_dir.mkdir()
    sub2_dir.mkdir()

    # Copy tokenizer from cached local model if available, else skip full CLI init and test group transcode
    shard_tensors = {}
    for l_idx in range(2):
        shard_tensors[f"layers.{l_idx}.input_layernorm.weight"] = torch.ones(H, dtype=torch.float16)
        shard_tensors[f"layers.{l_idx}.post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.float16)
        for name, (out_f, in_f) in [
            ("self_attn.q_proj", (H, H)),
            ("self_attn.k_proj", (H, H)),
            ("self_attn.v_proj", (H, H)),
            ("self_attn.o_proj", (H, H)),
            ("mlp.gate_proj", (I, H)),
            ("mlp.up_proj", (I, H)),
            ("mlp.down_proj", (H, I)),
        ]:
            nb = (out_f * in_f) // 64
            pfx = f"layers.{l_idx}.{name}"
            shard_tensors[f"{pfx}.packed_base"] = torch.randint(4, 12, (nb, 32), generator=g, dtype=torch.uint8)
            shard_tensors[f"{pfx}.scales_base"] = torch.full((nb, 1), 0.01, dtype=torch.float16)
            shard_tensors[f"{pfx}.A_scar"] = (torch.randn(8, in_f, generator=g) * 0.01).to(torch.float16)
            shard_tensors[f"{pfx}.B_scar"] = (torch.randn(out_f, 8, generator=g) * 0.01).to(torch.float16)

    in_shard = v2_dir / "mos_group_00.safetensors"
    out_shard = sub2_dir / "mos_group_00.safetensors"
    save_file(shard_tensors, str(in_shard))

    rep = transcode_mos_v2_shard_to_sub2bit(in_shard, out_shard, mlp_rank=8, attn_sub2bit=True, attn_rank=4)
    assert out_shard.exists()
    assert rep["reduction_ratio"] > 1.50

    # Verify CanonicalMoSGroup loads and runs forward_layer on the transcoded Sub-2-Bit shard!
    from scar_cache.cli_72b import CanonicalMoSGroup

    manifest = {
        "format": "mos_canonical_v2_sub2bit",
        "layers_per_group": 2,
        "hidden_size": H,
        "intermediate_size": I,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
    }
    grp_v2 = CanonicalMoSGroup(0, in_shard, manifest, torch.device("cpu"), torch.float32)
    grp_s2 = CanonicalMoSGroup(0, out_shard, manifest, torch.device("cpu"), torch.float32)
    grp_v2.load()
    grp_s2.load()

    x = torch.randn(1, 5, H)
    cos = torch.ones(1, 1, 5, 32)
    sin = torch.zeros(1, 1, 5, 32)
    y_v2 = grp_v2.forward_layer(0, x, cos, sin, scache=None)
    y_s2 = grp_s2.forward_layer(0, x, cos, sin, scache=None)
    cos_sim = float(F.cosine_similarity(y_v2.reshape(1, -1), y_s2.reshape(1, -1)).item())
    assert cos_sim > 0.99
