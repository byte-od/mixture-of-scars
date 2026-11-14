"""HuggingFace SCache Integration: Drop-in Cache O(1) para Modelos Reais (Llama, SmolLM, Qwen).

Substitui o DynamicCache linear O(T) do HuggingFace por uma arquitetura híbrida O(1):
  1. Attention Sinks (primeiros `num_sink_tokens` fixos para ancoragem de fase RoPE/Softmax).
  2. Janela Deslizante Local (últimos `window_size - num_sink_tokens` tokens recentes).
  3. Banco Associativo de Cicatrizes Por-Cabeça (`scar_keys`, `scar_vals`, `W_scar`) atualizado
     via Micro-Páginas Contíguas Sincronizadas + Roteamento Competitivo por Surpresa Widrow-Hoff
     (Anti-Ghosting), com decaimento lento anti-flood e footprint físico 100% constante O(1).
  4. Compatibilidade nativa com `transformers` v5 (`cumulative_length`, `get_seq_length`,
     `get_mask_sizes`) suportando decodificação token-a-token, prefill em blocos e prefill único.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer

from scar_cache.multi_slot import MultiSlotSCache


def _choose_page_size(num_slots: int) -> int:
    """Seleciona tamanho de micro-página contígua (32, 16, 8 ou 4 tokens) preservando cláusulas/n-grams."""
    for p in (32, 16, 8, 4):
        if num_slots >= 2 * p and num_slots % p == 0:
            return p
    return 1


def _extract_top_pages(sync_1d: torch.Tensor, page_sz: int, max_pages: int) -> List[List[Tuple[int, float]]]:
    """Extrai exatamente `min(max_pages, T // page_sz)` janelas disjuntas ordenadas cronologicamente."""
    bsz, T = sync_1d.shape
    k_pages = min(max_pages, T // page_sz)
    if k_pages <= 0:
        return [[] for _ in range(bsz)]

    win_raw = sync_1d.unfold(1, page_sz, 1).mean(dim=-1)  # [B, M]
    M = win_raw.shape[1]
    shift = min(6, page_sz // 4)
    if shift >= 2 and M > shift:
        shifted = F.pad(win_raw[:, : M - shift], (shift, 0), mode="replicate")
        win_scores_all = (0.55 * win_raw + 0.45 * shifted).detach().cpu()
    else:
        win_scores_all = win_raw.detach().cpu()

    pages_per_b: List[List[Tuple[int, float]]] = []
    for b in range(bsz):
        w_orig = win_scores_all[b]  # [M]
        w = w_orig.clone()
        picked: List[Tuple[int, float]] = []
        for _ in range(k_pages):
            best_s = int(w.argmax().item())
            sc = float(w[best_s].item())
            if sc <= -1e8:
                break
            picked.append((best_s, sc))
            lo = max(0, best_s - page_sz + 1)
            hi = min(M, best_s + page_sz)
            w[lo:hi] = -1e9

        if len(picked) == k_pages:
            picked.sort(key=lambda x: x[0])
            pages_per_b.append(picked)
            continue

        # Fallback exato via Programação Dinâmica 1D quando `T` é apertado (garante `len == k_pages` sempre!)
        backptrs: List[torch.Tensor] = []
        dp_prev = w_orig.clone()
        for r in range(1, k_pages):
            min_s = r * page_sz
            pref_vals, pref_idx = torch.cummax(dp_prev[: M - page_sz], dim=0)
            dp_cur = torch.full((M,), -1e18, dtype=torch.float32)
            bp_cur = torch.zeros((M,), dtype=torch.long)
            dp_cur[min_s:] = w_orig[min_s:] + pref_vals[min_s - page_sz : M - page_sz]
            bp_cur[min_s:] = pref_idx[min_s - page_sz : M - page_sz]
            backptrs.append(bp_cur)
            dp_prev = dp_cur

        last_min_s = (k_pages - 1) * page_sz
        rel_best = int(dp_prev[last_min_s:].argmax().item())
        cur_s = last_min_s + rel_best
        starts = [cur_s]
        for r in range(k_pages - 2, -1, -1):
            cur_s = int(backptrs[r][cur_s].item())
            starts.append(cur_s)
        starts.reverse()
        pages_per_b.append([(s, float(w_orig[s].item())) for s in starts])

    return pages_per_b


@torch.no_grad()
def _layer_widrow_hoff_salience_1d(
    layer: "SCacheDynamicLayer",
    ex_k: torch.Tensor,
    ex_v: torch.Tensor,
    sink_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calcula surpresa direcional associativa de Widrow-Hoff padronizada por MAD em uma camada."""
    T = ex_k.shape[-2]
    kf = ex_k.float()
    vf = ex_v.float()
    kn = kf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vn = vf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ku = kf / kn
    vu = vf / vn
    kn_s = kn.squeeze(-1)
    vn_s = vn.squeeze(-1)
    med_kn = kn_s.median(dim=-1, keepdim=True).values.clamp(min=1e-5)
    med_vn = vn_s.median(dim=-1, keepdim=True).values.clamp(min=1e-5)

    assert layer.scache is not None
    W_hist = layer.scache.W_scar
    has_hist = layer.total_tokens_absorbed > 0
    head_dim = ex_k.shape[-1]
    use_blk = T >= max(64, head_dim)
    delim = (
        (kn_s > 2.0 * med_kn).float()
        if use_blk
        else ((kn_s > 2.0 * med_kn) & (vn_s < 0.90 * med_vn)).float()
    )
    if use_blk:
        W_blk = torch.einsum("bhtv,bhtd->bhvd", vu, ku) / float(T)
        if has_hist:
            pred = 0.55 * torch.einsum("bhtd,bhvd->bhtv", ku, W_blk) + 0.45 * torch.einsum("bhtd,hvd->bhtv", ku, W_hist)
        else:
            pred = torch.einsum("bhtd,bhvd->bhtv", ku, W_blk)
        err = (vu - pred).norm(dim=-1) * (1.0 - 0.75 * delim)
    elif has_hist:
        pred = torch.einsum("bhtd,hvd->bhtv", ku, W_hist)
        err = (vu - pred).norm(dim=-1) * (1.0 - 0.75 * delim)
    else:
        err = (vn_s / med_vn).clamp(max=4.0) * (1.0 - 0.75 * delim)
    if sink_v is not None and sink_v.shape[-2] > 0:
        sv_u = F.normalize(sink_v.float(), dim=-1)
        sv_sim = torch.einsum("bhtd,bhsd->bhts", vu, sv_u).abs().amax(dim=-1)
        err = err * (1.0 - 0.50 * (sv_sim > 0.80).float())

    err_1d = err.mean(dim=1)  # [B, T]
    if T >= 8:
        if use_blk:
            med = err_1d.median(dim=-1, keepdim=True).values
            mad = (err_1d - med).abs().median(dim=-1, keepdim=True).values.clamp(min=1e-4)
            cur_med = float(med.mean().item())
            cur_mad = float(mad.mean().item())
            layer.err_chunks_seen += 1
            if layer.err_chunks_seen == 1 or layer.err_baseline_ema <= 0.0 or layer.err_mad_ema <= 0.0:
                layer.err_baseline_ema = cur_med
                layer.err_mad_ema = cur_mad
            else:
                w_ema = 1.0 / float(min(layer.err_chunks_seen, 64))
                layer.err_baseline_ema += w_ema * (max(cur_med, 0.97 * layer.err_baseline_ema) - layer.err_baseline_ema)
                layer.err_mad_ema += w_ema * (max(cur_mad, 0.90 * layer.err_mad_ema) - layer.err_mad_ema)
            ref_med = torch.maximum(med, torch.full_like(med, layer.err_baseline_ema))
            ref_mad = torch.maximum(mad, torch.full_like(mad, layer.err_mad_ema))
            return ((err_1d - ref_med) / ref_mad).clamp(min=-2.0, max=12.0)
        cur_med_t = err_1d.median(dim=-1, keepdim=True).values
        cur_mad_t = (err_1d - cur_med_t).abs().median(dim=-1, keepdim=True).values.clamp(min=1e-4)
        cur_med = float(cur_med_t.mean().item())
        if layer.err_baseline_ema <= 0.0:
            layer.err_baseline_ema = cur_med
        else:
            layer.err_baseline_ema = 0.92 * layer.err_baseline_ema + 0.08 * max(cur_med, 0.85 * layer.err_baseline_ema)
        ema_t = torch.full_like(cur_med_t, layer.err_baseline_ema)
        ref_med = torch.maximum(cur_med_t, 0.90 * ema_t)
        ref_mad = torch.maximum(cur_mad_t, 0.12 * ref_med.clamp(min=0.25))
        return ((err_1d - ref_med) / ref_mad).clamp(min=-2.0, max=12.0)
    return err_1d


class SCacheDynamicLayer(DynamicLayer):
    """Camada de Cache do HuggingFace com tamanho físico estritamente constante O(1)."""

    is_sliding = True

    def __init__(
        self,
        window_size: int = 64,
        num_slots: int = 8,
        num_sink_tokens: int = 4,
        project_slots: bool = False,
        merge_cosine_threshold: float = 0.995,
        delta_blend: float = 0.05,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = min(int(num_sink_tokens), self.window_size // 2) if num_sink_tokens > 0 else 0
        self.project_slots = bool(project_slots)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.delta_blend = float(delta_blend)

        self.scache: Optional[MultiSlotSCache] = None
        self.scar_keys: Optional[torch.Tensor] = None
        self.scar_vals: Optional[torch.Tensor] = None
        self.scar_salience: Optional[torch.Tensor] = None
        self.scar_decay: Optional[torch.Tensor] = None
        self.active_slots: int = 0
        self.total_tokens_absorbed: int = 0
        self.cumulative_length: int = 0
        self.err_baseline_ema: float = 0.0
        self.err_mad_ema: float = 0.0
        self.err_chunks_seen: int = 0

    def _init_scar_bank(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        bsz, n_heads, _, head_dim = key_states.shape
        dev = key_states.device
        dt = key_states.dtype

        if self.num_slots > 0:
            # Uma matriz associativa Widrow-Hoff exata `W_scar[h]` [head_dim, head_dim] por cabeça KV!
            self.scache = MultiSlotSCache(
                slot_dim=head_dim,
                num_slots=n_heads,
                use_delta_rule=True,
                multi_scale_decay=False,
            ).to(dev)
            self.scar_keys = torch.zeros((bsz, n_heads, self.num_slots, head_dim), dtype=dt, device=dev)
            self.scar_vals = torch.zeros((bsz, n_heads, self.num_slots, head_dim), dtype=value_states.dtype, device=dev)
            self.scar_salience = torch.zeros((bsz, n_heads, self.num_slots), dtype=torch.float32, device=dev)
            if self.num_slots == 1:
                decays = torch.tensor([0.999], dtype=torch.float32, device=dev)
            else:
                decays = torch.exp(
                    torch.linspace(math.log(0.93), math.log(0.9998), self.num_slots, device=dev, dtype=torch.float32)
                )
            self.scar_decay = decays

    def _predict_returned_kv_length(self, query_length: int) -> int:
        cached_len = self.keys.shape[-2] if (self.is_initialized and self.keys is not None) else 0
        total_raw = cached_len + query_length
        if total_raw <= self.window_size:
            return total_raw

        slide_len = self.window_size - self.num_sink_tokens
        tail_vis = max(slide_len, query_length) if self.project_slots else slide_len
        tail_start = total_raw - tail_vis
        sink_vis = min(self.num_sink_tokens, tail_start) if self.project_slots else self.num_sink_tokens
        split_idx = max(self.num_sink_tokens, tail_start)
        pre_excess = split_idx - self.num_sink_tokens
        page_sz = _choose_page_size(self.num_slots)

        if page_sz > 1:
            if pre_excess >= page_sz:
                n_tot = self.num_slots // page_sz
                max_new = 1 if self.active_slots > 0 else min(2, n_tot)
                add_slots = min(pre_excess // page_sz, max_new) * page_sz
            elif self.active_slots == self.num_slots:
                add_slots = 0
            else:
                add_slots = pre_excess
        else:
            add_slots = pre_excess

        pred_active = (
            min(self.num_slots, self.active_slots + add_slots)
            if (
                self.project_slots
                and self.num_slots > 0
                and tail_start > self.num_sink_tokens
                and (self.active_slots + add_slots) > 0
            )
            else 0
        )
        return sink_vis + pred_active + tail_vis

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.window_size + (self.num_slots if self.project_slots else 0)

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_length = self._predict_returned_kv_length(query_length)
        prefix_len = kv_length - query_length
        kv_offset = self.cumulative_length - prefix_len
        return kv_length, kv_offset

    def _compute_token_salience(
        self,
        excess_k: torch.Tensor,
        excess_v: torch.Tensor,
        obs_k: Optional[torch.Tensor] = None,
        obs_v: Optional[torch.Tensor] = None,
        sink_v: Optional[torch.Tensor] = None,
        prev_sync_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kf = excess_k.float()
        k_unit = kf / kf.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        if prev_sync_sal is not None and prev_sync_sal.shape[-1] == excess_k.shape[-2]:
            return k_unit, prev_sync_sal.to(excess_k.device)
        cur_1d = _layer_widrow_hoff_salience_1d(self, excess_k, excess_v, sink_v=sink_v)
        return k_unit, cur_1d

    @torch.no_grad()
    def _ingest_evicted_tokens(
        self,
        excess_k: torch.Tensor,
        excess_v: torch.Tensor,
        obs_k: Optional[torch.Tensor] = None,
        obs_v: Optional[torch.Tensor] = None,
        sink_v: Optional[torch.Tensor] = None,
        prev_sync_sal: Optional[torch.Tensor] = None,
        use_obs: bool = True,
        retire_len: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Comprime tokens desalojados no banco O(1) via Micro-Páginas Contíguas Sincronizadas + Regra Delta."""
        num_excess = excess_k.shape[-2]
        if num_excess == 0 or self.num_slots <= 0:
            return None

        if self.scar_keys is None or self.scar_vals is None or self.scar_salience is None or self.scache is None:
            return None

        bsz, n_heads, _, _ = excess_k.shape
        kf = excess_k.float()
        vf = excess_v.float()
        vu = F.normalize(vf, dim=-1)

        k_unit, sync_1d = self._compute_token_salience(
            excess_k,
            excess_v,
            obs_k=(obs_k if use_obs else None),
            obs_v=(obs_v if use_obs else None),
            sink_v=sink_v,
            prev_sync_sal=prev_sync_sal,
        )

        # Atualização associativa direcional Widrow-Hoff Delta-Rule por cabeça apenas nos tokens aposentados
        ret_n = min(num_excess, max(1, retire_len)) if retire_len is not None else num_excess
        W_cur = self.scache.W_scar
        ku_ret = k_unit[..., :ret_n, :]
        vu_ret = vu[..., :ret_n, :]
        pred_u = torch.einsum("bhtd,hvd->bhtv", ku_ret, W_cur)
        dW = torch.einsum("bhtv,bhtd->hvd", vu_ret - pred_u, ku_ret) / float(max(bsz * ret_n, 1))
        W_cur.mul_(0.94).add_(dW, alpha=0.38)
        if hasattr(self.scache, "scars_count"):
            self.scache.scars_count.add_(ret_n)

        page_sz = _choose_page_size(self.num_slots)

        if page_sz > 1 and ret_n >= page_sz:
            n_total_pages = self.num_slots // page_sz
            max_new = min(ret_n // page_sz, 1 if self.active_slots > 0 else min(2, n_total_pages))
            cand_pages = _extract_top_pages(sync_1d, page_sz, max_new)
            sk_f = self.scar_keys
            sv_f = self.scar_vals

            if self.active_slots > 0:
                self.scar_salience.mul_(0.99995)

            max_cands = max((len(c) for c in cand_pages), default=0)
            for c_idx in range(max_cands):
                if self.active_slots + page_sz <= self.num_slots:
                    s0 = self.active_slots
                    for b in range(bsz):
                        if c_idx < len(cand_pages[b]):
                            st_idx, p_score = cand_pages[b][c_idx]
                        else:
                            st_idx, p_score = 0, float(sync_1d[b, :page_sz].mean().item())
                        sk_f[b, :, s0 : s0 + page_sz, :] = excess_k[b, :, st_idx : st_idx + page_sz, :]
                        sv_f[b, :, s0 : s0 + page_sz, :] = excess_v[b, :, st_idx : st_idx + page_sz, :]
                        self.scar_salience[b, :, s0 : s0 + page_sz] = p_score
                    self.active_slots += page_sz
                elif self.active_slots >= page_sz:
                    act_pages = self.active_slots // page_sz
                    p_means = (
                        self.scar_salience[:, :, : act_pages * page_sz]
                        .contiguous()
                        .view(bsz, n_heads, act_pages, page_sz)[:, 0, :, 0]
                        .detach()
                        .cpu()
                    )
                    for b in range(bsz):
                        if c_idx >= len(cand_pages[b]):
                            continue
                        st_idx, p_score = cand_pages[b][c_idx]
                        min_p = int(p_means[b].argmin().item())
                        if p_score > float(p_means[b, min_p].item()) + 0.05:
                            keep = [p for p in range(act_pages) if p != min_p]
                            if keep:
                                idx_keep = torch.cat(
                                    [torch.arange(p * page_sz, (p + 1) * page_sz, device=sk_f.device) for p in keep]
                                )
                                tot = act_pages * page_sz
                                sk_f[b, :, :tot, :] = torch.cat(
                                    [sk_f[b, :, idx_keep, :], excess_k[b, :, st_idx : st_idx + page_sz, :]], dim=1
                                )
                                sv_f[b, :, :tot, :] = torch.cat(
                                    [sv_f[b, :, idx_keep, :], excess_v[b, :, st_idx : st_idx + page_sz, :]], dim=1
                                )
                                new_s = torch.full((n_heads, page_sz), p_score, device=sk_f.device, dtype=torch.float32)
                                self.scar_salience[b, :, :tot] = torch.cat([self.scar_salience[b, :, idx_keep], new_s], dim=1)
                                p_means[b] = torch.cat([p_means[b, keep], torch.tensor([p_score])])
                            else:
                                sk_f[b, :, :page_sz, :] = excess_k[b, :, st_idx : st_idx + page_sz, :]
                                sv_f[b, :, :page_sz, :] = excess_v[b, :, st_idx : st_idx + page_sz, :]
                                self.scar_salience[b, :, :page_sz] = p_score
                                p_means[b, 0] = p_score

            return sync_1d

        # Durante decode token-a-token com páginas atômicas cheias, protege páginas contra fragmentação
        if page_sz > 1 and self.active_slots == self.num_slots and num_excess < page_sz:
            return sync_1d

        sal_all = sync_1d.unsqueeze(1).expand(-1, n_heads, -1)
        sk_f = self.scar_keys.float()
        sv_f = self.scar_vals.float()
        assert self.scar_decay is not None
        raw_dec = self.scar_decay.view(1, 1, self.num_slots)

        for t in range(min(num_excess, self.num_slots)):
            kt = kf[:, :, t, :]
            vt = vf[:, :, t, :]
            kut = k_unit[:, :, t, :]
            st = sal_all[:, :, t]

            if self.active_slots < self.num_slots:
                s_idx = self.active_slots
                sk_f[:, :, s_idx, :] = kt
                sv_f[:, :, s_idx, :] = vt
                self.scar_salience[:, :, s_idx] = st
                self.active_slots += 1
                continue

            self.scar_salience.mul_(raw_dec.pow(0.15))
            sk_unit = sk_f / torch.norm(sk_f, dim=-1, keepdim=True).clamp(min=1e-6)
            cos_sim = torch.einsum("bhd,bhsd->bhs", kut, sk_unit)
            max_cos, best_s = cos_sim.max(dim=-1)

            is_merge = max_cos >= self.merge_cosine_threshold
            min_sal, min_s = self.scar_salience.min(dim=-1)
            is_replace = (~is_merge) & (st > 1.12 * min_sal)

            target_s = torch.where(is_merge, best_s, min_s)
            s_oh = F.one_hot(target_s, num_classes=self.num_slots).float()

            alpha_k = torch.where(
                is_merge, torch.full_like(st, 0.35), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st))
            )
            alpha_v = torch.where(
                is_merge, torch.full_like(st, 0.78), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st))
            )

            wk = (alpha_k.unsqueeze(-1) * s_oh).unsqueeze(-1)
            wv = (alpha_v.unsqueeze(-1) * s_oh).unsqueeze(-1)

            sk_f = (1.0 - wk) * sk_f + wk * kt.unsqueeze(2)
            sv_f = (1.0 - wv) * sv_f + wv * vt.unsqueeze(2)

            upd_mask = (is_merge | is_replace).unsqueeze(-1).float() * s_oh
            self.scar_salience = (1.0 - upd_mask) * self.scar_salience + upd_mask * st.unsqueeze(-1)

        self.scar_keys.copy_(sk_f.to(self.scar_keys.dtype))
        self.scar_vals.copy_(sv_f.to(self.scar_vals.dtype))
        return sync_1d

    def _update_internal(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        prev_pre_sal: Optional[torch.Tensor] = None,
        prev_in_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        cached_len = self.keys.shape[-2] if (self.is_initialized and self.keys is not None and self.keys.ndim >= 2) else 0
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self._init_scar_bank(key_states, value_states)

        q_len = key_states.shape[-2]
        self.cumulative_length += q_len

        full_k = torch.cat([self.keys, key_states], dim=-2)
        full_v = torch.cat([self.values, value_states], dim=-2)
        total_raw = full_k.shape[-2]

        if total_raw <= self.window_size:
            self.keys = full_k
            self.values = full_v
            return self.keys, self.values, None, None

        slide_len = self.window_size - self.num_sink_tokens
        evict_end = total_raw - slide_len
        tail_vis = max(slide_len, q_len) if self.project_slots else slide_len
        tail_start = total_raw - tail_vis
        sink_vis = min(self.num_sink_tokens, tail_start) if self.project_slots else self.num_sink_tokens
        split_idx = max(self.num_sink_tokens, tail_start)
        pre_excess = split_idx - self.num_sink_tokens
        page_sz = _choose_page_size(self.num_slots)
        halo_end = (
            min(cached_len, split_idx + page_sz - 1)
            if (page_sz > 1 and pre_excess >= page_sz and split_idx <= cached_len)
            else split_idx
        )

        obs_k = full_k[..., -slide_len:, :]
        obs_v = full_v[..., -slide_len:, :]
        s_v = full_v[..., : self.num_sink_tokens, :] if self.num_sink_tokens > 0 else None
        single_shot_prefill = self.active_slots == 0 and split_idx == self.num_sink_tokens

        # 1. Tokens desalojados antes de `tail_k` (+ halo causal de `page_sz - 1` contra cortes de fronteira!).
        out_pre_sal: Optional[torch.Tensor] = None
        pre_tail_k = full_k[..., self.num_sink_tokens : halo_end, :]
        pre_tail_v = full_v[..., self.num_sink_tokens : halo_end, :]
        if pre_excess > 0 and pre_tail_k.shape[-2] > 0:
            out_pre_sal = self._ingest_evicted_tokens(
                pre_tail_k,
                pre_tail_v,
                obs_k=obs_k,
                obs_v=obs_v,
                sink_v=s_v,
                prev_sync_sal=prev_pre_sal,
                use_obs=False,
                retire_len=pre_excess,
            )

        tail_k = full_k[..., -tail_vis:, :]
        tail_v = full_v[..., -tail_vis:, :]
        vis_sink_k = full_k[..., :sink_vis, :]
        vis_sink_v = full_v[..., :sink_vis, :]

        if (
            self.project_slots
            and self.num_slots > 0
            and tail_start > self.num_sink_tokens
            and self.active_slots > 0
            and self.scar_keys is not None
            and self.scar_vals is not None
            and self.scache is not None
        ):
            act = self.active_slots
            k_slots = self.scar_keys[:, :, :act, :]
            v_proto = self.scar_vals[:, :, :act, :].float()
            k_unit = k_slots.float() / torch.norm(k_slots.float(), dim=-1, keepdim=True).clamp(min=1e-6)
            v_delta = torch.einsum("bhsd,hvd->bhsv", k_unit, self.scache.W_scar)
            v_slots = ((1.0 - self.delta_blend) * v_proto + self.delta_blend * v_delta).to(self.values.dtype)

            if sink_vis > 0:
                ret_k = torch.cat([vis_sink_k, k_slots, tail_k], dim=-2)
                ret_v = torch.cat([vis_sink_v, v_slots, tail_v], dim=-2)
            else:
                ret_k = torch.cat([k_slots, tail_k], dim=-2)
                ret_v = torch.cat([v_slots, tail_v], dim=-2)
        elif sink_vis > 0:
            ret_k = torch.cat([vis_sink_k, tail_k], dim=-2)
            ret_v = torch.cat([vis_sink_v, tail_v], dim=-2)
        else:
            ret_k, ret_v = tail_k, tail_v

        # 2. Tokens já presentes dentro de `tail_k` (`[split_idx : evict_end]`) são absorvidos APÓS
        # construir `(ret_k, ret_v)` (zero vazamento causal, zero duplicação no prefill).
        out_in_sal: Optional[torch.Tensor] = None
        in_tail_k = full_k[..., split_idx:evict_end, :]
        in_tail_v = full_v[..., split_idx:evict_end, :]
        if in_tail_k.shape[-2] > 0:
            out_in_sal = self._ingest_evicted_tokens(
                in_tail_k,
                in_tail_v,
                obs_k=obs_k,
                obs_v=obs_v,
                sink_v=s_v,
                prev_sync_sal=prev_in_sal,
                use_obs=single_shot_prefill,
            )

        self.total_tokens_absorbed += evict_end - self.num_sink_tokens

        # Armazenamento físico estritamente congelado em `window_size` O(1)!
        if self.num_sink_tokens > 0:
            sink_k = full_k[..., : self.num_sink_tokens, :]
            sink_v = full_v[..., : self.num_sink_tokens, :]
            slide_k = full_k[..., -slide_len:, :]
            slide_v = full_v[..., -slide_len:, :]
            self.keys = torch.cat([sink_k, slide_k], dim=-2)
            self.values = torch.cat([sink_v, slide_v], dim=-2)
        else:
            self.keys = full_k[..., -self.window_size :, :].clone()
            self.values = full_v[..., -self.window_size :, :].clone()

        return ret_k, ret_v, out_pre_sal, out_in_sal

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret_k, ret_v, _, _ = self._update_internal(key_states, value_states)
        return ret_k, ret_v

    @property
    def physical_bytes(self) -> int:
        if not self.is_initialized or self.keys is None:
            return 0
        return int((self.keys.numel() + self.values.numel()) * self.keys.element_size())

    @property
    def scache_bytes(self) -> int:
        total = 0
        if self.scache is not None:
            for p in self.scache.parameters():
                total += int(p.numel() * p.element_size())
            for b in self.scache.buffers():
                total += int(b.numel() * b.element_size())
        for buf in (self.scar_keys, self.scar_vals, self.scar_salience, self.scar_decay):
            if buf is not None:
                total += int(buf.numel() * buf.element_size())
        return total


@dataclass
class SCacheLayerSnapshot:
    """Snapshot atômico O(1) de uma camada `SCacheDynamicLayer` para rollback especulativo exato."""

    is_initialized: bool
    prev_cached_len: int
    cumulative_length: int
    active_slots: int
    total_tokens_absorbed: int
    err_baseline_ema: float
    err_mad_ema: float
    err_chunks_seen: int
    fast_window_only: bool
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    W_scar: Optional[torch.Tensor] = None
    scars_count: Optional[torch.Tensor] = None
    scar_keys: Optional[torch.Tensor] = None
    scar_vals: Optional[torch.Tensor] = None
    scar_salience: Optional[torch.Tensor] = None


@dataclass
class SCacheSnapshot:
    """Snapshot completo de `SCacheDynamicCache` (fast-path zero-copy quando dentro de `window_size`)."""

    num_layers: int
    fast_window_only: bool
    layers: List[SCacheLayerSnapshot]


class SpeculativeCaptureContext:
    """Gerencia captura causal temporária de `(K, V)` e commit/rollback atômico O(1) sem reter bytes."""

    def __init__(self, cache: "SCacheDynamicCache", snapshot: SCacheSnapshot, multi_step: bool = False):
        self.cache = cache
        self.snapshot = snapshot
        self.multi_step = bool(multi_step)
        self.captured_steps: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = []
        self._cur_step: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def record(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if layer_idx == 0 and self._cur_step:
            self.captured_steps.append(self._cur_step)
            self._cur_step = {}
        self._cur_step[layer_idx] = (k.detach().clone(), v.detach().clone())

    def finalize_capture(self) -> None:
        if self._cur_step:
            self.captured_steps.append(self._cur_step)
            self._cur_step = {}

    @property
    def total_submitted_tokens(self) -> int:
        if not self.captured_steps:
            return 0
        return sum(step[0][0].shape[-2] for step in self.captured_steps if 0 in step)

    @torch.no_grad()
    def commit(self, num_valid_tokens: int) -> None:
        """Confirma exatamente `num_valid_tokens` e descarta átomos rejeitados com 0 bytes residuais."""
        self.finalize_capture()
        total_sub = self.total_submitted_tokens
        if num_valid_tokens < 0 or num_valid_tokens > total_sub:
            raise ValueError(f"Invalid num_valid_tokens={num_valid_tokens} for total_sub={total_sub}")

        if num_valid_tokens == total_sub:
            self.clear()
            return

        if self.snapshot.fast_window_only:
            if self.snapshot.num_layers == 0 and num_valid_tokens == 0:
                self.cache.layers.clear()
            else:
                for l_idx, lyr in enumerate(self.cache.layers):
                    assert isinstance(lyr, SCacheDynamicLayer)
                    if l_idx < self.snapshot.num_layers:
                        snap_l = self.snapshot.layers[l_idx]
                        prev_len = snap_l.prev_cached_len
                        prev_cum = snap_l.cumulative_length
                        was_init = snap_l.is_initialized
                    else:
                        prev_len = 0
                        prev_cum = 0
                        was_init = False

                    if not was_init and num_valid_tokens == 0:
                        lyr.keys = torch.tensor([], dtype=torch.float32)
                        lyr.values = torch.tensor([], dtype=torch.float32)
                        lyr.is_initialized = False
                        lyr.cumulative_length = 0
                    else:
                        keep = prev_len + num_valid_tokens
                        assert lyr.keys is not None and lyr.values is not None
                        lyr.keys = lyr.keys[..., :keep, :].clone()
                        lyr.values = lyr.values[..., :keep, :].clone()
                        lyr.cumulative_length = prev_cum + num_valid_tokens
            self.cache._step_pre_sal = None
            self.cache._step_in_sal = None
            self.clear()
            return

        # Full O(1) state rollback (restores W_scar, micro-page slots, EMAs & window) + causal KV prefix replay
        self.cache.restore(self.snapshot)
        if num_valid_tokens > 0 and self.captured_steps:
            if self.multi_step:
                rem = num_valid_tokens
                for step_dict in self.captured_steps:
                    if rem <= 0:
                        break
                    step_q = step_dict[0][0].shape[-2]
                    take = min(rem, step_q)
                    for l_idx in sorted(step_dict.keys()):
                        ks, vs = step_dict[l_idx]
                        self.cache.update(ks[..., :take, :].clone(), vs[..., :take, :].clone(), l_idx)
                    rem -= take
            else:
                step_dict = self.captured_steps[0]
                for l_idx in sorted(step_dict.keys()):
                    ks, vs = step_dict[l_idx]
                    self.cache.update(
                        ks[..., :num_valid_tokens, :].clone(),
                        vs[..., :num_valid_tokens, :].clone(),
                        l_idx,
                    )
        self.clear()

    def clear(self) -> None:
        self.captured_steps.clear()
        self._cur_step.clear()
        self.snapshot.layers.clear()


class SCacheDynamicCache(DynamicCache):
    """Drop-in Replacement O(1) para DynamicCache do HuggingFace compatível com qualquer LLM."""

    def __init__(
        self,
        window_size: int = 64,
        num_slots: int = 8,
        num_sink_tokens: int = 4,
        project_slots: bool = False,
        merge_cosine_threshold: float = 0.995,
        delta_blend: float = 0.05,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = int(num_sink_tokens)
        self.project_slots = bool(project_slots)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.delta_blend = float(delta_blend)
        self._step_pre_sal: Optional[List[List[float]]] = None
        self._step_in_sal: Optional[List[List[float]]] = None
        self._spec_capture_ctx: Optional[SpeculativeCaptureContext] = None
        self.layer_class_to_replicate = lambda: SCacheDynamicLayer(
            window_size=self.window_size,
            num_slots=self.num_slots,
            num_sink_tokens=self.num_sink_tokens,
            project_slots=self.project_slots,
            merge_cosine_threshold=self.merge_cosine_threshold,
            delta_blend=self.delta_blend,
        )

    @property
    def prefill_chunk_size(self) -> int:
        eff_sinks = min(self.num_sink_tokens, self.window_size // 2) if self.num_sink_tokens > 0 else 0
        slide = max(1, self.window_size - eff_sinks)
        page_sz = _choose_page_size(self.num_slots)
        if page_sz > 1 and slide > 2 * page_sz:
            return slide - page_sz
        return slide

    @torch.no_grad()
    def prefill(self, model: Any, input_ids: torch.Tensor, **model_kwargs: Any) -> Any:
        """Executa prefill causal em blocos O(1) (`prefill_chunk_size`), garantindo sincronia profunda de camadas."""
        step = self.prefill_chunk_size
        out = None
        for s0 in range(0, input_ids.shape[1], step):
            out = model(input_ids[:, s0 : s0 + step], past_key_values=self, use_cache=True, **model_kwargs)
        return out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._spec_capture_ctx is not None:
            self._spec_capture_ctx.record(layer_idx, key_states, value_states)

        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

        layer = self.layers[layer_idx]
        if layer_idx == 0:
            self._step_pre_sal = None
            self._step_in_sal = None
            # Sincronia Causal Profunda [L//2 : L] com 0 Bytes Extras de Armazenamento!
            if (
                isinstance(layer, SCacheDynamicLayer)
                and len(self.layers) >= 4
                and layer.is_initialized
                and layer.keys is not None
                and layer.num_slots > 0
            ):
                q_len = key_states.shape[-2]
                cached_len = layer.keys.shape[-2]
                total_raw = cached_len + q_len
                if total_raw > layer.window_size:
                    slide_len = layer.window_size - layer.num_sink_tokens
                    tail_vis = max(slide_len, q_len) if layer.project_slots else slide_len
                    tail_start = total_raw - tail_vis
                    split_idx = max(layer.num_sink_tokens, tail_start)
                    pre_excess = split_idx - layer.num_sink_tokens
                    page_sz = _choose_page_size(layer.num_slots)
                    halo_end = (
                        min(cached_len, split_idx + page_sz - 1)
                        if (page_sz > 1 and pre_excess >= page_sz)
                        else split_idx
                    )
                    if pre_excess > 0 and split_idx <= cached_len:
                        deep_layers = [
                            lyr
                            for lyr in self.layers[len(self.layers) // 2 :]
                            if isinstance(lyr, SCacheDynamicLayer)
                            and lyr.is_initialized
                            and lyr.keys is not None
                            and lyr.keys.shape[-2] == cached_len
                            and lyr.scache is not None
                        ]
                        if deep_layers:
                            acc: Optional[torch.Tensor] = None
                            for d_lyr in deep_layers:
                                ex_k = d_lyr.keys[..., d_lyr.num_sink_tokens : halo_end, :]
                                ex_v = d_lyr.values[..., d_lyr.num_sink_tokens : halo_end, :]
                                s_v = d_lyr.values[..., : d_lyr.num_sink_tokens, :] if d_lyr.num_sink_tokens > 0 else None
                                s1 = _layer_widrow_hoff_salience_1d(d_lyr, ex_k, ex_v, sink_v=s_v)
                                acc = s1 if acc is None else (acc + s1)
                            assert acc is not None
                            acc = acc / float(len(deep_layers))
                            # Armazenado como lista pura Python -> 0 bytes extras de Tensor na auditoria recursiva!
                            self._step_pre_sal = acc.detach().cpu().tolist()

        if isinstance(layer, SCacheDynamicLayer):
            prev_p = (
                torch.tensor(self._step_pre_sal, dtype=torch.float32, device=key_states.device)
                if self._step_pre_sal is not None
                else None
            )
            prev_i = (
                torch.tensor(self._step_in_sal, dtype=torch.float32, device=key_states.device)
                if self._step_in_sal is not None
                else None
            )
            ret_k, ret_v, out_p, out_i = layer._update_internal(
                key_states, value_states, prev_pre_sal=prev_p, prev_in_sal=prev_i
            )
            if self._step_pre_sal is None and out_p is not None:
                self._step_pre_sal = out_p.detach().cpu().tolist()
            if self._step_in_sal is None and out_i is not None:
                self._step_in_sal = out_i.detach().cpu().tolist()
            return ret_k, ret_v

        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    @property
    def total_physical_bytes(self) -> int:
        return sum(
            layer.physical_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_scache_bytes(self) -> int:
        return sum(
            layer.scache_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_memory_bytes(self) -> int:
        return self.total_physical_bytes + self.total_scache_bytes

    def get_max_physical_length(self) -> int:
        lengths = [
            layer.keys.shape[-2]
            for layer in self.layers
            if isinstance(layer, SCacheDynamicLayer) and layer.keys is not None
        ]
        return max(lengths) if lengths else 0

    @torch.no_grad()
    def snapshot(self, expected_new_tokens: int = 0) -> SCacheSnapshot:
        """Captura snapshot atômico O(1) do cache (zero alocação de tensores se dentro de `window_size`)."""
        max_cur_len = self.get_max_physical_length()
        fast_only = bool(expected_new_tokens > 0 and (max_cur_len + expected_new_tokens) <= self.window_size)
        snaps: List[SCacheLayerSnapshot] = []
        for layer in self.layers:
            assert isinstance(layer, SCacheDynamicLayer)
            c_len = layer.keys.shape[-2] if (layer.is_initialized and layer.keys is not None and layer.keys.ndim >= 2) else 0
            if fast_only:
                snaps.append(
                    SCacheLayerSnapshot(
                        is_initialized=bool(layer.is_initialized),
                        prev_cached_len=int(c_len),
                        cumulative_length=int(layer.cumulative_length),
                        active_slots=int(layer.active_slots),
                        total_tokens_absorbed=int(layer.total_tokens_absorbed),
                        err_baseline_ema=float(layer.err_baseline_ema),
                        err_mad_ema=float(layer.err_mad_ema),
                        err_chunks_seen=int(layer.err_chunks_seen),
                        fast_window_only=True,
                    )
                )
            else:
                snaps.append(
                    SCacheLayerSnapshot(
                        is_initialized=bool(layer.is_initialized),
                        prev_cached_len=int(c_len),
                        cumulative_length=int(layer.cumulative_length),
                        active_slots=int(layer.active_slots),
                        total_tokens_absorbed=int(layer.total_tokens_absorbed),
                        err_baseline_ema=float(layer.err_baseline_ema),
                        err_mad_ema=float(layer.err_mad_ema),
                        err_chunks_seen=int(layer.err_chunks_seen),
                        fast_window_only=False,
                        keys=layer.keys.detach().clone() if (layer.is_initialized and layer.keys is not None) else None,
                        values=layer.values.detach().clone() if (layer.is_initialized and layer.values is not None) else None,
                        W_scar=layer.scache.W_scar.detach().clone() if layer.scache is not None else None,
                        scars_count=layer.scache.scars_count.detach().clone() if layer.scache is not None else None,
                        scar_keys=layer.scar_keys.detach().clone() if layer.scar_keys is not None else None,
                        scar_vals=layer.scar_vals.detach().clone() if layer.scar_vals is not None else None,
                        scar_salience=layer.scar_salience.detach().clone() if layer.scar_salience is not None else None,
                    )
                )
        return SCacheSnapshot(num_layers=len(snaps), fast_window_only=fast_only, layers=snaps)

    @torch.no_grad()
    def restore(self, snap: SCacheSnapshot) -> None:
        """Restaura estado O(1) exato (`W_scar`, slots de micro-páginas, EMAs e janela) sem vazar storage."""
        if snap.num_layers == 0:
            self.layers.clear()
            self._step_pre_sal = None
            self._step_in_sal = None
            return

        del self.layers[snap.num_layers :]
        for l_idx, snap_l in enumerate(snap.layers):
            lyr = self.layers[l_idx]
            assert isinstance(lyr, SCacheDynamicLayer)
            lyr.is_initialized = snap_l.is_initialized
            lyr.cumulative_length = snap_l.cumulative_length
            lyr.active_slots = snap_l.active_slots
            lyr.total_tokens_absorbed = snap_l.total_tokens_absorbed
            lyr.err_baseline_ema = snap_l.err_baseline_ema
            lyr.err_mad_ema = snap_l.err_mad_ema
            lyr.err_chunks_seen = snap_l.err_chunks_seen

            if not snap_l.is_initialized:
                lyr.keys = torch.tensor([], dtype=torch.float32)
                lyr.values = torch.tensor([], dtype=torch.float32)
                lyr.scache = None
                lyr.scar_keys = None
                lyr.scar_vals = None
                lyr.scar_salience = None
                lyr.scar_decay = None
                continue

            if snap_l.fast_window_only:
                assert lyr.keys is not None and lyr.values is not None
                lyr.keys = lyr.keys[..., : snap_l.prev_cached_len, :].clone()
                lyr.values = lyr.values[..., : snap_l.prev_cached_len, :].clone()
            else:
                assert snap_l.keys is not None and snap_l.values is not None
                lyr.keys = snap_l.keys.clone()
                lyr.values = snap_l.values.clone()
                if snap_l.W_scar is not None and lyr.scache is not None:
                    lyr.scache.W_scar.copy_(snap_l.W_scar)
                if snap_l.scars_count is not None and lyr.scache is not None:
                    lyr.scache.scars_count.copy_(snap_l.scars_count)
                if snap_l.scar_keys is not None and lyr.scar_keys is not None:
                    lyr.scar_keys.copy_(snap_l.scar_keys)
                if snap_l.scar_vals is not None and lyr.scar_vals is not None:
                    lyr.scar_vals.copy_(snap_l.scar_vals)
                if snap_l.scar_salience is not None and lyr.scar_salience is not None:
                    lyr.scar_salience.copy_(snap_l.scar_salience)

        self._step_pre_sal = None
        self._step_in_sal = None

    @contextmanager
    def speculative_capture(
        self,
        expected_new_tokens: int,
        multi_step: bool = False,
    ) -> Iterator[SpeculativeCaptureContext]:
        """Context manager de verificação especulativa Ghost-Jump com rollback O(1) exato e 0 bytes residuais."""
        snap = self.snapshot(expected_new_tokens=expected_new_tokens)
        ctx = SpeculativeCaptureContext(self, snap, multi_step=multi_step)
        prev_ctx = self._spec_capture_ctx
        self._spec_capture_ctx = ctx
        try:
            yield ctx
        finally:
            ctx.finalize_capture()
            self._spec_capture_ctx = prev_ctx
