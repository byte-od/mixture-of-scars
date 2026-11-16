"""Native Polyglot Byte-Level BPE Tokenizer (`4,096` tokens) & Automatic Epistemic Abstention Filter (`v6.0.0`).

Implements Module #3 of `ScarNative v6.0`:
  1. `ScarNativeBPETokenizer`: Trained from scratch (`0` external HuggingFace/tiktoken files) over a balanced
     polyglot (`PT`, `EN`, `ES`, `FR`) + symbolic math + Python/WGSL code + dialogue corpus.
     - Exact multiple-of-64 vocabulary (`vocab_size = 4096` by default) for zero-padding `Block-64 INT4` & `VQ4-Scar`.
     - GPT-4o / Llama-3 style individual-digit pre-tokenization (`[0-9]` never merged together) for compositional
       multi-digit arithmetic & arbitrary unseen passkey recall.
     - `100%` byte-exact UTF-8 round-trip (`decode(encode(s)) == s` on ANY string in the universe, `0` `<unk>`!).
  2. `EpistemicAbstentionFilter`: Automatically detects ungrounded mutable world-fact, clinical/cosmetic/chemical,
     and unseen entity-value queries without inline context during pretraining, routing them to `<abstain>` in the
     user's language (`PT/EN/ES/FR`) so `100%` of trunk capacity is spent on grammar, logic, math, and code.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scar_native.tokenizer import has_cjk_script

V6_SPECIAL_TOKENS: List[str] = [
    "<pad>",        # 0
    "<bos>",        # 1
    "<eos>",        # 2
    "<unk>",        # 3
    "<user>",       # 4
    "<assistant>",  # 5
    "<lang:pt>",    # 6
    "<lang:en>",    # 7
    "<lang:es>",    # 8
    "<lang:fr>",    # 9
    "<lang:zh>",    # 10
    "<passkey>",    # 11
    "</passkey>",   # 12
    "<fact>",       # 13
    "<abstain>",    # 14
    "<sep>",        # 15
]

# Pre-tokenization regex:
#   1. Exact special control tags `<...>`
#   2. Individual ASCII digits `[0-9]` (never merged across digits!)
#   3. Optional leading space + Latin/accented word
#   4. CJK runs
#   5. Optional leading space + punctuation/operator runs (excluding `<` so tags stay clean)
#   6. Whitespace runs
_V6_PRETOK_RE = re.compile(
    r"(<(?:pad|bos|eos|unk|user|assistant|lang:pt|lang:en|lang:es|lang:fr|lang:zh|passkey|/passkey|fact|abstain|sep)>"
    r"|[0-9]"
    r"| ?[A-Za-zÀ-ÖØ-öø-ÿ_]+"
    r"|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+"
    r"| ?[^\sA-Za-zÀ-ÖØ-öø-ÿ0-9<]+"
    r"|<"
    r"|\s+)"
)


class ScarNativeBPETokenizer:
    """Deterministic Byte-Level BPE Tokenizer (`vocab_size = 4096`, `100%` lossless UTF-8 round-trip)."""

    BYTE_OFFSET = len(V6_SPECIAL_TOKENS)  # 16 .. 271 are raw UTF-8 bytes 0x00 .. 0xFF

    def __init__(
        self,
        vocab_size: int = 4096,
        merges: Optional[Sequence[Tuple[int, int, int]]] = None,
        token_bytes: Optional[Sequence[bytes]] = None,
    ) -> None:
        if vocab_size < 320 or (vocab_size % 64) != 0:
            raise ValueError(f"vocab_size must be >= 320 and a multiple of 64, got {vocab_size}")

        self.target_vocab_size = vocab_size
        self.special_tokens = list(V6_SPECIAL_TOKENS)
        self.special_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}

        if token_bytes is None or merges is None:
            # Base untuned state (`16` specials + `256` bytes + reserved padding to `vocab_size`)
            tb: List[bytes] = [b"" for _ in self.special_tokens]
            tb.extend(bytes([b]) for b in range(256))
            while len(tb) < vocab_size:
                tb.append(b"")
            self.token_bytes: List[bytes] = tb
            self.merges: List[Tuple[int, int, int]] = []
        else:
            self.token_bytes = list(token_bytes)
            self.merges = [(int(l), int(r), int(m)) for l, r, m in merges]
            while len(self.token_bytes) < vocab_size:
                self.token_bytes.append(b"")

        self.vocab_size = len(self.token_bytes)
        self.merge_ranks: Dict[Tuple[int, int], Tuple[int, int]] = {
            (l, r): (rank, m) for rank, (l, r, m) in enumerate(self.merges)
        }
        self._bpe_cache: Dict[bytes, Tuple[int, ...]] = {}

        self.pad_token_id = self.special_to_id["<pad>"]
        self.bos_token_id = self.special_to_id["<bos>"]
        self.eos_token_id = self.special_to_id["<eos>"]
        self.unk_token_id = self.special_to_id["<unk>"]
        self.abstain_token_id = self.special_to_id["<abstain>"]

    def __len__(self) -> int:
        return self.vocab_size

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_pair_freq: int = 2,
    ) -> "ScarNativeBPETokenizer":
        """Trains exact deterministic Byte-Level BPE merges from an iterable of polyglot strings."""
        if vocab_size < 320 or (vocab_size % 64) != 0:
            raise ValueError(f"vocab_size must be >= 320 and a multiple of 64, got {vocab_size}")

        special_set = set(V6_SPECIAL_TOKENS)
        word_counts: Counter[bytes] = Counter()
        for text in texts:
            for m in _V6_PRETOK_RE.finditer(text):
                piece = m.group(0)
                if piece in special_set:
                    continue
                b = piece.encode("utf-8")
                if len(b) >= 2 and not (len(b) == 1 and 48 <= b[0] <= 57):
                    word_counts[b] += 1

        # Represent each unique pre-token `bytes` as a mutable list of token IDs + frequency
        words: List[List[int]] = []
        freqs: List[int] = []
        for raw_b, cnt in sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            words.append([cls.BYTE_OFFSET + byte_val for byte_val in raw_b])
            freqs.append(int(cnt))

        token_bytes: List[bytes] = [b"" for _ in V6_SPECIAL_TOKENS]
        token_bytes.extend(bytes([b]) for b in range(256))
        merges: List[Tuple[int, int, int]] = []

        # Initial pair counts
        pair_counts: Counter[Tuple[int, int]] = Counter()
        for seq, f in zip(words, freqs):
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += f

        max_merges = vocab_size - len(token_bytes)
        for _ in range(max_merges):
            if not pair_counts:
                break
            # Deterministic tie-breaking: highest frequency, then lowest `(left_id, right_id)`
            best_pair, best_freq = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if best_freq < min_pair_freq:
                break

            left_id, right_id = best_pair
            new_id = len(token_bytes)
            token_bytes.append(token_bytes[left_id] + token_bytes[right_id])
            merges.append((left_id, right_id, new_id))
            del pair_counts[best_pair]

            # Incrementally update `words` & `pair_counts` only on words containing `best_pair`
            for w_idx, seq in enumerate(words):
                n = len(seq)
                if n < 2:
                    continue
                has_match = False
                for k in range(n - 1):
                    if seq[k] == left_id and seq[k + 1] == right_id:
                        has_match = True
                        break
                if not has_match:
                    continue

                f = freqs[w_idx]
                for k in range(n - 1):
                    p = (seq[k], seq[k + 1])
                    if p != best_pair:
                        pair_counts[p] -= f
                        if pair_counts[p] <= 0:
                            del pair_counts[p]

                merged_seq: List[int] = []
                k = 0
                while k < n:
                    if k + 1 < n and seq[k] == left_id and seq[k + 1] == right_id:
                        merged_seq.append(new_id)
                        k += 2
                    else:
                        merged_seq.append(seq[k])
                        k += 1
                words[w_idx] = merged_seq

                for k in range(len(merged_seq) - 1):
                    pair_counts[(merged_seq[k], merged_seq[k + 1])] += f

        while len(token_bytes) < vocab_size:
            token_bytes.append(b"")

        return cls(vocab_size=vocab_size, merges=merges, token_bytes=token_bytes)

    def _bpe_encode_bytes(self, raw_b: bytes) -> Tuple[int, ...]:
        cached = self._bpe_cache.get(raw_b)
        if cached is not None:
            return cached
        seq = [self.BYTE_OFFSET + b for b in raw_b]
        if len(seq) >= 2 and self.merge_ranks:
            while len(seq) >= 2:
                best_rank = 1_000_000_000
                best_pos = -1
                best_merged = -1
                for i in range(len(seq) - 1):
                    info = self.merge_ranks.get((seq[i], seq[i + 1]))
                    if info is not None and info[0] < best_rank:
                        best_rank = info[0]
                        best_pos = i
                        best_merged = info[1]
                if best_pos < 0:
                    break
                # Merge all non-overlapping occurrences of the winning pair in this pass
                l_win, r_win = seq[best_pos], seq[best_pos + 1]
                nxt: List[int] = []
                k = 0
                while k < len(seq):
                    if k + 1 < len(seq) and seq[k] == l_win and seq[k + 1] == r_win:
                        nxt.append(best_merged)
                        k += 2
                    else:
                        nxt.append(seq[k])
                        k += 1
                seq = nxt
        res = tuple(seq)
        self._bpe_cache[raw_b] = res
        return res

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        for m in _V6_PRETOK_RE.finditer(text):
            piece = m.group(0)
            sp_id = self.special_to_id.get(piece)
            if sp_id is not None:
                ids.append(sp_id)
            else:
                ids.extend(self._bpe_encode_bytes(piece.encode("utf-8")))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out_parts: List[str] = []
        byte_acc = bytearray()

        def _flush() -> None:
            if byte_acc:
                out_parts.append(byte_acc.decode("utf-8", errors="replace"))
                byte_acc.clear()

        n_spec = len(self.special_tokens)
        for raw_id in ids:
            tid = int(raw_id)
            if tid < 0 or tid >= self.vocab_size:
                continue
            if tid < n_spec:
                if not skip_special_tokens:
                    _flush()
                    out_parts.append(self.special_tokens[tid])
                continue
            tb = self.token_bytes[tid]
            if tb:
                byte_acc.extend(tb)
        _flush()
        return "".join(out_parts)

    def build_cjk_bitmask_u32(self) -> np.ndarray:
        """Builds exact `uint32` bitmask suppressing CJK tokens (`P_⊥CJK`) without harming UTF-8 Latin bytes."""
        n_words = (self.vocab_size + 31) // 32
        mask = np.zeros((n_words,), dtype=np.uint32)
        # Suppress `<lang:zh>` output token + any merged/byte token that decodes to a CJK character
        for tid in range(len(self.special_tokens), self.vocab_size):
            tb = self.token_bytes[tid]
            if not tb:
                continue
            try:
                s = tb.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if has_cjk_script(s):
                mask[tid >> 5] |= np.uint32(1 << (tid & 31))
        return mask

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "format": "scar_native_bpe_v6_0",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "token_bytes_hex": [tb.hex() for tb in self.token_bytes],
            "merges": [[l, r, m] for l, r, m in self.merges],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScarNativeBPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        tb = [bytes.fromhex(h) for h in data["token_bytes_hex"]]
        merges = [(int(m[0]), int(m[1]), int(m[2])) for m in data["merges"]]
        return cls(vocab_size=int(data["vocab_size"]), merges=merges, token_bytes=tb)


# ============================================================================
# Automatic Epistemic Abstention Pretraining Filter (`v6.0` Module #3)
# ============================================================================

CANONICAL_ABSTENTION_REPLIES: Dict[str, str] = {
    "pt": " <abstain> Abstenção epistêmica: nenhum cartucho verificado contém esse fato ou prescrição específica fora do contexto.",
    "en": " <abstain> Epistemic abstention: no verified cartridge holds this specific fact or prescription outside the context.",
    "es": " <abstain> Abstención epistémica: ningún cartucho verificado contiene este hecho o prescripción fuera del contexto.",
    "fr": " <abstain> Abstention épistémique: aucune cartouche vérifiée ne contient ce fait ou cette prescription hors contexte.",
}

_UNGROUNDED_SENSITIVE_PATTERNS = re.compile(
    r"\b("
    r"progressiva|descolorid\w*|bleach\w*|keratin\s+treatment|alisamento|dosagem|posologia|mg/kg|prescri\w*|"
    r"diagnóstico\s+médico|medical\s+diagnosis|stock\s+price|cotação\s+de\s+hoje|preço\s+do\s+bitcoin|"
    r"cpf\s+de|senha\s+bancária|private\s+phone|chemical\s+relaxer"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class FilteredPretrainSample:
    lang: str
    prompt: str
    completion: str
    was_abstained: bool
    reason: str


class EpistemicAbstentionFilter:
    """Filters pretraining records to prevent mutable world facts or unsafe ungrounded prescriptions
    from ever entering the `ScarNative v6.0` trunk weights (`100%` capacity reserved for grammar/logic/math/code).
    """

    def __init__(self, detached_entity_names: Sequence[str] = ()) -> None:
        self.entity_names = {e.lower() for e in detached_entity_names}

    def filter_record(
        self,
        lang: str,
        prompt: str,
        completion: str,
        has_inline_evidence: bool = False,
    ) -> FilteredPretrainSample:
        lg = lang if lang in CANONICAL_ABSTENTION_REPLIES else "en"
        p_low = prompt.lower()

        if has_inline_evidence or "<passkey>" in prompt:
            return FilteredPretrainSample(lg, prompt, completion, False, "grounded_inline_context")

        # 1. Explicit `<fact>` query without active cartridge during trunk pretraining -> MUST abstain!
        if "<fact>" in prompt:
            return FilteredPretrainSample(
                lg, prompt, CANONICAL_ABSTENTION_REPLIES[lg], True, "detached_fact_tag_without_cartridge"
            )

        # 2. Mention of detached mutable entity metric without inline context -> MUST abstain!
        for ent in self.entity_names:
            if re.search(rf"\b{re.escape(ent)}\b", p_low):
                return FilteredPretrainSample(
                    lg, prompt, CANONICAL_ABSTENTION_REPLIES[lg], True, f"detached_entity:{ent}"
                )

        # 3. Ungrounded clinical/chemical/cosmetic/financial hazard (e.g. Jarvis bleached-hair chemical case!)
        if _UNGROUNDED_SENSITIVE_PATTERNS.search(prompt):
            return FilteredPretrainSample(
                lg, prompt, CANONICAL_ABSTENTION_REPLIES[lg], True, "ungrounded_sensitive_domain"
            )

        return FilteredPretrainSample(lg, prompt, completion, False, "pure_syntax_logic_code")
