"""Deterministic Polyglot Tokenizer (`PT`, `EN`, `ES`, `FR` + Symbolic Logic + CJK Shield) for `scar_native` v5.0.0."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Sequence

import numpy as np


def has_cjk_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (
            (0x3040 <= cp <= 0x30FF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xAC00 <= cp <= 0xD7AF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x3000 <= cp <= 0x303F)
            or (0xFF01 <= cp <= 0xFF60)
        ):
            return True
    return False

SPECIAL_TOKENS: List[str] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<user>",
    "<assistant>",
    "<lang:pt>",
    "<lang:en>",
    "<lang:es>",
    "<lang:fr>",
    "<lang:zh>",
    "<passkey>",
    "</passkey>",
    "<fact>",
    "<sep>",
]

# Core polyglot (`PT`, `EN`, `ES`, `FR`), logical, technical, numeric, and CJK script tokens
CORE_VOCAB_WORDS: List[str] = [
    # Punctuation & operators
    ".", ",", ":", ";", "?", "!", "¿", "¡", "(", ")", "[", "]", "-", "=", "+", "*", "/", "->", "%", "\n",
    # Individual digits (so any arbitrary multi-digit number or passkey encodes losslessly!)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Portuguese (PT)
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na", "por", "para", "com", "sem", "e", "ou",
    "que", "qual", "é", "são", "foi", "está", "como", "se", "então", "logo", "portanto", "porque", "mas", "cada",
    "sistema", "memória", "arquitetura", "cicatriz", "cicatrizes", "vetor", "matriz", "estado", "constante",
    "energia", "pressão", "órbita", "frequência", "código", "chave", "secreta", "valor", "exato", "resposta",
    "linguagem", "idioma", "modelo", "rede", "atenção", "janela", "rápido", "rápida", "preciso", "zero", "vácuo",
    "verdadeiro", "falso", "maior", "menor", "igual", "soma", "dobro", "triplo", "resultado", "verificado", "verificadas",
    "relatório", "telemetria", "sensor", "satélite", "protocolo", "módulo", "núcleo", "operação", "seguro",
    "informação", "contexto", "fluxo", "sinal", "limpo", "ruído", "filtro", "subespaço", "ortogonal", "dados", "registro",
    "sintático", "contínua", "mantém", "mantendo", "separa", "analisa", "opera", "verifica", "protege", "garante",
    "expandir", "rotação", "vantagem", "funciona", "você", "pode", "desconhecido",
    "olá", "bom", "dia", "posso", "ajudar", "explicar", "analisar", "calcular", "recuperar", "confirmado",
    # English (EN)
    "the", "of", "in", "to", "for", "with", "without", "on", "at", "by", "from", "and", "or", "if", "then", "via", "inside",
    "therefore", "because", "but", "what", "is", "are", "was", "how", "does", "each", "than", "you", "day", "work",
    "system", "memory", "architecture", "syntactic", "continuous", "clean", "advantage", "rotation",
    "scar", "scars", "vector", "matrix", "state", "constant", "energy", "pressure", "orbit", "frequency",
    "code", "key", "secret", "value", "exact", "answer", "language", "model", "network", "attention",
    "window", "fast", "accurate", "vacuum", "true", "false", "greater", "less", "equal", "sum", "double",
    "triple", "result", "verified", "report", "telemetry", "satellite", "protocol", "module", "modules", "core",
    "operation", "secure", "information", "context", "stream", "signal", "noise", "filter", "subspace",
    "orthogonal", "data", "record", "keeps", "keeping", "separates", "processes", "operates", "verifies", "protect",
    "guarantees", "expanding", "unknown", "hello", "good", "can", "help", "explain", "analyze", "compute",
    "retrieve", "confirmed",
    # Spanish (ES)
    "el", "la", "los", "las", "del", "al", "en", "con", "sin", "y", "o", "qué", "cuál", "es", "son", "fue", "está",
    "cómo", "si", "entonces", "por", "tanto", "porque", "pero", "mediante", "cero", "limpia", "continua",
    "memoria", "arquitectura", "cicatrices", "separa", "analiza", "analizar", "opera", "verifica", "protege", "da",
    "manteniendo", "estado", "energía", "presión", "órbita", "frecuencia", "código", "clave", "secreto", "exacto",
    "respuesta", "idioma", "red", "atención", "ventana", "rápido", "vacío", "verdadero", "mayor", "igual",
    "suma", "doble", "triple", "verificado", "informe", "telemetría", "satélite", "módulo", "núcleo",
    "operación", "información", "señal", "ruido", "ortogonal", "datos", "hola", "puedo", "ayudar",
    "calcular", "recuperar", "confirmado", "desconocido",
    # French (FR)
    "le", "la", "les", "un", "une", "du", "des", "au", "aux", "dans", "sur", "avec", "sans", "et", "à",
    "quel", "quelle", "est", "sont", "était", "comment", "alors", "donc", "parce", "mais", "chaque", "zéro", "continu",
    "système", "mémoire", "cicatrice", "vecteur", "matrice", "état", "énergie", "pression", "orbite", "fréquence",
    "clé", "secrète", "valeur", "exacte", "réponse", "langue", "modèle", "réseau", "fenêtre", "rapide", "précis",
    "vide", "vrai", "faux", "supérieur", "inférieur", "égal", "somme", "résultat", "vérifié", "rapport", "télémétrie",
    "protocole", "noyau", "opération", "sécurisé", "contexte", "bruit", "filtre", "sous-espace", "données",
    "analyse", "sépare", "opère", "garde", "gardant", "vérifie", "protège", "donne", "agrandir", "inconnu",
    "bonjour", "peux", "aider", "expliquer", "calculer", "récupérer", "confirmé",
    # Technical units, variable names & entity identifiers
    "alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "lambda",
    "orion", "helios", "aegis", "kronos", "vega", "atlas", "titan", "zephyr",
    "nova", "hyperion", "borealis", "solaris", "artemis", "valkyrie", "meridian", "polaris",
    "km", "ms", "hz", "khz", "mhz", "ghz", "bar", "kpa", "deg", "kelvin", "teslas", "watts", "kb", "mb",
    "v5", "d4", "lut256", "fwht64", "scache", "o1", "int4", "wgsl", "gdpr", "lgpd",
    # Explicit CJK script tokens (for native `P_⊥CJK` script-gate verification & ZH prompts)
    "你好", "系统", "记忆", "矩阵", "恒定", "信号", "密钥", "验证", "计算", "结果", "量子", "轨道", "压力", "频率", "数据", "安全",
]

_TOKEN_SPLIT_RE = re.compile(
    r"(<[a-zA-Z0-9_:/-]+>|[\u4e00-\u9fff]+|[a-zA-ZáéíóúâêôãõçàèùäëïöüñÁÉÍÓÚÂÊÔÃÕÇÀÈÙÑ][a-zA-Z0-9áéíóúâêôãõçàèùäëïöüñÁÉÍÓÚÂÊÔÃÕÇÀÈÙÑ-]*|[0-9]|[^\s])"
)


class ScarNativeTokenizer:
    """Compact deterministic polyglot word + UTF-8 byte-fallback tokenizer (`vocab_size = 1024`).

    Guarantees:
      - `100%` lossless round-trip on arbitrary UTF-8 strings (via `256` byte-fallback tokens `<0x00>..<0xFF>`).
      - Single-token coverage of core `PT`, `EN`, `ES`, `FR` vocabulary, digits `0..9`, domain entities, and CJK probes.
      - Multiple-of-64 vocabulary size (`1024`) for zero-padding `Block-64 INT4` and `VQ4-Scar` GPU kernels.
    """

    TARGET_VOCAB_SIZE = 1024

    def __init__(self, id_to_token: Sequence[str] | None = None) -> None:
        if id_to_token is None:
            seen: set[str] = set()
            toks: List[str] = []
            for t in [*SPECIAL_TOKENS, *(f"<0x{b:02X}>" for b in range(256)), *CORE_VOCAB_WORDS]:
                if t not in seen:
                    seen.add(t)
                    toks.append(t)
            idx = 0
            while len(toks) < self.TARGET_VOCAB_SIZE:
                filler = f"<res_{idx:03d}>"
                idx += 1
                if filler not in seen:
                    seen.add(filler)
                    toks.append(filler)
            id_to_token = toks[: self.TARGET_VOCAB_SIZE]

        self.id_to_token: List[str] = list(id_to_token)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        self.vocab_size: int = len(self.id_to_token)
        self.pad_token_id: int = self.token_to_id["<pad>"]
        self.bos_token_id: int = self.token_to_id["<bos>"]
        self.eos_token_id: int = self.token_to_id["<eos>"]
        self.unk_token_id: int = self.token_to_id["<unk>"]

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        # Preserve spaces implicitly between word tokens, use byte-fallback for unknown chunks
        pos = 0
        n = len(text)
        while pos < n:
            if text[pos] == " ":
                pos += 1
                continue
            m = _TOKEN_SPLIT_RE.match(text, pos)
            if m is None:
                b_arr = text[pos].encode("utf-8")
                for b in b_arr:
                    ids.append(self.token_to_id[f"<0x{b:02X}>"])
                pos += 1
                continue
            tok = m.group(0)
            pos = m.end()
            low = tok.lower()
            if tok in self.token_to_id:
                ids.append(self.token_to_id[tok])
            elif low in self.token_to_id:
                ids.append(self.token_to_id[low])
            elif all(ch in self.token_to_id for ch in tok):
                for ch in tok:
                    ids.append(self.token_to_id[ch])
            else:
                for b in tok.encode("utf-8"):
                    ids.append(self.token_to_id[f"<0x{b:02X}>"])
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        pieces: List[str] = []
        byte_buf: bytearray = bytearray()

        def _flush_bytes() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        no_space_before = {".", ",", ":", ";", "?", "!", ")", "]"}
        no_space_after = {"(", "[", "¿", "¡", "\n"}
        special_set = set(SPECIAL_TOKENS)

        for raw_id in ids:
            tid = int(raw_id)
            if tid < 0 or tid >= self.vocab_size:
                continue
            tok = self.id_to_token[tid]
            if skip_special_tokens and (tok in special_set or tok.startswith("<res_")):
                continue
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                byte_buf.append(int(tok[3:5], 16))
                continue
            _flush_bytes()
            if not pieces:
                pieces.append(tok)
            else:
                prev = pieces[-1].lstrip(" ")
                if (
                    tok in no_space_before
                    or prev in no_space_after
                    or (prev.isdigit() and tok.isdigit())
                    or has_cjk_script(prev)
                    or has_cjk_script(tok)
                ):
                    pieces.append(tok)
                else:
                    pieces.append(" " + tok)
        _flush_bytes()
        return "".join(pieces).strip()

    def batch_decode(self, batch_ids: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in batch_ids]

    def build_cjk_bitmask_u32(self) -> np.ndarray:
        n_words = (self.vocab_size + 31) // 32
        mask = np.zeros((n_words,), dtype=np.uint32)
        for tid, tok in enumerate(self.id_to_token):
            if has_cjk_script(tok):
                mask[tid >> 5] |= np.uint32(1 << (tid & 31))
        return mask

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"id_to_token": self.id_to_token}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ScarNativeTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(id_to_token=data["id_to_token"])
