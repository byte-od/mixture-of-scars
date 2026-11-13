// ============================================================================
// Pure Zero-Dependency Qwen2.5 Byte-Level BPE Tokenizer (`tokenizer.json`)
// 100% Bit-Exact Token ID Parity with HuggingFace `AutoTokenizer` (EN + PT-BR)
// ============================================================================

const DEFAULT_SYSTEM_PROMPT = "Answer concisely in one precise technical sentence.";
const PT_CHAT_SYSTEM_PROMPT = "Responda de forma concisa, direta e precisa em português.";

const PT_DIACRITIC_RE = /[áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]/;
const PT_WORD_RE = /\b(ol[aá]|oi|tudo|bem|quem|voc[eê]|qual|quais|como|por\s*que|porque|explique|diferen[cç]a|entre|sobre|para|n[aã]o|s[aã]o|est[aá]|fale|resuma)\b/i;

// Exact GPT-2 / Qwen2.5 pre-tokenization split regex (Unicode property escapes, universal JS compatible)
const QWEN_SPLIT_RE = /(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

export function selectChatSystemPrompt(userQuery, defaultPrompt = DEFAULT_SYSTEM_PROMPT) {
  if (PT_DIACRITIC_RE.test(userQuery) || PT_WORD_RE.test(userQuery)) {
    return PT_CHAT_SYSTEM_PROMPT;
  }
  return defaultPrompt;
}

function buildByteUnicodeTables() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteToChar = new Array(256);
  const charToByte = new Map();
  for (let i = 0; i < 256; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteUnicodeTables();
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: false });

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class QwenBPETokenizer {
  constructor(tokenizerJson) {
    this.vocab = new Map();
    const rawVocab = tokenizerJson.model.vocab;
    for (const key of Object.keys(rawVocab)) {
      this.vocab.set(key, rawVocab[key]);
    }

    this.idToToken = new Map();
    for (const [tok, id] of this.vocab.entries()) {
      this.idToToken.set(id, tok);
    }

    this.specialTokens = new Set();
    this.addedTokens = [];
    for (const item of tokenizerJson.added_tokens || []) {
      this.vocab.set(item.content, item.id);
      this.idToToken.set(item.id, item.content);
      this.addedTokens.push(item);
      if (item.special) {
        this.specialTokens.add(item.id);
      }
    }

    // Sort added tokens longest-first so multi-char special tokens split cleanly
    const sortedAdded = this.addedTokens
      .map((t) => t.content)
      .sort((a, b) => b.length - a.length);
    this.addedSplitRe = sortedAdded.length > 0
      ? new RegExp(`(${sortedAdded.map(escapeRegExp).join("|")})`, "g")
      : null;
    this.addedSet = new Set(sortedAdded);

    this.bpeRanks = new Map();
    const merges = tokenizerJson.model.merges || [];
    for (let i = 0; i < merges.length; i++) {
      const m = merges[i];
      const key = Array.isArray(m) ? `${m[0]}\u0001${m[1]}` : m.replace(" ", "\u0001");
      this.bpeRanks.set(key, i);
    }

    this.useNFC = tokenizerJson.normalizer && tokenizerJson.normalizer.type === "NFC";
    this.bpeCache = new Map();
  }

  static async fromUrl(url = "./tokenizer.json") {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to load tokenizer from ${url}: HTTP ${resp.status}`);
    }
    const data = await resp.json();
    return new QwenBPETokenizer(data);
  }

  _bpe(tokenStr) {
    const cached = this.bpeCache.get(tokenStr);
    if (cached !== undefined) return cached;

    let word = Array.from(tokenStr);
    if (word.length <= 1) {
      this.bpeCache.set(tokenStr, word);
      return word;
    }

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestPair = null;
      for (let i = 0; i < word.length - 1; i++) {
        const key = `${word[i]}\u0001${word[i + 1]}`;
        const r = this.bpeRanks.get(key);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPair = [word[i], word[i + 1]];
        }
      }
      if (bestPair === null) break;

      const [first, second] = bestPair;
      const nextWord = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          nextWord.push(first + second);
          i += 2;
        } else {
          nextWord.push(word[i]);
          i += 1;
        }
      }
      word = nextWord;
    }

    if (this.bpeCache.size > 16384) {
      this.bpeCache.clear();
    }
    this.bpeCache.set(tokenStr, word);
    return word;
  }

  _encodeOrdinaryChunk(chunk) {
    const norm = this.useNFC ? chunk.normalize("NFC") : chunk;
    const ids = [];
    const pieces = norm.match(QWEN_SPLIT_RE);
    if (!pieces) return ids;

    for (const piece of pieces) {
      const bytes = UTF8_ENCODER.encode(piece);
      let mapped = "";
      for (let i = 0; i < bytes.length; i++) {
        mapped += BYTE_TO_CHAR[bytes[i]];
      }
      const bpeTokens = this._bpe(mapped);
      for (const bt of bpeTokens) {
        const id = this.vocab.get(bt);
        if (id !== undefined) {
          ids.push(id);
        }
      }
    }
    return ids;
  }

  encode(text) {
    if (!text) return [];
    if (!this.addedSplitRe) {
      return this._encodeOrdinaryChunk(text);
    }
    const parts = text.split(this.addedSplitRe);
    const ids = [];
    for (const part of parts) {
      if (!part) continue;
      if (this.addedSet.has(part)) {
        ids.push(this.vocab.get(part));
      } else {
        const sub = this._encodeOrdinaryChunk(part);
        for (let i = 0; i < sub.length; i++) ids.push(sub[i]);
      }
    }
    return ids;
  }

  applyChatTemplate(messages, addGenerationPrompt = true) {
    let prompt = "";
    for (const msg of messages) {
      prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }
    if (addGenerationPrompt) {
      prompt += "<|im_start|>assistant\n";
    }
    return this.encode(prompt);
  }

  decode(tokenIds, skipSpecialTokens = true) {
    const byteList = [];
    const chunks = [];

    const flushBytes = () => {
      if (byteList.length > 0) {
        chunks.push(UTF8_DECODER.decode(new Uint8Array(byteList)));
        byteList.length = 0;
      }
    };

    for (const id of tokenIds) {
      if (skipSpecialTokens && this.specialTokens.has(id)) {
        continue;
      }
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      if (this.addedSet.has(tok)) {
        flushBytes();
        if (!skipSpecialTokens) chunks.push(tok);
        continue;
      }
      for (const ch of tok) {
        const b = CHAR_TO_BYTE.get(ch);
        if (b !== undefined) {
          byteList.push(b);
        } else {
          flushBytes();
          chunks.push(ch);
        }
      }
    }
    flushBytes();
    return chunks.join("");
  }
}
