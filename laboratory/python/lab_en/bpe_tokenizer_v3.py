"""
bpe_tokenizer_v3.py — byte-level BPE tokenizer for TinyGPT v3.

RESTORED MODULE (bugfix): this file was referenced by tests/test_v3_session7.py
and tests/v3/full_train_v3.py but was never committed to the repository —
the repo .gitignore pattern `*_token*` accidentally excluded it at commit
time (the pattern has been narrowed to credential-file shapes instead).

Contract implemented (fixed by tests/test_v3_session7.py):
  - byte-level BPE: 256 byte tokens + learned merge tokens
  - fit(corpus) accepts bytes or str; n_merges defaults to vocab_size - 256
  - encode(text|bytes) -> list[int]; decode(ids) -> bytes (unknown -> b"?")
  - decode_to_str, encode_corpus -> np.ndarray[int64]
  - compression_ratio, save/load JSON roundtrip, encode cache
  - untrained tokenizer (0 merges) encodes as raw byte ids
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise

import numpy as np


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `idx`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizerV3:
    """Byte-level BPE tokenizer (256 base bytes + learned merges)."""

    def __init__(self, vocab_size: int = 1024):
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (byte alphabet), got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.merges: list[tuple[int, int]] = []
        self._ranks: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(self, corpus: bytes | str, n_merges: int | None = None) -> BPETokenizerV3:
        if isinstance(corpus, str):
            corpus = corpus.encode("utf-8")
        if n_merges is None:
            n_merges = self.vocab_size - 256
        n_merges = min(int(n_merges), self.vocab_size - 256)

        ids = list(corpus)
        for step in range(n_merges):
            counts = Counter(pairwise(ids))
            if not counts:
                break
            # deterministic tie-break: highest count, then lexicographic pair
            pair = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            if counts[pair] < 2:
                break
            idx = 256 + len(self.merges)
            self.merges.append(pair)
            self._ranks[pair] = step
            self._vocab[idx] = self._vocab[pair[0]] + self._vocab[pair[1]]
            ids = _merge(ids, pair, idx)
        self._cache.clear()
        return self

    # ------------------------------------------------------------------
    # encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: bytes | str) -> list[int]:
        if isinstance(text, str):
            text = text.encode("utf-8")
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)
        ids = list(text)
        # GPT-2 style: repeatedly apply the lowest-rank (earliest-learned) merge
        while len(ids) >= 2:
            pairs = set(pairwise(ids))
            best = min(pairs, key=lambda p: self._ranks.get(p, 1 << 30))
            if best not in self._ranks:
                break
            ids = _merge(ids, best, 256 + self._ranks[best])
        self._cache[text] = list(ids)
        return ids

    def decode(self, ids: list[int]) -> bytes:
        parts: list[bytes] = []
        for i in ids:
            b = self._vocab.get(int(i))
            parts.append(b if b is not None else b"?")
        return b"".join(parts)

    def decode_to_str(self, ids: list[int]) -> str:
        return self.decode(ids).decode("utf-8", errors="replace")

    def encode_corpus(self, corpus: bytes | str) -> np.ndarray:
        return np.array(self.encode(corpus), dtype=np.int64)

    def compression_ratio(self, corpus: bytes | str) -> float:
        if isinstance(corpus, str):
            corpus = corpus.encode("utf-8")
        n_ids = len(self.encode(corpus))
        return len(corpus) / max(n_ids, 1)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "kind": "bpe_tokenizer_v3",
            "version": 1,
            "vocab_size": self.vocab_size,
            "n_merges": len(self.merges),
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> BPETokenizerV3:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=int(data["vocab_size"]))
        for a, b in data["merges"]:
            idx = 256 + len(tok.merges)
            tok.merges.append((int(a), int(b)))
            tok._ranks[(int(a), int(b))] = len(tok.merges) - 1
            tok._vocab[idx] = tok._vocab[int(a)] + tok._vocab[int(b)]
        return tok

    @property
    def n_merges(self) -> int:
        return len(self.merges)
