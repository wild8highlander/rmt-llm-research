"""Tests for Session 7: BPE Tokenizer v3."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bpe_tokenizer_v3 import BPETokenizerV3


CORPUS = (
    b"def hello_world():\n"
    b"    print('Hello, World!')\n"
    b"    return 42\n"
    b"\n"
    b"def goodbye():\n"
    b"    print('Bye!')\n"
    b"    return 0\n"
) * 10


class TestBPETokenizerV3:

    def test_default_vocab_size(self):
        tok = BPETokenizerV3()
        assert tok.vocab_size == 1024

    def test_invalid_vocab_size_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be"):
            BPETokenizerV3(vocab_size=100)

    def test_fit_learns_merges(self):
        tok = BPETokenizerV3(vocab_size=300)  # 256 + 44 merges
        tok.fit(CORPUS)
        assert tok.n_merges > 0
        assert tok.n_merges <= 44

    def test_encode_decode_roundtrip(self):
        """encode → decode should preserve the original text."""
        tok = BPETokenizerV3(vocab_size=400)
        tok.fit(CORPUS)
        text = b"def hello_world():\n    return 42\n"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_str_input(self):
        """encode should accept str as well as bytes."""
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS)
        ids = tok.encode("def hello():")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_corpus_returns_ndarray(self):
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS)
        arr = tok.encode_corpus(CORPUS)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.int64
        assert len(arr) > 0

    def test_compression_ratio_above_one(self):
        """After training, BPE should compress the corpus (>1× ratio)."""
        tok = BPETokenizerV3(vocab_size=512)
        tok.fit(CORPUS)
        ratio = tok.compression_ratio(CORPUS)
        assert ratio > 1.5, f"compression ratio {ratio} should be > 1.5"

    def test_token_ids_in_vocab_range(self):
        """All encoded IDs should be in [0, vocab_size)."""
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS)
        ids = tok.encode(CORPUS)
        assert all(0 <= i < 300 for i in ids)

    def test_save_load_roundtrip(self, tmp_path):
        """Save then load should produce an identical tokenizer."""
        tok = BPETokenizerV3(vocab_size=400)
        tok.fit(CORPUS)
        path = str(tmp_path / "bpe.json")
        tok.save(path)
        tok2 = BPETokenizerV3.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.n_merges == tok.n_merges
        # Same encoding.
        text = b"def hello():\n    return 0\n"
        assert tok.encode(text) == tok2.encode(text)

    def test_untrained_tokenizer_encodes_as_bytes(self):
        """An untrained tokenizer (0 merges) should encode as raw bytes."""
        tok = BPETokenizerV3(vocab_size=256)
        # No fit() call.
        ids = tok.encode(b"hello")
        assert ids == [104, 101, 108, 108, 111]  # ASCII for 'hello'

    def test_decode_unknown_id_returns_placeholder(self):
        """decode should handle unknown IDs gracefully."""
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS)
        decoded = tok.decode([9999])  # unknown ID
        assert b"?" in decoded

    def test_decode_to_str_returns_str(self):
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS)
        s = tok.decode_to_str(tok.encode(b"hello"))
        assert isinstance(s, str)

    def test_fit_with_str_corpus(self):
        """fit should accept str as well as bytes."""
        tok = BPETokenizerV3(vocab_size=300)
        tok.fit(CORPUS.decode("utf-8"))
        assert tok.n_merges > 0

    def test_fit_with_n_merges_override(self):
        """n_merges should override the default vocab_size - 256."""
        tok = BPETokenizerV3(vocab_size=1024)
        tok.fit(CORPUS, n_merges=10)
        assert tok.n_merges <= 10

    def test_cache_speeds_up_repeated_encode(self):
        """Encoding the same text twice should use the cache (faster)."""
        import time
        tok = BPETokenizerV3(vocab_size=400)
        tok.fit(CORPUS)
        text = CORPUS[:1000]
        # First encode (cold cache).
        t0 = time.time()
        ids1 = tok.encode(text)
        t1 = time.time()
        # Second encode (warm cache).
        ids2 = tok.encode(text)
        t2 = time.time()
        assert ids1 == ids2
        # Cache should be at least as fast (often much faster).
        assert (t2 - t1) <= (t1 - t0) * 1.5

    def test_larger_vocab_gives_better_compression(self):
        """A 512-vocab tokenizer should compress better than 256-vocab."""
        tok_small = BPETokenizerV3(vocab_size=256)  # 0 merges
        tok_large = BPETokenizerV3(vocab_size=512)  # 256 merges
        tok_large.fit(CORPUS)
        r_small = tok_small.compression_ratio(CORPUS)
        r_large = tok_large.compression_ratio(CORPUS)
        assert r_large > r_small

    def test_merges_are_unique(self):
        """No merge pair should appear twice in the merge list."""
        tok = BPETokenizerV3(vocab_size=400)
        tok.fit(CORPUS)
        seen = set()
        for pair in tok.merges:
            assert pair not in seen, f"Duplicate merge: {pair}"
            seen.add(pair)
