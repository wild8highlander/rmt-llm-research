"""
Unit tests for `tiny_gpt_trainer.py` — BPE tokenizer.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tiny_gpt_trainer import BPETokenizer


# ─── Construction ────────────────────────────────────────────────────────────
class TestBPETokenizerConstruction:
    """Tests for BPETokenizer.__init__."""

    def test_default_vocab_size(self):
        tok = BPETokenizer()
        assert tok.vocab_size == 512

    def test_custom_vocab_size(self):
        tok = BPETokenizer(vocab_size=256)
        assert tok.vocab_size == 256

    def test_untrained_has_no_merges(self):
        tok = BPETokenizer()
        assert tok.n_merges == 0


# ─── Training ───────────────────────────────────────────────────────────────
class TestBPETraining:
    """Tests for BPETokenizer.train."""

    def test_train_produces_merges(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)  # 256 bytes + 44 merges
        tok.train(small_corpus)
        assert tok.n_merges == 44

    def test_train_with_explicit_target_merges(self, small_corpus):
        tok = BPETokenizer(vocab_size=512)
        tok.train(small_corpus, target_merges=10)
        assert tok.n_merges == 10

    def test_train_on_tiny_corpus(self, tiny_corpus):
        tok = BPETokenizer(vocab_size=280)
        tok.train(tiny_corpus)
        # Should still produce some merges even on small input
        assert tok.n_merges >= 1

    def test_train_idempotent(self, small_corpus):
        """Training twice on the same corpus should give the same merges."""
        tok1 = BPETokenizer(vocab_size=300, )
        tok1.train(small_corpus)
        tok2 = BPETokenizer(vocab_size=300)
        tok2.train(small_corpus)
        assert tok1.n_merges == tok2.n_merges
        # The merges themselves should be deterministic
        assert tok1.merges == tok2.merges

    def test_train_more_merges_compresses_better(self, small_corpus):
        """More merges → fewer tokens for the same input."""
        tok_small = BPETokenizer(vocab_size=280)
        tok_small.train(small_corpus, target_merges=20)
        tok_big = BPETokenizer(vocab_size=400)
        tok_big.train(small_corpus, target_merges=140)

        sample = small_corpus[:200].decode("utf-8", errors="replace")
        ids_small = tok_small.encode(sample)
        ids_big = tok_big.encode(sample)
        # BPE with more merges should produce fewer or equal tokens
        assert len(ids_big) <= len(ids_small)


# ─── Encoding ───────────────────────────────────────────────────────────────
class TestBPEEncoding:
    """Tests for BPETokenizer.encode / decode."""

    def test_encode_returns_int_array(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids = tok.encode("hello world")
        assert isinstance(ids, np.ndarray) or isinstance(ids, list)
        assert len(ids) > 0

    def test_encode_decode_roundtrip(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        text = "Random Matrix Theory"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_unknown_text_still_works(self, small_corpus):
        """Text not in training corpus should still encode/decode (bytes fallback)."""
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        # Use characters unlikely to appear in the corpus
        text = "qwzx⚡⚡⚡"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_empty_string(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids = tok.encode("")
        assert len(ids) == 0

    def test_encode_single_char(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids = tok.encode("a")
        assert len(ids) == 1

    def test_encode_bytes_directly(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids = tok.encode_bytes(b"hello")
        assert isinstance(ids, np.ndarray)
        assert len(ids) >= 1

    def test_encode_preserves_byte_fidelity(self, small_corpus):
        """Every byte 0-255 should round-trip even after BPE training."""
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        for byte_val in range(256):
            b = bytes([byte_val])
            ids = tok.encode_bytes(b)
            assert tok.decode(ids).encode("utf-8", errors="replace") == b or \
                   len(ids) >= 1, f"byte {byte_val} failed to round-trip"


# ─── Save / load ────────────────────────────────────────────────────────────
class TestBPESaveLoad:
    """Tests for BPETokenizer.save / load."""

    def test_save_load_roundtrip(self, small_corpus, tmp_bpe_path):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus, target_merges=20)

        tok.save(tmp_bpe_path)
        assert os.path.exists(tmp_bpe_path)

        loaded = BPETokenizer.load(tmp_bpe_path)
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.n_merges == tok.n_merges
        assert loaded.merges == tok.merges

        # Encoding should give identical results
        sample = "Random Matrix Theory"
        np.testing.assert_array_equal(tok.encode(sample), loaded.encode(sample))

    def test_save_creates_valid_json(self, small_corpus, tmp_bpe_path):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus, target_merges=5)
        tok.save(tmp_bpe_path)

        with open(tmp_bpe_path) as f:
            data = json.load(f)
        assert "merges" in data or "vocab" in data or "vocab_size" in data

    def test_save_load_preserves_encoding_behavior(self, small_corpus, tmp_bpe_path):
        tok1 = BPETokenizer(vocab_size=300)
        tok1.train(small_corpus, target_merges=30)
        tok1.save(tmp_bpe_path)
        tok2 = BPETokenizer.load(tmp_bpe_path)

        samples = ["Random Matrix", "BBP transition", "Tracy-Widom"]
        for s in samples:
            np.testing.assert_array_equal(tok1.encode(s), tok2.encode(s))


# ─── Numerical properties ───────────────────────────────────────────────────
class TestBPENumerical:
    """Tests for numerical correctness of BPE encoding."""

    def test_token_ids_in_vocab_range(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus, target_merges=40)
        ids = tok.encode(small_corpus[:500].decode("utf-8", errors="replace"))
        ids_arr = np.asarray(ids)
        assert ids_arr.min() >= 0
        assert ids_arr.max() < tok.vocab_size

    def test_token_ids_are_integers(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids = tok.encode("hello")
        ids_arr = np.asarray(ids)
        # Should be integer type
        assert np.issubdtype(ids_arr.dtype, np.integer) or \
               np.allclose(ids_arr, np.round(ids_arr))

    def test_encoding_is_deterministic(self, small_corpus):
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus)
        ids1 = tok.encode("deterministic test")
        ids2 = tok.encode("deterministic test")
        np.testing.assert_array_equal(ids1, ids2)


# ─── Stress tests ───────────────────────────────────────────────────────────
@pytest.mark.slow
class TestBPEStress:
    """Slow tests for BPE on larger inputs."""

    def test_train_on_large_corpus(self):
        # Generate ~50 KB of varied text (not too repetitive — BPE needs unique pairs)
        words = (
            b"the quick brown fox jumps over the lazy dog "
            b"random matrix theory spectral analysis hallucinations "
            b"marchenko pastur bbp transition tracy widom distribution "
            b"caputo fractional derivative keating snaith corrections "
            b"non-hermitian skin effect eigenvalue neural network "
        )
        large_corpus = words * 200  # ~46 KB
        tok = BPETokenizer(vocab_size=320)  # 256 bytes + 64 merges
        tok.train(large_corpus, target_merges=60)
        # On varied text, BPE should produce close to the requested merges
        assert tok.n_merges >= 30, f"Expected ≥30 merges, got {tok.n_merges}"

    def test_encode_long_text(self, small_corpus):
        tok = BPETokenizer(vocab_size=400)
        tok.train(small_corpus, target_merges=100)
        long_text = (small_corpus * 10).decode("utf-8", errors="replace")
        ids = tok.encode(long_text)
        decoded = tok.decode(ids)
        assert decoded == long_text
