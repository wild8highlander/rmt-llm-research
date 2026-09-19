"""
Property-based tests for TinyGPT + BPE + Adam using Hypothesis.

These tests run hundreds of randomly-generated inputs to catch edge cases
that example-based tests miss. They run as part of `make test-hypothesis`
and in CI under the `hypothesis` marker.

Strategy philosophy:
- Generate *valid* model configurations and corpora, never adversarial inputs.
- Use bounded search spaces so tests run in seconds, not minutes.
- Always include `@settings(max_examples=...)` so CI is deterministic.

If Hypothesis is not installed, the tests are skipped automatically.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


# Make the lab package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays
except ImportError:  # pragma: no cover
    pytest.skip("hypothesis not installed — run pip install hypothesis", allow_module_level=True)

from tiny_gpt import TinyGPT, TinyGPTConfig
from tiny_gpt_trainer import BPETokenizer, cross_entropy_loss, softmax


# ─── Strategies ──────────────────────────────────────────────────────────────


# A small-but-valid TinyGPT config: 1-2 layers, hidden 32-64, 2-4 heads.
# This keeps each property test under 1 second while still exercising
# the math non-trivially.
@st.composite
def small_config(draw):
    hidden = draw(st.sampled_from([32, 48, 64]))
    n_heads = draw(st.sampled_from([2, 4]))
    # hidden must be divisible by n_heads
    if hidden % n_heads != 0:
        n_heads = 2
    return TinyGPTConfig(
        vocab_size=draw(st.sampled_from([64, 128, 256])),
        hidden_dim=hidden,
        n_layers=draw(st.integers(min_value=1, max_value=2)),
        n_heads=n_heads,
        max_seq_len=draw(st.integers(min_value=8, max_value=16)),
        mlp_ratio=draw(st.sampled_from([2, 4])),
        use_mlp=True,
        use_layernorm=True,
        activation="gelu",
    )


# Random token sequences (within vocab + seq_len bounds)
def token_sequence(vocab_size: int, max_seq_len: int):
    return arrays(
        dtype=np.int64,
        shape=st.integers(min_value=2, max_value=max_seq_len),
        elements=st.integers(min_value=0, max_value=vocab_size - 1),
    )


# ASCII text under 200 chars (avoids Unicode edge cases in BPE)
small_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=200,
)


# ─── Property tests: TinyGPT forward pass ────────────────────────────────────


class TestForwardProperties:
    """Shape and basic invariants of the TinyGPT forward pass."""

    @given(cfg=small_config())
    @settings(max_examples=15, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_output_shape(self, cfg):
        """forward(x) must return logits of shape (T, vocab_size)."""
        model = TinyGPT(replace(cfg, seed=42))
        x = np.random.randint(0, cfg.vocab_size, size=cfg.max_seq_len)
        logits, _ = model.forward(x)
        assert logits.shape == (cfg.max_seq_len, cfg.vocab_size)

    @given(cfg=small_config())
    @settings(max_examples=10, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_is_deterministic(self, cfg):
        """Calling forward twice with the same input must produce identical logits."""
        model = TinyGPT(replace(cfg, seed=7))
        x = np.random.randint(0, cfg.vocab_size, size=cfg.max_seq_len)
        a, _ = model.forward(x)
        b, _ = model.forward(x)
        np.testing.assert_array_equal(a, b)

    @given(cfg=small_config())
    @settings(max_examples=10, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_finite(self, cfg):
        """Logits must be finite (no NaN / Inf) for any valid input."""
        model = TinyGPT(replace(cfg, seed=11))
        x = np.random.randint(0, cfg.vocab_size, size=cfg.max_seq_len)
        logits, _ = model.forward(x)
        assert np.all(np.isfinite(logits)), "Logits contain NaN or Inf"

    @given(cfg=small_config())
    @settings(max_examples=8, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
    def test_different_inputs_different_outputs(self, cfg):
        """Different inputs must produce different outputs (model is not constant)."""
        model = TinyGPT(replace(cfg, seed=99))
        x1 = np.zeros(cfg.max_seq_len, dtype=np.int64)
        x2 = np.ones(cfg.max_seq_len, dtype=np.int64) * (cfg.vocab_size - 1)
        if cfg.vocab_size > 1:
            a, _ = model.forward(x1)
            b, _ = model.forward(x2)
            assert not np.allclose(a, b), "Forward pass is input-invariant"


# ─── Property tests: BPE tokenizer ───────────────────────────────────────────


class TestBPEProperties:
    """Round-trip and basic invariants of the BPE tokenizer."""

    @given(text=small_text)
    @settings(max_examples=25, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_bpe_roundtrip_preserves_text(self, text):
        """encode(decode(encode(text))) == encode(text) — BPE must be a stable projection."""
        # vocab_size=512 gives room for 32 merges on top of the 256 byte tokens
        tok = BPETokenizer(vocab_size=512)
        tok.train(text.encode("utf-8"), target_merges=32)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # Round-trip: re-encoding the decoded text must yield the same ids
        ids2 = tok.encode(decoded)
        assert list(ids) == list(ids2), f"BPE round-trip not stable for {text[:50]!r}"

    @given(text=small_text)
    @settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_bpe_token_count_bounded(self, text):
        """Encoded length must be ≤ raw byte length (BPE never expands)."""
        tok = BPETokenizer(vocab_size=512)
        tok.train(text.encode("utf-8"), target_merges=32)
        ids = tok.encode(text)
        assert len(ids) <= len(text.encode("utf-8")), "BPE expanded the input"

    @given(text=small_text)
    @settings(max_examples=15, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_bpe_decoded_is_bytestring_prefix(self, text):
        """decoded text must start with the first byte of input (no leading corruption)."""
        tok = BPETokenizer(vocab_size=512)
        tok.train(text.encode("utf-8"), target_merges=16)
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        if text and decoded:
            assert decoded[0] == text[0] or text[0].encode() == decoded[:1], (
                f"First char mismatch: input={text[0]!r}, decoded={decoded[0]!r}"
            )

    @given(
        text1=small_text,
        text2=small_text,
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.too_slow])
    def test_bpe_concat_compositionality(self, text1, text2):
        """encode(a+b) must equal encode(a) + encode(b) ONLY when token boundaries align.

        Since this is NOT guaranteed for BPE in general, we only check that
        the concatenation length is at most len(encode(a)) + len(encode(b)).
        """
        tok = BPETokenizer(vocab_size=512)
        combined = text1 + text2
        tok.train(combined.encode("utf-8"), target_merges=16)
        a_ids = tok.encode(text1)
        b_ids = tok.encode(text2)
        ab_ids = tok.encode(combined)
        # Concatenation may have cross-boundary merges, but never expansions
        assert len(ab_ids) <= len(a_ids) + len(b_ids)


# ─── Property tests: softmax / cross-entropy numerics ────────────────────────


class TestNumericalStability:
    """Softmax and cross-entropy must be numerically stable."""

    @given(
        n=st.integers(min_value=2, max_value=20),
        scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=1000)
    def test_softmax_sums_to_one(self, n, scale):
        """softmax(x) must sum to 1.0 (within float tolerance) for any finite input."""
        x = np.random.randn(n) * scale
        p = softmax(x)
        assert np.all(p >= 0.0), "softmax produced negative probabilities"
        assert np.all(p <= 1.0 + 1e-6), "softmax produced probabilities > 1"
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)

    @given(
        n=st.integers(min_value=2, max_value=10),
        offset=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=1000)
    def test_softmax_shift_invariant(self, n, offset):
        """softmax(x + c) must equal softmax(x) for any constant c."""
        x = np.random.randn(n)
        a = softmax(x)
        b = softmax(x + offset)
        np.testing.assert_allclose(a, b, atol=1e-6, err_msg=f"Shift by {offset} changed softmax")

    @given(
        n=st.integers(min_value=2, max_value=20),
        target=st.integers(min_value=0, max_value=19),
    )
    @settings(max_examples=30, deadline=1000)
    def test_cross_entropy_non_negative(self, n, target):
        """Cross-entropy loss must be ≥ 0 for any valid probability distribution."""
        # Adjust target to be within range
        target = target % n
        logits = np.random.randn(n) * 2.0
        loss = cross_entropy_loss(logits, target)
        assert loss >= 0.0 - 1e-6, f"Cross-entropy is negative: {loss}"
        assert np.isfinite(loss), "Cross-entropy is not finite"

    @given(n=st.integers(min_value=2, max_value=20))
    @settings(max_examples=15, deadline=1000)
    def test_cross_entropy_zero_when_perfect(self, n):
        """Cross-entropy → 0 when the target class has all probability mass."""
        logits = np.full(n, -50.0)
        logits[0] = 50.0  # target=0 should get ~1.0 probability
        loss = cross_entropy_loss(logits, 0)
        assert loss < 0.01, f"Perfect prediction should have ~0 loss, got {loss}"


# ─── Property tests: weight initialization ───────────────────────────────────


class TestWeightInitProperties:
    """Weight initialization must produce well-conditioned matrices."""

    @given(cfg=small_config())
    @settings(max_examples=10, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
    def test_weights_finite(self, cfg):
        """All model weights must be finite after initialization."""
        model = TinyGPT(replace(cfg, seed=0))
        for layer in model.layers:
            for w in [
                layer.W_q,
                layer.W_k,
                layer.W_v,
                layer.W_o,
                layer.ln1_gamma,
                layer.ln1_beta,
                layer.ln2_gamma,
                layer.ln2_beta,
            ]:
                assert np.all(np.isfinite(w)), "Layer weights contain NaN/Inf"
            if cfg.use_mlp:
                for w in [layer.W_fc1, layer.W_fc2, layer.b_fc1, layer.b_fc2]:
                    assert np.all(np.isfinite(w)), "MLP weights contain NaN/Inf"

    @given(cfg=small_config())
    @settings(max_examples=10, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
    def test_layer_norm_init_to_identity(self, cfg):
        """LayerNorm must init to gamma=1, beta=0 (identity transform)."""
        model = TinyGPT(replace(cfg, seed=1))
        for layer in model.layers:
            np.testing.assert_allclose(layer.ln1_gamma, 1.0, atol=1e-6)
            np.testing.assert_allclose(layer.ln1_beta, 0.0, atol=1e-6)
            np.testing.assert_allclose(layer.ln2_gamma, 1.0, atol=1e-6)
            np.testing.assert_allclose(layer.ln2_beta, 0.0, atol=1e-6)

    @given(
        seed1=st.integers(min_value=0, max_value=1000),
        seed2=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=15, deadline=2000)
    def test_different_seeds_different_weights(self, seed1, seed2):
        """Two different RNG seeds must produce different weights."""
        if seed1 == seed2:
            return  # skip trivial case
        cfg = TinyGPTConfig(vocab_size=64, hidden_dim=32, n_layers=1, n_heads=2, max_seq_len=8)
        m1 = TinyGPT(replace(cfg, seed=seed1))
        m2 = TinyGPT(replace(cfg, seed=seed2))
        assert not np.allclose(m1.layers[0].W_q, m2.layers[0].W_q), (
            f"Seeds {seed1} and {seed2} produced identical weights"
        )


# ─── Property tests: generation ──────────────────────────────────────────────


class TestGenerationProperties:
    """Invariants of `TinyGPT.generate`."""

    @given(
        prompt_len=st.integers(min_value=1, max_value=8),
        n_new=st.integers(min_value=1, max_value=10),
        temperature=st.sampled_from([0.5, 0.8, 1.0, 1.5]),
    )
    @settings(max_examples=15, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_output_length(self, prompt_len, n_new, temperature):
        """Generated sequence must have exactly prompt_len + n_new tokens."""
        cfg = TinyGPTConfig(
            vocab_size=32, hidden_dim=16, n_layers=1, n_heads=2, max_seq_len=32, seed=3
        )
        model = TinyGPT(cfg)
        prompt = np.arange(prompt_len)
        out = model.generate(
            prompt,
            max_new_tokens=n_new,
            temperature=temperature,
            capture_hidden=False,
        )
        assert len(out["full_ids"]) == prompt_len + n_new, (
            f"Expected {prompt_len + n_new} tokens, got {len(out['full_ids'])}"
        )

    @given(prompt_len=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
    def test_greedy_generation_deterministic(self, prompt_len):
        """Greedy (temperature=0) generation must be deterministic."""
        cfg = TinyGPTConfig(
            vocab_size=16, hidden_dim=16, n_layers=1, n_heads=2, max_seq_len=16, seed=7
        )
        model = TinyGPT(cfg)
        prompt = np.arange(prompt_len)

        out1 = model.generate(prompt, max_new_tokens=5, temperature=0.0, capture_hidden=False)
        out2 = model.generate(prompt, max_new_tokens=5, temperature=0.0, capture_hidden=False)
        assert out1["output_ids"] == out2["output_ids"], "Greedy generation is not deterministic"

    @given(prompt_len=st.integers(min_value=1, max_value=5))
    @settings(max_examples=8, deadline=4000, suppress_health_check=[HealthCheck.too_slow])
    def test_generated_tokens_in_vocab(self, prompt_len):
        """Every generated token must be in [0, vocab_size)."""
        cfg = TinyGPTConfig(
            vocab_size=16, hidden_dim=16, n_layers=1, n_heads=2, max_seq_len=16, seed=13
        )
        model = TinyGPT(cfg)
        prompt = np.arange(prompt_len)

        out = model.generate(prompt, max_new_tokens=8, temperature=1.0, capture_hidden=False)
        assert all(0 <= t < cfg.vocab_size for t in out["output_ids"]), (
            f"Generated token out of vocab range: {out['output_ids']}"
        )
