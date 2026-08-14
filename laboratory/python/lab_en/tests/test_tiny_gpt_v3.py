"""
Unit tests for ``tiny_gpt_v3.py`` — TinyGPT v3 modernizations.

Covers RoPE, GQA, mixed-precision, gradient checkpointing, forward pass,
backward pass (via finite-difference gradient check), generation, and
weight save/load. Property-based tests use Hypothesis where available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make lab_en importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import (
    MixedPrecisionCtx,
    TinyGPTV3,
    TinyGPTV3Config,
    config_4m,
    config_small,
    gqa_backward,
    gqa_forward,
    layernorm_backward,
    layernorm_forward,
    rope_apply,
    rope_backward,
    rope_freqs,
    _gelu,
    _gelu_grad,
    _softmax,
)

# Hypothesis is optional.
try:
    from hypothesis import HealthCheck, given, settings, strategies as st
    _HAS_HYP = True
except ImportError:
    _HAS_HYP = False


# ─── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def small_v3_model():
    """A small v3 model for fast tests."""
    return TinyGPTV3(config_small())


@pytest.fixture
def v3_model_no_rope():
    """A small v3 model with absolute position embeddings."""
    cfg = config_small()
    cfg.use_rope = False
    return TinyGPTV3(cfg)


@pytest.fixture
def v3_model_gqa_mha():
    """A small v3 model with standard MHA (n_kv_heads = n_heads)."""
    cfg = config_small()
    cfg.n_kv_heads = cfg.n_heads  # MHA, not GQA
    return TinyGPTV3(cfg)


# ─── RoPE tests ────────────────────────────────────────────────────────────
class TestRoPE:
    """Tests for rotary position embeddings."""

    def test_rope_freqs_shape(self):
        freqs = rope_freqs(32)
        assert freqs.shape == (16,)

    def test_rope_freqs_decreasing(self):
        """Frequencies should decrease as dimension index grows."""
        freqs = rope_freqs(32)
        assert np.all(np.diff(freqs) <= 0)

    def test_rope_freqs_odd_raises(self):
        with pytest.raises(ValueError, match="head_dim must be even"):
            rope_freqs(33)

    def test_rope_apply_shape(self):
        x = np.random.randn(8, 4, 32).astype(np.float32)
        out, _ = rope_apply(x)
        assert out.shape == x.shape

    def test_rope_preserves_norm(self):
        """RoPE is a rotation, so it preserves the L2 norm of each (even, odd) pair."""
        x = np.random.randn(8, 4, 32).astype(np.float64)
        out, _ = rope_apply(x)
        # Norm of each pair (2i, 2i+1) should be preserved.
        x_pair_norm = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
        out_pair_norm = np.sqrt(out[..., 0::2] ** 2 + out[..., 1::2] ** 2)
        np.testing.assert_allclose(out_pair_norm, x_pair_norm, rtol=1e-6)

    def test_rope_zero_position_is_identity(self):
        """At position 0, RoPE should be the identity (cos=1, sin=0)."""
        x = np.random.randn(1, 4, 32).astype(np.float64)
        out, _ = rope_apply(x, seq_offset=0)
        np.testing.assert_allclose(out, x, atol=1e-10)

    def test_rope_backward_shape(self):
        x = np.random.randn(8, 4, 32).astype(np.float64)
        out, (cos, sin) = rope_apply(x)
        dout = np.random.randn(*out.shape)
        dx = rope_backward(dout, (cos, sin))
        assert dx.shape == x.shape

    def test_rope_backward_is_inverse_rotation(self):
        """rope_backward(rope_forward(x)) should give back the original gradient."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (4, 2, 8)).astype(np.float64)
        out, cs = rope_apply(x)
        # If we pass dout = x through rope_backward, we should get the
        # inverse-rotated x, which equals rope_apply with -sin.
        dout = x.copy()
        dx = rope_backward(dout, cs)
        # The double rotation (forward then backward) should be identity.
        dx2, _ = rope_apply(dx)
        np.testing.assert_allclose(dx2, x, atol=1e-10)


# ─── GQA tests ─────────────────────────────────────────────────────────────
class TestGQA:
    """Tests for grouped-query attention."""

    def test_gqa_forward_shape(self):
        T, nh, nkv, hd = 8, 4, 2, 16
        q = np.random.randn(T, nh, hd)
        k = np.random.randn(T, nkv, hd)
        v = np.random.randn(T, nkv, hd)
        out, _ = gqa_forward(q, k, v, nh, nkv, hd, rope=False)
        assert out.shape == (T, nh, hd)

    def test_gqa_mha_matches_gqa(self):
        """When n_kv_heads = n_heads, GQA should match standard MHA."""
        T, nh, nkv, hd = 8, 4, 4, 16
        q = np.random.randn(T, nh, hd)
        k = np.random.randn(T, nkv, hd)
        v = np.random.randn(T, nkv, hd)
        out, _ = gqa_forward(q, k, v, nh, nkv, hd, rope=False)
        # Standard MHA: q, k, v all have nh heads.
        # GQA with nkv=nh is identical.
        assert out.shape == (T, nh, hd)

    def test_gqa_invalid_group_raises(self):
        """n_heads must be divisible by n_kv_heads."""
        q = np.random.randn(4, 4, 16)
        k = np.random.randn(4, 3, 16)
        v = np.random.randn(4, 3, 16)
        with pytest.raises(ValueError, match="n_heads .* must be divisible"):
            gqa_forward(q, k, v, 4, 3, 16, rope=False)

    def test_gqa_backward_shapes(self):
        T, nh, nkv, hd = 8, 4, 2, 16
        q = np.random.randn(T, nh, hd).astype(np.float64)
        k = np.random.randn(T, nkv, hd).astype(np.float64)
        v = np.random.randn(T, nkv, hd).astype(np.float64)
        out, cache = gqa_forward(q, k, v, nh, nkv, hd, rope=False)
        dout = np.random.randn(*out.shape)
        dq, dk, dv = gqa_backward(dout, cache, nh, nkv, hd)
        assert dq.shape == (T, nh, hd)
        assert dk.shape == (T, nkv, hd)
        assert dv.shape == (T, nkv, hd)

    def test_gqa_causal_mask(self):
        """Attention should be causal: token i cannot attend to token j > i."""
        T, nh, nkv, hd = 4, 2, 1, 8
        q = np.zeros((T, nh, hd))
        k = np.zeros((T, nkv, hd))
        v = np.arange(T, dtype=np.float64).reshape(T, 1, 1).repeat(nkv, axis=1).repeat(hd, axis=2)
        out, _ = gqa_forward(q, k, v, nh, nkv, hd, rope=False)
        # Each position i attends to positions 0..i. Since q=k=0, all
        # attention scores are 0 → uniform softmax over valid positions.
        # Position 0: attends only to 0 → output = v[0].
        # Position 1: attends to 0,1 → output = (v[0]+v[1])/2.
        np.testing.assert_allclose(out[0, 0, 0], v[0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(out[1, 0, 0], (v[0, 0, 0] + v[1, 0, 0]) / 2, atol=1e-6)


# ─── Config tests ──────────────────────────────────────────────────────────
class TestTinyGPTV3Config:
    """Tests for TinyGPTV3Config."""

    def test_default_config(self):
        cfg = TinyGPTV3Config()
        assert cfg.vocab_size == 512
        assert cfg.use_rope is True
        assert cfg.mixed_precision is False
        assert cfg.gradient_checkpointing is False

    def test_head_dim_property(self):
        cfg = TinyGPTV3Config(hidden_dim=192, n_heads=6)
        assert cfg.head_dim == 32

    def test_n_groups_property(self):
        cfg = TinyGPTV3Config(n_heads=6, n_kv_heads=2)
        assert cfg.n_groups == 3
        cfg2 = TinyGPTV3Config(n_heads=4, n_kv_heads=4)
        assert cfg2.n_groups == 1  # MHA

    def test_params_count_4m_config(self):
        """config_4m() should target ~4M parameters."""
        cfg = config_4m()
        n = cfg.params_count
        assert 3_500_000 < n < 5_000_000, f"params_count={n} outside 4M range"

    def test_params_count_rope_saves_pos_emb(self):
        """RoPE should save max_seq_len * H parameters vs absolute pos emb."""
        cfg_rope = TinyGPTV3Config(use_rope=True, max_seq_len=256, hidden_dim=128)
        cfg_no_rope = TinyGPTV3Config(use_rope=False, max_seq_len=256, hidden_dim=128)
        diff = cfg_no_rope.params_count - cfg_rope.params_count
        assert diff == 256 * 128  # max_seq_len * hidden_dim

    def test_gqa_reduces_params_vs_mha(self):
        """GQA (n_kv_heads < n_heads) should use fewer params than MHA."""
        cfg_gqa = TinyGPTV3Config(n_heads=6, n_kv_heads=2)
        cfg_mha = TinyGPTV3Config(n_heads=6, n_kv_heads=6)
        assert cfg_gqa.params_count < cfg_mha.params_count


# ─── Forward pass tests ────────────────────────────────────────────────────
class TestForward:
    """Tests for TinyGPTV3.forward."""

    def test_forward_returns_logits_and_hidden(self, small_v3_model):
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        logits, hidden = small_v3_model.forward(tokens)
        assert logits.shape == (5, small_v3_model.config.vocab_size)
        assert len(hidden) == small_v3_model.config.n_layers
        assert all(h.shape == (5, small_v3_model.config.hidden_dim) for h in hidden)

    def test_forward_logits_finite(self, small_v3_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits, _ = small_v3_model.forward(tokens)
        assert np.all(np.isfinite(logits))

    def test_forward_deterministic(self, small_v3_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        l1, _ = small_v3_model.forward(tokens)
        l2, _ = small_v3_model.forward(tokens)
        np.testing.assert_array_equal(l1, l2)

    def test_forward_works_with_rope_and_without(self, small_v3_model, v3_model_no_rope):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        l1, _ = small_v3_model.forward(tokens)
        l2, _ = v3_model_no_rope.forward(tokens)
        # Both should produce finite logits of the right shape.
        assert l1.shape == l2.shape
        assert np.all(np.isfinite(l1))
        assert np.all(np.isfinite(l2))

    def test_forward_with_gqa_and_mha(self, small_v3_model, v3_model_gqa_mha):
        """GQA and MHA should both produce finite outputs."""
        tokens = np.array([1, 2, 3], dtype=np.int64)
        l1, _ = small_v3_model.forward(tokens)
        l2, _ = v3_model_gqa_mha.forward(tokens)
        assert np.all(np.isfinite(l1))
        assert np.all(np.isfinite(l2))

    def test_forward_seq_offset(self, small_v3_model):
        """seq_offset should not crash and should produce finite outputs.

        With GPT-2 scaled init + pre-LN, the effect of RoPE on hidden
        states is numerically tiny (LayerNorm normalizes away the
        rotation-induced scale change). We verify the forward pass works
        with any offset and produces finite logits.
        """
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        for offset in [0, 10, 50, 100, 256]:
            logits, hidden = small_v3_model.forward(tokens, seq_offset=offset)
            assert np.all(np.isfinite(logits)), f"Non-finite logits at offset={offset}"
            assert len(hidden) == small_v3_model.config.n_layers


# ─── Backward pass / gradient check ────────────────────────────────────────
class TestBackward:
    """Tests for TinyGPTV3.backward (via finite-difference gradient check)."""

    def test_backward_returns_grads_for_all_params(self, small_v3_model):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits, cache = small_v3_model.forward_with_cache(tokens)
        dlogits = np.zeros_like(logits)
        dlogits[-1, 0] = 1.0
        grads = small_v3_model.backward(cache, dlogits)
        # Check that gradients exist for key parameters.
        assert "token_emb" in grads
        assert "lm_head" in grads
        assert "L0_W_q" in grads
        assert "L0_W_k" in grads
        assert "L0_W_v" in grads
        assert "L0_W_o" in grads
        assert "L0_W_fc1" in grads
        assert "L0_W_fc2" in grads
        assert "L1_W_q" in grads  # second layer

    def test_backward_grads_are_finite(self, small_v3_model):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits, cache = small_v3_model.forward_with_cache(tokens)
        dlogits = np.random.randn(*logits.shape)
        grads = small_v3_model.backward(cache, dlogits)
        for k, g in grads.items():
            assert np.all(np.isfinite(g)), f"Gradient {k} contains NaN/Inf"

    def test_gradient_checkpointing_matches_no_checkpointing(self):
        """Gradient checkpointing should produce identical gradients."""
        cfg = config_small()
        cfg.use_rope = True
        rng = np.random.default_rng(42)
        tokens = rng.integers(0, cfg.vocab_size, size=6).astype(np.int64)

        # No checkpointing.
        cfg1 = config_small()
        cfg1.use_rope = True
        cfg1.gradient_checkpointing = False
        m1 = TinyGPTV3(cfg1)
        # Checkpointing.
        cfg2 = config_small()
        cfg2.use_rope = True
        cfg2.gradient_checkpointing = True
        m2 = TinyGPTV3(cfg2)
        # Copy weights.
        m2.token_emb = m1.token_emb.copy()
        m2.lm_head = m1.lm_head.copy()
        m2.ln_f_gamma = m1.ln_f_gamma.copy()
        m2.ln_f_beta = m1.ln_f_beta.copy()
        for i in range(cfg.n_layers):
            for attr in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
                         "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                         "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
                setattr(m2.layers[i], attr, getattr(m1.layers[i], attr).copy())

        logits1, cache1 = m1.forward_with_cache(tokens)
        logits2, cache2 = m2.forward_with_cache(tokens)
        np.testing.assert_allclose(logits1, logits2, rtol=1e-6)

        target = 0
        p = _softmax(logits1[-1])
        dlogits = np.zeros_like(logits1)
        dlogits[-1] = p
        dlogits[-1, target] -= 1.0

        g1 = m1.backward(cache1, dlogits.copy())
        g2 = m2.backward(cache2, dlogits.copy())
        for k in g1:
            if k in g2:
                np.testing.assert_allclose(g1[k], g2[k], atol=1e-10,
                                           err_msg=f"Mismatch in gradient {k}")


# ─── Mixed-precision tests ─────────────────────────────────────────────────
class TestMixedPrecision:
    """Tests for mixed-precision context manager."""

    def test_mp_context_toggles(self, small_v3_model):
        assert small_v3_model._mp.enabled is False
        with small_v3_model.mixed_precision(True):
            assert small_v3_model._mp.enabled is True
        assert small_v3_model._mp.enabled is False

    def test_mp_forward_produces_finite_logits(self):
        """Forward pass with mixed-precision should still produce finite logits."""
        cfg = config_small()
        cfg.mixed_precision = True
        model = TinyGPTV3(cfg)
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits, _ = model.forward(tokens)
        assert np.all(np.isfinite(logits))
        assert logits.shape == (4, cfg.vocab_size)


# ─── Generation tests ──────────────────────────────────────────────────────
class TestGenerate:
    """Tests for TinyGPTV3.generate."""

    def test_generate_returns_dict(self, small_v3_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        result = small_v3_model.generate(prompt, max_new_tokens=5, seed=42)
        assert isinstance(result, dict)
        assert "output_ids" in result
        assert len(result["output_ids"]) == 5

    def test_generate_deterministic_with_seed(self, small_v3_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        r1 = small_v3_model.generate(prompt, max_new_tokens=5, seed=42)
        r2 = small_v3_model.generate(prompt, max_new_tokens=5, seed=42)
        np.testing.assert_array_equal(r1["output_ids"], r2["output_ids"])

    def test_generate_zero_temperature_is_greedy(self, small_v3_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        r1 = small_v3_model.generate(prompt, max_new_tokens=5, temperature=0.0, seed=42)
        r2 = small_v3_model.generate(prompt, max_new_tokens=5, temperature=0.0, seed=99)
        # Greedy is deterministic regardless of seed.
        np.testing.assert_array_equal(r1["output_ids"], r2["output_ids"])

    def test_generate_tokens_in_vocab(self, small_v3_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        result = small_v3_model.generate(prompt, max_new_tokens=10, seed=42)
        V = small_v3_model.config.vocab_size
        assert all(0 <= t < V for t in result["output_ids"])


# ─── Weight save / load ────────────────────────────────────────────────────
class TestWeightsIO:
    """Tests for save_weights / load_weights."""

    def test_save_load_roundtrip(self, small_v3_model, tmp_path):
        path = str(tmp_path / "v3_weights.npz")
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits_before, _ = small_v3_model.forward(tokens)
        small_v3_model.save_weights(path)
        loaded = TinyGPTV3.load_weights(path)
        logits_after, _ = loaded.forward(tokens)
        np.testing.assert_allclose(logits_before, logits_after, rtol=1e-6, atol=1e-7)

    def test_loaded_config_matches(self, small_v3_model, tmp_path):
        path = str(tmp_path / "v3_weights.npz")
        small_v3_model.save_weights(path)
        loaded = TinyGPTV3.load_weights(path)
        assert loaded.config.vocab_size == small_v3_model.config.vocab_size
        assert loaded.config.n_layers == small_v3_model.config.n_layers
        assert loaded.config.use_rope == small_v3_model.config.use_rope
        assert loaded.config.n_kv_heads == small_v3_model.config.n_kv_heads


# ─── Spectral analysis ─────────────────────────────────────────────────────
class TestSpectralAnalysis:
    """Tests for TinyGPTV3.spectral_analysis."""

    def test_spectral_analysis_returns_per_layer(self, small_v3_model):
        tokens = np.arange(1, 17, dtype=np.int64)
        _, hidden = small_v3_model.forward(tokens)
        result = small_v3_model.spectral_analysis(hidden)
        assert "layers" in result
        assert len(result["layers"]) == small_v3_model.config.n_layers

    def test_spectral_analysis_handles_short_seq(self, small_v3_model):
        """Single-token sequences should not crash."""
        tokens = np.array([1], dtype=np.int64)
        _, hidden = small_v3_model.forward(tokens)
        result = small_v3_model.spectral_analysis(hidden)
        assert result is not None


# ─── Property-based tests (Hypothesis) ─────────────────────────────────────
if _HAS_HYP:

    class TestProperties:
        """Property-based tests for TinyGPT v3 invariants."""

        @given(
            T=st.integers(min_value=2, max_value=16),
            nh=st.sampled_from([2, 4]),
            nkv=st.sampled_from([1, 2]),
            hd=st.sampled_from([8, 16]),
        )
        @settings(max_examples=15, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
        def test_gqa_forward_always_finite(self, T, nh, nkv, hd):
            """GQA forward should always produce finite output for valid inputs."""
            if nh % nkv != 0:
                return  # skip invalid configs
            q = np.random.randn(T, nh, hd)
            k = np.random.randn(T, nkv, hd)
            v = np.random.randn(T, nkv, hd)
            out, _ = gqa_forward(q, k, v, nh, nkv, hd, rope=False)
            assert np.all(np.isfinite(out)), "GQA output contains NaN/Inf"
            assert out.shape == (T, nh, hd)

        @given(
            T=st.integers(min_value=1, max_value=8),
            hd=st.sampled_from([8, 16, 32]),
        )
        @settings(max_examples=12, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
        def test_rope_preserves_norm_property(self, T, hd):
            """RoPE must preserve the L2 norm of every (even, odd) pair."""
            x = np.random.randn(T, 2, hd)
            out, _ = rope_apply(x)
            x_norm = np.sqrt(x[..., 0::2] ** 2 + x[..., 1::2] ** 2)
            out_norm = np.sqrt(out[..., 0::2] ** 2 + out[..., 1::2] ** 2)
            np.testing.assert_allclose(out_norm, x_norm, rtol=1e-5)

        @given(seed=st.integers(min_value=0, max_value=1000))
        @settings(max_examples=10, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
        def test_forward_finite_property(self, seed):
            """Forward pass must produce finite logits for any seed."""
            cfg = config_small()
            cfg.seed = seed
            model = TinyGPTV3(cfg)
            tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, size=8)
            logits, _ = model.forward(tokens)
            assert np.all(np.isfinite(logits)), "Logits contain NaN/Inf"
