"""
Tests for Session 2: Dropout + GPT-2 Scaled Initialization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import (
    TinyGPTV3,
    TinyGPTV3Config,
    config_small,
    dropout_forward,
    dropout_backward,
    _init_layer_v3,
    _init_layer_v3_gpt2,
)


# ─── Dropout tests ────────────────────────────────────────────────────────
class TestDropout:

    def test_dropout_zero_p_is_identity(self):
        """p=0 should return x unchanged, mask=None."""
        rng = np.random.default_rng(0)
        x = np.random.randn(4, 8)
        out, mask = dropout_forward(x, p=0.0, training=True, rng=rng)
        np.testing.assert_array_equal(out, x)
        assert mask is None

    def test_dropout_not_training_is_identity(self):
        """At inference (training=False), dropout is identity."""
        rng = np.random.default_rng(0)
        x = np.random.randn(4, 8)
        out, mask = dropout_forward(x, p=0.5, training=False, rng=rng)
        np.testing.assert_array_equal(out, x)
        assert mask is None

    def test_dropout_training_drops_some_elements(self):
        """At p=0.5, roughly half the elements should be zero."""
        rng = np.random.default_rng(42)
        x = np.ones((100, 100))
        out, mask = dropout_forward(x, p=0.5, training=True, rng=rng)
        # ~50% should be zeroed.
        zero_frac = np.mean(out == 0)
        assert 0.4 < zero_frac < 0.6, f"zero_frac={zero_frac}"
        # Kept elements should be scaled by 1/(1-p) = 2.
        kept = out[out != 0]
        np.testing.assert_allclose(kept, 2.0, rtol=1e-6)

    def test_dropout_preserves_expected_value(self):
        """E[dropout(x)] = x when training (inverted dropout).

        With p=0.3 and 200 samples × 1000 elements, the mean should
        converge to ~1.0 within ~0.05 (CLT).
        """
        rng = np.random.default_rng(0)
        x = np.ones((1000,))
        outs = []
        for _ in range(200):
            out, _ = dropout_forward(x, p=0.3, training=True, rng=rng)
            outs.append(out)
        mean_out = np.mean(outs, axis=0)
        # Grand mean should be very close to 1.0.
        grand_mean = float(np.mean(mean_out))
        assert abs(grand_mean - 1.0) < 0.05, f"grand_mean={grand_mean}"

    def test_dropout_backward_with_mask(self):
        """dropout_backward should multiply by the mask."""
        rng = np.random.default_rng(0)
        x = np.random.randn(4, 8)
        _, mask = dropout_forward(x, p=0.5, training=True, rng=rng)
        dout = np.random.randn(4, 8)
        dx = dropout_backward(dout, mask)
        # Where mask is 0 (dropped), dx should be 0.
        # Where mask is 1/(1-p) (kept), dx should be dout * 1/(1-p).
        zero_positions = mask == 0
        assert np.all(dx[zero_positions] == 0)

    def test_dropout_backward_none_mask(self):
        """When mask is None (no dropout), backward is identity."""
        dout = np.random.randn(4, 8)
        dx = dropout_backward(dout, None)
        np.testing.assert_array_equal(dx, dout)

    def test_model_dropout_in_training_mode(self):
        """model.training=True with dropout>0 should give stochastic outputs."""
        cfg = config_small()
        cfg.dropout = 0.3
        model = TinyGPTV3(cfg)
        model.training = True
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        # Two forward passes should give different results (with high prob).
        logits1, _ = model.forward_with_cache(tokens)
        logits2, _ = model.forward_with_cache(tokens)
        assert not np.allclose(logits1, logits2, atol=1e-6)

    def test_model_dropout_in_eval_mode_is_deterministic(self):
        """model.training=False should give deterministic outputs."""
        cfg = config_small()
        cfg.dropout = 0.3
        model = TinyGPTV3(cfg)
        model.training = False
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits1, _ = model.forward_with_cache(tokens)
        logits2, _ = model.forward_with_cache(tokens)
        np.testing.assert_array_equal(logits1, logits2)

    def test_model_dropout_zero_matches_no_dropout(self):
        """dropout=0 should behave identically with/without training mode."""
        cfg = config_small()
        cfg.dropout = 0.0
        model = TinyGPTV3(cfg)
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        model.training = False
        logits1, _ = model.forward_with_cache(tokens)
        model.training = True
        logits2, _ = model.forward_with_cache(tokens)
        np.testing.assert_array_equal(logits1, logits2)


# ─── GPT-2 Scaled Init tests ──────────────────────────────────────────────
class TestGPT2ScaledInit:

    def test_residual_weights_smaller_than_non_residual(self):
        """Residual-path weights should have smaller std than non-residual."""
        rng = np.random.default_rng(42)
        layer = _init_layer_v3_gpt2(
            rng=rng, H=64, kv_dim=32, mlp_dim=256,
            use_layernorm=True, use_mlp=True,
            n_layers=10, init_scale=0.02,
        )
        residual_std = layer.W_q.std()
        non_residual_std = layer.W_k.std()
        # W_q (residual) should be smaller than W_k (non-residual).
        assert residual_std < non_residual_std, (
            f"residual std={residual_std} should be < non-residual std={non_residual_std}"
        )

    def test_init_scale_decreases_with_depth(self):
        """Deeper models should have smaller residual weights."""
        # n_layers=2
        rng = np.random.default_rng(42)
        layer_shallow = _init_layer_v3_gpt2(
            rng=rng, H=64, kv_dim=32, mlp_dim=256,
            use_layernorm=True, use_mlp=True,
            n_layers=2, init_scale=0.02,
        )
        # n_layers=20
        rng = np.random.default_rng(42)
        layer_deep = _init_layer_v3_gpt2(
            rng=rng, H=64, kv_dim=32, mlp_dim=256,
            use_layernorm=True, use_mlp=True,
            n_layers=20, init_scale=0.02,
        )
        # Deeper → smaller residual scale.
        assert layer_deep.W_q.std() < layer_shallow.W_q.std()

    def test_residual_scale_matches_formula(self):
        """W_q std should be ~ init_scale / sqrt(2 * n_layers)."""
        rng = np.random.default_rng(42)
        n_layers = 10
        init_scale = 0.02
        layer = _init_layer_v3_gpt2(
            rng=rng, H=256, kv_dim=64, mlp_dim=1024,
            use_layernorm=True, use_mlp=True,
            n_layers=n_layers, init_scale=init_scale,
        )
        expected_std = init_scale / np.sqrt(2 * n_layers)
        actual_std = layer.W_q.std()
        # Should match to within sampling noise (~10%).
        assert abs(actual_std - expected_std) / expected_std < 0.15, (
            f"expected std={expected_std}, got {actual_std}"
        )

    def test_default_init_scale_is_0_02(self):
        cfg = TinyGPTV3Config()
        assert cfg.init_scale == 0.02

    def test_model_with_gpt2_init_produces_finite_logits(self):
        """The scaled init should still produce finite forward outputs."""
        cfg = config_small()
        cfg.init_scale = 0.02
        model = TinyGPTV3(cfg)
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits, _ = model.forward(tokens)
        assert np.all(np.isfinite(logits))

    def test_model_with_gpt2_init_forward_backward_works(self):
        """Forward + backward should work end-to-end with GPT-2 init."""
        cfg = config_small()
        model = TinyGPTV3(cfg)
        model.training = False  # disable dropout for deterministic test
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        logits, cache = model.forward_with_cache(tokens)
        dlogits = np.random.randn(*logits.shape).astype(np.float32)
        grads = model.backward(cache, dlogits)
        # All gradients should be finite.
        for k, g in grads.items():
            assert np.all(np.isfinite(g)), f"Gradient {k} contains NaN/Inf"
