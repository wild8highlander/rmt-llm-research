"""
Tests for Session 1 additions: weight tying, label smoothing, gradient clipping.
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
    _softmax,
    clip_grad_norm_,
    config_small,
    label_smoothing_cross_entropy,
)


# ─── Weight tying ─────────────────────────────────────────────────────────
class TestWeightTying:
    def test_default_config_has_weight_tying(self):
        cfg = TinyGPTV3Config()
        assert cfg.weight_tying is True

    def test_weight_tying_reduces_param_count(self):
        """Tying should remove H * V parameters (the standalone lm_head)."""
        cfg_tied = TinyGPTV3Config(weight_tying=True, vocab_size=512, hidden_dim=128)
        cfg_untied = TinyGPTV3Config(weight_tying=False, vocab_size=512, hidden_dim=128)
        diff = cfg_untied.params_count - cfg_tied.params_count
        assert diff == 512 * 128  # V * H

    def test_forward_uses_tied_weights(self):
        """When tied, logits should match x_norm @ token_emb.T (not lm_head)."""
        cfg = config_small()
        cfg.weight_tying = True
        model = TinyGPTV3(cfg)
        # Corrupt the standalone lm_head; forward should be unaffected.
        model.lm_head[:] = 999.0
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits, _ = model.forward(tokens)
        assert np.all(np.isfinite(logits))
        # Now disable tying — should use the corrupted lm_head.
        cfg2 = config_small()
        cfg2.weight_tying = False
        cfg2.seed = cfg.seed
        model2 = TinyGPTV3(cfg2)
        model2.token_emb = model.token_emb.copy()
        model2.lm_head[:] = 999.0
        logits2, _ = model2.forward(tokens)
        # The untied (corrupted) version should give very different logits.
        assert not np.allclose(logits, logits2, atol=1e-3)

    def test_weight_tying_gradient_check(self):
        """With weight tying, gradients should still be correct (finite-diff)."""
        cfg = config_small()
        cfg.weight_tying = True
        model = TinyGPTV3(cfg)
        # Cast to float64 for clean finite differences.
        model.token_emb = model.token_emb.astype(np.float64)
        model.lm_head = model.lm_head.astype(np.float64)
        model.ln_f_gamma = model.ln_f_gamma.astype(np.float64)
        model.ln_f_beta = model.ln_f_beta.astype(np.float64)
        for layer in model.layers:
            for attr in (
                "W_q",
                "W_k",
                "W_v",
                "W_o",
                "b_q",
                "b_k",
                "b_v",
                "b_o",
                "ln1_gamma",
                "ln1_beta",
                "ln2_gamma",
                "ln2_beta",
                "W_fc1",
                "W_fc2",
                "b_fc1",
                "b_fc2",
            ):
                setattr(layer, attr, getattr(layer, attr).astype(np.float64))

        rng = np.random.default_rng(42)
        tokens = rng.integers(0, cfg.vocab_size, size=6).astype(np.int64)
        target = int(rng.integers(0, cfg.vocab_size))

        # Analytic gradient.
        logits, cache = model.forward_with_cache(tokens)
        p = _softmax(logits[-1])
        dlogits = np.zeros_like(logits)
        dlogits[-1] = p
        dlogits[-1, target] -= 1.0
        grads = model.backward(cache, dlogits)

        # Numeric gradient on a single token_emb row (used token).
        used_token = int(tokens[0])
        eps = 1e-6
        orig = model.token_emb[used_token, 0]
        model.token_emb[used_token, 0] = orig + eps
        l_plus = float(-np.log(_softmax(model.forward(tokens)[0][-1])[target] + 1e-12))
        model.token_emb[used_token, 0] = orig - eps
        l_minus = float(-np.log(_softmax(model.forward(tokens)[0][-1])[target] + 1e-12))
        model.token_emb[used_token, 0] = orig
        numeric_grad = (l_plus - l_minus) / (2 * eps)
        analytic_grad = float(grads["token_emb"][used_token, 0])
        # The analytic gradient includes contributions from both the
        # embedding path AND the LM head path (weight tying).
        assert abs(numeric_grad - analytic_grad) / (abs(numeric_grad) + 1e-8) < 1e-3, (
            f"tied grad mismatch: numeric={numeric_grad:.4e}, analytic={analytic_grad:.4e}"
        )


# ─── Label smoothing ──────────────────────────────────────────────────────
class TestLabelSmoothing:
    def test_zero_smoothing_matches_standard_ce(self):
        """smoothing=0 should reduce to standard cross-entropy."""
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 1, (4, 10))
        targets = np.array([0, 3, 7, 9])
        loss_sm, dlogits_sm = label_smoothing_cross_entropy(logits, targets, smoothing=0.0)
        # Standard CE.
        p = _softmax(logits)
        nll = -np.log(p[np.arange(4), targets] + 1e-12)
        loss_std = float(nll.mean())
        assert abs(float(loss_sm) - loss_std) < 1e-3

    def test_loss_is_non_negative(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 1, (4, 10))
        targets = np.array([0, 3, 7, 9])
        loss, _ = label_smoothing_cross_entropy(logits, targets, smoothing=0.1)
        assert loss >= 0.0

    def test_loss_decreases_with_better_predictions(self):
        """A confident correct prediction should have lower loss than a wrong one."""
        V = 5
        targets = np.array([2])
        # Confident correct.
        good = np.array([[0, 0, 10, 0, 0]], dtype=np.float64)
        # Confident wrong.
        bad = np.array([[0, 0, 0, 0, 10]], dtype=np.float64)
        loss_good, _ = label_smoothing_cross_entropy(good, targets, smoothing=0.1)
        loss_bad, _ = label_smoothing_cross_entropy(bad, targets, smoothing=0.1)
        assert loss_good < loss_bad

    def test_dlogits_shape(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 1, (4, 10))
        targets = np.array([0, 3, 7, 9])
        _, dlogits = label_smoothing_cross_entropy(logits, targets, smoothing=0.1)
        assert dlogits.shape == logits.shape

    def test_ignore_index_masks_loss(self):
        """Targets with ignore_index should contribute zero loss."""
        V = 5
        logits = np.array([[0, 0, 10, 0, 0], [0, 0, 0, 0, 10]], dtype=np.float64)
        targets = np.array([2, -100])  # second position is masked
        loss, dlogits = label_smoothing_cross_entropy(
            logits, targets, smoothing=0.0, ignore_index=-100
        )
        # Loss should be from only the first position.
        p = _softmax(logits[:1])
        expected = float(-np.log(p[0, 2] + 1e-12))
        assert abs(float(loss) - expected) < 1e-3
        # dlogits for the masked position should be zero.
        assert np.allclose(dlogits[1], 0.0, atol=1e-6)

    def test_invalid_smoothing_raises(self):
        with pytest.raises(ValueError, match="smoothing must be in"):
            label_smoothing_cross_entropy(np.zeros((2, 3)), np.array([0, 1]), smoothing=1.5)

    def test_gradient_check_label_smoothing(self):
        """Finite-difference check of the dlogits gradient."""
        rng = np.random.default_rng(42)
        logits = rng.normal(0, 1, (3, 8)).astype(np.float64)
        targets = np.array([1, 4, 6])
        smoothing = 0.1
        _, dlogits = label_smoothing_cross_entropy(logits, targets, smoothing=smoothing)

        # Numeric gradient on a single logit.
        i, j = 1, 3
        eps = 1e-6
        orig = logits[i, j]
        logits[i, j] = orig + eps
        l_plus, _ = label_smoothing_cross_entropy(logits, targets, smoothing=smoothing)
        logits[i, j] = orig - eps
        l_minus, _ = label_smoothing_cross_entropy(logits, targets, smoothing=smoothing)
        logits[i, j] = orig
        numeric = float((l_plus - l_minus) / (2 * eps))
        analytic = float(dlogits[i, j])
        assert abs(numeric - analytic) / (abs(numeric) + 1e-6) < 5e-3, (
            f"label smoothing grad mismatch: numeric={numeric:.4e}, analytic={analytic:.4e}"
        )


# ─── Gradient clipping ────────────────────────────────────────────────────
class TestGradientClipping:
    def test_no_clip_when_under_threshold(self):
        grads = {"a": np.ones(10, dtype=np.float32) * 0.1}
        norm = clip_grad_norm_(grads, max_norm=10.0)
        # Total norm = sqrt(10 * 0.01) = sqrt(0.1) ≈ 0.316
        assert abs(norm - np.sqrt(0.1)) < 1e-4
        # Gradients unchanged (within float32 precision).
        np.testing.assert_allclose(grads["a"], np.ones(10) * 0.1, atol=1e-6)

    def test_clips_when_over_threshold(self):
        grads = {"a": np.ones(100, dtype=np.float32) * 10.0}
        # Total norm = sqrt(100 * 100) = 100
        norm = clip_grad_norm_(grads, max_norm=1.0)
        assert norm > 1.0
        # After clipping, norm should be ~1.0.
        new_norm = float(np.sqrt(np.sum(grads["a"] ** 2)))
        assert abs(new_norm - 1.0) < 1e-3

    def test_zero_max_norm_skips(self):
        grads = {"a": np.ones(10) * 5.0}
        norm = clip_grad_norm_(grads, max_norm=0.0)
        assert norm == 0.0
        # Gradients unchanged.
        np.testing.assert_array_equal(grads["a"], np.ones(10) * 5.0)

    def test_multiple_arrays(self):
        grads = {
            "a": np.array([3.0, 4.0]),  # norm 5
            "b": np.array([0.0, 12.0]),  # norm 12
        }
        # Total norm = sqrt(5² + 12²) = 13
        norm = clip_grad_norm_(grads, max_norm=6.5)
        assert abs(norm - 13.0) < 1e-4
        # After clipping, total norm should be ~6.5.
        new_norm = float(np.sqrt(np.sum(grads["a"] ** 2) + np.sum(grads["b"] ** 2)))
        assert abs(new_norm - 6.5) < 1e-3

    def test_preserves_relative_ratios(self):
        """Clipping should scale all gradients by the same factor."""
        grads = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.0, 2.0, 0.0]),
        }
        original_ratio = np.linalg.norm(grads["b"]) / np.linalg.norm(grads["a"])
        clip_grad_norm_(grads, max_norm=0.5)
        new_ratio = np.linalg.norm(grads["b"]) / np.linalg.norm(grads["a"])
        assert abs(original_ratio - new_ratio) < 1e-4
