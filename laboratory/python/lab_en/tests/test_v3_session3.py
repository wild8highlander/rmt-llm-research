"""
Tests for Session 3: config_12m, cosine LR schedule, early stopping.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import (
    TinyGPTV3,
    TinyGPTV3Config,
    config_4m,
    config_12m,
    config_train_4m,
    config_small,
    cosine_lr_schedule,
    linear_lr_schedule,
    EarlyStopping,
)


# ─── config_12m tests ─────────────────────────────────────────────────────
class TestConfig12M:

    def test_config_12m_params_in_range(self):
        """12M config should target ~10-15M parameters."""
        cfg = config_12m()
        n = cfg.params_count
        assert 10_000_000 < n < 15_000_000, f"params_count={n} outside 12M range"

    def test_config_12m_has_all_training_features(self):
        """12M config should enable all Tier 1+2 features."""
        cfg = config_12m()
        assert cfg.weight_tying is True
        assert cfg.label_smoothing == 0.1
        assert cfg.grad_clip_norm == 1.0
        assert cfg.dropout == 0.1
        assert cfg.use_rope is True

    def test_config_12m_larger_than_4m(self):
        """12M config should have more params than 4M config."""
        n4 = config_4m().params_count
        n12 = config_12m().params_count
        assert n12 > n4 * 2  # at least 2x larger

    def test_config_train_4m_has_training_features(self):
        """config_train_4m should enable all Tier 1+2 features."""
        cfg = config_train_4m()
        assert cfg.weight_tying is True
        assert cfg.label_smoothing == 0.1
        assert cfg.grad_clip_norm == 1.0
        assert cfg.dropout == 0.1

    def test_config_train_4m_params_match_4m(self):
        """config_train_4m should have same param count as config_4m
        (weight tying saves lm_head params)."""
        n_plain = TinyGPTV3Config(
            vocab_size=512, hidden_dim=192, n_layers=10,
            n_heads=6, n_kv_heads=2, max_seq_len=256,
            use_rope=True, weight_tying=False,
        ).params_count
        n_train = config_train_4m().params_count
        # Tying should save H * V = 192 * 512 = 98304 params.
        assert n_plain - n_train == 192 * 512


# ─── Cosine LR schedule tests ──────────────────────────────────────────────
class TestCosineLRSchedule:

    def test_warmup_starts_near_zero(self):
        """At step 0 with warmup, LR should be small."""
        lr = cosine_lr_schedule(0, max_lr=1e-3, warmup_steps=100,
                                total_steps=1000)
        assert lr < 2e-5  # 1e-3 * 1/100 ≈ 1e-5

    def test_warmup_reaches_max_at_warmup_steps(self):
        """At the end of warmup, LR should equal max_lr."""
        lr = cosine_lr_schedule(99, max_lr=1e-3, warmup_steps=100,
                                total_steps=1000)
        # step 99 (0-indexed) is the last warmup step.
        assert abs(lr - 1e-3) / 1e-3 < 0.05  # within 5%

    def test_decay_reaches_min_at_end(self):
        """At total_steps, LR should be min_lr_ratio * max_lr."""
        lr = cosine_lr_schedule(999, max_lr=1e-3, warmup_steps=100,
                                total_steps=1000, min_lr_ratio=0.1)
        assert abs(lr - 1e-4) / 1e-4 < 0.05  # within 5% of 0.1 * 1e-3

    def test_lr_is_monotonic_decreasing_after_warmup(self):
        """After warmup, LR should monotonically decrease."""
        lrs = [
            cosine_lr_schedule(s, max_lr=1e-3, warmup_steps=100,
                              total_steps=1000, min_lr_ratio=0.1)
            for s in range(100, 1000)
        ]
        diffs = np.diff(lrs)
        # All diffs should be <= 0 (monotonic non-increasing).
        assert np.all(diffs <= 1e-12), f"LR increased at some step: max diff={diffs.max()}"

    def test_lr_always_positive(self):
        """LR should always be positive."""
        for s in range(0, 1000):
            lr = cosine_lr_schedule(s, max_lr=1e-3, warmup_steps=100,
                                    total_steps=1000, min_lr_ratio=0.1)
            assert lr > 0

    def test_zero_warmup_starts_at_max(self):
        """With warmup_steps=0, LR should start at max_lr."""
        lr = cosine_lr_schedule(0, max_lr=1e-3, warmup_steps=0,
                                total_steps=1000)
        assert abs(lr - 1e-3) < 1e-6

    def test_total_steps_zero_returns_max(self):
        """Edge case: total_steps=0 should return max_lr."""
        lr = cosine_lr_schedule(0, max_lr=1e-3, warmup_steps=10,
                                total_steps=0)
        assert lr == 1e-3


class TestLinearLRSchedule:

    def test_linear_decay_reaches_min(self):
        """Linear decay should reach min_lr_ratio * max_lr at the end."""
        lr = linear_lr_schedule(999, max_lr=1e-3, warmup_steps=100,
                                total_steps=1000, min_lr_ratio=0.0)
        assert lr < 2e-6  # ~0, but not exactly 0 due to step indexing

    def test_linear_monotonic_after_warmup(self):
        """After warmup, linear LR should monotonically decrease."""
        lrs = [
            linear_lr_schedule(s, max_lr=1e-3, warmup_steps=100,
                              total_steps=1000, min_lr_ratio=0.0)
            for s in range(100, 1000)
        ]
        diffs = np.diff(lrs)
        assert np.all(diffs <= 1e-12)


# ─── Early stopping tests ─────────────────────────────────────────────────
class TestEarlyStopping:

    def test_min_mode_improves_on_decrease(self):
        """In 'min' mode, a lower metric should reset the counter."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)
        assert es.counter == 0
        es.step(0.9)  # improvement
        assert es.counter == 0
        assert es.best == 0.9

    def test_min_mode_increments_on_no_improvement(self):
        """In 'min' mode, no improvement should increment counter."""
        es = EarlyStopping(patience=3, mode="min")
        es.step(1.0)
        es.step(1.0)  # no improvement
        assert es.counter == 1
        es.step(1.0)
        assert es.counter == 2

    def test_min_mode_stops_after_patience(self):
        """Should stop after patience consecutive non-improvements."""
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0)
        assert not es.stopped
        es.step(1.0)  # counter=1
        assert not es.stopped
        es.step(1.0)  # counter=2 → stop
        assert es.stopped
        assert es.step(1.0) is True  # stays stopped

    def test_max_mode_improves_on_increase(self):
        """In 'max' mode, a higher metric should reset the counter."""
        es = EarlyStopping(patience=3, mode="max")
        es.step(0.5)
        es.step(0.6)  # improvement
        assert es.counter == 0
        assert es.best == 0.6

    def test_min_delta_requires_significant_change(self):
        """min_delta filters out small improvements."""
        es = EarlyStopping(patience=2, mode="min", min_delta=0.05)
        es.step(1.0)
        es.step(0.98)  # improvement < min_delta → no improvement
        assert es.counter == 1

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="bad")

    def test_reset_clears_state(self):
        es = EarlyStopping(patience=2, mode="min")
        es.step(1.0)
        es.step(1.0)
        assert es.counter == 1
        es.reset()
        assert es.counter == 0
        assert es.stopped is False

    def test_realistic_training_scenario(self):
        """Simulate a training loop where loss plateaus then stops."""
        es = EarlyStopping(patience=3, mode="min", min_delta=0.001)
        losses = [2.0, 1.5, 1.2, 1.1, 1.05, 1.04, 1.04, 1.04, 1.04]
        stopped_at = None
        for i, loss in enumerate(losses):
            if es.step(loss):
                stopped_at = i
                break
        # Loss plateaus at 1.04 starting from index 5.
        # patience=3 → stop at index 5+3=8.
        assert stopped_at == 8
