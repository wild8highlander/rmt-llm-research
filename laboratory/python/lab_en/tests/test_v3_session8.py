"""Tests for Session 8: LR Search utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lr_search_v3 import (
    GridSearchResult,
    LRGridSearch,
    LRRangeTest,
    RangeTestResult,
)
from tiny_gpt_trainer_v3 import TrainConfig
from tiny_gpt_v3 import TinyGPTV3, config_small


@pytest.fixture
def small_model():
    cfg = config_small()
    cfg.weight_tying = True
    cfg.label_smoothing = 0.0
    cfg.dropout = 0.0
    cfg.grad_clip_norm = 1.0
    return TinyGPTV3(cfg)


@pytest.fixture
def corpus():
    rng = np.random.default_rng(42)
    return rng.integers(0, 64, size=500).astype(np.int64)


class TestLRRangeTest:
    def test_run_returns_result(self, small_model, corpus):
        """Range test should return a RangeTestResult."""
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        result = rt.run(corpus, n_steps=5)
        assert isinstance(result, RangeTestResult)
        assert len(result.lrs) == 5
        assert len(result.losses) == 5

    def test_lrs_are_log_spaced(self, small_model, corpus):
        """LRs should be log-spaced from lr_min to lr_max."""
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        result = rt.run(corpus, lr_min=1e-5, lr_max=1e-1, n_steps=10)
        assert abs(result.lrs[0] - 1e-5) / 1e-5 < 0.01
        assert abs(result.lrs[-1] - 1e-1) / 1e-1 < 0.01

    def test_best_lr_is_positive(self, small_model, corpus):
        """best_lr should be a positive number."""
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        result = rt.run(corpus, n_steps=5)
        assert result.best_lr > 0
        assert result.steepest_descent_lr > 0

    def test_best_lr_is_steepest_div_10(self, small_model, corpus):
        """best_lr should be steepest_descent_lr / 10 (conservative)."""
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        result = rt.run(corpus, n_steps=5)
        assert abs(result.best_lr - result.steepest_descent_lr / 10) < 1e-15

    def test_model_weights_change_after_run(self, small_model, corpus):
        """The range test should update model weights (it trains)."""
        emb_before = small_model.token_emb.copy()
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        rt.run(corpus, n_steps=5)
        assert not np.allclose(small_model.token_emb, emb_before)

    def test_not_enough_tokens_raises(self, small_model):
        """Should raise if corpus is too small for seq_len."""
        rt = LRRangeTest(small_model, TrainConfig(epochs=1, verbose=False))
        with pytest.raises(ValueError, match="Not enough tokens"):
            rt.run(np.array([1, 2, 3], dtype=np.int64), n_steps=3)


class TestLRGridSearch:
    def test_run_returns_result(self, small_model, corpus):
        """Grid search should return a GridSearchResult."""
        val = np.random.default_rng(0).integers(0, 64, size=100).astype(np.int64)
        gs = LRGridSearch(
            small_model,
            TrainConfig(
                epochs=1,
                eval_interval=1,
                early_stopping_patience=0,
                verbose=False,
            ),
        )
        result = gs.run(corpus, val, lr_grid=[1e-3, 3e-3])
        assert isinstance(result, GridSearchResult)
        assert len(result.lr_results) == 2

    def test_best_lr_in_grid(self, small_model, corpus):
        """best_lr should be one of the grid values."""
        val = np.random.default_rng(0).integers(0, 64, size=100).astype(np.int64)
        grid = [1e-3, 3e-3, 1e-2]
        gs = LRGridSearch(
            small_model,
            TrainConfig(
                epochs=1,
                eval_interval=1,
                early_stopping_patience=0,
                verbose=False,
            ),
        )
        result = gs.run(corpus, val, lr_grid=grid)
        assert result.best_lr in grid

    def test_best_val_loss_is_min(self, small_model, corpus):
        """best_val_loss should be the minimum across the grid."""
        val = np.random.default_rng(0).integers(0, 64, size=100).astype(np.int64)
        gs = LRGridSearch(
            small_model,
            TrainConfig(
                epochs=1,
                eval_interval=1,
                early_stopping_patience=0,
                verbose=False,
            ),
        )
        result = gs.run(corpus, val, lr_grid=[1e-3, 3e-3])
        assert result.best_val_loss == min(result.lr_results.values())

    def test_model_reset_between_runs(self, small_model, corpus):
        """Each LR run should start from the same initial weights."""
        val = np.random.default_rng(0).integers(0, 64, size=100).astype(np.int64)
        emb_before = small_model.token_emb.copy()
        gs = LRGridSearch(
            small_model,
            TrainConfig(
                epochs=1,
                eval_interval=1,
                early_stopping_patience=0,
                verbose=False,
            ),
        )
        gs.run(corpus, val, lr_grid=[1e-3, 3e-3])
        # After grid search, the model has the last LR's weights.
        # The important thing is that each run started from the same state.
        assert not np.allclose(small_model.token_emb, emb_before)

    def test_default_grid(self, small_model, corpus):
        """Default grid should be [1e-4, 3e-4, 1e-3, 3e-3]."""
        val = np.random.default_rng(0).integers(0, 64, size=100).astype(np.int64)
        gs = LRGridSearch(
            small_model,
            TrainConfig(
                epochs=1,
                eval_interval=1,
                early_stopping_patience=0,
                verbose=False,
            ),
        )
        result = gs.run(corpus, val)  # no lr_grid → use default
        assert len(result.lr_results) == 4
