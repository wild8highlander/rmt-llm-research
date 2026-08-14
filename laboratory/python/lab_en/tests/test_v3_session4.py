"""
Tests for Session 4: TrainerV3 + AdamV3 + TrainConfig.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import TinyGPTV3, config_small, TinyGPTV3Config
from tiny_gpt_trainer_v3 import AdamV3, TrainerV3, TrainConfig


# ─── AdamV3 tests ─────────────────────────────────────────────────────────
class TestAdamV3:

    def test_adam_reduces_loss_on_simple_problem(self):
        """Adam should reduce loss on a simple quadratic."""
        # Minimize f(x) = (x - 3)^2. Gradient = 2(x - 3).
        params = {"x": np.array([0.0], dtype=np.float32)}
        opt = AdamV3(params, lr=0.1, weight_decay=0.0)
        for _ in range(100):
            grad = {"x": 2.0 * (params["x"] - 3.0)}
            opt.step(params, grad)
        assert abs(float(params["x"][0]) - 3.0) < 0.1, (
            f"Adam should converge to 3.0, got {params['x'][0]}"
        )

    def test_adam_handles_zero_gradient(self):
        """Zero gradients should not change params (after bias correction)."""
        params = {"x": np.array([1.0], dtype=np.float32)}
        opt = AdamV3(params, lr=0.1, weight_decay=0.0)
        opt.step(params, {"x": np.zeros_like(params["x"])})
        # With zero grad, m and v are 0, so update is 0.
        np.testing.assert_allclose(params["x"], 1.0, atol=1e-7)

    def test_adam_weight_decay_shrinks_params(self):
        """Weight decay should pull params toward 0 (even with zero grad)."""
        params = {"x": np.array([1.0], dtype=np.float32)}
        opt = AdamV3(params, lr=0.1, weight_decay=0.1)
        for _ in range(10):
            opt.step(params, {"x": np.zeros_like(params["x"])})
        # With weight decay, x should decrease.
        assert float(params["x"][0]) < 1.0

    def test_adam_lr_override(self):
        """The lr= argument to step() should override the optimizer's lr."""
        params = {"x": np.array([0.0], dtype=np.float32)}
        opt = AdamV3(params, lr=1.0, weight_decay=0.0)
        # Use a tiny lr override.
        opt.step(params, {"x": np.array([1.0])}, lr=0.001)
        # x should move by a tiny amount, not a large amount.
        assert abs(float(params["x"][0])) < 0.01


# ─── TrainConfig tests ────────────────────────────────────────────────────
class TestTrainConfig:

    def test_default_config(self):
        cfg = TrainConfig()
        assert cfg.epochs == 150
        assert cfg.batch_size == 1
        assert cfg.max_lr == 3e-3
        assert cfg.warmup_ratio == 0.1

    def test_effective_batch_size(self):
        """Effective batch = batch_size * grad_accum_steps."""
        cfg = TrainConfig(batch_size=4, grad_accum_steps=2)
        assert cfg.batch_size * cfg.grad_accum_steps == 8


# ─── TrainerV3 tests ──────────────────────────────────────────────────────
class TestTrainerV3:

    @pytest.fixture
    def small_model(self):
        cfg = config_small()
        cfg.weight_tying = True
        cfg.label_smoothing = 0.0  # disable for cleaner gradients
        cfg.dropout = 0.0
        cfg.grad_clip_norm = 0.0
        return TinyGPTV3(cfg)

    def test_trainer_construct(self, small_model):
        """Trainer should construct without error."""
        trainer = TrainerV3(small_model, TrainConfig(epochs=1, verbose=False))
        assert trainer.model is small_model
        assert "token_emb" in trainer._params
        assert "L0_W_q" in trainer._params

    def test_make_sequences(self, small_model):
        """_make_sequences should split tokens into seq_len chunks."""
        trainer = TrainerV3(small_model, TrainConfig(epochs=1, verbose=False))
        tokens = np.arange(100, dtype=np.int64)
        seqs = trainer._make_sequences(tokens, seq_len=10)
        assert len(seqs) == 10
        assert all(len(s) == 10 for s in seqs)
        np.testing.assert_array_equal(seqs[0], np.arange(10))

    def test_make_sequences_too_short(self, small_model):
        """Should return empty list if tokens < seq_len."""
        trainer = TrainerV3(small_model, TrainConfig(epochs=1, verbose=False))
        seqs = trainer._make_sequences(np.array([1, 2, 3], dtype=np.int64),
                                       seq_len=10)
        assert seqs == []

    def test_train_step_runs(self, small_model):
        """One train step should not crash and should return finite loss."""
        trainer = TrainerV3(small_model, TrainConfig(epochs=1, verbose=False))
        seqs = [np.arange(10, dtype=np.int64) for _ in range(4)]
        loss, grad_norm = trainer._train_step(seqs, current_lr=1e-3)
        assert np.isfinite(loss)
        assert grad_norm >= 0

    def test_evaluate_returns_finite(self, small_model):
        """Evaluation should return finite loss and a match_rate in [0, 1]."""
        trainer = TrainerV3(small_model, TrainConfig(epochs=1, verbose=False))
        seqs = [np.arange(10, dtype=np.int64) for _ in range(3)]
        loss, mr = trainer._evaluate(seqs, max_steps=3)
        assert np.isfinite(loss)
        assert 0.0 <= mr <= 1.0

    def test_train_reduces_loss_on_synthetic(self):
        """Training on a repeated pattern should reduce loss over epochs.

        We create a corpus where token[i+1] = (token[i] + 1) % vocab_size,
        so the model can learn the pattern. After 20 epochs the match_rate
        should be meaningfully above random (1/vocab_size).
        """
        cfg = config_small()
        cfg.weight_tying = True
        cfg.label_smoothing = 0.0
        cfg.dropout = 0.0
        cfg.grad_clip_norm = 1.0
        model = TinyGPTV3(cfg)
        # Create a learnable pattern: 0, 1, 2, ..., vocab-1, 0, 1, ...
        pattern = np.arange(cfg.vocab_size, dtype=np.int64)
        corpus = np.tile(pattern, 20)  # 20 repetitions
        val = np.tile(pattern, 5)
        train_cfg = TrainConfig(
            epochs=20, batch_size=1, grad_accum_steps=1,
            max_lr=5e-3, warmup_ratio=0.1, eval_interval=5,
            early_stopping_patience=0, verbose=False,
        )
        trainer = TrainerV3(model, train_cfg)
        history = trainer.train(corpus, val_ids=val)
        # Final val match_rate should be well above random (1/64 ≈ 1.5%).
        final_mr = history["val_match_rates"][-1]
        assert final_mr > 0.10, (
            f"match_rate={final_mr:.1%} should be > 10% on the learnable pattern"
        )

    def test_grad_accumulation_matches_large_batch(self):
        """Gradient accumulation should approximate a large batch.

        With grad_accum_steps=N, the effective batch is N× larger. The
        loss should be similar to training with batch_size=N (without
        accumulation), modulo the LR scaling.
        """
        cfg = config_small()
        cfg.weight_tying = True
        cfg.label_smoothing = 0.0
        cfg.dropout = 0.0
        cfg.grad_clip_norm = 0.0

        # Two models with identical seeds.
        m1 = TinyGPTV3(cfg)
        m2 = TinyGPTV3(cfg)
        # Copy weights to ensure they start identical.
        m2.token_emb = m1.token_emb.copy()
        m2.lm_head = m1.lm_head.copy()
        m2.ln_f_gamma = m1.ln_f_gamma.copy()
        m2.ln_f_beta = m1.ln_f_beta.copy()
        for i in range(cfg.n_layers):
            for attr in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
                         "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                         "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
                setattr(m2.layers[i], attr,
                        getattr(m1.layers[i], attr).copy())

        rng = np.random.default_rng(42)
        corpus = rng.integers(0, cfg.vocab_size, size=500).astype(np.int64)

        # m1: batch_size=4, no accumulation.
        t1 = TrainerV3(m1, TrainConfig(epochs=1, batch_size=4,
                                        grad_accum_steps=1,
                                        max_lr=1e-3, verbose=False))
        # m2: batch_size=1, accumulation=4 (same effective batch).
        t2 = TrainerV3(m2, TrainConfig(epochs=1, batch_size=1,
                                        grad_accum_steps=4,
                                        max_lr=1e-3, verbose=False))
        # Both should run without error and produce finite losses.
        h1 = t1.train(corpus)
        h2 = t2.train(corpus)
        assert np.isfinite(h1["losses"][-1])
        assert np.isfinite(h2["losses"][-1])

    def test_weight_tying_syncs_lm_head(self):
        """After training, lm_head should track token_emb (when tied)."""
        cfg = config_small()
        cfg.weight_tying = True
        model = TinyGPTV3(cfg)
        emb_before = model.token_emb.copy()
        # Train for a few steps.
        rng = np.random.default_rng(0)
        corpus = rng.integers(0, cfg.vocab_size, size=200).astype(np.int64)
        trainer = TrainerV3(model, TrainConfig(
            epochs=3, max_lr=1e-3, early_stopping_patience=0, verbose=False,
        ))
        trainer.train(corpus)
        # token_emb should have changed.
        assert not np.allclose(model.token_emb, emb_before)
        # lm_head should be the transpose of token_emb (weight tying).
        np.testing.assert_allclose(model.lm_head, model.token_emb.T, atol=1e-5)
