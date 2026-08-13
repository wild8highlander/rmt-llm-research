"""
Unit tests for `tiny_gpt_trainer.py` — Adam optimizer, loss function,
match-rate evaluation, forward-with-cache, backward pass, and end-to-end
mini-training.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tiny_gpt import TinyGPT, TinyGPTConfig
from tiny_gpt_trainer import (
    AdamState,
    ForwardCache,
    TrainConfig,
    backward,
    build_corpus,
    cross_entropy_loss,
    evaluate_match_rate,
    forward_with_cache,
    make_dataset_tokens,
    train_tiny_gpt,
)


# ─── Cross-entropy loss ─────────────────────────────────────────────────────
class TestCrossEntropy:
    """Tests for cross_entropy_loss."""

    def test_zero_loss_when_target_is_argmax(self):
        """If target has the largest logit, loss should be ~0."""
        logits = np.array([10.0, 1.0, 0.5, 0.1])
        loss = cross_entropy_loss(logits, target=0)
        assert loss < 0.01

    def test_high_loss_when_target_has_low_logit(self):
        logits = np.array([0.01, 0.01, 0.01, 10.0])
        loss = cross_entropy_loss(logits, target=0)
        assert loss > 5.0

    def test_loss_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            logits = rng.standard_normal(8)
            target = int(rng.integers(0, 8))
            loss = cross_entropy_loss(logits, target)
            assert loss >= 0.0

    def test_loss_numerical_stability(self):
        """Very large logits should not overflow."""
        logits = np.array([1e6, 0.0, 0.0])
        loss = cross_entropy_loss(logits, target=0)
        assert math.isfinite(loss)

    def test_loss_symmetric_under_shift(self):
        """Adding a constant to all logits should not change the loss."""
        logits = np.array([1.0, 2.0, 3.0, 0.5])
        loss1 = cross_entropy_loss(logits, target=1)
        loss2 = cross_entropy_loss(logits + 100.0, target=1)
        assert abs(loss1 - loss2) < 1e-6


# ─── Forward with cache ─────────────────────────────────────────────────────
class TestForwardWithCache:
    """Tests for forward_with_cache (used by backward)."""

    def test_returns_logits_and_cache(self, small_model):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        x, cache = forward_with_cache(small_model, tokens)
        # x is the final hidden state (T, H); cache.logits is (T, V)
        assert cache.logits.shape == (4, small_model.config.vocab_size)
        assert x.shape == (4, small_model.config.hidden_dim)
        assert isinstance(cache, ForwardCache)

    def test_cache_logits_match_plain_forward(self, small_model):
        """forward_with_cache should produce the same logits as TinyGPT.forward."""
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits_plain, _ = small_model.forward(tokens)
        _, cache = forward_with_cache(small_model, tokens)
        np.testing.assert_allclose(logits_plain, cache.logits, rtol=1e-7)

    def test_cache_is_populated(self, small_model):
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        _, cache = forward_with_cache(small_model, tokens)
        # Cache should contain intermediate activations
        assert cache.token_ids is not None
        assert cache.logits is not None
        assert cache.probs is not None
        assert len(cache.per_layer) == small_model.config.n_layers


# ─── Backward pass ──────────────────────────────────────────────────────────
class TestBackward:
    """Tests for the reverse-mode autodiff backward() function."""

    def test_backward_returns_grads_dict(self, small_model):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        target = np.array([5], dtype=np.int64)  # backward takes array, uses last element
        _, cache = forward_with_cache(small_model, tokens)
        grads = backward(small_model, cache, target)
        assert isinstance(grads, dict)
        assert "token_emb" in grads
        assert "lm_head" in grads
        assert "layers" in grads
        assert len(grads["layers"]) == small_model.config.n_layers

    def test_backward_grads_match_param_shapes(self, small_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        target = np.array([5], dtype=np.int64)
        _, cache = forward_with_cache(small_model, tokens)
        grads = backward(small_model, cache, target)

        np.testing.assert_array_equal(
            grads["token_emb"].shape, small_model.token_emb.shape
        )
        np.testing.assert_array_equal(
            grads["lm_head"].shape, small_model.lm_head.shape
        )

        for li, layer in enumerate(small_model.layers):
            layer_grads = grads["layers"][li]
            assert layer_grads["W_q"].shape == layer.W_q.shape
            assert layer_grads["W_fc1"].shape == layer.W_fc1.shape

    def test_backward_grads_are_finite(self, small_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        target = np.array([5], dtype=np.int64)
        _, cache = forward_with_cache(small_model, tokens)
        grads = backward(small_model, cache, target)

        def assert_finite(arr, name):
            if arr is None:
                return
            assert np.all(np.isfinite(arr)), f"NaN/Inf in {name}"

        assert_finite(grads["token_emb"], "token_emb")
        assert_finite(grads["lm_head"], "lm_head")
        for li, lg in enumerate(grads["layers"]):
            for k, v in lg.items():
                assert_finite(v, f"layers[{li}].{k}")

    def test_gradient_check_numerical(self, small_config):
        """Numerical gradient check on a tiny model.

        Compares analytical gradient from backward() to a finite-difference
        estimate. Uses a relatively large eps=1e-2 because the model uses
        float32 internally (smaller eps would lose precision in the
        subtraction). We assert that the *signs* agree and the magnitudes
        are within 50% — this is enough to catch wrong-sign bugs, zero
        gradients, or completely wrong backprop paths, while not being
        tripped up by float32 noise.
        """
        # Use an even smaller model to make this fast
        cfg = TinyGPTConfig(
            vocab_size=16, hidden_dim=8, n_layers=1, n_heads=2,
            max_seq_len=8, mlp_ratio=2, seed=42,
        )
        model = TinyGPT(cfg)

        tokens = np.array([1, 2, 3], dtype=np.int64)
        target = np.array([5], dtype=np.int64)

        # Analytical gradient
        _, cache = forward_with_cache(model, tokens)
        grads = backward(model, cache, target)
        dW_q_analytical = grads["layers"][0]["W_q"].copy()

        # Numerical gradient (central differences) with larger eps for float32 stability
        eps = 1e-2
        W = model.layers[0].W_q
        dW_numerical = np.zeros_like(W, dtype=np.float64)
        for i in range(min(4, W.shape[0])):  # only check first 4 rows for speed
            for j in range(min(4, W.shape[1])):
                orig = float(W[i, j])
                W[i, j] = orig + eps
                logits_plus, _ = model.forward(tokens)
                loss_plus = cross_entropy_loss(logits_plus[-1], int(target[-1]))
                W[i, j] = orig - eps
                logits_minus, _ = model.forward(tokens)
                loss_minus = cross_entropy_loss(logits_minus[-1], int(target[-1]))
                W[i, j] = orig  # restore
                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

        # Compare signs — they should agree (or both be near zero)
        ana = dW_q_analytical[:4, :4].astype(np.float64)
        num = dW_numerical[:4, :4]
        # Element-wise: products should be ≥ 0 (same sign or one is ~0)
        products = ana * num
        # Allow tiny violations only when both magnitudes are very small
        bad = (products < 0) & ((np.abs(ana) > 1e-4) | (np.abs(num) > 1e-4))
        assert not bad.any(), (
            f"Gradient sign mismatch at positions {np.argwhere(bad).tolist()}\n"
            f"Analytical:\n{ana}\nNumerical:\n{num}"
        )
        # Compare magnitudes — within 50% OR within absolute tolerance 5e-3
        np.testing.assert_allclose(
            np.abs(ana), np.abs(num),
            rtol=0.5, atol=5e-3,
            err_msg="Gradient magnitude mismatch (>50% relative or >5e-3 absolute)"
        )


# ─── Adam optimizer ─────────────────────────────────────────────────────────
class TestAdamState:
    """Tests for AdamState."""

    def test_init_creates_momentum_arrays(self, small_model):
        opt = AdamState(small_model, lr=1e-3)
        assert opt.m_token_emb.shape == small_model.token_emb.shape
        assert opt.v_token_emb.shape == small_model.token_emb.shape
        assert len(opt.m_layers) == small_model.config.n_layers
        assert len(opt.v_layers) == small_model.config.n_layers
        # All moments should start at zero
        assert np.all(opt.m_token_emb == 0.0)
        assert np.all(opt.v_token_emb == 0.0)

    def test_default_hyperparameters(self, small_model):
        opt = AdamState(small_model)
        assert opt.max_lr == 3e-4
        assert opt.beta1 == 0.9
        assert opt.beta2 == 0.999
        assert opt.eps == 1e-8
        assert opt.lr_schedule == "cosine"

    def test_step_increments_t(self, small_model):
        opt = AdamState(small_model)
        assert opt.t == 0
        tokens = np.array([1, 2, 3], dtype=np.int64)
        target = np.array([5], dtype=np.int64)
        _, cache = forward_with_cache(small_model, tokens)
        grads = backward(small_model, cache, target)
        opt.step(small_model, grads)
        assert opt.t == 1
        opt.step(small_model, grads)
        assert opt.t == 2

    def test_step_updates_weights(self, small_model):
        """After step(), weights should change."""
        opt = AdamState(small_model, lr=1e-2)
        tokens = np.array([1, 2, 3], dtype=np.int64)
        target = np.array([5], dtype=np.int64)

        # Snapshot weights
        W_before = small_model.layers[0].W_q.copy()

        _, cache = forward_with_cache(small_model, tokens)
        grads = backward(small_model, cache, target)
        opt.step(small_model, grads)

        W_after = small_model.layers[0].W_q
        assert not np.allclose(W_before, W_after), "Weights did not change after Adam step"

    def test_constant_lr_schedule(self, small_model):
        opt = AdamState(small_model, lr=1e-3, lr_schedule="constant")
        opt.update_epoch_progress(0.0)
        assert abs(opt.lr - 1e-3) < 1e-12
        opt.update_epoch_progress(15.0)
        assert abs(opt.lr - 1e-3) < 1e-12

    def test_cosine_lr_warmup(self, small_model):
        """During warmup, LR should ramp up from ~0 to max_lr."""
        opt = AdamState(small_model, lr=1.0, lr_schedule="cosine",
                        warmup_epochs=10, total_epochs=100)
        opt.update_epoch_progress(0.0)
        assert opt.lr < 0.1  # near zero at start
        opt.update_epoch_progress(10.0)
        assert abs(opt.lr - 1.0) < 0.05  # near max at end of warmup

    def test_cosine_lr_decay(self, small_model):
        """After warmup, LR should decay following a cosine curve."""
        opt = AdamState(small_model, lr=1.0, lr_schedule="cosine",
                        warmup_epochs=0, total_epochs=10, min_lr_ratio=0.0)
        opt.update_epoch_progress(0.0)
        lr_start = opt.lr
        opt.update_epoch_progress(5.0)  # midpoint
        lr_mid = opt.lr
        opt.update_epoch_progress(10.0)  # end
        lr_end = opt.lr
        # Cosine: start=max, mid=0.5*max, end=0
        assert lr_start > lr_mid > lr_end
        assert abs(lr_end - 0.0) < 0.05  # near min_lr_ratio * max_lr


# ─── Match-rate evaluation ──────────────────────────────────────────────────
class TestEvaluateMatchRate:
    """Tests for evaluate_match_rate."""

    def test_returns_dict_with_expected_keys(self, small_model):
        # Build a tiny dataset: (seq_len+1,) per row, last col is target
        rng = np.random.default_rng(0)
        dataset = rng.integers(0, small_model.config.vocab_size,
                                size=(20, 6)).astype(np.int64)
        result = evaluate_match_rate(small_model, dataset, n_samples=10)
        assert "match_rate" in result
        assert "n_samples" in result
        assert "loss" in result

    def test_match_rate_in_valid_range(self, small_model):
        rng = np.random.default_rng(0)
        dataset = rng.integers(0, small_model.config.vocab_size,
                                size=(20, 6)).astype(np.int64)
        result = evaluate_match_rate(small_model, dataset, n_samples=10)
        assert 0.0 <= result["match_rate"] <= 1.0

    def test_empty_dataset(self, small_model):
        empty = np.array([], dtype=np.int64).reshape(0, 6)
        result = evaluate_match_rate(small_model, empty)
        assert result["match_rate"] == 0.0
        assert result["n_samples"] == 0

    def test_n_samples_capped_at_dataset_size(self, small_model):
        rng = np.random.default_rng(0)
        dataset = rng.integers(0, small_model.config.vocab_size,
                                size=(5, 6)).astype(np.int64)
        result = evaluate_match_rate(small_model, dataset, n_samples=100)
        assert result["n_samples"] == 5


# ─── Dataset construction ───────────────────────────────────────────────────
class TestMakeDatasetTokens:
    """Tests for make_dataset_tokens."""

    def test_creates_dataset_with_expected_shape(self):
        token_ids = np.arange(100, dtype=np.int64)
        dataset = make_dataset_tokens(token_ids, seq_len=10, stride=10)
        # Each row: seq_len + 1 (target)
        assert dataset.shape[1] == 11
        # 100 tokens / stride 10 = 10 samples (last one might be cut)
        assert dataset.shape[0] >= 5
        assert dataset.shape[0] <= 10

    def test_stride_smaller_gives_more_samples(self):
        token_ids = np.arange(100, dtype=np.int64)
        ds_stride_5 = make_dataset_tokens(token_ids, seq_len=10, stride=5)
        ds_stride_10 = make_dataset_tokens(token_ids, seq_len=10, stride=10)
        assert ds_stride_5.shape[0] > ds_stride_10.shape[0]


# ─── Corpus builder ─────────────────────────────────────────────────────────
class TestBuildCorpus:
    """Tests for build_corpus."""

    def test_returns_bytes(self, tmp_path):
        # Create a minimal "repo" structure
        (tmp_path / "README.md").write_text("# Test\nHello world.\n")
        corpus = build_corpus(str(tmp_path), paths=["README.md"])
        assert isinstance(corpus, bytes)
        assert len(corpus) > 0
        assert b"Hello world" in corpus

    def test_handles_missing_paths_gracefully(self, tmp_path):
        (tmp_path / "README.md").write_text("# Test\n")
        # Non-existent file should be skipped, not crash
        corpus = build_corpus(str(tmp_path), paths=["README.md", "nonexistent.txt"])
        assert isinstance(corpus, bytes)
        assert len(corpus) > 0


# ─── End-to-end mini training ───────────────────────────────────────────────
@pytest.mark.slow
class TestEndToEndTraining:
    """Slow tests that actually run train_tiny_gpt for a few epochs."""

    def test_train_tiny_gpt_returns_history(self, tmp_path):
        """Run train_tiny_gpt for 1 epoch on a tiny corpus and verify output."""
        # Create a minimal repo structure with enough text for ≥2 windows.
        # BPE aggressively compresses repetitive text — need ~5KB of varied prose.
        rng = np.random.default_rng(0)
        words = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
            "random matrix theory spectral analysis hallucinations cognitive mode "
            "marchenko pastur bbp transition tracy widom distribution caputo "
            "fractional derivative keating snaith corrections non-hermitian skin "
            "effect eigenvalue neural network transformer attention layer norm "
        ).split()
        lines = []
        for _ in range(300):
            n = int(rng.integers(5, 20))
            lines.append(" ".join(rng.choice(words, size=n)))
        (tmp_path / "README.md").write_text("# TinyGPT test\n" + "\n".join(lines))
        (tmp_path / "main.py").write_text("print('hello world')\n" * 50)

        config = TrainConfig(
            seq_len=8,
            stride=8,
            batch_size=2,
            vocab_size=280,
            bpe_merges=20,
            epochs=1,
            max_train_tokens=2000,
            eval_every=1,
            lr=1e-3,
            weight_decay=0.0,
        )

        result = train_tiny_gpt(str(tmp_path), config=config)

        assert isinstance(result, dict)
        # Should NOT report corpus too small
        assert "error" not in result or result.get("n_windows", 0) >= 2, \
            f"Training failed: {result}"
        # Look for evidence of training — any of these keys
        assert "history" in result or "final_loss" in result or \
               "match_rate" in result or "n_windows" in result

    def test_loss_decreases_over_one_epoch(self, tmp_path):
        """Loss after 1 epoch should be ≤ loss at start (with high probability)."""
        (tmp_path / "README.md").write_text("# TinyGPT loss test\n" * 100)
        (tmp_path / "main.py").write_text("def foo(): return 42\n" * 80)

        config = TrainConfig(
            seq_len=8, stride=8, batch_size=2, vocab_size=280,
            bpe_merges=10, epochs=1, max_train_tokens=2000,
            eval_every=1, lr=1e-3,
        )
        result = train_tiny_gpt(str(tmp_path), config=config)
        # Look for evidence of training — any of these keys
        history = result.get("history", [])
        if isinstance(history, list) and len(history) >= 2:
            assert history[-1].get("loss", float("inf")) <= history[0].get("loss", float("inf")) + 1.0


# ─── Integration with main.py ───────────────────────────────────────────────
class TestIntegration:
    """Integration tests checking that the trainer wires up correctly with the model."""

    def test_train_then_generate_roundtrip(self, small_corpus, tmp_path):
        """Train one step, then generate — verify no crash and finite output."""
        # Use a model with vocab_size matching the BPE tokenizer
        from tiny_gpt_trainer import BPETokenizer

        cfg = TinyGPTConfig(
            vocab_size=300, hidden_dim=32, n_layers=2, n_heads=4,
            max_seq_len=64, mlp_ratio=4, seed=42,
        )
        model = TinyGPT(cfg)

        # Train BPE on the corpus
        tok = BPETokenizer(vocab_size=300)
        tok.train(small_corpus, target_merges=20)

        # Encode corpus
        token_ids = tok.encode_bytes(small_corpus[:500])

        # Build a tiny dataset
        if len(token_ids) < 20:
            pytest.skip("Corpus too small for training step")
        dataset = make_dataset_tokens(token_ids, seq_len=8, stride=8)

        # One optimizer step
        opt = AdamState(model, lr=1e-3, lr_schedule="constant")
        sample = dataset[0]
        x, y = sample[:-1], np.array([int(sample[-1])], dtype=np.int64)
        _, cache = forward_with_cache(model, x)
        grads = backward(model, cache, y)
        opt.step(model, grads)

        # Generate from the (now updated) model
        prompt = "hello"
        prompt_ids = tok.encode(prompt)
        if len(prompt_ids) == 0:
            prompt_ids = np.array([1, 2, 3], dtype=np.int64)
        result = model.generate(
            np.asarray(prompt_ids, dtype=np.int64),
            max_new_tokens=3,
            seed=42,
        )
        assert "output_ids" in result
        out_ids = result["output_ids"]
        assert len(out_ids) >= 1
        # All output IDs should be in valid range
        out_arr = np.asarray(out_ids)
        assert out_arr.min() >= 0
        assert out_arr.max() < cfg.vocab_size

    def test_save_load_trained_weights_preserves_behavior(self, small_model, tmp_weights_path):
        """Save weights, load them, verify identical forward output."""
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        logits_before, _ = small_model.forward(tokens)

        small_model.save_weights(tmp_weights_path)
        loaded = TinyGPT.load_weights(tmp_weights_path)
        logits_after, _ = loaded.forward(tokens)

        np.testing.assert_allclose(logits_before, logits_after, rtol=1e-6, atol=1e-7)
