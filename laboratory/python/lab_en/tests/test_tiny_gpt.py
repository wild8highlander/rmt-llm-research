"""
Unit tests for `tiny_gpt.py` — model config, forward pass, generation,
spectral analysis, weight save/load, and low-level math helpers.
"""

from __future__ import annotations

import os

import numpy as np
from tiny_gpt import (
    TinyGPT,
    TinyGPTConfig,
    TinyLayer,
    _gelu,
    _gelu_grad,
    _init_layer,
    _softmax,
    decode,
    encode,
    layernorm_backward,
    layernorm_forward,
)


# ─── Config ──────────────────────────────────────────────────────────────────
class TestTinyGPTConfig:
    """Tests for the TinyGPTConfig dataclass."""

    def test_default_config_has_expected_dims(self):
        cfg = TinyGPTConfig()
        assert cfg.vocab_size == 512
        assert cfg.hidden_dim == 128
        assert cfg.n_layers == 12
        assert cfg.n_heads == 4
        assert cfg.max_seq_len == 256
        assert cfg.mlp_ratio == 4
        assert cfg.use_layernorm is True
        assert cfg.use_mlp is True
        assert cfg.activation == "gelu"

    def test_head_dim_property(self):
        cfg = TinyGPTConfig(hidden_dim=128, n_heads=4)
        assert cfg.head_dim == 32

    def test_mlp_dim_property(self):
        cfg = TinyGPTConfig(hidden_dim=128, mlp_ratio=4)
        assert cfg.mlp_dim == 512

    def test_head_dim_must_divide_hidden(self):
        # 128 / 3 is not integer — but head_dim property just returns int division
        cfg = TinyGPTConfig(hidden_dim=128, n_heads=3)
        assert cfg.head_dim == 42  # floor(128/3) = 42
        # NOTE: forward() would fail at runtime with non-divisible dims; config alone is permissive

    def test_params_count_matches_expected(self):
        """v2 architecture: 12 layers × hidden=128 × vocab=512 → ~2.5M params."""
        cfg = TinyGPTConfig()
        n = cfg.params_count
        # Documented in CHANGELOG as 2,543,104
        assert 2_400_000 < n < 2_700_000, f"params_count={n} outside expected range"

    def test_params_count_scales_with_layers(self):
        cfg_2 = TinyGPTConfig(n_layers=2)
        cfg_4 = TinyGPTConfig(n_layers=4)
        # Doubling layers should roughly double params (plus fixed embeddings)
        assert cfg_4.params_count > cfg_2.params_count
        # Per-layer params ~ 4*H^2 + 4*H + 4*H (LN) + 2*H*mlp + mlp + H ≈ 4*H^2 + 2*H*mlp
        per_layer_diff = (cfg_4.params_count - cfg_2.params_count) / 2
        expected_per_layer = 4 * 128 * 128 + 2 * 128 * 512  # rough
        assert abs(per_layer_diff - expected_per_layer) < 0.1 * expected_per_layer


# ─── Model construction ─────────────────────────────────────────────────────
class TestTinyGPTConstruction:
    """Tests for TinyGPT.__init__."""

    def test_default_model_constructs(self):
        model = TinyGPT()
        assert model.config.vocab_size == 512
        assert len(model.layers) == 12
        assert model.token_emb.shape == (512, 128)
        assert model.pos_emb.shape == (256, 128)
        assert model.lm_head.shape == (128, 512)

    def test_custom_config_constructs(self, small_config):
        model = TinyGPT(small_config)
        assert len(model.layers) == 2
        assert model.token_emb.shape == (64, 32)

    def test_each_layer_has_all_weights(self, small_model):
        for layer in small_model.layers:
            assert isinstance(layer, TinyLayer)
            assert layer.W_q.shape == (32, 32)
            assert layer.W_k.shape == (32, 32)
            assert layer.W_v.shape == (32, 32)
            assert layer.W_o.shape == (32, 32)
            assert layer.b_q.shape == (32,)
            assert layer.W_fc1.shape == (32, 128)  # H -> 4H
            assert layer.W_fc2.shape == (128, 32)  # 4H -> H

    def test_weights_are_finite(self, small_model):
        for layer in small_model.layers:
            for arr_name in ["W_q", "W_k", "W_v", "W_o", "b_q", "W_fc1", "W_fc2"]:
                arr = getattr(layer, arr_name)
                assert np.all(np.isfinite(arr)), f"{arr_name} has NaN/Inf"

    def test_seed_reproducibility(self):
        """Same seed → same initial weights."""
        m1 = TinyGPT(TinyGPTConfig(seed=42, n_layers=2, hidden_dim=32, vocab_size=64))
        m2 = TinyGPT(TinyGPTConfig(seed=42, n_layers=2, hidden_dim=32, vocab_size=64))
        np.testing.assert_array_equal(m1.token_emb, m2.token_emb)
        np.testing.assert_array_equal(m1.layers[0].W_q, m2.layers[0].W_q)

    def test_different_seed_gives_different_weights(self):
        m1 = TinyGPT(TinyGPTConfig(seed=1, n_layers=2, hidden_dim=32, vocab_size=64))
        m2 = TinyGPT(TinyGPTConfig(seed=2, n_layers=2, hidden_dim=32, vocab_size=64))
        assert not np.array_equal(m1.token_emb, m2.token_emb)


# ─── Forward pass ───────────────────────────────────────────────────────────
class TestForward:
    """Tests for TinyGPT.forward."""

    def test_forward_returns_logits_and_hidden(self, small_model):
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        logits, hidden = small_model.forward(tokens)
        assert logits.shape == (5, 64)  # (T, V)
        assert len(hidden) == 2  # one per layer
        assert all(h.shape == (5, 32) for h in hidden)  # (T, H)

    def test_forward_logits_are_finite(self, small_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits, _ = small_model.forward(tokens)
        assert np.all(np.isfinite(logits))

    def test_forward_works_with_single_token(self, small_model):
        tokens = np.array([5], dtype=np.int64)
        logits, hidden = small_model.forward(tokens)
        assert logits.shape == (1, 64)
        assert len(hidden) == 2

    def test_forward_works_at_max_seq_len(self, small_model):
        cfg = small_model.config
        tokens = np.arange(cfg.max_seq_len, dtype=np.int64) % cfg.vocab_size
        logits, _ = small_model.forward(tokens)
        assert logits.shape == (cfg.max_seq_len, cfg.vocab_size)

    def test_forward_hidden_states_captured(self, small_model):
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        _, hidden = small_model.forward(tokens)
        # _hidden_states should match the returned list
        assert small_model._hidden_states is not None
        assert len(small_model._hidden_states) == len(hidden)

    def test_forward_is_deterministic(self, small_model):
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits1, _ = small_model.forward(tokens)
        logits2, _ = small_model.forward(tokens)
        np.testing.assert_array_equal(logits1, logits2)

    def test_forward_different_inputs_different_outputs(self, small_model):
        tokens1 = np.array([1, 2, 3], dtype=np.int64)
        tokens2 = np.array([4, 5, 6], dtype=np.int64)
        logits1, _ = small_model.forward(tokens1)
        logits2, _ = small_model.forward(tokens2)
        assert not np.allclose(logits1, logits2)


# ─── Generation ─────────────────────────────────────────────────────────────
class TestGenerate:
    """Tests for TinyGPT.generate."""

    def test_generate_returns_dict_with_expected_keys(self, small_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        result = small_model.generate(prompt, max_new_tokens=5, seed=42)
        assert isinstance(result, dict)
        assert "output_ids" in result

    def test_generate_produces_expected_length(self, small_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        result = small_model.generate(prompt, max_new_tokens=5, seed=42)
        out = result["output_ids"]
        # Output includes prompt + new tokens (depending on API)
        assert len(out) >= 5

    def test_generate_is_deterministic_with_seed(self, small_model):
        prompt = np.array([1, 2, 3], dtype=np.int64)
        r1 = small_model.generate(prompt, max_new_tokens=5, seed=42)
        r2 = small_model.generate(prompt, max_new_tokens=5, seed=42)
        np.testing.assert_array_equal(r1["output_ids"], r2["output_ids"])

    def test_generate_zero_temperature_is_greedy(self, small_model):
        """At T=0, the newly-generated tokens should match argmax continuation."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        n_new = 3
        result = small_model.generate(prompt, max_new_tokens=n_new, temperature=0.0, seed=42)
        # Manually compute argmax continuation
        manual_new = []
        ctx = list(prompt)
        for _ in range(n_new):
            logits, _ = small_model.forward(np.array(ctx, dtype=np.int64))
            next_id = int(np.argmax(logits[-1]))
            manual_new.append(next_id)
            ctx.append(next_id)
        # output_ids may contain only new tokens, or prompt+new — compare new-only
        out = result["output_ids"]
        new_part = out[-n_new:] if len(out) > n_new else out
        np.testing.assert_array_equal(new_part, manual_new)

    def test_generate_high_temperature_is_stochastic(self, small_model):
        """At T=1.0 with different seeds, output should differ (with high prob)."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        r1 = small_model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=1)
        r2 = small_model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=2)
        assert not np.array_equal(r1["output_ids"], r2["output_ids"])


# ─── Spectral analysis ──────────────────────────────────────────────────────
class TestSpectralAnalysis:
    """Tests for TinyGPT.spectral_analysis."""

    def test_spectral_analysis_returns_per_layer_results(self, small_model):
        tokens = np.arange(1, 33, dtype=np.int64)  # 32 tokens
        _, hidden = small_model.forward(tokens)
        result = small_model.spectral_analysis(hidden)
        assert isinstance(result, dict)
        # Should have a list of per-layer results
        assert "layers" in result or isinstance(result, list) or len(result) > 0

    def test_spectral_analysis_handles_short_sequence(self, small_model):
        """Sequences shorter than 2 tokens should not crash."""
        tokens = np.array([1], dtype=np.int64)
        _, hidden = small_model.forward(tokens)
        # Should not raise even though cov is degenerate
        result = small_model.spectral_analysis(hidden)
        assert result is not None


# ─── Weight save / load ─────────────────────────────────────────────────────
class TestWeightsIO:
    """Tests for save_weights / load_weights round-trip."""

    def test_save_load_roundtrip(self, small_model, tmp_weights_path):
        # Forward to populate caches
        tokens = np.array([1, 2, 3], dtype=np.int64)
        logits_before, _ = small_model.forward(tokens)

        small_model.save_weights(tmp_weights_path)
        assert os.path.exists(tmp_weights_path)

        loaded = TinyGPT.load_weights(tmp_weights_path)
        logits_after, _ = loaded.forward(tokens)

        np.testing.assert_allclose(logits_before, logits_after, rtol=1e-6, atol=1e-7)

    def test_save_creates_npz_file(self, small_model, tmp_weights_path):
        small_model.save_weights(tmp_weights_path)
        # .npz files are ZIP archives
        assert tmp_weights_path.endswith(".npz")
        loaded_data = np.load(tmp_weights_path)
        assert "token_emb" in loaded_data.files
        assert "lm_head" in loaded_data.files

    def test_loaded_model_has_same_config(self, small_model, tmp_weights_path):
        small_model.save_weights(tmp_weights_path)
        loaded = TinyGPT.load_weights(tmp_weights_path)
        assert loaded.config.vocab_size == small_model.config.vocab_size
        assert loaded.config.hidden_dim == small_model.config.hidden_dim
        assert loaded.config.n_layers == small_model.config.n_layers


# ─── Low-level math helpers ─────────────────────────────────────────────────
class TestMathHelpers:
    """Tests for _softmax, _gelu, _gelu_grad, layernorm_forward/backward."""

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        s = _softmax(x, axis=-1)
        np.testing.assert_allclose(s.sum(axis=-1), 1.0, rtol=1e-7)

    def test_softmax_preserves_shape(self):
        x = np.random.randn(5, 7)
        s = _softmax(x)
        assert s.shape == x.shape

    def test_softmax_is_invariant_to_shift(self):
        x = np.array([1.0, 2.0, 3.0])
        s1 = _softmax(x)
        s2 = _softmax(x + 100.0)
        np.testing.assert_allclose(s1, s2, rtol=1e-7)

    def test_softmax_handles_large_values(self):
        """softmax([1000, 1000, 1000]) should not overflow to NaN."""
        x = np.array([1000.0, 1000.0, 1000.0])
        s = _softmax(x)
        assert np.all(np.isfinite(s))
        np.testing.assert_allclose(s, 1.0 / 3.0, rtol=1e-7)

    def test_gelu_zero_at_zero(self):
        """GELU(0) ≈ 0."""
        x = np.array([0.0])
        result = _gelu(x)
        assert abs(result[0]) < 1e-7

    def test_gelu_positive_for_positive_input(self):
        x = np.array([1.0, 2.0, 5.0])
        result = _gelu(x)
        assert np.all(result > 0)

    def test_gelu_grad_finite(self):
        x = np.linspace(-5, 5, 21)
        grad = _gelu_grad(x)
        assert np.all(np.isfinite(grad))

    def test_gelu_grad_negative_extreme_is_zero(self):
        """For tanh-approx GELU, grad → 0 as x → -∞ (function saturates to 0)."""
        grad_neg = _gelu_grad(np.array([-100.0]))[0]
        assert abs(grad_neg) < 1e-3

    def test_gelu_grad_positive_extreme_is_one(self):
        """For tanh-approx GELU, grad → 1 as x → +∞ (function becomes identity)."""
        grad_pos = _gelu_grad(np.array([100.0]))[0]
        assert abs(grad_pos - 1.0) < 1e-3

    def test_layernorm_zero_mean_unit_var(self):
        """Output of layernorm should be approximately zero-mean, unit-variance."""
        x = np.random.randn(10, 32) * 5 + 3
        gamma = np.ones(32)
        beta = np.zeros(32)
        out, cache_mean, cache_var = layernorm_forward(x, gamma, beta)
        # Per-sample (per-row) mean should be ~0
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        # Per-sample variance should be ~1
        np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-6)

    def test_layernorm_gamma_scales_output(self):
        x = np.random.randn(5, 8)
        gamma = np.full(8, 2.0)
        beta = np.zeros(8)
        out, _, _ = layernorm_forward(x, gamma, beta)
        # eps=1e-5 in layernorm introduces tiny bias — use atol=1e-3
        np.testing.assert_allclose(out.std(axis=-1), 2.0, atol=1e-3)

    def test_layernorm_beta_shifts_output(self):
        x = np.random.randn(5, 8)
        gamma = np.ones(8)
        beta = np.full(8, 1.5)
        out, _, _ = layernorm_forward(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 1.5, atol=1e-6)

    def test_layernorm_backward_shape(self):
        x = np.random.randn(5, 8)
        gamma = np.ones(8)
        beta = np.zeros(8)
        out, mean, var = layernorm_forward(x, gamma, beta)
        dout = np.random.randn(*out.shape)
        dx, dgamma, dbeta = layernorm_backward(dout, x, gamma, mean, var)
        assert dx.shape == x.shape
        assert dgamma.shape == gamma.shape
        assert dbeta.shape == beta.shape

    def test_init_layer_shapes(self):
        rng = np.random.default_rng(0)
        layer = _init_layer(rng, H=32, mlp_dim=128, use_layernorm=True, use_mlp=True)
        assert layer.W_q.shape == (32, 32)
        assert layer.W_fc1.shape == (32, 128)


# ─── Byte-level encoding ────────────────────────────────────────────────────
class TestByteEncoding:
    """Tests for encode() / decode() byte-level helpers."""

    def test_encode_returns_int_array(self):
        ids = encode("hello")
        assert isinstance(ids, (list, np.ndarray))
        assert len(ids) == 5  # 5 bytes in "hello"

    def test_encode_decode_roundtrip(self):
        text = "Hello, World!\n"
        ids = encode(text)
        assert decode(ids) == text

    def test_encode_handles_unicode(self):
        """Non-ASCII chars should encode to multiple bytes (UTF-8)."""
        text = "Привет, мир!"  # Russian
        ids = encode(text)
        # Each Cyrillic char is 2 bytes in UTF-8
        assert len(ids) >= len(text)
        assert decode(ids) == text

    def test_encode_empty_string(self):
        assert len(encode("")) == 0
        assert decode([]) == ""
