"""
Unit tests for ``probe_real_gpt2.py`` — GPT-2 activation probe.

Uses the :class:`SyntheticProbe` fallback so tests run without the
``transformers`` / ``torch`` dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from probe_real_gpt2 import (
    GPT2Probe,
    LayerSpectrum,
    PromptResult,
    SyntheticProbe,
    _mp_bounds_general,
)


# ─── SyntheticProbe tests ──────────────────────────────────────────────────
class TestSyntheticProbe:

    def test_construct(self):
        probe = SyntheticProbe(n_layers=4, hidden_dim=32, seed=42)
        assert probe.n_layers == 4
        assert probe.hidden_dim == 32

    def test_extract_hidden_states_shape(self):
        probe = SyntheticProbe(n_layers=4, hidden_dim=32, seed=42)
        hs, n_tokens = probe.extract_hidden_states("hello world test")
        assert hs.shape == (4, n_tokens, 32)
        assert n_tokens >= 3

    def test_analyze_prompt_returns_result(self):
        probe = SyntheticProbe(n_layers=4, hidden_dim=32, seed=42)
        result = probe.analyze_prompt("The quick brown fox")
        assert isinstance(result, PromptResult)
        assert result.prompt == "The quick brown fox"
        assert len(result.layers) == 4
        assert all(isinstance(l, LayerSpectrum) for l in result.layers)

    def test_analyze_prompt_bbp_transition(self):
        """With growing spike, BBP transition should occur at some layer."""
        probe = SyntheticProbe(n_layers=10, hidden_dim=64, seed=42)
        result = probe.analyze_prompt("The capital of France is Paris")
        # The transition should occur at a non-None layer.
        assert result.bbp_transition_layer is not None
        assert 0 <= result.bbp_transition_layer < 10

    def test_analyze_prompts_list(self):
        probe = SyntheticProbe(n_layers=4, hidden_dim=32, seed=42)
        results = probe.analyze_prompts(["hello", "world", "test"])
        assert len(results) == 3
        assert all(isinstance(r, PromptResult) for r in results)


# ─── LayerSpectrum validation ──────────────────────────────────────────────
class TestLayerSpectrum:

    def test_spectrum_fields(self):
        probe = SyntheticProbe(n_layers=2, hidden_dim=32, seed=42)
        result = probe.analyze_prompt("test prompt here")
        spec = result.layers[0]
        assert hasattr(spec, "layer")
        assert hasattr(spec, "lambda_max")
        assert hasattr(spec, "mp_upper")
        assert hasattr(spec, "signal_detected")
        assert hasattr(spec, "n_outlier_eigvals")
        assert hasattr(spec, "q")
        assert hasattr(spec, "bbp_theta")

    def test_spectrum_finite(self):
        probe = SyntheticProbe(n_layers=2, hidden_dim=32, seed=42)
        result = probe.analyze_prompt("test prompt here")
        for spec in result.layers:
            assert np.isfinite(spec.lambda_max)
            assert np.isfinite(spec.mp_upper)
            assert spec.mp_upper > 0


# ─── MP bounds (general) ───────────────────────────────────────────────────
class TestMPBoundsGeneral:

    def test_q_le_one(self):
        """For q ≤ 1, should match the standard MP bounds."""
        lo, hi = _mp_bounds_general(0.5, sigma2=1.0)
        sq = np.sqrt(0.5)
        np.testing.assert_allclose(lo, (1 - sq) ** 2, rtol=1e-10)
        np.testing.assert_allclose(hi, (1 + sq) ** 2, rtol=1e-10)

    def test_q_gt_one(self):
        """For q > 1, should still return valid bounds."""
        lo, hi = _mp_bounds_general(2.0, sigma2=1.0)
        sq = np.sqrt(2.0)
        np.testing.assert_allclose(lo, (1 - sq) ** 2, rtol=1e-10)
        np.testing.assert_allclose(hi, (1 + sq) ** 2, rtol=1e-10)

    def test_invalid_q(self):
        with pytest.raises(ValueError, match="q must be positive"):
            _mp_bounds_general(0.0)
        with pytest.raises(ValueError, match="q must be positive"):
            _mp_bounds_general(-1.0)


# ─── Reporting ─────────────────────────────────────────────────────────────
class TestReporting:

    def test_report_contains_key_info(self):
        probe = SyntheticProbe(n_layers=3, hidden_dim=32, seed=42)
        result = probe.analyze_prompt("test")
        report = GPT2Probe.report(result)
        assert "Prompt:" in report
        assert "BBP transition" in report
        assert "Layer" in report

    def test_to_json_serializes(self):
        probe = SyntheticProbe(n_layers=2, hidden_dim=32, seed=42)
        results = probe.analyze_prompts(["hello", "world"])
        json_str = GPT2Probe.to_json(results)
        import json
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert "prompt" in parsed[0]
        assert "layers" in parsed[0]

    def test_save_results(self, tmp_path):
        probe = SyntheticProbe(n_layers=2, hidden_dim=32, seed=42)
        results = probe.analyze_prompts(["hello"])
        path = str(tmp_path / "results.json")
        GPT2Probe.save_results(results, path)
        import os
        assert os.path.exists(path)
