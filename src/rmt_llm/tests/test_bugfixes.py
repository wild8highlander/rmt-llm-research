"""Regression tests for the bug-fix pass.

Covers two core-library bugs that were fixed (see CHANGELOG [Unreleased]):
  1. caputo_fractional.caputo_quadratic_acceleration() raised ValueError on
     EVERY call (it evaluated caputo_mean_collapse_time(mu, beta=1.0) outside
     the valid open interval (0, 1)). Fixed via the analytic beta -> 1 limit.
  2. free_probability.free_convolution_multiplicative() created an unseeded
     np.random.Generator internally, so results were not reproducible.
     Fixed via an optional `rng` parameter (backwards compatible).
"""

from __future__ import annotations

import numpy as np
import pytest

import caputo_fractional as cap
import free_probability as fp


class TestCaputoQuadraticAcceleration:
    def test_returns_finite_for_valid_mu(self):
        for mu in (0.05, 0.1, 0.3, 1.0, 2.0):
            val = cap.caputo_quadratic_acceleration(mu)
            assert np.isfinite(val)
            assert val > 0

    def test_matches_analytic_limit(self):
        """T(0.5)/T(1) = mu^(-2) / mu^(-1) = 1/mu."""
        for mu in (0.07, 0.25, 1.5):
            expected = 1.0 / mu
            got = cap.caputo_quadratic_acceleration(mu)
            assert got == pytest.approx(expected, rel=1e-12)

    def test_no_value_error_for_valid_mu(self):
        # Before the fix this raised ValueError("beta must be in (0, 1)")
        cap.caputo_quadratic_acceleration(0.1)  # must not raise

    def test_non_positive_mu_raises(self):
        with pytest.raises(ValueError):
            cap.caputo_quadratic_acceleration(0.0)
        with pytest.raises(ValueError):
            cap.caputo_quadratic_acceleration(-1.0)


class TestFreeConvolutionSeededRng:
    def test_seeded_rng_is_reproducible(self):
        a = np.array([0.5, 1.0, 2.0, 1.5])
        b = np.array([1.0, 0.8, 1.2, 0.9])
        r1 = fp.free_convolution_multiplicative(a, b, rng=np.random.default_rng(7))
        r2 = fp.free_convolution_multiplicative(a, b, rng=np.random.default_rng(7))
        np.testing.assert_allclose(r1, r2)

    def test_different_seeds_differ(self):
        a = np.array([0.5, 1.0, 2.0, 1.5])
        b = np.array([1.0, 0.8, 1.2, 0.9])
        r1 = fp.free_convolution_multiplicative(a, b, rng=np.random.default_rng(1))
        r2 = fp.free_convolution_multiplicative(a, b, rng=np.random.default_rng(2))
        assert not np.allclose(r1, r2)

    def test_default_call_still_works(self):
        a = np.array([0.5, 1.0, 2.0])
        b = np.array([1.0, 0.8, 1.2])
        out = fp.free_convolution_multiplicative(a, b)
        assert len(out) > 0
        assert np.all(np.isfinite(out))

    def test_negative_inputs_raise(self):
        with pytest.raises(ValueError):
            fp.free_convolution_multiplicative([1.0, -0.5], [1.0, 2.0])
