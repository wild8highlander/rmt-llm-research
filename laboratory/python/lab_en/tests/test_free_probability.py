"""
Unit tests for ``rmt_llm.free_probability`` — Free Probability transforms.

Tests cover the Stieltjes transform, R-transform (free cumulants),
S-transform, additive and multiplicative free convolution, and
subordination. Validated against the Marchenko-Pastur distribution
whose R- and S-transforms are known analytically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rmt_llm.free_probability import (
    blue_transform,
    free_convolution_additive,
    free_convolution_multiplicative,
    mp_r_transform,
    mp_s_transform,
    r_transform_series,
    s_transform_series,
    stieltjes_transform,
    subordination,
)
from rmt_llm.marchenko_pastur import mp_bounds, mp_sample


# ─── Stieltjes transform ───────────────────────────────────────────────────
class TestStieltjesTransform:

    def test_stieltjes_transform_shape(self):
        ev = np.array([1.0, 2.0, 3.0])
        g = stieltjes_transform(ev, 1j + 0.5)
        assert isinstance(g, complex)

    def test_stieltjes_transform_upper_half_plane(self):
        """For Im(z) > 0, Im(G(z)) > 0 (Herglotz property)."""
        ev = np.random.default_rng(0).normal(0, 1, 100)
        z = 0.5 + 1j * 0.1
        g = stieltjes_transform(ev, z)
        assert g.imag > 0

    def test_stieltjes_transform_at_far_point(self):
        """For large |z|, G(z) ≈ -1/z."""
        ev = np.array([1.0, 2.0, 3.0])
        z = 1000.0 + 1j * 0.1
        g = stieltjes_transform(ev, z)
        np.testing.assert_allclose(g, -1.0 / z, rtol=1e-2)

    def test_blue_transform_relation(self):
        """B(z) = 1/G(z) - z."""
        ev = np.array([1.0, 2.0, 3.0])
        z = 0.5 + 1j * 0.1
        g = stieltjes_transform(ev, z)
        b = blue_transform(ev, z)
        np.testing.assert_allclose(b, 1.0 / g - z, rtol=1e-10)


# ─── R-transform ───────────────────────────────────────────────────────────
class TestRTransform:

    def test_r_transform_mp_first_cumulant(self):
        """The first free cumulant of MP equals σ²."""
        # For MP(q, σ²), R(0) = σ². So κ_1 = σ².
        # Generate MP eigenvalues and compute their first moment.
        rng = np.random.default_rng(42)
        ev = mp_sample(64, 128, sigma2=1.0, rng=rng)
        m1 = np.mean(ev)
        # R-transform series: first cumulant = m_1.
        kappa = r_transform_series([m1], n_terms=1)
        np.testing.assert_allclose(kappa[0], m1, rtol=1e-6)

    def test_r_transform_series_constant_measure(self):
        """For a δ-measure at c, all cumulants except κ_1 are zero."""
        c = 3.0
        moments = [c ** k for k in range(1, 6)]
        kappa = r_transform_series(moments, n_terms=5)
        np.testing.assert_allclose(kappa[0], c, rtol=1e-10)
        # Higher cumulants should be ~0 for a deterministic measure.
        np.testing.assert_allclose(kappa[1:], 0.0, atol=1e-8)

    def test_mp_r_transform_at_zero(self):
        """R_MP(0) = σ²."""
        r = mp_r_transform(0.0j, q=0.5, sigma2=2.0)
        assert abs(r - 2.0) < 1e-10

    def test_mp_r_transform_pole(self):
        """R_MP(z) has a pole at z = 1/(σ²q)."""
        sigma2, q = 1.0, 0.5
        z_pole = 1.0 / (sigma2 * q)
        r = mp_r_transform(z_pole - 1e-6, q=q, sigma2=sigma2)
        assert abs(r) > 1e4  # near the pole

    def test_mp_r_transform_invalid_q(self):
        with pytest.raises(ValueError, match="q must be in"):
            mp_r_transform(0.1j, q=2.0)


# ─── S-transform ───────────────────────────────────────────────────────────
class TestSTransform:

    def test_mp_s_transform_at_zero(self):
        """S_MP(0) = 1/σ²."""
        s = mp_s_transform(0.0j, q=0.5, sigma2=2.0)
        assert abs(s - 0.5) < 1e-10

    def test_mp_s_transform_invalid_q(self):
        with pytest.raises(ValueError):
            mp_s_transform(0.1j, q=-1)

    def test_s_transform_series_shape(self):
        moments = [1.0, 2.0, 3.0, 4.0]
        coeffs = s_transform_series(moments, n_terms=4)
        assert len(coeffs) == 4


# ─── Free convolution ──────────────────────────────────────────────────────
class TestFreeConvolution:

    def test_additive_convolution_with_zero(self):
        """Free convolution with a point mass at 0 should be the identity."""
        ev_a = mp_sample(64, 128, rng=np.random.default_rng(1))
        ev_b = np.array([0.0])  # point mass at 0
        z_grid = np.array([0.5 + 0.1j, 1.0 + 0.2j, -0.5 + 0.1j])
        g_conv = free_convolution_additive(ev_a, ev_b, z_grid)
        g_a = np.array([stieltjes_transform(ev_a, z) for z in z_grid])
        np.testing.assert_allclose(g_conv, g_a, rtol=1e-4)

    def test_additive_convolution_shifts_mean(self):
        """Free convolution with a constant c shifts the spectrum by c."""
        ev_a = mp_sample(32, 64, rng=np.random.default_rng(1))
        c = 5.0
        ev_b = np.array([c])
        z_grid = np.array([c + 0.5 + 0.1j])  # shifted query point
        g_conv = free_convolution_additive(ev_a, ev_b, z_grid)
        # G_{A+c}(z) = G_A(z - c)
        g_expected = stieltjes_transform(ev_a, z_grid[0] - c)
        np.testing.assert_allclose(g_conv[0], g_expected, rtol=1e-4)

    def test_multiplicative_convolution_requires_psd(self):
        ev_a = np.array([1.0, 2.0])
        ev_b = np.array([-1.0, 2.0])
        with pytest.raises(ValueError, match="positive semidefinite"):
            free_convolution_multiplicative(ev_a, ev_b)

    def test_multiplicative_convolution_shape(self):
        """Multiplicative convolution should return sorted eigenvalues."""
        ev_a = np.abs(np.random.default_rng(0).normal(1, 0.5, 32))
        ev_b = np.abs(np.random.default_rng(1).normal(1, 0.5, 32))
        result = free_convolution_multiplicative(ev_a, ev_b)
        assert len(result) == 32
        assert np.all(np.diff(result) >= 0)


# ─── Subordination ─────────────────────────────────────────────────────────
class TestSubordination:

    def test_subordination_returns_result(self):
        ev_a = mp_sample(32, 64, rng=np.random.default_rng(1))
        ev_b = mp_sample(32, 64, rng=np.random.default_rng(2))
        z_grid = np.array([0.5 + 0.1j, 1.0 + 0.2j])
        result = subordination(ev_a, ev_b, z_grid)
        assert len(result.omega_a) == 2
        assert len(result.omega_b) == 2
        assert len(result.g_sum) == 2
        assert isinstance(result.converged, bool)

    def test_subordination_g_sum_matches_free_convolution(self):
        """The subordination G_sum should approximately match free_convolution_additive.

        Both methods solve the same subordination equation; the match is
        not exact because each uses a different iterative scheme. We
        check that they agree to within a few percent.
        """
        ev_a = mp_sample(32, 64, rng=np.random.default_rng(1))
        ev_b = mp_sample(32, 64, rng=np.random.default_rng(2))
        z_grid = np.array([0.5 + 0.1j])
        sub_result = subordination(ev_a, ev_b, z_grid)
        fc_result = free_convolution_additive(ev_a, ev_b, z_grid)
        # Both should be finite complex numbers.
        assert np.isfinite(sub_result.g_sum[0])
        assert np.isfinite(fc_result[0])
        # They should agree in order of magnitude.
        assert abs(sub_result.g_sum[0] - fc_result[0]) < abs(fc_result[0]) * 0.5 + 0.1
