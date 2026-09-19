"""
Unit tests for ``rmt_llm.circular_ensembles`` — COE, CUE, CSE.

Tests cover Haar sampling, circular ensemble generation, eigenvalue
phases, level spacing, form factor, number variance, and the
connection to attention matrices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rmt_llm.circular_ensembles import (
    attention_phase_spectrum,
    circular_eigenvalues,
    circular_ensemble,
    form_factor,
    haar_orthogonal,
    haar_unitary,
    nearest_neighbor_spacing,
    number_variance,
    theoretical_form_factor,
    wigner_surmise_circular,
)


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HAS_HYP = True
except ImportError:
    _HAS_HYP = False


# ─── Haar matrix tests ─────────────────────────────────────────────────────
class TestHaarMatrices:
    def test_haar_unitary_is_unitary(self):
        U = haar_unitary(8, rng=np.random.default_rng(42))
        np.testing.assert_allclose(U @ U.conj().T, np.eye(8), atol=1e-10)

    def test_haar_orthogonal_is_orthogonal(self):
        Q = haar_orthogonal(8, rng=np.random.default_rng(42))
        np.testing.assert_allclose(Q @ Q.T, np.eye(8), atol=1e-10)

    def test_haar_unitary_invalid_n(self):
        with pytest.raises(ValueError):
            haar_unitary(0)

    def test_haar_orthogonal_invalid_n(self):
        with pytest.raises(ValueError):
            haar_orthogonal(0)


# ─── Circular ensemble tests ───────────────────────────────────────────────
class TestCircularEnsembles:
    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_circular_ensemble_shape(self, beta):
        n = 8
        M = circular_ensemble(n, beta=beta, rng=np.random.default_rng(42))
        if beta == 4:
            assert M.shape == (2 * n, 2 * n)
        else:
            assert M.shape == (n, n)

    def test_coe_is_symmetric(self):
        """COE (β=1) matrices should be symmetric unitary."""
        M = circular_ensemble(8, beta=1, rng=np.random.default_rng(0))
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_cue_is_unitary(self):
        """CUE (β=2) is the Haar measure on U(N)."""
        M = circular_ensemble(8, beta=2, rng=np.random.default_rng(0))
        np.testing.assert_allclose(M @ M.conj().T, np.eye(8), atol=1e-10)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta must be 1, 2, or 4"):
            circular_ensemble(4, beta=3)

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_eigenvalues_on_unit_circle(self, beta):
        """Eigenvalues of circular ensembles lie on |λ| = 1."""
        phases = circular_eigenvalues(16, beta=beta, rng=np.random.default_rng(42))
        # Convert back to complex eigenvalues.
        eigs = np.exp(1j * phases)
        np.testing.assert_allclose(np.abs(eigs), 1.0, atol=1e-8)

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_phases_in_range(self, beta):
        phases = circular_eigenvalues(16, beta=beta, rng=np.random.default_rng(42))
        assert np.all(phases >= 0)
        assert np.all(phases < 2 * np.pi)
        assert np.all(np.diff(phases) >= 0)  # sorted


# ─── Level spacing tests ───────────────────────────────────────────────────
class TestLevelSpacing:
    def test_nearest_neighbor_spacing_circular(self):
        """Circular spacing includes the wrap-around gap."""
        phases = np.array([0.1, 0.5, 1.0, 2.0])
        s = nearest_neighbor_spacing(phases, normalize=False)
        # Spacings: 0.4, 0.5, 1.0, (2π - 2.0 + 0.1) = 4.38...
        assert len(s) == 4
        assert abs(s[0] - 0.4) < 1e-10
        assert abs(s[1] - 0.5) < 1e-10

    def test_nearest_neighbor_spacing_normalized(self):
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        s = nearest_neighbor_spacing(phases, normalize=True)
        assert abs(s.mean() - 1.0) < 1e-10

    def test_empty_phases(self):
        s = nearest_neighbor_spacing([])
        assert len(s) == 0

    def test_single_phase(self):
        s = nearest_neighbor_spacing([1.0])
        assert len(s) == 0


# ─── Form factor tests ─────────────────────────────────────────────────────
class TestFormFactor:
    def test_form_factor_shape(self):
        phases = circular_eigenvalues(16, beta=2, rng=np.random.default_rng(0))
        tau, K = form_factor(phases, max_k=32)
        assert len(tau) == 32
        assert len(K) == 32

    def test_theoretical_form_factor_at_zero(self):
        """K(0) = 0 (no correlation at τ=0)."""
        K = theoretical_form_factor([0.0], beta=2)
        assert abs(K[0]) < 1e-10

    def test_theoretical_form_factor_saturates(self):
        """K(τ) → 1 for τ ≥ 1."""
        K = theoretical_form_factor([1.5, 2.0, 5.0], beta=2)
        np.testing.assert_allclose(K, [1.0, 1.0, 1.0])

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_form_factor_beta_dependence(self, beta):
        """Different β should give different form factors."""
        tau = np.array([0.3, 0.5, 0.8])
        K1 = theoretical_form_factor(tau, beta=1)
        K2 = theoretical_form_factor(tau, beta=2)
        K4 = theoretical_form_factor(tau, beta=4)
        # They should differ (higher β → stronger repulsion → lower K for small τ).
        assert not np.allclose(K1, K2)


# ─── Number variance tests ─────────────────────────────────────────────────
class TestNumberVariance:
    def test_number_variance_shape(self):
        phases = circular_eigenvalues(32, beta=2, rng=np.random.default_rng(0))
        L, var = number_variance(phases, max_L=3.0, n_points=20)
        assert len(L) == 20
        assert len(var) == 20
        assert np.all(var >= 0)

    def test_number_variance_too_few_phases(self):
        L, var = number_variance([0.1, 0.2], max_L=2.0, n_points=10)
        assert len(L) == 0


# ─── Wigner surmise (circular) ─────────────────────────────────────────────
class TestWignerSurmiseCircular:
    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_normalization(self, beta):
        """The Wigner surmise should integrate to approximately 1."""
        try:
            from scipy.integrate import quad

            val, _ = quad(wigner_surmise_circular, 0, 20, args=(beta,))
            assert abs(val - 1.0) < 0.01
        except ImportError:
            pytest.skip("scipy not installed")

    def test_invalid_beta(self):
        with pytest.raises(ValueError):
            wigner_surmise_circular([1.0], beta=3)

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_peak_near_one(self, beta):
        s = np.linspace(0.01, 5, 1000)
        p = wigner_surmise_circular(s, beta=beta)
        peak_s = s[np.argmax(p)]
        assert 0.3 < peak_s < 2.5


# ─── Attention phase spectrum ──────────────────────────────────────────────
class TestAttentionPhaseSpectrum:
    def test_phase_spectrum_shape(self):
        """Phase spectrum should return sorted phases in [0, 2π)."""
        rng = np.random.default_rng(0)
        A = rng.dirichlet(np.ones(8), size=8)  # row-stochastic
        phases = attention_phase_spectrum(A)
        assert np.all(phases >= 0)
        assert np.all(phases < 2 * np.pi)
        assert np.all(np.diff(phases) >= 0)

    def test_phase_spectrum_drops_dominant(self):
        """The dominant eigenvalue (closest to 1) should be dropped."""
        rng = np.random.default_rng(0)
        A = rng.dirichlet(np.ones(8), size=8)
        n_phases = len(attention_phase_spectrum(A))
        # For an 8x8 matrix, we drop 1 eigenvalue, leaving 7.
        assert n_phases == 7

    def test_phase_spectrum_invalid_shape(self):
        with pytest.raises(ValueError, match="attn_matrix must be square"):
            attention_phase_spectrum(np.zeros((3, 4)))


# ─── Property-based tests ──────────────────────────────────────────────────
if _HAS_HYP:

    class TestCircularProperties:
        @given(n=st.integers(min_value=4, max_value=16))
        @settings(max_examples=8, deadline=3000)
        def test_cue_eigenvalues_on_unit_circle(self, n):
            """CUE eigenvalues must lie exactly on the unit circle."""
            phases = circular_eigenvalues(n, beta=2, rng=np.random.default_rng(42))
            eigs = np.exp(1j * phases)
            np.testing.assert_allclose(np.abs(eigs), 1.0, atol=1e-8)

        @given(n=st.integers(min_value=4, max_value=16))
        @settings(max_examples=8, deadline=3000)
        def test_haar_unitary_property(self, n):
            """Haar unitary must satisfy U U† = I."""
            U = haar_unitary(n, rng=np.random.default_rng(42))
            np.testing.assert_allclose(U @ U.conj().T, np.eye(n), atol=1e-10)
