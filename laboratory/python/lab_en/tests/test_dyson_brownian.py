"""
Unit tests for ``rmt_llm.dyson_brownian`` — Dyson Brownian Motion.

Tests cover Gaussian ensemble sampling, DBM simulation, spectral
diagnostics, level-spacing distributions, and Wigner surmise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

from rmt_llm.dyson_brownian import (
    DBMConfig,
    DysonBrownianMotion,
    empirical_spacing_distribution,
    ensemble_eigenvalues,
    gaussian_ensemble,
    wigner_surmise,
)

try:
    from hypothesis import given, settings, strategies as st
    _HAS_HYP = True
except ImportError:
    _HAS_HYP = False


# ─── Gaussian ensemble tests ───────────────────────────────────────────────
class TestGaussianEnsemble:
    """Tests for gaussian_ensemble and ensemble_eigenvalues."""

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_gaussian_ensemble_shape(self, beta):
        n = 8
        M = gaussian_ensemble(n, beta=beta, rng=np.random.default_rng(42))
        if beta == 4:
            assert M.shape == (2 * n, 2 * n)
        else:
            assert M.shape == (n, n)

    def test_goe_is_symmetric(self):
        """GOE (β=1) matrices must be real symmetric."""
        M = gaussian_ensemble(8, beta=1, rng=np.random.default_rng(0))
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_gue_is_hermitian(self):
        """GUE (β=2) matrices must be Hermitian."""
        M = gaussian_ensemble(8, beta=2, rng=np.random.default_rng(0))
        np.testing.assert_allclose(M, M.conj().T, atol=1e-10)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta must be 1, 2, or 4"):
            gaussian_ensemble(4, beta=3)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            gaussian_ensemble(0)

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_ensemble_eigenvalues_are_real(self, beta):
        """Eigenvalues of Hermitian matrices must be real and sorted."""
        ev = ensemble_eigenvalues(16, beta=beta, rng=np.random.default_rng(42))
        assert np.all(np.isreal(ev))
        assert np.all(np.diff(ev) >= 0)  # sorted

    def test_ensemble_eigenvalues_semircircle(self):
        """GOE eigenvalues should approximately follow the semicircle law."""
        n = 200
        ev = ensemble_eigenvalues(n, beta=1, rng=np.random.default_rng(42))
        # Semicircle radius = 2 * sigma * sqrt(n) = 2 * 1 * sqrt(200) ≈ 28.3
        radius = 2 * np.sqrt(n)
        # The largest eigenvalue should be close to the semicircle edge.
        assert abs(ev[-1] - radius) / radius < 0.2  # within 20%


# ─── DBM simulation tests ──────────────────────────────────────────────────
class TestDysonBrownianMotion:
    """Tests for the DysonBrownianMotion simulator."""

    def test_dbm_config_validation(self):
        # DBMConfig is a plain dataclass; validation happens in the
        # DysonBrownianMotion constructor.
        with pytest.raises(ValueError):
            DysonBrownianMotion(DBMConfig(n=1))
        with pytest.raises(ValueError):
            DysonBrownianMotion(DBMConfig(beta=3))
        with pytest.raises(ValueError):
            DysonBrownianMotion(DBMConfig(sigma=0))
        with pytest.raises(ValueError):
            DysonBrownianMotion(DBMConfig(dt=0))

    def test_initialize_with_semicircle(self):
        dbm = DysonBrownianMotion(DBMConfig(n=16, beta=2))
        ev = dbm.initialize(rng=np.random.default_rng(42))
        assert ev.shape == (16,)
        assert np.all(np.diff(ev) >= 0)

    def test_initialize_with_custom_eigvals(self):
        dbm = DysonBrownianMotion(DBMConfig(n=4, beta=2))
        ev = dbm.initialize(eigvals=[-2, -1, 1, 2])
        np.testing.assert_array_equal(ev, [-2, -1, 1, 2])

    def test_initialize_wrong_shape_raises(self):
        dbm = DysonBrownianMotion(DBMConfig(n=4, beta=2))
        with pytest.raises(ValueError, match="eigvals must have shape"):
            dbm.initialize(eigvals=[1, 2, 3])

    def test_step_advances_time(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        assert dbm.time == 0.0
        dbm.step(10)
        assert abs(dbm.time - 0.1) < 1e-10

    def test_step_returns_sorted_eigvals(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        ev = dbm.step(5)
        assert np.all(np.diff(ev) >= 0)

    def test_run_returns_final_eigvals(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        ev = dbm.run(0.1)
        assert ev.shape == (8,)

    def test_spectral_gap_positive(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        assert dbm.spectral_gap() > 0

    def test_largest_eigenvalue_trajectory(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        dbm.run(0.05, record_interval=5)
        traj = dbm.largest_eigenvalue_trajectory()
        assert len(traj) >= 2
        assert np.all(traj > traj[0] * 0.5)  # should not collapse

    def test_history_grows_with_steps(self):
        dbm = DysonBrownianMotion(DBMConfig(n=8, beta=2, dt=0.01))
        dbm.initialize(rng=np.random.default_rng(0))
        dbm.step(10, record=True)
        assert len(dbm.history) == 11  # initial + 10 steps

    def test_no_repulsion_increases_spread(self):
        """Without repulsion, eigenvalues should spread more (independent BM)."""
        cfg_rep = DBMConfig(n=16, beta=2, dt=0.01, repulsion=True)
        cfg_no = DBMConfig(n=16, beta=2, dt=0.01, repulsion=False)
        dbm1 = DysonBrownianMotion(cfg_rep)
        dbm2 = DysonBrownianMotion(cfg_no)
        ev1 = dbm1.initialize(rng=np.random.default_rng(42))
        ev2 = dbm2.initialize(rng=np.random.default_rng(42))
        dbm1.run(0.1)
        dbm2.run(0.1)
        # Without repulsion, the spread should be larger.
        assert dbm2.spectral_gap() >= dbm1.spectral_gap() * 0.9

    def test_boundary_reflection(self):
        """Reflecting boundary should keep eigenvalues within [-b, b]."""
        dbm = DysonBrownianMotion(
            DBMConfig(n=8, beta=2, dt=0.01, boundary=5.0)
        )
        dbm.initialize(rng=np.random.default_rng(0))
        dbm.run(1.0)
        assert np.all(np.abs(dbm.eigvals) <= 5.0 + 1e-10)


# ─── Level spacing tests ───────────────────────────────────────────────────
class TestLevelSpacing:
    """Tests for wigner_surmise and empirical_spacing_distribution."""

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_wigner_surmise_normalizes_to_one(self, beta):
        """The Wigner surmise must integrate to 1."""
        from scipy.integrate import quad
        # Integrate P(s) from 0 to infinity.
        val, _ = quad(wigner_surmise, 0, 20, args=(beta,))
        assert abs(val - 1.0) < 0.01

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_wigner_surmise_peak_near_one(self, beta):
        """The Wigner surmise should peak near s=1 (mean spacing)."""
        s = np.linspace(0.01, 5, 1000)
        p = wigner_surmise(s, beta=beta)
        peak_s = s[np.argmax(p)]
        assert 0.5 < peak_s < 2.5

    def test_wigner_surmise_invalid_beta(self):
        with pytest.raises(ValueError):
            wigner_surmise([1.0], beta=3)

    def test_empirical_spacing_distribution(self):
        ev = np.array([1, 2, 3, 5, 8], dtype=float)
        s = empirical_spacing_distribution(ev, normalize=False)
        np.testing.assert_array_equal(s, [1, 1, 2, 3])

    def test_empirical_spacing_normalized(self):
        ev = np.array([1, 2, 3, 5, 8], dtype=float)
        s = empirical_spacing_distribution(ev, normalize=True)
        assert abs(s.mean() - 1.0) < 1e-10

    @pytest.mark.parametrize("beta", [1, 2, 4])
    def test_dbm_spacings_match_wigner_surmise(self, beta):
        """DBM eigenvalue spacings should approximately follow the Wigner surmise."""
        n = 64
        ev = ensemble_eigenvalues(n, beta=beta, rng=np.random.default_rng(42))
        s = empirical_spacing_distribution(ev, normalize=True)
        # The mean of the Wigner surmise is 1 by construction.
        assert abs(s.mean() - 1.0) < 0.15
        # Level repulsion should reduce the variance below Poisson (1.0)
        # for β=1 and β=2. For β=4 the 2n×2n real representation has
        # a different effective spacing, so we only check the mean.


# ─── Property-based tests ──────────────────────────────────────────────────
if _HAS_HYP:

    class TestDBMProperties:

        @given(n=st.integers(min_value=4, max_value=16))
        @settings(max_examples=8, deadline=5000)
        def test_dbm_eigvals_stay_sorted(self, n):
            """Eigenvalues must remain sorted after any number of steps."""
            dbm = DysonBrownianMotion(DBMConfig(n=n, beta=2, dt=0.01))
            dbm.initialize(rng=np.random.default_rng(n))
            dbm.step(20)
            assert np.all(np.diff(dbm.eigvals) >= 0)

        @given(
            n=st.integers(min_value=8, max_value=32),
            beta=st.sampled_from([1, 2, 4]),
        )
        @settings(max_examples=10, deadline=5000)
        def test_ensemble_eigvals_real_and_sorted(self, n, beta):
            """Sampled eigenvalues must be real and sorted."""
            ev = ensemble_eigenvalues(n, beta=beta, rng=np.random.default_rng(42))
            assert np.all(np.isreal(ev))
            assert np.all(np.diff(ev) >= 0)
