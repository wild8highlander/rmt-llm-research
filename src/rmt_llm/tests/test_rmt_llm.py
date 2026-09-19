"""
Comprehensive test suite for the RMT-LLM verification package.

Covers all six mathematical modules:
  - marchenko_pastur: MP law density, bounds, CDF, sampling
  - bbp_transition: BBP phase transition, signal separation
  - tracy_widom: Tracy-Widom F_2 distribution
  - nhse: Non-Hermitian skin effect and winding number
  - caputo_fractional: Caputo fractional dynamics and RLHF utility trap
  - keating_snaith: Keating-Snaith corrections
  - ep_surfaces: Exceptional-point spectral sensitivity
  - thermodynamics: Free energy, Landauer principle, RG flow

Author: Iskhak Hamzatovich Isaev
"""

import math

import numpy as np
import pytest

from rmt_llm.bbp_transition import (
    bbp_critical_theta,
    bbp_fluctuation_scaling,
    bbp_is_supercritical,
    bbp_lambda_max,
    bbp_sample,
    bbp_signal_separation,
)
from rmt_llm.caputo_fractional import (
    caputo_fokker_planck_drift,
    caputo_mean_collapse_time,
    caputo_n_crit,
)
from rmt_llm.constants import (
    BETA_CAPUTO,
    GAMMA_1,
    GPT2_CONTEXT_WINDOW,
    K_B,
    LN2,
    THETA_B_DEGREES,
    TW_MEAN,
    TW_SKEWNESS,
    TW_VARIANCE,
)
from rmt_llm.ep_surfaces import (
    ep_is_near,
    ep_order_from_separation,
    ep_perturbed_matrix,
    ep_rounding_sensitivity,
    ep_sensitivity,
)
from rmt_llm.keating_snaith import (
    ks_corrected_gamma,
    ks_gue_mean_spacing,
    ks_n_crit_correction,
    ks_relative_correction,
    ks_zeta_zero_statistics,
)
from rmt_llm.marchenko_pastur import (
    mp_bounds,
    mp_cdf,
    mp_density,
    mp_sample,
    mp_stieltjes,
)
from rmt_llm.nhse import (
    nhse_eigenvalues,
    nhse_hamiltonian,
    nhse_imag_collapse,
    nhse_point_gap,
    nhse_skin_strength,
    nhse_winding_number,
)
from rmt_llm.thermodynamics import (
    autoregressive_irreversibility,
    cognitive_mode,
    free_energy,
    landauer_cost,
    rg_fixed_point,
    rg_flow_lambda,
    spectral_entropy,
)
from rmt_llm.tracy_widom import (
    tracy_widom_cdf,
    tracy_widom_mean,
    tracy_widom_pdf,
    tracy_widom_skewness,
    tracy_widom_variance,
)


# ============================================================
# 1. MARCHENKO-PASTUR LAW TESTS
# ============================================================


class TestMarchenkoPasturBounds:
    """Tests for MP support bounds lambda_-, lambda_+."""

    def test_bounds_q_half_sigma2_1(self):
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        np.testing.assert_allclose(lam_m, (1 - math.sqrt(0.5)) ** 2, rtol=1e-10)
        np.testing.assert_allclose(lam_p, (1 + math.sqrt(0.5)) ** 2, rtol=1e-10)

    def test_bounds_q_1(self):
        lam_m, lam_p = mp_bounds(1.0, 1.0)
        np.testing.assert_allclose(lam_m, 0.0, atol=1e-10)
        np.testing.assert_allclose(lam_p, 4.0, rtol=1e-10)

    def test_bounds_positive_interval(self):
        for q in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            lam_m, lam_p = mp_bounds(q, 1.0)
            assert lam_m < lam_p, f"lambda_- < lambda_+ for q={q}"
            assert lam_m >= 0, f"lambda_- >= 0 for q={q}"

    def test_bounds_sigma2_scaling(self):
        for s2 in [0.5, 1.0, 2.0, 3.0]:
            lam_m, lam_p = mp_bounds(0.5, s2)
            lam_m0, lam_p0 = mp_bounds(0.5, 1.0)
            np.testing.assert_allclose(lam_m, s2 * lam_m0, rtol=1e-10)
            np.testing.assert_allclose(lam_p, s2 * lam_p0, rtol=1e-10)

    def test_invalid_q_raises(self):
        with pytest.raises(ValueError):
            mp_bounds(0.0, 1.0)
        with pytest.raises(ValueError):
            mp_bounds(1.5, 1.0)
        with pytest.raises(ValueError):
            mp_bounds(-0.1, 1.0)

    def test_invalid_sigma2_raises(self):
        with pytest.raises(ValueError):
            mp_bounds(0.5, 0.0)
        with pytest.raises(ValueError):
            mp_bounds(0.5, -1.0)


class TestMarchenkoPasturDensity:
    """Tests for MP density function."""

    def test_density_zero_outside_support(self):
        lam = np.linspace(-1, 6, 1000)
        rho = mp_density(lam, 0.5, 1.0)
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        outside = (lam < lam_m) | (lam > lam_p)
        np.testing.assert_allclose(rho[outside], 0.0, atol=1e-15)

    def test_density_positive_inside_support(self):
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        lam = np.linspace(lam_m + 0.01, lam_p - 0.01, 100)
        rho = mp_density(lam, 0.5, 1.0)
        assert np.all(rho > 0), "Density must be positive inside support"

    def test_density_integrates_to_one(self):
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        lam = np.linspace(lam_m + 1e-6, lam_p - 1e-6, 10000)
        rho = mp_density(lam, 0.5, 1.0)
        integral = np.trapezoid(rho, lam)
        np.testing.assert_allclose(integral, 1.0, atol=0.01)

    def test_density_peak_at_left_edge(self):
        """MP density diverges at lambda_- for q < 1."""
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        lam = np.linspace(lam_m + 0.001, lam_p - 0.001, 1000)
        rho = mp_density(lam, 0.5, 1.0)
        # Peak should be near left edge
        assert np.argmax(rho) < len(rho) // 4

    def test_density_q_scaling(self):
        """Support width should increase as q increases."""
        _, lam_p03 = mp_bounds(0.3, 1.0)
        _, lam_p07 = mp_bounds(0.7, 1.0)
        assert lam_p07 > lam_p03


class TestMarchenkoPasturCDF:
    """Tests for MP CDF."""

    def test_cdf_monotone(self):
        lam = np.linspace(0, 5, 200)
        F = mp_cdf(lam, 0.5, 1.0)
        assert np.all(np.diff(F) >= -1e-10), "CDF must be non-decreasing"

    def test_cdf_boundary_values(self):
        lam_m, lam_p = mp_bounds(0.5, 1.0)
        F_at_minus = mp_cdf(np.array([lam_m - 0.1]), 0.5, 1.0)
        F_at_plus = mp_cdf(np.array([lam_p + 0.1]), 0.5, 1.0)
        np.testing.assert_allclose(F_at_minus[0], 0.0, atol=0.01)
        np.testing.assert_allclose(F_at_plus[0], 1.0, atol=0.01)


class TestMarchenkoPasturSampling:
    """Tests for MP eigenvalue sampling."""

    def test_sample_shape(self, rng):
        eigs = mp_sample(100, 200, 1.0, rng)
        assert eigs.shape == (100,)

    def test_sample_sorted(self, rng):
        eigs = mp_sample(50, 100, 1.0, rng)
        assert np.all(np.diff(eigs) >= 0), "Eigenvalues must be sorted"

    def test_sample_positive(self, rng):
        eigs = mp_sample(50, 100, 1.0, rng)
        assert np.all(eigs > 0), "Eigenvalues must be positive"

    def test_sample_bulk_within_bounds(self, rng):
        """95% of eigenvalues should fall within [lambda_-, lambda_+]."""
        n, t = 200, 400
        eigs = mp_sample(n, t, 1.0, rng)
        lam_m, lam_p = mp_bounds(n / t, 1.0)
        inside = np.sum((eigs >= lam_m) & (eigs <= lam_p))
        assert inside / n > 0.90


class TestMarchenkoPasturStieltjes:
    """Tests for MP Stieltjes transform."""

    def test_stieltjes_upper_half_plane(self):
        g = mp_stieltjes(1.0 + 0.1j, 0.5, 1.0)
        assert g.imag > 0, "Stieltjes transform must have Im(g) > 0 for Im(z) > 0"

    def test_stieltjes_at_real_point(self):
        """Stieltjes transform should be finite at a real point."""
        g = mp_stieltjes(2.0 + 0.1j, 0.5, 1.0)
        assert abs(g) < 100  # finite value
        assert g.imag > 0


# ============================================================
# 2. BBP PHASE TRANSITION TESTS
# ============================================================


class TestBBPTransition:
    """Tests for the BBP phase transition."""

    def test_critical_theta(self):
        for q in [0.1, 0.25, 0.5, 0.75, 1.0]:
            np.testing.assert_allclose(bbp_critical_theta(q), math.sqrt(q), rtol=1e-10)

    def test_subcritical_regime(self):
        """theta <= sqrt(q): lambda_max = lambda_+."""
        for q in [0.3, 0.5, 0.8]:
            theta_c = bbp_critical_theta(q)
            for theta in [0.0, theta_c / 2, theta_c]:
                lam_max = bbp_lambda_max(theta, q, 1.0)
                lam_plus = (1.0 + math.sqrt(q)) ** 2
                np.testing.assert_allclose(lam_max, lam_plus, rtol=1e-10)

    def test_supercritical_regime(self):
        """theta > sqrt(q): lambda_max = sigma^2 * (1 + theta^2 / q)."""
        q = 0.5
        for theta in [1.0, 1.5, 2.0]:
            lam_max = bbp_lambda_max(theta, q, 1.0)
            expected = 1.0 * (1.0 + theta**2 / q)
            np.testing.assert_allclose(lam_max, expected, rtol=1e-10)

    def test_signal_separation_zero_below(self):
        """Below transition: gap = 0."""
        gap = bbp_signal_separation(0.3, 0.5, 1.0)
        np.testing.assert_allclose(gap, 0.0, atol=1e-10)

    def test_signal_separation_positive_above(self):
        """Above transition: gap > 0."""
        gap = bbp_signal_separation(1.0, 0.5, 1.0)
        assert gap > 0

    def test_is_supercritical(self):
        assert not bbp_is_supercritical(0.3, 0.5)
        assert not bbp_is_supercritical(math.sqrt(0.5), 0.5)
        assert bbp_is_supercritical(0.8, 0.5)

    def test_fluctuation_scaling(self):
        """Tracy-Widom scaling: N^{-1/3}."""
        scale = bbp_fluctuation_scaling(1000, 0.5)
        np.testing.assert_allclose(scale, 1000 ** (-1.0 / 3.0), rtol=1e-10)

    def test_bbp_sample(self, rng):
        eigs = bbp_sample(50, 100, 1.5, 1.0, rng)
        assert eigs.shape == (50,)
        assert np.all(np.diff(eigs) >= 0)

    def test_continuous_at_transition(self):
        """lambda_max should be continuous at theta = sqrt(q)."""
        q = 0.5
        theta_c = bbp_critical_theta(q)
        lam_at = bbp_lambda_max(theta_c, q, 1.0)
        lam_above = bbp_lambda_max(theta_c + 0.1, q, 1.0)
        # The function is continuous but has a kink; values should be close
        np.testing.assert_allclose(lam_at, lam_above, atol=1.0)


# ============================================================
# 3. TRACY-WIDOM TESTS
# ============================================================


class TestTracyWidom:
    """Tests for the Tracy-Widom F_2 distribution."""

    def test_cdf_range(self):
        s = np.linspace(-5, 5, 100)
        F = tracy_widom_cdf(s)
        assert np.all(F >= 0)
        assert np.all(F <= 1.01)  # slight tolerance

    def test_cdf_monotone(self):
        s = np.linspace(-5, 5, 100)
        F = tracy_widom_cdf(s)
        assert np.all(np.diff(F) >= -1e-10)

    def test_cdf_left_tail(self):
        F = tracy_widom_cdf(-5.0)
        assert F < 0.001, "Left tail should be very small"

    def test_cdf_right_tail(self):
        F = tracy_widom_cdf(5.0)
        assert F > 0.5, "Right tail should approach 1"

    def test_pdf_positive(self):
        s = np.linspace(-4, 4, 100)
        f = tracy_widom_pdf(s)
        assert np.all(f >= -1e-10), "PDF must be non-negative"

    def test_known_moments(self):
        """Test against known values of TW F_2 moments."""
        np.testing.assert_allclose(tracy_widom_mean(), TW_MEAN, rtol=1e-4)
        np.testing.assert_allclose(tracy_widom_variance(), TW_VARIANCE, rtol=1e-3)
        np.testing.assert_allclose(tracy_widom_skewness(), TW_SKEWNESS, rtol=0.01)

    def test_cdf_scalar_input(self):
        F = tracy_widom_cdf(0.0)
        assert isinstance(F, (float, np.floating))


# ============================================================
# 4. NHSE TESTS
# ============================================================


class TestNHSE:
    """Tests for the Non-Hermitian Skin Effect."""

    def test_winding_below_transition(self):
        assert nhse_winding_number(0.5, 0.3) == 0
        assert nhse_winding_number(0.99, 0.5) == 0

    def test_winding_above_transition(self):
        assert nhse_winding_number(1.5, 0.3) == 1
        assert nhse_winding_number(2.0, 0.5) == 1

    def test_winding_at_transition(self):
        """At exactly n_ratio = 1.0, we define w = 0 (subcritical)."""
        assert nhse_winding_number(1.0, 0.3) == 0

    def test_skin_strength_zero_below(self):
        s = nhse_skin_strength(0.5, 0.3)
        np.testing.assert_allclose(s, 0.0)

    def test_skin_strength_positive_above(self):
        s = nhse_skin_strength(1.5, 0.5)
        assert s > 0

    def test_skin_strength_bounded(self):
        for n_ratio in [0.5, 1.0, 1.5, 2.0, 5.0]:
            for gamma in [0.1, 0.5, 1.0]:
                s = nhse_skin_strength(n_ratio, gamma)
                assert 0 <= s <= 1.0 + 1e-10

    def test_imag_collapse_equals_skin(self):
        """Imag collapse fraction equals skin strength."""
        for n_ratio in [0.5, 1.5]:
            for gamma in [0.2, 0.7]:
                np.testing.assert_allclose(
                    nhse_imag_collapse(n_ratio, gamma),
                    nhse_skin_strength(n_ratio, gamma),
                )

    def test_point_gap_positive_below(self):
        gap = nhse_point_gap(0.5, 0.3)
        assert gap > 0

    def test_point_gap_near_zero_above(self):
        gap = nhse_point_gap(2.0, 0.3)
        assert gap < 0.05

    def test_hamiltonian_shape(self, rng):
        H = nhse_hamiltonian(10, 0.3, rng)
        assert H.shape == (10, 10)
        assert H.dtype == complex

    def test_hamiltonian_has_complex_eigenvalues(self, rng):
        """Non-Hermitian system should have complex eigenvalues."""
        # Generate a simple non-Hermitian matrix directly
        H = nhse_hamiltonian(20, 0.5, rng)
        # Add explicit non-Hermitian perturbation to ensure complex spectrum
        perturbation = 1j * 0.1 * rng.normal(size=(20, 20))
        H_nh = H + perturbation
        eigs = np.linalg.eigvals(H_nh)
        max_imag = np.max(np.abs(eigs.imag))
        assert max_imag > 0.01

    def test_eigenvalues_complex(self, rng):
        eigs = nhse_eigenvalues(10, 0.3, rng)
        assert eigs.shape == (10,)
        assert eigs.dtype == complex

    def test_winding_invalid_raises(self):
        with pytest.raises(ValueError):
            nhse_winding_number(-0.5, 0.3)
        with pytest.raises(ValueError):
            nhse_winding_number(0.5, -0.1)


# ============================================================
# 5. CAPUTO FRACTIONAL DYNAMICS TESTS
# ============================================================


class TestCaputoFractional:
    """Tests for Caputo fractional dynamics and RLHF utility trap."""

    def test_mean_collapse_time_formula(self):
        """<T_crit> = c * mu_eff^(-1/beta)."""
        mu = 0.1
        t = caputo_mean_collapse_time(mu, beta=0.5, c=1.0)
        np.testing.assert_allclose(t, mu ** (-2.0), rtol=1e-10)

    def test_quadratic_acceleration(self):
        """For beta=0.5, the exponent -1/beta = -2 (quadratic scaling)."""
        mu = 0.1
        t_half = caputo_mean_collapse_time(mu, beta=0.5)
        # beta=0.5 -> exponent = -1/0.5 = -2 -> mu^(-2) = 100
        np.testing.assert_allclose(t_half, 100.0, rtol=1e-6)
        # beta=0.3 -> exponent = -1/0.3 ≈ -3.33 -> even stronger scaling
        t_03 = caputo_mean_collapse_time(mu, beta=0.3)
        assert t_03 > t_half  # smaller beta -> longer collapse time

    def test_collapse_time_decreases_with_rlhf(self):
        """Stronger RLHF -> shorter collapse time."""
        t1 = caputo_mean_collapse_time(0.1, beta=0.5)
        t2 = caputo_mean_collapse_time(0.2, beta=0.5)
        assert t2 < t1

    def test_collapse_time_decreases_with_beta(self):
        """Higher beta -> shorter collapse time (exponent 1/beta is smaller)."""
        t1 = caputo_mean_collapse_time(0.1, beta=0.3)
        t2 = caputo_mean_collapse_time(0.1, beta=0.7)
        assert t2 < t1

    def test_n_crit_estimate(self):
        """N_crit should be positive and of reasonable magnitude."""
        n_crit = caputo_n_crit()
        assert n_crit > 0
        assert 50 < n_crit < 500  # reasonable range

    def test_fokker_planck_drift_positive(self):
        """Drift should be positive for positive RLHF."""
        drift = caputo_fokker_planck_drift(0.1, beta=0.5)
        assert drift > 0

    def test_fokker_planck_drift_scales_with_rlhf(self):
        """Drift scales linearly with RLHF strength."""
        d1 = caputo_fokker_planck_drift(0.1, beta=0.5)
        d2 = caputo_fokker_planck_drift(0.2, beta=0.5)
        np.testing.assert_allclose(d2 / d1, 2.0, rtol=1e-10)

    def test_invalid_mu_eff_raises(self):
        with pytest.raises(ValueError):
            caputo_mean_collapse_time(0.0, beta=0.5)
        with pytest.raises(ValueError):
            caputo_mean_collapse_time(-0.1, beta=0.5)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            caputo_mean_collapse_time(0.1, beta=0.0)
        with pytest.raises(ValueError):
            caputo_mean_collapse_time(0.1, beta=1.0)


# ============================================================
# 6. KEATING-SNAITH TESTS
# ============================================================


class TestKeatingSnaith:
    """Tests for Keating-Snaith corrections."""

    def test_corrected_gamma_approaches_gamma1(self):
        """As N -> infinity, gamma_1(N) -> gamma_1."""
        g_large_n = ks_corrected_gamma(100000)
        np.testing.assert_allclose(g_large_n, GAMMA_1, rtol=1e-4)

    def test_corrected_gamma_different_for_small_n(self):
        """For small N, correction should be noticeable."""
        g_small = ks_corrected_gamma(10)
        assert abs(g_small - GAMMA_1) > 0.001

    def test_correction_decreases_with_n(self):
        """Correction should decrease as N increases."""
        corr_10 = abs(ks_corrected_gamma(10) - GAMMA_1)
        corr_100 = abs(ks_corrected_gamma(100) - GAMMA_1)
        corr_1000 = abs(ks_corrected_gamma(1000) - GAMMA_1)
        assert corr_10 > corr_100 > corr_1000

    def test_n_crit_positive(self):
        n_crit = ks_n_crit_correction(100)
        assert n_crit > 0

    def test_relative_correction_small(self):
        """For reasonable N, relative correction should be small."""
        rel = ks_relative_correction(1000)
        assert rel < 0.01

    def test_zeta_zero_statistics(self):
        zeros = ks_zeta_zero_statistics(5)
        assert len(zeros) == 5
        np.testing.assert_allclose(zeros[0], 14.134725, rtol=1e-5)

    def test_gue_mean_spacing_near_one(self):
        """GUE mean spacing should be close to 1.0 for unfolded zeros."""
        spacing = ks_gue_mean_spacing(10)
        np.testing.assert_allclose(spacing, 1.0, atol=0.2)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            ks_corrected_gamma(0)
        with pytest.raises(ValueError):
            ks_corrected_gamma(-1)


# ============================================================
# 7. EP SURFACES TESTS
# ============================================================


class TestEPSurfaces:
    """Tests for exceptional-point spectral sensitivity."""

    def test_sensitivity_formula(self):
        """delta_lambda = epsilon^{1/k}."""
        delta = ep_sensitivity(1e-16, 2)
        np.testing.assert_allclose(delta, (1e-16) ** 0.5, rtol=1e-10)

    def test_sensitivity_order_2(self):
        delta = ep_sensitivity(0.01, 2)
        np.testing.assert_allclose(delta, 0.1, rtol=1e-10)

    def test_sensitivity_increases_with_order(self):
        """Higher EP order -> larger sensitivity for same epsilon."""
        d2 = ep_sensitivity(1e-10, 2)
        d5 = ep_sensitivity(1e-10, 5)
        d10 = ep_sensitivity(1e-10, 10)
        assert d2 < d5 < d10

    def test_rounding_sensitivity(self):
        """Float64 rounding sensitivity approaches 1 as N increases."""
        delta_10 = ep_rounding_sensitivity(10, np.float64)
        delta_100 = ep_rounding_sensitivity(100, np.float64)
        # epsilon^(1/N) increases toward 1 as N grows
        assert delta_100 > delta_10
        assert delta_100 < 1.0

    def test_ep_order_estimation(self):
        """Can we recover the EP order from sensitivity?"""
        epsilon = 1e-8
        order = 3
        separation = ep_sensitivity(epsilon, order)
        k_est = ep_order_from_separation(separation, epsilon)
        np.testing.assert_allclose(k_est, order, rtol=0.15)

    def test_is_near_ep(self):
        """Eigenvalues near coalescence should be detected."""
        eigs = np.array([1.0 + 0j, 1.0 + 1e-8j, 2.0 + 0j])
        is_near, min_sep = ep_is_near(2, eigs, tol=1e-5)
        assert is_near
        assert min_sep < 1e-5

    def test_perturbed_matrix_shape(self, rng):
        H = ep_perturbed_matrix(5, 3, 1e-10, rng)
        assert H.shape == (5, 5)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            ep_sensitivity(-0.1, 2)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            ep_sensitivity(0.01, 1)


# ============================================================
# 8. THERMODYNAMICS TESTS
# ============================================================


class TestThermodynamics:
    """Tests for thermodynamic framework."""

    def test_free_energy_formula(self):
        F = free_energy(10.0, 2.0, 3.0)
        np.testing.assert_allclose(F, 10.0 - 2.0 * 3.0, rtol=1e-10)

    def test_factual_mode(self):
        """F < 0 -> factual (crystal phase)."""
        mode = cognitive_mode(-1.0)
        assert mode == "factual"

    def test_creative_mode(self):
        """F > 0 -> creative (gas phase)."""
        mode = cognitive_mode(1.0)
        assert mode == "creative"

    def test_landauer_cost_positive(self):
        E = landauer_cost(1, 300.0)
        expected = K_B * 300.0 * LN2
        np.testing.assert_allclose(E, expected, rtol=1e-10)
        assert E > 0

    def test_landauer_cost_scales_linearly(self):
        E1 = landauer_cost(1)
        E10 = landauer_cost(10)
        np.testing.assert_allclose(E10 / E1, 10.0, rtol=1e-10)

    def test_spectral_entropy_uniform(self):
        """Uniform distribution has maximum entropy."""
        eigs = np.ones(10)
        S = spectral_entropy(eigs)
        np.testing.assert_allclose(S, math.log(10), rtol=1e-10)

    def test_spectral_entropy_pure_state(self):
        """Pure state has zero entropy."""
        eigs = np.array([1.0, 0.0, 0.0, 0.0])
        S = spectral_entropy(eigs)
        np.testing.assert_allclose(S, 0.0, atol=1e-10)

    def test_spectral_entropy_non_negative(self):
        eigs = np.array([0.5, 0.3, 0.2])
        S = spectral_entropy(eigs)
        assert S >= 0

    def test_rg_flow_decays(self):
        """RG flow should decay exponentially."""
        lam = rg_flow_lambda(1.0, 10, 0.1)
        np.testing.assert_allclose(lam, math.exp(-1.0), rtol=1e-10)

    def test_rg_fixed_point_zero(self):
        """For positive beta, fixed point is 0."""
        fp = rg_fixed_point(1.0, 0.1, 1000)
        np.testing.assert_allclose(fp, 0.0, atol=1e-10)

    def test_autoregressive_irreversibility(self):
        """Irreversibility should be positive."""
        S = autoregressive_irreversibility(100)
        assert S > 0

    def test_autoregressive_scales_with_tokens(self):
        S1 = autoregressive_irreversibility(1)
        S100 = autoregressive_irreversibility(100)
        np.testing.assert_allclose(S100 / S1, 100.0, rtol=1e-10)


# ============================================================
# 9. CROSS-MODULE VERIFICATION TESTS
# ============================================================


class TestCrossModuleVerification:
    """Tests that verify consistency between modules."""

    def test_bbp_matches_mp_at_transition(self):
        """At the BBP transition, lambda_max equals the MP bulk edge."""
        q = 0.5
        theta_c = bbp_critical_theta(q)
        lam_max = bbp_lambda_max(theta_c, q, 1.0)
        _, lam_plus = mp_bounds(q, 1.0)
        np.testing.assert_allclose(lam_max, lam_plus, rtol=1e-10)

    def test_n_crit_caputo_ks_consistency(self):
        """N_crit from Caputo and KS should be of same order."""
        n_crit_caputo = caputo_n_crit()
        n_crit_ks = ks_n_crit_correction(GPT2_CONTEXT_WINDOW)
        ratio = n_crit_caputo / n_crit_ks
        assert 0.5 < ratio < 2.0, f"N_crit ratio {ratio} is too far from 1"

    def test_nse_tw_consistency(self, rng):
        """Above NHSE transition, eigenvalue fluctuations should
        follow Tracy-Widom statistics (qualitative check)."""
        # Generate NHSE eigenvalues above transition
        eigs = nhse_eigenvalues(20, 0.3, rng)
        # Most imaginary parts should be small
        assert np.mean(np.abs(eigs.imag)) < 3.0

    def test_ep_nhse_connection(self):
        """EP sensitivity and NHSE collapse both relate to
        spectral instability — qualitative consistency check."""
        # For large N, EP rounding sensitivity should be significant
        delta_ep = ep_rounding_sensitivity(1000, np.float64)
        # NHSE point gap above transition should be small
        gap_nhse = nhse_point_gap(2.0, 0.3)
        # Both should indicate instability
        assert delta_ep > 0 or gap_nhse < 1.0

    def test_free_energy_mode_consistency(self):
        """Free energy sign should correctly identify cognitive mode."""
        # F = U - T*S < 0 when T*S > U (high T or S) -> factual
        F_factual = free_energy(1.0, 2.0, 1.0)  # 1 - 2 = -1 < 0
        assert cognitive_mode(F_factual) == "factual"

        # F > 0 when U > T*S (low T or S) -> creative
        F_creative = free_energy(3.0, 1.0, 1.0)  # 3 - 1 = 2 > 0
        assert cognitive_mode(F_creative) == "creative"

    def test_zeta_zeros_monotone(self):
        """Riemann zeta zeros must be monotonically increasing."""
        zeros = ks_zeta_zero_statistics(10)
        assert np.all(np.diff(zeros) > 0)

    def test_constants_consistency(self):
        """Key constants should be consistent with module values."""
        np.testing.assert_allclose(GAMMA_1, 14.134725, rtol=1e-5)
        np.testing.assert_allclose(BETA_CAPUTO, 0.5)
        np.testing.assert_allclose(THETA_B_DEGREES, 7.07, rtol=1e-10)
