"""
Marchenko-Pastur Law — Bulk eigenvalue density of random covariance matrices.

For an N x T matrix X with i.i.d. entries of variance sigma^2, the empirical
spectral density of M = X X^T / T converges to the Marchenko-Pastur
distribution as N, T -> infinity with q = N/T fixed.

References:
  - Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
    of random matrices", Mat. Sb.

RCDA extension (RMT-Calibrated Directional Ablation, see README_RCDA.md):
  - mp_upper_edge(sigma2, c)   — lambda_+ = sigma2 * (1 + sqrt(c))^2, any c > 0
  - tw_scale(lambda_plus, N)   — Tracy-Widom edge-fluctuation scale lambda_+ * N^{-2/3}
  - bbp_threshold(c)           — critical spike strength theta_c = sqrt(c)
  - bbp_excess(theta_v, c)     — delta = max(0, (theta_v - theta_c) / theta_v) in [0, 1)
  - alpha_from_bbp(theta_v, c, kappa) — alpha_0 = min(1, kappa * delta / (1 - delta))
  - rayleigh_spike_strength(...) — invert the repo's BBP asymptotic for an observed
    Rayleigh quotient (see the docstring for the exact parametrization)

Conventions used throughout RCDA (fixed once, everywhere):
  * A weight matrix W in R^{m x n} (m = out_features, n = in_features) is treated
    as n samples (columns) in R^m, so its sample covariance is
        C = (1/n) * W W^T in R^{m x m},   aspect q = m/n,
    matching the hidden-state convention of rmt_intervention.py (T tokens as
    samples). q > 1 is allowed (continuous bulk edge, cf. _mp_bounds_general in
    probe_real_gpt2.py).
  * sigma_W^2 is estimated from the bulk: median(eig(C)) — the same working
    heuristic as in rmt_intervention.py ("conservative for supercritical data").
  * The plan's theta_c = sqrt(n/m) (n <= m) with permutation when n > m equals
    sqrt(min(q, 1/q)); the additive-spike parametrization used for calibration
    keeps theta_c = sqrt(q) of rmt_llm.bbp_transition (the repo formula), which
    for q <= 1 coincides with the plan and for q > 1 remains the general-q BBP
    threshold.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# RCDA calibration core (RMT-Calibrated Directional Ablation)
# ---------------------------------------------------------------------------

def mp_upper_edge(sigma2: float, c: float) -> float:
    """Upper MP bulk edge lambda_+ = sigma2 * (1 + sqrt(c))^2 for any c > 0.

    Same expression as ``mp_bounds`` but valid beyond q in (0, 1]: for c > 1
    the continuous part of the support still ends at sigma2*(1+sqrt(c))^2
    (cf. ``_mp_bounds_general`` in probe_real_gpt2.py).
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    return float(sigma2 * (1.0 + np.sqrt(c)) ** 2)


def tw_scale(lambda_plus: float, n: float) -> float:
    """Tracy-Widom edge-fluctuation scale of lambda_max.

    RCDA plan formula: TW-scale(lambda_+, N) = lambda_+ * N^{-2/3}.
    The fluctuation of the largest eigenvalue of a sample covariance matrix
    around the bulk edge is of this order (sigma = 1 in TW units; the F1/F2
    spread contributes an O(1) factor deliberately absorbed into the z-threshold).

    Parameters
    ----------
    lambda_plus : float
        Upper MP bulk edge (same units as the eigenvalues).
    n : float
        Sample size N behind the covariance (tokens, columns, ...).

    Returns
    -------
    scale : float
        One Tracy-Widom unit expressed in eigenvalue units.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if lambda_plus <= 0:
        raise ValueError(f"lambda_plus must be positive, got {lambda_plus}")
    return float(lambda_plus * float(n) ** (-2.0 / 3.0))


def rmt_z_score(lambda_max: float, lambda_plus: float, n: float) -> float:
    """RMT anomaly score of a layer: z = (lambda_max - lambda_+) / TW-scale.

    z <=~ 2 means "no RMT-confirmed structure above the bulk edge";
    the layer's spectrum is then indistinguishable from MP noise at the
    resolution of the sample.
    """
    scale = tw_scale(lambda_plus, n)
    return float((lambda_max - lambda_plus) / scale)


def bbp_threshold(c: float) -> float:
    """Critical spike strength theta_c = sqrt(c) of the BBP transition.

    This is the repo formula (rmt_llm.bbp_transition.bbp_critical_theta)
    generalised to any c > 0: for the aspect ratio q = m/n of the covariance
    at hand the threshold is sqrt(q). The RCDA plan writes sqrt(n/m) for
    n <= m and permutes for n > m, which equals sqrt(min(c, 1/c)) in the
    (n, m)-notation; with the convention q = m/n both agree for q <= 1.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    return float(np.sqrt(c))


def rayleigh_spike_strength(lambda_v: float, sigma2: float, q: float) -> float:
    """Spike strength theta_v of an observed Rayleigh quotient (repo parametrization).

    The repo's BBP asymptotic (rmt_llm.bbp_transition.bbp_lambda_max) reads
        lambda_max = sigma2 * (1 + theta^2 / q)   for the supercritical spike.
    Given an observed projection mass lambda_v = v^T C v of a unit direction v
    on the sample covariance C, the implied theta_v is therefore
        theta_v = sqrt(q * (lambda_v / sigma2 - 1))   (clipped at 0).
    Under the null lambda_v ~ sigma2 and theta_v ~ 0; theta_v > theta_c = sqrt(q)
    flags a supercritical direction (equivalent, up to the parametrization of
    the asymptote, to the plan's test lambda_v > lambda_+).
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    excess = lambda_v / sigma2 - 1.0
    if excess <= 0:
        return 0.0
    return float(np.sqrt(q * excess))


def bbp_excess(theta_v: float, c: float) -> float:
    """BBP excess delta = max(0, (theta_v - theta_c) / theta_v) in [0, 1).

    delta ~ 0        — the direction has barely left the bulk (fragile signal);
    delta -> 1       — a strong outlier; full subtraction may be justified.
    delta == 0       — the direction is inside the bulk: do not touch it.
    """
    if theta_v <= 0:
        return 0.0
    theta_c = bbp_threshold(c)
    if theta_v <= theta_c:
        return 0.0
    return float((theta_v - theta_c) / theta_v)


def alpha_from_bbp(theta_v: float, c: float, kappa: float = 0.5) -> float:
    """Calibrated ablation strength alpha_0 = min(1, kappa * delta / (1 - delta)).

    RCDA plan, section 5: the BBP excess delta is mapped to a starting alpha
    instead of a blind grid search. delta = 0 -> alpha_0 = 0 (do not touch);
    delta -> 1 -> alpha_0 -> kappa / (1 - delta) -> 1 (cap). ``kappa`` is the
    safety factor (0.5 by default). This is a starting point for the
    edit-measure-keep loop, not a replacement for behavioural validation.
    """
    delta = bbp_excess(theta_v, c)
    if delta <= 0.0:
        return 0.0
    if delta >= 1.0:  # pragma: no cover (delta < 1 by construction)
        return 1.0
    return float(min(1.0, kappa * delta / (1.0 - delta)))


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Compute the support bounds lambda_-, lambda_+ of the MP law.

    Parameters
    ----------
    q : float
        Aspect ratio N/T, must be in (0, 1] for the standard case.
    sigma2 : float
        Population variance, must be positive.

    Returns
    -------
    (lambda_minus, lambda_plus) : tuple of floats
        Support endpoints of the Marchenko-Pastur density.

    Raises
    ------
    ValueError
        If q not in (0, 1] or sigma2 <= 0.
    """
    if q <= 0 or q > 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    sqrt_q = np.sqrt(q)
    lam_minus = sigma2 * (1.0 - sqrt_q) ** 2
    lam_plus = sigma2 * (1.0 + sqrt_q) ** 2
    return lam_minus, lam_plus


def mp_density(lam: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Evaluate the Marchenko-Pastur density at given eigenvalues.

    rho(lambda) = (1 / (2*pi*sigma^2*lambda*q))
                  * sqrt((lambda_+ - lambda)(lambda - lambda_-))

    For q > 1, there is additionally a point mass of weight (1 - 1/q) at
    lambda = 0. This function returns only the continuous part.

    Parameters
    ----------
    lam : array_like
        Eigenvalues at which to evaluate the density.
    q : float
        Aspect ratio N/T in (0, 1].
    sigma2 : float
        Population variance, positive.

    Returns
    -------
    rho : ndarray
        Density values (0 outside the support).
    """
    lam = np.asarray(lam, dtype=float)
    lam_minus, lam_plus = mp_bounds(q, sigma2)

    # Mask: inside support and positive
    mask = (lam > lam_minus) & (lam < lam_plus) & (lam > 0)
    rho = np.zeros_like(lam)

    if np.any(mask):
        lam_m = lam[mask]
        rho[mask] = (
            1.0
            / (2.0 * np.pi * sigma2 * lam_m * q)
            * np.sqrt((lam_plus - lam_m) * (lam_m - lam_minus))
        )

    return rho


def mp_cdf(lam: ArrayLike, q: float, sigma2: float = 1.0, n_points: int = 10000) -> np.ndarray:
    """Compute the CDF of the Marchenko-Pastur distribution numerically.

    Parameters
    ----------
    lam : array_like
        Points at which to evaluate the CDF.
    q : float
        Aspect ratio N/T in (0, 1].
    sigma2 : float
        Population variance.
    n_points : int
        Number of quadrature points for numerical integration.

    Returns
    -------
    F : ndarray
        CDF values.
    """
    lam = np.asarray(lam, dtype=float)
    lam_minus, lam_plus = mp_bounds(q, sigma2)

    # Build a fine grid for numerical integration
    eps = (lam_plus - lam_minus) * 1e-6
    grid = np.linspace(lam_minus + eps, lam_plus - eps, n_points)
    density = mp_density(grid, q, sigma2)
    dx = grid[1] - grid[0]

    # Cumulative sum
    cum = np.cumsum(density) * dx

    # Interpolate at requested points
    result = np.zeros_like(lam)
    for i, l in enumerate(lam):
        if l <= lam_minus:
            result[i] = 0.0
        elif l >= lam_plus:
            result[i] = 1.0
        else:
            idx = np.searchsorted(grid, l)
            if idx == 0:
                result[i] = 0.0
            elif idx >= len(grid):
                result[i] = 1.0
            else:
                # Linear interpolation
                frac = (l - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
                result[i] = cum[idx - 1] + frac * (cum[idx] - cum[idx - 1])

    return result


def mp_sample(
    n: int, t: int, sigma2: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Sample eigenvalues from a random covariance matrix.

    Generates an N x T matrix with i.i.d. N(0, sigma2) entries and computes
    the eigenvalues of M = X X^T / T.

    Parameters
    ----------
    n : int
        Number of rows (model dimension).
    t : int
        Number of columns (sample size / sequence length).
    sigma2 : float
        Entry variance.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    eigenvalues : ndarray of shape (N,)
        Sorted eigenvalues of the sample covariance matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(sigma2)
    X = rng.normal(0, sigma, size=(n, t))
    M = X @ X.T / t
    eigenvalues = np.linalg.eigvalsh(M)
    return np.sort(eigenvalues)


def mp_stieltjes(z: complex, q: float, sigma2: float = 1.0) -> complex:
    """Compute the Stieltjes transform of the Marchenko-Pastur distribution.

    g(z) = (1 - q - z) / (2 * sigma^2 * q * z)
           + sqrt((1 - q - z)^2 - 4 * q * z) / (2 * sigma^2 * q * z)

    where sqrt is the branch with positive imaginary part for Im(z) > 0.

    Parameters
    ----------
    z : complex
        Point in the upper half-plane.
    q : float
        Aspect ratio.
    sigma2 : float
        Population variance.

    Returns
    -------
    g : complex
        Stieltjes transform value.
    """
    discriminant = (1.0 - q - z) ** 2 - 4.0 * q * z
    sqrt_disc = np.sqrt(discriminant)
    # Choose branch with positive imaginary part when Im(z) > 0
    if sqrt_disc.imag < 0:
        sqrt_disc = -sqrt_disc
    return ((1.0 - q - z) + sqrt_disc) / (2.0 * sigma2 * q * z)
