"""
Caputo Fractional-Time Memory — RLHF utility trap and hallucination onset.

The Caputo fractional derivative of order beta in (0, 1) models memory
effects in autoregressive generation. With beta ~ 0.5, the mean collapse
time scales as:

    <T_crit> proportional to (mu_eff)^(-1/beta)

Since 1/beta ~ 2 for beta ~ 0.5, even small RLHF pressure mu_eff
quadratically accelerates hallucination onset. This is the mathematical
core of the "utility trap" described in the research.

References:
  - Caputo (1967), "Linear models of dissipation whose Q is almost
    frequency independent", Geophys. J. R. Astr. Soc.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# Key constants
BETA_CAPUTO = 0.5  # Caputo memory parameter (from experimental fits)
GAMMA_1_RIEMANN = 14.134725  # First Riemann zeta zero (imaginary part)


def caputo_mean_collapse_time(mu_eff: float, beta: float = BETA_CAPUTO, c: float = 1.0) -> float:
    """Compute the mean hallucination collapse time.

    <T_crit> = c * (mu_eff)^(-1/beta)

    Parameters
    ----------
    mu_eff : float
        Effective RLHF drift strength (positive).
    beta : float
        Caputo memory parameter in (0, 1). Default: 0.5.
    c : float
        Proportionality constant.

    Returns
    -------
    t_crit : float
        Mean collapse time.

    Raises
    ------
    ValueError
        If mu_eff <= 0 or beta not in (0, 1).
    """
    if mu_eff <= 0:
        raise ValueError(f"mu_eff must be positive, got {mu_eff}")
    if beta <= 0 or beta >= 1:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    return c * mu_eff ** (-1.0 / beta)


def caputo_quadratic_acceleration(mu_eff: float) -> float:
    """Compute the quadratic acceleration factor for beta = 0.5.

    With beta = 0.5, the mean collapse time scales as (mu_eff)^(-2),
    meaning RLHF pressure is amplified quadratically.

    Parameters
    ----------
    mu_eff : float
        Effective RLHF drift strength.

    Returns
    -------
    acceleration : float
        Quadratic acceleration factor.

    Raises
    ------
    ValueError
        If mu_eff <= 0.

    Notes
    -----
    BUGFIX: this function previously called
    ``caputo_mean_collapse_time(mu_eff, beta=1.0)``, which *always* raised
    ``ValueError`` (beta must be in the open interval (0, 1)). The beta -> 1
    limit is analytically well-defined: <T_crit> = c * mu^(-1/beta) gives
    T(beta=1) = mu^(-1), so the acceleration factor is

        T(beta=0.5) / T(beta=1) = mu^(-2) / mu^(-1) = 1 / mu_eff.
    """
    if mu_eff <= 0:
        raise ValueError(f"mu_eff must be positive, got {mu_eff}")
    # analytic beta -> 1 limit of caputo_mean_collapse_time (c = 1): mu^(-1)
    t_beta_one = mu_eff ** (-1.0)
    return caputo_mean_collapse_time(mu_eff, beta=0.5) / t_beta_one


def caputo_derivative(
    f: ArrayLike,
    t: ArrayLike,
    beta: float,
    n_points: int | None = None,  # noqa: ARG001
) -> np.ndarray:
    """Compute the Caputo fractional derivative numerically via Grunwald-Letnikov.

    D^beta f(t) = (1/Gamma(1-beta)) * integral_0^t f'(tau) / (t-tau)^beta dtau

    Parameters
    ----------
    f : array_like
        Function values at time points t.
    t : array_like
        Time points (equally spaced).
    beta : float
        Fractional order in (0, 1).
    n_points : int, optional
        Number of points to use (defaults to len(t)).

    Returns
    -------
    d_beta_f : ndarray
        Caputo fractional derivative values.
    """
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(f) != len(t):
        raise ValueError("f and t must have the same length")

    n = len(t)
    dt = t[1] - t[0] if n > 1 else 1.0

    # Grunwald-Letnikov coefficients
    coeff = np.zeros(n)
    coeff[0] = 1.0
    for k in range(1, n):
        coeff[k] = coeff[k - 1] * (1.0 - (1.0 + beta) / k)

    # Compute the fractional derivative
    result = np.zeros(n)
    for j in range(1, n):
        s = 0.0
        for k in range(j + 1):
            s += coeff[k] * f[j - k]
        result[j] = s / (dt**beta)

    return result


def caputo_fokker_planck_drift(mu_rlhf: float, beta: float = BETA_CAPUTO, t: float = 1.0) -> float:
    """Compute the effective drift in the Fokker-Planck equation under RLHF.

    RLHF optimization creates an artificial drift term mu_rlhf in the
    Fokker-Planck equation, which accelerates the collapse of the
    autoregressive process.

    Parameters
    ----------
    mu_rlhf : float
        RLHF optimization strength.
    beta : float
        Caputo memory parameter.
    t : float
        Time.

    Returns
    -------
    drift : float
        Effective drift in the Fokker-Planck equation.
    """
    # The drift is amplified by the Caputo memory factor t^(1-beta) / Gamma(2-beta)
    from math import gamma as gamma_func

    return mu_rlhf * t ** (1.0 - beta) / gamma_func(2.0 - beta)


def caputo_n_crit(theta_b: float = 7.07, gamma_1: float = GAMMA_1_RIEMANN) -> float:
    """Estimate the critical token count N_crit from Caputo dynamics.

    The critical count is related to the first Riemann zeta zero and
    the rotation angle theta_b from the Navier-Stokes regularity analysis.

    Parameters
    ----------
    theta_b : float
        Rotation angle in degrees (from NS regularity analysis). Default: 7.07.
    gamma_1 : float
        First Riemann zeta zero (imaginary part). Default: 14.134725.

    Returns
    -------
    n_crit : float
        Estimated critical token count.
    """
    # N_crit ~ gamma_1 / (theta_b * pi / 180)
    # This is a simplified model connecting the spectral gap to the
    # autoregressive context window
    return gamma_1 / (theta_b * np.pi / 180.0)
