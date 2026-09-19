"""
BBP Phase Transition — Signal eigenvalue emergence from the random bulk.

For a rank-1 spiked covariance matrix Sigma = sigma^2 * I + theta * v v^T,
the largest eigenvalue lambda_max undergoes a phase transition at theta = sqrt(q):

  - Subcritical (theta <= sqrt(q)): lambda_max -> lambda_+  (bulk edge)
  - Supercritical (theta > sqrt(q)): lambda_max -> sigma^2 * (1 + theta^2 / q)

This is the Baik-Ben Arous-Peche (BBP) transition, the key marker for
cognitive mode detection in LLM activations.

References:
  - Baik, Ben Arous & Peche (2005), "Phase transition of the largest
    eigenvalue for nonnull complex sample covariance matrices", Ann. Probab.
"""

from __future__ import annotations

import numpy as np


def bbp_critical_theta(q: float) -> float:
    """Compute the critical signal strength theta_c = sqrt(q).

    Parameters
    ----------
    q : float
        Aspect ratio N/T in (0, 1].

    Returns
    -------
    theta_c : float
        Critical signal strength.
    """
    if q <= 0 or q > 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return np.sqrt(q)


def bbp_lambda_max(theta: float, q: float, sigma2: float = 1.0) -> float:
    """Compute the asymptotic largest eigenvalue under a rank-1 spike.

    Parameters
    ----------
    theta : float
        Signal strength (spike magnitude).
    q : float
        Aspect ratio N/T in (0, 1].
    sigma2 : float
        Background variance.

    Returns
    -------
    lambda_max : float
        Asymptotic largest eigenvalue.
    """
    theta_c = bbp_critical_theta(q)
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

    if theta <= theta_c:
        return lam_plus
    return sigma2 * (1.0 + theta**2 / q)


def bbp_signal_separation(theta: float, q: float, sigma2: float = 1.0) -> float:
    """Compute the gap between lambda_max and the bulk edge lambda_+.

    A positive gap indicates supercritical regime (signal detected).

    Parameters
    ----------
    theta : float
        Signal strength.
    q : float
        Aspect ratio N/T in (0, 1].
    sigma2 : float
        Background variance.

    Returns
    -------
    gap : float
        lambda_max - lambda_+.  Positive = signal detected.
    """
    lam_max = bbp_lambda_max(theta, q, sigma2)
    lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    return lam_max - lam_plus


def bbp_is_supercritical(theta: float, q: float) -> bool:
    """Check whether the BBP transition has occurred.

    Parameters
    ----------
    theta : float
        Signal strength.
    q : float
        Aspect ratio N/T in (0, 1].

    Returns
    -------
    is_supercritical : bool
        True if theta > sqrt(q).
    """
    return theta > bbp_critical_theta(q)


def bbp_fluctuation_scaling(n: int, q: float) -> float:  # noqa: ARG001 (q kept for API compat)
    """Compute the fluctuation scale of lambda_max near the transition.

    In the critical regime (theta ~ sqrt(q)), the fluctuations are of order
    N^{-1/3} (Tracy-Widom scaling). In the sub/supercritical regime, they
    are of order N^{-1/2} (Gaussian scaling).

    Parameters
    ----------
    n : int
        Matrix dimension.
    q : float
        Aspect ratio.

    Returns
    -------
    scale : float
        Fluctuation scale.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    # Tracy-Widom regime (critical)
    return n ** (-1.0 / 3.0)


def bbp_sample(
    n: int, t: int, theta: float, sigma2: float = 1.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Sample eigenvalues from a rank-1 spiked covariance matrix.

    Generates X = sigma * G + theta * v e_1^T / sqrt(T) where G is i.i.d.
    N(0,1), v is a unit vector, and e_1 is the first standard basis vector.

    Parameters
    ----------
    n : int
        Matrix dimension.
    t : int
        Sample size.
    theta : float
        Signal strength.
    sigma2 : float
        Background variance.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    eigenvalues : ndarray of shape (N,)
        Sorted eigenvalues.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(sigma2)
    X = rng.normal(0, sigma, size=(n, t))
    # Add rank-1 spike along the first row
    X[0, :] += theta / np.sqrt(t)
    M = X @ X.T / t
    eigenvalues = np.linalg.eigvalsh(M)
    return np.sort(eigenvalues)
