"""
Marchenko-Pastur Law — Bulk eigenvalue density of random covariance matrices.

For an N x T matrix X with i.i.d. entries of variance sigma^2, the empirical
spectral density of M = X X^T / T converges to the Marchenko-Pastur
distribution as N, T -> infinity with q = N/T fixed.

References:
  - Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
    of random matrices", Mat. Sb.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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
