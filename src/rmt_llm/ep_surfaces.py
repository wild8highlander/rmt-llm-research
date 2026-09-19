"""
Exceptional-Point (EP) Surfaces — Spectral sensitivity divergence.

At an exceptional point of order k, eigenvalues of a non-Hermitian matrix
coalesce and the spectral sensitivity diverges:

    delta_lambda ~ epsilon^{1/k},  k ~ O(N)

This means that even infinitesimal perturbations (e.g., floating-point
rounding errors) can cause macroscopic spectral shifts, providing another
path to hallucination onset in LLMs.

References:
  - Kato (1980), "Perturbation Theory for Linear Operators", Springer.
  - Heiss (2012), "The physics of exceptional points", J. Phys. A.
"""

from __future__ import annotations

import numpy as np


def ep_sensitivity(epsilon: float, order: int) -> float:
    """Compute the eigenvalue sensitivity at an EP of given order.

    delta_lambda ~ epsilon^{1/k}

    Parameters
    ----------
    epsilon : float
        Perturbation strength (positive).
    order : int
        Exceptional point order k >= 2.

    Returns
    -------
    delta_lam : float
        Spectral shift magnitude.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if epsilon == 0:
        return 0.0
    return epsilon ** (1.0 / order)


def ep_rounding_sensitivity(n: int, dtype: type = np.float64) -> float:
    """Compute the spectral sensitivity due to floating-point rounding.

    With machine epsilon eps_mach ~ 2^{-53} for float64, and EP order
    k ~ O(N), the rounding-induced spectral shift is:

    delta_lambda ~ (eps_mach)^{1/N}

    Parameters
    ----------
    n : int
        Matrix dimension (EP order ~ N).
    dtype : type
        Floating-point dtype.

    Returns
    -------
    delta_lam : float
        Rounding-induced spectral shift.
    """
    eps_mach = np.finfo(dtype).eps
    return eps_mach ** (1.0 / n)


def ep_order_from_separation(separation: float, epsilon: float = 1e-16) -> float:
    """Estimate the EP order from observed eigenvalue separation.

    Given observed separation delta_lambda and perturbation epsilon,
    estimate the EP order k ~ log(epsilon) / log(delta_lambda).

    Parameters
    ----------
    separation : float
        Observed eigenvalue separation.
    epsilon : float
        Perturbation strength.

    Returns
    -------
    k : float
        Estimated EP order.
    """
    if separation <= 0 or epsilon <= 0:
        raise ValueError("separation and epsilon must be positive")
    return np.log(epsilon) / np.log(separation)


def ep_is_near(
    n: int,  # noqa: ARG001 (n kept for API compat)
    eigenvalues: np.ndarray,
    tol: float = 1e-6,
) -> tuple[bool, float]:
    """Check if a set of eigenvalues is near an exceptional point.

    An EP is characterized by near-coalescence of eigenvalues. We check
    the minimum pairwise separation relative to the spectral radius.

    Parameters
    ----------
    n : int
        Expected EP order.
    eigenvalues : ndarray
        Complex eigenvalues.
    tol : float
        Tolerance relative to spectral radius.

    Returns
    -------
    (is_near, min_separation) : tuple
        Whether eigenvalues are near an EP, and the minimum separation.
    """
    if len(eigenvalues) < 2:
        return False, float("inf")

    # Compute pairwise distances
    diffs = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    np.fill_diagonal(diffs, np.inf)
    min_sep = np.min(diffs)

    # Spectral radius
    spectral_radius = np.max(np.abs(eigenvalues))
    relative_sep = min_sep / max(spectral_radius, 1e-15)

    return relative_sep < tol, min_sep


def ep_perturbed_matrix(
    size: int, order: int = 2, epsilon: float = 1e-10, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a Jordan-block matrix near an EP of given order.

    The unperturbed matrix is a Jordan block of the given order, which has
    an EP of that order. The perturbation breaks the degeneracy.

    Parameters
    ----------
    size : int
        Overall matrix dimension (>= order).
    order : int
        EP order (Jordan block size).
    epsilon : float
        Perturbation strength.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    H : ndarray of shape (size, size)
        Perturbed matrix near an EP.
    """
    if rng is None:
        rng = np.random.default_rng()

    H = np.zeros((size, size), dtype=complex)

    # Jordan block of given order
    for i in range(order - 1):
        H[i, i + 1] = 1.0

    # Add perturbation
    for i in range(order):
        H[i, i] += epsilon * rng.normal()

    # Fill remaining dimensions with random diagonal entries
    for i in range(order, size):
        H[i, i] = rng.normal() + 1j * rng.normal()

    return H
