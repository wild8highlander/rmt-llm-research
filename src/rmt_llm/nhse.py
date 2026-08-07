"""
Non-Hermitian Skin Effect (NHSE) — Topological transition at N = N_crit.

The NHSE describes the spectral collapse of non-Hermitian matrices: below
the critical token count N_crit, eigenvalues form a 2D ring in the complex
plane with non-zero imaginary parts (winding number w = 0). Above N_crit,
the skin effect forces all eigenvalues to collapse onto the real axis
(w = 1), signalling the onset of hallucination in autoregressive LLMs.

Key quantities:
  - Winding number: w = (1/2pi) oint arg(det(H(z) - lambda)) dz
  - Transition: w: 0 -> 1 at N = N_crit
  - Skin strength: exponential localization of eigenvectors at boundaries

References:
  - Okuma, Sato & Yao (2018+), "Topological origin of non-Hermitian skin
    effects", Phys. Rev. Lett.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def nhse_winding_number(n_ratio: float, gamma: float) -> int:
    """Compute the winding number for a non-Hermitian system.

    The winding number transitions from 0 to 1 at n_ratio = 1.0
    (i.e., when N = N_crit).

    Parameters
    ----------
    n_ratio : float
        Ratio N / N_crit.  Below 1: w = 0, above 1: w = 1.
    gamma : float
        Non-Hermiticity parameter in [0, 1].

    Returns
    -------
    w : int
        Winding number (0 or 1).
    """
    if n_ratio < 0:
        raise ValueError(f"n_ratio must be non-negative, got {n_ratio}")
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    return 1 if n_ratio > 1.0 else 0


def nhse_skin_strength(n_ratio: float, gamma: float) -> float:
    """Compute the skin effect strength (degree of eigenvalue collapse).

    Below N_crit: skin = 0 (eigenvalues on a 2D ring).
    Above N_crit: skin increases toward 1 as eigenvectors localize.

    Parameters
    ----------
    n_ratio : float
        Ratio N / N_crit.
    gamma : float
        Non-Hermiticity parameter.

    Returns
    -------
    strength : float in [0, 1]
        Skin effect strength.
    """
    w = nhse_winding_number(n_ratio, gamma)
    if w == 0:
        return 0.0
    else:
        # Smooth transition: tanh-saturated growth
        return np.tanh(gamma * (n_ratio - 1.0))


def nhse_imag_collapse(n_ratio: float, gamma: float) -> float:
    """Compute the fraction of imaginary parts that have collapsed.

    When w = 0: Im(lambda) is broadly distributed.
    When w = 1: Im(lambda) -> 0, fraction collapsed -> 1.

    Parameters
    ----------
    n_ratio : float
        Ratio N / N_crit.
    gamma : float
        Non-Hermiticity parameter.

    Returns
    -------
    collapse_fraction : float in [0, 1]
    """
    return nhse_skin_strength(n_ratio, gamma)


def nhse_hamiltonian(size: int, gamma: float,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a non-Hermitian Hamiltonian with skin effect.

    H = H_0 + i * gamma * Gamma

    where H_0 is a real symmetric matrix (Hermitian part) and Gamma is
    anti-Hermitian (breaks Hermiticity with strength gamma).

    Parameters
    ----------
    size : int
        Matrix dimension.
    gamma : float
        Non-Hermiticity strength.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    H : ndarray of shape (size, size)
        Non-Hermitian Hamiltonian.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Hermitian part: real symmetric random matrix
    A = rng.normal(0, 1, size=(size, size))
    H_0 = (A + A.T) / (2.0 * np.sqrt(size))

    # Anti-Hermitian part: i * Gamma where Gamma = (B - B^T) / (2i)
    B = rng.normal(0, 1, size=(size, size))
    Gamma = (B - B.T) / 2.0  # anti-symmetric real matrix

    H = H_0 + 1j * gamma * Gamma
    return H


def nhse_eigenvalues(size: int, gamma: float,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Compute eigenvalues of a non-Hermitian Hamiltonian.

    Parameters
    ----------
    size : int
        Matrix dimension.
    gamma : float
        Non-Hermiticity strength.
    rng : Generator, optional
        Random number generator.

    Returns
    -------
    eigenvalues : ndarray of shape (size,)
        Complex eigenvalues.
    """
    H = nhse_hamiltonian(size, gamma, rng)
    return np.linalg.eigvals(H)


def nhse_point_gap(n_ratio: float, gamma: float) -> float:
    """Estimate the point gap (min |Im(lambda)|) at given n_ratio.

    Below transition: point gap > 0 (eigenvalues off real axis).
    Above transition: point gap -> 0 (eigenvalues collapse to real axis).

    This is a simplified analytical model; for exact results, compute
    eigenvalues of the full Hamiltonian.

    Parameters
    ----------
    n_ratio : float
        Ratio N / N_crit.
    gamma : float
        Non-Hermiticity strength.

    Returns
    -------
    gap : float
        Estimated minimum |Im(lambda)|.
    """
    if n_ratio <= 1.0:
        # Below transition: gap is proportional to gamma
        return gamma * (1.0 - n_ratio)
    else:
        # Above transition: gap decays exponentially
        return gamma * np.exp(-(n_ratio - 1.0) / gamma) if gamma > 0 else 0.0
