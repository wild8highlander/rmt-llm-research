"""Marchenko-Pastur law with the rigorous Silverstein fixed-point equation.

This module implements the *derived* (not postulated) spectral bulk:

* support edges ``lambda_pm = sigma^2 (1 +- sqrt(q))^2``  (Step 5 of the
  formal-proof program);
* the Stieltjes transform of the limiting spectral distribution in closed
  form, obtained as the physical root of the self-consistent (Silverstein)
  quadratic  (Step 4);

    sigma^2 q z m(z)^2 + (z + sigma^2 (q - 1)) m(z) + 1 = 0,

  where ``m(z) = int (lambda - z)^{-1} dmu(lambda)`` and
  ``G(z) = -m(z) = (1/N) Tr (zI - C)^{-1}``.

The quadratic is equivalent to the classical Silverstein equation

    m(z) = int 1 / ( t (1 - q - q z m(z)) - z ) dH(t),   H = delta_{sigma^2},

(see monograph chapter 3 for the derivation and the equivalence proof).
Every routine is verified against direct resolvent simulation in
``experiments/verify_mp_law.py``.

References
----------
- Marchenko & Pastur (1967), Mat. Sb. 72.
- Silverstein (1995), Ann. Probab. 23(4) 1911-1922.
- Bai & Silverstein (2010), "Spectral Analysis of Large Dimensional
  Random Matrices", 2nd ed., Springer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "mp_edges",
    "mp_atom_weight",
    "mp_density",
    "mp_cdf",
    "mp_stieltjes",
    "mp_resolvent_G",
    "silverstein_residual",
]


def mp_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Support edges ``lambda_pm = sigma^2 (1 +- sqrt(q))^2`` of the MP law.

    Parameters
    ----------
    q : float
        Aspect ratio ``N / T`` (any positive value; for ``q > 1`` the law
        acquires an atom of weight ``1 - 1/q`` at zero).
    sigma2 : float
        Population variance, must be positive.

    Returns
    -------
    (lambda_minus, lambda_plus) : tuple of floats
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    s = np.sqrt(q)
    return sigma2 * (1.0 - s) ** 2, sigma2 * (1.0 + s) ** 2


def mp_atom_weight(q: float) -> float:
    """Weight of the point mass at zero (present iff ``q > 1``)."""
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    return max(0.0, 1.0 - 1.0 / q)


def mp_density(lam: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Continuous part of the Marchenko-Pastur density.

    ``rho(lambda) = sqrt((lambda_+ - lambda)(lambda - lambda_-))
                    / (2 pi sigma^2 q lambda)``

    on ``[lambda_-, lambda_+]`` and zero elsewhere.  For ``q > 1`` an atom
    of weight ``mp_atom_weight(q)`` sits at zero (not included here).
    """
    lam = np.asarray(lam, dtype=float)
    lamm, lamp = mp_edges(q, sigma2)
    rho = np.zeros_like(lam)
    mask = (lam > lamm) & (lam < lamp)
    if np.any(mask):
        x = lam[mask]
        rho[mask] = np.sqrt((lamp - x) * (x - lamm)) / (2.0 * np.pi * sigma2 * q * x)
    return rho


def mp_cdf(lam: ArrayLike, q: float, sigma2: float = 1.0, n: int = 20001) -> np.ndarray:
    """CDF of the MP law (including the zero atom, if any) by quadrature.

    Parameters
    ----------
    lam : array_like
        Evaluation points.
    q, sigma2 : float
        MP parameters.
    n : int
        Quadrature resolution over the support.
    """
    lam = np.asarray(lam, dtype=float)
    lamm, lamp = mp_edges(q, sigma2)
    grid = np.linspace(lamm, lamp, n)
    rho = mp_density(grid, q, sigma2)
    cdf_support = np.concatenate(
        ([0.0], np.cumsum((rho[1:] + rho[:-1]) * 0.5 * np.diff(grid)))
    )
    atom = mp_atom_weight(q)
    out = np.empty_like(lam)
    out[lam <= 0] = 0.0
    between = (lam > 0) & (lam < lamm)
    out[between] = atom
    mid = (lam >= lamm) & (lam < lamp)
    if np.any(mid):
        out[mid] = atom + np.interp(lam[mid], grid, cdf_support)
    out[lam >= lamp] = 1.0
    return out


def mp_stieltjes(z: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Limiting Stieltjes transform ``m(z) = int (lambda - z)^{-1} dmu``.

    Physical root of the Silverstein quadratic

        sigma^2 q z m^2 + (z + sigma^2 (q - 1)) m + 1 = 0,

    selected as ``m = (-(z + sigma^2 (q-1)) + s(z)) / (2 sigma^2 q z)``
    with the branch ``s(z) = sqrt(z - lambda_-) sqrt(z - lambda_+)`` taken
    as the product of principal square roots.  This branch is analytic on
    ``C \\ [lambda_-, lambda_+]``, satisfies ``Im m > 0`` on the upper
    half-plane, ``m ~ -1/z`` at infinity and ``m(z) < 0`` for real
    ``z > lambda_+``.

    Parameters
    ----------
    z : array_like (complex)
        Evaluation points off the support (or on it, in the boundary limit).
    q, sigma2 : float
        MP parameters.
    """
    z = np.asarray(z, dtype=complex)
    lamm, lamp = mp_edges(q, sigma2)
    b = z + sigma2 * (q - 1.0)
    sd = np.sqrt(z - lamm) * np.sqrt(z - lamp)
    return (-b + sd) / (2.0 * sigma2 * q * z)


def mp_resolvent_G(z: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """``G(z) = (1/N) Tr (z I_N - C)^{-1}`` in the limit, i.e. ``G = -m``.

    ``G > 0`` for real ``z > lambda_+``, ``G ~ 1/z`` at infinity.
    """
    return -mp_stieltjes(z, q, sigma2)


def silverstein_residual(z: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Residual of the Silverstein fixed-point equation at ``m = mp_stieltjes(z)``.

    The equation is  ``m = 1 / (sigma^2 (1 - q - q z m) - z)``;  the returned
    value is ``m * (sigma^2 (1 - q - q z m) - z) - 1`` (zero at the solution).
    Used by ``verify_mp_law.py`` as an internal consistency check of the
    closed form against the fixed-point form.
    """
    z = np.asarray(z, dtype=complex)
    m = mp_stieltjes(z, q, sigma2)
    return m * (sigma2 * (1.0 - q - q * z * m) - z) - 1.0
