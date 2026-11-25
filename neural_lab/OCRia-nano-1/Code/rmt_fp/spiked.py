"""BBP spiked-covariance transition: threshold, outliers, overlaps.

Spiked population model (Johnstone / Baik-Ben Arous-Peche normalization):

    population  Sigma = sigma^2 I + sum_k mu_k v_k v_k^T,   mu_1 >= ... ,
    sample      C = (1/T) Sigma^{1/2} Z Z^T Sigma^{1/2},  q = N/T -> (0, inf).

Derived results (monograph chapters 5-6):

* BBP threshold            mu_c = sigma^2 sqrt(q)                          (Step 7)
* outlier location         lambda_k = (sigma^2 + mu_k) (1 + q sigma^2 / mu_k)
                           for mu_k > mu_c, else the eigenvalue sticks to
                           the bulk edge lambda_+                          (Step 8)
* eigenvector overlap      |<v_hat_k, v_k>|^2 = (1 - q sigma^4 / mu_k^2)
                                            / (1 + q sigma^2 / mu_k)       (Step 8)
* exact inverse estimator  mu_hat(lambda) = (A - sqrt(A^2 - 4 q sigma^4)) / 2,
                           A = lambda - sigma^2 (1 + q)                    (Step 14)

Consistency identities (checked numerically in ``experiments/``):
lambda_out(mu_c) = lambda_+, overlap(mu_c) = 0, overlap -> 1 as mu -> inf.

References
----------
- Baik, Ben Arous & Peche (2005), Ann. Statist. 33(5).
- Paul (2007), Ann. Statist. 35(4).
- Nadler (2008), Ann. Statist. 36(6) (real Gaussian case).
- Benaych-Georges & Nadler (2012), Random Matrices Theory Appl. 1(1).
- Bloemendal, Knowles, Lee & Vu (2016), Probab. Theory Relat. Fields 164.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .mp import mp_edges

__all__ = [
    "bbp_threshold",
    "bbp_outlier",
    "bbp_outlier_pole_equation",
    "bbp_overlap",
    "inverse_bbp_spike",
    "estimate_sigma2_from_bulk",
]


def bbp_threshold(q: float, sigma2: float = 1.0) -> float:
    """Critical additive spike strength ``mu_c = sigma^2 sqrt(q)``."""
    if q <= 0 or sigma2 <= 0:
        raise ValueError(f"q and sigma2 must be positive, got q={q}, sigma2={sigma2}")
    return sigma2 * np.sqrt(q)


def bbp_outlier(mu: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Deterministic equivalent of a spiked eigenvalue (additive spike ``mu``).

    ``lambda(mu) = (sigma^2 + mu) (1 + q sigma^2 / mu)``  for ``mu > mu_c``;
    for ``mu <= mu_c`` the spiked eigenvalue does not separate and the
    function returns the bulk edge ``lambda_+``.
    """
    mu = np.asarray(mu, dtype=float)
    lamm, lamp = mp_edges(q, sigma2)
    mu_c = bbp_threshold(q, sigma2)
    out = np.full_like(mu, lamp, dtype=float)
    sup = mu > mu_c
    if np.any(sup):
        m = mu[sup]
        out[sup] = (sigma2 + m) * (1.0 + q * sigma2 / m)
    return out


def bbp_outlier_pole_equation(mu: float, q: float, sigma2: float = 1.0) -> float:
    """Outlier location from the spiked-resolvent pole equation (numeric root).

    Solves  ``(sigma^2 + mu) (1 - q + q lambda G_bulk(lambda)) = lambda``
    for ``lambda > lambda_+``, where ``G_bulk`` is the MP resolvent.  Used
    by the verification experiment as an independent cross-check of
    ``bbp_outlier``; both must agree (they do, to machine precision, for
    every ``q`` tested).
    """
    if mu <= bbp_threshold(q, sigma2):
        return mp_edges(q, sigma2)[1]
    from scipy.optimize import brentq

    lamm, lamp = mp_edges(q, sigma2)

    def f(lam: float) -> float:
        z = complex(lam, 0.0)
        sd = np.sqrt(z - lamm) * np.sqrt(z - lamp)
        g = (z + sigma2 * (q - 1.0) - sd) / (2.0 * sigma2 * q * lam)
        return (sigma2 + mu) * (1.0 - q + q * lam * g.real) - lam

    hi = (sigma2 + mu) * (1.0 + q * sigma2 / mu)
    return brentq(f, lamp * (1.0 + 1e-9), hi * (1.0 + 1e-6), xtol=1e-12)


def bbp_overlap(mu: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Squared overlap of the sample and population eigenvectors.

    ``|<v_hat, v>|^2 = (1 - q sigma^4 / mu^2) / (1 + q sigma^2 / mu)``
    for ``mu > mu_c``; zero otherwise.
    """
    mu = np.asarray(mu, dtype=float)
    mu_c = bbp_threshold(q, sigma2)
    out = np.zeros_like(mu, dtype=float)
    sup = mu > mu_c
    if np.any(sup):
        m = mu[sup]
        out[sup] = (1.0 - q * sigma2**2 / m**2) / (1.0 + q * sigma2 / m)
    return out


def inverse_bbp_spike(lam: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Exact inverse-BBP estimator of the additive spike from an eigenvalue.

    Inverts ``lambda = (sigma^2 + mu)(1 + q sigma^2 / mu)``: the spike
    solves ``mu^2 - A mu + q sigma^4 = 0`` with ``A = lambda - sigma^2 (1+q)``,
    and the physical (large-root) branch is

        mu_hat(lambda) = (A + sqrt(A^2 - 4 q sigma^4)) / 2,

    defined for ``lambda >= lambda_+`` (returns 0 below the edge).  The
    first-order expansion is ``mu_hat = lambda - sigma^2 (1 + q)
    - q sigma^4 / lambda + O(mu^{-3})``; note the *minus* sign of the
    second-order term.
    """
    lam = np.asarray(lam, dtype=float)
    lamm, lamp = mp_edges(q, sigma2)
    a = lam - sigma2 * (1.0 + q)
    disc = a * a - 4.0 * q * sigma2**2
    out = np.zeros_like(lam)
    ok = (lam >= lamp) & (disc >= 0.0)
    if np.any(ok):
        out[ok] = 0.5 * (a[ok] + np.sqrt(disc[ok]))
    return out


def estimate_sigma2_from_bulk(eigs: np.ndarray, q: float, iters: int = 25) -> float:
    """Estimate the effective noise variance from the empirical bulk.

    Simple robust fixed-point fit: find ``sigma2`` whose MP edge ``lambda_+``
    matches a high quantile of the bulk eigenvalues, then refine with the
    moment identity ``E[lambda] = sigma2`` restricted to the bulk.  This is
    the operational way the phenomenological code attaches ``sigma2`` to
    real activation matrices; the estimator itself is heuristic and lives
    outside the derived part of the theory.
    """
    eigs = np.sort(np.asarray(eigs, dtype=float).ravel())
    n_fit = max(8, int(0.9 * eigs.size))
    bulk = eigs[:n_fit]
    sigma2 = max(float(np.mean(bulk)), 1e-12)
    for _ in range(iters):
        _, lamp = mp_edges(q, sigma2)
        inside = bulk[bulk <= lamp * (1.0 + 1e-9)]
        if inside.size < 8:
            break
        sigma2 = max(float(np.mean(inside)), 1e-12)
    return sigma2
