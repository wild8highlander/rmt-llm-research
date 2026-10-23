"""Spectral filtering operators with exact optimality statements.

Step 14 of the program distinguishes three *different* problems, each with
its own exact solution (monograph chapter 10):

1. Best rank-r approximation of the *empirical* covariance in Frobenius
   norm  --  solved exactly by hard truncation (Eckart-Young-Mirsky
   theorem): keep the top-r sample eigenvalues unchanged.

2. Estimation of the *population* spike from an observed eigenvalue above
   the MP edge  --  solved exactly by the inverse-BBP soft shrinker
   (algebraic inversion of the Step 8 formula).

3. Minax hard thresholding of singular values under the oracle Frobenius
   risk  --  solved by the Gavish-Donoho threshold; for the eigenvalues of
   C the threshold is ``lambda*_GD = sigma^2 * tau*(q)^2`` with

       tau*(q) = sqrt( 2 (q + 1) + 8 q / (q + 1 + sqrt(q^2 + 14 q + 1)) ).

   Sanity anchor: tau*(1) = 4 / sqrt(3) (the celebrated 4/sqrt(3) result).

All operators act on the full eigenspectrum and return a filtered copy.
Relative risks are compared empirically in ``experiments/verify_shrinkage.py``.

References
----------
- Eckart & Young (1936), Psychometrika 1; Mirsky (1960), Num. Math. 86.
- Gavish & Donoho (2014), IEEE Trans. Inform. Theory 60(8), 5040-5053.
- Donoho & Gavish (2018), Comm. Pure Appl. Math. (optimal shrinkage).
"""

from __future__ import annotations

import numpy as np

from .mp import mp_edges
from .spiked import inverse_bbp_spike

__all__ = [
    "hard_truncation",
    "hard_mp_threshold",
    "inverse_bbp_shrinker",
    "first_order_inverse_shrinker",
    "gavish_donoho_tau",
    "gavish_donoho_threshold",
    "frobenius_relative_error",
]


def _apply(eigs: np.ndarray, fn) -> np.ndarray:
    out = np.array(fn(np.asarray(eigs, dtype=float)), dtype=float)
    return np.maximum(out, 0.0)


def hard_truncation(eigs: np.ndarray, r: int) -> np.ndarray:
    """Eckart-Young solution of  min ||C_emp - C_hat||_F  s.t. rank <= r."""
    eigs = np.asarray(eigs, dtype=float).ravel()
    if not 0 <= r <= eigs.size:
        raise ValueError(f"rank r={r} out of range for {eigs.size} eigenvalues")
    out = np.zeros_like(eigs)
    order = np.argsort(eigs)[::-1]
    out[order[:r]] = eigs[order[:r]]
    return out


def hard_mp_threshold(eigs: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Keep eigenvalues above the MP edge, zero the rest (rank-free MP clip)."""
    _, lamp = mp_edges(q, sigma2)
    eigs = np.asarray(eigs, dtype=float)
    return np.where(eigs > lamp, eigs, 0.0)


def inverse_bbp_shrinker(eigs: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Exact inverse-BBP estimator of the population eigenvalues above the edge."""
    eigs = np.asarray(eigs, dtype=float)
    return _apply(eigs, lambda l: inverse_bbp_spike(l, q, sigma2) + sigma2)


def first_order_inverse_shrinker(eigs: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """First-order de-biased estimator of the population eigenvalue.

    Large-spike expansion of the exact inverse-BBP operator, estimating the
    same object (``sigma^2 + mu``) as :func:`inverse_bbp_shrinker`:

        sigma^2 + mu_hat ~ lambda - q sigma^2 - q sigma^4 / lambda,

    for ``lambda > lambda_+``; the second-order term has a *minus* sign,
    matching the expansion of the exact inverse-BBP formula.
    """

    def fn(l: np.ndarray) -> np.ndarray:
        _, lamp = mp_edges(q, sigma2)
        # above the edge: first-order de-biased spike estimate + bulk level;
        # below the edge: the bulk level sigma^2 itself (as in the exact inverse)
        out = np.where(
            l > lamp,
            l - q * sigma2 - q * sigma2**2 / np.maximum(l, 1e-300),
            sigma2,
        )
        return out

    return _apply(np.asarray(eigs, dtype=float), fn)


def gavish_donoho_tau(q: float) -> float:
    """Optimal hard threshold for *singular values* (in units of sigma)."""
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    return float(np.sqrt(2.0 * (q + 1.0) + 8.0 * q / (q + 1.0 + np.sqrt(q**2 + 14.0 * q + 1.0))))


def gavish_donoho_threshold(eigs: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Gavish-Donoho optimal hard threshold translated to eigenvalues of C.

    The GD threshold applies to singular values of the data matrix X
    (entries of noise variance sigma2): keep s_i > sigma * tau*(q).  Since
    lambda_i = s_i^2 / T and the MP edge matches at leading order, the
    eigenvalue threshold is  lambda > sigma2 * tau*(q)^2.
    """
    eigs = np.asarray(eigs, dtype=float)
    lam_star = sigma2 * gavish_donoho_tau(q) ** 2
    return np.where(eigs > lam_star, eigs, 0.0)


def frobenius_relative_error(eigs_hat: np.ndarray, eigs_target: np.ndarray) -> float:
    """||diag(eigs_hat) - diag(eigs_target)||_F / ||diag(eigs_target)||_F."""
    eigs_hat = np.asarray(eigs_hat, dtype=float)
    eigs_target = np.asarray(eigs_target, dtype=float)
    n = max(eigs_hat.size, eigs_target.size)
    a = np.zeros(n)
    b = np.zeros(n)
    a[: eigs_hat.size] = np.sort(eigs_hat)[::-1]
    b[: eigs_target.size] = np.sort(eigs_target)[::-1]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))
