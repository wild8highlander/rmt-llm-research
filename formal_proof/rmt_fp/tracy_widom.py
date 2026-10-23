"""Tracy-Widom edge fluctuations with the Johnstone normalization.

Derived statement (monograph chapter 9, Step 13 of the program): for a
real white Wishart sample covariance ``C = (1/T) X X^T``, ``X`` an
``N x T`` matrix of i.i.d. ``N(0, sigma^2)`` entries,

    (lambda_max - mu_NT) / s_NT  -->  F_1        (beta = 1, real entries),

with Johnstone's centering and scaling

    mu_NT = sigma^2 (sqrt(T - 1) + sqrt(N))^2 / T,
    s_NT  = sigma^2 (sqrt(T - 1) + sqrt(N)) (1/sqrt(T - 1) + 1/sqrt(N))^{1/3} / T.

For *complex* entries the limit is ``F_2`` (beta = 2).  Real activations
of language models therefore follow the beta = 1 law; the beta = 2 law is
the complex-ensemble counterpart.  Both are provided here so the
distinction is verifiable numerically (``experiments/verify_tracy_widom.py``).

The Tracy-Widom reference samples themselves are generated from the
beta-ensemble matrix models (GOE / GUE largest eigenvalues standardized
by their asymptotic edge), which requires no external tables:

    GOE:  xi_1 = (lambda_max - 2 sqrt(N)) N^{1/6}   -> F_1,
    GUE:  xi_2 = (lambda_max - 2 sqrt(N)) N^{1/6}   -> F_2.

Known moments used as calibration anchors (Wikipedia / Bornemann tables):
    F1: mean -1.2065335745820, var 1.607781034581,  skew 0.29346452408
    F2: mean -1.7710868074110, var 0.8131947928329, skew 0.224084203610

References
----------
- Johnstone (2001), Ann. Statist. 29(2), 295-327.
- Tracy & Widom (1994), Comm. Math. Phys. 159, 151-174; (1996) 177, 727-754.
- El Karoui (2006), Ann. Statist. 34(6) (improved finite-N centering).
- Dumitriu & Edelman (2002), J. Math. Phys. 43, 5830 (beta-ensembles).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "johnstone_center_scale",
    "el_karoui_center_scale",
    "wishart_lambda_max",
    "goe_lambda_max",
    "gue_lambda_max",
    "tw_reference_sample",
    "TW_MEAN",
    "TW_VAR",
    "TW_SKEW",
]


TW_MEAN = {1: -1.2065335745820, 2: -1.7710868074110}
TW_VAR = {1: 1.607781034581, 2: 0.8131947928329}
TW_SKEW = {1: 0.29346452408, 2: 0.224084203610}


def johnstone_center_scale(n: int, p: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Johnstone (2001) centering/scaling for the largest eigenvalue.

    Parameters
    ----------
    n : int
        Number of samples ``T``.
    p : int
        Dimension ``N``.
    sigma2 : float
        Entry variance.

    Returns
    -------
    (mu, s) : tuple of floats
        Center and scale for ``C = (1/n) X X^T``.
    """
    rt = np.sqrt(n - 1.0)
    rp = np.sqrt(p)
    mu = sigma2 * (rt + rp) ** 2 / n
    s = sigma2 * (rt + rp) * (1.0 / rt + 1.0 / rp) ** (1.0 / 3.0) / n
    return float(mu), float(s)


def el_karoui_center_scale(n: int, p: int, sigma2: float = 1.0) -> tuple[float, float]:
    """El Karoui (2006) improved finite-N centering/scaling."""
    rt = np.sqrt(n - 0.5)
    rp = np.sqrt(p - 0.5)
    mu = sigma2 * (rt + rp) ** 2 / n
    s = sigma2 * (rt + rp) * (1.0 / rt + 1.0 / rp) ** (1.0 / 3.0) / n
    return float(mu), float(s)


def wishart_lambda_max(
    n_samples: int,
    n: int,
    p: int,
    sigma2: float = 1.0,
    complex_: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample the largest eigenvalue of ``C = (1/n) X X^T`` repeatedly.

    Parameters
    ----------
    n_samples : int
        Number of independent matrices.
    n, p : int
        Samples ``T`` and dimension ``N``.
    sigma2 : float
        Entry variance.
    complex_ : bool
        If True, use complex Gaussian entries (beta = 2 universality class).
    rng : numpy.random.Generator, optional
    """
    rng = rng or np.random.default_rng()
    out = np.empty(n_samples)
    for i in range(n_samples):
        if complex_:
            x = (rng.standard_normal((p, n)) + 1j * rng.standard_normal((p, n))) * np.sqrt(
                sigma2 / 2.0
            )
        else:
            x = rng.standard_normal((p, n)) * np.sqrt(sigma2)
        out[i] = np.linalg.eigvalsh(x @ x.T.conj() if complex_ else x @ x.T)[-1]
    return out


def goe_lambda_max(n_samples: int, p: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Largest eigenvalue of a ``p x p`` GOE matrix (off-diagonal var 1)."""
    rng = rng or np.random.default_rng()
    out = np.empty(n_samples)
    for i in range(n_samples):
        a = rng.standard_normal((p, p))
        h = (a + a.T) / np.sqrt(2.0)
        out[i] = np.linalg.eigvalsh(h)[-1]
    return out


def gue_lambda_max(n_samples: int, p: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Largest eigenvalue of a ``p x p`` GUE matrix (off-diagonal var 1)."""
    rng = rng or np.random.default_rng()
    out = np.empty(n_samples)
    for i in range(n_samples):
        a = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
        h = (a + a.conj().T) / 2.0
        out[i] = np.linalg.eigvalsh(h)[-1]
    return out


def tw_reference_sample(
    beta: int,
    n_samples: int,
    p: int = 160,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Standardized Tracy-Widom sample of index ``beta`` via beta-ensembles.

    The GOE/GUE largest eigenvalue is standardized by the asymptotic edge
    ``2 sqrt(p)`` and the ``p^{-1/6}`` scale, yielding samples whose law
    converges to ``F_beta`` as ``p`` grows.  ``p = 160`` already matches
    the known means to ~0.01 (verified in the experiment).
    """
    rng = rng or np.random.default_rng()
    lam = gue_lambda_max(n_samples, p, rng) if beta == 2 else goe_lambda_max(n_samples, p, rng)
    return (lam - 2.0 * np.sqrt(p)) * p ** (1.0 / 6.0)
