"""Layer operator calculus: RMSNorm, GELU/SiLU covariance kernels.

Step 2 of the program, derived part (monograph chapter 2).  Two results
are proved and implemented here:

1. RMSNorm trace conservation (exact, assumption-free lemma).  For

       Norm(x) = sqrt(N) x / ||x||

   every normalized vector lies on the sphere of radius sqrt(N), hence for
   the normalized activation matrix X (columns on the sphere)

       Tr C = (1/T) sum_t ||x_t||^2 = N     exactly,

   i.e. the mean eigenvalue of the covariance is pinned to 1.  This is a
   hard constraint that any MP-type fit to real activations must respect.

2. Nonlinear covariance kernels under the Gaussian assumption.  If the
   pre-activation is scalar Gaussian x ~ N(0, gamma^2), Price's theorem /
   Stein's lemma give closed forms for the GELU moments:

       E[GELU(x)]  = gamma^2 phi(0) / sqrt(1 + gamma^2),
       E[GELU(x)^2] = gamma^2 E[Phi(x)^2]
                      + gamma^4 [ 2 E[phi(x)^2] - 2 gamma^2 E[phi(x)^2]/(1+gamma^2) ],
       E[Phi(x)^2]  = 1/4 + arcsin(gamma^2/(1+gamma^2)) / (2 pi),
       E[phi(x)^2]  = (2 pi)^{-1} (1 + 2 gamma^2)^{-1/2},

   and the bivariate kernel  E[SiLU(u) SiLU(v)]  for a correlated Gaussian
   pair is evaluated by exact Gauss-Hermite quadrature.  These kernels are
   the microscopic replacement for postulated Gaussian ensembles: they
   transform the covariance deterministically, layer by layer.

The Gaussian assumption is stated explicitly: it is exact for linear
random-weights models and an approximation for trained networks.

References
----------
- Price (1958), IRE Trans. Inform. Theory 4, 69-72.
- Stein (1972), Proc. 6th Berkeley Symp. Math. Statist. Probab. 2.
- Hendrycks & Gimpel (2016), arXiv:1606.08415 (GELU); Shazeer (2020),
  arXiv:2002.05202 (SiLU / SwiGLU).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "rmsnorm",
    "rmsnorm_trace_check",
    "gelu",
    "silu",
    "gelu_moments",
    "bivariate_kernel",
    "gelu_covariance",
]


def rmsnorm(x: np.ndarray) -> np.ndarray:
    """``Norm(x) = sqrt(N) x / ||x||`` (column-wise for 2-D input)."""
    if x.ndim == 1:
        return np.sqrt(x.size) * x / np.linalg.norm(x)
    return np.sqrt(x.shape[0]) * x / np.linalg.norm(x, axis=0, keepdims=True)


def rmsnorm_trace_check(n_dims: int = 256, n_cols: int = 64, seed: int = 0) -> float:
    """Empirical check that Tr C = N after RMSNorm (returns the deviation)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_dims, n_cols)) * rng.uniform(0.1, 5.0, size=(1, n_cols))
    xn = rmsnorm(x)
    return float(abs(np.trace(xn @ xn.T) / n_cols - n_dims))


def gelu(x: ArrayLike) -> np.ndarray:
    """Gaussian-error linear unit  ``x * Phi(x)``."""
    from scipy.special import erf

    x = np.asarray(x, dtype=float)
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def silu(x: ArrayLike) -> np.ndarray:
    """Sigmoid-linear unit  ``x * sigmoid(x)``."""
    x = np.asarray(x, dtype=float)
    return x / (1.0 + np.exp(-x))


def gelu_moments(gamma2: float) -> tuple[float, float, float]:
    """Closed-form (E[GELU], E[GELU^2], Var) for x ~ N(0, gamma^2)."""
    g2 = float(gamma2)
    if g2 <= 0:
        raise ValueError(f"gamma2 must be positive, got {g2}")
    gamma = np.sqrt(g2)
    inv2pi = 1.0 / np.sqrt(2.0 * np.pi)
    e_gelu = g2 * inv2pi / np.sqrt(1.0 + g2)
    e_phi2 = 0.25 + np.arcsin(g2 / (1.0 + g2)) / (2.0 * np.pi)
    e_phi_sq = inv2pi**2 / np.sqrt(1.0 + 2.0 * g2)
    e_x_phiphi = g2 * e_phi_sq / (1.0 + g2)
    e_gelu2 = g2 * e_phi2 + g2**2 * (2.0 * e_phi_sq - 2.0 * e_x_phiphi)
    return float(e_gelu), float(e_gelu2), float(e_gelu2 - e_gelu**2)


def bivariate_kernel(
    fn,
    var_u: float,
    var_v: float,
    rho: float,
    n_points: int = 80,
) -> float:
    """E[fn(u, v)] for (u, v) centered bivariate normal via Gauss-Hermite.

    ``fn`` maps ``(u, v) -> float``.  Uses the Cholesky parameterization and
    the tensor-product Gauss-Hermite rule; exact for smooth integrands.
    """
    if abs(rho) > 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {rho}")
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_points)
    weights = weights / weights.sum()
    su, sv = np.sqrt(var_u), np.sqrt(var_v)
    total = 0.0
    for i in range(n_points):
        # u = su * x, v = sv * (rho x + sqrt(1-rho^2) y)
        u = su * nodes[i]
        vs = sv * (rho * nodes[i] + np.sqrt(1.0 - rho**2) * nodes)
        total += weights[i] * np.sum(weights * fn(u, vs))
    return float(total)


def gelu_covariance(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Covariance of y = GELU(W x) for x ~ N(0, C), via the bivariate kernel.

    Entry (a, b) uses the joint Gaussian of the two pre-activations
    (u_a, u_b) with variances ``w_a^T C w_a``, ``w_b^T C w_b`` and
    correlation ``w_a^T C w_b / sqrt(...)``.  O(K^2) kernel evaluations.
    """
    K = W.shape[0]
    var = np.einsum("kd,de,ke->k", W, C, W)
    cov_uv = W @ C @ W.T
    sd = np.sqrt(np.maximum(var, 1e-300))
    corr = cov_uv / np.outer(sd, sd)
    corr = np.clip(corr, -1.0 + 1e-12, 1.0 - 1e-12)
    out = np.empty((K, K))
    for a in range(K):
        for b in range(K):
            if a == b:
                _, _, var_g = gelu_moments(var[a])
                out[a, b] = var_g
            else:
                mean_a, _, _ = gelu_moments(var[a])
                mean_b, _, _ = gelu_moments(var[b])
                joint = bivariate_kernel(
                    lambda u, v: gelu(u) * gelu(v), var[a], var[b], corr[a, b]
                )
                out[a, b] = joint - mean_a * mean_b
    return out
