"""Entropy response of the decoder to Marchenko-Pastur bulk noise.

Derived second-order statement (monograph chapter 8; the *corrected* and
rigorous version of the information bridge):

Let the last-position logit vector be perturbed by bulk noise,

    z = z0 + W x_noise / tau,    x_noise ~ N(0, Sigma_noise),

where ``Sigma_noise`` has the Marchenko-Pastur bulk spectrum and ``tau``
is the softmax temperature.  Writing ``p = softmax(z0)`` and
``A = W Sigma_noise W^T / tau^2``, the conditional entropy of the next-token
distribution satisfies, for near-uniform baselines (|z0| small),

    E[H(softmax(z))] = H(p) - (1/2) Tr( (diag(p) - p p^T) A ) + O(tau^{-3}).

At the uniform baseline ``p = 1/K`` this reduces to

    E[H] = log K - (1/(2K)) [ Tr(A) - (1/K) 1^T A 1 ] + O(tau^{-3}),

and if the decoder rows are isotropic w.r.t. the noise (A = s2bar I / tau^2),

    E[H] = H0 - (1 - 1/K) s2bar / (2 tau^2) + O(tau^{-3}),
    s2bar = (1/K) Tr(W Sigma_noise W^T) -> sigma2_eff * <||w_i||^2>,

with ``sigma2_eff = int lambda rho_MP(lambda) dlambda`` the first spectral
moment of the bulk.  The correction is *negative*: each realized predictive
distribution is more peaked than the baseline (concavity of the entropy in
the logits).  The *mixture* (average predictive) distribution, by contrast,
flattens toward uniform; both readings are provided, and the Monte-Carlo
checker validates the conditional-entropy formula.

The key lemma (entropy Hessian at the uniform point) is elementary:

    d^2/dz_j dz_k  H(softmax z) = -[p_j (delta_jk - p_k)]  at z -> 0.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "entropy_logit_response",
    "entropy_response_uniform",
    "entropy_response_isotropic",
    "mc_conditional_entropy",
    "mc_mixture_entropy",
]


def entropy_logit_response(p: np.ndarray, A: np.ndarray) -> float:
    """Leading-order delta H = -(1/2) Tr((diag(p) - p p^T) A) (near-uniform p)."""
    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    hess = np.diag(p) - np.outer(p, p)
    return float(-0.5 * np.trace(hess @ A))


def entropy_response_uniform(W: np.ndarray, Sigma_noise: np.ndarray, tau: float) -> float:
    """Exact uniform-baseline formula with A = W Sigma W^T / tau^2."""
    K = W.shape[0]
    A = W @ Sigma_noise @ W.T / tau**2
    ones = np.ones(K)
    quad = float(ones @ A @ ones)
    return float(-(1.0 / (2.0 * K)) * (np.trace(A) - quad / K))


def entropy_response_isotropic(W: np.ndarray, Sigma_noise: np.ndarray, tau: float) -> float:
    """Isotropic-decoder closed form  -(1 - 1/K) s2bar / (2 tau^2)."""
    K = W.shape[0]
    s2bar = float(np.trace(W @ Sigma_noise @ W.T) / K)
    return -(1.0 - 1.0 / K) * s2bar / (2.0 * tau**2)


def mc_conditional_entropy(
    z0: np.ndarray,
    W: np.ndarray,
    Sigma_noise: np.ndarray,
    tau: float,
    n_trials: int = 4000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo estimate of E[H(softmax(z0 + W xi / tau))]."""
    rng = rng or np.random.default_rng()
    K = z0.size
    L = np.linalg.cholesky(Sigma_noise + 1e-12 * np.eye(Sigma_noise.shape[0]))
    total = 0.0
    for _ in range(n_trials):
        xi = L @ rng.standard_normal(Sigma_noise.shape[0])
        logit = z0 + W @ xi / tau
        logit -= logit.max()
        e = np.exp(logit)
        p = e / e.sum()
        total -= float(p @ np.log(p + 1e-300))
    return total / n_trials


def mc_mixture_entropy(
    z0: np.ndarray,
    W: np.ndarray,
    Sigma_noise: np.ndarray,
    tau: float,
    n_trials: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte-Carlo estimate of H(E[p]) -- entropy of the averaged prediction."""
    rng = rng or np.random.default_rng()
    L = np.linalg.cholesky(Sigma_noise + 1e-12 * np.eye(Sigma_noise.shape[0]))
    acc = np.zeros_like(z0)
    for _ in range(n_trials):
        xi = L @ rng.standard_normal(Sigma_noise.shape[0])
        logit = z0 + W @ xi / tau
        logit -= logit.max()
        e = np.exp(logit)
        acc += e / e.sum()
    p = acc / n_trials
    return float(-p @ np.log(p + 1e-300))
