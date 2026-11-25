"""Spectral definition of the hallucination split (Step 10: a definition).

This module implements the *definitional* content of the program: given a
covariance eigendecomposition and the bulk edge ``lambda_+``, the Hilbert
space of activations splits orthogonally into the signal subspace
(eigenvalues above the edge) and the bulk-noise subspace (eigenvalues
inside the MP band).  The normalized projector ratio

    r(x) = ||P_noise x||^2 / ||P_signal x||^2

is the operational quantity tracked by the phenomenological detectors in
``src/rmt_llm``.  Here it is provided as an exact linear-algebra utility
plus a synthetic-data harness, with the explicit caveat recorded in the
monograph: the split is a *definition* parameterized by ``eta_crit``; the
probabilistic statements about ``r(x)`` (Steps 11-12) are treated
separately and only the entropy response is derived rigorously.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spectral_split",
    "noise_signal_ratio",
    "hallucination_flag",
    "spiked_activation_matrix",
]


def spectral_split(eigs: np.ndarray, lambda_plus: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the signal (above edge) and noise (inside bulk) subspaces."""
    eigs = np.asarray(eigs, dtype=float)
    sig = np.where(eigs > lambda_plus)[0]
    noise = np.where((eigs >= 0) & (eigs <= lambda_plus))[0]
    return sig, noise


def noise_signal_ratio(
    x: np.ndarray,
    vecs: np.ndarray,
    eigs: np.ndarray,
    lambda_plus: float,
    eps: float = 1e-12,
) -> float:
    """``||P_noise x||^2 / ||P_signal x||^2`` from a full eigendecomposition.

    Parameters
    ----------
    x : (N,) vector (e.g. last-position activation).
    vecs : (N, N) eigenvectors as columns.
    eigs : (N,) eigenvalues.
    lambda_plus : float
        Bulk edge separating signal from noise.
    """
    sig, noise = spectral_split(eigs, lambda_plus)
    coeff_sig = vecs[:, sig].T @ x if sig.size else np.zeros(0)
    coeff_noise = vecs[:, noise].T @ x if noise.size else np.zeros(0)
    ps = float(coeff_sig @ coeff_sig)
    pn = float(coeff_noise @ coeff_noise)
    return pn / max(ps, eps)


def hallucination_flag(
    x: np.ndarray,
    vecs: np.ndarray,
    eigs: np.ndarray,
    lambda_plus: float,
    eta_crit: float = 1.0,
) -> bool:
    """Definition: x is flagged iff the noise projection dominates."""
    return bool(noise_signal_ratio(x, vecs, eigs, lambda_plus) > eta_crit)


def spiked_activation_matrix(
    n_dims: int,
    n_tokens: int,
    spikes: np.ndarray,
    sigma2: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic activation matrix with MP bulk plus rank-r semantic spikes.

    Returns ``(X, pop_spike_dirs, spike_strengths)`` where
    ``X = sigma * Z + sum_k sqrt(mu_k) v_k z_k^T`` mimics the
    signal-plus-bulk structure postulated for real LLM activations.
    """
    rng = np.random.default_rng(seed)
    spikes = np.asarray(spikes, dtype=float)
    z = rng.standard_normal((n_dims, n_tokens)) * np.sqrt(sigma2)
    dirs = np.linalg.qr(rng.standard_normal((n_dims, spikes.size)))[0]
    x = z.copy()
    for k, mu in enumerate(spikes):
        coeff = rng.standard_normal(n_tokens)
        coeff /= np.linalg.norm(coeff)
        x += np.sqrt(mu) * np.outer(dirs[:, k], coeff) * np.sqrt(n_tokens)
    return x, dirs, spikes
