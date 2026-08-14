"""
Circular Ensembles — COE, CUE, CSE.

The circular ensembles (Dyson, 1962) are probability distributions on
the **unitary group** U(N). Their eigenvalues lie on the unit circle
``|λ| = 1`` and model systems with a conserved quantum number
(periodic boundary conditions, Floquet systems, etc.).

The three ensembles are:

- **COE** (Circular Orthogonal Ensemble, β=1): invariant under
  conjugation by orthogonal matrices. Eigenvalues come in complex-
  conjugate pairs. Generated as ``U U^T`` where ``U`` is Haar-unitary.

- **CUE** (Circular Unitary Ensemble, β=2): invariant under
  conjugation by unitary matrices. The Haar measure on U(N) itself.

- **CSE** (Circular Symplectic Ensemble, β=4): invariant under
  conjugation by symplectic matrices. Generated as ``U U^D`` where
  ``U^D`` is the dual (quaternion transpose).

In the RMT-LLM framework, circular ensembles model the spectral
distribution of **attention matrices** viewed as quantum channels:
the eigenvalue phases encode the rotation induced by each attention
head on the token-basis.

References:
  - Dyson (1962), "The threefold way. Algebraic structure of symmetry
    groups and ensembles in quantum mechanics", J. Math. Phys.
  - Mezzadri (2007), "How to generate random matrices from the
    classical compact groups", Notices AMS.

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: CC-BY-NC-SA-4.0
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


CircularBeta = Literal[1, 2, 4]


# ---------------------------------------------------------------------------
# Haar-distributed unitary matrices
# ---------------------------------------------------------------------------
def haar_unitary(
    n: int, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a Haar-distributed unitary matrix.

    Uses the QR decomposition of a complex Ginibre matrix (Mezzadri,
    2007). The diagonal phases of ``R`` are corrected to ensure the
    Haar measure.

    Args:
        n: Matrix dimension.
        rng: Random number generator.

    Returns:
        Complex unitary matrix of shape ``(n, n)``.

    Raises:
        ValueError: If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if rng is None:
        rng = np.random.default_rng()
    # Complex Ginibre: entries ~ N(0, 1/√2) + i N(0, 1/√2).
    A = (rng.normal(0, 1, (n, n)) + 1j * rng.normal(0, 1, (n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(A)
    # Correct the QR decomposition for Haar measure.
    d = np.diag(R)
    ph = d / np.abs(d)
    U = Q * ph[np.newaxis, :]
    return U


def haar_orthogonal(
    n: int, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a Haar-distributed orthogonal matrix.

    Args:
        n: Matrix dimension.
        rng: Random number generator.

    Returns:
        Real orthogonal matrix of shape ``(n, n)``.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if rng is None:
        rng = np.random.default_rng()
    A = rng.normal(0, 1, (n, n))
    Q, R = np.linalg.qr(A)
    # Correct signs for Haar measure.
    d = np.diag(R)
    signs = np.sign(d)
    signs[signs == 0] = 1.0
    return Q * signs[np.newaxis, :]


# ---------------------------------------------------------------------------
# Circular ensemble generators
# ---------------------------------------------------------------------------
def circular_ensemble(
    n: int, beta: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a matrix from the circular β-ensemble.

    Args:
        n: Matrix dimension.
        beta: 1 (COE), 2 (CUE), or 4 (CSE).
        rng: Random number generator.

    Returns:
        Unitary matrix of shape ``(n, n)`` (or ``(2n, 2n)`` for β=4).
        For β=1 the result is symmetric unitary; for β=4 it is
        self-dual quaternion.

    Raises:
        ValueError: If ``beta`` is not in {1, 2, 4}.
    """
    if beta not in (1, 2, 4):
        raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
    if rng is None:
        rng = np.random.default_rng()

    if beta == 2:
        # CUE = Haar unitary.
        return haar_unitary(n, rng)
    if beta == 1:
        # COE = U U^T for Haar U.
        U = haar_unitary(n, rng)
        return U @ U.T
    # beta == 4
    # CSE: build a 2n x 2n symplectic unitary.
    # Use the quaternion representation: each entry is a 2x2 block
    # [[a, b], [-b*, a*]].
    U = haar_unitary(n, rng)
    Re = np.real(U)
    Im = np.imag(U)
    top = np.hstack([Re, -Im])
    bot = np.hstack([Im, Re])
    M = np.vstack([top, bot])
    # Self-dual: M = M^D where M^D = J M^T J^{-1}, J = [[0, I], [-I, 0]].
    # For the CSE we return U U^D.
    n2 = 2 * n
    J = np.zeros((n2, n2))
    J[:n, n:] = np.eye(n)
    J[n:, :n] = -np.eye(n)
    MD = J @ M.T @ J.T
    return M @ MD


def circular_eigenvalues(
    n: int, beta: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample eigenvalues (phases) from a circular β-ensemble.

    Args:
        n: Matrix dimension.
        beta: 1 (COE), 2 (CUE), or 4 (CSE).
        rng: Random number generator.

    Returns:
        Sorted complex eigenvalue phases on the unit circle
        ``|λ| = 1``, as angles in ``[0, 2π)``.
    """
    M = circular_ensemble(n, beta, rng)
    eigs = np.linalg.eigvals(M)
    phases = np.angle(eigs) % (2 * np.pi)
    return np.sort(phases)


# ---------------------------------------------------------------------------
# Spectral statistics
# ---------------------------------------------------------------------------
def nearest_neighbor_spacing(
    phases: ArrayLike, normalize: bool = True,
) -> np.ndarray:
    """Compute nearest-neighbor spacings on the unit circle.

    Args:
        phases: Sorted phases in ``[0, 2π)``.
        normalize: If True, rescale spacings to have mean 1.

    Returns:
        Array of spacings (length ``len(phases)`` for circular topology).
    """
    p = np.sort(np.asarray(phases, dtype=np.float64))
    n = len(p)
    if n < 2:
        return np.array([])
    # Circular spacing: include wrap-around.
    spacings = np.diff(p)
    wrap = (p[0] + 2 * np.pi) - p[-1]
    spacings = np.append(spacings, wrap)
    if normalize and spacings.mean() > 0:
        spacings = spacings / spacings.mean()
    return spacings


def form_factor(
    phases: ArrayLike, max_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the spectral form factor ``K(τ)``.

    The form factor is the Fourier transform of the two-level
    correlation function. For the circular ensembles, it has a
    universal large-N limit::

        K_β(τ) = τ - (β τ / 2) ln(1 + 2 τ / β)   for τ < 1
        K_β(τ) = 1                                  for τ ≥ 1

    (Here ``τ = k / N`` is the scaled wavenumber.)

    Args:
        phases: Sorted eigenvalue phases.
        max_k: Maximum wavenumber. Default: ``2 * N``.

    Returns:
        Tuple ``(tau, K)`` of arrays.
    """
    p = np.asarray(phases, dtype=np.float64)
    n = len(p)
    if max_k is None:
        max_k = 2 * n
    ks = np.arange(1, max_k + 1)
    # K(k) = (1/N) |Σ_j e^{i k θ_j}|^2
    K = np.zeros(len(ks), dtype=np.float64)
    for idx, k in enumerate(ks):
        s = np.sum(np.exp(1j * k * p))
        K[idx] = abs(s) ** 2 / n
    tau = ks / n
    return tau, K


def number_variance(
    phases: ArrayLike, max_L: float = 5.0, n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the number variance ``Σ²(L)``.

    The number variance measures the fluctuation of the count of
    eigenvalues in an interval of length ``L`` (in units of mean
    spacing). For Poisson statistics ``Σ²(L) = L``; for the circular
    ensembles it grows logarithmically (level repulsion).

    Args:
        phases: Sorted eigenvalue phases.
        max_L: Maximum interval length (in units of mean spacing).
        n_points: Number of ``L`` values.

    Returns:
        Tuple ``(L, Sigma2)`` of arrays.
    """
    p = np.sort(np.asarray(phases, dtype=np.float64))
    n = len(p)
    if n < 4:
        return np.array([]), np.array([])
    spacings = np.diff(p)
    mean_spacing = spacings.mean() if len(spacings) > 0 else 1.0
    if mean_spacing <= 0:
        return np.array([]), np.array([])

    Ls = np.linspace(0.1, max_L, n_points)
    variances = np.zeros(len(Ls))
    for idx, L in enumerate(Ls):
        # Count eigenvalues in sliding windows of length L * mean_spacing.
        window = L * mean_spacing
        counts = []
        for start in p:
            end = start + window
            # Handle circular wrap.
            shifted = (p - start) % (2 * np.pi)
            count = np.sum(shifted <= window)
            counts.append(count)
        variances[idx] = np.var(counts)
    return Ls, variances


# ---------------------------------------------------------------------------
# Theoretical predictions
# ---------------------------------------------------------------------------
def wigner_surmise_circular(s: ArrayLike, beta: int = 2) -> np.ndarray:
    """Wigner surmise for the circular ensembles.

    For the circular β-ensemble, the nearest-neighbor spacing
    distribution is::

        P_β(s) = c_β s^β exp(-a_β s²)

    (same functional form as the Gaussian ensembles, but with
    slightly different constants due to the compact domain).

    Args:
        s: Normalized spacings (mean 1).
        beta: 1, 2, or 4.

    Returns:
        Probability density.
    """
    if beta not in (1, 2, 4):
        raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
    s = np.asarray(s, dtype=np.float64)
    if beta == 1:
        a, b = np.pi / 2, np.pi / 4
    elif beta == 2:
        a, b = 32.0 / np.pi ** 2, 4.0 / np.pi
    else:
        a, b = (2 ** 18) / (3 ** 6 * np.pi ** 3), 64.0 / (9 * np.pi)
    return a * s ** beta * np.exp(-b * s ** 2)


def theoretical_form_factor(tau: ArrayLike, beta: int = 2) -> np.ndarray:
    """Theoretical large-N form factor for the circular β-ensemble.

    Args:
        tau: Scaled wavenumber ``k / N``.
        beta: 1, 2, or 4.

    Returns:
        Form factor values.
    """
    if beta not in (1, 2, 4):
        raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
    tau = np.asarray(tau, dtype=np.float64)
    K = np.where(
        tau < 1.0,
        tau - (beta * tau / 2.0) * np.log1p(2.0 * tau / beta),
        1.0,
    )
    return K


# ---------------------------------------------------------------------------
# Connection to attention matrices
# ---------------------------------------------------------------------------
def attention_phase_spectrum(
    attn_matrix: ArrayLike, normalize: bool = True,
) -> np.ndarray:
    """Compute the phase spectrum of an attention matrix.

    Attention matrices (after softmax) are real but not symmetric.
    Their eigenvalues are generally complex. When row-stochastic
    (each row sums to 1), the largest eigenvalue is 1, and the rest
    lie inside the unit disk. The *phases* of the subdominant
    eigenvalues encode the "rotation" induced by attention.

    This function extracts those phases and returns them sorted on
    ``[0, 2π)`` for comparison with circular-ensemble predictions.

    Args:
        attn_matrix: Attention weight matrix ``(T, T)``.
        normalize: If True, normalize the matrix to have spectral
            radius 1 before extracting phases.

    Returns:
        Sorted phases of the subdominant eigenvalues.
    """
    A = np.asarray(attn_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"attn_matrix must be square, got shape {A.shape}")
    eigs = np.linalg.eigvals(A)
    if normalize:
        radius = np.max(np.abs(eigs))
        if radius > 0:
            eigs = eigs / radius
    # Drop the dominant eigenvalue (closest to 1).
    idx_dom = np.argmax(np.abs(eigs))
    mask = np.ones(len(eigs), dtype=bool)
    mask[idx_dom] = False
    sub = eigs[mask]
    phases = np.angle(sub) % (2 * np.pi)
    return np.sort(phases)
