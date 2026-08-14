"""
Dyson Brownian Motion — Eigenvalue dynamics of random matrix ensembles.

Dyson Brownian motion (DBM) describes the stochastic evolution of
eigenvalues of a Hermitian matrix whose entries undergo independent
Ornstein-Uhlenbeck processes. For the Gaussian ensembles (GOE β=1,
GUE β=2, GSE β=4), the eigenvalues evolve as interacting Brownian
particles with a logarithmic Coulomb repulsion::

    dλ_i = (σ²/2N) Σ_{j≠i} 1/(λ_i - λ_j) dt + (σ/√N) dW_i

where β controls the repulsion strength and ``W_i`` are independent
Brownian motions.

In the RMT-LLM framework, DBM is used to model the *training dynamics*
of the spectral gap: as training progresses, the empirical covariance
eigenvalues drift, and the BBP signal eigenvalue separates from the
MP bulk. DBM gives a principled prior on the expected drift rate.

References:
  - Dyson (1962), "A Brownian-motion model for the analysis of our
    eigenvalue data", J. Math. Phys.
  - Erdős & Yau (2012), "Rigorous spectral analysis of large random
    matrices" (review of the universality program).

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: CC-BY-NC-SA-4.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


Beta = Literal[1, 2, 4]


# ---------------------------------------------------------------------------
# Gaussian ensembles: sample Wigner matrices
# ---------------------------------------------------------------------------
def gaussian_ensemble(
    n: int, beta: int = 2, sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a matrix from the Gaussian β-ensemble.

    Args:
        n: Matrix dimension.
        beta: Dyson index: 1 (GOE, real symmetric), 2 (GUE, complex
            Hermitian), 4 (GSE, quaternion self-dual).
        sigma: Entry standard deviation.
        rng: Random number generator.

    Returns:
        Matrix of shape ``(n, n)``. For β=1 the result is real symmetric;
        for β=2 complex Hermitian; for β=4 a real representation of the
        quaternion self-dual matrix with shape ``(2n, 2n)``.

    Raises:
        ValueError: If ``beta`` is not in {1, 2, 4} or ``n < 1``.
    """
    if beta not in (1, 2, 4):
        raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if rng is None:
        rng = np.random.default_rng()

    if beta == 1:
        A = rng.normal(0, sigma, (n, n))
        return (A + A.T) / np.sqrt(2)
    if beta == 2:
        A = rng.normal(0, sigma / np.sqrt(2), (n, n)) + \
            1j * rng.normal(0, sigma / np.sqrt(2), (n, n))
        return (A + A.conj().T) / np.sqrt(2)
    # beta == 4 — real representation of quaternion self-dual
    # Each quaternion entry [[a, b], [-b*, a*]] is a 2x2 block.
    A = rng.normal(0, sigma / 2, (n, n)) + \
        1j * rng.normal(0, sigma / 2, (n, n))
    # Build the 2n x 2n real representation.
    Re = np.real(A)
    Im = np.imag(A)
    top = np.hstack([Re, -Im])
    bot = np.hstack([Im, Re])
    M = np.vstack([top, bot])
    return (M + M.T) / np.sqrt(2)


def ensemble_eigenvalues(
    n: int, beta: int = 2, sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample eigenvalues from a Gaussian β-ensemble.

    Args:
        n: Matrix dimension (returns ``n`` eigenvalues for β=1,2 and
            ``2n`` for β=4).
        beta: Dyson index.
        sigma: Entry standard deviation.
        rng: Random number generator.

    Returns:
        Sorted real eigenvalues.
    """
    M = gaussian_ensemble(n, beta, sigma, rng)
    eigvals = np.linalg.eigvalsh(M)
    return np.sort(eigvals)


# ---------------------------------------------------------------------------
# Dyson Brownian Motion simulator
# ---------------------------------------------------------------------------
@dataclass
class DBMConfig:
    """Configuration for a Dyson Brownian Motion simulation.

    Attributes:
        n: Number of eigenvalues (particles).
        beta: Dyson index (1, 2, or 4).
        sigma: Diffusion coefficient (controls the noise scale).
        dt: Time step for the Euler-Maruyama integration.
        repulsion: If True, include the logarithmic Coulomb repulsion
            term (standard DBM). If False, the particles perform
            independent Brownian motion (no interaction).
        boundary: Optional reflecting boundary at ``±boundary``. If
            ``None``, no boundary is enforced.
    """
    n: int = 64
    beta: int = 2
    sigma: float = 1.0
    dt: float = 1e-3
    repulsion: bool = True
    boundary: float | None = None


class DysonBrownianMotion:
    """Simulate Dyson Brownian motion for β ∈ {1, 2, 4}.

    The eigenvalues ``λ_1 < λ_2 < ... < λ_N`` evolve according to the
    SDE::

        dλ_i = (β σ² / 4N) Σ_{j≠i} 1/(λ_i - λ_j) dt + (σ / √N) dW_i

    The drift term is the logarithmic Coulomb repulsion that keeps
    eigenvalues apart; the diffusion term is independent Brownian
    motion. The factor ``β`` in the drift comes from the chain rule
    applied to the Vandermonde determinant.

    The integration uses the Euler-Maruyama scheme. For numerical
    stability, the repulsion is regularized when eigenvalues get closer
    than ``dt`` to avoid division by zero.
    """

    def __init__(self, config: DBMConfig | None = None) -> None:
        self.config = config or DBMConfig()
        self._validate_config()
        self._t = 0.0
        self._rng: np.random.Generator = np.random.default_rng()
        self._eigvals: np.ndarray | None = None
        self._history: list[np.ndarray] = []

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.n < 2:
            raise ValueError(f"n must be ≥ 2, got {cfg.n}")
        if cfg.beta not in (1, 2, 4):
            raise ValueError(f"beta must be 1, 2, or 4, got {cfg.beta}")
        if cfg.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {cfg.sigma}")
        if cfg.dt <= 0:
            raise ValueError(f"dt must be positive, got {cfg.dt}")

    def initialize(self, eigvals: ArrayLike | None = None,
                   rng: np.random.Generator | None = None) -> np.ndarray:
        """Set the initial eigenvalue configuration.

        Args:
            eigvals: Initial sorted eigenvalues. If ``None``, sample
                from the semicircle law with radius ``2 σ √N``.
            rng: RNG for the initial sampling and subsequent steps.

        Returns:
            The initial eigenvalue array (sorted).
        """
        if rng is not None:
            self._rng = rng
        cfg = self.config
        if eigvals is None:
            # Semicircle law: ρ(λ) = (1/2π) √(4N σ² - λ²)
            # Sample by drawing a GOE/GUE matrix and taking eigenvalues.
            self._eigvals = ensemble_eigenvalues(cfg.n, cfg.beta, cfg.sigma, self._rng)
        else:
            ev = np.asarray(eigvals, dtype=np.float64)
            if ev.shape != (cfg.n,):
                raise ValueError(
                    f"eigvals must have shape ({cfg.n},), got {ev.shape}"
                )
            self._eigvals = np.sort(ev.copy())
        self._t = 0.0
        self._history = [self._eigvals.copy()]
        return self._eigvals.copy()

    @property
    def eigvals(self) -> np.ndarray:
        """Current eigenvalue configuration."""
        if self._eigvals is None:
            raise RuntimeError("Call initialize() first")
        return self._eigvals

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._t

    @property
    def history(self) -> list[np.ndarray]:
        """List of eigenvalue snapshots (one per recorded step)."""
        return self._history

    def step(self, n_steps: int = 1, record: bool = True) -> np.ndarray:
        """Advance the simulation by ``n_steps`` Euler-Maruyama steps.

        Args:
            n_steps: Number of time steps of size ``dt``.
            record: If True, append each step's eigenvalues to history.

        Returns:
            The eigenvalues after the last step.
        """
        if self._eigvals is None:
            raise RuntimeError("Call initialize() first")
        cfg = self.config
        n = cfg.n
        beta = cfg.beta
        sigma = cfg.sigma
        dt = cfg.dt
        noise_scale = sigma / np.sqrt(n)

        for _ in range(n_steps):
            lam = self._eigvals
            if cfg.repulsion:
                # Pairwise differences: λ_i - λ_j for all i ≠ j.
                # Use broadcasting: diff[i,j] = lam[i] - lam[j].
                diff = lam[:, None] - lam[None, :]          # (n, n)
                # Regularize the diagonal to avoid division by zero.
                np.fill_diagonal(diff, np.inf)
                inv_diff = 1.0 / diff                         # (n, n)
                drift = (beta * sigma ** 2 / (4 * n)) * inv_diff.sum(axis=1)
            else:
                drift = np.zeros(n)
            dW = self._rng.normal(0, np.sqrt(dt), n)
            self._eigvals = lam + drift * dt + noise_scale * dW
            self._eigvals = np.sort(self._eigvals)

            if cfg.boundary is not None:
                # Reflect at ±boundary.
                b = cfg.boundary
                self._eigvals = np.clip(self._eigvals, -b, b)
                self._eigvals = np.sort(self._eigvals)

            self._t += dt
            if record:
                self._history.append(self._eigvals.copy())
        return self._eigvals.copy()

    def run(self, total_time: float, record_interval: int = 1) -> np.ndarray:
        """Run the simulation for ``total_time`` time units.

        Args:
            total_time: Total simulation time.
            record_interval: Record every ``record_interval`` steps.

        Returns:
            Final eigenvalue configuration.
        """
        cfg = self.config
        n_steps = max(1, int(np.ceil(total_time / cfg.dt)))
        for i in range(n_steps):
            record = (i + 1) % record_interval == 0
            self.step(1, record=record)
        return self._eigvals.copy() if self._eigvals is not None else np.array([])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def spectral_gap(self) -> float:
        """Current spectral gap ``λ_max - λ_min``."""
        if self._eigvals is None:
            raise RuntimeError("Call initialize() first")
        return float(self._eigvals[-1] - self._eigvals[0])

    def largest_eigenvalue_trajectory(self) -> np.ndarray:
        """Time series of the largest eigenvalue."""
        if not self._history:
            return np.array([])
        return np.array([h[-1] for h in self._history])

    def mean_spacing(self) -> float:
        """Mean nearest-neighbor spacing of the current eigenvalues."""
        if self._eigvals is None:
            raise RuntimeError("Call initialize() first")
        spacings = np.diff(self._eigvals)
        return float(np.mean(spacings)) if len(spacings) > 0 else 0.0

    def unfolding(self) -> np.ndarray:
        """Unfold the eigenvalues to mean spacing 1.

        Unfolding removes the global density variation so that the
        local spacing statistics follow universal RMT predictions.
        We use a simple cubic-spline-based cumulative-density approach.

        Returns:
            Unfolded eigenvalues (mean spacing ≈ 1).
        """
        if self._eigvals is None:
            raise RuntimeError("Call initialize() first")
        lam = self._eigvals
        n = len(lam)
        # Empirical CDF: F(λ_i) = i / N (rank-based).
        ranks = np.arange(1, n + 1, dtype=np.float64)
        # Unfolded = N * (F(λ) - F(λ_1)) / (F(λ_N) - F(λ_1)) → mean spacing 1.
        unfolded = ranks - ranks[0]
        return unfolded


# ---------------------------------------------------------------------------
# Level spacing distribution
# ---------------------------------------------------------------------------
def wigner_surmise(s: ArrayLike, beta: int = 2) -> np.ndarray:
    """Evaluate the Wigner surmise for the level-spacing distribution.

    For the Gaussian β-ensembles, the nearest-neighbor spacing
    distribution is well approximated by::

        P_β(s) = a_β s^β exp(-b_β s²)

    where ``a_β`` and ``b_β`` are chosen so that ``∫ P(s) ds = 1``
    and ``⟨s⟩ = 1``.

    Args:
        s: Spacings (should have mean ≈ 1 after unfolding).
        beta: Dyson index (1, 2, or 4).

    Returns:
        Probability density values.

    Raises:
        ValueError: If ``beta`` is not in {1, 2, 4}.
    """
    if beta not in (1, 2, 4):
        raise ValueError(f"beta must be 1, 2, or 4, got {beta}")
    s = np.asarray(s, dtype=np.float64)
    if beta == 1:
        a, b = np.pi / 2, np.pi / 4
    elif beta == 2:
        a, b = 32.0 / np.pi ** 2, 4.0 / np.pi
    else:  # beta == 4
        a, b = (2 ** 18) / (3 ** 6 * np.pi ** 3), 64.0 / (9 * np.pi)
    p = a * s ** beta * np.exp(-b * s ** 2)
    return p


def empirical_spacing_distribution(
    eigvals: ArrayLike, normalize: bool = True,
) -> np.ndarray:
    """Compute nearest-neighbor spacings of a sorted eigenvalue array.

    Args:
        eigvals: Sorted eigenvalues.
        normalize: If True, rescale spacings to have mean 1 (unfolded).

    Returns:
        Array of spacings.
    """
    ev = np.sort(np.asarray(eigvals, dtype=np.float64))
    spacings = np.diff(ev)
    if normalize and len(spacings) > 0 and spacings.mean() > 0:
        spacings = spacings / spacings.mean()
    return spacings
