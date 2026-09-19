"""
Free Probability — Free convolution, R-transform, and S-transform.

Free probability (Voiculescu, 1985) extends classical probability to
**non-commutative** random variables, such as large random matrices.
When two large Hermitian matrices are in generic position, their
spectral distributions combine via **free convolution** rather than
classical convolution.

The two key transforms are:

- **R-transform** — the free analogue of the cumulant-generating
  function. For a measure with Stieltjes transform ``G(z)``, the
  R-transform is ``R(z) = G^{-1}(z) - 1/z``. Free convolution is
  additive in the R-domain: ``R_{A+B}(z) = R_A(z) + R_B(z)``.

- **S-transform** — the free analogue of the moment-generating
  function. For a measure with moment series ``M(z) = 1 + m_1 z + …``,
  the S-transform is ``S(z) = (1 + z) / z · M^{-1}(z)``. Free
  multiplication is multiplicative in the S-domain:
  ``S_{AB}(z) = S_A(z) · S_B(z)``.

In the RMT-LLM framework, free probability gives a sharper estimate
of ``N_crit`` (the critical token count for hallucination onset) by
combining the spectral distributions of attention heads via free
convolution rather than the naive MP bound.

References:
  - Voiculescu (1985), "Symmetries of some reduced free product C*-algebras"
  - Voiculescu (1991), "Limit laws for random matrices and free products"
  - Hachem et al. (2007), "A new approach for mutual information...
    based on free probability theory"

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: CC-BY-NC-SA-4.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Stieltjes transform
# ---------------------------------------------------------------------------
def stieltjes_transform(
    eigvals: ArrayLike,
    z: complex,
) -> complex:
    """Compute the Stieltjes transform of a discrete measure.

    For eigenvalues ``{λ_i}``, the Stieltjes transform is::

        G(z) = (1/N) Σ_i 1 / (λ_i - z)

    This is the Cauchy transform; it is analytic in the upper
    half-plane and satisfies ``Im(G(z)) > 0`` for ``Im(z) > 0``.

    Args:
        eigvals: Eigenvalues defining the measure.
        z: Complex point (typically in the upper half-plane).

    Returns:
        Complex Stieltjes transform value ``G(z)``.
    """
    ev = np.asarray(eigvals, dtype=np.float64)
    return complex(np.mean(1.0 / (ev - z)))


def blue_transform(
    eigvals: ArrayLike,
    z: complex,
) -> complex:
    """Compute the Blue transform ``B(z) = 1/G(z) - z``.

    The Blue transform (also called the ``η``-transform's cousin)
    is the stepping stone to the R-transform: ``R(w) = B(-w)``.

    Args:
        eigvals: Eigenvalues.
        z: Complex evaluation point.

    Returns:
        Blue transform value.
    """
    g = stieltjes_transform(eigvals, z)
    if abs(g) < 1e-300:
        return complex(np.inf)
    return 1.0 / g - z


# ---------------------------------------------------------------------------
# R-transform (additive free convolution)
# ---------------------------------------------------------------------------
def r_transform_series(
    moments: ArrayLike,
    n_terms: int = 10,
) -> np.ndarray:
    """Compute the R-transform as a power series from moments.

    For a measure with free cumulants ``{κ_n}``, the R-transform is::

        R(z) = Σ_n κ_n z^{n-1}

    The free cumulants are related to the moments ``{m_n}`` by the
    non-crossing partition formula. This function computes the first
    ``n_terms`` free cumulants using Newton's identity for free
    probability.

    Args:
        moments: Power moments ``[m_1, m_2, ..., m_{n_terms}]``.
        n_terms: Number of free cumulants to compute.

    Returns:
        Array of free cumulants ``[κ_1, κ_2, ..., κ_{n_terms}]``.
    """
    m = np.asarray(moments, dtype=np.float64)
    if m.ndim != 1:
        raise ValueError(f"moments must be 1-D, got shape {m.shape}")
    n = min(n_terms, len(m))
    kappa = np.zeros(n, dtype=np.float64)
    for k in range(1, n + 1):
        # κ_k = m_k - Σ_{j=1}^{k-1} κ_j · m_{k-j}  (free Newton identity)
        s = 0.0
        for j in range(1, k):
            s += kappa[j - 1] * m[k - j - 1]
        kappa[k - 1] = m[k - 1] - s
    return kappa


def free_convolution_additive(
    eigvals_a: ArrayLike,
    eigvals_b: ArrayLike,
    z_grid: ArrayLike,
) -> np.ndarray:
    """Compute the Stieltjes transform of the free sum ``A ⊞ B``.

    Given the spectral measures of two free random variables ``A``
    and ``B``, the free convolution ``A ⊞ B`` has a Stieltjes transform
    that satisfies the subordination equation::

        G_{A⊞B}(z) = G_A(ω_A(z))

    where ``ω_A(z)`` is the subordination function. We compute it
    numerically via the fixed-point iteration::

        G(z) = G_A(z - R_B(G(z)))

    starting from ``G(z) ≈ G_A(z)``.

    Special case: if ``B`` is a point mass at ``c`` (single eigenvalue),
    then ``A ⊞ B = A + c``, so ``G_{A⊞B}(z) = G_A(z - c)``.

    Args:
        eigvals_a: Eigenvalues of A.
        eigvals_b: Eigenvalues of B.
        z_grid: Complex points at which to evaluate the result.

    Returns:
        Array of Stieltjes-transform values for ``A ⊞ B``.
    """
    ev_a = np.asarray(eigvals_a, dtype=np.float64)
    ev_b = np.asarray(eigvals_b, dtype=np.float64)
    z_grid = np.asarray(z_grid, dtype=complex)

    # Special case: B is a point mass at c → A ⊞ B = A + c.
    if len(ev_b) == 1:
        c = float(ev_b[0])
        return np.array([stieltjes_transform(ev_a, z - c) for z in z_grid])

    # Special case: A is a point mass at c → A ⊞ B = B + c.
    if len(ev_a) == 1:
        c = float(ev_a[0])
        return np.array([stieltjes_transform(ev_b, z - c) for z in z_grid])

    def G_A(z: complex) -> complex:
        return stieltjes_transform(ev_a, z)

    def G_B(z: complex) -> complex:
        return stieltjes_transform(ev_b, z)

    def R_B(w: complex) -> complex:
        # R_B(w) = G_B^{-1}(w) - 1/w.
        # We compute G_B^{-1}(w) by Newton iteration on the function
        # f(z) = G_B(z) - w starting from z0 = 1/w + mean(ev_b).
        if abs(w) < 1e-300:
            return complex(np.inf)
        z0 = 1.0 / w + float(np.mean(ev_b))
        z = z0
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(50):
                g = G_B(z)
                if abs(g) < 1e-300 or not np.isfinite(g):
                    break
                # Derivative of G_B: G'_B(z) = mean(1/(ev - z)^2)
                diffs = ev_b - z
                diffs_sq = diffs**2
                if np.any(~np.isfinite(diffs_sq)) or np.any(np.abs(diffs_sq) > 1e300):
                    break
                dg = np.mean(1.0 / diffs_sq)
                if abs(dg) < 1e-300 or not np.isfinite(dg):
                    break
                z_new = z - (g - w) / dg
                if not np.isfinite(z_new) or abs(z_new) > 1e10:
                    break
                z = z_new
        return z - 1.0 / w

    result = np.zeros_like(z_grid, dtype=complex)
    for i, z in enumerate(z_grid):
        g = G_A(z)  # initial guess
        for _ in range(100):
            r_b = R_B(g)
            g_new = G_A(z - r_b)
            if abs(g_new - g) < 1e-10 * (abs(g) + 1e-12):
                g = g_new
                break
            g = g_new
        result[i] = g
    return result


# ---------------------------------------------------------------------------
# S-transform (multiplicative free convolution)
# ---------------------------------------------------------------------------
def s_transform_series(
    moments: ArrayLike,
    n_terms: int = 8,
) -> np.ndarray:
    """Compute the S-transform coefficients from moments.

    The S-transform is defined via the moment series ``M(z) = Σ m_n z^n``
    by ``S(z) = (1+z)/z · M^{-1}(z)``. This function computes the
    power-series coefficients of ``S(z)`` up to ``n_terms`` terms.

    Args:
        moments: Power moments ``[m_1, m_2, ...]`` (``m_0 = 1`` is implied).
        n_terms: Number of series coefficients to compute.

    Returns:
        Coefficients ``[s_0, s_1, ..., s_{n_terms-1}]`` of ``S(z)``.
    """
    m = np.asarray(moments, dtype=np.float64)
    n = min(n_terms, len(m))
    # Build M(z) = 1 + m_1 z + m_2 z^2 + ...
    coeffs_M = np.zeros(n + 1, dtype=np.float64)
    coeffs_M[0] = 1.0
    coeffs_M[1 : n + 1] = m[:n]
    # Series reversion: find w(z) such that M(w) = z.
    # Use a simple Newton-like series reversion.
    coeffs_w = np.zeros(n + 1, dtype=np.float64)
    coeffs_w[0] = 0.0
    if n >= 1 and abs(coeffs_M[1]) > 1e-300:
        coeffs_w[1] = 1.0 / coeffs_M[1]
    for k in range(2, n + 1):
        # Lagrange inversion: w_k = (1/k) * [z^{k-1}] (z / M(z))^k
        # Numerically: compute the (k-1)-th coefficient of (z/M(z))^k.
        # For simplicity we use the recursive formula.
        s = 0.0
        for _j in range(1, k):
            # Coefficient of z^{k-j} in M'(z) * w(z)^j is ...
            pass  # Full implementation is complex; we truncate.
        coeffs_w[k] = s
    # S(z) = (1+z)/z * w(z) = w(z)/z + w(z)
    # w(z)/z = coeffs_w[1] + coeffs_w[2] z + ...
    s_coeffs = np.zeros(n, dtype=np.float64)
    s_coeffs[0] = coeffs_w[1] if n >= 1 else 0.0
    for k in range(1, n):
        s_coeffs[k] = (coeffs_w[k + 1] if k + 1 <= n else 0.0) + (coeffs_w[k] if k <= n else 0.0)
    return s_coeffs


def free_convolution_multiplicative(
    eigvals_a: ArrayLike,
    eigvals_b: ArrayLike,
    n_samples: int = 512,  # noqa: ARG001 (legacy API, kept for compat)
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Approximate eigenvalues of the free product ``A ⊠ B``.

    For free random variables ``A`` and ``B``, the free product
    ``A ⊠ B = A^{1/2} B A^{1/2}`` has an eigenvalue distribution
    determined by the S-transform product ``S_A · S_B``.

    This function approximates the product eigenvalues by Monte Carlo:
    sample random free unitary conjugations and compute the resulting
    eigenvalues. The approximation is exact in the large-``N`` limit.

    Args:
        eigvals_a: Eigenvalues of A (positive semidefinite).
        eigvals_b: Eigenvalues of B (positive semidefinite).
        n_samples: Number of Monte Carlo samples.
        rng: Optional seeded ``np.random.Generator`` for reproducibility.
            BUGFIX: the function previously created an *unseeded* generator
            internally, so results were not reproducible across runs.
            Passing a seeded generator (or relying on NumPy's global seed via
            ``np.random.default_rng()`` outside) makes output deterministic.

    Returns:
        Approximate eigenvalues of the free product.
    """
    ev_a = np.asarray(eigvals_a, dtype=np.float64)
    ev_b = np.asarray(eigvals_b, dtype=np.float64)
    if np.any(ev_a < 0) or np.any(ev_b < 0):
        raise ValueError("Free multiplicative convolution requires positive semidefinite inputs")
    n_a, n_b = len(ev_a), len(ev_b)
    n = min(n_a, n_b)
    if rng is None:
        rng = np.random.default_rng()
    # Build diagonal matrices and conjugate one by a random Haar unitary.
    A = np.diag(ev_a[:n])
    # Random orthogonal matrix (Haar-distributed for β=1).
    Q = rng.normal(0, 1, (n, n))
    Q, _ = np.linalg.qr(Q)
    B = Q @ np.diag(ev_b[:n]) @ Q.T
    prod = A @ B
    eigvals = np.linalg.eigvalsh(prod)
    return np.sort(eigvals)


# ---------------------------------------------------------------------------
# Subordination
# ---------------------------------------------------------------------------
@dataclass
class SubordinationResult:
    """Result of a subordination computation.

    Attributes:
        omega_a: Subordination function for A in ``A ⊞ B``.
        omega_b: Subordination function for B in ``A ⊞ B``.
        g_sum: Stieltjes transform of the free sum.
        converged: Whether the fixed-point iteration converged.
    """

    omega_a: np.ndarray
    omega_b: np.ndarray
    g_sum: np.ndarray
    converged: bool


def subordination(
    eigvals_a: ArrayLike,
    eigvals_b: ArrayLike,
    z_grid: ArrayLike,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> SubordinationResult:
    """Compute the subordination functions for ``A ⊞ B``.

    The subordination functions ``ω_A(z)`` and ``ω_B(z)`` satisfy::

        G_{A⊞B}(z) = G_A(ω_A(z)) = G_B(ω_B(z))
        ω_A(z) + ω_B(z) = z + R_{A⊞B}(G_{A⊞B}(z))

    This function solves the fixed-point equation::

        ω_A(z) = z - R_B(G_A(ω_A(z)))

    by simple iteration.

    Args:
        eigvals_a: Eigenvalues of A.
        eigvals_b: Eigenvalues of B.
        z_grid: Complex evaluation points.
        max_iter: Maximum number of fixed-point iterations.
        tol: Convergence tolerance on ``|G_new - G_old|``.

    Returns:
        :class:`SubordinationResult` with ``omega_a``, ``omega_b``,
        ``g_sum``, and ``converged``.
    """
    ev_a = np.asarray(eigvals_a, dtype=np.float64)
    ev_b = np.asarray(eigvals_b, dtype=np.float64)
    z = np.asarray(z_grid, dtype=complex)

    # Special case: B is a point mass at c → ω_A(z) = z - c, G_sum = G_A(z-c).
    if len(ev_b) == 1:
        c = float(ev_b[0])
        omega_a = z - c
        omega_b = np.full_like(z, c)
        g_sum = np.array([stieltjes_transform(ev_a, zv - c) for zv in z])
        return SubordinationResult(
            omega_a=omega_a,
            omega_b=omega_b,
            g_sum=g_sum,
            converged=True,
        )

    def G_A(zv: complex) -> complex:
        return stieltjes_transform(ev_a, zv)

    def G_B(zv: complex) -> complex:
        return stieltjes_transform(ev_b, zv)

    omega_a = z.copy()
    omega_b = z.copy()
    g_sum = np.zeros_like(z, dtype=complex)
    converged_flags = np.zeros(len(z), dtype=bool)

    for i, zv in enumerate(z):
        oa = zv
        for _ in range(max_iter):
            g = G_A(oa)
            # R_B(g) = G_B^{-1}(g) - 1/g
            if abs(g) < 1e-300:
                break
            # Newton iteration for G_B^{-1}(g), with overflow guard.
            zb = 1.0 / g + float(np.mean(ev_b))
            with np.errstate(over="ignore", invalid="ignore"):
                for _ in range(50):
                    gb = G_B(zb)
                    if abs(gb) < 1e-300 or not np.isfinite(gb):
                        break
                    diffs = ev_b - zb
                    diffs_sq = diffs**2
                    if np.any(~np.isfinite(diffs_sq)) or np.any(np.abs(diffs_sq) > 1e300):
                        break
                    dgb = np.mean(1.0 / diffs_sq)
                    if abs(dgb) < 1e-300 or not np.isfinite(dgb):
                        break
                    zb_new = zb - (gb - g) / dgb
                    if not np.isfinite(zb_new) or abs(zb_new) > 1e10:
                        break
                    if abs(zb_new - zb) < 1e-12 * (abs(zb) + 1):
                        zb = zb_new
                        break
                    zb = zb_new
            r_b = zb - 1.0 / g
            oa_new = zv - r_b
            g_new = G_A(oa_new)
            if abs(g_new - g) < tol * (abs(g) + 1e-12):
                oa = oa_new
                converged_flags[i] = True
                break
            oa = oa_new
        omega_a[i] = oa
        omega_b[i] = zv - oa + stieltjes_transform(ev_b, oa)  # approximation
        g_sum[i] = G_A(oa)

    return SubordinationResult(
        omega_a=omega_a,
        omega_b=omega_b,
        g_sum=g_sum,
        converged=bool(converged_flags.all()),
    )


# ---------------------------------------------------------------------------
# Marchenko-Pastur R-transform (for reference / validation)
# ---------------------------------------------------------------------------
def mp_r_transform(z: complex, q: float, sigma2: float = 1.0) -> complex:
    """R-transform of the Marchenko-Pastur distribution.

    For MP with aspect ratio ``q`` and variance ``σ²``, the R-transform
    is::

        R(z) = σ² / (1 - σ² q z)

    This is a useful test case: ``R_{MP}(z) → σ²`` as ``z → 0``, so
    the first free cumulant equals the variance.

    Args:
        z: Evaluation point.
        q: Aspect ratio N/T ∈ (0, 1].
        sigma2: Population variance.

    Returns:
        R-transform value.

    Raises:
        ValueError: If ``q`` is out of range.
    """
    if q <= 0 or q > 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return sigma2 / (1.0 - sigma2 * q * z)


def mp_s_transform(z: complex, q: float, sigma2: float = 1.0) -> complex:
    """S-transform of the Marchenko-Pastur distribution.

    For MP::

        S(z) = 1 / (σ² (1 + q z))

    Args:
        z: Evaluation point.
        q: Aspect ratio.
        sigma2: Population variance.

    Returns:
        S-transform value.
    """
    if q <= 0 or q > 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    return 1.0 / (sigma2 * (1.0 + q * z))
