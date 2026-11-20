"""
Marchenko-Pastur law — bulk eigenvalue density of random covariance matrices,
plus the RMT runtime used by the Gemma3-1B-IT RCDA-5 kit (z-scores in TW units,
BBP spike thresholds, spectral Tracy-Widom control of a weight edit).

For an N x T matrix X with i.i.d. entries of variance sigma^2, the empirical
spectral density of M = X X^T / T converges to the Marchenko-Pastur
distribution as N, T -> infinity with q = N/T fixed.

References:
  - Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets
    of random matrices", Mat. Sb.
  - Baik, Ben Arous & Peche (2005) — phase transition of the largest
    eigenvalue for spiked sample covariance matrices.
  - Johnstone (2001, 2008) — Tracy-Widom centering/scaling of the largest
    sample eigenvalue.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Standardized Tracy-Widom-1 moments (mean / sd of the TW1 distribution).
TW1_MEAN = -1.2068
TW1_SD = 1.2680


# --------------------------------------------------------------------------
# MP law (bulk)
# --------------------------------------------------------------------------

def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Support bounds (lambda_-, lambda_+) of the MP law for aspect ratio q in (0, 1]."""
    if q <= 0 or q > 1:
        raise ValueError(f"q must be in (0, 1], got {q}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    sqrt_q = np.sqrt(q)
    lam_minus = sigma2 * (1.0 - sqrt_q) ** 2
    lam_plus = sigma2 * (1.0 + sqrt_q) ** 2
    return lam_minus, lam_plus


def mp_density(lam: ArrayLike, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP density rho(lambda) at given eigenvalues (continuous part)."""
    lam = np.asarray(lam, dtype=float)
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    mask = (lam > lam_minus) & (lam < lam_plus) & (lam > 0)
    rho = np.zeros_like(lam)
    if np.any(mask):
        lam_m = lam[mask]
        rho[mask] = (
            1.0 / (2.0 * np.pi * sigma2 * lam_m * q)
            * np.sqrt((lam_plus - lam_m) * (lam_m - lam_minus))
        )
    return rho


def mp_cdf(lam: ArrayLike, q: float, sigma2: float = 1.0, n_points: int = 10000) -> np.ndarray:
    """Numerical CDF of the MP law."""
    lam = np.asarray(lam, dtype=float)
    lam_minus, lam_plus = mp_bounds(q, sigma2)
    eps = (lam_plus - lam_minus) * 1e-6
    grid = np.linspace(lam_minus + eps, lam_plus - eps, n_points)
    density = mp_density(grid, q, sigma2)
    dx = grid[1] - grid[0]
    cum = np.cumsum(density) * dx
    result = np.zeros_like(lam)
    for i, l in enumerate(lam):
        if l <= lam_minus:
            result[i] = 0.0
        elif l >= lam_plus:
            result[i] = 1.0
        else:
            idx = np.searchsorted(grid, l)
            if idx == 0:
                result[i] = 0.0
            elif idx >= len(grid):
                result[i] = 1.0
            else:
                frac = (l - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
                result[i] = cum[idx - 1] + frac * (cum[idx] - cum[idx - 1])
    return result


def mp_sample(n: int, t: int, sigma2: float = 1.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Eigenvalues of a sampled N x T Wishart matrix M = X X^T / T."""
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(sigma2)
    X = rng.normal(0, sigma, size=(n, t))
    M = X @ X.T / t
    return np.sort(np.linalg.eigvalsh(M))


def mp_stieltjes(z: complex, q: float, sigma2: float = 1.0) -> complex:
    """Stieltjes transform of the MP distribution (branch with Im g > 0)."""
    discriminant = (1.0 - q - z) ** 2 - 4.0 * q * z
    sqrt_disc = np.sqrt(discriminant)
    if sqrt_disc.imag < 0:
        sqrt_disc = -sqrt_disc
    return ((1.0 - q - z) + sqrt_disc) / (2.0 * sigma2 * q * z)


def mp_fit_sigma2(eigenvalues: ArrayLike, q: float) -> float:
    """Estimate the population variance sigma^2 as the median of the bulk.

    The median of the MP spectrum is a robust bulk-level estimator (the
    outlier spike only shifts the mean/upper tail, not the median).
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=float))
    return float(max(np.median(ev), 1e-12))


# --------------------------------------------------------------------------
# Tracy-Widom z-scores and BBP thresholds (runtime used by the RCDA kit)
# --------------------------------------------------------------------------

def johnstone_center_scale(n: int, t: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Johnstone (2001) centering/scaling of the largest eigenvalue of
    (1/t) X X^T for an n x t noise matrix with entry variance sigma^2.

    Returns (mu_np, sigma_np) such that (lam_max - mu_np) / sigma_np -> TW1.
    """
    n_eff, t_eff = float(n - 1), float(t)
    mu = sigma2 * (np.sqrt(t_eff) + np.sqrt(n_eff)) ** 2
    sigma_np = sigma2 * (np.sqrt(t_eff) + np.sqrt(n_eff)) * \
        (1.0 / np.sqrt(t_eff) + 1.0 / np.sqrt(n_eff)) ** (1.0 / 3.0)
    return float(mu), float(sigma_np)


def tw_units(z_raw: float) -> float:
    """Convert a raw Johnstone z ((lam - mu)/sigma_np, distributed as TW1)
    into standardized 'TW units' (approximately N(0, 1) under the null)."""
    return (z_raw - TW1_MEAN) / TW1_SD


def rmt_z_score(lam1: float, sigma2: float, n: int, t: int) -> float:
    """z-score (in TW units) of an observed top eigenvalue lam1 of (1/t)XX^T
    against the null of i.i.d. noise with variance sigma2 (n x t matrix)."""
    mu, s = johnstone_center_scale(n, t, sigma2)
    if s <= 0:
        return 0.0
    return tw_units((lam1 - mu) / s)


def bbp_spike_threshold(q: float, sigma2: float = 1.0) -> float:
    """BBP phase-transition threshold: a population spike stronger than this
    separates from the MP bulk edge lambda_+ (Baik-Ben Arous-Peche, 2005).
    For a rank-1 spike of strength s (s > threshold), the limiting top
    eigenvalue is s * (1 + q / (s / sigma2 - 1))."""
    return float(sigma2 * (1.0 + np.sqrt(q)) ** 2)


def rayleigh_spike_strength(lam1: float, lam_plus: float) -> float:
    """Relative prominence of a top eigenvalue above the MP edge:
    (lam1 - lam_plus) / lam_plus. Negative values mean 'inside the bulk'."""
    return float((lam1 - lam_plus) / max(lam_plus, 1e-12))


# --------------------------------------------------------------------------
# Spectral Tracy-Widom control of a weight edit (the RCDA gate)
# --------------------------------------------------------------------------

def spectral_tw_control(w_base: np.ndarray, w_edit: np.ndarray,
                        v: np.ndarray, tol: float = 4.0,
                        target_min_shift: float = 2.0) -> dict:
    """Tracy-Widom control of a weight edit W_edit vs W_base along direction v.

    The edit delta = W_edit - W_base is decomposed into
      * the target part  T = v_hat (v_hat^T delta)   — the component that
        writes along the measured loop direction v (output space), and
      * the residual   R = delta - T.

    Verdict semantics (mirrors the MNN RCDA kit):
      * "pass"      — the residual spectrum is statistically indistinguishable
                      from an i.i.d. random-matrix bulk (|z_resid| < tol), and
                      either the edit is a no-op or the target component sticks
                      out above the bulk edge (BBP shift >= target_min_shift
                      TW units). The edit is spectrally clean: all structured
                      power lives in the intended direction.
      * "violation" — the residual carries structured outliers outside v
                      (uncontrolled spectral damage), or the edit has no
                      target component at all (a "random burn" with the same
                      Frobenius mass but no directionality).

    Returns dict with: verdict, z_resid_tw, target_shift_tw_units,
    target_index, target_cos, sigma2_resid, lam_plus_resid, rel_frobenius.
    """
    wb = np.asarray(w_base, dtype=np.float64)
    we = np.asarray(w_edit, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).ravel()
    delta = we - wb
    m, n = delta.shape
    rel = float(np.linalg.norm(delta) / max(np.linalg.norm(wb), 1e-12))

    if float(np.linalg.norm(delta)) < 1e-12:
        return {"verdict": "pass", "z_resid_tw": 0.0,
                "target_shift_tw_units": 0.0, "target_index": -1,
                "target_cos": 0.0, "sigma2_resid": 0.0, "lam_plus_resid": 0.0,
                "rel_frobenius": 0.0}

    vh = v / max(np.linalg.norm(v), 1e-12)
    if vh.shape[0] != m:
        raise ValueError(f"v dim {vh.shape[0]} != matrix rows {m}")

    target = np.outer(vh, vh @ delta)          # P_v * delta (output space)
    resid = delta - target

    # ---- residual bulk (analyze the smaller-dimension side so q <= 1) ----
    transposed = m > n
    Rm = resid.T if transposed else resid
    Tm = target.T if transposed else target
    mm, nn = Rm.shape                          # nn = sample count
    lam_r = np.linalg.eigvalsh((Rm @ Rm.T) / max(nn, 1))
    lam_r = np.clip(lam_r, 0.0, None)
    q = mm / nn
    sigma2 = mp_fit_sigma2(lam_r, q)
    lam_plus = mp_bounds(q, sigma2)[1]
    z_resid = rmt_z_score(float(lam_r[-1]), sigma2, mm, nn)

    # ---- target component: how far the v-aligned part sticks out ----
    s_t = float(np.linalg.norm(Tm, ord=2))              # top singular value of T
    lam_t = s_t ** 2 / max(nn, 1)
    mu, s_np = johnstone_center_scale(mm, nn, sigma2)
    target_shift = tw_units((lam_t - mu) / s_np) if s_np > 0 else 0.0

    # ---- top left singular vector of the full edit vs v ----
    u, s, _ = np.linalg.svd(delta, full_matrices=False)
    target_index = -1
    target_cos = 0.0
    if u.shape[1]:
        coss = np.abs(u.T @ vh)                 # |cos| of each left sing. vec vs v
        target_index = int(np.argmax(coss))
        target_cos = float(coss[target_index])

    clean_bulk = abs(z_resid) < tol
    has_target = target_shift >= target_min_shift or target_cos >= 0.5
    verdict = "pass" if (clean_bulk and has_target) else "violation"

    return {"verdict": verdict,
            "z_resid_tw": round(float(z_resid), 3),
            "target_shift_tw_units": round(float(target_shift), 3),
            "target_index": target_index,
            "target_cos": round(target_cos, 4),
            "sigma2_resid": float(sigma2),
            "lam_plus_resid": float(lam_plus),
            "rel_frobenius": rel}
