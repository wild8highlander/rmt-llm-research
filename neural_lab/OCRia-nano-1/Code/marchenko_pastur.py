"""Marchenko-Pastur toolkit for the Gemma3-1B-IT RCDA kit.

Same primitives as the repository's src/rmt_llm/marchenko_pastur.py and
bbp_transition.py, packaged standalone so the intervention kit runs without
installing the package:

  * mp_bounds / mp_density            - the null hypothesis
  * fit_sigma2                        - robust sigma^2 from a real ESD
  * bbp_lambda_max                    - where a spike of strength theta lands
  * spike_scan                        - lambda_v vs lambda_+ and theta_v vs theta_c
  * tw_check                          - Tracy-Widom-flavoured spectral purity gate
                                        (bulk KS-distance vs the MP fit, top
                                        direction excluded as the intended spike)

Every number printed in the RCDA reports comes from these functions, not from
hand fitting.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Marchenko-Pastur density and support
# --------------------------------------------------------------------------

def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP support edges lambda_- / lambda_+ for aspect ratio q<=1."""
    q = min(q, 1.0)
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def mp_density(lam, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP density rho(lam) = sqrt((l+ - l)(l - l-)) / (2 pi sigma^2 q l)."""
    lam = np.asarray(lam, dtype=float)
    l_minus, l_plus = mp_bounds(q, sigma2)
    rho = np.zeros_like(lam)
    inside = (lam >= l_minus) & (lam <= l_plus) & (lam > 0)
    li = lam[inside]
    rho[inside] = np.sqrt((l_plus - li) * (li - l_minus)) / (
        2 * np.pi * sigma2 * q * li
    )
    return rho


def mp_cdf(lam, q: float, sigma2: float = 1.0, n_grid: int = 4000) -> np.ndarray:
    """MP CDF evaluated at lam (numeric integration of mp_density)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    l_minus, l_plus = mp_bounds(q, sigma2)
    grid = np.linspace(max(l_minus, 1e-12), l_plus, n_grid)
    rho = mp_density(grid, q, sigma2)
    cdf_grid = np.cumsum(rho) * (grid[1] - grid[0])
    out = np.zeros_like(lam)
    out[lam <= l_minus] = 0.0
    out[lam >= l_plus] = 1.0
    mid = (lam > l_minus) & (lam < l_plus)
    out[mid] = np.interp(lam[mid], grid, cdf_grid)
    return out


def wishart_eigs(W: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvalues of the sample-covariance matrix built from W (m x n).

    Returns (eigs, q) with eigs of (1/n) W W^T when m<=n (q = m/n), else of
    (1/m) W^T W (q = n/m) - always the n<=1 orientation of the MP law.
    """
    W = np.asarray(W, dtype=np.float64)
    m, n = W.shape
    if m <= n:
        lam = np.linalg.eigvalsh((W @ W.T) / n)
        q = m / n
    else:
        lam = np.linalg.eigvalsh((W.T @ W) / m)
        q = n / m
    return lam, min(q, 1.0)


def fit_sigma2(eigs: np.ndarray, trim_hi: float = 0.02) -> float:
    """Robust sigma^2: trimmed mean of the bulk (top `trim_hi` excluded so a
    BBP spike cannot inflate the fit)."""
    eigs = np.sort(np.asarray(eigs, dtype=float))
    k = int(len(eigs) * trim_hi)
    return float(np.mean(eigs[: len(eigs) - k])) if len(eigs) > k else float(np.mean(eigs))


def ks_distance(eigs: np.ndarray, q: float, sigma2: float) -> float:
    """KS distance between the empirical CDF of the bulk and the MP CDF."""
    e = np.sort(np.asarray(eigs, dtype=float))
    emp = np.arange(1, len(e) + 1) / len(e)
    theo = mp_cdf(e, q, sigma2)
    return float(np.max(np.abs(emp - theo)))


# --------------------------------------------------------------------------
# BBP transition
# --------------------------------------------------------------------------

def bbp_lambda_max(theta: float, q: float, sigma2: float = 1.0) -> float:
    """Position of the largest eigenvalue under a rank-one spike of strength
    theta: sigma^2 (1 + theta^2 / q) when theta > sqrt(q), else the upper edge."""
    q = min(q, 1.0)
    if theta > np.sqrt(q):
        return sigma2 * (1 + theta**2 / q)
    return sigma2 * (1 + np.sqrt(q)) ** 2


def spike_scan(W: np.ndarray, v: np.ndarray) -> dict:
    """The RCDA matrix scan on one weight matrix W with probe direction v.

    lambda_v = ||W^T v||^2      (Rayleigh quotient of the loop direction)
    lambda_+ = MP upper edge from the fitted bulk
    theta_v  = sqrt(lambda_v / sigma2) - normalised spike strength
    theta_c  = sqrt(q)          - the BBP critical strength

    supercritical <=> lambda_v / lambda_+ > 1 and theta_v > theta_c
    """
    W = np.asarray(W, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    lam, q = wishart_eigs(W)
    sigma2 = fit_sigma2(lam)
    l_minus, l_plus = mp_bounds(q, sigma2)
    # Rayleigh quotient of the probe direction on the SAME normalisation as the
    # Wishart spectrum.  RCDA convention: v lives on the OUTPUT side of the
    # writer tensor (the space the residual/feature state is written into),
    # so lambda_v = ||W^T v||^2 / n  with n = fan-in (the sample axis).
    # Kit patch (rectangular tensors): prefer the output-side branch when the
    # dimensions disambiguate - for square matrices this is identical to the
    # parent repository's original condition.
    m, n = W.shape
    if W.shape[0] == len(v) and W.shape[1] != len(v):
        Wv = W.T @ v          # output-side v on a rectangular tensor
    elif W.shape[1] == len(v):
        Wv = W.T @ v          # input-side v (parent behaviour for square W)
    elif W.shape[0] == len(v):
        Wv = W.T @ v
    else:
        Wv = W @ v
    lam_v = float(Wv @ Wv) / float(max(W.shape[0], W.shape[1]))
    theta_v = float(np.sqrt(max(lam_v, 0.0) / sigma2))
    theta_c = float(np.sqrt(q))
    return {
        "lambda_v": lam_v,
        "lambda_plus": float(l_plus),
        "lambda_ratio": float(lam_v / l_plus),
        "theta_v": theta_v,
        "theta_c": theta_c,
        "sigma2": sigma2,
        "q": q,
        "supercritical": bool(lam_v > l_plus and theta_v > theta_c),
    }


# --------------------------------------------------------------------------
# Spectral purity gate ("TW verdict")
# --------------------------------------------------------------------------

def tw_check(W: np.ndarray, ks_threshold: float = 0.05,
             spike_exclusion: float = 1.10) -> dict:
    """Spectral-purity verdict for a (possibly edited) weight matrix.

    The bulk of the edited matrix must still follow the MP law (KS distance to
    the fitted MP CDF below `ks_threshold`) after excluding the single top
    eigenvalue when it exceeds the fitted upper edge by `spike_exclusion` - the
    excluded direction is the *intended* edit direction (v), everything else
    must stay in the bulk. A matrix that fails picked up unstructured damage.
    """
    lam, q = wishart_eigs(W)
    lam_sorted = np.sort(lam)[::-1]
    sigma2 = fit_sigma2(lam_sorted)
    _, l_plus = mp_bounds(q, sigma2)
    excluded = None
    bulk = lam_sorted
    if lam_sorted[0] > spike_exclusion * l_plus:
        excluded = float(lam_sorted[0])
        bulk = lam_sorted[1:]
    ks = ks_distance(bulk, q, sigma2)
    return {
        "verdict": "pass" if ks < ks_threshold else "violation",
        "ks_distance": ks,
        "ks_threshold": ks_threshold,
        "sigma2": sigma2,
        "lambda_plus": float(l_plus),
        "lambda_max": float(lam_sorted[0]),
        "excluded_top_eig": excluded,
        "q": q,
        "n_bulk": int(len(bulk)),
    }
