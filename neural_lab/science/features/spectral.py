"""
features.spectral — spectral features of hidden-state covariances.

For each layer, a covariance of representations across positions (T, H)
is built -> eigenvalues -> comparison against the Marchenko–Pastur edge:

  q_ratio    = H / T                    (covariance shape)
  mp_upper   = sigma2 * (1 + sqrt(q))^2 (upper edge of the bulk)
  gap_ratio  = lambda_max / mp_upper    (>1.05 — a "BBP outlier" beyond the bulk)

Additionally: the effective rank (entropy exponent of the spectrum),
the participation ratio, and a free estimate of N_crit (kappa2/kappa1^2).
Only documented tensors, no synthetic data.
"""

from __future__ import annotations

import numpy as np


def covariance_spectral_features(h: np.ndarray, layer: int) -> dict | None:
    """Spectral features of one layer; None if there are fewer than 4 positions.

    Convention for rank-deficient covariances (T < H — the usual case for
    hidden states): sigma2 = trace(cov)/H — the TOTAL data variance,
    including the null directions; eigenvalues are filtered only for bulk
    statistics (effective rank and similar). Validated against control
    nulls: for white noise gap_ratio ∈ [0.85, 1.15]
    (see tests/test_spectral.py).
    """
    if h is None or h.ndim != 2 or h.shape[0] < 4 or h.shape[1] < 2:
        return None
    T, H = h.shape
    cov = np.cov(h.T)
    sigma2 = float(np.trace(cov) / H)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-12]
    if len(ev) == 0 or sigma2 <= 1e-18:
        return None
    lam_max = float(ev[-1])
    q_ratio = H / float(T)
    mp_upper = sigma2 * (1.0 + np.sqrt(q_ratio)) ** 2
    p = ev / ev.sum()
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    participation = float(ev.sum() ** 2 / max((ev**2).sum(), 1e-18))
    kappa1 = float(ev.mean())
    kappa2 = float((ev**2).mean() - kappa1**2)
    ncrit_free = kappa2 / max(kappa1**2, 1e-18)
    return {
        "layer": layer,
        "T": int(T),
        "H": int(H),
        "q_ratio": q_ratio,
        "lambda_max": lam_max,
        "sigma2": sigma2,
        "mp_upper": float(mp_upper),
        "gap_ratio": lam_max / max(mp_upper, 1e-18),
        "spectral_gap": lam_max - float(ev[0]),
        "effective_rank": eff_rank,
        "participation_ratio": participation,
        "ncrit_free": ncrit_free,
    }


def snapshot_features(final_hidden: list[np.ndarray]) -> dict:
    """Per-layer aggregate of the final generation step.

    Returns a dict with gap_ratio_L{i}, effrank_L{i} for every layer plus
    the summary mean/last quantities. The number of layers is inferred
    from the data.
    """
    per_layer = []
    for li, h in enumerate(final_hidden):
        f = covariance_spectral_features(h, li)
        if f is not None:
            per_layer.append(f)
    if not per_layer:
        raise ValueError("no layer produced a valid covariance spectrum")
    out: dict[str, float] = {}
    for f in per_layer:
        out[f"gap_ratio_L{f['layer']}"] = f["gap_ratio"]
        out[f"effrank_L{f['layer']}"] = f["effective_rank"]
    out["gap_ratio_mean"] = float(np.mean([f["gap_ratio"] for f in per_layer]))
    out["gap_ratio_last"] = per_layer[-1]["gap_ratio"]
    out["effrank_mean"] = float(np.mean([f["effective_rank"] for f in per_layer]))
    out["effrank_last"] = per_layer[-1]["effective_rank"]
    out["participation_mean"] = float(np.mean([f["participation_ratio"] for f in per_layer]))
    out["ncrit_free_last"] = per_layer[-1]["ncrit_free"]
    out["lambda_max_last"] = per_layer[-1]["lambda_max"]
    return out
