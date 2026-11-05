"""
stage4_interpretability.heavy_tail — heavy-tailed self-regularization
diagnostics for weight matrices (Martin & Mahoney 2018-2021; the
WeightWatcher line of work).

For each weight matrix W (m x n): eigenvalues of W^T W (m <= n).
The tail of the empirical spectral density above xmin is fitted with a
power law  p(lam) ~ lam^(-alpha)  using the Clauset-Shalizi-Newman /
Alstott continuous MLE:

    alpha = 1 + n_tail / sum_i ln(x_i / xmin),

xmin chosen by minimizing the KS distance. Interpretation bands (MM):
  alpha < 2     — strongly self-regularized ("solid") layer;
  2 <= alpha < 6 — weakly regularized / borderline;
  alpha >= 6    — random-matrix-like (untrained) tail.

Stage-4 use: the per-layer alpha profile is an INDEPENDENT lens on the
same spectrum the activation gap_ratio sees; correlating alpha with
gap_ratio links the WeightWatcher view of training quality to the
runtime activation spike.
"""

from __future__ import annotations

import numpy as np


def powerlaw_alpha(eigs: np.ndarray, xmin_grid: np.ndarray | None = None) -> dict:
    """MLE power-law alpha above xmin (Alstott-Clauset). eigs must be > 0."""
    x = np.asarray(eigs, dtype=np.float64)
    x = x[x > 0]
    if len(x) < 10:
        return {"alpha": float("nan"), "xmin": float("nan"), "ks": float("nan"),
                "n_tail": int(len(x)), "note": "too few eigenvalues"}
    if xmin_grid is None:
        xmin_grid = np.unique(np.quantile(x, np.linspace(0.4, 0.95, 40)))
    best = (np.inf, np.nan)
    for xm in xmin_grid:
        tail = x[x >= xm]
        n = len(tail)
        if n < 5:
            continue
        alpha = 1.0 + n / np.log(tail / xm).sum()
        if alpha <= 1.5:
            continue
        # KS distance of the fitted power law above xm (normalized rank plot)
        s = np.sort(tail)
        cdf_model = 1.0 - (xm / s) ** (alpha - 1.0)
        cdf_data = (np.arange(1, n + 1) - 0.5) / n
        ks = float(np.max(np.abs(cdf_model - cdf_data)))
        if ks < best[0]:
            best = (ks, (float(alpha), float(xm)))
    if np.isinf(best[0]):
        return {"alpha": float("nan"), "xmin": float("nan"), "ks": float("nan"),
                "n_tail": 0, "note": "no valid fit"}
    alpha, xm = best[1]
    return {"alpha": alpha, "xmin": xm, "ks": float(best[0]),
            "n_tail": int((x >= xm).sum()), "n": int(len(x))}


def weight_eigs(W: np.ndarray) -> np.ndarray:
    """Eigenvalues of W^T W for (m, n) with m <= n (auto-transposed)."""
    W = np.asarray(W, dtype=np.float64)
    if W.shape[0] > W.shape[1]:
        W = W.T
    cov = W.T @ W / W.shape[0]
    return np.clip(np.linalg.eigvalsh(cov), 1e-18, None)


def mm_table(named_weights: dict[str, np.ndarray]) -> list[dict]:
    """Per-matrix MM diagnostics: alpha, xmin, KS, lambda_max, and the MM band."""
    rows = []
    for name, W in named_weights.items():
        eigs = weight_eigs(W)
        fit = powerlaw_alpha(eigs)
        band = ("strong" if fit["alpha"] < 2 else "weak" if fit["alpha"] < 6 else "random-like") if np.isfinite(fit["alpha"]) else "n/a"
        rows.append({
            "matrix": name,
            "alpha": fit["alpha"],
            "xmin": fit["xmin"],
            "ks": fit["ks"],
            "lambda_max": float(eigs.max()),
            "band": band,
            "shape": list(np.asarray(W).shape),
        })
    return rows


def correlate_alpha_with_gap(mm_rows: list[dict], gap_profile: list[float]) -> dict:
    """Pearson r between per-layer alpha (layer-ordered matrices) and the
    activation gap_ratio profile. Requires matrices named 'L{i}...'.

    A layer typically owns SEVERAL weight matrices (e.g. GPT-NeoX has
    dense_h_to_4h and dense_4h_to_h per layer); their alphas are AVERAGED
    into one per-layer value — overwriting would silently drop matrices."""
    by_layer: dict[int, list[float]] = {}
    for r in mm_rows:
        name = r["matrix"]
        if ".h." in name or ".layers." in name:
            try:
                layer = int(name.split(".h.")[-1].split(".")[0]) if ".h." in name else int(name.split(".layers.")[-1].split(".")[0])
            except ValueError:
                continue
            if np.isfinite(r["alpha"]):
                by_layer.setdefault(layer, []).append(r["alpha"])
    n = min(len(by_layer), len(gap_profile))
    if n < 3:
        return {"pearson_r": float("nan"), "note": f"only {n} aligned layers"}
    layers = sorted(by_layer)[:n]
    a = np.asarray([float(np.mean(by_layer[l])) for l in layers])
    b = np.asarray(gap_profile[:n])
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {"pearson_r": float("nan"), "note": "degenerate profile"}
    return {"pearson_r": float(np.corrcoef(a, b)[0, 1]), "n_layers": n,
            "matrices_per_layer": {str(l): len(v) for l, v in sorted(by_layer.items())}}
