"""
predictors.calibration — probability calibration and its measurement.

A high ROC-AUC does not yet mean that "an 80 % error probability"
corresponds to ~80 % errors in a comparable group of cases. Here:

  platt_fit/apply  — Platt scaling of logits on OOF predictions
                     (calibration is tuned on TRAIN data only);
  ece              — expected calibration error (equal-mass bins);
  reliability_bins — the data for the calibration curve in the report.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (no overflow at large |z|)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def platt_fit(p: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """Platt calibration: sigmoid(a * logit(p) + b); returns (a, b)."""
    z = _logit(np.asarray(p, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        s = _sigmoid_stable(a * z + b)
        w = np.clip(s * (1 - s), 1e-10, None)
        grad = np.array([np.sum((y - s) * z), np.sum(y - s)])
        h = np.array(
            [
                [np.sum(w * z * z), np.sum(w * z)],
                [np.sum(w * z), np.sum(w)],
            ]
        ) + 1e-10 * np.eye(2)
        step = np.linalg.solve(h, grad)
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def platt_apply(p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * _logit(np.asarray(p, dtype=np.float64)) + b
    return _sigmoid_stable(z)


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Equal-mass bins: (boundaries, mean p, frequency of y, size)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows = []
    for idx in bins:
        if len(idx) == 0:
            continue
        rows.append(
            {
                "p_mean": float(p[idx].mean()),
                "freq": float(y[idx].mean()),
                "n": len(idx),
            }
        )
    return rows


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-mass bins."""
    rows = reliability_bins(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["p_mean"] - r["freq"]) for r in rows) / max(n, 1))
