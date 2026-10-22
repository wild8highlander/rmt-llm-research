"""
evaluation.metrics — метрики предсказания ошибок (чистый NumPy).

Совместно оцениваются: ранжирование (ROC-AUC, PR-AUC), точность
вероятностей (Brier, log loss) и калибровка (ECE — в predictors).
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("y and p must be equal-length 1-D arrays")
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be binary")
    return y, p


def roc_auc(y, p) -> float:
    """Rank-based AUC с корректной обработкой связей (Mann–Whitney)."""
    y, p = _check(y, p)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC undefined: need both classes")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # средний ранг связки
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y, p) -> float:
    """PR-AUC = average precision (шаги по порогам при убывании p)."""
    y, p = _check(y, p)
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        raise ValueError("AP undefined: no positives")
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, _EPS)
    recall = tp / n_pos
    # AP = сумма precision * dR (учитывая связки: берем precision в конце связки)
    ap, prev_recall = 0.0, 0.0
    i = 0
    while i < len(ys):
        j = i
        while j + 1 < len(ys) and p[order][j + 1] == p[order][i]:
            j += 1
        ap += precision[j] * (recall[j] - prev_recall)
        prev_recall = recall[j]
        i = j + 1
    return float(ap)


def brier(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y, p = _check(y, p)
    pc = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
