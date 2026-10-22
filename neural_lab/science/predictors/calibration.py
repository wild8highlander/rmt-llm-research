"""
predictors.calibration — калибровка вероятностей и её измерение.

Высокий ROC-AUC ещё не означает, что «вероятность ошибки 80 %»
соответствует ~80 % ошибок в сопоставимой группе случаев. Здесь:

  platt_fit/apply  — Платт-масштабирование логитов на OOF-предсказаниях
                     (калибровка настраивается ТОЛЬКО на train-данных);
  ece              — expected calibration error (равно-массовые бины);
  reliability_bins — данные для кривой калибровки в отчёте.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Численно устойчивый сигмоид (без overflow на больших |z|)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def platt_fit(p: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """Платт-калибровка: sigmoid(a * logit(p) + b); возвращает (a, b)."""
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
    """Равно-массовые бины: (границы, среднее p, частота y, размер)."""
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
    """Expected calibration error на равно-массовых бинах."""
    rows = reliability_bins(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["p_mean"] - r["freq"]) for r in rows) / max(n, 1))
