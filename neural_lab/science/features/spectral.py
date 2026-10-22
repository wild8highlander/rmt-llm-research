"""
features.spectral — спектральные признаки ковариаций скрытых состояний.

Для каждого слоя строится ковариация представлений по позициям (T, H) ->
собственные значения -> сравнение с границей Марченко–Пастура:

  q_ratio    = H / T                    (форма ковариации)
  mp_upper   = sigma2 * (1 + sqrt(q))^2 (верхняя граница bulk)
  gap_ratio  = lambda_max / mp_upper    (>1.05 — «BBP-выброс» за bulk)

Дополнительно: эффективный ранг (экспонента энтропии спектра),
participation ratio и free-оценка N_crit (kappa2/kappa1^2).
Только документированные тензоры, без синтетики.
"""

from __future__ import annotations

import numpy as np


def covariance_spectral_features(h: np.ndarray, layer: int) -> dict | None:
    """Спектральные признаки одного слоя; None если позиций < 4.

    Конвенция для rank-дефицитных ковариаций (T < H — обычный случай
    скрытых состояний): sigma2 = trace(cov)/H — ПОЛНАЯ дисперсия данных,
    включая нулевые направления; собственные значения фильтруются только
    для статистик bulk (эффективный ранг и т.п.). Проверено
    контрольными нулями: у белого шума gap_ratio ∈ [0.85, 1.15]
    (см. tests/test_spectral.py).
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
    """Агрегат по слоям последнего шага генерации.

    Возвращает словарь gap_ratio_L{i}, effrank_L{i} для каждого слоя и
    сводные mean/last величины. Число слоёв выводится из данных.
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
