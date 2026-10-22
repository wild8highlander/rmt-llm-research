"""
features.temporal — динамика признаков в ходе генерации.

Предсказательные сигналы могут сидеть не в статике, а в том, КАК меняется
внутреннее состояние по мере порождения ответа:

  lam_traj_*  — траектория lambda_max ковариаций ПОТОКА состояний
                последнего слоя: на каждом шаге s берётся ковариация
                векторов последней позиции за скользящее окно из w
                последних шагов генерации (определена с шага w-1;
                первые шаги заполняются первым валидным значением);
  ent_traj_*  — траектория энтропии выходных распределений;
  ce_traj_*   — траектория self-CE выбранных токенов.

Наклон оценивается МНК по нормированному времени t in [0, 1].
"""

from __future__ import annotations

import numpy as np


_WINDOW = 8  # окно потока состояний (шагов генерации)


def _slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    t = np.linspace(0.0, 1.0, n)
    tm, ym = t.mean(), y.mean()
    denom = ((t - tm) ** 2).sum()
    if denom <= 1e-18:
        return 0.0
    return float(((t - tm) * (y - ym)).sum() / denom)


def state_lambda_trajectory(step_hidden_last: np.ndarray, window: int = _WINDOW) -> np.ndarray:
    """lambda_max ковариаций потока состояний последнего слоя по шагам.

    step_hidden_last: (steps, layers, H). На шаге s ковариация строится
    по векторам последней позиции за окно из min(window, s+1) шагов;
    значения для первых шагов (окно < 4) заполняются первым валидным.
    """
    if step_hidden_last.ndim != 3:
        raise ValueError("step_hidden_last must be (steps, layers, H)")
    steps, _layers, H = step_hidden_last.shape
    if H < 2:
        raise ValueError("hidden dim too small for covariance")
    stream = step_hidden_last[:, -1, :]  # (steps, H) — последний слой
    lam = np.full(steps, np.nan)
    for s in range(steps):
        lo = max(0, s - window + 1)
        W = stream[lo : s + 1]
        if W.shape[0] < 4:
            continue
        cov = np.cov(W.T)
        ev = np.linalg.eigvalsh(cov)
        lam[s] = float(max(ev[-1], 0.0))
    valid = np.where(~np.isnan(lam))[0]
    if len(valid) == 0:
        raise ValueError("no valid covariance window in state stream")
    lam[: valid[0]] = lam[valid[0]]
    return lam


def temporal_features(
    step_hidden_last: np.ndarray,
    ce_traj: np.ndarray,
    entropy_traj: np.ndarray,
) -> dict:
    """Временные признаки наблюдения (см. модульный докстринг)."""
    if step_hidden_last is None or step_hidden_last.ndim != 3:
        raise ValueError("step_hidden_last must be (steps, layers, H)")
    if len(ce_traj) < 2 or len(entropy_traj) < 2:
        raise ValueError("trajectories need at least 2 steps")

    lam = np.log1p(state_lambda_trajectory(step_hidden_last))
    return {
        "lam_traj_slope": _slope(lam),
        "lam_traj_mean": float(lam.mean()),
        "lam_traj_std": float(lam.std()),
        "ent_traj_slope": _slope(np.asarray(entropy_traj, dtype=float)),
        "ent_traj_delta": float(entropy_traj[-1] - entropy_traj[0]),
        "ce_traj_slope": _slope(np.asarray(ce_traj, dtype=float)),
        "ce_traj_std": float(np.asarray(ce_traj, dtype=float).std()),
    }
