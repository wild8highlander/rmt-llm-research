"""
features.temporal — feature dynamics during generation.

Predictive signals may live not in the statics but in HOW the internal
state changes as the answer is being produced:

  lam_traj_*  — the lambda_max trajectory of the covariances of the
                last-layer STATE STREAM: at each step s, the covariance
                is taken over last-position vectors within a sliding
                window of the last w generation steps (defined from step
                w-1 on; early steps are filled with the first valid
                value);
  ent_traj_*  — the entropy trajectory of the output distributions;
  ce_traj_*   — the self-CE trajectory of the chosen tokens.

The slope is estimated by OLS on normalized time t in [0, 1].
"""

from __future__ import annotations

import numpy as np


_WINDOW = 8  # state-stream window (generation steps)


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
    """lambda_max of last-layer state-stream covariances across steps.

    step_hidden_last: (steps, layers, H). At step s the covariance is
    built over last-position vectors within a window of min(window, s+1)
    steps; values for early steps (window < 4) are filled with the first
    valid one.
    """
    if step_hidden_last.ndim != 3:
        raise ValueError("step_hidden_last must be (steps, layers, H)")
    steps, _layers, H = step_hidden_last.shape
    if H < 2:
        raise ValueError("hidden dim too small for covariance")
    stream = step_hidden_last[:, -1, :]  # (steps, H) — last layer
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
    """Temporal features of an observation (see the module docstring)."""
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
