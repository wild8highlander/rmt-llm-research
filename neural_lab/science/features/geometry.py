"""
features.geometry — representation geometry across layers.

A minimal set: representation norms per layer and drift between adjacent
layers (cosine similarity of last-position mean vectors). Complements the
spectral group without mixing in confidence features.
"""

from __future__ import annotations

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def geometry_features(final_hidden: list[np.ndarray]) -> dict:
    """final_hidden: list[layer] (T, H) — hidden states of the final step."""
    if not final_hidden or len(final_hidden) < 2:
        raise ValueError("geometry features need at least 2 layers")
    last_pos = np.stack([h[-1] for h in final_hidden])  # (layers, H)
    norms = np.linalg.norm(last_pos, axis=1)
    drifts = [_cos(last_pos[i], last_pos[i + 1]) for i in range(len(last_pos) - 1)]
    return {
        "rep_norm_mean": float(norms.mean()),
        "rep_norm_last": float(norms[-1]),
        "drift_mean": float(np.mean(drifts)),
        "drift_last": float(drifts[-1]),
    }
