"""
stage2_detector.spectral_features — the RMT detector on HF hidden states.

Bridges `science.features.spectral` (the H1-validated spectral feature
extractor) to the hidden states of real HF models:

  for every decoder layer, covariance of the last-step hidden states
  across positions (T, H) -> eigenvalues -> MP-edge comparison:

    gap_ratio    = lambda_max / (sigma2 * (1+sqrt(q))^2)   (BBP outlier)
    effrank      = exp(entropy of the eigenvalue spectrum)
    ncrit_free   = kappa2 / kappa1^2                        (free N_crit est.)

  plus the LAST-TOKEN hidden vector per layer — the input space for the
  linear probes (Azaria & Mitchell protocol).
"""

from __future__ import annotations

import numpy as np

from science.features.spectral import covariance_spectral_features


def spectral_summary(final_hidden: list[np.ndarray]) -> dict:
    """Per-layer spectral features for one generation (see snapshot_features
    in science.features.spectral for the aggregation convention)."""
    per_layer = []
    for li, h in enumerate(final_hidden):
        f = covariance_spectral_features(np.asarray(h, dtype=np.float64), li)
        if f is not None:
            per_layer.append(f)
    if not per_layer:
        raise ValueError("no layer produced a valid covariance spectrum")
    out: dict[str, float] = {
        "gap_ratio_mean": float(np.mean([f["gap_ratio"] for f in per_layer])),
        "gap_ratio_last": per_layer[-1]["gap_ratio"],
        "gap_ratio_max": float(np.max([f["gap_ratio"] for f in per_layer])),
        "effrank_mean": float(np.mean([f["effective_rank"] for f in per_layer])),
        "effrank_last": per_layer[-1]["effective_rank"],
        "participation_mean": float(np.mean([f["participation_ratio"] for f in per_layer])),
        "ncrit_free_last": per_layer[-1]["ncrit_free"],
        "lambda_max_last": per_layer[-1]["lambda_max"],
        "spike_layer": int(int(np.argmax([f["gap_ratio"] for f in per_layer]))),
    }
    for f in per_layer:
        out[f"gap_ratio_L{f['layer']}"] = f["gap_ratio"]
        out[f"effrank_L{f['layer']}"] = f["effective_rank"]
    return out


def last_token_hidden(final_hidden: list[np.ndarray], prompt_len: int) -> np.ndarray:
    """(L, H) matrix of last-token hidden vectors (the probe input space).

    The last token is the LAST GENERATED token; positions are absolute, so
    the prompt length must be passed to slice the full-sequence tensors.
    """
    rows = []
    for h in final_hidden:
        t = h[min(prompt_len, h.shape[0]) - 1] if prompt_len > 0 else h[-1]
        rows.append(t)
    return np.stack(rows)


def spectral_feature_matrix(summaries: list[dict], include_per_layer: bool = False) -> tuple[np.ndarray, list[str]]:
    """Design matrix (n, d) over the summary keys with deterministic order.
    Returns (X, keys) — the key order is part of the feature contract."""
    if not summaries:
        raise ValueError("empty summary list")
    keys = [
        k
        for k in summaries[0]
        if not k.startswith(("gap_ratio_L", "effrank_L")) or include_per_layer
    ]
    keys = sorted(keys)
    return np.stack([[s[k] for k in keys] for s in summaries]).astype(np.float64), keys
