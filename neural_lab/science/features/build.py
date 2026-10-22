"""
features.build — assembling the frozen feature vector of an observation.

The feature vector and its group breakdown are FROZEN in the H1 protocol
(EXPERIMENT_PROTOCOL.md, registry.yaml). Group composition:

  confidence (7) — standard confidence indicators (baseline group B);
  spectral  (15) — covariance spectrum of hidden states (MP gap, ranks);
  temporal   (7) — state/entropy/self-CE dynamics across steps;
  geometry   (4) — representation geometry across layers.

The internal group internal = spectral + temporal + geometry (26 at 4
layers) — "features requiring access to internal states". The combined
model D = confidence + internal (33 at 4 layers).

Protocol v1.1: per-layer spectral features generalize to any model depth
— the names L0..L{n-1} are built from the actual number of layers (the
v1.0 scheme is the special case n_layers=4). Vector size: 2*n_layers+25.
"""

from __future__ import annotations

import numpy as np

from .confidence import confidence_features
from .geometry import geometry_features
from .spectral import snapshot_features
from .temporal import temporal_features


LAYER_KEYS = (
    "gap_ratio_L0",
    "gap_ratio_L1",
    "gap_ratio_L2",
    "gap_ratio_L3",
    "effrank_L0",
    "effrank_L1",
    "effrank_L2",
    "effrank_L3",
)


def feature_groups(n_layers: int) -> dict[str, list[str]]:
    """Feature groups for a model of depth n_layers (protocol v1.1).

    For n_layers=4 this returns exactly the frozen v1.0 scheme (same name
    order — the as_matrix column order is pinned by the list). Per-layer
    names are L0..L{n-1}; the other groups are depth-invariant.
    """
    if n_layers < 2:
        raise ValueError(f"need >= 2 hidden layers, got {n_layers}")
    layer_keys = [f"gap_ratio_L{i}" for i in range(n_layers)] + [
        f"effrank_L{i}" for i in range(n_layers)
    ]
    groups: dict[str, list[str]] = {
        "confidence": [
            "ce_first",
            "ce_mean",
            "ppl_answer",
            "maxprob_first",
            "maxprob_mean",
            "entropy_mean",
            "margin_first",
        ],
        "spectral": [
            *layer_keys,
            "gap_ratio_mean",
            "gap_ratio_last",
            "effrank_mean",
            "effrank_last",
            "participation_mean",
            "ncrit_free_last",
            "lambda_max_last",
        ],
        "temporal": [
            "lam_traj_slope",
            "lam_traj_mean",
            "lam_traj_std",
            "ent_traj_slope",
            "ent_traj_delta",
            "ce_traj_slope",
            "ce_traj_std",
        ],
        "geometry": [
            "rep_norm_mean",
            "rep_norm_last",
            "drift_mean",
            "drift_last",
        ],
    }
    groups["internal"] = [n for g in ("spectral", "temporal", "geometry") for n in groups[g]]
    groups["combined"] = groups["confidence"] + groups["internal"]
    return groups


# Frozen scheme of protocol v1.0 (TinyGPT v3 S-cell, 4 layers).
# For a different depth use feature_groups(n_layers).
FEATURE_GROUPS: dict[str, list[str]] = feature_groups(4)


def build_feature_vector(observation: dict) -> dict[str, float]:
    """Full feature vector of one observation (all groups at once).

    observation — raw measurements from instrumentation.capture. Returns
    a flat name->float dict; any non-numeric / non-finite feature raises
    ValueError (the observation is excluded with the reason recorded).
    """
    conf = confidence_features(observation["per_step_logits"], observation["output_ids"])
    spec = snapshot_features(observation["final_hidden"])
    n_layers = len(observation["final_hidden"])
    if n_layers < 2:
        raise ValueError(f"need >= 2 hidden layers, got {n_layers}")
    groups = feature_groups(n_layers)
    temp = temporal_features(
        observation["step_hidden_last"], conf.pop("_ce_traj"), conf.pop("_entropy_traj")
    )
    geom = geometry_features(observation["final_hidden"])

    vec: dict[str, float] = {}
    vec.update(conf)
    vec.update(spec)
    vec.update(temp)
    vec.update(geom)

    expected = set(groups["combined"])
    missing = expected - set(vec)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)[:5]}")
    for name in expected:
        v = float(vec[name])
        if not np.isfinite(v):
            raise ValueError(f"non-finite feature {name}")
        vec[name] = v
    return {k: vec[k] for k in groups["combined"]}


def feature_names(group: str) -> list[str]:
    return list(FEATURE_GROUPS[group])


def as_matrix(vectors: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """List of dicts -> matrix (n, d) in the FROZEN column order."""
    return np.array([[v[n] for n in names] for v in vectors], dtype=np.float64)
