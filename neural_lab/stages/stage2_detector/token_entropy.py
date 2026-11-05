"""
stage2_detector.token_entropy — token-level confidence baselines.

Standard confidence signals computed from the DIRECT per-step measurements
of common.generation (per-step log-prob / entropy of emitted tokens):

  mean_entropy, max_entropy, last_entropy — distributional uncertainty;
  mean_logprob (= mean NLL negated), perplexity — answer confidence.

All detectors are oriented so that HIGHER score = more likely hallucinated.
"""

from __future__ import annotations

import numpy as np


def token_entropy_features(per_step: list[dict]) -> dict | None:
    """Feature dict from generate_with_capture's `per_step` records; None if
    the answer is empty (no tokens were emitted)."""
    if not per_step:
        return None
    ent = np.asarray([s["entropy"] for s in per_step], dtype=np.float64)
    lp = np.asarray([s["logprob"] for s in per_step], dtype=np.float64)
    return {
        "mean_entropy": float(ent.mean()),
        "max_entropy": float(ent.max()),
        "last_entropy": float(ent[-1]),
        "mean_logprob": float(lp.mean()),
        "perplexity": float(np.exp(-lp.mean())),
        "n_tokens": int(len(per_step)),
    }


def detector_scores_from_features(feats: dict) -> dict[str, float]:
    """Baseline detector scores; ALL oriented higher = more hallucinated."""
    if feats is None:
        return {}
    return {
        "tok_mean_entropy": feats["mean_entropy"],
        "tok_max_entropy": feats["max_entropy"],
        "tok_last_entropy": feats["last_entropy"],
        "tok_neg_logprob": -feats["mean_logprob"],
        "tok_log_perplexity": float(np.log(feats["perplexity"])),
    }
