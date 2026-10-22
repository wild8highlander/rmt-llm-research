"""
features.confidence — standard confidence indicators (baseline group).

Computed from per_step_logits (the logits of the last position of each
generation step) and the actually chosen tokens:

  ce_first / ce_mean   — cross-entropy of the first / step-averaged token;
  ppl_answer           — exp(ce_mean);
  maxprob_first / mean — the max probability of the distribution;
  entropy_mean         — the mean entropy of the distributions;
  margin_first         — the p1 - p2 gap at the first step.

These are the "features available before the reference answer" — the
base group for checking that spectral features ADD information.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def confidence_features(per_step_logits: np.ndarray, output_ids: list[int]) -> dict:
    """per_step_logits: (steps, vocab); output_ids: the chosen tokens."""
    if per_step_logits.ndim != 2 or per_step_logits.shape[0] == 0:
        raise ValueError("per_step_logits must be (steps, vocab) with steps >= 1")
    steps = min(per_step_logits.shape[0], len(output_ids))
    if steps == 0:
        raise ValueError("no generated tokens to score")
    logp = _log_softmax(per_step_logits[:steps])
    probs = np.exp(logp)
    chosen = logp[np.arange(steps), np.asarray(output_ids[:steps], dtype=np.int64)]

    ce_per_step = -chosen
    maxprob = probs.max(axis=-1)
    entropy = -(probs * np.log(probs + _EPS)).sum(axis=-1)
    srt = np.sort(probs, axis=-1)
    margin = srt[:, -1] - srt[:, -2]

    return {
        "ce_first": float(ce_per_step[0]),
        "ce_mean": float(ce_per_step.mean()),
        "ppl_answer": float(np.exp(ce_per_step.mean())),
        "maxprob_first": float(maxprob[0]),
        "maxprob_mean": float(maxprob.mean()),
        "entropy_mean": float(entropy.mean()),
        "margin_first": float(margin[0]),
        # per-step trajectories are needed by the temporal features; returned separately
        "_ce_traj": ce_per_step,
        "_entropy_traj": entropy,
    }
