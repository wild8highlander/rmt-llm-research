"""
stage2_detector.detectors — unified detector interface for the AUROC race.

A detector maps one scored unit (TruthfulQA answer, HaluEval candidate,
SelfCheck sentence) to a score with the orientation HIGHER = more likely
hallucinated. The unit carries: benchmark, item_id (grouping key for
clustered CIs), binary label.

Detectors implemented:
  tok_*      — token-entropy family (5 baselines from one measurement);
  sem_ent    — semantic entropy over the model's own K samples
               (Farquhar et al. 2024; Jaccard backend by default);
  spec_*     — the RMT detector: single spectral features
               (gap_ratio_last, ncrit_free_last) and the probe-on-spectral
               (logreg over the spectral summary vector, fitted honestly
               out-of-fold).

The probe on LAST-TOKEN HIDDEN STATES (the Azaria–Mitchell/Marks–Tegmark
baseline) is evaluated separately in run_stage2 because it is fitted
per-layer; here we only carry the units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from science.evaluation import average_precision, bootstrap_metric_ci, roc_auc


# The closed detector vocabulary (P2.9). A new detector = implement its
# scorer, add its name here, add a test, bump the protocol version
# (see docs/ADD_A_DETECTOR.md).
KNOWN_DETECTORS = frozenset({
    # token-entropy family (token_entropy.detector_scores_from_features)
    "tok_mean_entropy",
    "tok_max_entropy",
    "tok_last_entropy",
    "tok_neg_logprob",
    "tok_log_perplexity",
    # semantic entropy (semantic_entropy.semantic_entropy_score)
    "sem_ent",
    # RMT spectral single features + spectral probe (run_stage2)
    "spec_gap_last",
    "spec_ncrit_free",
    "spec_probe",
    # linear probe on last-token hidden states (probes.evaluate_probe_layers)
    "probe_hidden",
})


@dataclass
class Unit:
    unit_id: str
    benchmark: str
    item_id: str  # grouping key for clustered bootstrap
    label: int  # 1 = hallucinated / nonfactual
    scores: dict[str, float]  # detector_name -> score (higher = worse)

    def __post_init__(self):
        # an open dict let a typo'd detector name ("spec_gap_las") surface
        # only at the very END of a run; fail at construction instead
        unknown = set(self.scores) - KNOWN_DETECTORS
        if unknown:
            raise ValueError(
                f"unknown detector name(s) {sorted(unknown)}; known: "
                f"{sorted(KNOWN_DETECTORS)}"
            )

    def score(self, name: str) -> float:
        """Score by detector name; KeyError on an unknown OR unscored name."""
        if name not in KNOWN_DETECTORS:
            raise KeyError(f"unknown detector {name!r}; known: {sorted(KNOWN_DETECTORS)}")
        return self.scores[name]  # KeyError when not scored for this unit


def permutation_p(null: np.ndarray, stat: float, two_sided: bool = True) -> float:
    """Permutation p-value with the +1 correction: the smallest attainable
    p is 1/(n_null + 1), never exactly 0 (Davison & Hinkley 1997).

    null: permutation replicates of the statistic; stat: the observed value.
    The observed statistic is treated as part of the null ensemble, and a
    tolerance of 1e-12 absorbs float round-off in the comparisons.
    """
    null = np.asarray(null, dtype=np.float64)
    if null.size == 0 or not np.isfinite(stat):
        return float("nan")
    if two_sided:
        extreme = (np.abs(null) >= abs(stat) - 1e-12).sum()
        return float(min(1.0, (1.0 + extreme) / (1.0 + null.size)))
    extreme_ge = (null >= stat - 1e-12).sum()
    extreme_le = (null <= stat + 1e-12).sum()
    p_one = (1.0 + min(extreme_ge, extreme_le)) / (1.0 + null.size)
    return float(min(1.0, 2.0 * p_one))


def evaluate_detector(units: list[Unit], detector: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC + PR-AUC + 95% clustered bootstrap CI for one detector.

    Clustering is BY ITEM (candidates/sentences of one item are correlated;
    the bootstrap resamples items, not units)."""
    scores = np.asarray([u.scores[detector] for u in units], dtype=np.float64)
    labels = np.asarray([u.label for u in units], dtype=np.int64)
    if len(np.unique(labels)) < 2:
        raise ValueError(f"degenerate labels for detector {detector!r}")
    auc = roc_auc(labels, scores)
    ap = average_precision(labels, scores)
    boot = bootstrap_metric_ci(
        labels, scores, [u.item_id for u in units], lambda y, p: roc_auc(y, p), n_boot=n_boot, seed=seed,
    )
    out = {
        "detector": detector,
        "auroc": float(auc),
        "pr_auc": float(ap),
        "ci95": [boot["ci_low"], boot["ci_high"]],
        "n_units": int(len(units)),
    }
    if len(units) < 10:
        out["power_warning"] = f"n={len(units)}; AUROC is unstable below n≈20 and the CI is dominated by resampling noise"
    return out


def paired_delta_auc(
    units: list[Unit], det_a: str, det_b: str, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Clustered paired bootstrap of AUROC(det_a) - AUROC(det_b) over the
    units that carry BOTH scores (e.g. spectral features exist only on the
    generation-scored benchmark)."""
    units = [u for u in units if det_a in u.scores and det_b in u.scores]
    y = np.asarray([u.label for u in units], dtype=np.int64)
    pa = np.asarray([u.scores[det_a] for u in units], dtype=np.float64)
    pb = np.asarray([u.scores[det_b] for u in units], dtype=np.float64)
    items = np.asarray([u.item_id for u in units])
    uniq = np.unique(items)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(items == g)[0] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc(y[idx], pa[idx]) - roc_auc(y[idx], pb[idx]))
    deltas = np.asarray(deltas)
    d0 = roc_auc(y, pa) - roc_auc(y, pb)
    # two-sided p with the +1 correction: p=0 is impossible, the floor is
    # 1/(n_valid + 1) per side
    p = 2 * min(
        (1.0 + float((deltas <= 0).sum())) / (1.0 + len(deltas)),
        (1.0 + float((deltas >= 0).sum())) / (1.0 + len(deltas)),
    )
    return {
        "delta_auc": float(d0),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "p_two_sided": float(min(1.0, p)),
        "n_paired": int(len(units)),
    }
