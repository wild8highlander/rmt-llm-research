"""
stage2_detector.probes — linear probes on hidden states (the strongest
baseline class from the literature):

  Azaria & Mitchell (2023), "The Internal State of an LLM Knows When
    It's Lying" — logistic probes on last-token activations;
  Marks & Tegmark (2023), "The Geometry of Truth" — linear
    truth-directions probed across datasets.

Protocol implemented here (honest, no leakage):
  - probe input: last-token hidden vector per layer (L, H) -> layer chosen;
  - the AUROC is evaluated OUT-OF-FOLD: leave-one-benchmark-out (train on
    the other benchmarks, test on the held-out one) AND within-benchmark
    grouped 5-fold (grouping = benchmark item, so candidates/sentences of
    one item never straddle the split);
  - the logistic regression is the repo's own NumPy implementation
    (science.predictors.LogisticRegression, ADR-001): standardization
    statistics come from the training fold only.
"""

from __future__ import annotations

import numpy as np

from science.evaluation import roc_auc
from science.predictors import LogisticRegression


def grouped_folds(group_ids: list, n_folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-based fold assignment: whole groups go to one fold (train/test
    index arrays). Groups are distributed round-robin by sorted hash order
    for determinism."""
    import hashlib

    rng = np.random.default_rng(seed)
    groups = sorted(set(group_ids), key=lambda g: hashlib.sha256(str(g).encode()).hexdigest())
    order = rng.permutation(len(groups))
    fold_of_group = {groups[i]: int(order[i] % n_folds) for i in range(len(groups))}
    folds = []
    for k in range(n_folds):
        test_idx = np.where([fold_of_group.get(g, -1) == k for g in group_ids])[0]
        train_idx = np.where([fold_of_group.get(g, -1) != k for g in group_ids])[0]
        if len(test_idx) and len(train_idx):
            folds.append((train_idx, test_idx))
    return folds


def probe_layer_auroc(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    l2: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Cross-validated probe AUROC on a fixed feature matrix (n, d).

    Returns (mean AUROC, out-of-fold predicted probabilities)."""
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(l2=l2)
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        raise ValueError("no folds produced valid predictions (degenerate labels?)")
    return roc_auc(y[ok], oof[ok]), oof


def cross_benchmark_probe(
    X: np.ndarray,
    y: np.ndarray,
    bench_ids: list[str],
    l2: float = 1.0,
) -> dict:
    """Leave-one-benchmark-out probe evaluation.

    For every benchmark b: train on the others, test on b. Returns per-fold
    AUROCs and the pooled out-of-fold AUROC (each test benchmark is scored
    by a probe that never saw it)."""
    out_aucs = {}
    oof = np.full(len(y), np.nan)
    for b in sorted(set(bench_ids)):
        test_idx = np.where(np.asarray(bench_ids) == b)[0]
        train_idx = np.where(np.asarray(bench_ids) != b)[0]
        if len(test_idx) == 0 or len(train_idx) == 0 or len(np.unique(y[train_idx])) < 2:
            continue
        clf = LogisticRegression(l2=l2)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])
        oof[test_idx] = p
        out_aucs[b] = roc_auc(y[test_idx], p)
    ok = ~np.isnan(oof)
    pooled = roc_auc(y[ok], oof[ok]) if ok.any() else float("nan")
    return {"per_benchmark": out_aucs, "pooled_oof_auc": pooled, "oof_proba": oof}


def evaluate_probe_layers(
    last_hidden: list[np.ndarray],
    y: np.ndarray,
    group_ids: list,
    bench_ids: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """Probe AUROC per layer (within-benchmark grouped folds, pooled across
    benchmarks), plus the leave-one-benchmark-out check for the best layer.

    last_hidden: list of (L, H) arrays, one per scored unit.
    Returns {"per_layer_auc": [...], "best_layer": int,
             "best_layer_lob": {...}, "oof_proba": [...],
             "oof_source": "lobo" | "grouped_oof", "oof_note": str}.

    oof_proba: out-of-fold probabilities used as the probe_hidden detector
    score. With >= 2 benchmarks this is the leave-one-benchmark-out OOF
    (each benchmark scored by a probe that never saw it). With a SINGLE
    benchmark LOBO is undefined (its only fold has an empty train set), so
    the within-benchmark grouped OOF is used instead and the limitation is
    reported in oof_note — cross-benchmark generalization is then NOT
    claimed."""
    L = last_hidden[0].shape[0]
    per_layer = []
    for li in range(L):
        X = np.stack([lh[li] for lh in last_hidden])
        folds = grouped_folds(group_ids, n_folds=n_folds, seed=seed)
        auc, _ = probe_layer_auroc(X, y, folds)
        per_layer.append(float(auc))
    best = int(np.argmax(per_layer))
    X_best = np.stack([lh[best] for lh in last_hidden])
    folds_best = grouped_folds(group_ids, n_folds=n_folds, seed=seed)
    _, oof_grouped = probe_layer_auroc(X_best, y, folds_best)
    lob = cross_benchmark_probe(X_best, y, bench_ids)
    n_benchmarks = len(set(bench_ids))
    if n_benchmarks >= 2 and not np.isnan(lob["oof_proba"]).any():
        return {
            "per_layer_auc": per_layer,
            "best_layer": best,
            "best_layer_lob": lob,
            "oof_proba": lob["oof_proba"],
            "oof_source": "lobo",
            "oof_note": f"leave-one-benchmark-out over {n_benchmarks} benchmarks",
        }
    return {
        "per_layer_auc": per_layer,
        "best_layer": best,
        "best_layer_lob": lob,
        "oof_proba": oof_grouped,
        "oof_source": "grouped_oof",
        "oof_note": (
            f"LOBO not applicable with {n_benchmarks} benchmark(s); probe_hidden uses "
            "within-benchmark grouped OOF — cross-benchmark generalization NOT evaluated"
        ),
    }
