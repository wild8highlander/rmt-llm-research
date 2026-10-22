"""
data.splits — leakage-resistant splits.

Key principle: split not only randomly across examples but also across
groups. Repeated generations of one task (several sample_id of one
task_id) must NEVER land in different parts — otherwise the predictor
memorizes the task instead of learning to predict.

Schemes:
  grouped_kfold_v1     — K folds over tasks stratified by (family, band);
                         the primary scheme of the H1 protocol;
  leave_one_family_out — a hard generalization test: the whole cluster
                         of tasks of one family goes to test.
"""

from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    """Train/test independence violation (task-group intersection)."""


def grouped_kfold(
    task_ids: list[str],
    task_keys: list[tuple[str, str]],  # (family, band) for stratification
    k: int = 5,
    seed: int = 42,
) -> dict[int, list[str]]:
    """K folds over task_id stratified by (family, band).

    Within each stratum tasks are shuffled by the seeded RNG and dealt
    round-robin across folds — the family distribution over folds is
    balanced and the composition is deterministic.
    """
    if len(task_ids) != len(task_keys):
        raise ValueError("task_ids and task_keys must have equal length")
    strata: dict[tuple[str, str], list[str]] = {}
    for tid, key in zip(task_ids, task_keys):
        strata.setdefault(key, []).append(tid)
    rng = np.random.default_rng(seed)
    folds: dict[int, list[str]] = {f: [] for f in range(k)}
    for s_idx, key in enumerate(sorted(strata)):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        # rotate the starting fold across strata: without this, high-numbered
        # folds would be short-changed on tasks from short strata
        offset = s_idx % k
        for i, tid in enumerate(ids):
            folds[(i + offset) % k].append(tid)
    check_no_task_overlap(folds)
    return folds


def leave_one_family_out(
    task_ids: list[str], families: list[str]
) -> dict[str, dict[str, list[str]]]:
    """For each family: train = the other families, test = this one.

    Returns {family: {"train": [...], "test": [...]}}.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for fam in sorted(set(families)):
        test = [t for t, f in zip(task_ids, families) if f == fam]
        train = [t for t, f in zip(task_ids, families) if f != fam]
        if not train:
            raise LeakageError(f"leave_one_family_out: no train tasks for {fam}")
        out[fam] = {"train": train, "test": test}
    return out


def check_no_task_overlap(folds: dict[int, list[str]]) -> None:
    """Hard check: one task cannot appear in two folds."""
    seen: set[str] = set()
    for fold, ids in folds.items():
        dup = seen.intersection(ids)
        if dup:
            raise LeakageError(f"task overlap between folds at fold {fold}: {sorted(dup)[:3]}")
        seen.update(ids)


def family_overlap_report(folds: dict, task_family: dict[str, str]) -> dict:
    """Report: which families are represented in each fold (for the primary
    split, family intersections are ALLOWED by design — the strict variant
    is checked separately by leave_one_family_out)."""
    return {str(fold): sorted({task_family[t] for t in ids}) for fold, ids in folds.items()}
