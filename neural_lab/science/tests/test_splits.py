"""Tests of the splits and the leakage checks."""

import numpy as np
import pytest
from science.data import (
    check_no_task_overlap,
    family_overlap_report,
    grouped_kfold,
    leave_one_family_out,
)
from science.data.splits import LeakageError
from science.evaluation import assert_no_task_overlap, duplicate_prompt_check


def _task_pool(n=40):
    fams = ["f1", "f2", "f3", "f4"]
    bands = ["indist", "near", "far"]
    ids = [f"task_{i:03d}" for i in range(n)]
    keys = [(fams[i % 4], bands[i % 3]) for i in range(n)]
    return ids, keys


def test_grouped_kfold_disjoint_and_full_cover():
    ids, keys = _task_pool()
    folds = grouped_kfold(ids, keys, k=5, seed=42)
    check_no_task_overlap(folds)  # does not raise
    all_ids = sorted(t for v in folds.values() for t in v)
    assert all_ids == sorted(ids)
    assert all(len(v) > 0 for v in folds.values())


def test_grouped_kfold_deterministic():
    ids, keys = _task_pool()
    a = grouped_kfold(ids, keys, k=5, seed=7)
    b = grouped_kfold(ids, keys, k=5, seed=7)
    assert a == b
    c = grouped_kfold(ids, keys, k=5, seed=8)
    assert a != c


def test_grouped_kfold_family_balance():
    """Stratification: every family is present in >= k-1 folds."""
    ids, keys = _task_pool(80)
    folds = grouped_kfold(ids, keys, k=5, seed=42)
    rep = family_overlap_report(folds, {t: k[0] for t, k in zip(ids, keys)})
    for fam in ("f1", "f2", "f3", "f4"):
        present = sum(1 for fl in rep.values() if fam in fl)
        assert present >= 4  # out of 5 folds


def test_kfold_raises_on_overlap():
    with pytest.raises(LeakageError):
        check_no_task_overlap({0: ["a", "b"], 1: ["b", "c"]})


def test_lofo_partitions():
    ids = ["a1", "a2", "b1", "b2", "c1"]
    fams = ["fa", "fa", "fb", "fb", "fc"]
    out = leave_one_family_out(ids, fams)
    assert set(out) == {"fa", "fb", "fc"}
    for fam, sp in out.items():
        assert not set(sp["train"]) & set(sp["test"])
        assert sorted(sp["train"] + sp["test"]) == sorted(ids)
        assert all(f == fam for f in np.array(fams)[[ids.index(t) for t in sp["test"]]])


def test_leakage_helpers():
    assert_no_task_overlap(["a", "b"], ["c"])  # ok
    with pytest.raises(LeakageError):
        assert_no_task_overlap(["a"], ["a"], tag="x")
    duplicate_prompt_check(["p1", "p2"], ["p3"])  # ok
    with pytest.raises(LeakageError):
        duplicate_prompt_check(["p1"], ["p1", "p4"], tag="y")
