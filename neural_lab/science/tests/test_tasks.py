"""Tests of tasks: call parsing, independent verification, labels."""

import numpy as np
import pytest
from science.data import (
    build_task_set,
    format_ok,
    label_observation,
    numeric_fields,
    numeric_ok,
    parse_call_args,
    relative_error,
    verify_truth,
)


def test_parse_call_args():
    name, args = parse_call_args("mp_bounds(q=0.5, sigma2=1.0)")
    assert name == "mp_bounds"
    assert args == {"q": 0.5, "sigma2": 1.0}
    assert parse_call_args("no_args_here") is None
    assert parse_call_args("foo(a=notanumber)") is None


def test_numeric_fields_and_ok():
    gt = " => lambda_minus = 0.281552, lambda_plus = 1.443448"
    assert numeric_fields(gt) == [0.281552, 1.443448]
    good = " => lambda_minus = 0.290000, lambda_plus = 1.500000"
    bad = " => lambda_minus = 0.900000, lambda_plus = 1.500000"
    assert numeric_ok(good, gt, tol=0.15)
    assert not numeric_ok(bad, gt, tol=0.15)


def test_format_ok():
    gt = " => lambda_max = 2.0"
    assert format_ok(" => lambda_max = 3.0", gt)
    assert not format_ok("ответ: 3.0", gt)


def test_relative_error():
    gt = " => x = 1.0"
    assert relative_error(" => x = 1.1", gt) == pytest.approx(0.1)
    assert np.isnan(relative_error("нет чисел", gt))


def test_verify_truth_mp_bounds():
    """Independent checker: the library recomputation matches the corpus shape."""
    truth = verify_truth("mp_bounds(q=0.5, sigma2=1.0)")
    assert truth is not None
    assert "lambda_minus" in truth
    assert "lambda_plus" in truth
    import marchenko_pastur as mp

    lm, lp = mp.mp_bounds(0.5, 1.0)
    assert f"{lm:.6f}" in truth
    assert f"{lp:.6f}" in truth


def test_verify_truth_unknown_family():
    assert verify_truth("some_unknown_fn(q=1.0)") is None
    assert verify_truth("tracy_widom_mean()") is not None


def test_label_observation_grid():
    gt = " => lambda_max = 2.0"
    gen = " => lambda_max = 2.4"  # 20% error
    lab = label_observation(gen, gt, [0.15, 0.25, 0.40])
    assert lab["numeric_ok_015"] == 0
    assert lab["numeric_ok_025"] == 1
    assert lab["numeric_ok_040"] == 1
    assert lab["format_ok"] is True
    assert lab["rel_error"] == pytest.approx(0.2)


def test_build_task_set_end_to_end(tmp_path):
    """Mini-corpus -> tasks with verification; an RHS mismatch is excluded."""
    import marchenko_pastur as mp

    lm, lp = mp.mp_bounds(0.5, 1.0)
    good = f"mp_bounds(q=0.5, sigma2=1.0) => lambda_minus = {lm:.6f}, lambda_plus = {lp:.6f}"
    bad = "mp_bounds(q=0.6, sigma2=1.0) => lambda_minus = 999.0, lambda_plus = 999.0"
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("=== EVAL PAIRS ===\n" + good + "\n" + bad + "\n", encoding="utf-8")

    def fake_encode(s):
        return list(range(len(s)))

    tasks, exclusions, stats = build_task_set(
        str(corpus), max_prompt_tokens=64, encode_fn=fake_encode, n_indist=10, n_near=2
    )
    assert stats["indist_rhs_mismatch"] == 1
    reasons = [e["reason"] for e in exclusions]
    assert "corpus_rhs_mismatch" in reasons
    indist = [t for t in tasks if t.band == "indist"]
    assert len(indist) == 1
    assert indist[0].family == "mp_bounds"
    # near-tasks are created and verified
    assert all(t.ground_truth.startswith(" =>") for t in tasks)
    far = [t for t in tasks if t.band == "far"]
    assert len(far) == 37  # the canonical held-out grid
