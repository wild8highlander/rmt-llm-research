"""Stage 3 unit tests: first-error labeling, survival fit recovers a known
beta, changepoint detection, and the cross-model scaling regression."""

from __future__ import annotations

import numpy as np
import pytest

from stages.stage3_ncrit.error_labels import (
    completed_numbers,
    first_error_position,
    incremental_token_texts,
    numeric_prefix_checker,
    short_answer_checker,
)
from stages.stage3_ncrit.numeric_tasks import build_task_set, make_checker, sanity
from stages.stage3_ncrit.scaling_test import beta_consistency, fit_scaling, verdict
from stages.stage3_ncrit.survival_fit import (
    binned_error_rate,
    bootstrap_survival,
    detect_breakpoint,
    fit_survival,
    weibull_mean,
)


# --- checkers -----------------------------------------------------------------


def test_completed_numbers_only_terminated():
    assert completed_numbers("123 + 4") == [123.0]  # trailing 4 not complete while streaming
    assert completed_numbers("123 + 4", final=True) == [123.0, 4.0]  # complete at the end
    assert completed_numbers("42, 7.5") == [42.0]
    assert completed_numbers("42, 7.5", final=True) == [42.0, 7.5]
    assert completed_numbers("") == []


def test_numeric_prefix_checker_detects_first_wrong_number():
    chk = numeric_prefix_checker([100.0], tol=0.05)
    assert chk("The answer is 1")
    assert chk("The answer is 10")
    assert chk("The answer is 101", final=True)  # within the 5% tolerance
    assert not chk("The answer is 999", final=True)  # generation ended -> wrong
    assert not chk("The answer is 999.")  # terminated by punctuation -> wrong now


def test_first_error_position_walks_tokens():
    pieces = ["The", " answer", " is", " 5", "00", ".", " Next"]
    t, n = first_error_position(pieces, numeric_prefix_checker([100.0]))
    assert t == 5 and n == len(pieces)  # provable only when '500' terminates
    t2, _ = first_error_position(["The", " answer", " is", " 100", "."],
                                 numeric_prefix_checker([100.0]))
    assert t2 is None  # censored: never wrong


def test_incremental_token_texts_reconstructs():
    class FakeTok:
        def decode(self, ids):
            return "".join("abcdefghij"[i] for i in ids)

    out = incremental_token_texts(FakeTok(), [0, 1, 2])
    assert out == ["a", "b", "c"]


def test_short_answer_checker():
    chk = short_answer_checker(["Paris"], ["London", "Berlin"])
    assert chk("The capital is")
    assert chk("The capital is Par")
    assert not chk("The capital is London")


def test_numeric_tasks_sanity():
    assert sanity()
    tasks = build_task_set(n_tasks=8, seed=3)
    assert len(tasks) == 8
    for t in tasks:
        chk = make_checker(t)
        assert chk(t["answer"], final=True)  # the true generated answer passes
        wrong = str(int(float(t["answer"])) * 7 + 1)
        assert not chk(wrong, final=True)


# --- survival fit --------------------------------------------------------------


def simulate_survival(c: float, beta: float, gamma: float, mu: np.ndarray, n_per: int,
                      censor_at: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts, ds, mus = [], [], []
    for m in mu:
        tau = c * m ** (-1.0 / beta)
        t = tau * rng.weibull(gamma, size=n_per)
        d = (t <= censor_at).astype(float)
        t = np.minimum(t, censor_at)
        ts.append(t)
        ds.append(d)
        mus.append(np.full(n_per, m))
    return np.concatenate(ts), np.concatenate(ds), np.concatenate(mus)


def test_survival_fit_recovers_known_beta():
    c_true, beta_true, gamma_true = 40.0, 0.5, 1.2
    mu = np.asarray([0.5, 1.0, 2.0])
    t, d, m = simulate_survival(c_true, beta_true, gamma_true, mu, n_per=250,
                                censor_at=1e9, seed=7)
    fit = fit_survival(t, d, mu=m, n_starts=6, seed=0)
    assert fit["beta"] == pytest.approx(beta_true, abs=0.08)
    assert fit["c"] == pytest.approx(c_true, rel=0.35)
    assert fit["gamma"] == pytest.approx(gamma_true, abs=0.15)


def test_survival_fit_without_mu_is_weibull():
    rng = np.random.default_rng(3)
    tau, gamma = 20.0, 1.0
    t = tau * rng.weibull(gamma, 400)
    d = np.ones_like(t)
    fit = fit_survival(t, d, mu=None)
    assert fit["tau"] == pytest.approx(tau, rel=0.15)
    assert fit["gamma"] == pytest.approx(gamma, abs=0.1)
    assert weibull_mean(fit["tau"], fit["gamma"]) > 0


def test_bootstrap_gives_cis():
    rng = np.random.default_rng(5)
    t = 20.0 * rng.weibull(1.0, 300)
    d = np.ones_like(t)
    boot = bootstrap_survival(t, d, mu=None, n_boot=120, seed=0)
    lo, hi = boot["c"]["ci95"]
    assert lo < boot["c"]["mean"] < hi
    assert boot["c"]["ci95"][0] > 0


def test_binned_error_rate_normalizes_by_at_risk():
    t = np.asarray([2.0, 4.0, 6.0, 8.0])
    d = np.asarray([1.0, 1.0, 1.0, 0.0])
    curve = binned_error_rate(t, d, n_bins=4)
    assert len(curve["rate"]) == 4
    assert sum(curve["errors"]) == 3
    assert curve["at_risk"][0] == 4 and curve["at_risk"][-1] == 1


def test_detect_breakpoint_finds_rate_change():
    rng = np.random.default_rng(11)
    n1, n2 = 400, 400
    t1 = rng.exponential(40.0, n1)  # rare errors early
    t2 = 200.0 + rng.exponential(8.0, n2)  # frequent errors late
    t = np.concatenate([t1, t2])
    t = t[t < 400.0]
    d = np.ones_like(t)
    bp = detect_breakpoint(t, d)
    assert bp["breakpoint"] is not None
    assert 120.0 < bp["breakpoint"] < 280.0
    assert bp["p_perm"] < 0.05


# --- scaling --------------------------------------------------------------------


def test_fit_scaling_recovers_slope_two():
    c, beta = 40.0, 0.5
    mu = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    tau = c * mu ** (-1.0 / beta)
    tau_ci = np.column_stack([tau * 0.8, tau * 1.25])
    res = fit_scaling(tau, tau_ci, mu, n_boot=300, seed=0)
    assert res["slope"] == pytest.approx(2.0, abs=0.05)
    assert res["implied_beta"] == pytest.approx(0.5, abs=0.02)
    lo, hi = res["slope_ci95"]
    assert lo < 2.0 < hi


def test_beta_consistency_and_verdict():
    good = [{"beta": 0.48, "ci95": [0.42, 0.55]}, {"beta": 0.52, "ci95": [0.45, 0.6]}]
    bad = [{"beta": 0.3, "ci95": [0.25, 0.35]}, {"beta": 0.8, "ci95": [0.7, 0.9]}]
    assert beta_consistency(good)["all_consistent"]
    assert not beta_consistency(bad)["all_consistent"]

    scaling_ok = {"slope": 2.0, "slope_ci95": [1.8, 2.2], "implied_beta": 0.5}
    assert "supported" in verdict(scaling_ok, beta_consistency(good))
    scaling_diff_beta = {"slope": 3.0, "slope_ci95": [2.8, 3.2], "implied_beta": 1 / 3}
    v = verdict(scaling_diff_beta, beta_consistency(good))
    assert "partially supported" in v
    scaling_bad = {"slope": 0.4, "slope_ci95": [0.1, 0.7], "implied_beta": 2.5}
    assert "not supported" in verdict(scaling_bad, beta_consistency(bad))
