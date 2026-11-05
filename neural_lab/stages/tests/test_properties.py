"""Property-based tests (hypothesis) for the stage statistics — the plan's
fuzz list: invariants that must hold for ANY input, not just fixtures."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st  # noqa: E402

from stages.common.passports import data_hash  # noqa: E402
from stages.stage3_ncrit.scaling_test import fit_scaling  # noqa: E402
from stages.stage3_ncrit.survival_fit import (  # noqa: E402
    binned_error_rate,
    fit_survival,
    weibull_mean,
)
from stages.stage4_interpretability.heavy_tail import powerlaw_alpha  # noqa: E402
from stages.stage5_rlhf.compare_pairs import paired_permutation  # noqa: E402


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False), min_size=5, max_size=40),
    st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False), min_size=5, max_size=40),
)
def test_paired_permutation_invariants(a, b):
    if len(a) != len(b):  # hypothesis draws the two lists independently
        return
    res = paired_permutation(a, b, n_perm=200, seed=0)
    assert 0.0 <= res["p_two_sided"] <= 1.0
    assert res["p_two_sided"] >= 1.0 / 201  # +1 correction floor
    lo, hi = res["ci95"]
    assert lo <= res["delta_mean"] <= hi


@settings(max_examples=40, deadline=None)
@given(st.dictionaries(
    st.text(min_size=1, max_size=8),
    st.integers(-1000, 1000), min_size=0, max_size=20))
def test_data_hash_key_order_invariant(d):
    shuffled = dict(reversed(list(d.items())))
    assert data_hash(d) == data_hash(shuffled)


def test_data_hash_list_order_sensitive():
    assert data_hash([1, 2, 3]) != data_hash([3, 2, 1])


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.floats(1.0, 500, allow_nan=False, allow_infinity=False), min_size=8, max_size=300),
    st.lists(st.sampled_from([0.0, 1.0]), min_size=8, max_size=300),
)
def test_binned_error_rate_conserves_errors(t, d):
    # t >= 1: token positions start at 1 (the curve's window edge)
    if len(t) != len(d):
        return
    curve = binned_error_rate(np.asarray(t), np.asarray(d), n_bins=5)
    assert sum(curve["errors"]) == int(sum(d))
    assert all(r >= 0 for r in curve["rate"])


@settings(max_examples=30, deadline=None)
@given(
    c=st.floats(5, 100, allow_nan=False),
    slope=st.floats(0.5, 3.0, allow_nan=False),
    n=st.integers(3, 8),
)
def test_fit_scaling_recovers_known_slope(c, slope, n):
    mu = np.linspace(0.25, 4.0, n)
    tau = c * mu ** (-slope)
    tau_ci = np.column_stack([tau * 0.8, tau * 1.25])
    res = fit_scaling(tau, tau_ci, mu, n_boot=200, seed=0)
    assert res["slope"] == pytest.approx(slope, rel=0.05, abs=0.05)


@settings(max_examples=30, deadline=None)
@given(
    gamma=st.floats(0.7, 2.0, allow_nan=False),
    n=st.integers(50, 400),
)
def test_fit_survival_recovers_weibull(gamma, n):
    rng = np.random.default_rng(0)
    tau = 20.0
    t = tau * rng.weibull(gamma, n)
    fit = fit_survival(t, np.ones_like(t), mu=None, n_starts=3, seed=0)
    assert fit["tau"] == pytest.approx(tau, rel=0.35)
    assert fit["gamma"] == pytest.approx(gamma, abs=0.4)
    assert weibull_mean(fit["tau"], fit["gamma"]) > 0


@settings(max_examples=15, deadline=None)
@given(alpha_true=st.floats(2.2, 4.5, allow_nan=False))
def test_powerlaw_alpha_recovers_pure_tail(alpha_true):
    # inverse-CDF sampling of p(x) ~ x^-alpha above xmin=1
    n = 4000
    u = np.random.default_rng(1).uniform(0, 1, n)
    x = (1.0 - u) ** (-1.0 / (alpha_true - 1.0))
    fit = powerlaw_alpha(x, xmin_grid=np.linspace(1.0, 5.0, 30))
    assert fit["alpha"] == pytest.approx(alpha_true, abs=0.25)
