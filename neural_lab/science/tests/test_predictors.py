"""Tests of the predictors: logistic regression, baselines, calibration."""

import numpy as np
import pytest
from science.evaluation import roc_auc
from science.predictors import (
    ConstantRateBaseline,
    FamilyRateBaseline,
    LogisticRegression,
    ece,
    platt_apply,
    platt_fit,
)


def _separable(n=200, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.normal(1.2, 1.0, (n // 2, d))
    X_neg = rng.normal(-1.2, 1.0, (n // 2, d))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=float)
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_logreg_separable_high_auc():
    X, y = _separable()
    m = LogisticRegression(l2=1.0).fit(X, y)
    p = m.predict_proba(X)
    assert roc_auc(y, p) > 0.9


def test_logrec_oof_no_leak_stats():
    """Standardization keeps train statistics; predict on new data works."""
    X, y = _separable(n=120, seed=1)
    Xtr, Xte, ytr = X[:80], X[80:], y[:80]
    m = LogisticRegression(l2=1.0).fit(Xtr, ytr)
    p = m.predict_proba(Xte)
    assert len(p) == 40
    assert (p > 0).all()
    assert (p < 1).all()


def test_logreg_single_class_raises():
    X = np.random.default_rng(2).normal(size=(20, 3))
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.ones(20))


def test_constant_rate_baseline():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=float)
    m = ConstantRateBaseline().fit(None, y)
    p = m.predict_proba(np.zeros((3, 1)))
    assert np.allclose(p, 3 / 8)
    # a degenerate train is not forbidden: the class constant is predicted
    m2 = ConstantRateBaseline().fit(None, np.ones(4))
    assert np.allclose(m2.predict_proba(np.zeros((2, 1))), 1.0)


def test_family_rate_baseline_smoothing():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    fam = np.array(["a", "a", "a", "b", "b", "b", "b", "b"])
    m = FamilyRateBaseline(smoothing=1.0).fit(None, y, families=fam)
    p = m.predict_proba(np.zeros((2, 1)), families=np.array(["a", "unknown"]))
    assert p[0] == pytest.approx((3 + 1) / (3 + 2))  # smoothed 3/3
    assert p[1] == pytest.approx(3 / 8)  # unknown family -> global rate


def test_platt_recovers_well_calibrated():
    rng = np.random.default_rng(3)
    p_true = rng.uniform(0.05, 0.95, 4000)
    y = (rng.random(4000) < p_true).astype(float)
    a, b = platt_fit(p_true, y)
    pc = platt_apply(p_true, a, b)
    # on perfectly calibrated data Platt must return ~identity
    assert abs(a - 1.0) < 0.15
    assert abs(b) < 0.1
    assert ece(y, pc, n_bins=10) < 0.03


def test_ece_detects_miscalibration():
    rng = np.random.default_rng(4)
    y = (rng.random(3000) < 0.5).astype(float)
    overconfident = np.full(3000, 0.95)  # always almost certain at base rate 0.5
    calibrated = np.full(3000, 0.5)
    assert ece(y, overconfident) > 0.3
    assert ece(y, calibrated) < 0.05
