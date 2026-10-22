"""Тесты метрик и кластерной неопределённости."""

import numpy as np
import pytest
from science.evaluation import (
    average_precision,
    bootstrap_metric_ci,
    brier,
    cluster_permutation_paired,
    label_permutation_auc_p,
    log_loss,
    roc_auc,
)


def test_roc_auc_perfect_and_ties():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # все связи


def test_roc_auc_known_value_with_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.4, 0.4, 0.2, 0.6])
    # пары (pos, neg): (0.4,0.4)tie=0.5, (0.4,0.2)win, (0.6,0.4)win, (0.6,0.2)win
    assert roc_auc(y, p) == pytest.approx((0.5 + 1 + 1 + 1) / 4)


def test_roc_auc_single_class_raises():
    with pytest.raises(ValueError):
        roc_auc(np.ones(4), np.linspace(0, 1, 4))


def test_average_precision_known():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1])
    # precision на шагах: 1/1, 1/2, 2/3 -> AP = 1*1/2 + (2/3)*1/2 = 5/6
    assert average_precision(y, p) == pytest.approx(5 / 6)


def test_brier_logloss():
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.2])
    assert brier(y, p) == pytest.approx(0.04)
    assert log_loss(y, p) == pytest.approx(-np.log(0.8))


def _clustered(n_tasks=60, per_task=4, seed=0, effect=0.0):
    """Синтетика с кластерной структурой: у задачи свой скрытый сдвиг.
    effect=0 — слабый сигнал (AUC ~0.6–0.7); effect=1 — сильный (AUC ~1)."""
    rng = np.random.default_rng(seed)
    strength = 0.8 + 8.0 * effect
    y, p, tids = [], [], []
    for t in range(n_tasks):
        base = rng.random() < 0.4
        shift = rng.normal(0, 2.0)
        for _ in range(per_task):
            y_i = int(base)
            p_i = 1 / (1 + np.exp(-(strength * (y_i - 0.5) * 2 + shift)))
            y.append(y_i)
            p.append(p_i)
            tids.append(f"t{t:03d}")
    return np.array(y, float), np.array(p), np.array(tids)


def test_bootstrap_ci_contains_point():
    y, p, tids = _clustered(seed=1)
    ci = bootstrap_metric_ci(y, p, tids, roc_auc, n_boot=300, seed=2)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n_valid"] == 300


def test_cluster_permutation_paired_null_and_signal():
    y, p, tids = _clustered(seed=3)
    # одинаковые предсказания -> разница 0 -> p = 1
    res = cluster_permutation_paired(y, p, p, tids, n_perm=200, seed=4)
    assert res["p_value"] == 1.0
    # усиливаем разделение классов -> ΔAUC большой, p маленький
    _, p_strong, _ = _clustered(seed=3, effect=1.0)
    res2 = cluster_permutation_paired(y, p_strong, p, tids, n_perm=500, seed=5)
    assert res2["observed_delta"] > 0.3
    assert res2["p_value"] < 0.05


def test_label_permutation_noise_not_significant():
    rng = np.random.default_rng(6)
    n = 240
    y = (rng.random(n) < 0.4).astype(float)
    tids = np.array([f"t{i // 4:03d}" for i in range(n)])
    strata = np.array(["s"] * n)
    p = rng.random(n)  # шум, не связанный с меткой
    res = label_permutation_auc_p(y, p, tids, strata, n_perm=300, seed=7)
    assert res["p_value"] > 0.05  # шум не должен давать значимый AUC
