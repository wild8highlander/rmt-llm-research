"""Тесты доверительных признаков, временной динамики и геометрии."""

import numpy as np
import pytest
from science.features.confidence import confidence_features
from science.features.geometry import geometry_features
from science.features.temporal import state_lambda_trajectory, temporal_features


def _logits(n_steps=10, vocab=32, seed=0, sharp_first=True):
    rng = np.random.default_rng(seed)
    lg = rng.normal(0, 1, (n_steps, vocab))
    if sharp_first:
        lg[0, 3] = 12.0  # первый шаг почти уверен
    return lg


def test_confidence_features_known_values():
    lg = _logits(seed=10)
    ids = [3] * 10
    out = confidence_features(lg, ids)
    p0 = np.exp(lg[0] - lg[0].max())
    p0 /= p0.sum()
    assert out["maxprob_first"] == pytest.approx(p0.max(), rel=1e-9)
    assert out["ce_first"] == pytest.approx(-np.log(p0[3]), rel=1e-9)
    assert 0 < out["entropy_mean"] <= np.log(32) + 1e-9
    assert out["ppl_answer"] == pytest.approx(np.exp(out["ce_mean"]), rel=1e-9)
    assert "_ce_traj" in out
    assert len(out["_ce_traj"]) == 10


def test_confidence_rejects_empty():
    with pytest.raises(ValueError):
        confidence_features(np.zeros((0, 8)), [])


def test_state_lambda_trajectory_shape_and_finite():
    rng = np.random.default_rng(11)
    stream = rng.normal(0, 1, (12, 4, 16))
    lam = state_lambda_trajectory(stream, window=8)
    assert lam.shape == (12,)
    assert np.all(np.isfinite(lam))
    # после накопления окна (шаг >= 7) значения должны расти со всплеском
    stream[8:, :, 0] += 10.0
    lam2 = state_lambda_trajectory(stream, window=8)
    assert lam2[-1] > lam2[7]


def test_temporal_features_keys():
    rng = np.random.default_rng(12)
    stream = rng.normal(0, 1, (10, 4, 16))
    ce = np.linspace(3.0, 1.0, 10)  # падающий CE
    ent = rng.uniform(0, 2, 10)
    out = temporal_features(stream, ce, ent)
    assert set(out) == {
        "lam_traj_slope",
        "lam_traj_mean",
        "lam_traj_std",
        "ent_traj_slope",
        "ent_traj_delta",
        "ce_traj_slope",
        "ce_traj_std",
    }
    assert out["ce_traj_slope"] < 0  # CE падает -> наклон отрицательный
    assert np.isfinite(list(out.values())).all()


def test_temporal_rejects_short():
    with pytest.raises(ValueError):
        temporal_features(np.ones((10, 4, 8)), np.array([1.0]), np.array([1.0]))


def test_geometry_features():
    rng = np.random.default_rng(13)
    layers = [rng.normal(0, 1, (16, 8)) for _ in range(4)]
    out = geometry_features(layers)
    assert set(out) == {"rep_norm_mean", "rep_norm_last", "drift_mean", "drift_last"}
    # идентичные соседние слои -> косинус 1
    same = [np.ones((4, 8)) for _ in range(3)]
    assert geometry_features(same)["drift_mean"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        geometry_features(layers[:1])
