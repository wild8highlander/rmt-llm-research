"""Tests of the spectral features: RMT-null control with random matrices."""

import numpy as np
import pytest
from science.features.spectral import covariance_spectral_features, snapshot_features


def _random_cov(T=64, H=96, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (T, H))


def test_mp_null_gap_ratio_near_one():
    """Pure Gaussian noise: lambda_max sits at the MP edge (gap ≈ 1)."""
    f = covariance_spectral_features(_random_cov(seed=1), layer=0)
    assert f is not None
    assert 0.85 < f["gap_ratio"] < 1.15, f["gap_ratio"]
    assert f["q_ratio"] == pytest.approx(96 / 64)


def test_planted_spike_detected():
    """Plus one dominant component -> gap_ratio substantially > 1."""
    rng = np.random.default_rng(2)
    h = rng.normal(0, 0.3, (64, 96))
    h += 3.0 * rng.normal(0, 1, (64, 1))  # a common factor
    f = covariance_spectral_features(h, layer=0)
    assert f["gap_ratio"] > 1.3
    assert f["effective_rank"] < 96  # the spectrum is concentrated


def test_effective_rank_white_noise_full():
    rng = np.random.default_rng(3)
    h = rng.normal(0, 1, (256, 32))
    f = covariance_spectral_features(h, layer=0)
    # white noise has a flat spectrum: eff_rank is close to min(T,H)-1
    assert f["effective_rank"] > 0.75 * 32


def test_too_few_positions_returns_none():
    assert covariance_spectral_features(np.ones((3, 8)), 0) is None
    assert covariance_spectral_features(None, 0) is None


def test_snapshot_features_aggregate():
    rng = np.random.default_rng(4)
    hidden = [rng.normal(0, 1, (64, 96)) for _ in range(4)]
    out = snapshot_features(hidden)
    for k in (
        "gap_ratio_L0",
        "gap_ratio_L3",
        "gap_ratio_mean",
        "gap_ratio_last",
        "effrank_mean",
        "effrank_last",
        "participation_mean",
        "ncrit_free_last",
        "lambda_max_last",
    ):
        assert k in out
        assert np.isfinite(out[k])
    assert out["gap_ratio_mean"] == pytest.approx(
        np.mean([out[f"gap_ratio_L{i}"] for i in range(4)])
    )


def test_rank_deficient_covariance_convention():
    """T < H: sigma2 = trace/H, not the mean of the non-zero eigenvalues.
    For white noise (T=16, H=96, rank <= 15) gap_ratio must stay ~1."""
    rng = np.random.default_rng(6)
    h = rng.normal(0, 1, (16, 96))
    f = covariance_spectral_features(h, layer=0)
    assert f is not None
    assert 0.7 < f["gap_ratio"] < 1.3
