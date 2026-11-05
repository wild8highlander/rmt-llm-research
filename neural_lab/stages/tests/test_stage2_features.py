"""Stage 2 unit tests: token entropy, semantic entropy, spectral detector,
probes, and the AUROC race machinery — synthetic data, no model needed."""

from __future__ import annotations

import numpy as np
import pytest

from science.evaluation import roc_auc
from stages.stage2_detector.detectors import Unit, evaluate_detector, paired_delta_auc
from stages.stage2_detector.semantic_entropy import (
    JaccardBackend,
    discrete_semantic_entropy,
    jaccard_similarity,
    predictive_entropy,
    semantic_entropy_score,
)
from stages.stage2_detector.spectral_features import (
    last_token_hidden,
    spectral_feature_matrix,
    spectral_summary,
)
from stages.stage2_detector.token_entropy import detector_scores_from_features, token_entropy_features

# --- token entropy -----------------------------------------------------------


def test_token_entropy_features_and_orientation():
    low = [{"entropy": 0.1, "logprob": -0.1}] * 5  # confident answer
    high = [{"entropy": 5.0, "logprob": -5.0}] * 5  # uncertain answer
    f_low = token_entropy_features(low)
    f_high = token_entropy_features(high)
    assert f_low and f_high
    s_low = detector_scores_from_features(f_low)
    s_high = detector_scores_from_features(f_high)
    for k in s_low:
        assert s_high[k] > s_low[k], f"{k} must be oriented higher=more hallucinated"
    assert token_entropy_features([]) is None


# --- semantic entropy --------------------------------------------------------


def test_jaccard_similarity_basics():
    assert jaccard_similarity("the cat sat", "the cat sat") == 1.0
    assert jaccard_similarity("Paris", "London") < 0.2
    assert jaccard_similarity("", "Paris") == 0.0


def test_discrete_semantic_entropy_extremes():
    same = JaccardBackend().cluster(["Paris", "Paris", "Paris"])
    assert discrete_semantic_entropy(same) == pytest.approx(0.0, abs=1e-9)
    distinct = JaccardBackend().cluster(["Paris", "London", "Tokyo", "Madrid"])
    h = discrete_semantic_entropy(distinct)
    assert h == pytest.approx(np.log(4), rel=1e-6)  # uniform over 4 clusters


def test_semantic_entropy_score_monotone():
    agreeing = ["Paris, France", "Paris France", "paris", "Paris"]  # one cluster
    mixed = ["Paris", "London", "Tokyo", "Madrid"]  # four clusters
    assert semantic_entropy_score(agreeing) == pytest.approx(0.0, abs=1e-9)
    assert semantic_entropy_score(mixed) > semantic_entropy_score(agreeing)


def test_predictive_entropy_uniform_beats_degenerate():
    answers = ["a", "b", "c", "d"]
    h_uniform = predictive_entropy(answers, [-1.0] * 4)
    h_degenerate = predictive_entropy(answers, [-100.0, -1.0, -1.0, -1.0])
    assert h_uniform > h_degenerate


# --- spectral features -------------------------------------------------------


def test_spectral_summary_and_matrix_order():
    rng = np.random.default_rng(0)
    # planted strong first direction in layer 2 -> gap_ratio peaks there
    layers = []
    for li in range(3):
        base = rng.normal(0, 1, size=(40, 16))
        if li == 2:
            base[:, 0] *= 6.0
        layers.append(base)
    summ = spectral_summary(layers)
    assert 0 <= summ["spike_layer"] <= 2
    X, keys = spectral_feature_matrix([summ, summ])
    assert X.shape == (2, len(keys))
    assert "gap_ratio_last" in keys and "ncrit_free_last" in keys
    lt = last_token_hidden(layers, prompt_len=40)
    assert lt.shape == (3, 16)
    np.testing.assert_allclose(lt[0], layers[0][-1])


def test_spectral_single_feature_separates_planted_signal():
    """Synthetic race: hidden states with a strong spike are 'hallucinated'
    (label 1) — the spectral feature must rank them above the bulk."""
    rng = np.random.default_rng(1)
    units, summ_list = [], []
    labels = []
    for i in range(30):
        spike = i % 2 == 1
        base = rng.normal(0, 1, size=(40, 16))
        if spike:
            base[:, 0] *= 8.0
        s = spectral_summary([base])
        units.append(Unit(f"u{i}", "bench", f"item{i}", int(spike),
                          {"spec_gap_last": s["gap_ratio_last"]}))
        summ_list.append(s)
        labels.append(spike)
    res = evaluate_detector(units, "spec_gap_last", n_boot=200, seed=0)
    assert res["auroc"] > 0.9
    d = paired_delta_auc(units, "spec_gap_last", "spec_gap_last", n_boot=100)
    assert d["delta_auc"] == 0.0 and d["p_two_sided"] == 1.0


# --- probes -------------------------------------------------------------------


def test_probe_recovers_planted_direction():
    from stages.stage2_detector.probes import evaluate_probe_layers

    rng = np.random.default_rng(2)
    H, L, n = 16, 3, 60
    last_hidden, labels, groups, bench = [], [], [], []
    w = rng.normal(0, 1, H)
    w /= np.linalg.norm(w)
    for i in range(n):
        y = i % 2
        layers = []
        for li in range(L):
            h = rng.normal(0, 1, (30, H))
            layers.append(h)
        # last-token hidden: planted direction only in layer 1
        lh = np.stack([rng.normal(0, 1, H) for _ in range(L)])
        lh[1] += (2 * y - 1) * 3.0 * w
        last_hidden.append(lh)
        labels.append(y)
        groups.append(f"g{i}")
        bench.append("b" if i < n // 2 else "c")
    res = evaluate_probe_layers(last_hidden, np.asarray(labels, float), groups, bench)
    assert res["best_layer"] == 1
    assert res["per_layer_auc"][1] > 0.9
    assert res["per_layer_auc"][0] < res["per_layer_auc"][1]


def test_probe_degenerate_labels_raise():
    from stages.stage2_detector.probes import grouped_folds, probe_layer_auroc

    X = np.random.default_rng(0).normal(size=(10, 4))
    y = np.ones(10)
    with pytest.raises(Exception):
        probe_layer_auroc(X, y, grouped_folds([f"g{i}" for i in range(10)]))


def test_grouped_folds_no_group_straddles():
    from stages.stage2_detector.probes import grouped_folds

    group_ids = [f"g{i % 7}" for i in range(35)]
    folds = grouped_folds(group_ids, n_folds=5)
    assert folds
    for tr, te in folds:
        assert not (set(np.asarray(group_ids)[tr]) & set(np.asarray(group_ids)[te]))
