"""Тесты сборщика признаков: замороженный состав групп и контроль качества."""

import numpy as np
import pytest
from science.features import FEATURE_GROUPS, as_matrix, build_feature_vector


def _fake_observation(n_layers=4, steps=10, T=32, H=24, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (steps, 64))
    logits[:, 7] += 6.0
    out_ids = [7] * steps
    final_hidden = [rng.normal(0, 1, (T, H)) for _ in range(n_layers)]
    step_hidden_last = rng.normal(0, 1, (steps, n_layers, H))
    return {
        "per_step_logits": logits,
        "output_ids": out_ids,
        "final_hidden": final_hidden,
        "step_hidden_last": step_hidden_last,
    }


def test_feature_vector_frozen_composition():
    vec = build_feature_vector(_fake_observation())
    assert set(vec) == set(FEATURE_GROUPS["combined"])
    assert len(FEATURE_GROUPS["confidence"]) == 7
    assert len(FEATURE_GROUPS["spectral"]) == 15
    assert len(FEATURE_GROUPS["temporal"]) == 7
    assert len(FEATURE_GROUPS["geometry"]) == 4
    assert len(FEATURE_GROUPS["combined"]) == 33
    # нет пересечений групп
    for g1 in ("confidence", "spectral", "temporal", "geometry"):
        for g2 in ("confidence", "spectral", "temporal", "geometry"):
            if g1 < g2:
                assert not set(FEATURE_GROUPS[g1]) & set(FEATURE_GROUPS[g2])
    assert all(np.isfinite(v) for v in vec.values())


def test_layer_depth_generalization_v1_1():
    """Протокол v1.1: пер-слоевые признаки следуют за глубиной модели."""
    from science.features import feature_groups

    # v1.0 = частный случай v1.1 для 4 слоёв (байт-в-байт те же списки)
    assert feature_groups(4) == FEATURE_GROUPS
    # 3 слоя теперь валидны: 2*3+25 = 31 признак
    vec3 = build_feature_vector(_fake_observation(n_layers=3))
    assert len(vec3) == 31
    assert "gap_ratio_L2" in vec3
    assert "gap_ratio_L3" not in vec3
    # 6 слоёв (M-ячейка): 37 признаков, имена L0..L5
    g6 = feature_groups(6)
    vec6 = build_feature_vector(_fake_observation(n_layers=6))
    assert len(vec6) == 37
    assert set(vec6) == set(g6["combined"])
    assert "gap_ratio_L5" in vec6
    assert "effrank_L5" in vec6
    # 10 слоёв (L-ячейка): 45 признаков
    vec10 = build_feature_vector(_fake_observation(n_layers=10))
    assert len(vec10) == 45
    assert "gap_ratio_L9" in vec10
    # остальные группы инвариантны к глубине
    for g in ("confidence", "temporal", "geometry"):
        assert g6[g] == FEATURE_GROUPS[g]


def test_single_layer_rejected():
    # v1.1: >= 2 слоёв валидно (см. test_layer_depth_generalization_v1_1);
    # отклоняется только вырожденный случай без пары слоёв
    with pytest.raises(ValueError, match="layers"):
        build_feature_vector(_fake_observation(n_layers=1))


def test_nonfinite_rejected():
    obs = _fake_observation()
    obs["per_step_logits"] = obs["per_step_logits"].copy()
    obs["per_step_logits"][0, 0] = np.nan
    with pytest.raises(ValueError):
        build_feature_vector(obs)


def test_as_matrix_fixed_column_order():
    vecs = [build_feature_vector(_fake_observation(seed=s)) for s in range(3)]
    names = FEATURE_GROUPS["confidence"]
    X = as_matrix(vecs, names)
    assert X.shape == (3, len(names))
    for j, n in enumerate(names):
        assert all(vecs[i][n] == X[i, j] for i in range(3))
