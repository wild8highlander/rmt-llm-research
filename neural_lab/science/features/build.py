"""
features.build — сборка замороженного вектора признаков наблюдения.

Вектор признаков и разбивка по группам ЗАФИКСИРОВАНЫ в протоколе H1
(EXPERIMENT_PROTOCOL.md, registry.yaml). Состав групп:

  confidence (7) — стандартные показатели уверенности (baseline-группа B);
  spectral  (15) — спектр ковариаций скрытых состояний (MP-гэп, ранги);
  temporal   (7) — динамика состояний/энтропии/self-CE по шагам;
  geometry   (4) — геометрия представлений между слоями.

Внутренняя группа internal = spectral + temporal + geometry (26 при 4 слоях) —
«признаки, требующие доступа к внутренним состояниям». Комбинированная
модель D = confidence + internal (33 при 4 слоях).

Протокол v1.1: пер-слоевые спектральные признаки обобщаются на любую
глубину модели — имена L0..L{n-1} строятся из фактического числа слоёв
(схема v1.0 = частный случай n_layers=4). Размер вектора: 2*n_layers+25.
"""

from __future__ import annotations

import numpy as np

from .confidence import confidence_features
from .geometry import geometry_features
from .spectral import snapshot_features
from .temporal import temporal_features


LAYER_KEYS = (
    "gap_ratio_L0",
    "gap_ratio_L1",
    "gap_ratio_L2",
    "gap_ratio_L3",
    "effrank_L0",
    "effrank_L1",
    "effrank_L2",
    "effrank_L3",
)


def feature_groups(n_layers: int) -> dict[str, list[str]]:
    """Группы признаков для модели глубиной n_layers (протокол v1.1).

    Для n_layers=4 возвращает ровно замороженную схему v1.0 (тот же порядок
    имен — порядок столбцов as_matrix фиксируется списком). Пер-слоевые
    имена L0..L{n-1}; остальные группы инвариантны к глубине.
    """
    if n_layers < 2:
        raise ValueError(f"need >= 2 hidden layers, got {n_layers}")
    layer_keys = [f"gap_ratio_L{i}" for i in range(n_layers)] + [
        f"effrank_L{i}" for i in range(n_layers)
    ]
    groups: dict[str, list[str]] = {
        "confidence": [
            "ce_first",
            "ce_mean",
            "ppl_answer",
            "maxprob_first",
            "maxprob_mean",
            "entropy_mean",
            "margin_first",
        ],
        "spectral": [
            *layer_keys,
            "gap_ratio_mean",
            "gap_ratio_last",
            "effrank_mean",
            "effrank_last",
            "participation_mean",
            "ncrit_free_last",
            "lambda_max_last",
        ],
        "temporal": [
            "lam_traj_slope",
            "lam_traj_mean",
            "lam_traj_std",
            "ent_traj_slope",
            "ent_traj_delta",
            "ce_traj_slope",
            "ce_traj_std",
        ],
        "geometry": [
            "rep_norm_mean",
            "rep_norm_last",
            "drift_mean",
            "drift_last",
        ],
    }
    groups["internal"] = [n for g in ("spectral", "temporal", "geometry") for n in groups[g]]
    groups["combined"] = groups["confidence"] + groups["internal"]
    return groups


# Замороженная схема протокола v1.0 (TinyGPT v3 S-ячейка, 4 слоя).
# Для другой глубины используйте feature_groups(n_layers).
FEATURE_GROUPS: dict[str, list[str]] = feature_groups(4)


def build_feature_vector(observation: dict) -> dict[str, float]:
    """Полный вектор признаков одного наблюдения (все группы сразу).

    observation — сырые измерения из instrumentation.capture. Возвращает
    плоский словарь имя->float; любой нечисловой/не-конечный признак
    вызывает ValueError (наблюдение исключается с записью причины).
    """
    conf = confidence_features(observation["per_step_logits"], observation["output_ids"])
    spec = snapshot_features(observation["final_hidden"])
    n_layers = len(observation["final_hidden"])
    if n_layers < 2:
        raise ValueError(f"need >= 2 hidden layers, got {n_layers}")
    groups = feature_groups(n_layers)
    temp = temporal_features(
        observation["step_hidden_last"], conf.pop("_ce_traj"), conf.pop("_entropy_traj")
    )
    geom = geometry_features(observation["final_hidden"])

    vec: dict[str, float] = {}
    vec.update(conf)
    vec.update(spec)
    vec.update(temp)
    vec.update(geom)

    expected = set(groups["combined"])
    missing = expected - set(vec)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)[:5]}")
    for name in expected:
        v = float(vec[name])
        if not np.isfinite(v):
            raise ValueError(f"non-finite feature {name}")
        vec[name] = v
    return {k: vec[k] for k in groups["combined"]}


def feature_names(group: str) -> list[str]:
    return list(FEATURE_GROUPS[group])


def as_matrix(vectors: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """Список словарей -> матрица (n, d) в ЗАФИКСИРОВАННОМ порядке столбцов."""
    return np.array([[v[n] for n in names] for v in vectors], dtype=np.float64)
