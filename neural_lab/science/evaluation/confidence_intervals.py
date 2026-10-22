"""
evaluation.confidence_intervals — неопределённость метрик С УЧЁТОМ группировки.

Наблюдения внутри одной задачи коррелированы (повторные генерации), поэтому
и бутстреп, и перестановки делаются на уровне КЛАСТЕРОВ (задач), а не
отдельных примеров — иначе интервалы будут систематически занижены.
"""

from __future__ import annotations

import numpy as np

from .metrics import roc_auc


def _clusters(task_ids: np.ndarray) -> list[np.ndarray]:
    uniq = np.unique(task_ids)
    return [np.where(task_ids == t)[0] for t in uniq]


def bootstrap_metric_ci(
    y, p, task_ids, stat_fn, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """Кластерный бутстреп произвольной метрики: точечная оценка + CI.

    Вырожденные выборки (например, один класс) дают stat_fn=None в части
    бутстреп-реплик; если точечная оценка не определена, возвращается
    point=None (вырождение фиксируется, а не прячется).
    """
    y = np.asarray(y)
    p = np.asarray(p)
    task_ids = np.asarray(task_ids)
    try:
        point = float(stat_fn(y, p))
    except ValueError:
        point = None
    clusters = _clusters(task_ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        try:
            stats[b] = stat_fn(y[idx], p[idx])
        except ValueError:
            stats[b] = np.nan
    stats = stats[~np.isnan(stats)]
    if point is None or len(stats) == 0:
        return {
            "point": None,
            "ci_low": None,
            "ci_high": None,
            "n_boot": n_boot,
            "n_valid": len(stats),
            "note": "degenerate (metric undefined)",
        }
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "n_valid": len(stats),
    }


def cluster_bootstrap_delta(
    y, p_a, p_b, task_ids, metric_fn=roc_auc, n_boot: int = 2000, seed: int = 42
) -> dict:
    """CI для парной разницы метрик (модель D минус модель B) на одних и тех
    же бутстреп-выборках — корректная парная корреляция различий."""
    delta_fn = lambda yy, pp: metric_fn(yy, pp[:, 0]) - metric_fn(yy, pp[:, 1])
    return bootstrap_metric_ci(
        y, np.column_stack([p_a, p_b]), task_ids, delta_fn, n_boot=n_boot, seed=seed
    )


def cluster_permutation_paired(
    y, p_a, p_b, task_ids, metric_fn=roc_auc, n_perm: int = 2000, seed: int = 42
) -> dict:
    """Кластерный парный перестановочный тест: при нулевой гипотезе
    «предсказания A и B обменяемы», целиком ЗАДАЧА (все её повторные
    генерации) случайно меняется местами между A и B. Возвращает
    двусторонний p-value для разницы метрик."""
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    task_ids = np.asarray(task_ids)
    observed = float(metric_fn(y, p_a) - metric_fn(y, p_b))
    clusters = _clusters(task_ids)
    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for _ in range(n_perm):
        pa_parts, pb_parts = [], []
        for c in clusters:
            if rng.random() < 0.5:
                pa_parts.append(p_a[c])
                pb_parts.append(p_b[c])
            else:  # обмен целой задачей между моделями A и B
                pa_parts.append(p_b[c])
                pb_parts.append(p_a[c])
        pa = np.concatenate(pa_parts)
        pb = np.concatenate(pb_parts)
        try:
            d = float(metric_fn(y, pa) - metric_fn(y, pb))
        except ValueError:
            continue
        valid += 1
        if abs(d) >= abs(observed) - 1e-12:
            count += 1
    return {
        "observed_delta": observed,
        "p_value": (count + 1) / (valid + 1),
        "n_perm": n_perm,
        "n_valid": valid,
    }


def label_permutation_auc_p(y, p, task_ids, strata, n_perm: int = 500, seed: int = 42) -> dict:
    """Перестановочный тест AUC против нуля «признаки не связаны с меткой».

    Метки перемешиваются на уровне ЗАДАЧ (все повторные генерации задачи
    получают одинаковую (перевёрнутую) метку) внутри страт
    (band, family) — сохраняет структуру классов и кластеров.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    task_ids = np.asarray(task_ids)
    strata = np.asarray(strata)
    observed = float(roc_auc(y, p))
    uniq_tasks = np.unique(task_ids)
    task_stratum = {t: strata[np.where(task_ids == t)[0][0]] for t in uniq_tasks}
    task_label = {t: int(y[np.where(task_ids == t)[0][0]]) for t in uniq_tasks}
    # диагностика пространства рандомизации: сколько страт вообще можно
    # перемешать (в страте должны быть оба класса)
    strata_classes = {}
    for s in np.unique(strata):
        labs_s = {task_label[t] for t in uniq_tasks if task_stratum[t] == s}
        strata_classes[s] = labs_s
    n_strata_total = len(strata_classes)
    n_strata_permutable = int(sum(1 for v in strata_classes.values() if len(v) == 2))
    rng = np.random.default_rng(seed)
    count, valid = 0, 0
    for _ in range(n_perm):
        y_perm_tasks = {}
        for s in np.unique(strata):
            tasks_s = [t for t in uniq_tasks if task_stratum[t] == s]
            labs = np.array([task_label[t] for t in tasks_s])
            flip = rng.random() < 0.5
            perm = rng.permutation(labs) if not flip else 1 - rng.permutation(labs)
            if set(np.unique(perm)) != {0, 1}:
                # страта с одним классом неперемешиваема — сохраняем метки
                # (стандартная практика: рандомизация только внутри
                # переставимых страт)
                for t, v in zip(tasks_s, labs):
                    y_perm_tasks[t] = int(v)
                continue
            for t, v in zip(tasks_s, perm):
                y_perm_tasks[t] = int(v)
        if not y_perm_tasks:
            continue
        y_perm = np.array([y_perm_tasks[t] for t in task_ids])
        try:
            a = roc_auc(y_perm, p)
        except ValueError:
            continue
        valid += 1
        if a >= observed - 1e-12:
            count += 1
    return {
        "observed_auc": observed,
        "p_value": (count + 1) / (valid + 1),
        "n_perm": n_perm,
        "n_valid": valid,
        "n_strata_total": n_strata_total,
        "n_strata_permutable": n_strata_permutable,
        "note": (
            "weak randomization space: большинство страт "
            "одноклассовы — p-value близок к 1 и неинтерпретируем"
            if n_strata_permutable < max(2, n_strata_total // 2)
            else "ok"
        ),
    }
