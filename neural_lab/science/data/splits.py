"""
data.splits — разбиения, устойчивые к утечкам.

Ключевой принцип: разделять не только случайно по примерам, но и по
группам. Повторные генерации одной задачи (несколько sample_id одной
task_id) НИКОГДА не должны попадать в разные части — иначе предсказатель
запоминает задачу, а не учится предсказывать.

Схемы:
  grouped_kfold_v1    — K фолдов по задачам со стратификацией по
                        (family, band); первичная схема протокола H1;
  leave_one_family_out — жёсткая проверка обобщения: весь район задач
                        одного семейства уходит в тест.
"""

from __future__ import annotations

import numpy as np


class LeakageError(RuntimeError):
    """Нарушение независимости train/test (пересечение групп задач)."""


def grouped_kfold(
    task_ids: list[str],
    task_keys: list[tuple[str, str]],  # (family, band) для стратификации
    k: int = 5,
    seed: int = 42,
) -> dict[int, list[str]]:
    """K фолдов по task_id со стратификацией по (family, band).

    Внутри каждой страты задачи перемешиваются сидированным RNG и
    раздаются по фолдам round-robin — распределение семейств по фолдам
    сбалансировано, состав детерминирован.
    """
    if len(task_ids) != len(task_keys):
        raise ValueError("task_ids and task_keys must have equal length")
    strata: dict[tuple[str, str], list[str]] = {}
    for tid, key in zip(task_ids, task_keys):
        strata.setdefault(key, []).append(tid)
    rng = np.random.default_rng(seed)
    folds: dict[int, list[str]] = {f: [] for f in range(k)}
    for s_idx, key in enumerate(sorted(strata)):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        # ротация стартового фолда по стратам: без неё фолды с большими
        # номерами недополучали бы задачи из коротких страт
        offset = s_idx % k
        for i, tid in enumerate(ids):
            folds[(i + offset) % k].append(tid)
    check_no_task_overlap(folds)
    return folds


def leave_one_family_out(
    task_ids: list[str], families: list[str]
) -> dict[str, dict[str, list[str]]]:
    """Для каждого семейства: train = остальные семейства, test = это.

    Возвращает {family: {"train": [...], "test": [...]}}.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for fam in sorted(set(families)):
        test = [t for t, f in zip(task_ids, families) if f == fam]
        train = [t for t, f in zip(task_ids, families) if f != fam]
        if not train:
            raise LeakageError(f"leave_one_family_out: no train tasks for {fam}")
        out[fam] = {"train": train, "test": test}
    return out


def check_no_task_overlap(folds: dict[int, list[str]]) -> None:
    """Жёсткая проверка: одна задача не может быть в двух фолдах."""
    seen: set[str] = set()
    for fold, ids in folds.items():
        dup = seen.intersection(ids)
        if dup:
            raise LeakageError(f"task overlap between folds at fold {fold}: {sorted(dup)[:3]}")
        seen.update(ids)


def family_overlap_report(folds: dict, task_family: dict[str, str]) -> dict:
    """Отчёт: какие семейства представлены в каждом фолде (для primary
    split пересечения семейств ДОПУСТИМЫ по дизайну — строгий вариант
    проверяется leave_one_family_out отдельно)."""
    return {str(fold): sorted({task_family[t] for t in ids}) for fold, ids in folds.items()}
