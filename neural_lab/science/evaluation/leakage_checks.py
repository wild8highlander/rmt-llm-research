"""
evaluation.leakage_checks — формальные проверки отсутствия утечек.

Утечки — главный источник самообмана в предсказательных лабораториях.
Проверки запускаются на каждом разбиении ДО обучения:

  1. ни одна задача (task_id) не входит одновременно в train и test;
  2. одинаковые промпты не встречаются в разных частях;
  3. отчёт о пересечении СЕМЕЙСТВ (в primary-схеме допустимо по дизайну,
     строгий контроль обобщения даёт leave_one_family_out).
"""

from __future__ import annotations

from ..data.splits import LeakageError, family_overlap_report


__all__ = [
    "LeakageError",
    "assert_no_task_overlap",
    "duplicate_prompt_check",
    "family_overlap_report",
]


def assert_no_task_overlap(train_ids: list[str], test_ids: list[str], tag: str = "") -> None:
    dup = set(train_ids).intersection(test_ids)
    if dup:
        raise LeakageError(f"task leakage{(' [' + tag + ']') if tag else ''}: {sorted(dup)[:5]}")


def duplicate_prompt_check(
    train_prompts: list[str], test_prompts: list[str], tag: str = ""
) -> dict:
    """Точное совпадение промптов в разных частях — скрытая утечка."""
    seen = set(train_prompts)
    leaked = sorted({p for p in test_prompts if p in seen})
    if leaked:
        raise LeakageError(
            f"prompt leakage{(' [' + tag + ']') if tag else ''}: "
            f"{len(leaked)} duplicate prompts, e.g. {leaked[:2]}"
        )
    return {"tag": tag, "n_test": len(test_prompts), "duplicates": 0}
