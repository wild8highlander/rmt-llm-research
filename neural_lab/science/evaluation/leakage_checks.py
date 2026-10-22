"""
evaluation.leakage_checks — formal leakage-absence checks.

Leakage is the main source of self-deception in predictive laboratories.
The checks run on every split BEFORE fitting:

  1. no task (task_id) appears in both train and test;
  2. identical prompts never appear in different parts;
  3. a FAMILY-overlap report (allowed by design in the primary scheme;
     strict generalization control is provided by leave_one_family_out).
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
    """Exact prompt matches across parts — a hidden leak."""
    seen = set(train_prompts)
    leaked = sorted({p for p in test_prompts if p in seen})
    if leaked:
        raise LeakageError(
            f"prompt leakage{(' [' + tag + ']') if tag else ''}: "
            f"{len(leaked)} duplicate prompts, e.g. {leaked[:2]}"
        )
    return {"tag": tag, "n_test": len(test_prompts), "duplicates": 0}
