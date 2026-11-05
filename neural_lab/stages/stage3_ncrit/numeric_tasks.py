"""
stage3_ncrit.numeric_tasks — token-verifiable tasks for the N_crit curve.

The H1 convention (truth recomputed by an independent checker) adapted to
token-level first-error measurement on a real LLM: arithmetic recall with
a few-shot prompt. The answer is a chain of numbers, so partial
generation is checkable and the FIRST WRONG NUMBER is a well-defined
collapse time in tokens.

Families: add3 (a+b+c), mul2 (a*b), sub2, sqrt. Difficulty is scaled by
the operand magnitude; `band` (indist/near/far) mirrors the H1 gradient:
  indist — 2-digit operands (models have seen millions of these),
  near   — 3-digit operands,
  far    — 4-5-digit operands / chained multiplications.
"""

from __future__ import annotations

import math

import numpy as np


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else f"{n:.4f}"


def make_task(family: str, rng: np.random.Generator) -> dict:
    if family == "add3":
        digits = rng.choice([2, 3, 4])
        a, b, c = (int(rng.integers(10 ** (digits - 1), 10**digits)) for _ in range(3))
        truth = a + b + c
        text = f"{a} + {b} + {c} = "
    elif family == "mul2":
        digits = rng.choice([2, 2, 3])
        a, b = int(rng.integers(10 ** (digits - 1), 10**digits)), int(rng.integers(11, 100))
        truth = a * b
        text = f"{a} * {b} = "
    elif family == "sub2":
        digits = rng.choice([3, 4])
        a, b = int(rng.integers(10 ** (digits - 1), 10**digits)), int(rng.integers(10 ** (digits - 1), 10**digits))
        truth = abs(a - b)
        text = f"{max(a, b)} - {min(a, b)} = "
    elif family == "sqrt":
        base = int(rng.integers(11, 100))
        truth = base
        text = f"{base * base} = sqrt( ) : the square root of {base * base} is "
    else:
        raise ValueError(f"unknown family {family!r}")
    return {
        "family": family,
        "prompt": text,
        "truth": [float(truth)],
        "truth_text": _fmt(truth),
        "answer": _fmt(truth),
    }


def few_shot_header(n_shot: int = 3, seed: int = 0) -> str:
    """Fixed few-shot prefix (tasks themselves are drawn per-run and hashed
    into the passport)."""
    rng = np.random.default_rng(seed)
    shots = []
    fams = ["add3", "mul2", "sub2"]
    for f in fams[: max(1, min(3, n_shot))]:
        t = make_task(f, rng)
        shots.append(f"{t['prompt']}{t['answer']}")
    return "\n".join(shots) + "\n"


def build_task_set(n_tasks: int = 60, families: list[str] | None = None, seed: int = 0) -> list[dict]:
    families = families or ["add3", "mul2", "sub2", "sqrt"]
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_tasks):
        fam = families[i % len(families)]
        t = make_task(fam, rng)
        t["task_id"] = f"ncrit-{fam}-{i}"
        tasks.append(t)
    return tasks


def make_checker(task: dict, tol: float = 0.05):
    from stages.stage3_ncrit.error_labels import numeric_prefix_checker

    return numeric_prefix_checker(task["truth"], tol=tol)


def sanity() -> bool:
    t = make_task("mul2", np.random.default_rng(0))
    chk = make_checker(t)
    # the checker sees ONLY the generated continuation (no prompt numbers)
    ok_full = chk(t["answer"], final=True)
    wrong = chk(str(int(float(t["answer"])) * 10 + 1), final=True)
    return ok_full and not wrong


if __name__ == "__main__":
    print("sanity:", sanity())
