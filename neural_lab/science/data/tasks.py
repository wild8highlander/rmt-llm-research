"""
data.tasks — a task set with independently verifiable answers.

Truth is established NOT by a corpus string but by recomputing the same
formula with the `src/rmt_llm` library (an independent checker by
construction). Corpus pairs whose right-hand side disagrees with the
recomputation are excluded, with the reason written to the exclusion log.

Every task carries a provenance band:
  indist — the pair was present in the training corpus (the model "knows" the answer);
  near   — parameters perturbed by 4-8 % relative to the corpus values;
  far    — fresh parameters absent from the corpus (held-out).

H1 target variable: label_numeric_ok — every numeric field of the answer
within tolerance tol of truth. Primary tolerance 0.25; the sensitivity
grid [0.15, 0.25, 0.40] was frozen in the protocol BEFORE the run.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..storage.schemas import Task


# --- single implementation of answer labelling (canonical for H1; the
# duplicates in research/*/scripts stay as they are — see AUDIT.md,
# duplication) ---

_NUM_RE = re.compile(r"-?\d+\.\d+")


def numeric_fields(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def _answer_part(gen: str) -> str:
    """Answer segment: the text after the LAST '=>' (the model often
    keeps typing and echoes the prompt tail — numeric fields from the
    echo must not participate in the comparison with ground truth)."""
    if "=>" in gen:
        return gen.rsplit("=>", 1)[1]
    return gen


def format_ok(gen: str, gt: str) -> bool:
    """Structural correctness: '=>' present and every ground-truth field
    name appears in the answer segment."""
    if "=>" not in gen:
        return False
    keys = [k.strip() for k in gt.split("=>")[1].split(",")]
    names = [k.split("=")[0].strip() for k in keys]
    answer = _answer_part(gen)
    return all(n in answer for n in names)


def numeric_ok(gen: str, gt: str, tol: float = 0.25) -> bool:
    """All numeric fields of the ANSWER SEGMENT within tolerance tol."""
    gt_nums = numeric_fields(gt)
    gen_nums = numeric_fields(_answer_part(gen))
    if not gt_nums or not gen_nums:
        return False
    k = min(len(gt_nums), len(gen_nums))
    return all(abs(g - t) <= tol * max(abs(t), 1e-6) for g, t in zip(gen_nums[:k], gt_nums[:k]))


def relative_error(gen: str, gt: str) -> float:
    """Mean relative error over the first k numbers of the answer segment."""
    gt_nums = numeric_fields(gt)
    gen_nums = numeric_fields(_answer_part(gen))
    if not gt_nums or not gen_nums:
        return float("nan")
    k = min(len(gt_nums), len(gen_nums))
    errs = [abs(g - t) / max(abs(t), 1e-6) for g, t in zip(gen_nums[:k], gt_nums[:k])]
    return float(sum(errs) / len(errs))


# --- formula-call parsing and independent verification ---

_CALL_RE = re.compile(r"^\s*([a-z_][a-z0-9_]*)\((.*)\)\s*$")


def parse_call_args(prompt: str) -> tuple[str, dict[str, float]] | None:
    """'mp_bounds(q=0.5, sigma2=1.0)' -> ('mp_bounds', {'q': 0.5, ...})."""
    m = _CALL_RE.match(prompt)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    args: dict[str, float] = {}
    for chunk in argstr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            return None
        k, v = chunk.split("=", 1)
        k, v = k.strip(), v.strip()
        if v in ("true", "false"):
            return None
        try:
            args[k] = float(v)
        except ValueError:
            return None
    return name, args


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def verify_truth(prompt: str) -> str | None:
    """Recompute the formula with the library and return the reference
    continuation (same shape as the corpus pairs). None — family without
    a verifier, or the arguments failed to parse."""
    parsed = parse_call_args(prompt)
    if parsed is None:
        return None
    name, args = parsed

    if name == "mp_bounds":
        import marchenko_pastur as mp

        lm, lp = mp.mp_bounds(args["q"], args["sigma2"])
        return f" => lambda_minus = {_fmt(lm)}, lambda_plus = {_fmt(lp)}"
    if name == "bbp_lambda_max":
        import bbp_transition as bbp

        return (
            f" => lambda_max = {_fmt(bbp.bbp_lambda_max(args['theta'], args['q'], args['sigma2']))}"
        )
    if name == "caputo_mean_collapse_time":
        import caputo_fractional as cap

        t = cap.caputo_mean_collapse_time(args["mu_eff"], args["beta"], args.get("c", 1.0))
        return f" => <T_crit> = {_fmt(t)}"
    if name == "tracy_widom_mean":
        import tracy_widom as tw

        return f" => E[F2] = {_fmt(tw.tracy_widom_mean())}"
    if name == "tracy_widom_variance":
        import tracy_widom as tw

        return f" => Var[F2] = {_fmt(tw.tracy_widom_variance())}"
    return None


# --- extraction of in-dist pairs from the corpus ---

_PAIR_RE = re.compile(r"^([a-z_][a-z0-9_]*\(.*\))\s*=>\s*(.*)$")

_VERIFIED_FAMILIES = (
    "mp_bounds",
    "bbp_lambda_max",
    "caputo_mean_collapse_time",
    "tracy_widom_mean",
    "tracy_widom_variance",
)


def extract_indist_prompts(
    corpus_path: str,
    max_prompt_tokens: int,
    encode_fn: Callable[[str], list[int]],
    n_max: int = 48,
    seed: int = 42,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Extract unique corpus pairs for the verifiable families.

    Returns (pairs, exclusion log). A pair is (prompt, corpus_rhs,
    family). The prompt must fit in half of the context window.
    """
    import numpy as np

    exclusions: list[dict] = []
    seen: set[str] = set()
    pairs: list[tuple[str, str, str]] = []
    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()
    for line in text.splitlines():
        m = _PAIR_RE.match(line)
        if not m:
            continue
        prompt, rhs = m.group(1).strip(), m.group(2).strip()
        family = prompt.split("(", 1)[0]
        if family not in _VERIFIED_FAMILIES:
            exclusions.append({"prompt": prompt[:80], "reason": "family_without_verifier"})
            continue
        if prompt in seen:
            exclusions.append({"prompt": prompt[:80], "reason": "duplicate_prompt"})
            continue
        seen.add(prompt)
        n_tok = len(encode_fn(prompt))
        if n_tok > max_prompt_tokens:
            exclusions.append({"prompt": prompt[:80], "reason": "prompt_too_long", "n_tok": n_tok})
            continue
        pairs.append((prompt, rhs, family))

    # deterministic subsampling
    if len(pairs) > n_max:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(pairs), size=n_max, replace=False))
        dropped = [p for i, p in enumerate(pairs) if i not in set(idx.tolist())]
        for p, _, _ in dropped:
            exclusions.append({"prompt": p[:80], "reason": "sampled_out"})
        pairs = [pairs[i] for i in idx.tolist()]
    return pairs, exclusions


def near_band_tasks(
    indist_pairs: list[tuple[str, str, str]],
    n_max: int = 16,
    jitter_grid: tuple[float, ...] = (0.04, -0.04, 0.07),
    seed: int = 42,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Perturbed tasks (band 'near'): corpus parameters x (1 ± 4-7 %).

    Truth is recomputed with the library; a pair is accepted only if the
    perturbed parameters differ from the corpus ones and the formula
    evaluates.
    """
    import numpy as np

    exclusions: list[dict] = []
    rng = np.random.default_rng(seed + 1)
    out: list[tuple[str, str, str]] = []
    pool = list(indist_pairs)
    rng.shuffle(pool)
    ji = 0
    for prompt, _, _ in pool:
        if len(out) >= n_max:
            break
        parsed = parse_call_args(prompt)
        if parsed is None:
            continue
        name, args = parsed
        if name == "tracy_widom_mean" or name == "tracy_widom_variance":
            continue  # no parameters — nothing to perturb
        jitter = jitter_grid[ji % len(jitter_grid)]
        ji += 1
        new_args = {k: v * (1.0 + jitter) for k, v in args.items()}
        call = name + "(" + ", ".join(f"{k}={_fmt(v)}" for k, v in new_args.items()) + ")"
        truth = verify_truth(call)
        if truth is None:
            exclusions.append({"prompt": call[:80], "reason": "near_verification_failed"})
            continue
        out.append((call, truth, name))
    return out, exclusions


def far_band_tasks() -> list[tuple[str, str, str]]:
    """HELD-OUT (band 'far'): a fixed grid of fresh parameters —
    a port of the reference gt_pairs() from the research scripts
    (determinism)."""
    import bbp_transition as bbp
    import caputo_fractional as cap
    import marchenko_pastur as mp
    import tracy_widom as tw

    pairs = []
    for q in [0.15, 0.35, 0.55, 0.65, 0.85]:
        for s2 in [0.75, 1.5]:
            lm, lp = mp.mp_bounds(q, s2)
            pairs.append(
                (
                    f"mp_bounds(q={q}, sigma2={s2})",
                    f" => lambda_minus = {_fmt(lm)}, lambda_plus = {_fmt(lp)}",
                    "mp_bounds",
                )
            )
    for th in [0.3, 0.5, 0.9, 1.1]:
        for q in [0.35, 0.55]:
            for s2 in [0.75, 1.5]:
                lm = bbp.bbp_lambda_max(th, q, s2)
                pairs.append(
                    (
                        f"bbp_lambda_max(theta={th}, q={q}, sigma2={s2})",
                        f" => lambda_max = {_fmt(lm)}",
                        "bbp_lambda_max",
                    )
                )
    for mu in [0.07, 0.13, 0.25]:
        for beta in [0.35, 0.45, 0.65]:
            t = cap.caputo_mean_collapse_time(mu, beta, 1.0)
            pairs.append(
                (
                    f"caputo_mean_collapse_time(mu_eff={mu}, beta={beta}, c=1.0)",
                    f" => <T_crit> = {_fmt(t)}",
                    "caputo_mean_collapse_time",
                )
            )
    pairs.append(
        ("tracy_widom_mean()", f" => E[F2] = {_fmt(tw.tracy_widom_mean())}", "tracy_widom_mean")
    )
    pairs.append(
        (
            "tracy_widom_variance()",
            f" => Var[F2] = {_fmt(tw.tracy_widom_variance())}",
            "tracy_widom_variance",
        )
    )
    return pairs


def build_task_set(
    corpus_path: str,
    max_prompt_tokens: int,
    encode_fn: Callable[[str], list[int]],
    n_indist: int = 48,
    n_near: int = 16,
) -> tuple[list[Task], list[dict], dict]:
    """Assemble the full task set across all three bands, with verification.

    Independence check: the corpus right-hand side is compared against
    the library recomputation; a mismatch -> exclusion with a reason
    (corpus-drift detection). Returns (tasks, exclusions, stats).
    """
    exclusions: list[dict] = []
    tasks: list[Task] = []
    stats = {"indist_candidates": 0, "indist_rhs_mismatch": 0}

    indist, exc = extract_indist_prompts(corpus_path, max_prompt_tokens, encode_fn, n_indist)
    exclusions.extend(exc)
    stats["indist_candidates"] = len(indist)
    for i, (prompt, rhs, family) in enumerate(indist):
        truth = verify_truth(prompt)
        if truth is None:
            exclusions.append({"prompt": prompt[:80], "reason": "verification_failed"})
            continue
        # independent-checker-vs-corpus verification
        a, b = numeric_fields(truth), numeric_fields(rhs)
        if (
            not a
            or not b
            or len(a) != len(b)
            or any(abs(x - y) > 1e-6 * max(abs(y), 1e-6) for x, y in zip(a, b))
        ):
            stats["indist_rhs_mismatch"] += 1
            exclusions.append({"prompt": prompt[:80], "reason": "corpus_rhs_mismatch"})
            continue
        tasks.append(
            Task(
                task_id=f"indist_{i:03d}",
                family=family,
                band="indist",
                prompt=prompt,
                ground_truth=truth,
                params=(parse_call_args(prompt) or (None, {}))[1],
                in_dist=True,
            )
        )

    near, exc = near_band_tasks(indist, n_max=n_near)
    exclusions.extend(exc)
    for i, (prompt, truth, family) in enumerate(near):
        tasks.append(
            Task(
                task_id=f"near_{i:03d}",
                family=family,
                band="near",
                prompt=prompt,
                ground_truth=truth,
                params=(parse_call_args(prompt) or (None, {}))[1],
                in_dist=False,
            )
        )

    for i, (prompt, truth, family) in enumerate(far_band_tasks()):
        tasks.append(
            Task(
                task_id=f"far_{i:03d}",
                family=family,
                band="far",
                prompt=prompt,
                ground_truth=truth,
                params=(parse_call_args(prompt) or (None, {}))[1],
                in_dist=False,
            )
        )

    # deduplicate prompts across ALL bands (the leakage checks require
    # unique prompts between split parts): priority indist > near > far —
    # the "known" version of a task stays, the duplicate in the higher
    # band is excluded with the reason recorded
    seen_prompts: set[str] = set()
    deduped: list[Task] = []
    priority = {"indist": 0, "near": 1, "far": 2}
    for t in sorted(tasks, key=lambda t: (priority[t.band], t.task_id)):
        if t.prompt in seen_prompts:
            exclusions.append(
                {
                    "prompt": t.prompt[:80],
                    "reason": "duplicate_prompt_across_bands",
                    "dropped_band": t.band,
                }
            )
            continue
        seen_prompts.add(t.prompt)
        deduped.append(t)
    # restore deterministic order: band, then the original task_id
    tasks = sorted(deduped, key=lambda t: (priority[t.band], t.task_id))
    for band, pref in (("indist", "indist"), ("near", "near"), ("far", "far")):
        for i, t in enumerate(x for x in tasks if x.band == band):
            t.task_id = f"{pref}_{i:03d}"

    stats["n_tasks"] = len(tasks)
    stats["by_band"] = {b: sum(1 for t in tasks if t.band == b) for b in ("indist", "near", "far")}
    stats["by_family"] = {}
    for t in tasks:
        stats["by_family"][t.family] = stats["by_family"].get(t.family, 0) + 1
    return tasks, exclusions, stats


def label_observation(generated: str, ground_truth: str, tolerance_grid: list[float]) -> dict:
    """Labels for one observation across the tolerance grid + aux labels.

    The primary H1 target is label_numeric_ok at tol=0.25 (the primary
    protocol element); the remaining tolerances are the pre-declared
    sensitivity analysis.
    """
    labels: dict[str, float | int | bool] = {
        "format_ok": bool(format_ok(generated, ground_truth)),
        "rel_error": relative_error(generated, ground_truth),
    }
    for tol in tolerance_grid:
        # canonical key: numeric_ok_015 / 025 / 040 (independent of float repr)
        key = f"numeric_ok_{round(tol * 100):03d}"
        labels[key] = int(numeric_ok(generated, ground_truth, tol=tol))
    return labels
