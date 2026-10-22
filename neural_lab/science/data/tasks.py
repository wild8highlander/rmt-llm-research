"""
data.tasks — набор задач с независимо верифицируемыми ответами.

Истинность ответа устанавливается НЕ строкой из корпуса, а повторным
вычислением той же формулы библиотекой `src/rmt_llm` (независимый
проверяющий по построению). Пары корпуса, чья правая часть расходится с
пересчётом, исключаются с записью причины в журнал исключений.

Происхождение каждой задачи промаркировано (band):
  indist — пара присутствовала в обучающем корпусе (модель «знает» ответ);
  near   — параметры возмущены на 4–8 % относительно корпусных;
  far    — свежие параметры, которых нет в корпусе (held-out).

Целевая переменная H1: label_numeric_ok — все числовые поля ответа в
допуске tol от истины. Первичный допуск 0.25; сетка чувствительности
[0.15, 0.25, 0.40] зафиксирована в протоколе ДО запуска.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..storage.schemas import Task


# --- единая реализация меток ответа (канон для H1; дубликаты в
# research/*/scripts остаются как есть — см. AUDIT.md, дублирование) ---

_NUM_RE = re.compile(r"-?\d+\.\d+")


def numeric_fields(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def _answer_part(gen: str) -> str:
    """Сегмент ответа: текст после ПОСЛЕДНЕГО '=>' (модель часто
    продолжает строку и эхом повторяет хвост промпта — числовые поля
    эха не должны попадать в сравнение с эталоном)."""
    if "=>" in gen:
        return gen.rsplit("=>", 1)[1]
    return gen


def format_ok(gen: str, gt: str) -> bool:
    """Структурная корректность: есть '=>' и все поля эталона — в сегменте ответа."""
    if "=>" not in gen:
        return False
    keys = [k.strip() for k in gt.split("=>")[1].split(",")]
    names = [k.split("=")[0].strip() for k in keys]
    ans = _answer_part(gen)
    return all(n in ans for n in names)


def numeric_ok(gen: str, gt: str, tol: float = 0.25) -> bool:
    """Все числовые поля СЕГМЕНТА ОТВЕТА в пределах допуска tol."""
    gt_nums = numeric_fields(gt)
    gen_nums = numeric_fields(_answer_part(gen))
    if not gt_nums or not gen_nums:
        return False
    k = min(len(gt_nums), len(gen_nums))
    return all(abs(g - t) <= tol * max(abs(t), 1e-6) for g, t in zip(gen_nums[:k], gt_nums[:k]))


def relative_error(gen: str, gt: str) -> float:
    """Средняя относительная ошибка первых k чисел сегмента ответа."""
    gt_nums = numeric_fields(gt)
    gen_nums = numeric_fields(_answer_part(gen))
    if not gt_nums or not gen_nums:
        return float("nan")
    k = min(len(gt_nums), len(gen_nums))
    errs = [abs(g - t) / max(abs(t), 1e-6) for g, t in zip(gen_nums[:k], gt_nums[:k])]
    return float(sum(errs) / len(errs))


# --- разбор вызовов формул и независимая верификация ---

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
    """Пересчитывает формулу библиотекой и возвращает эталонное
    продолжение (той же формы, что и корпусные пары). None — семейство
    без верификатора или аргументы не разобрались."""
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


# --- извлечение in-dist пар из корпуса ---

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
    """Извлекает уникальные пары из корпуса для верифицируемых семейств.

    Возвращает (пары, журнал исключений). Пара — (prompt, corpus_rhs,
    family). Промпт должен помещаться в половину окна контекста.
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

    # детерминированная подвыборка
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
    """Возмущённые задачи (band 'near'): параметры корпуса x (1 ± 4–7 %).

    Истина пересчитывается библиотекой; пара принимается, только если
    возмущённые параметры отличаются от корпусных и формула вычисляется.
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
            continue  # нет параметров — возмущать нечего
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
    """HELD-OUT (band 'far'): фиксированная сетка свежих параметров —
    перенос эталонного gt_pairs() из research-скриптов (детерминизм)."""
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
    """Собирает полный набор задач всех трёх полос с верификацией.

    Проверка независимости: правая часть корпуса сверяется с пересчётом
    библиотеки; расхождение -> исключение с причиной (обнаружение дрейфа
    корпуса). Возвращает (задачи, исключения, статистика).
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
        # проверка независимого проверяющего против корпуса
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

    # дедупликация промптов по ВСЕМ полосам (проверка утечек требует
    # уникальных промптов между частями разбиения): приоритет indist >
    # near > far — «известная» версия задачи остаётся, дубликат в старшей
    # полосе исключается с записью причины
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
    # восстановим детерминированный порядок: band, затем исходный task_id
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
    """Метки одного наблюдения по сетке допусков + вспомогательные метки.

    Первичная цель H1 — label_numeric_ok при tol=0.25 (первичный элемент
    протокола); остальные допуски — заранее объявленный sensitivity.
    """
    labels: dict[str, float | int | bool] = {
        "format_ok": bool(format_ok(generated, ground_truth)),
        "rel_error": relative_error(generated, ground_truth),
    }
    for tol in tolerance_grid:
        # канонический ключ: numeric_ok_015 / 025 / 040 (независимо от repr float)
        key = f"numeric_ok_{round(tol * 100):03d}"
        labels[key] = int(numeric_ok(generated, ground_truth, tol=tol))
    return labels
