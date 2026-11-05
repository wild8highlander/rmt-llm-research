"""
stage3_ncrit.error_labels — token-level first-error labeling with censoring.

For each generated trajectory we need the position of the FIRST token at
which the answer is provably wrong, with right-censoring when the answer
stays correct to the end (a hallucination that never happens is still
data: censored observations carry information about T_crit).

Checkers (each returns False = provably wrong, True = provably fine so
far, checked at every generated token):

  numeric_prefix_checker — tasks with numeric ground truth (the H1
    convention: truth is recomputed independently, here in plain Python).
    A number that has COMPLETED (followed by a non-digit or the end of
    text) and contradicts the corresponding truth number is an error.
    The correspondence is positional: k-th completed number vs k-th truth
    number, tolerance `tol` relative.
  short_answer_checker — short verifiable answers (TruthfulQA-style):
    an error is committed when the normalized prefix stops being a prefix
    of ANY correct answer and starts matching an incorrect answer.

The honest default is numeric tasks: partial credit is well defined, so
the first-error time is well defined. TruthfulQA-style answers commit to
a single token; the checker is provided for completeness and its
limitations are documented.
"""

from __future__ import annotations

import re

from stages.stage2_detector.benchmarks import normalize_answer

_COMPLETED_STREAM_RE = re.compile(r"-?\d+(?:\.\d+)?(?=[^0-9.]|\.(?!\d))")
_COMPLETED_FINAL_RE = re.compile(r"-?\d+(?:\.\d+)?(?=[^0-9.]|\.(?!\d)|$)")


def completed_numbers(prefix: str, final: bool = False) -> list[float]:
    """Numbers in the prefix that are fully emitted.

    A number is terminated when the following character is neither a digit
    nor a decimal point that continues the number ('7.5' is one number;
    '500. Next' terminates 500 with a sentence period). During STREAMING
    (final=False) a number still at the very end of the prefix is NOT
    complete — more digits may follow; at the end of generation
    (final=True) the trailing number is complete as well.
    """
    pattern = _COMPLETED_FINAL_RE if final else _COMPLETED_STREAM_RE
    return [float(m.group()) for m in pattern.finditer(prefix)]


def numeric_prefix_checker(truth: list[float], tol: float = 0.25):
    """check(prefix, final=False) -> True while no completed number
    contradicts truth. `final=True` also validates the trailing number
    (use at the last token of a finished trajectory)."""

    def check(prefix: str, final: bool = False) -> bool:
        got = completed_numbers(prefix, final=final)
        k = min(len(got), len(truth))
        for g, t in zip(got[:k], truth[:k]):
            if abs(g - t) > tol * max(abs(t), 1e-6):
                return False
        return True

    return check


def short_answer_checker(correct: list[str], incorrect: list[str]):
    """check(prefix, final=False) -> True while the prefix is still
    compatible with a correct answer. A prefix is WRONG once (a) it matches
    no correct answer as a prefix-extension AND (b) it contains an
    incorrect answer. Conservative: censoring (never erroring) is preferred
    over a false first-error label."""

    def check(prefix: str, final: bool = False) -> bool:
        np_prefix = normalize_answer(prefix)
        if any(nr.startswith(np_prefix) or np_prefix.startswith(nr)
               for nr in (normalize_answer(c) for c in correct if normalize_answer(c))):
            return True
        for inc in incorrect:
            ni = normalize_answer(inc)
            if ni and ni in np_prefix:
                return False
        return True

    return check


def first_error_position(
    token_texts: list[str],
    check,
) -> tuple[int | None, int]:
    """Walk the generated tokens; return (first_error_index_or_None, n_tokens).

    token_texts[i] is the decoded text of token i (incremental). The check
    receives the cumulative prefix; on the LAST token it is called with
    final=True (trailing numbers/text are then complete)."""
    prefix = ""
    for i, tok in enumerate(token_texts):
        prefix += tok
        if not check(prefix, final=(i == len(token_texts) - 1)):
            return i, len(token_texts)
    return None, len(token_texts)


def incremental_token_texts(tokenizer, output_ids: list[int]) -> list[str]:
    """Decode each generated token as the cumulative-decode diff (byte-level
    BPE safe: pieces are only defined up to merge boundaries).

    WARNING: byte-level BPE can merge a leading space with the next token,
    so decode([a, b]) is not guaranteed to equal decode([a]) + decode([b]);
    the cumulative diff then puts the merged text on the LAST piece of the
    merge. Fine for prefix checkers (the full prefix is always correct) —
    do NOT use the individual pieces as standalone token texts."""
    out = []
    prev = ""
    for i in range(len(output_ids)):
        cum = tokenizer.decode(output_ids[: i + 1])
        out.append(cum[len(prev):])
        prev = cum
    return out


def trajectory_table(trajs: list[dict]) -> list[dict]:
    """(t_first_error, censored, n_tokens) records from observe results."""
    rows = []
    for tr in trajs:
        t, n = tr["first_error"], tr["n_tokens"]
        rows.append(
            {
                "t_first_error": (float(t) if t is not None else float(n)),
                "censored": t is None,
                "n_tokens": int(n),
            }
        )
    return rows
