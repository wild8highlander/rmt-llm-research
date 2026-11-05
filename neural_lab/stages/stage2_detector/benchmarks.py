"""
stage2_detector.benchmarks — public hallucination benchmarks with
independently established labels.

Three benchmarks, three labeling protocols (all label-based, deterministic,
no LLM judge in the default path):

  TruthfulQA (generation config, truthfulqa/truthful_qa)
      The model answers the question; the item is labeled hallucinated=1
      iff the answer matches none of `correct_answers` (normalized
      containment + token-overlap judge; a configurable external judge
      callable can be injected for paper-grade runs).
  HaluEval QA (pminervini/HaluEval, config `qa`)
      Candidate-scoring protocol: the item provides a right answer and a
      hallucinated answer; both are scored as continuations of
      question+knowledge. Each candidate is one unit, label = is_hallucinated.
  SelfCheckGPT wiki_bio (potsawee/wiki_bio_gpt3_hallucination, official
      author mirror of the paper data)
      Sentence-level labels from the GPT-3 annotation
      (accurate / minor_inaccurate / major_inaccurate). The detector reads
      (passage prefix + sentence); `gpt3_text_samples` (20 GPT-3 samples)
      enable the sampling-based signals of the original paper.

Limitations are documented where they exist (string judge for TruthfulQA);
nothing here invents labels.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Candidate:
    text: str
    is_hallucinated: bool


@dataclass
class Sentence:
    text: str
    is_nonfactual: bool  # major/minor inaccurate -> True (SelfCheckGPT labels)


@dataclass
class BenchmarkItem:
    item_id: str
    benchmark: str  # truthfulqa | halueval | selfcheckgpt
    context: str  # what the model reads (question / question+knowledge / passage)
    label: int | None = None  # item-level label if defined (TruthfulQA: filled after generation)
    candidates: list[Candidate] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    correct_answers: list[str] = field(default_factory=list)
    incorrect_answers: list[str] = field(default_factory=list)
    question: str = ""
    gpt3_samples: list[str] = field(default_factory=list)  # SelfCheckGPT reference samples

    @property
    def has_sentence_labels(self) -> bool:
        return bool(self.sentences)


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.I)
_NONALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/spaces — TruthfulQA-style."""
    t = _NONALNUM.sub(" ", text.lower())
    t = _ARTICLE_RE.sub(" ", t)
    return " ".join(t.split())


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of normalized token sets."""
    ta, tb = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def matches_any(answer: str, refs: list[str], contain_threshold: float = 0.75) -> bool:
    """Normalized containment or token overlap >= threshold with any ref."""
    na = normalize_answer(answer)
    if not na:
        return False
    for r in refs:
        nr = normalize_answer(r)
        if not nr:
            continue
        if nr in na or na in nr:
            return True
        if token_overlap(answer, r) >= contain_threshold:
            return True
    return False


def truthfulqa_judge(answer: str, correct: list[str], incorrect: list[str]) -> int:
    """1 = hallucinated. String judge (documented limitation): the answer is
    hallucinated unless it matches a correct answer. An incorrect match
    (the answer matches a KNOWN wrong reference) is the stronger evidence
    of hallucination — see truthfulqa_judge_detailed for the graded form
    used in diagnostics."""
    return truthfulqa_judge_detailed(answer, correct, incorrect)["label"]


def truthfulqa_judge_detailed(answer: str, correct: list[str], incorrect: list[str]) -> dict:
    """Graded judge: distinguishes 'simply did not match a correct answer'
    from 'matched a KNOWN incorrect answer'. Both are hallucinated=1 (the
    binary label is unchanged), but the evidence differs:
      'correct-match'    -> label 0;
      'incorrect-match'  -> label 1, matches a known wrong answer;
      'no-match'         -> label 1, no reference matched at all."""
    if matches_any(answer, correct):
        return {"label": 0, "evidence": "correct-match"}
    if matches_any(answer, incorrect):
        return {"label": 1, "evidence": "incorrect-match"}
    return {"label": 1, "evidence": "no-match"}


def make_truthfulqa_judge() -> Callable[[str, list[str], list[str]], int]:
    """Injectable judge factory (swap for an LLM judge in paper-grade runs)."""
    return truthfulqa_judge


def load_truthfulqa(n_items: int = 100, seed: int = 0, cache_dir: str | None = None) -> list[BenchmarkItem]:
    """TruthfulQA generation/validation. Label is set AFTER generation by the
    judge, so items ship with label=None and the reference answer lists."""
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed).select(range(min(n_items, len(ds))))
    items = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        items.append(
            BenchmarkItem(
                item_id=f"truthfulqa-{i}",
                benchmark="truthfulqa",
                context=f"Q: {q}\nA:",
                question=q,
                correct_answers=[ex["best_answer"]] + list(ex["correct_answers"]),
                incorrect_answers=list(ex["incorrect_answers"]),
            )
        )
    return items


def load_halueval(n_items: int = 100, seed: int = 0, cache_dir: str | None = None) -> list[BenchmarkItem]:
    """HaluEval QA: two labeled candidates per item (right / hallucinated)."""
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "qa", split="data", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed).select(range(min(n_items, len(ds))))
    items = []
    for i, ex in enumerate(ds):
        q, k = ex["question"], ex.get("knowledge") or ""
        ctx = (f"Passage: {k}\n\nQuestion: {q}\nAnswer:" if k else f"Question: {q}\nAnswer:")
        items.append(
            BenchmarkItem(
                item_id=f"halueval-{i}",
                benchmark="halueval",
                context=ctx,
                question=q,
                candidates=[
                    Candidate(text=ex["right_answer"].strip(), is_hallucinated=False),
                    Candidate(text=ex["hallucinated_answer"].strip(), is_hallucinated=True),
                ],
            )
        )
    return items


def load_selfcheckgpt(
    n_items: int = 50,
    seed: int = 0,
    passage_sentences: int = 5,
    cache_dir: str | None = None,
) -> list[BenchmarkItem]:
    """SelfCheckGPT wiki_bio: sentence-level labels from the GPT-3 annotation.

    Only the FIRST `passage_sentences` sentences of the annotated GPT-3 text
    are kept (the first sentence is factual context; annotations cover the
    rest). Labels: accurate -> False, minor/major_inaccurate -> True.
    """
    from datasets import load_dataset

    ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed).select(range(min(n_items, len(ds))))
    items = []
    for i, ex in enumerate(ds):
        sents = list(ex["gpt3_sentences"])[:passage_sentences]
        anns = list(ex["annotation"])[:passage_sentences]
        # passage prefix = source wiki passage, as in the paper (input text)
        prefix = ex["wiki_bio_text"].split("\n")[0].strip()
        sentences = [
            Sentence(text=s, is_nonfactual=("accurate" not in a)) for s, a in zip(sents, anns) if a
        ]
        if not sentences:
            continue
        items.append(
            BenchmarkItem(
                item_id=f"selfcheckgpt-{i}",
                benchmark="selfcheckgpt",
                context=prefix,
                sentences=sentences,
                gpt3_samples=list(ex.get("gpt3_text_samples") or []),
            )
        )
    return items


LOADERS = {
    "truthfulqa": load_truthfulqa,
    "halueval": load_halueval,
    "selfcheckgpt": load_selfcheckgpt,
}


def load_benchmarks(names: list[str], n_items: int = 100, seed: int = 0, cache_dir: str | None = None) -> list[BenchmarkItem]:
    out: list[BenchmarkItem] = []
    for name in names:
        if name not in LOADERS:
            raise ValueError(f"unknown benchmark {name!r}; known: {sorted(LOADERS)}")
        out.extend(LOADERS[name](n_items=n_items, seed=seed, cache_dir=cache_dir))
    return out
