"""
stage2_detector.semantic_entropy — semantic entropy (Farquhar et al., 2024).

Discrete semantic entropy over K sampled answers:

  1. sample K answers at temperature>0;
  2. cluster answers that mean the same thing (bidirectional entailment in
     the paper; here two backends: deterministic token-Jaccard clustering
     by default, optional NLI backend via a HF entailment pipeline);
  3. p(c) = m_c / K,  H_semantic = -sum_c p(c) ln p(c).

Higher semantic entropy = the model's samples disagree about MEANING =
uncertainty = hallucination risk. The score is an uncertainty feature:
higher -> more likely hallucinated, consistent with the other detectors.

Reference: Farquhar, Kossen, Kuhn & Gal (2024), "Detecting hallucinations
in large language models using semantic entropy", Nature 630.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class JaccardBackend:
    """Deterministic, dependency-free clustering: single-linkage greedy on
    token-Jaccard similarity. Good enough for short answers; for long free
    text prefer the NLI backend."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def cluster(self, answers: list[str]) -> list[int]:
        n = len(answers)
        if n == 0:
            return []
        assign = [0] * n  # cluster ids assigned greedily
        clusters: list[list[int]] = []
        for i, a in enumerate(answers):
            found = -1
            for c_idx, members in enumerate(clusters):
                if any(jaccard_similarity(a, answers[j]) >= self.threshold for j in members):
                    found = c_idx
                    break
            if found < 0:
                clusters.append([i])
                assign[i] = len(clusters) - 1
            else:
                clusters[found].append(i)
                assign[i] = found
        return assign


class NliBackend:
    """Bidirectional entailment clustering (paper-faithful) via a HF NLI
    pipeline. Two answers are in one cluster iff entailment holds in at
    least one direction with prob >= threshold. Requires an NLI model the
    first time it is downloaded (e.g. 'cross-encoder/nli-deberta-v3-base');
    NOT part of the default offline path."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.5, device: str = "cpu"):
        from transformers import pipeline  # guarded: optional dependency path

        self.threshold = float(threshold)
        self._nli = pipeline("text-classification", model=model_name, device=device,
                             truncation=True, max_length=256)

    def _entails(self, premise: str, hypothesis: str) -> bool:
        res = self._nli({"text": premise, "text_pair": hypothesis}, top_k=None)
        scores = {r["label"].lower(): float(r["score"]) for r in (res if isinstance(res, list) else [res])}
        ent = max(v for k, v in scores.items() if "entail" in k) if any("entail" in k for k in scores) else 0.0
        return ent >= self.threshold

    def cluster(self, answers: list[str]) -> list[int]:
        n = len(answers)
        assign = [0] * n
        clusters: list[list[int]] = []
        for i, a in enumerate(answers):
            found = -1
            for c_idx, members in enumerate(clusters):
                j = members[0]
                if self._entails(a, answers[j]) or self._entails(answers[j], a):
                    found = c_idx
                    break
            if found < 0:
                clusters.append([i])
                assign[i] = len(clusters) - 1
            else:
                clusters[found].append(i)
                assign[i] = found
        return assign


def discrete_semantic_entropy(cluster_ids: list[int]) -> float:
    """H = -sum_c (m_c/K) ln(m_c/K) over clusters (nats). 0 = all samples
    agree semantically; ln(K) = all samples are different meanings."""
    k = len(cluster_ids)
    if k == 0:
        raise ValueError("no samples")
    counts = np.bincount(np.asarray(cluster_ids, dtype=int))
    p = counts[counts > 0] / k
    return float(-(p * np.log(p)).sum())


def semantic_entropy_score(answers: list[str], backend: Callable | None = None) -> float:
    """End-to-end: cluster + entropy. `backend.cluster(answers) -> ids`."""
    backend = backend or JaccardBackend()
    return discrete_semantic_entropy(backend.cluster(answers))


def predictive_entropy(answers: list[str], seq_logprobs: list[float]) -> float:
    """Predictive entropy of the sample distribution (needs per-sample total
    log-prob): H = -sum_i w_i ln w_i with w_i = softmax of length-normalized
    log-probs (Kuhn et al. variant). Provided for completeness; the discrete
    semantic entropy above is the primary signal."""
    if len(answers) != len(seq_logprobs) or not answers:
        raise ValueError("answers/logprobs length mismatch")
    a = np.asarray(seq_logprobs, dtype=np.float64) / max(len(a) for a in answers)
    a = a - a.max()
    w = np.exp(a)
    w /= w.sum()
    return float(-(w * np.log(np.clip(w, 1e-12, None))).sum())
