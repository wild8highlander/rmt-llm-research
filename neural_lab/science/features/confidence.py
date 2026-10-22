"""
features.confidence — стандартные показатели уверенности (baseline-группа).

Вычисляются из per_step_logits (логиты последней позиции каждого шага
генерации) и фактически выбранных токенов:

  ce_first / ce_mean   — cross-entropy первого / средней по шагам token'а;
  ppl_answer           — exp(ce_mean);
  maxprob_first / mean — максимум вероятности распределения;
  entropy_mean         — средняя энтропия распределений;
  margin_first         — разрыв p1 - p2 первого шага.

Это признаки «доступные до эталонного ответа» — базовая группа для
проверки того, что спектральные признаки ДОБАВЛЯЮТ информацию.
"""

from __future__ import annotations

import numpy as np


_EPS = 1e-12


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def confidence_features(per_step_logits: np.ndarray, output_ids: list[int]) -> dict:
    """per_step_logits: (steps, vocab); output_ids: выбранные токены."""
    if per_step_logits.ndim != 2 or per_step_logits.shape[0] == 0:
        raise ValueError("per_step_logits must be (steps, vocab) with steps >= 1")
    steps = min(per_step_logits.shape[0], len(output_ids))
    if steps == 0:
        raise ValueError("no generated tokens to score")
    logp = _log_softmax(per_step_logits[:steps])
    probs = np.exp(logp)
    chosen = logp[np.arange(steps), np.asarray(output_ids[:steps], dtype=np.int64)]

    ce_per_step = -chosen
    maxprob = probs.max(axis=-1)
    entropy = -(probs * np.log(probs + _EPS)).sum(axis=-1)
    srt = np.sort(probs, axis=-1)
    margin = srt[:, -1] - srt[:, -2]

    return {
        "ce_first": float(ce_per_step[0]),
        "ce_mean": float(ce_per_step.mean()),
        "ppl_answer": float(np.exp(ce_per_step.mean())),
        "maxprob_first": float(maxprob[0]),
        "maxprob_mean": float(maxprob.mean()),
        "entropy_mean": float(entropy.mean()),
        "margin_first": float(margin[0]),
        # пер-шаговые траектории нужны temporal-признакам; отдаём отдельно
        "_ce_traj": ce_per_step,
        "_entropy_traj": entropy,
    }
