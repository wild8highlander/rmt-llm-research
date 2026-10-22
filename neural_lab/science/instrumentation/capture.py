"""
instrumentation.capture — захват наблюдений из реальной модели.

Наблюдение = одна генерация ответа на задачу. Сохраняются ТОЛЬКО
непосредственно измеренные величины (сырые наблюдения):

  per_step_logits  — логиты последней позиции на каждом шаге генерации;
  final_hidden     — скрытые состояния последнего шага по слоям (T, H);
  step_hidden_last — вектор последней позиции по слоям на каждом шаге
                     (для временной динамики; компактно);

Ничего синтетического: ни reasoning trace, ни искусственных метрик —
см. AUDIT.md (раздел «синтетические объекты»). Сырые наблюдения
отделяются от производных признаков (features/) и от меток (data/).
"""

from __future__ import annotations

import time

import numpy as np


def capture_observation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 28,
    temperature: float = 0.8,
    seed: int | None = None,
) -> dict:
    """Генерирует ответ и возвращает сырые измерения (без признаков и меток).

    temperature=0 — жадная генерация (детерминированная), temperature>0 —
    сэмплирование с заданным seed (повторные генерации одной задачи дают
    разброс, который сам по себе является признаком).
    """
    tok = tokenizer
    ids = tok.encode(prompt)[: model.config.max_seq_len // 2]
    max_new = max(0, min(max_new_tokens, model.config.max_seq_len - len(ids)))
    if max_new == 0:
        raise ValueError(f"prompt too long for context window: {prompt[:60]!r}")

    t0 = time.perf_counter()
    out = model.generate(
        np.array(ids, dtype=np.int64),
        max_new_tokens=max_new,
        temperature=temperature,
        seed=seed,
        capture_hidden=True,
    )
    elapsed = time.perf_counter() - t0

    out_ids = list(out["output_ids"])
    per_step_logits = np.stack(out["per_step_logits"]) if out["per_step_logits"] else None
    if per_step_logits is None or len(out_ids) == 0:
        raise ValueError(f"empty generation for prompt {prompt[:60]!r}")

    snapshots = out["hidden_snapshots"]  # list[step][layer] (T, H)
    final_hidden = [h.copy() for h in snapshots[-1]]
    step_hidden_last = (
        np.stack(
            [[h[-1].copy() for h in snap] for snap in snapshots]  # (steps, layers, H)
        )
        if snapshots
        else None
    )

    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(out_ids),
        "output_ids": out_ids,
        "per_step_logits": per_step_logits,  # (steps, vocab) float64
        "final_hidden": final_hidden,  # list[layer] (T, H)
        "step_hidden_last": step_hidden_last,  # (steps, layers, H) float64
        "n_prompt_tokens": len(ids),
        "n_new_tokens": len(out_ids),
        "temperature": temperature,
        "seed": seed,
        "elapsed_ms": round(elapsed * 1000, 1),
    }
