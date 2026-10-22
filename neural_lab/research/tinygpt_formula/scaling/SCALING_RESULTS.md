# Scaling + Hallucination A/B — Results

Generated: 2026-09-19 03:59:35

The key metric is **confident_hallucination_rate**: the share of formatted
answers (format_ok=True) with wrong numbers (the "Utility Trap", ±15%
tolerance).

## HELD-OUT (37 fresh tasks — the parameters were NOT in the corpus)

| Cell | Parameters | Epochs | val_loss | numeric_score | format_score | **confident hallucinations** | teacher_forced |
|---|---|---|---|---|---|---|---|
| S_control | 447,360 | 40 | 2.039 | 0.0% | 64.9% | **100.0%** (24/24) | 35.6% |
| S_formula | 447,360 | 40 | 1.954 | 0.0% | 40.5% | **100.0%** (15/15) | 36.7% |
| S_untrained | 447,360 | 0 | — | 0.0% | 0.0% | 0/0 — no formatted answers | — |

## IN-DISTRIBUTION (corpus pairs — the model SAW them during training)

The main question is checked here: does the model reproduce the TRUTH where
it has knowledge? The control model learned lies — and formats them
confidently.

| Cell | Parameters | Epochs | numeric_score | format_score | **confident hallucinations** |
|---|---|---|---|---|---|
| S_control | 447,360 | 40 | 3.3% | 18.3% | **81.8%** (9/11) |
| S_formula | 447,360 | 40 | 8.3% | 25.0% | **66.7%** (10/15) |
| S_untrained | 447,360 | 0 | 0.0% | 0.0% | 0/0 — no formatted answers |

Interpretation: if the formula model's numeric_score is HIGHER and its
confident_hallucination_rate is LOWER than the same-size control model's
(especially in-distribution), training on the repository's true formulas
REDUCES hallucinations; the S->XL gap dynamics show how the effect depends
on model scale.
