# Scaling + Hallucination A/B — Results

Generated: 2026-10-22 16:40:03

The key metric is **confident_hallucination_rate**: the share of formatted
answers (format_ok=True) with wrong numbers (the "Utility Trap", ±15% tolerance).

## HELD-OUT (37 fresh tasks — the parameters were NOT in the corpus)

| Cell | Parameters | Epochs | val_loss | numeric_score | format_score | **confident hallucinations** | teacher_forced |
|---|---|---|---|---|---|---|---|
| S_control | 447,360 | 40 | 2.039 | 0.0% | 64.9% | **100.0%** (24/24) | 35.6% |
| S_formula | 447,360 | 40 | 1.954 | 0.0% | 40.5% | **100.0%** (15/15) | 36.7% |
| S_untrained | 447,360 | 0 | — | 0.0% | 0.0% | 0/0 — no formatted answers | — |
| M_formula | 1,107,712 | 60 | 2.461 | 2.7% | 13.5% | **80.0%** (4/5) | 37.3% |
| M_untrained | 1,107,712 | 0 | — | 0.0% | 0.0% | 0/0 — no formatted answers | — |
| L_formula | 4,055,424 | 40 | 2.332 | 2.7% | 32.4% | **91.7%** (11/12) | 36.0% |
| L_untrained | 4,055,424 | 0 | — | 0.0% | 0.0% | 0/0 — no formatted answers | — |

## IN-DISTRIBUTION (pairs from the corpus — the model SAW them in training)

The main question is checked here: does the model reproduce the TRUTH
where it has knowledge? The control model learned the lie — and formats
it confidently.

| Cell | Parameters | Epochs | numeric_score | format_score | **confident hallucinations** |
|---|---|---|---|---|---|
| S_control | 447,360 | 40 | 3.3% | 18.3% | **81.8%** (9/11) |
| S_formula | 447,360 | 40 | 8.3% | 25.0% | **66.7%** (10/15) |
| S_untrained | 447,360 | 0 | 0.0% | 0.0% | 0/0 — no formatted answers |
| M_formula | 1,107,712 | 60 | 6.7% | 16.7% | **60.0%** (6/10) |
| M_untrained | 1,107,712 | 0 | 0.0% | 0.0% | 0/0 — no formatted answers |
| L_formula | 4,055,424 | 40 | 6.7% | 21.7% | **69.2%** (9/13) |
| L_untrained | 4,055,424 | 0 | 0.0% | 0.0% | 0/0 — no formatted answers |

Interpretation: if the formula model's numeric_score is HIGHER and its
confident_hallucination_rate is LOWER than the same-size control model's
(especially in-distribution), then training on the repository's true
formulas REDUCES hallucinations; how the gap moves across the S→XL sizes
shows how the effect depends on model scale.

---

## Addendum (2026-10-22): M/L cells trained, H1 replicated across the ladder

The M/L cells were trained on 2 CPUs with resume chunks: M — the full
ladder specification (60 epochs); L — a shortened but completed cosine
schedule (40 epochs, parity with the S budget; the 150-epoch ladder
targets GPUs; the override is recorded in `cell_report.json`). The
cells share the runner's BPE tokenizer (functionally identical to the
canonical one on corpus bytes; a byte-level difference in the tail
merges does not affect corpus encoding).

The predictive H1 check at these scales —
`neural_lab/science/results/h1_scaling/REPORT.md` (the same 113-task
dataset): the direction D > B replicated in 3/3 cells, and the L
bootstrap CI of ΔAUC excludes 0. XL (19.1M) requires a GPU; the pipeline
is smoke-verified.
