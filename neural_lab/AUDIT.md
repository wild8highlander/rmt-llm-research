# AUDIT — neural_lab: factual state (October 2025)

> An architectural and scientific audit of what `neural_lab/` actually
> contains, performed against the code (not the documentation). It
> answers the development plan's questions: what is really computed,
> what is merely claimed, where the synthetic parts are, where the
> duplicates are, and which defects are critical. Final recommendations
> are in `PREDICTION_ROADMAP.md`; the first repaired experiment is in
> `EXPERIMENT_PROTOCOL.md` + `science/`.

---

## 1. What neural_lab actually is

`neural_lab/` consists of two independent parts:

| Part | Technology | Role |
|---|---|---|
| Web app (`src/`, Next.js 15 + TS + Tailwind + shadcn/ui) | 6 tabs: Overview, Playground, Training, Benchmark, Questions, Deployment | a showcase for the trained model; the Playground calls `research/tinygpt_formula/scripts/infer.py` through an API route |
| Research program (`research/tinygpt_formula/`) | pure Python 3.10+ / NumPy | a 5-stage pipeline: corpus → training → evaluation → inference → open questions; plus `scaling/` (the S→XL ladder + the A/B "formulas vs hallucinations" study) |

The separation is sound and healthy: the research folder is
self-contained (it can be deleted without damage), and the web app only
reads artifacts (`fs/promises` in the API routes). The DB layer
(`src/lib/db.ts`, Prisma) is template code from the generator — it is
not used by the lab tabs and does not work without `prisma/schema`; it
is junk, not a defect of the science (recommendation: delete it or
explicitly mark it as unused).

The artifacts in `research/tinygpt_formula/model/` are committed (JSON);
the `*.npz` weights are deliberately gitignored (2 MB — below the
threshold of usefulness), but their reproduction is documented and
deterministic: seed 42, ~17 min CPU, val_loss 1.954 reproduces on another
machine to within ~1e-3.

## 2. The run → result path (traced)

`build_corpus.py` (277 pairs from 10+ core functions + monograph
formulas) → `train_formula_model.py` (TinyGPT v3, per-epoch history,
resume) → `evaluate_model.py` (held-out benchmark of 37 tasks + gains +
spectral probe) → `open_questions.py` (Q1–Q4) → `infer.py` (web).

Verified: the web app's charts and tables are reconstructed from the
saved JSON files (training_history.json, evaluation_report.json,
open_questions_results.json, scaling_report.json) — there are no
hand-written summaries. This satisfies the "reports from data, not from
hands" requirement.

## 3. Synthetic objects and stubs — actual status

| Object | Status | Verdict |
|---|---|---|
| `reasoning_trace` in the docstring of `generate()` (lab_en/tiny_gpt_v3.py) | **claimed but never returned** — there is no synthetic trace in the code, only a misleading docstring | ✅ fixed in this audit (docstring aligned with the facts) |
| "Spectral probe" λ_max vs the MP bound | computed on real hidden states (capture_hidden) | a real measurement |
| "Free probability" (Q1, R-cumulants) | κ1/κ2 on real covariances; the "free estimate of N_crit" = κ2/κ1² is a heuristic, not free probability in the strict sense | a real measurement; the interpretation is inflated by the "SOLVED" wording |
| Q2 Caputo-SFT | μ_SFT is estimated from the loss curve; the RLHF branch is analytics only — no experiment | partially confirmed; the "SOLVED" status is overstated |
| Q3 "lie detector" | real λ_max for known vs unseen, but 4×4 tasks; AUC 0.75 on one trajectory and 0.29 on another | the measurement is honest, the statistics are negligible (acknowledged in the README) |
| Confidence/proxy-perplexity in infer.py | cross-entropy of the model's own tokens | a real measurement |

Main conclusion: **there is no crude synthetic data** — every metric is
computed from documented tensors. The problems are in (a) the misleading
docstring, (b) inflated "SOLVED" statuses under weak statistics, and
(c) one mathematical defect (§4).

## 4. Mathematical correctness of the RMT calculations

One systematic convention defect was found (reproduced in
`evaluate_model.py`, `open_questions.py`, `spectral_analysis()`):

**Rank-deficient covariances.** Hidden states have T positions ≪ H=96,
so the (H×H) covariance has rank ≤ T−1; `sigma2` was computed as the
mean of the **non-zero** eigenvalues after filtering — this inflates σ²
by a factor of H/(T−1) and, with it, the MP bound, deflating the
gap_ratio. At T=64, H=96 the bias is ≈ 1.5×: pure noise receives a
gap_ratio ≈ 0.6 instead of ≈ 1.

The fix (implemented in `science/features/spectral.py`, verified with
control zeros): σ² = trace(cov)/H — the full variance; under this
convention white noise gives gap_ratio ∈ [0.85, 1.15] and a planted
spike is detected. **The research scripts were left as they are**
(historical reports must correspond to the code as it was at run time);
all new computations go through `science/`. For the old reports: the
conclusions about "spikes after training" survive (the spike ≫ the
bias), but the absolute gap_ratio values in the old artifacts are
deflated.

Additionally verified: `mp_bounds`, `bbp_lambda_max`, `caputo_*`,
`tracy_widom_*` are numerically consistent with one another and are used
as an independent "verifier" for H1 task labeling (corpus right-hand
sides cross-checked against recomputation: no discrepancies found).

## 5. Thresholds and fitting: what about N_crit = 114 and "γ₁/θ_b"

Values like `ncrit_threshold` in the laboratory interfaces are heuristic
descriptors not derivable from a fixed data model; the definitions of N
(tokens / sample / dimension) are nowhere fixed. In the hypothesis
registry (`science/hypotheses/registry.yaml`, H4) this is made explicit:
until the definitions are pre-registered, any agreement of thresholds
counts as a coincidence of metrics. A direct consequence for the web
app: the Questions tab must label the Q1–Q4 statuses as "empirical on a
single trajectory" rather than "SOLVED" (interface fix — in the roadmap,
stage 06).

## 6. Dependency map and duplication

```
tiny_gpt_v3.py (lab_en) ←─ train/evaluate/open_questions/infer/scaling
src/rmt_llm/* (mp, bbp, cap, tw, ...) ←─ the same + science/data (verification)
```

Duplications (the canon is now `science/`):

| Function | evaluate_model.py | scaling_experiment.py | science/ (canon) |
|---|---|---|---|
| gt_pairs() | ✔ | ✔ (copy) | `data/tasks.py::far_band_tasks` |
| format_ok/numeric_ok | ✔ | ✔ (with underscores) | `data/tasks.py` (+ echo-segment fix) |
| quick_eval | ✔ | ✔ (its own copy) | not needed (metrics live in evaluation/) |
| spectral probe | ✔ | ✘ | `features/spectral.py` (the corrected convention) |

The duplication risk materialized during this very audit: the research
scripts' copy of `numeric_ok` compares numbers across the entire
generated string, including the echo of the prompt tail — this inflates
negatives (see EXPERIMENT_PROTOCOL.md, the v1.0.1 changelog).
Canonicalization into `science/` is recommended to continue.

## 7. The "done / partial / hypothesis" separation

| Module | Status | Basis |
|---|---|---|
| Corpus + BPE + training (research) | ✅ done | determinism, resume, history, reproduced |
| Held-out benchmark (research) | ✅ done | reproducible, honest tolerances |
| Spectral probe (research) | ⚠️ partial | the σ² convention is biased (§4), single model |
| Q1–Q4 (research) | ⚠️ partial | the measurements are real, the "SOLVED" statuses are inflated, the statistics are 4×4 |
| Scaling ladder | ⚠️ point S only | M/L/XL had not been run |
| Web app | ✅ works | reads artifacts; Prisma is junk |
| **science/ (new)** | ✅ done | 52 tests; a full H1 run is fixed in `science/results/h1_v1/` |

## 8. Critical defects found and eliminated

1. **The σ² convention for rank-deficient covariances** (§4) — fixed in
   `science/features/spectral.py`, covered by a control-zero test.
2. **Prompt echo in numeric labeling** — the answer is segmented at the
   last `=>` (`science/data/tasks.py`); the research copies keep the old
   behavior (historical reports).
3. **Reusing one target's predictions for another** (the secondary
   format_ok target received constants from a degenerate primary fit) —
   fixed: each target is fitted separately.
4. **A leak between task bands**: `tracy_widom_mean()` was in both
   indist and far — the leakage check caught it on the first run;
   prompt deduplication across bands was added.
5. **The reasoning_trace docstring** — fixed (§3).
6. **The label permutation test** returned p≈1 silently under
   single-class strata — randomization-space diagnostics added.

## 9. Prioritized backlog

1. (high, partially done) Run H1 on the M/L cells of the scaling
   ladder — done (see `science/results/h1_scaling/REPORT.md`): positives
   14 → 19 (+36%, not "an order of magnitude" — the target remains rare
   at a ~3.4% base rate); the effect direction replicated 3/3, and the L
   CI excludes 0. Remaining: XL requires a GPU; power growth is now tied
   primarily to task-set expansion (≥ 40 positives), not just model size.
2. (high) Reduce Q3 to an H1-style protocol: pre-register
   known/fabricated sets (dozens of tasks per class), fix the weights.
3. (medium) Canonicalization: the research scripts should import
   `science.data.tasks` instead of their copies (after the historical
   reports are pinned with a tag).
4. (medium) Web app: a "Science" tab from `results/h1_v1/report.json`;
   delete/disable the Prisma layer; label the Q1–Q4 statuses.
5. (low) CI: a dedicated `science` job (`pytest neural_lab/science/tests`);
   an integration test with real weights only when the artifact is
   present (currently: skip — weights are not in git).
