# RCDA-9 for Gemma3-1B-IT: the formula round — every monograph formula, corpus and loss

Written: November 19, 2026. The fifth formula-training round on
**`google/gemma-3-1b-it`** (public mirror `unsloth/gemma-3-1b-it`, snapshot
`5b11413a…`, Gemma license) and the second executed end-to-end through
[**LlamaFactory 0.9.5**](https://github.com/hiyouga/LlamaFactory) — the
round the repository's own theory asked for: **take every formula of the
monographs and teach it to the model**, in both senses of "teach":

- **formulas arm — the corpus.** A 95-chunk LoRA SFT corpus assembled from
  **the twelve restored monograph DOCX editions** (311 numbered equations
  and formula lines, verbatim with context), the formal-proof monograph
  (252 LaTeX theorem formulas), the 12 executable modules of `src/rmt_llm`
  and 105 computational pairs `"func(params) => result"` computed by the
  library itself. Stock `llamafactory-cli train`, pt stage, LoRA r=8
  all-linear (the RCDA-8 data-arm geometry) — 6.52 M trainable parameters,
  285 steps.
- **pen arm — the loss.** The repository's formula
  `pen = (vᵀh/‖v‖)²`, phase-matched over the writer band, re-trained as a
  band-masked LoRA (r=16, 1.31 M parameters, 150 steps) exactly as the
  RCDA-8 round did — a from-scratch replication inside this round.

One frozen protocol, one evaluator, two **new** formula-knowledge axes
(held-out corpus PPL and a third-grid numeric QA), and a static audit of
the predecessor round.

**The result, measured on the full frozen protocol (20 bad + 20 good,
greedy 192, raw completion mode, 4-gram loops, threshold 0.5):**

| metric | base | formulas LoRA (corpus) | pen LoRA (loss) |
| --- | --- | --- | --- |
| ρ_bad (loops ≥ 0.5) | 0.60 | **0.10** | 0.80 |
| ρ_good | 0.10 | **0.00** | 0.15 |
| loop_share mean (bad 20) | 0.562 | **0.193** | 0.785 |
| loop_waste mean (bad 20), tok | 103.3 | **36.9** | 150.7 |
| hard-12 loop share | 0.827 | **0.259** | 0.825 |
| mild-8 loop share | 0.164 | **0.096** | 0.725 |
| prompts fully repaired (of 12) | — | **10 / 12** | 1 / 12 |
| KL(base ‖ arm), control | 0 | **0.079** | 0.421 |
| PPL control (mean) | 10.9 | **10.6** | 11.4 |
| corpus val PPL (held-out blocks) | 369.2 | **99.7** | 447.1 |
| QA exact (held-out grid, of 57) | 9 | **4** | 0 |
| trainable parameters | — | 6 522 880 (0.65 %) | 1 310 720 (0.13 %) |

**Four headlines.** (1) The **probe replicates exactly** — re-run from
scratch on the pinned snapshot, stage 1 lands on the committed RCDA-8
numbers: writers L10/L11/L12, z_diff 212.595 / 219.827 / 205.125, ρ_bad
0.60, mean loop_share 0.5619, band L6–L16. (2) The **formula corpus is the
strongest anti-loop intervention measured on this protocol so far** —
10/12 hard loopers repaired at 150 fewer wasted tokens per trajectory, no
migration, and the smallest abilities price of any arm in the RCDA-5/6/8/9
series (KL 0.079, PPL even improving 10.9 → 10.6). (3) The corpus **teaches
the form, not the arithmetic**: held-out corpus PPL melts 369.2 → 99.7
(−73 %) while exact numeric QA on unseen parameter grids *drops* (9/57 →
4/57) — a clean dissociation of distribution learning from computation.
(4) The **pen arm replicates every qualitative signature** of RCDA-8
(penalty −97.2 %, TW-dirty edit 16/16, cos(u₁,v_deep) ≤ 0.431, defect
migration mild-8 0.164 → 0.725) **but not the magnitudes** (1/12 vs 3/12
repairs, ρ_bad 0.80 vs 0.70, mini-bench endpoint 0.794 vs 0.529) — an
honest datum on the run-to-run variance of CPU-budget interventions. All
numbers are fields of the committed JSONs in `results/`.

## 1. Setup

### 1.1 The monographs, restored

The twelve DOCX editions behind `docs/monographs/` (M01–M06, EN+RU, plus
the merged PDF) are **not in the current git tree** — they were dropped
when the root-level `monographs/` folder was reorganized into
`docs/monographs/` (commit `cf5cb49`, November 9: only the READMEs were
moved). This round recovered all thirteen files from the git object
history (blob SHAs `695cb194…`, `16ca954…`, … `52542146…`), verified each
zip container, and extracted the formulas **verbatim**:

- numbered equations — `ρ(λ) = (q / 2π) · √((λ₊ − λ)(λ − λ₋)) / λ    (2.1)`
  — kept exactly as written, with one line of introducing context;
- the M03/M04 editions embed their equations as PNG images under
  `Equation N-M:` captions — for those, the ASCII formulation from the
  accompanying prose is used and linked to the caption (noted in the
  manifest);
- the RU editions of M03/M04 carry machine-translation damage
  («творческая рация», «ambda_max»); their formula lines are excluded and
  the EN editions serve as canon for those two monographs (noted in the
  manifest);
- one content fix against the tinygpt precedent: its RULES table printed
  `E[F2] = -1.2065…` — that is E[F1] (β=1). The library returns
  E[F2] = −1.7711, which is what the corpus now teaches.

### 1.2 The frozen protocol (unchanged)

- **Model**: `unsloth/gemma-3-1b-it` @ snapshot `5b11413a…` — the same
  weights as RCDA-5/6/8. 1.00 B params, 26 layers, d = 1152, vocab 262 144.
- **Mode**: raw completion (no chat template), greedy decoding. In chat
  mode the defect does not manifest (RCDA-5 sentinel scan).
- **Window**: `MAX_NEW = 192`.
- **Sets**: 20 bad (`bad_g01…g20`), 20 good, 5 control (RU + EN). Frozen,
  byte-identical to the RCDA-8 sets (copied, not regenerated).
- **Loop metric**: 4-gram repeat labels; a trajectory loops when
  `loop_share ≥ 0.5`; ρ_bad is the looping fraction of the bad set.

### 1.3 Stage 1 — the spectral probe, re-run (replication check)

`scripts/stage1_probe.py` — the RCDA-8 stage 1 ported verbatim, executed
from scratch (fresh greedy generations, fresh pools, fresh directions):

| quantity | this round (re-run) | RCDA-8 committed |
| --- | --- | --- |
| ρ_bad (base, greedy 192) | **0.60** | 0.60 |
| loop_share mean (bad 20) | **0.5619** | 0.5619 |
| writer layers | **L10 / L11 / L12** | L10 / L11 / L12 |
| z_diff at writers | **212.595 / 219.827 / 205.125** | 212.6 / 219.8 / 205.1 |
| band with z_diff > 130 | L6–L16 | L6–L16 |

The phase probe (early vs deep, `early_cycles = 2`) gives
cos(v_early, v_deep) ≈ 0.72–0.97 inside the band — strongly aligned, as in
RCDA-8. The penalty band stays at the pre-registered **L8–L15**
(o_proj + down_proj). *The probe numbers of the predecessor round are
genuine and replicate exactly.*

### 1.4 The validation box (provenance, stated plainly)

CPU-only sandbox: **2 cores, 4 GB RAM, no GPU**, torch 2.15.0+cpu,
transformers 5.6.0, LlamaFactory 0.9.5, peft 0.18.1. Weights bf16, LoRA
fp32, logits/losses fp32, gradient checkpointing on. Wall-clock: probe
≈ 7 min, base eval ≈ 16 min, formulas arm ≈ 26 min (285 steps), pen arm
≈ 18 min (150 steps + 3 mini-benches), arm evals ≈ 18 min each. As in
RCDA-8: **a budget run** — the *sign* of each finding is the claim.

## 2. The formula corpus

Assembled by `scripts/stage2_build_corpus.py` (deterministic, fixed
grids, no RNG):

| source | blocks / lines |
| --- | --- |
| restored monographs M01–M05 (EN+RU) + M06 (EN+RU) | 311 formula lines (M06 → val) |
| formal_proof/monograph chapters EN | 252 LaTeX formulas (ch. 1–10 train, 11–12 val) |
| src/rmt_llm doc blocks | 82 blocks |
| RULES (corrected) | 12 |
| computational pairs — train grid | 105 |
| computational pairs — val grid | 70 |
| computational pairs — held-out QA grid | 57 |

Splits: **train** = modules + rules + M01–M05 + proof ch.1–10 + train-grid
pairs (82 605 bytes); **val** = M06 + proof ch.11–12 + val-grid pairs
(23 385 bytes) — the PPL axis is perplexity on a *monograph the model
never trained on*; **QA** = 57 pairs on a third parameter grid, exact
numeric match within 5 % relative tolerance. The LlamaFactory view is 95
semantic chunks (≈900 chars, cutoff 512). Manifest with SHA-256:
`data/corpus_manifest.json`.

## 3. The formulas arm — stock LlamaFactory on the monograph corpus

- **Recipe**: `stage: pt` (raw-completion LM, matching the evaluation
  mode), packing off, LoRA r=8, α=16 on all linear projections — the exact
  geometry of the RCDA-8 data arm, so the two rounds are comparable arm to
  arm. lr 5e-5 cosine, 3 epochs = **285 steps**, loss 3.36 → 1.91
  (final-10 mean; the corpus is hard — rare tokens, two languages, three
  notations).
- **What it is not**: it is not the anti-loop dataset. Nothing in the
  corpus mentions loops, repetition or the defect. Any anti-loop effect is
  a *side effect of the domain*.

## 4. The pen arm — the formula as a loss, re-run

The RCDA-8 recipe verbatim (band-masked LoRA r=16, α=32 on
o_proj+down_proj, L8–L15, 1 310 720 trainable parameters, lr 2e-4 linear,
150 steps, anchor MSE on trailing-32 logits, β=10, cap=1.0), re-trained
from scratch inside this round on this round's probe directions.

**Training**: pen_norm 0.946 (first-10 mean) → **0.0261 (last-10 mean,
−97.2 %)** — RCDA-8: 0.954 → 0.0206 (−97.9 %). The anchor rode under the
cap (final MSE 0.41). The logged mini-benchmark (8 hardest bad prompts,
greedy 192) is **non-monotonic this run**:

| bench | ρ_mini | mean loop_share | RCDA-8 committed |
| --- | --- | --- | --- |
| step 0 | 1.00 | 0.921 | 1.00 / 0.921 |
| step 50 | 1.00 | 0.865 | 1.00 / 0.846 |
| step 100 | 0.50 | 0.566 | 0.625 / 0.612 |
| step 150 | 0.875 | **0.794** | 0.625 / 0.529 |

The full evaluation confirms what the step-150 bench already showed: this
run's endpoint is worse than RCDA-8's committed one (§5). Same recipe,
same seed, same snapshot — different run. That is what CPU-budget
stability looks like, and it is reported as such.

## 5. The one evaluator, three arms

### 5.1 The defect

- **formulas**: ρ_bad 0.60 → **0.10**, mean loop_share 0.562 → **0.193
  (−66 %)**, waste 103 → **37 tok**, hard-12 0.827 → 0.259, **10 of 12
  hard loopers fully repaired** (`bad_g01` 0.958 → 0.96 stays; nine
  others drop below 0.16), and the mild-8 subset got *cleaner than base*
  (0.164 → 0.096). No migration — the corpus is the first intervention on
  this model that repairs the hard subset *and* the mild subset at once.
- **pen**: hard-12 0.827 → 0.825 (1/12 repaired), ρ_bad 0.60 → 0.80, and
  the defect **migrated** into the mild subset (0.164 → 0.725) — the same
  migration signature as RCDA-6 (0.16 → 0.64) and RCDA-8 (0.164 → 0.691).
  Four rounds, four geometries, one invariant.

### 5.2 The abilities and the formula knowledge

- **formulas**: KL **0.079** (the smallest of any intervention arm in the
  series), PPL 10.9 → **10.6**, ρ_good 0.10 → 0.00. The formula corpus
  leaves general text essentially untouched while rewriting the
  raw-completion loop behaviour.
- **formulas, knowledge axes**: held-out corpus PPL **369.2 → 99.7
  (−73 %)** — the model genuinely learned the formula language of a
  monograph it never saw (M06, the ten-paths-to-N₍ᵣᵢₜ₎ monograph). But
  exact numeric QA on the held-out grid **fell**: 9/57 → 4/57 (partial
  11 → 6). The corpus teaches the *form* (notation, structure, style of
  the laws) — it does not teach the *arithmetic*. A budget LoRA on 95
  chunks is a stylist, not a calculator.
- **pen**: KL 0.421, PPL +4.6 % (11.4), corpus val PPL 447 (worse than
  base — the penalty actively damages the formula domain), QA 0/57.

### 5.3 The forensics (each edit as a matrix)

- **pen** (16 band tensors): rel-Frobenius **1.4–4.1 %**, TW violation
  **16/16**, cos(u₁ of ΔW, v_deep) ≤ **0.431**. Same verdict as every
  Gemma round: the optimizer finds penalty-compliant solutions that are
  neither aligned with v nor spectrally clean.
- **formulas** (52 tensors, all-linear): rel-Frobenius **0.18–0.87 %**,
  TW violation 52/52 — a data-arm edit is also structured, but an order
  smaller per tensor.

## 6. The RCDA-8 audit (static, no model)

`../audit/rcda8_audit.md` — 72 checks of every number printed in the RCDA-8
README/REPORT against the committed JSONs, plus script-logic review:
**70 PASS, 1 WARN, 1 FAIL**. The one failure: RCDA-8 REPORT §4.3 claims
the LoRA edit's rel-Frobenius is "0.4–1.5 %" while its own
`part8_eval_rcda8.json` says **1.35–4.14 %** (median 2.0 %) — the claim
understates the edit magnitude ≈3×. This round's re-run measured
1.4–4.1 %, i.e. the JSON was right and the prose was wrong. Everything
else — z_diff, ρ values, penalty descent, mini-benches, repair counts,
KL/PPL, TW ×16, cos ≤ 0.44, dataset sizes, adapter configs, loop-metric
logic — checks out against the artifacts.

## 7. What this round says

1. **Teaching the formulas as text beats projecting them away — by every
   axis.** At comparable budget and *smaller* abilities cost, the corpus
   arm repairs 10/12 vs the pen arm's 1/12 (and RCDA-8's 3/12), with no
   migration. The strongest known recipe against the greedy loop defect
   of Gemma3-1B-IT is now: fine-tune on the repository's own mathematics.
2. **Why it works is visible in the corpus itself.** The formula language
   is maximally anti-repetitive: rare tokens, dense structure, no 4-gram
   cycles. Training on it re-shapes the raw-completion prior exactly where
   the defect lives. The RCDA-8 data arm (38 curated anti-loop examples,
   5/12) pointed the same way; the monograph corpus is the same mechanism
   at full domain scale.
3. **Form is not computation.** The −73 % held-out corpus PPL against the
   4/57 exact-QA is a clean, honest dissociation: distribution learning
   and arithmetic generalization are different problems, and a budget
   LoRA buys only the first.
4. **The migration is the law; the magnitudes are not.** The pen arm
   replicates all qualitative signatures of RCDA-8 and none of the exact
   effect sizes. Budget-run digits should be read as one draw from a
   distribution, not as constants — including RCDA-8's.
5. **The audit closes the loop on authenticity.** One real discrepancy
   found and documented (rel-Frobenius prose vs JSON); the probe and base
   numbers replicate exactly; everything in this round is a committed
   JSON field.

## 8. Honest limits

CPU-scale budget run (285/150 steps, 20-prompt suite, LoRA ranks 8/16);
the pen-arm variance across identical-budget runs (this round vs RCDA-8)
shows the run-to-run noise floor directly; the QA grid, while held out,
shares its functional forms with the train grid (the pairs test
interpolation of known formulas, not new mathematics); the corpus contains
the monographs' own claim structure, so the model learns what the
monographs say — including any errors they contain (the one known
constant error, E[F1] mislabeled as E[F2] in the tinygpt RULES, was
corrected against the library before training and is documented in the
manifest).

## 9. Reproduction

```bash
# the whole round, resumable (a CPU box with 4 GB RAM is enough)
python3 scripts/pipeline.py

# or stage by stage
python3 scripts/stage1_probe.py                               # probe (re-run)
python3 scripts/stage2_build_corpus.py                        # corpus (rebuild)
llamafactory-cli train configs/sft_formula_cpu.yaml           # formulas arm
python3 scripts/stage3_pen_train.py                           # pen arm
python3 scripts/stage4_eval.py --tag base                     # evaluator
python3 scripts/stage4_eval.py --tag formulas \
    --lora results/formulas_arm/gemma3_formula_lora
python3 scripts/stage4_eval.py --tag pen --lora results/pen_arm/adapter
python3 scripts/stage5_summarise.py                           # summary
python3 scripts/make_figures.py                               # figures + GIFs
python3 ../audit/rcda8_audit_runner.py                       # the RCDA-8 audit

# full-scale GPU replication (>= 16 GB VRAM)
llamafactory-cli train configs/sft_formula_gpu.yaml
python3 scripts/stage3_pen_train.py --steps 600 --lr 3e-4
```

## 10. Artifacts

| path | contents |
| --- | --- |
| `results/probe/` | fresh generations, `probe_report.json` (writers L10–12, z_diff 212.595/219.827/205.125), `probe_phase_report.json`, direction vectors (npz) |
| `results/formulas_arm/gemma3_formula_lora/` | the trained corpus LoRA + `trainer_state.json` (loss 3.36 → 1.91) |
| `results/pen_arm/` | the trained band LoRA + `training_log.json` (150 steps, −97.2 %, 4 mini-benches) + anchor cache |
| `results/part9_eval_{base,formulas,pen}.json` | the one evaluator, three arms — every number of §5 |
| `results/qa_{base,formulas,pen}.jsonl` | per-pair QA generations (prompt, completion, expected, hits) |
| `results/summary_round9.json` | the cross-arm summary |
| `data/` | the frozen sets, the corpus (train/val/QA), manifest with SHA-256 |
| `figures/` | 10 static figures + 6 animations, rebuilt from the JSONs |
| [`../audit/`](../audit/README.md) | the RCDA-8 static audit (JSON + MD protocol) |

---

<p align="center">
  <sub><a href="README.md">↑ the RCDA-9 round</a> ·
  <a href="../README.md">the kit root →</a> ·
  <a href="../LlamaFactory/RCDA8_REPORT.md">the RCDA-8 report →</a></sub>
</p>
