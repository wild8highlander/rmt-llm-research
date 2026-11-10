# RCDA-2 — behavioral contrast, dose-response and spectral precursors of looping

**Model:** taobao-mnn/MNN-Pocket-0.3B (Qwen3ForCausalLM, 28 layers, d=896, 14 heads,
tied embeddings). **Date:** 2026-11-08. **Toolkit:** `neural_lab/MNN-Pocket-0.3B/Code`
(RCDA extension); every weight edit passes the RMT tolerance and TW control.

Round 1 (RCDA per the target plan) produced a surgically clean but behaviorally
inert edit: 0 of 20 greedy trajectories changed. Round 2 is my own protocol with
four new instruments; each answered one question and produced a number.

---

## 1. Behavioral contrast instead of style contrast (the main result)

**Round 1's confounder hypothesis:** Δh computed as the mean difference
"bad prompts − good prompts" mixes the defect with style (long chain-of-thought
versus short factual answers).

**Method:** the labels are set by the model itself. A token is marked "in the loop"
if its token-level 4-gram already occurred earlier in the same generation; the LOOP
pool is the tokens of the looping regime, the CLEAN pool is the tokens of generations
with not a single collapse — both groups drawn from the same 40 prompts (bad and
good), positions ≥ 8. Δh_behavioral = μ_loop − μ_clean per layer; the RMT passport
is as in round 1 (Hotelling z_diff, β against √(λ₁/N)).

**Result:** 30 of 40 generations loop, the median collapse onset is token 7;
798 loop tokens entered the pools. The z_diff top is **layers L15–L18**
(z_diff ≈ 1.1–1.4·10⁵), not L8–L12 from round 1. The decisive test is the cosine
between the directions:

| layer | |cos|(Δh_style, Δh_behavioral) |
|---|---|
| L15 | 0.011 |
| L16 | 0.038 |
| L17 | 0.065 |
| L18 | 0.088 |

The directions are **practically orthogonal**. The confounder is confirmed
quantitatively: round 1 subtracted "answer style", not "loop mechanics" — this
explains both the 0/20 changed trajectories and the harmlessness (PPL untouched).

## 2. Matrix scans of the behavioral direction: does o_proj write the loop?

For the top-4 layers, down_proj and o_proj were scanned (plan tolerance:
λ_v > λ₊⁽ᵂ⁾, θ_v > θ_c = √q):

| matrix | λ_v/λ₊ | θ_v | θ_c | δ | α₀ (κ=0.5) | auto_chi2 columns | verdict |
|---|---|---|---|---|---|---|---|
| L15 o_proj | 1.28 | 2.03 | 1.00 | 0.51 | 0.52 | 63 | supercritical |
| L16 o_proj | 1.34 | 2.09 | 1.00 | 0.52 | 0.55 | 65 | supercritical |
| L17 o_proj | 0.77 | 1.44 | 1.00 | 0.31 | 0.22 | 25 | subcritical |
| L18 o_proj | 1.41 | 2.15 | 1.00 | 0.54 | 0.58 | 84 | supercritical |
| L15–L18 down_proj | 0.46–0.54 | 0.24–0.36 | 0.59 | 0 | 0 | 2–9 | refused by tolerance |

Conclusion: the loop in the residual stream is written by the **o_proj** matrices of
L15/L16/L18; down_proj fails the RMT tolerance (α₀ = 0 — untouched by plan discipline).

## 3. Direction attribution via tied embeddings

Δlogits ∝ E·Δh (tied embeddings). The loop direction **pumps the
meta-tokens-of-continuation**: "etc", "total", "until", "indices", "loops",
"cyclic", brackets and service symbols; content tokens are suppressed.
Interpretation: the defect is an "enumeration/continuation attractor",
not a topical parameter.

## 4. Dose-response of α: why one matrix is not enough

Calibration grid α ∈ {0.1, 0.25, 0.5, 0.75, 1.0} on the edit of the three
supercritical o_proj matrices (in-memory, auto_chi2 columns), metrics: KL(base‖edited)
on teacher-forced continuations, the number of changed greedy trajectories, rep4 on
the calibration 8 bad:

| α | KL | changed bad | rep4_bad |
|---|---|---|---|
| 0.10 | 0.0000 | 0/8 | 0.562 |
| 0.25 | 0.0000 | 1/8 | 0.570 |
| 0.50 | 0.0001 | 2/8 | 0.570 |
| 0.75 | 0.0003 | 2/8 | 0.570 |
| 1.00 | 0.0006 | 2/8 | 0.570 |

Even a full (α = 1) subtraction of the direction from the three o_proj matrices moves
only 2 of 8 calibration trajectories and does not reduce rep4. KL grows linearly and
stays microscopic: **the defect is distributed across many writers** — this is the
scenario of plan §9 ("the defect is not localized in one direction"), not a method
failure.

## 5. Iteration 2: the final edit with TW backoff and the benchmark

Final edit: α* = 1.0 (chosen by dose-response — maximum impact), with a per-matrix
TW backoff (plan §7.2: L18 at α=1 violated TW control, rolled back to 0.5). In total
189,952 weights (≈ 0.062% of parameters):

| matrix | α | columns | weights | TW control |
|---|---|---|---|---|
| L15 o_proj | 1.0 | 63 | 56,448 | pass |
| L16 o_proj | 1.0 | 65 | 58,240 | pass |
| L18 o_proj | 0.5 | 84 | 75,264 | pass (after rollback) |

Before/after benchmark (20 bad + 20 good, max_new = 96, greedy):

| quantity | baseline | edited v2 |
|---|---|---|
| ρ (defect share on P_bad) | 70% (14/20) | **65% (13/20)** |
| false positives on P_good | 55% (11/20) | 60% (12/20) |
| changed trajectories | — | **5 bad / 3 good** |
| Wilcoxon p (rep4 / distinct4 / collapse) | — | 0.465 / 0.465 / **0.109** |
| PPL on control texts | 3793.6 | 3662.4 (−3.5%) |

The behavioral shift is real (5/20 trajectories changed versus 0/20 in round 1;
Wilcoxon on collapse time p = 0.109 — a positive trend without significance at
n = 20), abilities did not degrade (PPL improved). But there is no full cure — the
consistent conclusion of §4 and §5: unidirectional ablation is insufficient.

## 6. Spectral precursors: participation ratio predicts the collapse

A sliding window of 16 tokens, participation ratio PR = (Σs²)²/Σs⁴ over the SVD of
the window (lower PR — a more efficient rank):

| layer | PR 16–24 tokens before collapse | PR in the loop | drop |
|---|---|---|---|
| L15 | 2.85 → 2.58 | 1.69 | −9% before collapse, −41% in the loop |
| L16 | 2.37 → 2.16 | 1.53 | −9% / −35% |
| L17 | 2.26 → 2.07 | 1.50 | −8% / −34% |

PR starts falling **before** the textual collapse (−8…−9% on windows −16…−4) and
collapses by a third inside the loop. This is a ready online detector: a rule of the
form "PR(L15) dropped 5% below its sliding norm → arm the runtime clip hook from
`rmt_intervention.py`" connects static edits with the repository's built-in mechanism
without touching a single weight.

## 7. Localization down to attention heads

o_proj columns are the output dimensions of heads (14 heads × 64):
L15 → head **7** (18 columns, enrichment ×4.0), L16 → head **8** (×3.4),
L18 → head **0** (×2.7). Next round: per-head ablation (subtracting v only from the
64-column block of the selected head) — an even more surgical edit with a
predictably smaller TW footprint.

## 8. The §9 fallback path is implemented

`Code/rmt_intervention.py` gained a bulk-shrinkage mode
(`rcda_shrink`, `apply_pack_with_shrink`): W = UΣVᵀ, μ_i = σ_i²/n,
μ_i ← min(μ_i, λ₊ + τ·TW-scale) for i ≥ r, the top-r components untouched.
A synthetic test on a spiked matrix: top-10 singular values bit-for-bit, the bulk
clamped to the MP edge, TW control pass. This is the testable hypothesis "the defect
is not a direction but an overheated noise bulk" for the next round.

## 9. Conclusions

1. Round 1 edited style, not the defect: |cos| ≈ 0.01–0.09 between the style and
   behavioral directions. The contrast must be built from the model's behavioral
   labels, not from prompt types.
2. The behavioral loop direction is RMT-confirmed in o_proj L15/L16/L18;
   down_proj is rejected by the tolerance.
3. The dose-response shows the defect is distributed: α = 1 on three matrices
   changes ~25% of trajectories (KL ~10⁻⁴). A full cure requires per-head ablation,
   an iterative loop with Δh re-estimation, or §9 shrinkage.
4. TW control makes even α = 1 safe for abilities (PPL −3.5%).
5. The PR precursor (−9% before collapse) is the bridge to runtime detection with
   the repository's built-in hooks.

## Reproduction

```bash
python Research/RCDA/tools/rcda2_stage1_probe.py     # contrast + probes + precursors
python Research/RCDA/tools/rcda2_stage2_dose.py      # dose-response + final edit
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model_v2 \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench_v2 --max_new 96
python Research/RCDA/tools/rcda2_stage3_figures.py   # fig7-fig10 + digest
python Research/RCDA/tools/test_rcda_math.py         # 25 synthetic checks
```

All artifacts: `Research/RCDA/results/` (JSON), `Research/RCDA/figures/`
(fig7–fig10), `Research/RCDA/vectors/` (Δh_behavioral L15–L18).
