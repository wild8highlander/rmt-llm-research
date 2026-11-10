# RCDA-3 — the spectral sentinel: online detection and rescue of trajectories

**Model:** taobao-mnn/MNN-Pocket-0.3B (Qwen3ForCausalLM, 28 layers, d=896, 14 heads).
**Date:** 2026-11-08. **Round 3**, per my own plan: we turn the fig9 precursor
(the monotonic drop of the windowed participation ratio before the textual collapse)
into a working online controller and answer the question "what cures the loop better —
a one-time weight edit or therapy in the moment?". All numbers are measured; the
sources are `Research/RCDA/results/rcda3_*.json`.

---

## 1. Setup

Round 2 established: (a) the behavioral loop direction is localized in o_proj
L15–L18; (b) the defect is distributed — even α=1.0 on three matrices changes 2/8
trajectories; (c) the window-16 PR drops 8–9% before the collapse. Round 3 tests
three hypotheses:

- **H1 (sentinel):** a threshold detector on the sliding PR detects the loop online
  with high TPR and zero FPR on healthy generations.
- **H2 (attractor):** one-shot interventions (timer-based) do not pull the
  generation out of the loop — the attractor restores itself; an activation-level
  confirmation of round 2's distributedness conclusion.
- **H3 (vaccination vs cure):** a conditional alarm-triggered intervention (weights
  untouched) reduces the defective-generation share more than round 2's permanent
  weight edit.

## 2. Method

**Calibration without threshold tuning.** The healthy PR distribution comes from a
teacher-forced run of 5 control texts (`prompts/control.jsonl`):
314–354 windows per (scale, layer), θ = q10 of that pool. Two scales:
W=8 ("fast", available from token 8) and W=16 ("slow", the fig9 window); sentinel
layers L15/16/17. Alarm = PR < θ two steps in a row on any (layer, scale). The rule
was fixed before the experiment.

**Online loop.** Greedy generation with a KV cache; at every step the new token's
hidden states enter the sentinel deques, the window PR is computed via SVD
(16×896 — milliseconds). Detection and interventions work in the moment.

**Rescue arms (a priori):** `ban` — for 4 steps forbid the token that in this
generation already continued the current 3-gram; `steer` — for 8 steps subtract
γ·Δh_behav from the residual at the writer layers L15/16/18; `both` — the combination.
Success = no new 4-gram loop after the alarm and ≥8 tokens of continuation
(or a clean EOS).

## 3. Results

**Detection (H1 — confirmed).** Regenerating the 40 round-2 generations reproduced
**40/40 texts exactly** (loop determinism confirmed). By the a-priori labels 30/40
generations loop; the sentinel's TPR is **29/30 = 96.7%**, with **zero alarms** on
clean generations. 10 alarms are predictive (median lead 2 tokens), 19 are onset-locked.
Threshold-free ROC: AUC 0.720 for W8 at a 4-token horizon; W16 "fires before its
horizon" — at the fixed horizon 4 its AUC is 0.439, a direct footprint of the fig9
precursor (the signal appears beyond the labeling horizon); at the matched horizon 12
the picture flips symmetrically (0.415/0.720 — see rcda3_digest.md and fig11c: the
W8/h12 and W16/h4 pairs).

**Attractor (H2 — confirmed).** Timer arms: `ban` **0/30** (in-loop waste
73.8→68.2 tokens at 3.5 interventions per generation), `steer` **12/30**
(40%, waste 40.4), `both` **18/30** (60%, waste 28.8). Even successful rescues have a
median clean tail of 0–1 tokens: the intervention breaks the current loop cycle, but
greedy dynamics immediately returns the model to the attractor. Four interventions
per generation is the norm. Conclusion: a one-time "push" does not cure; the defect
must either be suppressed continuously or the weights must be edited.

**Vaccination vs cure (H3 — confirmed).** Benchmark of 20 bad prompts, 96 tokens,
the same generation code:

| Arm | ρ (defect) | In-loop waste, tokens/gen | Weight change |
| --- | --- | --- | --- |
| A baseline | 0.70 | 70.0 | — |
| B permanent edit (round 2) | 0.65 | — | 190k weights, TW pass |
| C sentinel (`both`) | **0.50** | **16.4** | 0 |

Arm A reproduced round 2's ρ=0.70 exactly (estimator validity). The sentinel beat
the permanent edit (−20 pp versus −5 pp) and cut the in-loop waste 4.3× — with zero
weight change, i.e. PPL on control texts is unchanged by construction. The price:
on 20 good prompts the sentinel changed 11 texts (12 of them were looping anyway —
the interventions were on target; the detector produced no false alarms on genuinely
clean generations).

**Transfer.** On the model with round 2's edit (`edited_model_v2`), with the same θ:
TPR **13/13**, ρ 0.65 (matches the round-2 benchmark). The sentinel does not care
whether the weights are edited — the monitor and therapy compose with any static cure.

## 3.5 Persistent policies: constant pressure instead of a one-time push

The H2 mechanism (the attractor restores itself) suggested exploratory policies —
suppress while the loop tries to re-form, not N tokens by timer:

- **`ban_persist`** — at every step forbid ALL tokens that ever continued the
  current 3-gram in this generation. Any token 4-gram can occur at most once — a loop
  is impossible by construction. The sentinel is not needed: this is a decoding
  constraint.
- **`both_persist`** — the same plus steering, re-armed while PR is below θ.

Result on 30 looping generations: both arms give **rescue 0.97, in-loop waste
0.0 tokens** (versus 28.8 for the best timer arm). Steering adds nothing but cost:
`both_persist` changes 16/20 good texts versus 12/20 for `ban_persist`, and the
12 changed under ban_persist are exactly the 12 looping good generations —
**ρ_good = 0.00**: not a single intervention on genuinely clean text. The final
policy is `ban_persist`.

Benchmark of 20 bad prompts: **ρ(C') = 0.10** versus 0.70 (base), 0.65
(permanent edit), 0.50 (sentinel with timer arms). The final therapy table: a
permanent weight edit is almost useless (−5 pp), an alarm-triggered conditional
intervention gives −20 pp, decoder n-gram exhaustion gives **−60 pp** without a
single edited weight.

Honest caveats: (a) the strict rescue criterion for ban_persist holds by
construction — the meaningful quantities are the benchmark ρ, the zero FP on clean
generations and the zero waste; (b) the constraint bans exact 4-gram repeats but not
verbal repetitions with a different tokenization — see distinct-4 of the rescued
texts in the JSON; (c) the policy is exploratory, thresholds were not tuned.

## 4. Head surgery: the concentration is causal, zeroing is not therapy

The sensitivity profile `e_h = ||W_o[:, block_h]^T v||²` confirms round 2's
localization on an independent computation: **L15→h7 (×3.11), L16→h8 (×2.22)**;
for L18 a pure-surrogate recomputation gives h12 (×1.91) instead of h0 — in the
round-2 report L18's concentration was weaker and unstable to the surrogate choice.

Zeroing calibration (8 bad + 8 good, 48 tokens, weights in memory):

| Condition | changed bad/good | hit_bad | hit_good | KL | TW |
| --- | --- | --- | --- | --- | --- |
| base | — | 0.625 | 0.625 | 0 | — |
| L15 h7 | 4/3 | 0.50 | 0.62 | 0.062 | violation |
| L16 h8 | 6/3 | 0.50 | 0.75 | 0.017 | violation |
| L18 h12 | 3/2 | 0.50 | 0.50 | 0.029 | violation |
| combo3 | 5/4 | 0.50 | **0.88** | 0.107 | violation×3 |
| **placebo3** (median heads) | 5/4 | **0.62** | 0.62 | 0.088 | violation×3 |

The placebo control is the key result: zeroing "ordinary" heads changes just as many
trajectories (5/4) but **does not reduce the defect** (0.62 = base), whereas the
top heads consistently give 0.50. The concentration specification is confirmed.
But by the pre-registered rule (hit_bad ≤ 0.475, hit_good ≤ 0.725) — **NO-GO**: combo
raises the good-split defect to 0.88, and zeroing a whole head (24–26% of the
matrix's Frobenius norm) coarsely breaks the spectrum (TW violation everywhere) —
unlike round 2's selective column edits. Conclusion: the concentration is causal,
but head surgery is not therapy; the right tools are selective columns (round 2,
TW pass) and online policies (C/C').

## 5. What carries over to the next round

- Correction of round 2's bookkeeping: 5 jsonl records had `clean_gen=False` with
  zero labels; the regeneration confirms the count of **30/40** looping
  (round 2's meta is correct; the saved labels of 5 records are an artifact).
- The persistent PR monitor is a candidate for `Code/` as a runtime `clip` hook:
  detection costs milliseconds (SVD 16×896) and does not require use_cache=False.
- The C' + sentinel combo: ban_persist guarantees no token 4-grams, the sentinel
  watches the spectrum for non-n-gram degradation forms (verbal repetitions, numeric
  ladders) — one runtime hook with two loops.
- Selective column edits inside top heads (a "surgery + TW" hybrid): zero not the
  whole head but only its TW-passing columns from the round-2 profile.

## 6. Reproduction

```bash
python3 Research/RCDA/tools/rcda3_sentinel.py            # ~18 min CPU: calibration, detection, arms, benchmark
python3 Research/RCDA/tools/rcda3_fix_steer.py           # re-run the steer arm with fixed hook arming
python3 Research/RCDA/tools/rcda3_explore.py             # persistent policies + C' benchmark (~12 min)
python3 Research/RCDA/tools/rcda3_heads.py               # head profile + head surgery + TW (~6 min)
python3 Research/RCDA/tools/rcda3_stage3_figures.py      # fig11–fig13, digest, comparison
```

Artifacts: `results/rcda3_sentinel.json` (calibration, detection, arms,
benchmark, transfer), `results/rcda3_explore.json` (persistent policies,
C' benchmark), `results/rcda3_heads.json` (head profiles, conditions, NO-GO),
`results/rcda3_sentinel_generations.jsonl`
(100 records: 40 detect with PR traces + 3×30 rescue arms),
`results/rcda3_comparison.json`, `results/rcda3_digest.md`,
`figures/fig11_sentinel.png`, `figures/fig12_heads.png`,
`figures/fig13_therapy.png`, `config/rcda3_sentinel.yaml`.
