# 🧪 EXPERIMENT_PROTOCOL — pre-registered before the first training step

The parent program's rule: **register the pipeline before the run, measure
after, never re-fit the story to the numbers.** This document is the OCRia
kit's registration. Deviations discovered *during* the run are recorded in
[`AUDIT.md`](AUDIT.md) as measured lessons, together with the control runs
that motivated them.

## 0. Subject

`llamaindex/OCRia-nano-1` — an 8.1M-parameter document-parsing stack:
line detector, script router (4 groups + flip), four CTC recognisers
(latin 791 chars, cjk, rtl, world), LayoutNet. Files pinned by sha256 in
[`RUN_PASSPORT.json`](RUN_PASSPORT.json).

Trainable organ: `rec_latin` (the latin CTC recogniser, 0.8M params, int8
QDQ ONNX → fp32 rebuild, parity-gated). Frozen organs: detector, router,
cjk/rtl/world recognisers, LayoutNet (out of scope — internal features).
The router freeze is lifted **only** by registered addendum §8 (round 3):
a head edit that adds a 5th `reject` class while keeping the trunk and the
flip head bit-frozen.

## 1. Corpora (registered before any model run)

| batch | n | content | role |
|---|---|---|---|
| clean | 120 | rendered latin text (5 fonts, sizes 15–23), fitted so GT = visible glyphs | abilities / anchor |
| degraded | 60 | clean + blur/noise/lowres/contrast | F3 misreads |
| structured | 120 | 7 non-text pattern families (scribble, wave, halftone, lineart, grid, seal, band) | F1/F2 |
| noise | 60 | gaussian/uniform/speckle/checker/stripes/snp | F1 |
| blank | 30 | white/gradient/vignette | F1 specificity |
| control | 40 | held-out rendered text | parity corpus (MSE/KL) |
| router | 200 | 4 scripts × 20 text + 40 non-text + flipped copies | F4/F5 |

Seed 7. Ground truth exact by construction.

## 2. Families and endpoints (registered)

- **F1** phantom text: `phantom_rate` over the 210 non-text crops (primary
  endpoint), `phantom_char_rate`, blank specificity.
- **F2** loops: char 4-gram `loop_share` ≥ 0.5 → `rho` (the parent's metric,
  ported to characters), `mean_loop_share`.
- **F3** misreads: CER on clean/degraded/control; exact match on clean.
- **F4** routing: `nontext_route_rate`, routing error on text crops.
- **F5** flip: `flip_miss_rate` per script.
- **F6** over-confidence: AUC(conf) vs AUC(v·ℓ) on phantom-vs-abstained
  frames (the Utility Trap port).

Primary endpoint: **F1 phantom rate, trained arms vs base.** Abilities gate:
clean CER must stay within +0.5pp and control KL must stay < 0.1 for a cure
to count (pre-registered — the shipped arms pass at 0.05pp/0.0095 and
0.06pp/0.083).

## 3. Probe (registered)

- State labelling: emitted frames (argmax ≠ 0, conf ≥ 0.5) vs confident-blank
  frames of the same trajectories; contrast stages = the CTC-timeline stages
  (body8, mix0, mix1, mix2, head).
- v per stage: unit Δμ between the two state populations.
- Writer statistic: z_diff in pooled-σ units; writers = top-|z_diff| stages.
- Matrix scans: the parent's `spike_scan` (λ_v vs λ₊, θ_v vs θ_c) per writer
  tensor with its output-side probe direction; supercritical ⇔ λ_v > λ₊ AND
  θ_v > θ_c.
- Utility Trap: frame-level AUC contest, conf vs v·ℓ.

## 4. Formula audit (registered)

All twelve parent formulas on OCRia weights/states; N/A reported as N/A
(RMSNorm). Verdicts: pass (number matches theory within stated tolerance),
structural-finding (number is a measured property of the trained model),
N/A. No formula may be silently dropped.

## 5. Training arms (registered)

Protocol shared by both arms: Adam, 150 steps, batch 1, lr drift-matched
per arm, grad-clip 1.0, anchor β=10 cap=1.0 scope='good', frozen emission
masks refreshed every 32 steps (online self-labelling), mini-benchmark at
0/75/150, seed 0.

- **classic**: trainable = mix2.fc2; pen = head edge-energy
  `mean_frames Σ_c relu(ℓ_c − ℓ_0)² / K`, per-crop init-normalised.
- **rcda_x**: trainable = mix{0,1,2}.fc2; pen = head edge-energy + mix-stage
  one-sided v-clamps `relu(v·h − c_ref)²` to the abstained reference; PLUS
  MP-bulk leash (report), entropy-response band (±0.25 nats), single-spike
  inverse-BBP projection every 25 steps (keep top-2 intended spikes, shrink
  excess beyond 2·λ₊ — knob fixed after the first control run, see AUDIT §5).

Registered deviations-from-parent (with their control runs): the symmetric
square is replaced by the one-sided clamp (control run showed phantom
0.42 → 1.0 for the symmetric form); the pen point moved from mix2 state to
head logits (control run showed no argmax flips at mix2).

## 6. Evaluation (registered)

One evaluator for all arms (`ocria_benchmark.py`): the full family table,
abilities parity (MSE + KL on the control corpus), per edited tensor
rel_frobenius, relative TW gate (ΔKS < 0.05), cos(u₁(ΔW), v) alignment.
Frozen organs reported from the probe once.

## 7. Reporting

`ocria_summarise.py` merges everything into `summary_all_arms.json` and
`RESULTS_SUMMARY.md`; `make_graphics.py` renders figures whose every data
point is a JSON field. The README quotes only numbers that exist in
`Results/`.

## 8. Registered addendum — Round 3: the router reject round

Added after the two shipped arms, **before the round's first training
step**, by user request ("add a round for the router — F4 requires a 5th
class *reject* — this is a head edit"). This is the kit's first
**head edit**: a new output class is grafted onto a frozen organ, not a
weight edit.

### 8.0 Subject and surgery

`Model/router.onnx` — the deployed script router. Trunk (stem + 6 mobile
blocks) and the flip head stay bit-frozen. The group head
`group.weight [4,192] / group.bias [4]` grows a 5th row: **reject**
(class 4 = textless crop → abstain from routing). Init of the new row:
LDA prototype — `w = unit(mu_nontext − mu_text)`, scaled to the median
script-row norm, bias at the midpoint of the projected means.

### 8.1 Data (registered)

| role | crops | content |
|---|---|---|
| fit | 40 non-text + 80 text | the pre-registered `router` corpus (F4/F5) |
| anchor (scope 'good') | 160 text | clean (120) + control (40) — the ability corpora, centred on a white canvas to the router's fixed 32×512 |
| transfer (unseen) | 210 non-text | structured / noise / blank — families the router corpus never contains |

Width normalisation is white-pad/centre-crop to 512 (registered after a
measured pre-check: resize degrades even the BASE router to 0.892 clean
latin accuracy; padding keeps 0.9667).

### 8.2 Arms and round protocol

Adam, batch 1, grad-clip 1.0, seed 0, violation-driven sampling (the
`refresh_every` self-labelling of §5 ported to a classifier head: every 32
steps the violation sets — text crops where reject still wins, non-text
where reject still loses — are re-labelled from the current head; violators
sampled 3:1 nt:text), up to 960 steps, lr ×0.5 at 320/640, early stop when
both violation sets empty, mini-benchmark at 0/320/640/960.

- **reject_row**: trainable = the new row only (192 + 1 params). The 4
  script rows stay bit-exact → script-logit parity is exact by
  construction; the edit is a single rank-1 BBP spike by design.
- **reject_full**: trainable = all 5 rows (+ reject bias; script biases
  frozen); CE on text (script targets) + non-text (reject target) + the
  hard anchor β·relu(mse₄ − cap), β=10, cap=0.02, of the 4 script logits
  vs the frozen base.

### 8.3 Gates (registered, all decided post-export)

- **G1 text parity** — row arm: script rows bitwise unchanged; full arm:
  zero new routing errors on fit text, ability route acc ≥ base − 0.02,
  anchor mse ≤ cap. Both arms: reject never wins an ability text crop.
- **G2 flip** — flip head untouched: flip outputs bitwise identical.
- **G3 trunk** — every non-head initializer array identical to the base
  graph.
- **G4 cure** — fit non-text reject rate ≥ 0.90 AND reject never wins a
  fit text crop.

Winner: the gate-passing arm with the higher transfer reject rate; tie →
`reject_row` (minimal intervention). No gate may be re-fitted to the
outcome; control-run failures stay in the training log (AUDIT §9).
