# 🧾 AUDIT — what was run, what was found, what was honestly negative

Every claim in the README traces to a committed JSON. This audit walks the
chain, records the measured lessons (negative results kept on purpose —
three from the recogniser arms, two more from round 3) and states the
limitations without cosmetics.

## 1. Model identity and rebuild parity

| check | value | where |
|---|---|---|
| source | `llamaindex/OCRia-nano-1` (HF), files sha256-pinned | [`RUN_PASSPORT.json`](RUN_PASSPORT.json) |
| int8 → fp32 rebuild parity | exact decoded-string parity **97.9%**, conf deviation 0.0079 | `Results/rebuild_parity.json` |
| ReLU map derivation | from the QDQ chains (uint8 zero_point=0 ⇒ clip ⇒ ReLU); mix.1/2 quantizers have zp=62/78 ⇒ signed | `rebuild_latin.py` docstring |
| residual mismatch | quantisation round-off on saturated crops; documented, accepted | — |

The probe re-measures the **deployed int8** graph for all corpora (phantom
0.119) so the fp32 world (phantom 0.157) is never silently substituted for the
deployed behaviour — the difference is itself a measured finding: int8
activation rounding suppresses marginal phantom emissions.

## 2. The pre-registered chain

```
make_corpora.py (seed 7)  →  rebuild_latin.py  →  ocria_probe.py (seed 0)
  → spectral_audit.py (seed 0)  →  ocria_train.py (seed 0, both arms)
  → ocria_benchmark.py (seed 0, 3 arms)  →  ocria_summarise.py  →  make_graphics.py
```

One corpora generator feeds parity, probe, training and evaluation, so no
number can drift between stages. One evaluator measures all three arms.

## 3. Measured lesson #1 — the symmetric square backfires on logits

The parent kit's penalty is `pen = (vᵀh / ‖v‖)²` — symmetric in the sign of the
projection. On the recogniser's **logit states** this is actively harmful:

| run | pen form | mini-bench phantom @150 |
|---|---|---|
| control run (kept in git history of this audit) | `(vᵀℓ)²` at head | **1.0** (base 0.417 → everything emits) |
| shipped classic arm | `relu(vᵀℓ − edge)²` (one-sided) | **0.083** |

Mechanism: frames that are already on the abstained side sit at negative
projections; the square drives them toward the hyperplane *from below*, i.e.
raises phantom-class logits. The one-sided clamp is the BBP-style energy
leash: only energy **above** the abstention edge is penalised.

## 4. Measured lesson #2 — a global direction does not flip per-frame argmax

With the penalty at the 192-dim mix2 state (`(vᵀh)²`), 150 steps moved the
penalty down 27% but the mini-bench phantom rate never moved: the population
direction v localises *where* the defect is written (z_diff ≈ 14, AUC 0.96)
but the per-frame emission decision is a 792-way argmax margin. The shipped
penalty therefore acts at the head logits, frame by frame, against the blank
edge — the quantity the argmax actually reads.

## 5. Measured lesson #3 — the projection knob

| rcda_x projection setting | extra spikes shrunk / step-25 | phantom @150 |
|---|---|---|
| 1.1·λ₊, keep top-1 | 65–76 | 0.25 |
| **2.0·λ₊, keep top-2 (shipped)** | 32–41 | **0.167** |

The aggressive setting fights Adam — it re-cuts the useful edit components
every 25 steps. The shipped setting keeps the two intended spikes per tensor
and only removes unstructured excess above 2·λ₊.

## 6. The leverage verdict

classic (single writer, 9.2% of params) beats rcda_x (three writers, 27.7%,
full formula stack) on the primary endpoint (phantom −94% vs −79%) *and* on
abilities drift (KL 0.0095 vs 0.083). The extra formulas of rcda_x buy a
sharper geometry separation (0.834 → 0.902) and near-perfect edit alignment
(cos(u₁, v) 0.78–0.85) — at 3× the drift. This replicates the parent
program's *"SGD beats surgery"* finding on a different organ.

## 7. Limitations (stated, not hidden)

1. **fp32 world.** Training and evaluation run on the fp32 rebuild; deployment
   would re-quantise. The int8 deployed graph is measured at the probe stage,
   and both numbers are reported everywhere they matter.
2. **Abilities vs robustness corners.** The hard anchor holds clean/control
   text (+0.05pp CER); degraded reads drift +1.7pp (classic) / +2.6pp (rcda_x).
   The anchor scope is 'good' (ability data), exactly as the parent protocol
   defines it.
3. **Single organ trained.** The router (F4/F5) and detector are measured and
   frozen. Curing the router's missing reject option would require a
   classification-head change (a 5th output class), not a parameter edit —
   out of scope for an RCDA-style weight intervention, stated as such.
   **→ LIFTED by round 3** (user request): registered addendum §8 of the
   protocol, gated to exact parity on everything the base already did right;
   measured in §9 below. The detector and LayoutNet remain out of scope.
4. **LayoutNet out of scope.** Its `feats`/`pairf` inputs are pipeline-internal
   feature engineering; measuring it without the real feature builder would be
   fabrication, so it is not measured.
5. **F2 (CTC loops) is ~0 at base** on these corpora. The metric is the
   parent's verbatim port and stays in the evaluator as a guard against
   regressions; it is not artificially induced to manufacture a cure.
6. **Corpora are synthetic.** Rendered text and programmatic patterns give
   exact ground truth and full control, but absolute rates on real-world
   scans will differ; the *relative* arm comparisons are the transportable
   claim.
7. **Single seed.** The parent kits also commit single-seed CPU runs with
   pinned passports; variance runs are a GPU-scale follow-up (as in the
   parent's XL-queued scaling ladder).

## 8. Provenance

Every result file carries the args and seeds that produced it
(`Results/*_training_log.json` → `args`; `probe_report.json` → `protocol`;
`RUN_PASSPORT.json` → model file hashes + environment versions).

## 9. Round 3 addendum — the router reject round (a head, not weights)

Registered as protocol addendum §8 before the round's first training step
(user request: "F4 requires a 5th class *reject* — this is a head edit").
The surgery: `group.weight [4,192] → [5,192]`, the new row = **reject**;
trunk and flip head bit-frozen (verified post-export, not assumed).

**Shipped result** ([`eval_router_reject.json`](Results/eval_router_reject.json),
arm `reject_row`, all 4 gates pass):

| gate | verdict |
|---|---|
| G1 text parity | script rows bit-identical (script rows max\|Δ\| = **0.0**); ability routing identical to base (clean 0.9667 / control 1.0); **0** reject wins on 160 ability crops |
| G2 flip | flip outputs **bitwise identical** on all crops |
| G3 trunk | every non-head initializer array identical |
| G4 cure | fit non-text reject **1.0** (40/40), fit-text reject wins **0** |

Transfer (never-trained families): reject **0.911** macro-mean of the
family rates (structured 0.833 / noise 0.900 / blank 1.000; pooled
184/210 = 0.876) — the 26 residual misroutes (20 structured, 6 noise,
0 blank) are the honest cost; the head edit is a rank-1 single BBP spike
(λmax/λ₊ = 3.71, σ₁ share 1.0).

### Measured lesson #4 — the reject row learns whiteness first

Control run: violation-driven training anchored ONLY on the router corpus
(text = full-frame 512-px renders). Fit reject reached 1.0 with text parity
exact — but the transfer text route collapsed to **0.53**: the reject row
had learned *margins/whiteness*, not *textlessness*. The ability corpora
(200–480 px renders) are centred on white canvases, and the pooled
mean/max features of padded text sit closer to the blank prototype than to
any script. Fix (shipped): the round's anchor scope 'good' includes
clean + control, exactly like the recogniser arms.

### Measured lesson #5 — text-heavy passes collapse the cure

Control run: one full pass = 40 non-text ×3 + all 240 text crops (a 1:2
text-heavy cycle), constant lr 5e-3. Dynamics: fit reject 0.925 → 1.0
through the non-text segment, then **0.5** by step 360 — Adam's per-sign
normalisation let the tiny-but-consistent text-side gradients (p_rej ≈ 1e-3
on already-correct crops) drag the reject row below the script logits on
half the non-text corpus. Fix (shipped): violation-driven sampling — only
crops the head still gets wrong receive gradient, violators 3:1 nt:text,
lr ×0.5 at 320/640. The shipped run converges with fit reject 1.0,
transfer 0.911 and ability parity exact.

### The leverage verdict, third occurrence

`reject_row` (193 trainable params, exact parity by construction) passes
all gates; `reject_full` (961 params, supervised fit + anchor) reaches a
marginally better macro-mean transfer reject (0.920 vs 0.911) but wins
**2 ability text crops** (reject-win rate 0.0125) — G1 fails, the arm is
rejected.
For the third time in this kit the minimal lever wins: phantom cure
(classic > rcda_x), and now the head cure (row > full).
