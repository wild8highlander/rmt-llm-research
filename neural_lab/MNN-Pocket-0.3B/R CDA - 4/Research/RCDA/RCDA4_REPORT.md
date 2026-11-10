# RCDA-4: Weight-Space Programming — reading, verifying and writing weights

Written: November 9, 2026. Plan: `RCDA4_PLAN.md` (a previous session). All three
parts were executed on the RCDA bench (MNN-Pocket-0.3B, CPU) and on MIT's public
weights (Qwen3.5-0.8B). Primary source: Takehi, Teoh, Zhang, Khattab (MIT CSAIL),
"Sparks of Weight-Space Programming", November 2, 2026 —
https://rikiyat.github.io/blog/weight-space-programming

---

## 1. Method: the math of the primitive and the architecture

### 1.1 Why the trick works: projective direction subtraction

The looping defect in a small LM is encoded as a **behavioral direction** in the
residual stream: on loop generations the increments of the hidden states
systematically write along one vector `v ∈ R^d` (for our model d = 896; v is measured
by the Petrov–Ring SVD spectrum on activation windows, PM-energy share 0.9995 —
rounds 1–2). Each tensor `W`'s contribution to this direction is the component of its
output along `v`. For a matrix `W` (out × in) the output on input `x` is `Wx`, and its
projection on `v` is `v(vᵀWx)/‖v‖²`. Eliminating the contribution along `v` entirely
means multiplying from the left by the projector:

```text

W' = (I − P_v) W,   P_v = v vᵀ / ‖v‖²

```

The MIT agent did exactly this in a restricted form: it subtracted `v` from **64
selected output columns** of the MLP blocks of layers 15 and 19 (131,072 weights =
0.016% of the model). Restricting the edit to a subset of columns S:

```text

W'[:, S] = (I − P_v) W[:, S],   W'[:, ∉S] = W[:, ∉S]

```

— removes only the part of the selected neurons' contribution that writes the loop
and leaves everything else untouched. In our RCDA-2 notation the same is expressed
through the **TW gate**: the spectral health condition of the edit is that all
eigen-components of `C = WWᵀ/n`, EXCEPT the one whose left singular vector is closest
to `v`, must stay within ±3 Tracy–Widom scale units (`spectral_tw_control`, plan
section 7.2). Coarse edits (zeroing a whole head — round 3) violate TW: they remove
the whole spectrum rather than one component; projective subtraction passes.

### 1.2 Harness architecture (implemented, not a description)

Components (all scripts saved in `Research/RCDA/tools/`):

```text

┌────────────┐  ```python cells```  ┌──────────────────┐
│  GLM agent │ ───────────────────► │ driver rcda4_wsp │
│ (z-ai chat)│ ◄─────────────────── │  (transcript,    │
└────────────┘  stdout ≤ 3 KB/cell  │   budgets)       │
                                    └────────┬─────────┘
                                             │ stdin/stdout protocol
                                             ▼
                                    ┌──────────────────┐
                                    │ rcda4_kernel.py  │ persistent
                                    │ persistent       │ python session
                                    │ session          │
                                    └──┬───────────┬───┘
                     boot_c.py: numpy  │           │  boot_b.py: sd + model
                     + A/B/C matrices  │           │  + benchmark() (budget 10)
                                ▼      ▼           ▼
                       weight matrices  state_dict → model → metrics
                       (Part C)         (Part B: edit + verification)

```

Budgets and rules (pre-registered): Part C — 8 cells at most; Part B —
25 cells, 10 benchmark() calls, the finale = `save_edited()` + `FINAL_JSON`;
gates: ρ_mini ≤ 0.35, TW pass, KL ≤ 0.15 on control texts.

### 1.3 The plan's three parts and their status

| Part | Question | Data | Status |
| --- | --- | --- | --- |
| C | Can GLM **read** the spectral concentration from raw weights? | our W_o L15/16/18 under blind labels A/B/C | done |
| A | Does the **published MIT edit** work independently? | base + edited Qwen3.5-0.8B from HF, our benchmark | done |
| B | Can GLM **write** the edit itself (blind vs aided)? | our defect bench, state_dict, budget | done |

## 2. Part C — reading weights (done)

GLM received three 896×896 matrices (the base's o_proj of layers 15/16/18) under
blind labels A/B/C (secret shuffling: A→L16, B→L18, C→L15), a Python session with
numpy and a neutral task: "find the head whose 64 output columns are structurally
distinct". 8 cells, 10 minutes.

The agent's answer: `{"A": [12, 3, 8], "B": [9, 10, 1], "C": [7, 13, 11]}`.

| Layer (label) | Measured top (concentration) | Agent's answer | Outcome |
| --- | --- | --- | --- |
| L15 (C) | **h7** (×3.11) | h7, 13, 11 | **top-1 exact hit** |
| L16 (A) | h8 (×2.22) | 12, 3, h8 | top-3 hit |
| L18 (B) | h12 (×1.91) | 9, 10, 1 | miss |

Conclusion: a non-frontier model reconstructed the strongest spectral concentration
(L15→h7) exactly from the weight numbers alone, and the second one (L16→h8) within
the top-3; the miss is on the weakest signal (L18 ×1.91 — the layer whose
concentration was already unstable in round 3). This is an independent confirmation
both of our concentration metric and of MIT exp3's "reading" of LLM weights without
running the network.

## 3. Part A — independent check of the MIT edit (completed, Gate A failed)

Protocol: base and edited Qwen3.5-0.8B from HF (1.7 GB each), our set of 8 bad RU +
4 good RU + 6 GSM8K train EN, sampling per the card (t=0.6, top-p 0.95, top-k 20,
thinking on), a 512-token budget, one sample per prompt.

Metric correction (fixed BEFORE the edited run): on the 512-token horizon the base
does not produce hard 4-gram loops (loop_share 0.11–0.35, line repeats nearly
absent: 0.0087) — this model's pathology, like MIT's, is "reasoning never finishes".
Therefore ρ := the share of generations that did not close thinking within the budget
(`</think>` by token id 248069; the first detector over decoded text was incorrect due
to stripping special tokens — fixed before the edited run, the id's validity confirmed
after the run).

| Metric (18 gens) | base | MIT edited |
| --- | --- | --- |
| finished (bad RU) | 0/8 | 0/8 |
| finished (GSM8K EN) | 0/6 | 0/6 |
| ρ "did not finish within 512" (bad/good/gsm) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| loop_share mean (bad RU) | 0.228 | 0.191 |
| line_repeat mean (bad RU) | 0.0087 | 0.000 |
| truncation @512 | 100% | 100% |
| distinct4 (bad RU) | 0.771 | 0.808 |

**Gate A (ρ_ed ≤ 0.5·ρ_base = 0.5) — failed: ρ_ed = 1.0.** The published edit does
not transfer to our benchmark: on our set (Russian bad prompts + GSM8K train EN) the
edited release behaves indistinguishably from the base — thinking does not close
within the budget for a single generation, and the loop_share drop 0.228 → 0.191 is
within the spread of one sample per prompt.

What is excluded: finish detection is valid (the token id is in the vocabulary, the
texts truncate mid-reasoning, not in a loop); sampling and the chat template are taken
from the repositories; the edit was applied by the authors themselves (the same
weights-space as in their paper). The likely explanation: the edit was tuned to their
diagnostic set (English) and a different generation horizon; on long GSM8K reasoning
and on Russian prompts the pathology is the same in both versions. RCDA-4's own
result: **the weight-space-edit effect does not generalize beyond the set it was
tuned on** — that is exactly why the verification was run. For Part B this is a
reason to read results carefully: the "edit → behavior" transfer function is narrow
even for the method's authors.

## 4. Part B — the GLM programmer: both arms failed (completed)

Bench: MNN-Pocket-0.3B, state_dict in a persistent kernel, budget 25 cells +
10 benchmark() calls (8 bad prompts, greedy 96, ρ_mini; base = 1.0). The aided arm
additionally received v_L{15,16,18}.npy and diagnostics.json (head alignments with v:
top heads 7/8/12 — matched ground truth). Final evaluation `rcda4_eval_b.py`:
20 bad + 20 good (greedy 96), KL(base||edited) on control texts, TW control, the
cosine of the edit's top direction with v.

**Blind (25/25 cells, 10/10 benchmark):** the agent never reached the narrow
component — it cycled through global perturbations (LM-head noise → attn o_proj noise
→ gate_proj scaling → RMS noise → embed noise → final-norm noise → down_proj scaling
→ amplified noise). Its best "win" of ρ_mini 0.75 was achieved by breaking the model:
generations collapsed to 1 token (loop_share 0.0 at zero length — cures the loop by
killing generation). Massive gate_proj edits across many layers, the instruction "do
not rewrite large chunks" violated, `save_edited()` never called (budget exhausted) —
the edit was not saved and is unavailable for the final eval. Qualitatively this is
exactly MIT's exp6 outcome in its failing branches: without diagnostics the programmer
mutes the model globally.

**Aided (25/25, 10/10):** the agent read the diagnostics (v vectors, head
alignments), then 12 of 25 cells died in execution errors (dtype/shape bf16); an
intermediate edit was saved at cell 7 — **model.layers.16.q_proj**; the final
self-report claim ("zeroed L18 head 2 o_proj") never reached disk: the file and the
self-report diverge. Final eval of the saved edit:

| Metric | aided | Manual RCDA-2 (target) |
| --- | --- | --- |
| edited tensor | L16 **q_proj** (1) | L16+L18 o_proj (overlay) |
| ρ_bad (20 gens) | 0.85 | 0.65 |
| ρ_good | 0.60 (broken) | 0.05 |
| KL(base‖edited), ctrl | 0.065 | 0.09 |
| TW verdict | **violation** | pass |
| cos(edit top direction, v_L16) | **0.054** | ~1 (built from v) |

The agent chose the right layer (16 — one of the three loop layers), but the wrong
tensor (q_proj instead of o_proj) and the wrong direction (cosine with v 0.054): the
defect remained (0.85), good generations degraded (0.6), the spectral control failed.
An important side result from cell 3: the inner products
v15·v16/(‖v15‖‖v16‖)=0.944, v16·v18=0.883, v15·v18=0.833 — **the loop directions of
the three layers are nearly collinear**: it is one loop observed in three places, and
editing one component of one layer cannot remove it by construction (which manual
RCDA-2 confirmed: two layers were required).

## 5. RCDA-4 conclusions

1. **Reading weights — yes.** Part C: a non-frontier GLM, from raw 896×896 matrices
   under blind labels, found the most concentrated head exactly (top-1) and the second
   one within top-3; the miss is naturally on the weakest signal. Our concentration
   metric is reproducible from the weights alone, without running the network.
2. **The primitive's math — confirmed.** The edit = projective subtraction
   W'[:,S] = (I − P_v)·W[:,S]; TW control separates it from coarse edits
   (pass vs violation on synthetic data, and on the agent's real edit).
3. **Transferability of the published edit — no.** Part A, Gate A failed: on our set
   (Russian bad + GSM8K train EN, 512 tokens) MIT's edited release is
   indistinguishable from the base (0/18 thinking completions for both). The
   weight-space-programming effect does not generalize beyond the authors'
   diagnostic set.
4. **Writing the edit by an agent — does not reproduce.** Part B: blind drowns in
   global perturbations and breaks the model; aided finds the layer but not the
   tensor and not the direction (TW violation, cos 0.054, the defect remains). Unlike
   MIT's exp6/8, on our bench the method works neither blindly nor with diagnostics —
   under honest budgets (25 cells / 10 benchmark).
5. **The architecture — reproducible and reusable**: kernel+driver+bootstrap
   separate the agent from the environment, incremental transcripts survive a session
   drop, pre-registered gates turn a "trick" into a testable protocol. RCDA-4 closes
   RCDA4_PLAN.md's question: the trick has a real architecture — and real limits.

## 6. Reproduction

```bash

python3 Research/RCDA/tools/rcda4_partC_extract.py   # the blind matrices A/B/C
python3 Research/RCDA/tools/rcda4_wsp.py ...          # the C/B agent sessions
python3 Research/RCDA/tools/rcda4_partA_bench.py ...  # the base/edited benchmark
python3 Research/RCDA/tools/rcda4_eval_b.py ...       # the final Part-B evaluation
python3 Research/RCDA/tools/rcda4_figures.py          # fig14

```

Artifacts: `results/` (in the zip: `partC_partC_transcript.json` etc.) — agent
transcripts (partC, partB blind/aided), benchmark records
(partA_{base,edited}.jsonl + summary), the aided-edit evaluation
(partB_eval_aided.json), the C secret mapping, v_L{15,16,18}.npy,
edited_state_aided.safetensors, figures/fig14_rcda4.png. The blind arm has no final
artifact by definition: the agent never called `save_edited()`
(section 4), no eval is run for it — that is part of the result.

## 7. Limitations

- **Part A: one sample per prompt** (18 generations per arm) at t=0.6 — enough for
  the binary conclusion "0/18 vs 0/18", but it does not measure variance; the
  512-token budget is hard and chosen to equal our other horizon; the authors' horizon
  is larger — transferability to 32k was not tested.
- **Part B: a budget of 25 cells / 10 benchmark()** — half the aided cells died in
  execution errors (dtype/shape bf16); a more generous budget or a better tool-API
  might change the outcome, hence the phrasing "does not reproduce under honest
  budgets", not "impossible in principle".
- **The blind arm did not save its edit** — its final evaluation is limited to the
  self-report (ρ_mini 0.75 with broken generations), not an independent eval;
  reconstructing the edit from the transcript is possible but was not done.
- **One MIT repository and one bench model**: the transferability conclusions apply
  to the layers-15/19 edit of Qwen3.5-0.8B and to our MNN-Pocket-0.3B defect; other
  model/edit pairs were not tested.
