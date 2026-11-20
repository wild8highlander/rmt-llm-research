# RESULTS SUMMARY — Gemma3-1B-IT, the formula-training rounds (RCDA-5 → RCDA-6)

Written: November 16, 2026. Cross-experiment summary of the two Gemma3-1B-IT
rounds. One question runs through both: **does baking the formula
`(I − P_v)` into training as a gradient penalty move the repetition-loop
defect — and what exactly does it buy?** Every number below is the measured
record of the frozen protocol (20 bad + 20 good sentinel prompts, greedy 192,
raw completion mode; 4-gram loop labels, threshold 0.5).

**Model**: [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)
— the LiteRT packaging of the base checkpoint `google/gemma-3-1b-it`
(1.04 B params, 26 layers, hidden 1152); trained from the public mirror
`unsloth/gemma-3-1b-it` (checksum-verified; Gemma license applies).

## The four states, one evaluator

| metric | base | RCDA-5 SGD-150 | RCDA-5 Adam-150 | RCDA-6 SGD-150 |
| --- | --- | --- | --- | --- |
| **trainable tensors** | — | o_proj L10–12 (3) | o_proj L10–12 (3) | **o_proj+down_proj L8–15 (16)** |
| trainable params | — | 3.54 M (0.35 %) | 3.54 M | **73.1 M (7.31 %)** |
| penalty direction | — | collapse-point v | collapse-point v | **phase-matched (v_early/v_deep)** |
| ρ_bad (loops ≥ 0.5) | 0.60 | 1.00 | 0.95 | **0.85** |
| ρ_good | 0.10 | 0.10 | 0.05 | 0.10 |
| loop_share mean (bad) | 0.562 | 0.911 | 0.908 | **0.627** |
| loop_waste mean (bad), tok | 147.1 | 180.1 | 177.9 | **160.1** |
| hard-12 loop share¹ | 0.827 | 0.908 | 0.884 | **0.620** |
| mild-8 loop share¹ | 0.164 | 0.915 | 0.944 | **0.637** |
| prompts fully repaired¹ | — | 0 / 12 | 0 / 12 | **2 / 12** |
| pen_norm (init 1.0, final) | 1.0 | 0.0084 | 0.2236 | 0.028 |
| TW verdict | — | violation ×3 | violation ×3 | violation ×16 |
| cos(u₁ of edit, v) | — | 0.32 / 0.18 / 0.31 | 0.08 / 0.05 / 0.14 | 0.03–0.33 (v_deep), ≤ 0.66 (v_early) |
| KL(base‖arm) | 0 | 0.2213 | 0.0248 | 0.3074 |
| PPL control (mean) | 10.9 | 10.2 | 11.0 | 14.1 |

¹ hard-12 = the 12 bad prompts with base loop share ≥ 0.5; mild-8 = the rest.
"Repaired" = looping at base, not looping after the edit.

## What the formula-training buys, round by round

1. **RCDA-5 (the transfer test, `R CDA - 5/`)**: the machinery transfers and
   abilities hold (PPL parity), but the behavioral outcome is *negative* —
   the penalty collapses (−99 %) while every hard looper gets worse
   (0.83 → 0.91) and every mild prompt breaks (0.16 → 0.92). Verdict: the
   penalized direction is a correlate of the loop, not its cause; the edit is
   not spectrally clean (TW ×3).
2. **RCDA-6 (this round, `R CDA - 6/`)**: the two pre-registered changes —
   the wider o_proj+down_proj band and the phase-conditioned directions —
   deliver the **first measured прирост against both the base and RCDA-5**:
   - **vs RCDA-5 SGD**: loop share −31 % (0.911 → 0.627), ρ_bad −15 %
     (1.00 → 0.85), loop waste −11 % (180 → 160);
   - **vs base, on the subset that defines the defect**: hard-12 loop share
     **0.827 → 0.620 (−25 %)**, two prompts fully repaired
     (`bad_g02` 0.96 → 0.24, `bad_g11` 0.59 → 0.28) — the RCDA-5 arms moved
     the same subset the wrong way (→ 0.91);
   - **the price**: the defect migrates (mild-8 0.16 → 0.64) and the edit
     costs abilities drift the small edit did not (PPL +29 %, KL 0.31);
   - **unchanged**: no TW-clean solution (×16), no convergence of the edit's
     top direction to the penalized one — the MNN-Pocket "spectrally clean
     solution of the same class" remains architecture-specific.
3. **The scaling law this pair of rounds suggests**: the behavioral leverage
   of formula-training scales with how well the penalized direction matches
   the defect's actual writing structure (writer types × loop phase), not
   with the parameter count alone. RCDA-5 aimed a phase-mixed direction at
   one writer type of a distributed circuit and made things worse; RCDA-6
   aimed phase-pure directions at both writer types of the band and bought a
   partial, migrating repair. The next lever named by the measurement:
   phase-only (deep) penalties + a tighter anchor budget.

## Artifacts

| Experiment | Report | Record | Weights |
| --- | --- | --- | --- |
| RCDA-5 | [`R CDA - 5/Research/RCDA/RCDA5_REPORT.md`](../R%20-%20CDA%20%E2%80%94%205/Research/RCDA/RCDA5_REPORT.md) | `results/` (training logs ×2, part5 evals ×3, probe, digest, vectors) | full checkpoint, 17 volumes ≤ 95 MB |
| RCDA-6 | [`R CDA - 6/Research/RCDA/RCDA6_REPORT.md`](../R%20-%20CDA%20%E2%80%94%206/Research/RCDA/RCDA6_REPORT.md) | `results/` (phase probe, training log, part6 eval + rows + TW, PPL, digest, vectors) | 16-tensor delta, 2 volumes ≤ 91 MB |

Parent experiment: [`neural_lab/MNN-Pocket-0.3B/R CDA - 5`](../../MNN-Pocket-0.3B/R%20CDA%20-%205/README.md)
— the positive result that started the transfer program.
