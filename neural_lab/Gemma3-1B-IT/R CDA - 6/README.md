# <sub>🧬</sub> R CDA — 6 · the expanded trainable set, phase-conditioned — Gemma3-1B-IT

<p align="center">
  <img src="https://img.shields.io/badge/model-Gemma3--1B--IT-38E1C0" alt="model badge">
  <img src="https://img.shields.io/badge/trainable-73.1M%20params%20%2F%207.31%25-00D4FF" alt="trainable badge">
  <img src="https://img.shields.io/badge/hard--loopers-0.827%20%E2%86%92%200.620-12B886" alt="result badge">
  <img src="https://img.shields.io/badge/full--set%20rho__bad-0.85%2C%20migration%20recorded-FFB020" alt="migration badge">
</p>

The direct continuation of
[`R CDA - 5`](../R%20CDA%20-%205/README.md). RCDA-6 executes the hypothesis
that closes the RCDA-5 report — *"the loop is written jointly by attention
output and MLP down_proj across a wider band (L8–L15), and the probe
direction must be conditioned on the loop phase, not only the collapse
point"* — with the same model, the same frozen sentinel sets, the same
evaluator, and only two changes:

1. **Expanded trainable set**: `o_proj` **and** `down_proj` of layers
   **8–15** — 73 138 176 parameters (7.31 % of the model, 21× RCDA-5);
2. **Phase-conditioned directions**: the frozen loop trajectories are split
   into pre / early / deep phases; the penalty projects each position on the
   direction of ITS phase (the deep phase separates 2.5–3.5× better and
   points 0.48–0.89 away from the early one).

**Model used:**
[litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)
(LiteRT on-device packages of the base checkpoint `google/gemma-3-1b-it`;
1.04 B parameters, instruction-tuned, 26 layers, hidden 1152, vocab
262 144). Training ran on the identical PyTorch weights (public mirror
`unsloth/gemma-3-1b-it`, checksum-verified; Gemma license applies).

**The final form of the training** (150 steps, batch 1, SGD momentum 0.9,
lr 3e-3, grad-clip 1.0, fp32 masters over bf16 compute; the loss formula is
RCDA-5's, the target is phase-matched):

```
L = pen_norm + β·relu(mse − cap)
pen_norm = mean_l mean_t ( v̂_{phase(t), l}ᵀ h_t )² / pen0_j    # dimensionless
phase(pre) = phase(early) → v̂_early ;  phase(deep) → v̂_deep
mse       = MSE(trailing 32 logits, base logits)               # the anchor
β = 10, cap = 1.0
```

## Headline results (measured)

| metric | base | RCDA-5 SGD | RCDA-5 Adam | **RCDA-6** |
| --- | --- | --- | --- | --- |
| ρ_bad (20 bad prompts, greedy 192) | 0.60 | 1.00 | 0.95 | **0.85** |
| hard-12 loop share (the prompts that hard-loop at base) | 0.827 | 0.908 | 0.884 | **0.620 (−25 %, 2 fully repaired)** |
| mild-8 loop share | 0.164 | 0.915 | 0.944 | **0.637** |
| loop_share mean (all bad) | 0.562 | 0.911 | 0.908 | **0.627** |
| ρ_good | 0.10 | 0.10 | 0.05 | **0.10** |
| pen_norm (init 1.0) | 1.0 | 0.0084 | 0.2236 | **0.028 (mean last 50)** |
| TW verdict | — | violation ×3 | violation ×3 | **violation ×16** |
| KL(base‖arm) | 0 | 0.2213 | 0.0248 | **0.3074** |
| PPL control (mean) | 10.9 | 10.2 | 11.0 | **14.1** |

**What this experiment established.** The phase hypothesis is real at the
probe level (z_deep 247–393 vs z_early 75–118; the phase directions differ),
and it is real behaviorally: RCDA-6 is the **first arm that improves the
hard-loop attractor over the base model itself** — RCDA-5's edit pushed the
same prompts from 0.83 to 0.91, RCDA-6 pulls them to 0.62 and fully repairs
two. The defect did not die — it migrated to the mild prompts (0.16 → 0.64)
— and the 7.3 %-parameter edit costs abilities drift (PPL +29 %, KL 0.31)
that the small RCDA-5 edit did not. The spectrally clean solution of the
MNN-Pocket class remains out of reach on Gemma3 (TW violation ×16). The
прирост against RCDA-5 is measured end to end: loop share −31 %, ρ_bad −15 %,
loop waste −11 %. Full analysis:
[`RCDA6_REPORT.md`](Research/RCDA/RCDA6_REPORT.md).

## What is inside

| Path | Contents |
|------|----------|
| [`Code/`](Code) | the toolkit: `gemma_probe_phase.py`, `gemma_rcds6_train.py`, `gemma_benchmark.py`, `gemma_lab.py` (+cycle/phase), `marchenko_pastur.py` |
| [`config/`](config) | `gemma_rcda6.yaml` — the final configuration as actually run |
| [`prompts/`](prompts) | the same frozen experiment inputs as RCDA-5 (`bad.jsonl` 20, `good.jsonl` 20, `control.jsonl` 5) |
| [`Research/RCDA/`](Research/RCDA) | [`RCDA6_REPORT.md`](Research/RCDA/RCDA6_REPORT.md) + [`RCDA6_PLAN.md`](Research/RCDA/RCDA6_PLAN.md) (pre-registered), `results/` — the complete JSON record (phase probe, training log with benches, part6 eval + rows + TW cache, PPL check, spectral digest, `rcds6_vectors.npz`), `figures/` figC1–C6, `tools/` (`rcda6_minibench.py`, `rcda6_ppl.py`, `rcda6_analyze.py`, `rcda6_figures.py`) |
| [`Graphic/`](Graphic) | the figures, mirrored from `Research/RCDA/figures/` |
| [`Model ZIP/`](Model%20ZIP) | the trained **delta** (16 tensors, 146 MB bf16) as a self-assembling checksummed archive — 2 volumes ≤ 91 MB (GitHub-safe), + `SHA256SUMS.txt`; applies onto the public base checkpoint |
| [`Docs/`](Docs) | [`UNPACK_MODEL.md`](Docs/UNPACK_MODEL.md) — archive reassembly + delta application; [`RCDA_RUNBOOK.md`](Docs/RCDA_RUNBOOK.md) — how to re-run |

**Start here:** [RCDA-6 report](Research/RCDA/RCDA6_REPORT.md) →
[`figC5_migration.png`](Graphic/figC5_migration.png) (the per-prompt story)
→ the training record
[`Research/RCDA/results/training_log.json`](Research/RCDA/results/training_log.json).

> Sibling experiment: [`neural_lab/Gemma3-1B-IT/R CDA - 5`](../R%20CDA%20-%205/README.md)
> — the transfer test this round builds on. Cross-experiment summary:
> [`../RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md).

## Reproduction (one CPU evening, ~1.5 h)

```bash
python3 Code/gemma_probe_phase.py --model unsloth/gemma-3-1b-it \
        --probe <rcda5-work>/probe --band 8 9 10 11 12 13 14 15
python3 Code/gemma_rcds6_train.py --model unsloth/gemma-3-1b-it --steps 150 \
        --lr 3e-3 --optim sgd
python3 Code/gemma_benchmark.py   --model unsloth/gemma-3-1b-it \
        --edited-state <work>/rcda6/trained_state.safetensors \
        --tag rcda6_trained --out part6 --probe <work>/probe_phase
python3 Research/RCDA/tools/rcda6_ppl.py
python3 Research/RCDA/tools/rcda6_analyze.py
python3 Research/RCDA/tools/rcda6_figures.py
```

Everything runs on CPU (the full program was executed on 2 cores / 4 GB RAM,
chunked through atomic checkpoints). The protocol deviations of the recorded
run — the dropped step-75 mini-bench, the standalone benches — are documented
in the report.
