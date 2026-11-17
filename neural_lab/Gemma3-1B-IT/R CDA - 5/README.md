# <sub>🧬</sub> R CDA — 5 · the formula as a loss, ported to Gemma3-1B-IT

<p align="center">
  <img src="https://img.shields.io/badge/model-Gemma3--1B--IT-38E1C0" alt="model badge">
  <img src="https://img.shields.io/badge/transfer-test-FFB020" alt="transfer badge">
  <img src="https://img.shields.io/badge/result-negative%2C%20recorded%20honestly-FF3B6B" alt="result badge">
</p>

An exact structural analog of
[`neural_lab/MNN-Pocket-0.3B/R CDA - 5`](../../MNN-Pocket-0.3B/R%20CDA%20-%205/):
the RCDA-5 experiment — **bake the formula `(I − P_v)` into training as a
gradient penalty** on the loop layers, with a hard abilities anchor — applied
to **Gemma3-1B-IT**.

**Model used:**
[https://huggingface.co/litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)
(LiteRT on-device packages of the base checkpoint `google/gemma-3-1b-it`;
1.04 B parameters, instruction-tuned, 26 layers, hidden 1152, vocab 262 144).
Training ran on the identical PyTorch weights (public mirror
`unsloth/gemma-3-1b-it`, checksum-verified; Gemma license applies).

**The final form of the training** (150 steps, batch 1, CPU, fp32 masters over
bf16 compute; trainable = o_proj of the writer layers **10/11/12** found by the
spectral probe):

```
L = pen_norm + β·relu(mse − cap)
pen_norm = mean_l mean_t (v̂_lᵀ h_t)² / pen0_j      # the formula, dimensionless
mse      = MSE(trailing 32 logits, base logits)    # the abilities anchor
β = 10, cap = 1.0
```

Two optimizer arms: **SGD** (momentum 0.9, lr 3e-3, grad-clip 1.0 — the
user-specified primary) and **Adam** (lr 3e-5 — the exact MNN RCDA-5 recipe,
control).

## Headline results (measured, all three arms)

| metric | base | trained (SGD-150) | adam (Adam-150) |
| --- | --- | --- | --- |
| ρ_bad (20 bad prompts, greedy 192) | 0.60 | **1.00** | **0.95** |
| loop_waste mean (bad), tokens | 147.1 | 180.1 | 177.9 |
| pen_norm (init 1.0) | 1.0 | **0.0084 (−99.2 %)** | **0.2236 (−78 %)** |
| TW verdict | — | violation ×3 | violation ×3 |
| cos(u₁ of the edit, v) | — | 0.32 / 0.18 / 0.31 | 0.08 / 0.05 / 0.14 |
| KL(base‖arm), control | 0 | 0.2213 | 0.0248 |
| PPL control (mean) | 10.9 | 10.2 | 11.0 |

**What this experiment established.** The probe/penalty/anchor/evaluator
machinery transfers, and abilities survive (PPL parity). The MNN result does
**not**: on Gemma3-1B-IT the penalty is minimized (−99 %) yet the behavioral
defect *strengthens* (every bad prompt ends up looping), and the SGD edit is
not spectrally clean. The loop of Gemma3-1B-IT is not a 3-o_proj-localized
defect — the penalized direction is a correlate, not the cause. This negative
transfer is the experiment's result, recorded with the same honesty standard
as the rest of the repository. Full analysis:
[`RCDA5_REPORT.md`](Research/RCDA/RCDA5_REPORT.md).

## What is inside

| Path | Contents |
|------|----------|
| [`Code/`](Code) | the toolkit: `gemma_probe.py`, `gemma_rcds_train.py`, `gemma_benchmark.py`, `gemma_lab.py`, `marchenko_pastur.py` (MP law + TW control) |
| [`config/`](config) | `gemma_rcda5.yaml` — the final configuration as actually run |
| [`prompts/`](prompts) | the frozen experiment inputs: sentinel-selected `bad.jsonl` (20), `good.jsonl` (20), `control.jsonl` (5) |
| [`Research/RCDA/`](Research/RCDA) | [`RCDA5_REPORT.md`](Research/RCDA/RCDA5_REPORT.md) + [`RCDA5_PLAN.md`](Research/RCDA/RCDA5_PLAN.md) (pre-registered), `results/` — the complete JSON record (probe, selection, training logs ×2, part5 evals ×3, PPL check, spectral digest, `rcds5_vectors.npz`), `figures/` fig15 + figG1–G3, `tools/` (`rcda5_ppl.py`, `rcda5_analyze.py`, `rcda5_figures.py`) |
| [`Graphic/`](Graphic) | the figures, mirrored from `Research/RCDA/figures/` |
| [`Model ZIP/`](Model%20ZIP) | the fine-tuned model (SGD arm) as a self-assembling checksummed split archive — 17 volumes, each ≤ 95 MB (GitHub-safe), + `SHA256SUMS.txt` |
| [`Docs/`](Docs) | [`UNPACK_MODEL.md`](Docs/UNPACK_MODEL.md) — archive reassembly; [`RCDA_RUNBOOK.md`](Docs/RCDA_RUNBOOK.md) — how to re-run |

**Start here:** [RCDA-5 report](Research/RCDA/RCDA5_REPORT.md) → the gallery in
[`Graphic/`](Graphic) → the training record
[`Research/RCDA/results/training_log.json`](Research/RCDA/results/training_log.json)
(150 steps × {mse, pen_raw, pen_norm, pen_layers} + benches + forensics).

> Parent experiment: [`neural_lab/MNN-Pocket-0.3B/R CDA - 5`](../../MNN-Pocket-0.3B/R%20CDA%20-%205/README.md)
> Continuation: [`neural_lab/Gemma3-1B-IT/R CDA - 6`](../R%20CDA%20-%206/README.md) — the expanded trainable set (o_proj+down_proj, L8–L15) with phase-conditioned directions, run per this round's closing hypothesis; cross-round summary: [`../RESULTS_SUMMARY.md`](../RESULTS_SUMMARY.md).

> — the canonical round-5 snapshot. The two folders share the methodology, the
> evaluator design and the reporting format; the models (and, as it turned out,
> the conclusions) differ.

## Reproduction (one CPU evening, ~2.5 h total)

```bash
python3 Code/gemma_probe.py       --model unsloth/gemma-3-1b-it
python3 Code/gemma_rcds_train.py  --model unsloth/gemma-3-1b-it --steps 150 \
                                  --lr 3e-3 --optim sgd --bench-at 75 150
python3 Code/gemma_benchmark.py   --model unsloth/gemma-3-1b-it \
                                  --edited-state <work>/trained_state.safetensors \
                                  --tag trained --out part5
python3 Research/RCDA/tools/rcda5_ppl.py      # PPL parity
python3 Research/RCDA/tools/rcda5_analyze.py  # TW/cos forensics + vectors
python3 Research/RCDA/tools/rcda5_figures.py  # fig15 + figG1–G3
```

Everything runs on CPU (the full program was executed on 2 cores / 4 GB RAM);
`google/gemma-3-1b-it` requires accepting the Gemma license on Hugging Face —
any local copy of the same weights works via `--model`.
