# Delivery notes — how to merge this into `rmt-llm-research`

This folder contains the complete RCDA-8 LlamaFactory round for
`neural_lab/Gemma3-1B-IT` plus refreshed READMEs for every folder of the
kit. It was produced by a real end-to-end run on the real model
(`unsloth/gemma-3-1b-it` @ snapshot `5b1141…`) through LlamaFactory 0.9.5
on a CPU-only box (2 cores, 4 GB RAM). Every number is a field of a
committed JSON — nothing hand-typed.

## What to copy where

Copy the contents of `gemma3-llamafactory-round/` into
`neural_lab/Gemma3-1B-IT/` in your repository (folders merge, READMEs
replace the old ones):

| Item | Action |
|------|--------|
| `LlamaFactory/` | **NEW folder** — the whole RCDA-8 round (report, pipeline, data, configs, results, adapters, figures) |
| `README.md` | **REPLACES** the kit README — maximally extended: real-model numbers on top, three-layer provenance, animation wall, the GPT-2 stand-in numbers collapsed into an archive section |
| `Gemma3-1B-IT.md` | **REPLACES** the patient card — real probe numbers (writers L10/11/12), real baselines |
| `Code/README.md`, `Docs/README.md`, `Graphic/README.md`, `Graphics/README.md`, `JSON-Results/README.md`, `Model-ZIP/README.md`, `Model-Z/README.md`, `RCDA-5/README.md`, `RCDA-6/README.md`, `RCDA-7/README.md`, `Reports-MD/README.md`, `Scripts/README.md`, `assets/README.md` | **NEW or REPLACED** — one README per folder; every one labels its provenance layer (validation / real-Gemma / RCDA-8) and links the round lineage |

Nothing else in your repository is touched: all existing files
(`R CDA - */Research/…`, `JSON-Results/*`, `Model-ZIP/*`, `Model-Z/*`,
`Graphic/*`, `Graphics/*`, `Scripts/*`, `Reports-MD/*`) stay exactly as
they are — the new READMEs document them as the record they already are.

## Headline of the round (measured, greedy 192, one evaluator)

| metric | base | RCDA-8 LoRA (formula) | SFT LoRA (data) |
|---|---|---|---|
| ρ_bad | 0.60 | 0.70 | **0.40** |
| loop_share mean (bad) | 0.562 | 0.656 | **0.393** |
| hard-12 loop share | 0.827 | 0.632 | **0.590** |
| fully repaired | — | 3/12 | **5/12** |
| KL / PPL | 0 / 10.9 | 0.512 / 12.0 | **0.120 / 8.6** |

The formula arm repairs hard loopers for the first time on the real
model (and the defect still migrates); the data arm wins every axis.

## Reproduce the round

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "llamafactory==0.9.5" transformers peft accelerate datasets
python3 LlamaFactory/scripts/pipeline.py      # resumable, ~75 min on 2 CPU cores
python3 LlamaFactory/scripts/make_figures.py
```

Full-scale GPU recipes: `LlamaFactory/configs/sft_antiloop_gpu.yaml` and
`stage3_rcda8_train.py --steps 600 --lr 3e-4`.

## Note on LlamaFactory

LlamaFactory is a self-hosted framework — its demo site (LLaMA Board)
builds configs; the training itself runs wherever the framework is
installed. This round ran it here, on CPU, through its CLI (data arm)
and its trainer classes (formula arm). No cloud training occurred and no
number is projected: every figure, table and GIF reads from the JSONs in
`LlamaFactory/results/`.
