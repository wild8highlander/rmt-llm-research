# Scripts — Developer Utilities

Helper scripts for the TinyGPT v2 training pipeline and cross-language
laboratory synchronization. These are **developer tools**, not part of the
runtime library — they live outside `laboratory/` to keep the lab tree clean.

## Contents

| Script | Purpose |
|--------|---------|
| `train_v2_chunk.py` | Chunked training of TinyGPT v2 with checkpoint recovery. Splits a 30-epoch run into 5-epoch invocations so the process can be safely interrupted and resumed. |
| `sync_ru_trainer.py` | Synchronizes `lab_en/tiny_gpt_trainer.py` to `lab_ru/` with translated Russian docstrings and UI strings. Run after modifying the EN trainer. |

## Usage

All scripts should be run from the repository root:

```bash
cd /path/to/rmt-llm-research

# Train TinyGPT v2 in chunks (5 epochs per invocation)
python scripts/train_v2_chunk.py 0 5    # epochs 1-5
python scripts/train_v2_chunk.py 5 10   # epochs 6-10 (resumes from checkpoint)
python scripts/train_v2_chunk.py 25 30  # final chunk

# Sync EN trainer changes to RU
python scripts/sync_ru_trainer.py
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINYGPT_REPO` | auto-detected | Absolute path to repository root. Override only if running from an unusual location. |
| `TINYGPT_LAB` | `lab_en` | Which lab to target (`lab_en` or `lab_ru`). |

## Outputs

The chunked trainer writes to:

- `laboratory/python/{lab}/results/models/tiny_gpt_trained.npz` — model weights (resume point)
- `laboratory/python/{lab}/results/models/tiny_gpt_bpe.json` — BPE tokenizer
- `laboratory/python/{lab}/results/models/training_history.json` — per-epoch metrics
- `scripts/train_v2_log.txt` — append-only training log

## Notes

- All paths are resolved relative to the repository root via `__file__`. No hardcoded `/home/...` paths — the scripts work on any machine after `git clone`.
- The chunked trainer uses deterministic seeding, so resuming from a checkpoint produces identical results to a single uninterrupted run.
