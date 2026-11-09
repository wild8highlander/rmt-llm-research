# stages/common — shared infrastructure

Utilities used by all stage runners:

- `cli.py` — common argument parsing for `run_stage*.py`
- `hf_model.py` — Hugging Face model loading/caching
- `generation.py` — generation loop with deterministic settings
- `passports.py` — run passports (config + environment snapshots)
- `index.py`, `reporting.py` — results indexing and report rendering

The results contract is in [`../results_schema/`](../results_schema/README.md).
