# 🧮 stage 3 — N_crit estimation

The critical-pressure estimator: fits the Caputo-fractional order and the
collapse statistics across the pressure grid, producing the N_crit
surface and the fits that feed stage 5.

| Path | Role |
|------|------|
| [`run_stage3.py`](run_stage3.py) | the stage entry point (CLI, resumable) |
| [`../results_schema/`](../results_schema/README.md) | the `fit.json` / `report.json` contracts |
| [`../README.md`](../README.md) | the pipeline overview |

Offline smoke run (tiny random model, seconds):

```bash
python neural_lab/stages/stage3_ncrit/run_stage3.py --n-tasks 4 \
    --max-new-tokens 8 --n-boot 50 --out /tmp/stage3-smoke
```

Committed pythia-70m smoke artifacts:
[`../results/smoke_pythia70m/stage3/`](../results/smoke_pythia70m/stage3/)
(`fit.json` validated against the schema at write time).
