# MNN-Pocket-0.3B — hands-on RMT kit for a real 0.3B model

Working directory of the intervention experiments on the MNN-Pocket-0.3B
transformer: probing spectra layer by layer, fitting Marchenko-Pastur,
detecting BBP outliers and applying weight-space interventions.

- [`Code/`](Code/README.md) — probe, benchmark and intervention scripts
  (`pocket_rmt_probe.py`, `benchmark_rmt.py`, `rmt_intervention.py`,
  `reassemble_model.py`, ...).
- [`Docs/`](Docs/README.md) — unpacking instructions for the split model
  archive.
- [`Graphic/`](Graphic/README.md) — result figures (per-layer spectra, MP
  fit, BBP theta, outliers, benchmark summary).
- [`Model ZIP/`](Model%20ZIP/README.md) — the split archive
  `rmt-pocket-interventions.zip` (+ `.z01`–`.z06`), one volume per GitHub
  100 MB limit.
- `rmt-llm-research_rcda2.zip`, `rcda3.zip`, `rcda4.zip` — snapshots of the
  RCDA rounds (repo at the end of rounds 2–4). Historical: the live results
  live in `neural_lab/science/` and `results/`.

Related research programs: [`../research/`](../research/README.md) and
[`../science/`](../science/README.md).
