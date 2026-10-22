<p align="center">
  <img src="../../../assets/logos/sc_res.banner.svg" width="760" alt="science/results — animated logo: trophy with sparkles">
</p>

# 🏆 science/results — Fixed Results

<p align="center">
  <em>Every published number, with its passport and generated report</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--results-fbbf24" alt="results"></a>
  <img src="https://img.shields.io/badge/4%20result%20sets-fbbf24" alt="4 result sets">
</p>

Where runs become citable. Each subfolder is one frozen result set: the
raw traces (`observations/labels/predictions.jsonl`), the task set, the
passport with full provenance, the machine-readable `report.json`, and a
`REPORT.md` generated from that JSON — never edited by hand, so the
numbers cannot drift from the data. Result folders are append-only
history: a new question means a new run, not an edit of an old folder.

## Result sets

| Folder | What it is | Headline |
|---|---|---|
| [`h1_v1/`](h1_v1/REPORT.md) | H1 on the S cell (447K, protocol v1.0) | primary underpowered (ΔAUC +0.206, p=0.097); secondary «Utility Trap» significant (0.871 vs 0.794, p=0.0025) |
| [`h1_v1_M/`](h1_v1_M/REPORT.md) | H1 on the M cell (1.11M, protocol v1.1) | ΔAUC(D−B) +0.037 [−0.042, +0.127] — direction preserved, not significant |
| [`h1_v1_L/`](h1_v1_L/REPORT.md) | H1 on the L cell (4.06M, protocol v1.1) | ΔAUC(D−B) +0.116 [+0.012, +0.215] — bootstrap CI excludes 0 for the first time |
| [`h1_scaling/`](h1_scaling/REPORT.md) | cross-cell aggregation over S/M/L | direction D > B replicated 3/3; confidence baseline non-monotonic across scale |

## Anatomy of a result folder

```text
h1_v1_L/
├── passport.json        — sha256 of weights / corpus / dataset, seeds, env
├── tasks.json           — the exact task set used
├── observations.jsonl   — per-observation internal-state traces
├── labels.jsonl         — independently recomputed ground truth
├── predictions.jsonl    — per-model predictions
├── report.json          — the numbers (source of truth)
└── REPORT.md            — human view, generated from report.json
```

## Rules

- Read `report.json`, trust `passport.json`, quote `REPORT.md`.
- Weights are not committed (`*.npz` gitignored) — reproduction means
  re-training the cell (same seed) and re-running the experiment.
- The registry status of each hypothesis ([`hypotheses/`](../hypotheses/README.md))
  must stay consistent with what these folders show.

---

<p align="center">
  <img src="../../../assets/logos/sc_res.avatar.svg" width="56" alt="science/results animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../storage/README.md">storage →</a>
  </sub>
</p>
