# Results — Experiment Artifacts

Machine-generated outputs of the 3D research pipeline. Everything in this
tree is produced by `laboratory/python/lab_en/research_3d.py` (and its
cross-language twins) — do not edit by hand.

```text
results/
├── reports/     # per-run reports: JSON, CSV, YAML, Markdown, HTML, TeX, ...
└── charts/      # per-run chart bundles: PNG / SVG / PDF triplets
```

Naming convention: `<timestamp>_<experiment-id>.<ext>` (for example
`20260813_112416_ALL_3D.json`). Files ending in `.pdf.txt`, `.xlsx.txt`, ...
are textual stubs produced when an optional export backend was unavailable
in the run environment; the canonical machine-readable formats are `.json`,
`.csv` and `.yaml`.

To regenerate:

```bash
cd laboratory/python/lab_en
python main.py            # menu item "Research 3D"
```

Status artifacts of the live experiments are also mirrored into the
Neural Lab app (`neural_lab/`).
