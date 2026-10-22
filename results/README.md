<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🗃️ results — Experiment Artifacts

<p align="center">
  <em>Machine-generated outputs of the 3D research pipeline</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-results-f472b6" alt="repository root"></a>
  <img src="https://img.shields.io/badge/do%20not%20edit-8ca3c7" alt="do not edit">
</p>

Machine-generated outputs of the 3D research pipeline. Everything in
this tree is produced by `laboratory/python/lab_en/research_3d.py` (and
its cross-language twins) — do not edit by hand.

```text
results/
├── reports/     # per-run reports: JSON, CSV, YAML, Markdown, HTML, TeX, …
└── charts/      # per-run chart bundles: PNG / SVG / PDF triplets
```

Naming convention: `<timestamp>_<experiment-id>.<ext>` (for example
`20260813_112416_ALL_3D.json`). Files ending in `.pdf.txt`,
`.xlsx.txt`, … are textual stubs produced when an optional export
backend was unavailable in the run environment; the canonical
machine-readable formats are `.json`, `.csv` and `.yaml`.

## Regenerate

```bash
cd laboratory/python/lab_en
python main.py            # menu item "Research 3D"
```

Related: the per-run mirrors at
[`laboratory/results/`](../laboratory/results/README.md) and the curated
[chart gallery](../assets/charts/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="charts/README.md">charts →</a>
  </sub>
</p>
