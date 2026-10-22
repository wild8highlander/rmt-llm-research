<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🖼️ charts — Run Gallery

<p align="center">
  <em>Per-run chart bundles produced by the 3D research pipeline</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-results-f472b6" alt="results/"></a>
  <img src="https://img.shields.io/badge/machine----generated-8ca3c7" alt="machine-generated">
</p>

Each subfolder is one research run — named `YYYYMMDD_HHMMSS_<experiment>`
— holding the PNG / SVG / PDF triplets for every figure of that run.
The bundles are produced by `research_3d.py` and its cross-language
twins; nothing here is hand-edited, which is what makes any figure in a
paper traceable to its exact experiment.

| Folder | Run |
|---|---|
| `20260813_112352_3d_exp_12/` | 3D experiment 12 |
| `20260814_*`, `ALL_3D` bundles | aggregated 3D scenes across experiments |

Regenerate:

```bash
cd laboratory/python/lab_en
python main.py    # menu item "Research 3D"
```

Curated, publication-grade visualizations live separately in
[`assets/charts/`](../../assets/charts/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← results</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="../reports/README.md">reports →</a>
  </sub>
</p>
