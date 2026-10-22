<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 📊 charts — Gallery

<p align="center">
  <em>Publication-grade 600 dpi figures — theory panels and repository metrics</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-charts-00ffa3" alt="assets/"></a>
  <img src="https://img.shields.io/badge/600%20dpi-22d3ee" alt="600 dpi"><img src="https://img.shields.io/badge/11%20figures-a78bfa" alt="11 figures">
</p>

Two families of figures in the same dark-spectrum style. The **theory panels**
are faithful demonstrations of the mathematics implemented in
`src/rmt_llm/` (synthetic data, marked on-chart); the **metrics
figures** are computed from the actual repository tree and CI
configuration.

## Theory panels

| File | Content |
|---|---|
| `marchenko_pastur.png` | MP density vs empirical eigenvalues of an activation gram matrix |
| `circle_law.png` | Girko circular law — eigenvalues filling the unit disk |
| `tracy_widom.png` | convergence of largest-eigenvalue fluctuations to Tracy–Widom |
| `dyson_brownian.png` | Dyson Brownian motion — eigenvalue repulsion trajectories |
| `bbp_transition.png` | Baik–Ben Arous–Péché spiked transition at β = 1 |
| `spectral_entropy.png` | entropy separation of grounded vs hallucinating continuations |
| `entropy_heatmap.png` | spectral entropy map — attention heads × layers |
| `hallucination_roc.png` | ROC of the spectral hallucination detector |
| `training_dynamics.png` | TinyGPT v2 loss / match-rate reference run |

## Repository metrics

| File | Content |
|---|---|
| `repo_languages.png` | polyglot composition donut (counted from the tree) |
| `repo_dashboard.png` | repository health — KPI cards, modules, CI matrix, stack radar |

Regenerate:

```bash
python scripts/gen_charts.py     # matplotlib, 600 dpi, custom colormap
```

---

<p align="center">
  <sub>
    <a href="../README.md">← assets</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../branding/README.md">branding →</a>
  </sub>
</p>
