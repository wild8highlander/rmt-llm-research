<p align="center">
  <a href="../../README.md"><img src="../../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🌌 RMT-LLM Viz — Python

<p align="center">
  <em>Interactive 3D visualization of the spectral mathematics, NumPy + Matplotlib</em><br><br>
  <img src="../../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-python-3776AB" alt="python/"></a>
  <img src="https://img.shields.io/badge/python-3776AB" alt="Python"><img src="https://img.shields.io/badge/numpy%20%2B%20matplotlib-22d3ee" alt="NumPy+Matplotlib">
</p>

`rmt_llm_viz` — an interactive 3D visualization and analysis toolkit for
Random Matrix Theory applied to LLM spectral analysis. Pure Python
(NumPy + Matplotlib), no GUI framework required: a menu-driven console
launches each scene, and every scene exports to PNG/SVG for papers.

## Features

- Marchenko–Pastur 3D density surfaces
- BBP phase-transition animated 3D landscapes
- Non-Hermitian Skin Effect eigenvalue ring/collapse visualization
- EP-surface sensitivity 3D ridgeline plots
- Thermodynamic free-energy landscape across transformer layers
- Cross-module consistency verification dashboard

## Usage

```bash
pip install -e ".[dev]"        # from the repository root
python python/rmt_llm_viz/main.py
```

Siblings: the [JavaFX suite](../../../java/java/rmt-llm-viz/README.md) and the
[Julia suite](../../../julia/julia/RMTLLMViz/README.md) render the same scenes.

---

<p align="center">
  <sub>
    <a href="../README.md">← python</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="../../assets/charts/README.md">chart gallery →</a>
  </sub>
</p>
