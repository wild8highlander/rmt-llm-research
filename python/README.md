# RMT-LLM Visualization Suite (Python)

`rmt_llm_viz` — an interactive 3D visualization and analysis toolkit for
Random Matrix Theory applied to LLM spectral analysis. Pure Python
(NumPy + Matplotlib), no GUI framework required.

## Features

- Marchenko-Pastur 3D density surfaces
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

A menu-driven console launches the interactive plots; every scene is also
exportable as PNG/SVG for papers.

Cross-implementation twin: the JavaFX suite in `java/rmt-llm-viz/` renders
the same mathematics — both are kept numerically consistent.
