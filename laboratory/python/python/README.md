<p align="center">
  <a href="../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🐍 python — Python Suite

<p align="center">
  <em>The interactive 3D visualization toolkit, NumPy + Matplotlib only</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-python-3776AB" alt="repository root"></a>
  <img src="https://img.shields.io/badge/python-3776AB" alt="Python"><img src="https://img.shields.io/badge/no%20gui%20framework-22d3ee" alt="no GUI framework">
</p>

[`rmt_llm_viz/`](rmt_llm_viz/README.md) — the Python member of the
visualization trio. A menu-driven console launches interactive 3D
scenes of the RMT mathematics; every scene exports to PNG/SVG for
papers, and the whole toolkit runs anywhere NumPy runs.

## What it renders

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

Siblings: [JavaFX edition](../../java/java/rmt-llm-viz/README.md) ·
[Julia edition](../../julia/julia/RMTLLMViz/README.md). The core package they
visualize lives in [`src/rmt_llm/`](../../../src/rmt_llm/README.md); the
research lab in [`laboratory/python/`](../README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="rmt_llm_viz/README.md">the toolkit →</a>
  </sub>
</p>
