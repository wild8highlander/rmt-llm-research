<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧪 Julia Laboratory (EN)

<p align="center">
  <em>The Julia twin of the RMT-LLM research laboratory — interactive menu, 3D spectral research, chart & report export</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/cross--language-twin-22d3ee" alt="status">
</p>

The `Julia` laboratory is a self-contained port of the RMT-LLM research
laboratory: the same Marchenko–Pastur bounds, BBP transition landscapes,
NHSE skin-effect collapse and per-layer free-energy scenes as the Python
original, re-expressed natively for the `Julia` toolchain. It is one of
eight cross-language twins that keep the experiment identical everywhere —
only the runtime differs. Numerical outputs are cross-checked against the
Python core (`src/rmt_llm/`) and the Julia verifier, so a result produced
here can be compared byte-for-byte with its siblings.

Typical session: pick a scenario from the interactive menu, run the
spectral sweep, then export chart bundles (PNG / SVG / PDF triplets) and
multi-format reports (JSON · CSV · Markdown · HTML · TeX) — the same
artifacts you will find in `results/`.

## 🚀 Quick start

```bash
cd laboratory/julia/lab_en
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia main.jl
```

## 📂 Contents

| File | Role |
|---|---|
| `main.jl` | menu-driven entry point |
| `research.jl` / `research_3d.jl` | RMT spectral experiments and the 3D research pipeline |
| `charts.jl` / `reports.jl` | chart bundles and multi-format reports |
| `parameters.jl` / `model_downloader.jl` | configuration and model download helper |
| `Project.toml` | package environment (stdlib + LinearAlgebra/Plots) |

Results land in `results/` inside this folder (`charts/`, `charts_3d/`,
`models/`, `reports/`) — mirrored by `laboratory/results/` at the
laboratory root.

## 🌐 Cross-language family

The same lab exists in: [Python](../../python/) · [Java](../../java/) · [Rust](../../rust/) · [Go](../../go/) · [C++](../../cpp/) · [R](../../r/) — and in
[TinyGPT-driven notebooks](../../../notebooks/README.md) and the
[3D visualization suites](../../python/python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
