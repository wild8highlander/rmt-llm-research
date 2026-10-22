<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🧪 Julia Laboratory (RU)

<p align="center">
  <em>Русская версия лаборатории RMT-LLM на Julia — интерактивное меню, 3D-исследования, экспорт графиков и отчётов</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-julia-9558B2" alt="Julia">
  <img src="https://img.shields.io/badge/cross--language-twin-22d3ee" alt="status">
</p>

Лаборатория `Julia` — автономный порт исследовательской лаборатории
RMT-LLM: те же границы Марченко–Пастура, ландшафты BBP-перехода, коллапс
скин-эффекта NHSE и сцены свободной энергии по слоям, что и в Python-
оригинале, но в нативной экосистеме `Julia`. Это один из восьми
межъязыковых близнецов: эксперимент везде идентичен, отличается только
рантайм. Численные результаты сверяются с Python-ядром (`src/rmt_llm/`)
и Julia-верификатором, поэтому артефакты сопоставимы между реализациями.

Типичная сессия: выберите сценарий в интерактивном меню, запустите
спектральный проход и экспортируйте наборы графиков (PNG / SVG / PDF)
и отчёты в нескольких форматах (JSON · CSV · Markdown · HTML · TeX) —
те же артефакты, что живут в `results/`.

## 🚀 Quick start

```bash
cd laboratory/julia/lab_ru
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
[3D visualization suites](../../../python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
