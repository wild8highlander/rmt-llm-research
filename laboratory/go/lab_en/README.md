<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧪 Go Laboratory (EN)

<p align="center">
  <em>The Go twin of the RMT-LLM research laboratory — interactive menu, 3D spectral research, chart & report export</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-go-00ADD8" alt="Go">
  <img src="https://img.shields.io/badge/cross--language-twin-22d3ee" alt="status">
</p>

The `Go` laboratory is a self-contained port of the RMT-LLM research
laboratory: the same Marchenko–Pastur bounds, BBP transition landscapes,
NHSE skin-effect collapse and per-layer free-energy scenes as the Python
original, re-expressed natively for the `Go` toolchain. It is one of
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
cd laboratory/go/lab_en
go run .
go test ./...    # parity tests
```

## 📂 Contents

| File | Role |
|---|---|
| `main.go` | menu-driven entry point |
| `research_3d.go` | 3D research pipeline |
| `research_3d_test.go` | parity tests against reference values |
| `go.mod` | module definition |

Results land in `results/` inside this folder (`charts/`, `charts_3d/`,
`models/`, `reports/`) — mirrored by `laboratory/results/` at the
laboratory root.

## 🌐 Cross-language family

The same lab exists in: [Python](../../python/) · [Julia](../../julia/) · [Java](../../java/) · [Rust](../../rust/) · [C++](../../cpp/) · [R](../../r/) — and in
[TinyGPT-driven notebooks](../../../notebooks/README.md) and the
[3D visualization suites](../../python/python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
