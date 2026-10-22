<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🧪 Java Laboratory (EN)

<p align="center">
  <em>The Java twin of the RMT-LLM research laboratory — interactive menu, 3D spectral research, chart & report export</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-java-ED8B00" alt="Java">
  <img src="https://img.shields.io/badge/cross--language-twin-22d3ee" alt="status">
</p>

The `Java` laboratory is a self-contained port of the RMT-LLM research
laboratory: the same Marchenko–Pastur bounds, BBP transition landscapes,
NHSE skin-effect collapse and per-layer free-energy scenes as the Python
original, re-expressed natively for the `Java` toolchain. It is one of
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
cd laboratory/java/lab_en
javac *.java
java Main
```

## 📂 Contents

| File | Role |
|---|---|
| `Main.java` | menu-driven entry point |
| `Research.java` / `Research3D.java` | RMT spectral experiments and the 3D research pipeline |
| `Charts.java` / `Reports.java` | chart bundles and multi-format reports |
| `Scenarios.java` / `Parameters.java` | scenario presets and configuration |
| `ModelDownloader.java` | model download helper |

Results land in `results/` inside this folder (`charts/`, `charts_3d/`,
`models/`, `reports/`) — mirrored by `laboratory/results/` at the
laboratory root.

## 🌐 Cross-language family

The same lab exists in: [Python](../../python/) · [Julia](../../julia/) · [Rust](../../rust/) · [Go](../../go/) · [C++](../../cpp/) · [R](../../r/) — and in
[TinyGPT-driven notebooks](../../../notebooks/README.md) and the
[3D visualization suites](../../../python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
