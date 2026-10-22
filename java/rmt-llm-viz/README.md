<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# ☕ RMT-LLM Viz — JavaFX

<p align="center">
  <em>The RMT mathematics as interactive 3D scenes, Java edition</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-java-ED8B00" alt="java/"></a>
  <img src="https://img.shields.io/badge/java%20%2F%20javafx-ED8B00" alt="Java/JavaFX"><img src="https://img.shields.io/badge/jdk%2017%2B-8ca3c7" alt="JDK 17+">
</p>

A JavaFX desktop application that renders the RMT-LLM mathematics in
interactive 3D: Marchenko–Pastur surfaces, BBP transition landscapes,
NHSE eigenvalue collapse, and per-layer free-energy scenes. It is the
Java member of the visualization trio — same scenes, same formulas, same
constants as the [Python](../../python/rmt_llm_viz/README.md) and
[Julia](../../julia/RMTLLMViz/README.md) twins.

## Build and run

Requires JDK 17+ (OpenJFX is pulled in by the Gradle plugin):

```bash
cd java/rmt-llm-viz
./gradlew run          # or: gradle run
```

## Project layout

`src/main/java/com/rmt/llm/viz/`:

| File | Role |
|---|---|
| `RMTLLMVizApp.java` | JavaFX application entry point |
| `RMTMath.java` | the RMT formulas — MP, BBP, Tracy–Widom, NHSE |
| `RMTVerifier.java` | cross-implementation consistency checks |
| `RMTConstants.java` | shared constants mirroring `src/rmt_llm/constants.py` |
| `Complex.java` | minimal complex-number arithmetic |

---

<p align="center">
  <sub>
    <a href="../README.md">← java</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="../../python/rmt_llm_viz/README.md">Python viz →</a>
  </sub>
</p>
