# RMT-LLM Visualization Suite (Java / JavaFX)

A JavaFX desktop application that renders the RMT-LLM mathematics in
interactive 3D: Marchenko-Pastur surfaces, BBP transition landscapes,
NHSE eigenvalue collapse, and per-layer free-energy scenes.

## Build and run

Requires JDK 17+ (OpenJFX is pulled in by the Gradle plugin):

```bash
cd java/rmt-llm-viz
./gradlew run          # or: gradle run
```

Project layout (`src/main/java/com/rmt/llm/viz/`):

| File | Role |
|---|---|
| `RMTLLMVizApp.java` | JavaFX application entry point |
| `RMTMath.java` | the RMT formulas (MP, BBP, Tracy-Widom, NHSE) |
| `RMTVerifier.java` | cross-implementation consistency checks |
| `RMTConstants.java` | shared constants mirroring `src/rmt_llm/constants.py` |
| `Complex.java` | minimal complex-number arithmetic |

The module mirrors the Python package `python/rmt_llm_viz/` — the CI
cross-implementation job keeps their reference values aligned.
