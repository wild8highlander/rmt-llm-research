<p align="center">
  <a href="../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# ☕ java — Java Suite

<p align="center">
  <em>The JavaFX visualization twin and the Java laboratory</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-java-ED8B00" alt="repository root"></a>
  <img src="https://img.shields.io/badge/java%2017%2B-ED8B00" alt="Java 17+"><img src="https://img.shields.io/badge/javafx-8ca3c7" alt="JavaFX">
</p>

The Java corner of the polyglot experiment. Two residents:

| Path | Content |
|---|---|
| [`rmt-llm-viz/`](rmt-llm-viz/README.md) | **JavaFX desktop app** — Marchenko–Pastur surfaces, BBP landscapes, NHSE collapse and per-layer free-energy scenes as interactive 3D |
| (laboratory twin) | the Java research lab lives at [`laboratory/java/`](../README.md) — menu-driven experiments, chart and report export |

## Quick taste

```bash
cd java/rmt-llm-viz
./gradlew run          # JDK 17+; OpenJFX pulled in by the Gradle plugin
```

The math constants mirror `src/rmt_llm/constants.py` one-to-one
(`RMTConstants.java`), and `RMTVerifier.java` re-runs the
cross-implementation consistency checks from inside the GUI — the same
contract the [Julia verifier](../../julia/julia/RMTLLMVerify/README.md) enforces
on the numbers side.

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="rmt-llm-viz/README.md">the viz app →</a>
  </sub>
</p>
