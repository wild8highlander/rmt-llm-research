# Java Laboratory

Java port of the lab pipeline (`lab_en/` English, `lab_ru/` Russian).
A Gradle-driven console application covering scenarios, the 1D/3D research
runs, TinyGPT inference, and multi-format report export.

```bash
cd laboratory/java/lab_en
gradle run               # or build the jar and run it directly
```

| File | Role |
|---|---|
| `Main.java` | interactive menu entry point |
| `Research.java` / `Research3D.java` | 1D and 3D experiment engines |
| `TinyGPT.java` | TinyGPT forward pass + generation |
| `Charts.java` / `Reports.java` | chart and report export |
| `Scenarios.java` | adversarial verification scenarios |
| `SimpleJson.java` | minimal JSON writer for the shared schema |
