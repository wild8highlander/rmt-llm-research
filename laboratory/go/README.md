<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🐹 Go Laboratory

<p align="center">
  <em>Go twins of the laboratory — quick builds, parity-tested</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/go--go-00ADD8" alt="Go">
</p>

Two language variants of the Go research laboratory — the same
scenarios, artifacts and export formats as every other twin in
[`laboratory/`](../README.md). Pick `go` if it is the toolchain you
already have installed; the experiment is identical everywhere.

| Folder | Content |
|---|---|
| [`lab_en/`](lab_en/README.md) | English variant — menu-driven lab |
| [`lab_ru/`](lab_ru/README.md) | Russian variant — same code, translated UI |

`research_3d_test.go` asserts parity against the reference values.

```bash
cd laboratory/go/lab_en
go run .
go test ./...
```

Results land in each lab's `results/` folder and are mirrored in
[`laboratory/results/`](../results/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="lab_en/README.md">lab_en →</a>
  </sub>
</p>
