<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🦀 Rust Laboratory

<p align="center">
  <em>Rust twins of the laboratory — memory-safe, single static binary</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/rust--rust-DEA584" alt="Rust">
</p>

Two language variants of the Rust research laboratory — the same
scenarios, artifacts and export formats as every other twin in
[`laboratory/`](../README.md). Pick `rust` if it is the toolchain you
already have installed; the experiment is identical everywhere.

| Folder | Content |
|---|---|
| [`lab_en/`](lab_en/README.md) | English variant — menu-driven lab |
| [`lab_ru/`](lab_ru/README.md) | Russian variant — same code, translated UI |

3D research pipeline included (`research_3d.rs`); identical scenarios to the other twins.

```bash
cd laboratory/rust/lab_en
cargo build --release
./target/release/rmt_llm_lab_en
```

Results land in each lab's `results/` folder and are mirrored in
[`laboratory/results/`](../results/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="lab_en/README.md">lab_en →</a>
  </sub>
</p>
