# Rust Laboratory

Rust port of the lab pipeline (`lab_en/` English, `lab_ru/` Russian).
Pure `std` + `rand` implementation — no external crates — covering the 3D
research loop, TinyGPT inference, and scenario verification.

```bash
cd laboratory/rust/lab_en
cargo test               # unit tests
cargo run --release      # interactive CLI
cargo clippy             # lint (advisory in CI)
```

Both crates are format-checked with `cargo fmt` and linted with clippy in
the pre-commit pipeline. Reports follow `laboratory/shared/schema.json`.
