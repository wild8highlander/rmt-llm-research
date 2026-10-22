<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧬 src — Source of Truth

<p align="center">
  <em>The NumPy core every other implementation is verified against</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-src-00ffa3" alt="repository root"></a>
  <img src="https://img.shields.io/badge/python-3776AB" alt="Python"><img src="https://img.shields.io/badge/numpy----only-22d3ee" alt="NumPy only">
</p>

This folder holds the canonical implementation of the RMT-LLM
mathematical framework. The package [`rmt_llm/`](rmt_llm/README.md)
inside is the single source of truth for every formula in the project —
Marchenko–Pastur bounds, the BBP transition threshold, Tracy–Widom
fluctuations, the Non-Hermitian Skin Effect criterion, thermodynamic
costs and more. All other language implementations (Julia, Java, Go,
Rust, C++, R) are cross-verified against it in CI, which is what turns a
collection of ports into one trustworthy experiment.

```text
src/
└── rmt_llm/         # the core verification package
    ├── *.py         # one module per mathematical object
    └── tests/       # 171+ tests incl. hypothesis property tests
```

Start points: the [package README](rmt_llm/README.md), the
[test suite](rmt_llm/tests/README.md), and the
[API reference](../docs/api/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="rmt_llm/README.md">the package →</a>
  </sub>
</p>
