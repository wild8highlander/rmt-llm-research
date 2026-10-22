<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧪 Test Suite

<p align="center">
  <em>171+ tests that keep the mathematics honest</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-tests-22d3ee" alt="rmt_llm/"></a>
  <img src="https://img.shields.io/badge/pytest-0a9edc" alt="pytest"><img src="https://img.shields.io/badge/hypothesis-1f6feb" alt="hypothesis"><img src="https://img.shields.io/badge/pytest----benchmark-a78bfa" alt="pytest-benchmark">
</p>

The verification layer of the core package. Beyond classic unit tests,
the suite uses **Hypothesis** property-based tests (mathematical
invariants must hold for whole families of random matrices, not just
fixed fixtures) and **pytest-benchmark** performance baselines, so a
regression in either correctness or speed fails CI before it can reach
the cross-implementation consistency checks.

| File | Role |
|---|---|
| `conftest.py` | shared fixtures — deterministic RNG seeds, matrix builders |
| `test_rmt_llm.py` | the mathematical contract of every module |
| `test_bugfixes.py` | regression tests — each fixed bug is pinned here forever |

## Run

```bash
pytest src/rmt_llm/tests/ -v          # full suite (~12 s)
make test                             # same via Makefile
make test-hypothesis                  # property-based tests only
make test-bench                       # performance benchmarks
```

The numbers these tests assert on are the same reference values the
[Julia verifier](../../../laboratory/julia/julia/RMTLLMVerify/README.md) re-checks on
Julia 1.9 / 1.10 / 1.11 — one contract, two languages.

---

<p align="center">
  <sub>
    <a href="../README.md">← rmt_llm</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../../../laboratory/julia/julia/RMTLLMVerify/README.md">Julia verifier →</a>
  </sub>
</p>
