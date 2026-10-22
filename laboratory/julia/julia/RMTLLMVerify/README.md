<p align="center">
  <a href="../../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🔬 RMTLLMVerify

<p align="center">
  <em>The Julia verifier — every reference value, re-derived independently</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-julia-9558B2" alt="julia/"></a>
  <img src="https://img.shields.io/badge/julia%201.9%20%7C%201.10%20%7C%201.11-9558B2" alt="Julia 1.9–1.11"><img src="https://img.shields.io/badge/pkg.test-00ffa3" alt="Pkg.test">
</p>

The verification package for the RMT-LLM framework, written in pure
Julia (stdlib + LinearAlgebra/Plots). It re-derives the spectral
statistics — Marchenko–Pastur, BBP, Tracy–Widom — the 10 paths to
`N_crit`, and the RLHF utility-trap dynamics, and asserts them against
the Python reference values. CI runs `Pkg.test()` on Julia 1.9, 1.10
and 1.11, which is what makes the "cross-implementation consistency"
claim more than a promise.

## Usage

```bash
cd julia/RMTLLMVerify
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Its numeric twin is the Python core [`src/rmt_llm/`](../../src/rmt_llm/README.md);
its visual twin is [`RMTLLMViz/`](../RMTLLMViz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← julia</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="../RMTLLMViz/README.md">RMTLLMViz →</a>
  </sub>
</p>
