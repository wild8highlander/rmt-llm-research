<p align="center">
  <a href="../README.md"><img src="../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🔮 julia — Julia Packages

<p align="center">
  <em>The independent verifier and the Julia visualization twin</em><br><br>
  <img src="../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-julia-9558B2" alt="repository root"></a>
  <img src="https://img.shields.io/badge/julia%201.9–1.11-9558B2" alt="Julia 1.9–1.11">
</p>

Two independent Julia projects mirror parts of the Python core — one
owns the numbers, the other owns the pictures:

| Package | Purpose |
|---|---|
| [`RMTLLMVerify/`](RMTLLMVerify/README.md) | **verification package** — spectral statistics (Marchenko–Pastur, BBP, Tracy–Widom), the 10 paths to `N_crit`, and the RLHF utility-trap dynamics. CI runs `Pkg.test()` against the Python reference values on Julia 1.9 / 1.10 / 1.11 |
| [`RMTLLMViz/`](RMTLLMViz/README.md) | the Julia twin of the 3D visualization suite — MP density surfaces, BBP landscapes, NHSE ring collapse, EP ridgelines |

## Usage

```bash
cd julia/RMTLLMVerify
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Both packages are pure Julia (stdlib + LinearAlgebra/Plots) and are kept
numerically consistent with `src/rmt_llm/` — the "Cross-Implementation
Consistency" CI job asserts Python-vs-Julia equality on reference
values, so a formula is only considered implemented when two languages
agree on it.

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="RMTLLMVerify/README.md">the verifier →</a>
  </sub>
</p>
