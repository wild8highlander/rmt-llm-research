# Julia Packages

Two independent Julia projects mirroring parts of the Python core:

| Package | Purpose |
|---|---|
| [`RMTLLMVerify/`](RMTLLMVerify/) | verification package for the RMT-LLM framework — spectral statistics (Marchenko-Pastur, BBP, Tracy-Widom), the 10 paths to N_crit, and the RLHF utility-trap dynamics. CI runs `Pkg.test()` against the Python reference values on Julia 1.9 / 1.10 / 1.11. |
| [`RMTLLMViz/`](RMTLLMViz/) | the Julia twin of the 3D visualization suite — MP density surfaces, BBP landscapes, NHSE ring collapse, EP ridgelines. |

## Usage

```bash
cd julia/RMTLLMVerify
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'
```

Both packages are pure Julia (stdlib + LinearAlgebra/Plots) and are kept
numerically consistent with `src/rmt_llm/` — the "Cross-Implementation
Consistency" CI job asserts Python-vs-Julia equality on reference values.
