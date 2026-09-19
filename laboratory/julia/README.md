# Julia Laboratory

Julia port of the lab pipeline (`lab_en/` English, `lab_ru/` Russian).
Implements the scenario runner, the 1D/3D research experiments, TinyGPT
training, and the report exporter using LinearAlgebra-based code.

```bash
cd laboratory/julia/lab_en
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. main.jl          # interactive menu
julia --project=. run_full_lab.jl  # full pipeline
```

Outputs are JSON/YAML reports compatible with `laboratory/shared/schema.json`,
so results from all 8 language implementations can be compared directly.
Julia is also exercised in CI via the `julia/RMTLLMVerify` test suite.
