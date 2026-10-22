<p align="center">
  <a href="../../README.md"><img src="../../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🖼️ RMTLLMViz

<p align="center">
  <em>The Julia twin of the 3D visualization suite</em><br><br>
  <img src="../../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-julia-9558B2" alt="julia/"></a>
  <img src="https://img.shields.io/badge/julia-9558B2" alt="Julia"><img src="https://img.shields.io/badge/plots.jl-a78bfa" alt="Plots.jl">
</p>

The Julia member of the 3D visualization trio: Marchenko–Pastur density
surfaces, BBP transition landscapes, NHSE ring collapse and
exceptional-point ridgelines — computed with the same constants as the
verifier and exported as static scenes for papers and talks. Being pure
Julia, it starts fast, needs no GUI framework, and runs anywhere the
verifier runs.

## Usage

```bash
cd julia/RMTLLMViz
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. src/RMTLLMViz.jl
```

Siblings: [`RMTLLMVerify/`](../RMTLLMVerify/README.md) (numbers),
[`python/rmt_llm_viz/`](../../../python/python/rmt_llm_viz/README.md) (Python
edition), [`java/rmt-llm-viz/`](../../../java/java/rmt-llm-viz/README.md)
(JavaFX edition).

---

<p align="center">
  <sub>
    <a href="../README.md">← julia</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="../../../python/python/rmt_llm_viz/README.md">Python viz →</a>
  </sub>
</p>
