# rmt_llm — Core Verification Package

Pure-NumPy implementation of the mathematical framework applying Random
Matrix Theory (RMT) to the spectral analysis of LLM activations. This package
is the single source of truth for every formula: all other implementations
(Julia, Java, Go, Rust, C++, R) are cross-verified against it in CI.

## Modules

| Module | Content |
|---|---|
| `marchenko_pastur` | MP bounds and density — the noise floor of activation spectra |
| `bbp_transition` | BBP critical threshold theta_c — signal-vs-noise phase transition |
| `tracy_widom` | Tracy-Widom fluctuations of the largest eigenvalue |
| `nhse` | Non-Hermitian Skin Effect winding number criterion |
| `caputo_fractional` | Caputo fractional dynamics — entropic collapse timelines |
| `keating_snaith` | Keating-Snaith corrected gamma — N_crit estimation |
| `ep_surfaces` | Exceptional-point sensitivity surfaces |
| `thermodynamics` | Landauer cost and autoregressive irreversibility |
| `free_probability` | Free cumulants and multiplicative free convolution |
| `circular_ensembles` | Circular ensembles — spectral rigidity probes |
| `dyson_brownian` | Dyson Brownian motion — eigenvalue repulsion dynamics |
| `constants` | Shared physical and numerical constants (N_crit, gamma_1, ...) |

## Layout

```text
src/rmt_llm/
├── __init__.py        # public API
├── __main__.py        # `rmt-llm-verify` CLI entry point
├── tests/             # 101 pytest tests (unit + regression + cross-impl)
└── *.py               # the modules above
```

## Quick start

```bash
pip install -e ".[dev]"     # from the repository root
pytest src/rmt_llm/tests/   # run the verification suite
rmt-llm-verify              # run the verification CLI
```

Requires Python 3.10+ and NumPy only — no heavy scientific stack.

Author: Iskhak Hamzatovich Isaev · ORCID 0009-0003-7299-0701
