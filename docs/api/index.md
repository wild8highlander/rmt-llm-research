# API Reference

This is the auto-generated API reference for `rmt-llm-research`. Every public
function, class, and constant is documented here, with type hints and
docstrings extracted from the source.

For tutorials and walkthroughs, see [Tutorials](../tutorials/index.md).

---

## Module index

| Module | What it contains | Source |
|--------|------------------|--------|
| [`rmt_llm.marchenko_pastur`](rmt-llm.md#marchenko_pastur) | MP density, bounds, CDF, Stieltjes transform | [`src/rmt_llm/marchenko_pastur.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/marchenko_pastur.py) |
| [`rmt_llm.bbp_transition`](rmt-llm.md#bbp_transition) | BBP phase transition, signal separation, fluctuation scaling | [`src/rmt_llm/bbp_transition.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/bbp_transition.py) |
| [`rmt_llm.tracy_widom`](rmt-llm.md#tracy_widom) | F₂ CDF/PDF, moments (mean, variance, skewness) | [`src/rmt_llm/tracy_widom.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/tracy_widom.py) |
| [`rmt_llm.nhse`](rmt-llm.md#nhse) | Winding number, skin strength, eigenvalue sampling | [`src/rmt_llm/nhse.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/nhse.py) |
| [`rmt_llm.caputo_fractional`](rmt-llm.md#caputo_fractional) | Caputo fractional-time memory, RLHF drift, Fokker-Planck, N_crit | [`src/rmt_llm/caputo_fractional.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/caputo_fractional.py) |
| [`rmt_llm.keating_snaith`](rmt-llm.md#keating_snaith) | Finite-N corrections to γ₁, N_crit | [`src/rmt_llm/keating_snaith.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/keating_snaith.py) |
| [`rmt_llm.ep_surfaces`](rmt-llm.md#ep_surfaces) | Exceptional point sensitivity, rounding effects | [`src/rmt_llm/ep_surfaces.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/ep_surfaces.py) |
| [`rmt_llm.thermodynamics`](rmt-llm.md#thermodynamics) | Free energy, Landauer cost, spectral entropy, RG flow | [`src/rmt_llm/thermodynamics.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/thermodynamics.py) |
| [`rmt_llm.constants`](rmt-llm.md#constants) | Centralized numerical constants (N_crit, γ, etc.) | [`src/rmt_llm/constants.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/src/rmt_llm/constants.py) |
| [`tiny_gpt`](tinygpt.md) | 2.5M-param transformer model | [`laboratory/python/lab_en/tiny_gpt.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/python/lab_en/tiny_gpt.py) |
| [`tiny_gpt_trainer.BPETokenizer`](bpe.md) | Pure-NumPy BPE tokenizer | [`laboratory/python/lab_en/tiny_gpt_trainer.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/python/lab_en/tiny_gpt_trainer.py) |
| [`tiny_gpt_trainer`](trainer.md) | Trainer, AdamState, generate_sample, loss functions | [`laboratory/python/lab_en/tiny_gpt_trainer.py`](https://github.com/wild8highlander/rmt-llm-research/blob/main/laboratory/python/lab_en/tiny_gpt_trainer.py) |

---

## Quick lookup

### By use case

| If you want to… | Use this |
|------------------|----------|
| Compute MP eigenvalue bounds | [`rmt_llm.marchenko_pastur.mp_bounds`](rmt-llm.md#rmt_llm.marchenko_pastur.mp_bounds) |
| Sample a random matrix | [`rmt_llm.marchenko_pastur.sample_wishart`](rmt-llm.md#rmt_llm.marchenko_pastur.sample_wishart) |
| Detect a BBP spike | [`rmt_llm.bbp_transition.bbp_is_spiked`](rmt-llm.md#rmt_llm.bbp_transition.bbp_is_spiked) |
| Compute Tracy-Widom CDF | [`rmt_llm.tracy_widom.tracy_widom_cdf`](rmt-llm.md#rmt_llm.tracy_widom.tracy_widom_cdf) |
| Compute N_crit | [`rmt_llm.constants.N_CRIT`](rmt-llm.md#rmt_llm.constants.N_CRIT) |
| Build a TinyGPT model | [`tiny_gpt.TinyGPT`](tinygpt.md#tiny_gpt.TinyGPT) |
| Train TinyGPT | [`tiny_gpt_trainer.train`](trainer.md#tiny_gpt_trainer.train) |
| Generate text | [`tiny_gpt_trainer.generate_sample`](trainer.md#tiny_gpt_trainer.generate_sample) |
| Load trained weights | [`tiny_gpt_trainer.load_trained_model`](trainer.md#tiny_gpt_trainer.load_trained_model) |

### By type

| Type | Examples |
|------|----------|
| Functions | `mp_bounds`, `bbp_is_spiked`, `tracy_widom_cdf` |
| Classes | `TinyGPT`, `TinyGPTConfig`, `BPETokenizer`, `AdamState`, `TrainConfig` |
| Constants | `N_CRIT`, `GAMMA_VALUE`, `MP_DEFAULT_Q` |
| Dataclasses | `TinyGPTConfig`, `TinyLayer`, `TrainConfig` |
