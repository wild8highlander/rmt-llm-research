# Code — probe, benchmark, intervene

Scripts for the MNN-Pocket-0.3B experiments (run from this directory):

| script | purpose |
| --- | --- |
| `pocket_rmt_probe.py` | extracts per-layer activation/weight spectra of the pocket model |
| `probe_real_gpt2.py` | same spectral probe for a reference GPT-2 checkpoint |
| `benchmark_rmt.py` | behavioral benchmark before/after an intervention |
| `rmt_intervention.py` | the RMT-guided weight-space edit utilities (spectral TW control) |
| `marchenko_pastur.py` | MP-law fit helpers used by the probe |
| `plot_benchmark.py` | renders the benchmark figures into `../Graphic/` |
| `reassemble_model.py` | rebuilds the model from the split archive in `../Model ZIP/` |

The theory behind these scripts is documented in `docs/monographs/` and the
NumPy core in `src/rmt_llm/`.
