# laboratory/results/models — trained checkpoints

`tiny_gpt_local.npz` — the TinyGPT checkpoint produced by the local
Python training run (`laboratory/python/lab_en/`): weights + optimizer
state in a single NumPy archive. Reload it with `numpy.load(...)` and the
model code from the lab; the trainer is described in
[`../../python/lab_en/`](../../python/lab_en/README.md).

Checkpoints are reproducible from the committed training scripts; treat
this directory as a cache, not a source of truth.
