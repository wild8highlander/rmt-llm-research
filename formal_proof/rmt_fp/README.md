# formal_proof/rmt_fp — the formal-model library

Python package backing the formal proof: the idealized layer operator and
its spectral theory.

- `mp.py` — Marchenko-Pastur machinery
- `spiked.py` — spiked (low-rank perturbation) models, BBP threshold
- `shrinkage.py` — spectral shrinkage operators
- `layer_ops.py` — composition rules for layer operators
- `entropy.py` — entropy/imaginary-part identities
- `hallucination.py` — the formal hallucination-split theorem

Everything here is verified numerically by [`../experiments/`](../experiments/README.md).
