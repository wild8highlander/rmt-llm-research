"""
neural_lab/science — the predictive research core of the laboratory.

Three layers are separated (see EXPERIMENT_PROTOCOL.md and
PREDICTION_ROADMAP.md):

  instrumentation  — what is actually observed in the model (logits,
  activations);
  features         — what is mathematically computed from observations;
  predictors       — what outcome is predicted from the features.

Plus the cross-cutting loops:
  data             — tasks with independent answer verification and the
  splits;
  evaluation       — metrics, confidence intervals, leakage checks;
  storage          — machine-readable passports and run artifacts;
  experiments      — reproducible runs (run_h1.py);
  results          — frozen results (required to be committed).

Principle: the prediction is saved BEFORE the laboratory learns the
right answer; the final metric is compared against a baseline and the
permutation null.
"""

__version__ = "0.1.0"

PROTOCOL_VERSION = "1.1"
