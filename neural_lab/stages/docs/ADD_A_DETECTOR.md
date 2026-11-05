# ADD_A_DETECTOR — how to add a detector to the Stage-2 race

The detector race is a CLOSED vocabulary (`detectors.KNOWN_DETECTORS`):
a typo'd name fails at `Unit` construction, not at the end of a run.
Adding a detector is four steps, in this order:

1. **Implement the scorer.** A detector maps one scored unit to a float
   with the orientation **higher = more likely hallucinated**. If it
   derives from existing measurements (per-step log-probs, entropies,
   hidden states), add a function to the corresponding module
   (`token_entropy.py`, `semantic_entropy.py`, `spectral_features.py`)
   and wire it where the units are built (`run_stage2.observe_units` or
   the `add_*_units` helpers). Do not re-derive measurements — raw
   observations are captured once, features are derived from them.
2. **Register the name** in `stage2_detector/detectors.py::KNOWN_DETECTORS`.
   The name is part of the report contract (`units.jsonl`, AUROC table)
   and of `results_schema/v1`; do not rename an existing entry — add a
   new one and deprecate the old.
3. **Add tests.** At minimum: the scorer's orientation (a known-hallucinated
   construction scores higher than a known-good one) and its presence in
   the race for a fixture run. Property-style invariants (score is finite,
   survives the `Unit` validation) belong in `tests/`.
4. **Bump the protocol.** A new detector changes what the race measures —
   that is a protocol change: append to `EXPERIMENT_PROTOCOLS.md`
   (changelog section), bump `PROTOCOL_VERSION` in `stages/__init__.py`,
   and update the `S2` entry in `science/hypotheses/registry.yaml` if the
   baselines list changes. The `protocol_doc_hash` in every passport
   makes the freeze verifiable.

What NOT to do: tune the new detector on the eval labels and then report
its AUROC. Anything fitted on labels (like the probes and the spectral
probe) must stay out-of-fold by construction.
