# Docs/ — how to run and unpack (R CDA - 6)

| Document | Contents |
|----------|----------|
| [`RCDA_RUNBOOK.md`](RCDA_RUNBOOK.md) | the full re-run recipe: phase probe → 150-step training (atomic checkpoints) → mini-bench → evaluator → PPL → digest → figures → packaging, with the measured expectations at every step |
| [`UNPACK_MODEL.md`](../../Docs/UNPACK_MODEL.md) | reassembling the `Model ZIP/` split archive, verifying the delta checksum, and applying the 16-tensor delta onto the `google/gemma-3-1b-it` base checkpoint |

The one-paragraph version: the trained artifact of RCDA-6 is a **delta**
(the 16 tensors the formula-training changed), shipped as a 2-volume
self-assembling ZIP in [`../Model ZIP/`](../../Model%20ZIP); reconstruct the
model by applying the delta onto the public base checkpoint
(`google/gemma-3-1b-it` — the litert-community/Gemma3-1B-IT weights).
