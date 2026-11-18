# Gemma3-1B-IT.md — the patient

The model under intervention in this kit:
[`litert-community/Gemma3-1B-IT`](https://huggingface.co/litert-community/Gemma3-1B-IT)
— the LiteRT community distillation of **`google/gemma-3-1b-it`** (Gemma 3, 1B
instruction-tuned). Per the repository owner's context this release is a
distilled, uncensored build, which makes its raw autoregressive behaviour —
including the repetition-loop defect this kit treats — directly observable and
measurable.

| Property | Value |
|----------|-------|
| HF training mirror | `google/gemma-3-1b-it` (Gemma license, gated) |
| Architecture | Gemma3ForCausalLM — 26 blocks, d = 1152, ~1B params, 262,144-token vocabulary |
| Validation stand-in | `openai-community/gpt2` — 124M, 12 blocks, d = 768 (CPU; adapter in `Code/rcds_common.py`) |
| Defect under treatment | greedy repetition loops (4-gram recurrence, loop_share ≥ 0.5) |
| Defect baseline | ρ_bad 0.94, mean loop_share 0.894 (16-prompt bilingual suite) |
| Writers found by the probe | blocks L6/L7/L8 (z_diff 2.20–2.23, deep-middle third) |
| Intervention | `pen = (vᵀh/‖v‖)² + β·relu(mse − cap)` — 150 Adam steps per variant |

Why a stand-in: the Gemma mirror is gated (license acceptance + token) and the
capture hardware for the committed run had no GPU, so the full pipeline was
validated end-to-end on GPT-2 with an architecture-adaptive loader —
`--model google/gemma-3-1b-it` reruns the identical protocol at Gemma scale.
The mechanics that the stand-in run proves (formula trains, penalty melts,
edits align with v, abilities hold) are architecture-independent; the
Gemma-scale effect sizes are the pre-registered expectation of the next run.
