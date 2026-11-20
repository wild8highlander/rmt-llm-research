# Gemma3-1B-IT.md — the patient

The model under intervention in this kit:
[`litert-community/Gemma3-1B-IT`](https://huggingface.co/litert-community/Gemma3-1B-IT)
— the LiteRT community distillation of **`google/gemma-3-1b-it`** (Gemma 3, 1B
instruction-tuned). Per the repository owner's context this release is a
distilled, uncensored build — restraint filters cut out, a heretic by
construction — which makes its raw autoregressive behaviour, including the
repetition-loop defect this kit treats, directly observable and measurable.
Its training, in this repository, is training **by the repository's own
formulas**: the RCDA penalty `pen = (vᵀh/‖v‖)²`, not a generic recipe.

| Property | Value |
|----------|-------|
| Patient (on-device build) | `litert-community/Gemma3-1B-IT` |
| HF training mirror | `google/gemma-3-1b-it` (Gemma license, gated) |
| Weights actually trained (rounds 5–8) | `unsloth/gemma-3-1b-it` @ snapshot `5b1141…` (hash-pinned, checksum-verified) |
| Architecture | Gemma3ForCausalLM — 26 blocks, d = 1152, 1.00 B params, 262,144-token vocabulary |
| Defect under treatment | greedy repetition loops (4-gram recurrence, loop_share ≥ 0.5) |
| Defect baseline (real model, greedy 192) | ρ_bad **0.60**, mean loop_share **0.562** (20-prompt bilingual suite, raw completion mode) |
| Writers found by the probe (real model) | blocks **L10/L11/L12** — z_diff **212.6 / 219.8 / 205.1**, band L8–L15 |
| Validation stand-in | `openai-community/gpt2` — 124M, 12 blocks, d = 768 (CPU; adapter in `Code/rcds_common.py`) |
| Interventions | `pen = (vᵀh/‖v‖)² + β·relu(mse − cap)` — full-matrix (RCDA-5/6) and LoRA via LlamaFactory (RCDA-8, 150 steps) |
| Best measured outcome (RCDA-8 round) | formula arm **3/12** hard loopers repaired; data arm **5/12**, ρ_bad 0.60 → **0.40**, PPL 10.9 → **8.6** |

Why a stand-in ever ran: the Gemma mirror is gated (license acceptance +
token) and the early capture hardware had no GPU, so the pipeline mechanics
were validated end-to-end on GPT-2 with an architecture-adaptive loader.
The real-model runs came later (`Reports-MD/`, `JSON-Results/`) and the
RCDA-8 round (`LlamaFactory/`) completes them: same snapshot, same
protocol, trained through LlamaFactory 0.9.5 on a CPU-only box. What the
record now holds: the probe replicates byte-exact (writers L10/11/12, ρ_bad
0.60), the formula arm repairs hard loopers for the first time, and the
defect migrates under every edit geometry tried so far.
