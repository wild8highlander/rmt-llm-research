# 🗜️ Model ZIP — the intervention archive

<p align="center">
  <img src="https://img.shields.io/badge/archive-gemma3--rcds--interventions.zip-00D4FF" alt="archive badge">
  <img src="https://img.shields.io/badge/size-35.6%20MB%20(single%20volume%2C%20%3C%20100%20MB)-38E1C0" alt="size badge">
  <img src="https://img.shields.io/badge/contents-fp16%20deltas%20%C3%973%20variants%20%2B%20v%20directions-B026FF" alt="contents badge">
  <img src="https://img.shields.io/badge/layer-validation%20(GPT--2%20stand--in)-FF3B6B" alt="layer badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The trained artefact of the validation layer, shipped the way an
intervention should be shipped: not a multi-GB model dump, but the **fp16
deltas** that turn an untouched base model into the trained one.

| Member | Contents | SHA-256 (pinned in `SHA256SUMS.txt` / `manifest.json`) |
|--------|----------|------------|
| `rcda5_deltas.npz` | 3 × o_proj deltas (writer layers) | `322c93a9…` |
| `rcda6_deltas.npz` | 16 × (o_proj + down_proj) band deltas | `43a8ca46…` |
| `rcda7_deltas.npz` | same tensor set, deep-phase-only penalty | `e0240673…` |
| `rcds5_vectors.npz` | the probe directions v per layer | `b228688a…` |

**Apply** (rebuild a trained model from any pristine base checkpoint):

```bash
python Code/gemma_apply_deltas.py --deltas Model\ ZIP/rcda6_deltas.npz \
    --model google/gemma-3-1b-it --save out_dir
```

> **Provenance.** These deltas are the *validation-run* tensors (GPT-2
> stand-in, see the [archive section](../README.md#-archive--the-gpt-2-stand-in-validation-layer)
> of the main README) — they prove the shipping format and the apply
> path. The **real-model** trained artifacts are the RCDA-5 SGD full-model
> archive ([`../Model — Z/`](../Model%20%E2%80%94%20Z/README.md)) and the
> RCDA-8 round's peft LoRA adapters
> ([`../LlamaFactory/adapters/`](../LlamaFactory/adapters/)) — the LoRA
> format is this archive's successor: 2.6 MB instead of 16 MB of deltas,
> applicable with one `PeftModel.from_pretrained` call, no rebuild script.
