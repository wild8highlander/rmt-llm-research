# 📦 Docs — unpacking and archives

<a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>

Everything about turning the shipped archives back into a working model.

| Document | Contents |
|----------|----------|
| [`UNPACK_MODEL.md`](UNPACK_MODEL.md) | the intervention archive (`Model ZIP/gemma3-rcds-interventions.zip`): verify → unpack → apply the deltas; plus the split-volume procedure for a full-model archive under the GitHub 100 MB rule |

## The three ways this kit ships a trained model

| Artifact | Folder | What it is | How to apply |
|----------|--------|------------|--------------|
| fp16 deltas (validation layer) | [`../Model ZIP/`](../Model%20ZIP/README.md) | `W′ − W` of the three RCDA variants + the v-directions, 35.6 MB | `Code/gemma_apply_deltas.py` |
| full fine-tuned model | [`../Model — Z/`](../Model%20%E2%80%94%20Z/README.md) | the RCDA-5 SGD arm, base + trained o_proj L10–12, 1.59 GB in 17 self-assembling volumes | `cat` + `unzip` + the unpack script |
| LoRA adapters (real-model round) | [`../LlamaFactory/adapters/`](../LlamaFactory/adapters/) | peft adapters of the RCDA-8 formula arm and the SFT data arm | `PeftModel.from_pretrained(base, path)` |
