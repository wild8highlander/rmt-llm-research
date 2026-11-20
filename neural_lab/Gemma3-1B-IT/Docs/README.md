<p align="center">
  <img src="../assets/logos/gemma-rcda-avatar.svg" width="48" alt="RCDA kit avatar">
</p>

# 📦 Docs — unpacking and archives

<p align="center">
  <img src="https://img.shields.io/badge/guides-1-00D4FF" alt="guides badge">
  <img src="https://img.shields.io/badge/artifact%20formats-4-B026FF" alt="formats badge">
  <img src="https://img.shields.io/badge/checksums-SHA--256%20pinned-38E1C0" alt="checksums badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

Everything about turning the shipped archives back into a working model.

| Document | Contents |
|----------|----------|
| [`UNPACK_MODEL.md`](UNPACK_MODEL.md) | the intervention archive (`Model-ZIP/gemma3-rcds-interventions.zip`): verify → unpack → apply the deltas; plus the split-volume procedure for a full-model archive under the GitHub 100 MB rule |

## The four ways this kit ships a trained model

| Artifact | Folder | What it is | How to apply |
|----------|--------|------------|--------------|
| fp16 deltas (validation layer) | [`../Model-ZIP/`](../Model%20ZIP/README.md) | `W′ − W` of the three RCDA variants + the v-directions, 35.6 MB | `Code/gemma_apply_deltas.py` |
| full fine-tuned model | [`../Model-Z/`](../Model%20%E2%80%94%20Z/README.md) | the RCDA-5 SGD arm, base + trained o_proj L10–12, 1.59 GB in 17 self-assembling volumes | `cat` + `unzip` + the unpack script |
| LoRA adapters (RCDA-8) | [`../LlamaFactory/adapters/`](../LlamaFactory/adapters/) | peft adapters of the RCDA-8 formula arm and the SFT data arm | `PeftModel.from_pretrained(base, path)` |
| LoRA adapters (RCDA-9) | [`../RCDA-9/results/`](../R%20-%20CDA%20%E2%80%94%208/results) | the formula-corpus arm (10/12 repaired) + the pen re-run | `PeftModel.from_pretrained(base, path)` |

```mermaid
flowchart LR
    A["SHA256SUMS.txt"] --> V["verify every volume"]
    V --> C["cat volumes → zip"]
    C --> U["unzip"]
    U --> P["unpack script / apply deltas / PeftModel"]
    P --> M["trained model"]
```

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> ·
  <a href="UNPACK_MODEL.md">the unpack guide →</a> ·
  <a href="../Model-Z/README.md">the model archive →</a></sub>
</p>
