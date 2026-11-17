# <sub>📦</sub> Model ZIP — the split archive of the fine-tuned Gemma3-1B-IT (RCDA-5, SGD arm)

<p align="center">
  <img src="https://img.shields.io/badge/model-Gemma3--1B--IT%20RCDA--5-38E1C0" alt="model badge">
  <img src="https://img.shields.io/badge/size-1.59%20GB-00D4FF" alt="size badge">
  <img src="https://img.shields.io/badge/volumes-17-FFB020" alt="volume badge">
  <img src="https://img.shields.io/badge/max%20volume-95%20MB-FF3B6B" alt="limit badge">
</p>

`gemma3-rcds-kit` — the fine-tuned model (base `google/gemma-3-1b-it` +
the trained o_proj of layers 10/11/12) with `training_log.json`,
`ppl_check.json`, the provenance README and a checksum self-check script,
packaged as a **self-assembling split archive** to stay under the 100 MB
per-file GitHub limit:

```text
gemma3-rcds-interventions.zip.01 … gemma3-rcds-interventions.zip.17   (16 × 95.0 MB + 1 × 68.7 MB)
SHA256SUMS.txt                                    ← SHA-256 of every volume
```

## Unpack

Put **all 17 volumes** in one directory (every volume is required), then:

```bash
sha256sum -c SHA256SUMS.txt          # verify every volume
cat gemma3-rcds-interventions.zip.* > gemma3-rcds-interventions.zip
unzip gemma3-rcds-interventions.zip
cd gemma3-rcds-interventions && python3 unpack_gemma3_rcda5.py
```

Expected self-check verdict:

```text
model.safetensors  1999.8 MB
sha256             90489a1a58383cc3619b1a88465eb2edfaf83a8a3a570ab0a9d611536af6ad5e
verdict            OK — checksum verified
```

7-Zip / WinRAR can open `gemma3-rcds-interventions.zip.001` directly as a split set.
Step-by-step text: [`../Docs/UNPACK_MODEL.md`](../Docs/UNPACK_MODEL.md).

## Inside the kit

| Path | Contents |
|------|----------|
| `model/` | `model.safetensors` (bf16, 2000 MB — the full fine-tuned checkpoint), `config.json`, `generation_config.json`, tokenizer (`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`), `SHA256SUMS.txt` |
| `results/` | `training_log.json` (the 150-step SGD record) and `ppl_check.json` |
| `unpack_gemma3_rcda5.py` | the checksum self-check |
| `README.txt` | provenance: base checkpoint `google/gemma-3-1b-it` (= the litert-community/Gemma3-1B-IT weights), mirror used, license note |

The merged archive's SHA-256 is
`3bbc33e494626f0dddabcf3490c8854d5685361589314c6e0e4e04ae066c9d0a`
(round-trip verified: volumes → merged ZIP → extracted `model.safetensors`
→ SHA-256 match).

## Why a split archive

GitHub caps any single file at 100 MB. Seventeen ≤ 95 MB volumes keep the
kit pushable with plain git — no LFS, no external hosting — and the
reassembly is a one-command, checksum-verified step. The model weights
differ from the base checkpoint in exactly three tensors
(`model.layers.{10,11,12}.self_attn.o_proj.weight`), but the kit ships the
**full checkpoint** so it is offline-ready with no reassembly of weight
parts. Gemma Terms of Use apply to the weights.

---

<p align="center">
  <sub><a href="../README.md">← R CDA - 5 root</a> · <a href="../Docs/UNPACK_MODEL.md">unpacking guide →</a></sub>
</p>
