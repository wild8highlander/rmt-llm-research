# <sub>📦</sub> Model ZIP — the RCDA-6 trained delta, self-assembling split archive

<p align="center">
  <img src="https://img.shields.io/badge/model-Gemma3--1B--IT%20RCDA--6-38E1C0" alt="model badge">
  <img src="https://img.shields.io/badge/size-111%20MB-00D4FF" alt="size badge">
  <img src="https://img.shields.io/badge/volumes-2-FFB020" alt="volume badge">
  <img src="https://img.shields.io/badge/max%20volume-91%20MB-FF3B6B" alt="limit badge">
</p>

`gemma3-rcda6-interventions` — what the RCDA-6 training actually changed,
packaged as a **self-assembling split archive** (every volume ≤ 95 MB,
GitHub-safe):

```text
gemma3-rcda6-interventions.zip.01   (91 MB)
gemma3-rcda6-interventions.zip.02   (20 MB)
SHA256SUMS.txt                      ← SHA-256 of every volume
```

**What is inside** — the trained **delta state**, not a full checkpoint:

| Path | Contents |
|------|----------|
| `model/trained_state.safetensors` | the 16 tensors the training changed (`o_proj` + `down_proj` of layers 8–15, bf16, 146.3 MB) — 7.31 % of the model |
| `model/SHA256.txt` | `e34299c023eb4162a66deecae4f06d997c491c1c3bf1c9a3c14271a54a30893a` |
| `results/training_log.json` | the 150-step record (loss components, benches at 0/150, final TW/cos analysis) |
| `results/ppl_check.json` | the abilities PPL check (base 10.9 → trained 14.1) |
| `verify_rcda6_delta.py` | checksum self-check |
| `README.txt` | provenance + the 6-line apply snippet |

## Unpack

Put **both volumes** in one directory, then:

```bash
sha256sum -c SHA256SUMS.txt
cat gemma3-rcda6-interventions.zip.01 gemma3-rcda6-interventions.zip.02 \
    > gemma3-rcda6-interventions.zip
unzip gemma3-rcda6-interventions.zip
cd gemma3-rcda6-interventions && python3 verify_rcda6_delta.py
```

Expected verdict: `OK — checksum verified` (round-trip verified at packaging
time). Step-by-step: [`../Docs/UNPACK_MODEL.md`](../../Docs/UNPACK_MODEL.md).

## Apply the delta to the base model

```python
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import torch

m = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it",
                                         dtype=torch.bfloat16)
delta = load_file("gemma3-rcda6-interventions/model/trained_state.safetensors")
with torch.no_grad():
    for k, w in delta.items():
        dict(m.named_parameters())[k].data.copy_(w)
```

The base checkpoint is **not** shipped: RCDA-6 edits 16 tensors of
`google/gemma-3-1b-it` (= the litert-community/Gemma3-1B-IT weights), so the
delta + the public base reconstructs the trained model exactly — and the
whole kit stays a two-volume, plain-git-pushable archive instead of RCDA-5's
seventeen. Gemma Terms of Use apply to the base weights.

---

<p align="center">
  <sub><a href="../README.md">← R CDA - 6 root</a> · <a href="../Docs/UNPACK_MODEL.md">unpacking guide →</a></sub>
</p>
