# HOW TO UNPACK THE MODEL (gemma3-rcda6-interventions)

The RCDA-6 trained **delta** (the 16 tensors the experiment changed) is
stored as a **self-assembling split archive** — one plain ZIP cut into byte
volumes, each under 95 MB so the kit stays pushable to GitHub without LFS:

    Model ZIP/gemma3-rcda6-interventions.zip.01   (91 MB)
    Model ZIP/gemma3-rcda6-interventions.zip.02   (20 MB)
    Model ZIP/SHA256SUMS.txt          ← SHA-256 of every volume

* total ≈ 111 MB — the ZIP holds the 146.3 MB bf16 delta (it does not
  compress) plus the training record and the PPL check;
* **the base checkpoint is NOT inside** — the delta applies onto
  `google/gemma-3-1b-it` (the litert-community/Gemma3-1B-IT weights).

## 1. Download BOTH volumes into ONE folder

Verify them (optional but recommended):

    sha256sum -c SHA256SUMS.txt

## 2. Reassemble the ZIP (one command)

Linux / macOS / Windows (WSL, Git Bash):

    cat gemma3-rcda6-interventions.zip.01 gemma3-rcda6-interventions.zip.02 \
        > gemma3-rcda6-interventions.zip

(7-Zip / WinRAR can also open `.zip.001` directly as a split set.)

## 3. Extract and verify

    unzip gemma3-rcda6-interventions.zip
    cd gemma3-rcda6-interventions
    python3 verify_rcda6_delta.py

Expected:

    trained_state.safetensors  146.3 MB
    sha256  e34299c023eb4162a66deecae4f06d997c491c1c3bf1c9a3c14271a54a30893a
    verdict OK — checksum verified

## 4. Apply the delta to the base model

    pip install torch transformers safetensors

```python
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
import torch

m = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it",
                                         dtype=torch.bfloat16)
delta = load_file("model/trained_state.safetensors")
with torch.no_grad():
    for k, w in delta.items():
        dict(m.named_parameters())[k].data.copy_(w)
# m is now the RCDA-6 trained model
```

The edit is confined to `model.layers.{8..15}.self_attn.o_proj.weight` and
`model.layers.{8..15}.mlp.down_proj.weight` — 73 138 176 weights, 7.31 % of
the model. Gemma license terms apply — accept the license on the base model
card before redistributing.

## 5. (Optional) package a full checkpoint yourself

If you want an offline full checkpoint like RCDA-5's kit:

```python
from safetensors.torch import save_file
# ... apply the delta as above, then:
save_file({k: v.data for k, v in m.state_dict().items()}, "model.safetensors")
```

then zip + `split -b 95000000` it the same way. The RCDA-5 folder ships the
full-checkpoint kit (17 volumes) if you want to see that packaging variant.
