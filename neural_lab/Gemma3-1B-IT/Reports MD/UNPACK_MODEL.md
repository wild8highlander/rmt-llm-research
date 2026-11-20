# HOW TO UNPACK THE MODEL (gemma3-rcds-kit)

The fine-tuned model (Gemma3-1B-IT, RCDA-5 SGD arm) is stored as a
**self-assembling split archive** — one plain ZIP cut into byte volumes, each
under 100 MB so the kit stays pushable to GitHub without LFS:

```text
Model ZIP/gemma3-rcds-interventions.zip.01 … gemma3-rcds-interventions.zip.17
Model ZIP/SHA256SUMS.txt          ← SHA-256 of every volume
```

- total size ≈ 1.59 GB (the ZIP holds the 2.0 GB bf16 `model.safetensors`
  plus config/tokenizer — safetensors compresses ~20 %);
- every volume is exactly ≤ 95 000 000 bytes (GitHub hard limit: 100 MB).

## 1. Download ALL 17 volumes into ONE folder

Every part is required — one missing volume makes the kit unreadable.
Verify them (optional but recommended):

```bash
sha256sum -c SHA256SUMS.txt
```

## 2. Reassemble the ZIP (one command)

Linux / macOS / Windows (WSL, Git Bash):

```bash
cat gemma3-rcds-interventions.zip.* > gemma3-rcds-interventions.zip
```

The merged archive's SHA-256 is
`3bbc33e494626f0dddabcf3490c8854d5685361589314c6e0e4e04ae066c9d0a`.

## 3. Extract

```bash
unzip gemma3-rcds-interventions.zip
```

(7-Zip / WinRAR can also open `gemma3-rcds-interventions.zip.001` directly as a split
set — "Open archive → .001 file".) You get:

```text
gemma3-rcds-interventions/
├── README.txt                  ← provenance + quick start
├── unpack_gemma3_rcda5.py      ← checksum self-check
├── model/                      ← model.safetensors (2000 MB) + config + tokenizer
└── results/                    ← training_log.json + ppl_check.json
```

## 4. Verify the weights (one command, ~10 s)

```bash
cd gemma3-rcds-interventions
python3 unpack_gemma3_rcda5.py
```

Expected:

```text
model.safetensors  1999.8 MB
sha256             90489a1a58383cc3619b1a88465eb2edfaf83a8a3a570ab0a9d611536af6ad5e
verdict            OK — checksum verified
```

## 5. Use the model (offline)

```text
pip install torch transformers safetensors
python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("gemma3-rcds-interventions/model")
t = AutoTokenizer.from_pretrained("gemma3-rcds-interventions/model")
ids = t("Дождь идёт. Дождь идёт. Дождь идёт. Дождь идёт.",
            return_tensors="pt").input_ids
out = m.generate(ids, max_new_tokens=192, do_sample=False)
print(t.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
EOF
```

The checkpoint differs from `google/gemma-3-1b-it` in exactly three tensors
(`model.layers.{10,11,12}.self_attn.o_proj.weight`). Gemma license terms
apply — accept the license on the base model card before redistributing.
