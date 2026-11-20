# UNPACK_MODEL.md — from archives to a working model

Written: November 18, 2026. Covers the two archives of the Gemma3-1B-IT kit.

## 1. The intervention archive (what this kit ships)

`Model-ZIP/gemma3-rcds-interventions.zip` — **35.6 MB, single volume** (under
the GitHub 100 MB per-file limit, no splitting needed). It contains the fp16
intervention deltas of all three rounds + the probe directions:

```text
rcda5_deltas.npz     3 tensors   (o_proj L6/L7/L8)
rcda6_deltas.npz     6 tensors   (o_proj + down_proj L6/L7/L8)
rcda7_deltas.npz     6 tensors   (same tensor set, RCDA-7 values)
rcds5_vectors.npz    the behavioural directions v_L{l}
manifest.json        per-member SHA-256 + provenance + apply command
```

```bash
# verify
sha256sum -c SHA256SUMS.txt            # in Model-ZIP/
# unpack
unzip gemma3-rcds-interventions.zip -d rcds_interventions/
# apply to the base model (any of the three variants)
python ../Code/gemma_apply_deltas.py \
    --deltas rcds_interventions/rcda5_deltas.npz \
    --model google/gemma-3-1b-it --save gemma3_rcda5_model
```

`--model openai-community/gpt2` rebuilds the validation-run model instead
(the committed deltas were produced on the stand-in — see the provenance
section of the [kit README](../README.md)).

## 2. A full-model archive (when *you* produce one)

A fine-tuned Gemma3-1B in fp16 is ~2.4 GB — over the GitHub 100 MB per-file
limit, so it must be shipped as a **self-assembling split archive**, the same
way the sibling kit ships `rmt-pocket-interventions.zip` + `.z01…`:

```bash
# split into 100 MB volumes (zip multi-volume syntax; 7z works too)
zip -s 100m gemma3-rcds-model.zip gemma3_rcda5_model/
#   → gemma3-rcds-model.zip + gemma3-rcds-model.z01 gemma3-rcds-model.z02 …
sha256sum gemma3-rcds-model* >> SHA256SUMS.txt

# self-assembly on the reader's side
cat gemma3-rcds-model.z01 gemma3-rcds-model.z02 … gemma3-rcds-model.zip > combined.zip
# or: zip -s 0 gemma3-rcds-model.zip --out combined.zip
unzip combined.zip -d gemma3_rcda5_model
sha256sum -c SHA256SUMS.txt
```

Rules the sibling kit established and this kit keeps:

1. every volume and the final assembly get a SHA-256 line in `SHA256SUMS.txt`;
2. the manifest inside the archive states which model, which round and which
   protocol produced it;
3. volumes are never regenerated — a new run means a new archive and a new
   manifest line.

## 3. Sizes to expect at Gemma scale

| Artifact | fp16 size |
|----------|-----------|
| RCDA-5 deltas (3 × o_proj, 1152×1152) | ~8 MB |
| RCDA-6/7 deltas (3 × o_proj + 3 × down_proj, 1152×6912) | ~56 MB |
| full edited model | ~2.4 GB → split archive |

The committed archive (GPT-2-scale tensors) is 35.6 MB — every member's
SHA-256 is pinned in `manifest.json` and the archive itself in
`SHA256SUMS.txt`.
