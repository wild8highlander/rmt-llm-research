# 🗜️ Model ZIP — the intervention archive

<p align="center">
  <img src="https://img.shields.io/badge/archive-gemma3--rcds--interventions.zip-00D4FF" alt="archive badge">
  <img src="https://img.shields.io/badge/size-35.6%20MB%20(single%20volume%2C%20%3C%20100%20MB)-38E1C0" alt="size badge">
  <img src="https://img.shields.io/badge/contents-fp16%20deltas%20%C3%973%20variants%20%2B%20v%20directions-B026FF" alt="contents badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

The trained artefact of the program, shipped the way an intervention should be
shipped: not a 2.4 GB model dump, but the **fp16 deltas** that turn an
untouched base model into the trained one.

## What is inside

| Member | Contents |
|--------|----------|
| `rcda5_deltas.npz` | ΔW for `o_proj` of L6/L7/L8 (the RCDA-5 trainable set) |
| `rcda6_deltas.npz` | ΔW for `o_proj + down_proj` of L6/L7/L8 (the RCDA-6 hypothesis set) |
| `rcda7_deltas.npz` | same tensor set, RCDA-7 trained values (deep-phase penalty + tight anchor) |
| `rcds5_vectors.npz` | the behavioural directions `v_L{l}` from the probe |
| `manifest.json` | per-member SHA-256, provenance, the apply command, expected Gemma-scale sizes |

## Integrity

```
c2571e9e8f2deb1050516baadf1e6206f0938bc27ba627f5115878c29f35ae37  gemma3-rcds-interventions.zip
```

Verify with `sha256sum -c SHA256SUMS.txt`; per-member hashes are pinned inside
`manifest.json`.

## Apply

```bash
unzip gemma3-rcds-interventions.zip -d rcds_interventions/
python ../Code/gemma_apply_deltas.py \
    --deltas rcds_interventions/rcda5_deltas.npz \
    --model google/gemma-3-1b-it --save gemma3_rcda5_model
```

**Provenance.** The committed deltas are the *validation-run* tensors
(GPT-2 124M stand-in, CPU — see the [kit README](../README.md) provenance
section). The Gemma-scale deltas are one command away:
`Code/gemma_rcds_train.py --variant rcda5 --model google/gemma-3-1b-it …`
(≈8 MB for RCDA-5, ≈56 MB for RCDA-6/7 — still single-volume). The full-model
split-archive procedure lives in [`../Docs/UNPACK_MODEL.md`](../Docs/UNPACK_MODEL.md).

---

<p align="center">
  <sub><a href="../README.md">← kit root</a> · <a href="../Docs/UNPACK_MODEL.md">unpack guide →</a></sub>
</p>
