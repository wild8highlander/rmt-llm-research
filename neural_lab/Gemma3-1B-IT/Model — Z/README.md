# <sub>📦</sub> Model — Z — the fine-tuned model archive (RCDA-5, SGD arm)

<p align="center">
  <img src="https://img.shields.io/badge/model-Gemma3--1B--IT%20RCDA--5%20SGD-38E1C0" alt="model badge">
  <img src="https://img.shields.io/badge/size-1.59%20GB-00D4FF" alt="size badge">
  <img src="https://img.shields.io/badge/volumes-17-FFB020" alt="volume badge">
  <img src="https://img.shields.io/badge/max%20volume-95%20MB-FF3B6B" alt="limit badge">
  <img src="https://img.shields.io/badge/layer-real%20Gemma%20(unsloth%20mirror)-B026FF" alt="layer badge">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-Gemma3--1B--IT-FFB020" alt="parent"></a>
</p>

`gemma3-rcds-kit` — the fine-tuned model of the real-Gemma RCDA-5 round
(base `unsloth/gemma-3-1b-it` + the trained o_proj of layers 10/11/12,
SGD arm) with `training_log.json`, `ppl_check.json`, the provenance README
and a checksum self-check script, packaged as a **self-assembling split
archive** to stay under the 100 MB per-file GitHub limit:

```text
gemma3-rcds-interventions.zip.01 … gemma3-rcds-interventions.zip.17   (16 × 95.0 MB + 1 × 68.7 MB)
SHA256SUMS.txt                                    ← SHA-256 of every volume
Training/                                         ← the RCDA-6 intervention archive (2 volumes)
```

## Unpack

Put **all 17 volumes** in one directory (every volume is required), then:

```bash
sha256sum -c SHA256SUMS.txt          # verify every volume
cat gemma3-rcds-interventions.zip.* > gemma3-rcds-interventions.zip
unzip gemma3-rcds-interventions.zip
cd gemma3-rcds-interventions && python3 unpack_gemma3_rcda5.py
```

## The honest caveat, and the successor format

This model is the RCDA-5 SGD arm — the round whose behavioral outcome was
**negative** (ρ_bad 0.60 → 1.00: suppressing the v-projection at
L10–L12 made every bad prompt loop harder; see
[`../Reports MD/RCDA5_REPORT.md`](../Reports%20MD/RCDA5_REPORT.md)). It is
shipped as the *measured record of that negative result*, not as a cure.

The successor artifacts are **LoRA adapters**
([`../LlamaFactory/adapters/`](../LlamaFactory/adapters/)): the RCDA-8
formula arm (3/12 hard loopers repaired, the first of any formula arm on
this model) and the SFT data arm (5/12, ρ_bad 0.40) — a few MB each,
applied with `PeftModel.from_pretrained`, no split volumes needed. The
era of 17-volume archives is over.
