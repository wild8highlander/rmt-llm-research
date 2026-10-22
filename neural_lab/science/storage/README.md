<p align="center">
  <img src="../../../assets/logos/sc_store.banner.svg" width="760" alt="science/storage — animated logo: archive cube with streaming bits">
</p>

# 📦 science/storage — Provenance & Artifacts

<p align="center">
  <em>Passports, hashes, and the discipline of traceable runs</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--storage-60a5fa" alt="storage"></a>
  <img src="https://img.shields.io/badge/sha256-22d3ee" alt="sha256"><img src="https://img.shields.io/badge/schemas-60a5fa" alt="schemas">
</p>

The memory of the science core. Storage answers one question for every
number the core has ever published: *exactly which inputs produced
it?* It does that with passports — structured records binding a run to
the sha256 of the model weights, the corpus, and the dataset, plus
seeds and environment versions — and with schema-checked artifact
serialization, so a result folder written today parses identically a
year from now.

## Modules

| File | Responsibility |
|---|---|
| `artifacts.py` | writes/reads run artifacts (passport, JSONL traces, report JSON) with atomic writes and stable ordering |
| `schemas.py` | the record schemas; every artifact is validated before it is allowed to represent a result |
| `__init__.py` | public surface |

## The passport

```json
{
  "experiment_id": "error_prediction_rmt_v1",
  "hypothesis_id": "H1",
  "model_revision": "51f72ab690c4…",
  "dataset_hash": "411eb0e9bb23…",
  "split_id": "grouped_kfold_v1",
  "seed": 42,
  "feature_groups": ["confidence", "spectral"],
  "prediction_target": "answer_incorrect",
  "primary_metric": "roc_auc",
  "protocol_version": "1.1"
}
```

A run without a valid passport is not a result — the runners in
[`experiments/`](../experiments/README.md) refuse to emit one, and the
aggregators refuse to consume one.

## Why it matters in practice

The S/M/L ladder comparison only works because all three cells can
prove they used the same dataset (`411eb0e9bb23…`) and the same split
(`grouped_kfold_v1`), while honestly declaring different weight hashes
and protocol versions (1.0 for S, 1.1 for M/L). That mix of
*comparable where it matters* and *explicit where it differs* is exactly
what the storage layer is for.

---

<p align="center">
  <img src="../../../assets/logos/sc_store.avatar.svg" width="56" alt="science/storage animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../tests/README.md">tests →</a>
  </sub>
</p>
