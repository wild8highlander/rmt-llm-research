<p align="center">
  <img src="../../../assets/logos/model.banner.svg" width="760" alt="model — animated logo: chip with pins and a live signal waveform">
</p>

# 🧠 model — Artifacts

<p align="center">
  <em>Weights, tokenizer and every evaluation artifact of the formula model</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/artifacts-f472b6" alt="artifacts"><img src="https://img.shields.io/badge/numpy%20weights-3776AB" alt="NumPy weights">
</p>

Everything the trained model leaves behind. These files are
**machine-generated** — retraining overwrites them deterministically
(fixed seeds), so any figure produced from them is reproducible. Treat
this folder as the *single source of truth* for every number quoted in
the README set: if a README table and an artifact here ever disagree,
the artifact wins and the README is stale. Weights themselves are
gitignored (binary churn without review value); every *derived* report
is committed, which is enough to reproduce or audit any published claim.

## Files

| File | Content |
|---|---|
| `tiny_gpt_formula.npz` | model weights |
| `tiny_gpt_formula_bpe.json` | BPE tokenizer merges |
| `training_history.json` | per-epoch loss / perplexity / match-rate |
| `run_config.json` | exact training configuration |
| `evaluation_report.json` | held-out scores: format / numeric / teacher-forced |
| `open_questions_results.json` | empirical answers to the ROADMAP open questions |

## Provenance chain

Each artifact knows what produced it, and the chain is fully connected:

```text
build_corpus.py          train_formula_model.py         evaluate_model.py / open_questions.py
     └── data/  ───────▶   tiny_gpt_formula.npz  ───────▶  evaluation_report.json
         (corpus hash)       + _bpe.json                  open_questions_results.json
                             + training_history.json
                             + run_config.json
```

`run_config.json` records architecture, seeds, schedule, and epoch
budget; `training_history.json` records what actually happened each
epoch. Together they make any `model/` state reconstructible — which is
why the science core can pin `model_revision` (sha256 of the weights) in
its experiment passports and the H1 runs remain traceable to the exact
weights that produced them.

## Who consumes what

| Consumer | Artifacts used |
|---|---|
| web app Training tab | `training_history.json` |
| web app Benchmark tab | `evaluation_report.json` |
| web app Questions tab | `open_questions_results.json` |
| web app Playground (via `/api/infer`) | `tiny_gpt_formula.npz` + `_bpe.json` |
| science core (H1) | weights + corpus hash via passport `model_revision` |

## Regeneration matrix

| To regenerate | Run |
|---|---|
| corpora (upstream) | `scripts/build_corpus.py` |
| weights + tokenizer + history | `scripts/train_formula_model.py` |
| evaluation report | `scripts/evaluate_model.py` |
| open-questions results | `scripts/open_questions.py` |

Regenerate with
[`scripts/train_formula_model.py`](../scripts/README.md); the web view
of these artifacts is the [`neural_lab/`](../../../README.md)
application.

---

<p align="center">
  <img src="../../../assets/logos/model.avatar.svg" width="56" alt="model animated avatar"><br>
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scaling/README.md">scaling study →</a>
  </sub>
</p>
