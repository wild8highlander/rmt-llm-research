<p align="center">
  <a href="../../../README.md"><img src="../../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧠 model — Artifacts

<p align="center">
  <em>Weights, tokenizer and every evaluation artifact of the formula model</em><br><br>
  <img src="../../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/artifacts-f472b6" alt="artifacts"><img src="https://img.shields.io/badge/numpy%20weights-3776AB" alt="NumPy weights">
</p>

Everything the trained model leaves behind. These files are
**machine-generated** — retraining overwrites them deterministically
(fixed seeds), so any figure produced from them is reproducible.

| File | Content |
|---|---|
| `tiny_gpt_formula.npz` | model weights |
| `tiny_gpt_formula_bpe.json` | BPE tokenizer merges |
| `training_history.json` | per-epoch loss / perplexity / match-rate |
| `run_config.json` | exact training configuration |
| `evaluation_report.json` | held-out scores: format / numeric / teacher-forced |
| `open_questions_results.json` | empirical answers to the ROADMAP open questions |

Regenerate with
[`scripts/train_formula_model.py`](../scripts/README.md); the web view
of these artifacts is the [`neural_lab/`](../../../README.md)
application.

---

<p align="center">
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scaling/README.md">scaling study →</a>
  </sub>
</p>
