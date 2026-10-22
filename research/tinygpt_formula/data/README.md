<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🗂️ data — Corpora

<p align="center">
  <em>The formula corpus and its control twin</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/corpus-00ffa3" alt="corpus"><img src="https://img.shields.io/badge/train%20%2B%20val%20%2B%20control-22d3ee" alt="train+val+control">
</p>

The training and evaluation corpora for the Formula-Edition experiment —
and, crucially, its **control twin**. The experiment is an A/B design:
the `formula` corpus carries true project formulas, the `control` corpus
is a structurally identical but content-free counterpart. Any
difference in confident-hallucination behaviour between the two models
is therefore attributable to the content, not to surface statistics.

| File | Role |
|---|---|
| `formula_corpus_train.txt` / `formula_corpus_val.txt` | formula corpus — train / held-out validation |
| `formula_corpus_train_control.txt` / `formula_corpus_val_control.txt` | the control corpus — same shape, no true formulas |
| `corpus_stats.json` | sizes, token counts and overlap statistics for both corpora |

Build or rebuild the corpora with
[`scripts/build_corpus.py`](../scripts/README.md); the study that
consumes them is [`scaling/`](../scaling/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../model/README.md">model artifacts →</a>
  </sub>
</p>
