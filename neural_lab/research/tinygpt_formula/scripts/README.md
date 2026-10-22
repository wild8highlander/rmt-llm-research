<p align="center">
  <a href="../../../README.md"><img src="../../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# ⚙️ scripts — Pipeline

<p align="center">
  <em>Corpus → training → evaluation → inference → open questions</em><br><br>
  <img src="../../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/5%20stages-00ffa3" alt="5 stages"><img src="https://img.shields.io/badge/cpu----friendly-22d3ee" alt="CPU friendly">
</p>

The five-stage pipeline of the Formula-Edition experiment. Run the
stages in order and the full study reproduces in roughly 20 minutes on a
CPU — every stage is deterministic and writes its artifacts next door in
[`model/`](../model/README.md).

| Script | Stage |
|---|---|
| `build_corpus.py` | 1 — build the formula + control corpora into [`data/`](../data/README.md) |
| `train_formula_model.py` | 2 — train TinyGPT (NumPy, BPE, fixed seeds) |
| `evaluate_model.py` | 3 — held-out evaluation: format / numeric / teacher-forced |
| `infer.py` | 4 — interactive inference; also backs the [`neural_lab/`](../../../README.md) playground |
| `open_questions.py` | 5 — empirical answers to the ROADMAP open questions |

Cloud variant: the notebooks in [`colab/`](../colab/README.md) run the
same stages on free GPU servers.

---

<p align="center">
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../colab/README.md">cloud notebooks →</a>
  </sub>
</p>
