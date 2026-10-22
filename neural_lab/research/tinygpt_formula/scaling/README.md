<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 📈 scaling — A/B Study

<p align="center">
  <em>How hallucination confidence scales with parameters — formula vs control</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/study-a78bfa" alt="study"><img src="https://img.shields.io/badge/a%2Fb%20design-22d3ee" alt="A/B design">
</p>

The parameter-scaling study at the heart of the Formula-Edition
experiment: train a ladder of models on the **formula** corpus and its
**control** twin, then measure how confidently-wrong output grows with
size in each arm. The headline artifact is the plot
`hallucination_vs_params.png` — the empirical counterpart of the
theoretical collapse arguments in the papers.

| File | Content |
|---|---|
| `scaling_experiment.py` | the ladder runner — train, evaluate, log per run |
| `build_control_corpus.py` | constructs the control corpus twin |
| `scaling_report.json` | machine-readable results for every rung |
| `SCALING_RESULTS.md` | the written analysis and conclusions |
| `runs/` | per-rung artifacts |
| `TERMUX_PROOT_UBUNTU.md` | how the ladder was run on an Android/Termux device |

Start with `SCALING_RESULTS.md`; the corpora come from
[`data/`](../data/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scripts/README.md">pipeline scripts →</a>
  </sub>
</p>
