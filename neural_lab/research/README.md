<p align="center">
  <a href="../README.md"><img src="../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🔬 research — Standalone Programs

<p align="center">
  <em>Self-contained research programs on top of the core</em><br><br>
  <img src="../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="repository root"></a>
  <img src="https://img.shields.io/badge/tinygpt%20formula-00ffa3" alt="TinyGPT Formula">
</p>

Standalone research programs built on top of the core package. Each
folder is self-contained: it does not modify the `src/rmt_llm/` core
and ships its own scripts, data, and model artifacts — a research
program can be deleted without touching anything else, or copied out of
the repository and still work.

| Folder | Content |
|---|---|
| [`tinygpt_formula/`](tinygpt_formula/README.md) | **TinyGPT v3 Formula-Edition** — train the repository's NumPy-only transformer on the project's own formulas, evaluate held-out generalization, run the RMT diagnostics, and answer the ROADMAP open questions empirically. Includes the parameter-scaling A/B study (`formula` vs `control` corpus) that measures how true formula content affects confident hallucinations |

Inside the folder, five stages form the pipeline:
[`scripts/`](tinygpt_formula/scripts/README.md) →
[`data/`](tinygpt_formula/data/README.md) →
[`model/`](tinygpt_formula/model/README.md) →
[`scaling/`](tinygpt_formula/scaling/README.md), with
[`colab/`](tinygpt_formula/colab/README.md) as the free-GPU variant.

Start with [`tinygpt_formula/README.md`](tinygpt_formula/README.md) —
the full pipeline reproduces in ~20 minutes on CPU.

---

<p align="center">
  <sub>
    <a href="../README.md">← root</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../README.md">🏠 repository root</a> · <a href="tinygpt_formula/README.md">the program →</a>
  </sub>
</p>
