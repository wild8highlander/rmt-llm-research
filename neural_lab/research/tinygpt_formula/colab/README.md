<p align="center">
  <a href="../../../README.md"><img src="../../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# ☁️ colab — Notebooks

<p align="center">
  <em>Run the formula experiments on free cloud GPUs</em><br><br>
  <img src="../../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/colab-F9AB00" alt="Colab"><img src="https://img.shields.io/badge/2%20notebooks-22d3ee" alt="2 notebooks">
</p>

Two Colab-ready notebooks that run the TinyGPT Formula-Edition pipeline
on free compute — useful when a local CPU cannot fit the scaling ladder.
Each notebook is self-contained: it clones the repository, installs the
dependencies, and drives the same scripts as the local
[`scripts/`](../scripts/README.md) pipeline.

| Notebook | Content |
|---|---|
| `RMT_TinyGPT_Free_Server.ipynb` | train + evaluate the formula model on a free server, with RMT diagnostics |
| `Scaling_Ladder_Hallucinations.ipynb` | the parameter-scaling A/B ladder measuring hallucination vs model size |

Open in Colab: `File → Upload notebook`, or push the repository to your
own GitHub and use the Colab GitHub launcher.

---

<p align="center">
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scaling/README.md">scaling study →</a>
  </sub>
</p>
