<p align="center">
  <img src="../../../assets/logos/colab.banner.svg" width="760" alt="colab — animated logo: cloud with an activity bolt and compute sparkles">
</p>

# ☁️ colab — Notebooks

<p align="center">
  <em>Run the formula experiments on free cloud GPUs</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/folder-research-a78bfa" alt="tinygpt_formula/"></a>
  <img src="https://img.shields.io/badge/colab-F9AB00" alt="Colab"><img src="https://img.shields.io/badge/2%20notebooks-22d3ee" alt="2 notebooks">
</p>

Two Colab-ready notebooks that run the TinyGPT Formula-Edition pipeline
on free compute — useful when a local CPU cannot fit the scaling ladder.
Each notebook is self-contained: it clones the repository, installs the
dependencies, and drives the same scripts as the local
[`scripts/`](../scripts/README.md) pipeline. Because the notebooks call
the *same* scripts with the *same* seeds as a local run, a GPU-trained
cell and a CPU-trained cell of the same rung are directly comparable —
the notebook is a hardware choice, not a methodological fork.

## Notebooks

| Notebook | Content | Typical budget |
|---|---|---|
| `RMT_TinyGPT_Free_Server.ipynb` | train + evaluate the formula model on a free server, with RMT diagnostics | ~30–60 min (S), ~2–4 h (L) |
| `Scaling_Ladder_Hallucinations.ipynb` | the parameter-scaling A/B ladder measuring hallucination vs model size | per-cell: ~30–60 min (M), ~2–4 h (L) |

## Walkthrough

1. Open Colab → `File → Upload notebook`, or push the repository to your
   own GitHub and use the Colab GitHub launcher.
2. Set the runtime to GPU (T4 is enough for every rung up to L).
3. Run the setup cell — it clones the repo and installs NumPy only.
4. Run the training cell; it drives the ladder script with resume
   chunks, so a disconnected session continues where it stopped.
5. Download the per-cell artifacts (`runs/<cell>/`) and drop them into
   the repository's `scaling/runs/` before regenerating reports —
   the reports are always built locally from committed artifacts.

## Practical notes

- Free Colab sessions are time-limited and can disconnect; the resume
  chunks exist exactly for this. Save `runs/` after every chunk, not at
  the end.
- Kaggle is a drop-in alternative if Colab quotas are exhausted; the
  notebooks assume only a standard Python 3 + GPU image.
- XL (~12M params, 150-epoch budget) is the first rung that genuinely
  needs the GPU session — it is currently queued and is the next
  milestone of the ladder.

## Which notebook when

- **Local CPU is fine** (S-rung reproduction, ~20 min) → skip this
  folder entirely and use [`scripts/`](../scripts/README.md).
- **Want M/L cells** (hours of CPU) → `Scaling_Ladder_Hallucinations.ipynb`
  on a free T4; the resume chunks absorb session disconnects.
- **Want a trained model *plus* RMT diagnostics in one session** →
  `RMT_TinyGPT_Free_Server.ipynb`, which appends the spectral probe to
  the standard train + evaluate flow.

Open in Colab: `File → Upload notebook`, or push the repository to your
own GitHub and use the Colab GitHub launcher.

---

<p align="center">
  <img src="../../../assets/logos/colab.avatar.svg" width="56" alt="colab animated avatar"><br>
  <sub>
    <a href="../README.md">← tinygpt_formula</a> · <a href="README.md">top</a> · <a href="../README.md">↑ parent</a> · <a href="../../README.md">🏠 repository root</a> · <a href="../scaling/README.md">scaling study →</a>
  </sub>
</p>
