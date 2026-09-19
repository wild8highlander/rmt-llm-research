# Research

Standalone research programs built on top of the core package. Each folder
is self-contained: it does not modify the `src/rmt_llm/` core and ships its
own scripts, data, and model artifacts.

| Folder | Content |
|---|---|
| [`tinygpt_formula/`](tinygpt_formula/) | **TinyGPT v3 Formula-Edition** — train the repository's NumPy-only transformer on the project's own formulas, evaluate held-out generalization, run the RMT diagnostics, and answer the ROADMAP open questions empirically. Includes the parameter-scaling A/B study (`formula` vs `control` corpus) that measures how true formula content affects confident hallucinations. |

Start with [`tinygpt_formula/README.md`](tinygpt_formula/README.md) — the
full pipeline (corpus → training → evaluation → open questions → scaling)
reproduces in ~20 minutes on CPU.
