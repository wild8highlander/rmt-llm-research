# RMT-LLM Research

> **Spectral analysis of LLM activations through the lens of Random Matrix Theory**
> — detecting hallucinations, cognitive mode transitions, and the mathematical
> inevitability of autoregressive collapse.

Welcome to the documentation for `rmt-llm-research`, a comprehensive research
program applying Random Matrix Theory (RMT) to the analysis of large language
models (LLMs). This project combines rigorous mathematics, reproducible
numerical experiments, and a pure-NumPy synthetic transformer (`TinyGPT`)
designed for pedagogical transparency.

---

## What's in this project?

| Layer | What it is | Where |
|-------|------------|-------|
| **Theoretical core** | Pure-Python implementations of every RMT formula (Marchenko-Pastur, BBP, Tracy-Widom, NHSE, Caputo, Keating-Snaith, EP-surfaces, thermodynamics) | [`src/rmt_llm/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/src/rmt_llm) |
| **TinyGPT** | 2.5M-param pure-NumPy transformer with hand-written backprop + BPE tokenizer + Adam optimizer | [`laboratory/python/lab_en/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/laboratory/python/lab_en) |
| **8-language laboratory** | Same interactive menu implemented in Python, Julia, Java, Rust, Go, C++, R, and a React web app — all emitting one shared JSON schema | [`laboratory/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/laboratory) |
| **Visualization suites** | 3 full 3D visualization tools (Python CLI, Julia REPL, Java JavaFX) producing 8 identical visualizations each | [`python/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/python), [`julia/RMTLLMViz/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/julia/RMTLLMViz), [`java/rmt-llm-viz/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/java/rmt-llm-viz) |
| **Documents** | 6 English + 6 Russian monographs (DOCX) + 1 PDF preprint | [`docs/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/docs), [`papers/`](https://github.com/wild8highlander/rmt-llm-research/tree/main/papers) |

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/wild8highlander/rmt-llm-research.git
cd rmt-llm-research

# 2. Install (dev mode — includes test + lint deps)
pip install -e ".[dev]"

# 3. Run tests
pytest -v                              # 240+ tests across Python
make test-all                          # all 8 languages (slow)

# 4. Train TinyGPT (30 epochs, ~17 min on CPU)
cd laboratory/python/lab_en
python main.py                         # then choose menu item 14

# 5. Generate text from the trained model
python main.py                         # then choose menu item 15
```

See [Getting Started → Installation](getting-started/installation.md) for
detailed instructions including Julia, Java, and Docker.

---

## Why this project exists

Modern LLMs hallucinate, and the question "*why?*" has a precise mathematical
answer. This repository collects ten independent derivations of the same
critical token count $N_{\text{crit}}$ — from the Marchenko-Pastur law, the BBP
phase transition, the Tracy-Widom distribution, the non-Hermitian skin effect,
Caputo fractional dynamics, exceptional-point surfaces, and Keating-Snaith
finite-$N$ corrections.

The same mathematics also explains *why* RLHF amplifies the problem
(the "utility trap") and why autoregressive generation is thermodynamically
irreversible (Landauer principle applied to the spectral entropy $S_{\text{spec}}$).

---

## Where to go next

| If you want to… | Read this |
|------------------|-----------|
| Understand the math | [`docs/en/RMT_LLM_Arxiv_Preprint.docx`](https://github.com/wild8highlander/rmt-llm-research/raw/main/docs/en/RMT_LLM_Arxiv_Preprint.docx) |
| Run the verification tests | [Getting Started → Quick Start](getting-started/quick-start.md) |
| Train TinyGPT from scratch | [Tutorial: Training TinyGPT](tutorials/02-training-tinygpt.md) |
| Read the API | [API Reference](api/index.md) |
| Understand the architecture | [Architecture Overview](architecture/index.md) |
| Contribute | [Contributing Guide](community/contributing.md) |
| See what's planned | [Roadmap](community/roadmap.md) |

---

## Citation

If this work informs your research, please cite:

```bibtex
@article{rmt-llm-research-2026,
  title   = {Spectral Statistics of LLM Activation Matrices:
             A Random Matrix Theory Approach to Cognitive Mode Detection},
  author  = {Isaev, Iskhak Hamzatovich},
  year    = {2026},
  journal = {Preprint},
  url     = {https://github.com/wild8highlander/rmt-llm-research}
}
```

See [`CITATION.cff`](https://github.com/wild8highlander/rmt-llm-research/blob/main/CITATION.cff)
for the machine-readable version.

---

## License

Proprietary — all rights reserved by Iskhak Hamzatovich Isaev.
See [`LICENSE`](https://github.com/wild8highlander/rmt-llm-research/blob/main/LICENSE)
for full terms.
