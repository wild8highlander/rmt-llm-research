# Jupyter Notebooks

Hands-on companions to the documentation tutorials. Each notebook is
executable top-to-bottom on a CPU-only machine (Python 3.10+, NumPy,
Matplotlib; see `pyproject.toml` extras `notebooks`).

| Notebook | Content |
|---|---|
| `01_quickstart.ipynb` | install, verify, and run the first RMT analysis |
| `02_training_tinygpt.ipynb` | train TinyGPT from scratch with the BPE tokenizer |
| `03_generation_and_sampling.ipynb` | temperature / top-k / top-p sampling playground |
| `04_multilingual_labs.ipynb` | run the same lab from the 8 language implementations |
| `05_tinygpt_v3_modernizations.ipynb` | TinyGPT v3 upgrades: RoPE, GQA, pre-LN, weight tying |
| `rmt_llm_verification.ipynb` | full verification walkthrough of the core package |

Launch:

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

Rendered equivalents of the first four tutorials live in `docs/tutorials/`.
