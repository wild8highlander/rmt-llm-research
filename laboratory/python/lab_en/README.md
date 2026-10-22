<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research — Black Aurora"></a>
</p>

# 🧪 Python Laboratory (EN)

<p align="center">
  <em>The Python twin of the RMT-LLM research laboratory — interactive menu, 3D spectral research, chart & report export</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-python-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/version-v1.5.1-00ffa3" alt="status">
</p>

The `Python` laboratory is a self-contained port of the RMT-LLM research
laboratory: the same Marchenko–Pastur bounds, BBP transition landscapes,
NHSE skin-effect collapse and per-layer free-energy scenes as the Python
original, re-expressed natively for the `Python` toolchain. It is one of
eight cross-language twins that keep the experiment identical everywhere —
only the runtime differs. Numerical outputs are cross-checked against the
Python core (`src/rmt_llm/`) and the Julia verifier, so a result produced
here can be compared byte-for-byte with its siblings.

Typical session: pick a scenario from the interactive menu, run the
spectral sweep, then export chart bundles (PNG / SVG / PDF triplets) and
multi-format reports (JSON · CSV · Markdown · HTML · TeX) — the same
artifacts you will find in `results/`.

### ✅ Test suite

101 unit tests cover the model, the BPE tokenizer, the Adam optimizer, the full backward pass (with numerical gradient checking) and end-to-end training — run them with `pytest tests/`.

> The `en` trainer is kept in sync with the EN original by `scripts/sync_ru_trainer.py` — run it after editing the EN trainer.

## 🚀 Quick start

```bash
cd laboratory/python/lab_en
python main.py    # interactive menu — item 14: Research 3D
```

## 📂 Contents

| File | Role |
|---|---|
| `main.py` | menu-driven entry point — training, generation, research, reports |
| `tiny_gpt_v3.py` / `tiny_gpt.py` | the transformer (v3: RoPE, GQA, pre-LN; v2: classic GPT-2 style block) |
| `tiny_gpt_trainer_v3.py` / `tiny_gpt_trainer.py` | BPE training pipeline with checkpoint recovery |
| `bpe_tokenizer_v3.py` | byte-pair encoding tokenizer (256 merges, pure NumPy) |
| `research.py` / `research_3d.py` | RMT spectral experiments + 3D research pipeline |
| `charts.py` / `charts_3d.py` / `reports.py` | chart bundles (PNG/SVG/PDF) and multi-format reports |
| `corpus_builder_v3.py` / `lr_search_v3.py` / `parameters.py` / `scenarios.py` | corpus tooling, learning-rate search, configuration, scenario presets |
| `probe_real_gpt2.py` / `model_downloader.py` | real-GPT-2 activation probes and model download helper |
| `tests/` | 101 unit tests — model, tokenizer, Adam, backward pass, end-to-end training |

Results land in `results/` inside this folder (`charts/`, `charts_3d/`,
`models/`, `reports/`) — mirrored by `laboratory/results/` at the
laboratory root.

## 🌐 Cross-language family

The same lab exists in: [Julia](../../julia/) · [Java](../../java/) · [Rust](../../rust/) · [Go](../../go/) · [C++](../../cpp/) · [R](../../r/) — and in
[TinyGPT-driven notebooks](../../../notebooks/README.md) and the
[3D visualization suites](../../../python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
