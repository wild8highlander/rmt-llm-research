<p align="center">
  <a href="../../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="72" alt="RMT × LLM Research"></a>
</p>

# 🧪 Python Laboratory (RU)

<p align="center">
  <em>Русская версия лаборатории RMT-LLM на Python — интерактивное меню, 3D-исследования, экспорт графиков и отчётов</em><br><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt=""><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-laboratory-1C2340" alt="laboratory/"></a>
  <img src="https://img.shields.io/badge/language-python-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/cross--language-twin-22d3ee" alt="status">
</p>

Лаборатория `Python` — автономный порт исследовательской лаборатории
RMT-LLM: те же границы Марченко–Пастура, ландшафты BBP-перехода, коллапс
скин-эффекта NHSE и сцены свободной энергии по слоям, что и в Python-
оригинале, но в нативной экосистеме `Python`. Это один из восьми
межъязыковых близнецов: эксперимент везде идентичен, отличается только
рантайм. Численные результаты сверяются с Python-ядром (`src/rmt_llm/`)
и Julia-верификатором, поэтому артефакты сопоставимы между реализациями.

Типичная сессия: выберите сценарий в интерактивном меню, запустите
спектральный проход и экспортируйте наборы графиков (PNG / SVG / PDF)
и отчёты в нескольких форматах (JSON · CSV · Markdown · HTML · TeX) —
те же артефакты, что живут в `results/`.

## ✅ Тесты

101 модульный тест покрывает модель, BPE-токенизатор, оптимизатор Adam, полный backward pass (с численной проверкой градиентов) и end-to-end обучение — запуск: `pytest tests/`.

> The `ru` trainer is kept in sync with the EN original by `scripts/sync_ru_trainer.py` — run it after editing the EN trainer.

## 🚀 Quick start

```bash
cd laboratory/python/lab_ru
python main.py    # интерактивное меню — пункт 14: Research 3D
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
[3D visualization suites](../../python/python/rmt_llm_viz/README.md).

---

<p align="center">
  <sub>
    <a href="../README.md">← laboratory root</a> · <a href="../README.md">↑ parent</a> · <a href="../../../README.md">🏠 repository root</a>
  </sub>
</p>
