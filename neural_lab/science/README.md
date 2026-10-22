<p align="center">
  <a href="../../README.md"><img src="../../../assets/branding/aurora-icon.svg" width="64" alt="RMT × LLM Research"></a><br>
  <img src="../../../assets/branding/aurora-divider.svg" width="420" alt="">
</p>

# 🧪 science — Predictive Research Core

<p align="center">
  <em>hypothesis → measurement → prediction → validation → falsification</em><br><br>
  <a href="../../README.md"><img src="https://img.shields.io/badge/folder-science-a78bfa" alt="science"></a>
  <img src="https://img.shields.io/badge/numpy--only-00ffa3" alt="NumPy only">
  <img src="https://img.shields.io/badge/tests-52-22d3ee" alt="52 tests">
</p>

Научное ядро `neural_lab`: не каталог эффектных экспериментов, а
воспроизводимый конвейер, который формулирует проверяемые гипотезы,
извлекает признаки из реальной модели, предсказывает результаты на
новых данных и измеряет, насколько его предсказания лучше случайного
угадывания и простых baseline'ов.

## Три разделённых слоя

| Слой | Модуль | Ответственность |
|---|---|---|
| Instrumentation | [`instrumentation/`](instrumentation/) | что реально наблюдается в модели (логиты, скрытые состояния; sha256 весов) |
| Features | [`features/`](features/) | что математически вычисляется из наблюдений (спектр/MP, энтропии, динамика, геометрия) |
| Prediction | [`predictors/`](predictors/) + [`evaluation/`](evaluation/) | какой исход прогнозируется и насколько это лучше baseline |

Сквозные контуры: [`data/`](data/) — задачи с независимой верификацией
(истина пересчитывается библиотекой `src/rmt_llm`) и разбиения без
утечек; [`storage/`](storage/) — паспорта запусков с provenance;
[`experiments/`](experiments/) — CLI-прогоны; [`results/`](results/) —
зафиксированные результаты; [`hypotheses/`](hypotheses/registry.yaml) —
реестр гипотез с критериями фальсификации.

## Быстрый старт

```bash
# 1. обучить модель, если весов ещё нет (weights gitignored, ~17 мин CPU)
python3 research/tinygpt_formula/scripts/train_formula_model.py

# 2. полный прогон эксперимента H1 (~3 мин CPU после обучения)
RMT_LLM_BASE=research/tinygpt_formula \
  python3 neural_lab/science/experiments/run_h1.py

# смок-прогон (жадная генерация, ~1 мин)
SMOKE=1 python3 neural_lab/science/experiments/run_h1.py --out /tmp/h1_smoke

# 3. тесты ядра
python3 -m pytest neural_lab/science/tests/ -q
```

Результаты прогона: `results/h1_v1/REPORT.md` (генерируется из
`report.json`, не из рук). Паспорт прогона: `results/h1_v1/passport.json`
(sha256 весов/корпуса/датасета, seeds, версии окружения).

## Документы

| Документ | Содержание |
|---|---|
| [`../AUDIT.md`](../AUDIT.md) | фактический аудит: что измеряется, где синтетика, где дубли |
| [`../EXPERIMENT_PROTOCOL.md`](../EXPERIMENT_PROTOCOL.md) | предзарегистрированный протокол H1 (v1.0.1) с журналом изменений |
| [`../PREDICTION_ROADMAP.md`](../PREDICTION_ROADMAP.md) | план 01→06 с критериями готовности |
| [`hypotheses/registry.yaml`](hypotheses/registry.yaml) | H1–H4: формулировки, статусы, критерии, результаты |

## Главный результат прогона H1 (v1.0.1, TinyGPT v3 447K)

- **Первичная цель** (предсказание численной ошибки, 14 позитивов):
  underpowered — ΔAUC(D−B) = +0.206 [+0.084, +0.319], p = 0.097;
  направление положительное, строгий критерий не достигнут.
  Абляция: без спектральной группы комбинированная модель падает до
  уровня уверенности (0.621 → 0.412) — прирост обеспечивает спектр.
- **Вторичная цель** (format_ok — «Ловушка Полезности»): **значимо** —
  внутренние спектральные состояния AUC 0.871 [0.833, 0.905] против
  стандартной уверенности 0.794 [0.740, 0.845]; ΔAUC = +0.075
  [+0.038, +0.115], перестановочный p = 0.0025.

Отрицательный/неубедительный результат — тоже результат: он зафиксирован
в реестре гипотез и определяет следующий шаг (масштабирование на
M/L/XL ячейки для мощности).

---

<p align="center">
  <sub>
    <a href="../README.md">← neural_lab</a> · <a href="../../README.md">🏠 repository root</a> ·
    <a href="../research/tinygpt_formula/README.md">research program →</a>
  </sub>
</p>
