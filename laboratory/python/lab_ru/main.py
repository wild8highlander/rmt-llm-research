#!/usr/bin/env python3
"""
main.py — Лаборатория RMT-LLM (русская версия)
================================================

Интерактивное меню исследовательской лаборатории проекта RMT-LLM.

Главное меню:
  1. Запустить предустановленный сценарий (синтетическая ИИ-сеть)
  2. Запустить исследовательский эксперимент (система измерений)
  3. Кастомный запуск — интерактивный мастер (бесконечные параметры)
  4. Кастомный запуск — JSON-конфиг
  5. Скачать модель из реестра (HuggingFace / ONNX)
  6. Сгенерировать только отчёты (из существующего results.json)
  7. Сгенерировать только графики (из существующего results.json)
  8. Запустить ВСЕ сценарии + эксперименты → полный отчёт
  9. Показать пространство параметров
 10. Кросс-имплементационная верификация
  0. Выход

Каждый запуск создаёт:
  - results/charts/ : PNG (600 DPI) + PDF + SVG + интерактивный Plotly HTML
  - results/reports/: 13 форматов (txt, md, csv, html, json, pdf, docx,
                       yaml, xml, latex, parquet, xlsx, sqlite)
  - results/logs/   : полные логи запуска задачи

Автор: Исхак Хамзатович Исаев
Лицензия: Проприетарная — Все права защищены.
"""

from __future__ import annotations

import json
import os
import sys
import time
import datetime
from typing import Any, Dict, List, Optional

# Добавляем родительский каталог в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
import charts
import charts_3d
import reports
import research
import research_3d
import scenarios as scen
import model_downloader as md
import tiny_gpt_trainer as trainer
from tiny_gpt import TinyGPT, TinyGPTConfig, encode, decode


# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(LAB_ROOT, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
CHARTS_3D_DIR = os.path.join(RESULTS_DIR, "charts_3d")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for d in (RESULTS_DIR, CHARTS_DIR, CHARTS_3D_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Баннер и меню
# ---------------------------------------------------------------------------
BANNER = r"""
==============================================================================
   ЛАБОРАТОРИЯ RMT-LLM  v1.0.0  (Русская версия)
   Теория случайных матриц встречается с большими языковыми моделями
   -----------------------------------------------------------------------------
   Исследовательская лаборатория для проверки новости:
   «Claude, Gemini, ChatGPT взломаны — раскрыты скрытые цепочки рассуждений.
    Модели лгут, галлюцинируют и хранят PII.»
   -----------------------------------------------------------------------------
   Автор  : Исхак Хамзатович Исаев
   ORCID  : 0009-0003-7299-0701
   Лицензия: Проприетарная — Все права защищены.
==============================================================================
"""

MENU = """
---------------------------- ГЛАВНОЕ МЕНЮ ----------------------------
  1. Запустить предустановленный сценарий (синтетическая ИИ-сеть)
  2. Запустить исследовательский эксперимент (система измерений)
  3. Кастомный запуск — интерактивный мастер (бесконечные параметры)
  4. Кастомный запуск — JSON-конфиг
  5. Скачать модель из реестра (HuggingFace / ONNX)
  6. Сгенерировать только отчёты (из существующего results.json)
  7. Сгенерировать только графики (из существующего results.json)
  8. Запустить ВСЕ сценарии + эксперименты → полный отчёт
  9. Показать пространство параметров
 10. Кросс-имплементационная верификация
 11. Запустить 3D-исследовательский эксперимент (v1.1.0)
 12. Запустить ВСЕ 3D-эксперименты → 3D-графики + отчёты
 13. Сгенерировать только 3D-графики (из существующего results.json)
 14. Обучить TinyGPT на реальном корпусе (v1.2.0) → улучшает match_rate
 15. Сгенерировать текст из обученного TinyGPT
  0. Выход
----------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.path: Optional[str] = None

    def log(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.lines.append(line)
        print(line)

    def save(self, dir_path: str, name: str = "run.log") -> str:
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, name)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        return self.path


# ---------------------------------------------------------------------------
# Действия меню
# ---------------------------------------------------------------------------
def action_run_scenario(logger: Logger) -> None:
    """Пункт 1: выбрать сценарий, запустить, сгенерировать отчёты + графики."""
    scens = scen.list_scenarios_for_menu()
    print("\n--- Доступные сценарии ---")
    for i, s in enumerate(scens, 1):
        name = s.get("name_ru", s.get("name", s["id"]))
        desc = s.get("description_ru", s.get("description", ""))
        print(f"  {i:2d}. [{s['id']}] {name}")
        print(f"       Ожидаемое поведение: {s.get('expected_behavior', 'неизвестно')}")
        print(f"       {desc[:80]}...")
    choice = input("\nНомер сценария: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(scens)):
        print("Неверный выбор.")
        return
    chosen = scens[int(choice) - 1]
    logger.log(f"Пользователь выбрал сценарий {chosen['id']}")
    name = chosen.get("name_ru", chosen.get("name", chosen["id"]))

    # Получаем базовые параметры (используем значения по умолчанию)
    use_wizard = input("Использовать интерактивный мастер параметров? (y/N): ").strip().lower() in ("y", "yes", "д", "да")
    if use_wizard:
        params = P.interactive_wizard()
    else:
        params = {p.name: p.default for p in P.default_parameter_space()}
    logger.log(f"Параметры: {json.dumps({k: v for k, v in params.items()}, default=str)[:200]}")

    # Запуск
    results = scen.run_scenario(chosen, params, logs=logger.lines)
    logger.log(f"Сценарий завершён. Доля совпадений: {results['metrics']['match_rate']:.2%}")

    # Генерация графиков + отчётов
    _emit_outputs(results, logger, suffix=chosen["id"])


def action_run_experiment(logger: Logger) -> None:
    """Пункт 2: выбрать исследовательский эксперимент, запустить, сгенерировать вывод."""
    print("\n--- Доступные исследовательские эксперименты ---")
    for k, exp in research.EXPERIMENTS.items():
        print(f"  {k}. {exp.name}")
        print(f"     {exp.description}")
    choice = input("\nНомер эксперимента: ").strip()
    if choice not in research.EXPERIMENTS:
        print("Неверный выбор.")
        return
    logger.log(f"Пользователь выбрал эксперимент {choice}: {research.EXPERIMENTS[choice].name}")

    params = {p.name: p.default for p in P.default_parameter_space()}
    tweak = input("Подправить параметры? (y/N): ").strip().lower()
    if tweak in ("y", "yes", "д", "да"):
        params = P.interactive_wizard()

    results = research.run_experiment(choice, params)
    logger.log(f"Эксперимент завершён за {results['elapsed_seconds']:.3f}с")
    _emit_outputs(results, logger, suffix=f"exp_{choice}")


def action_custom_wizard(logger: Logger) -> None:
    """Пункт 3: полностью кастомный запуск через интерактивный мастер."""
    logger.log("Запуск кастомного режима (интерактивный мастер)")
    params = P.interactive_wizard()
    logger.log(f"Кастомные параметры: {json.dumps({k: v for k, v in params.items()}, default=str)}")

    # Выбор, что запустить
    print("\nЧто выполнить с этими параметрами?")
    print("  1. Запустить сценарий")
    print("  2. Запустить исследовательский эксперимент")
    print("  3. Только посчитать спектральную сигнатуру TinyGPT")
    sub = input("Выбор: ").strip()
    if sub == "1":
        scens = scen.list_scenarios_for_menu()
        for i, s in enumerate(scens, 1):
            print(f"  {i}. {s.get('name_ru', s.get('name', s['id']))}")
        idx = input("Номер сценария: ").strip()
        if idx.isdigit() and 1 <= int(idx) <= len(scens):
            results = scen.run_scenario(scens[int(idx) - 1], params, logs=logger.lines)
            _emit_outputs(results, logger, suffix=f"custom_scen_{scens[int(idx)-1]['id']}")
    elif sub == "2":
        for k, exp in research.EXPERIMENTS.items():
            print(f"  {k}. {exp.name}")
        k = input("Номер эксперимента: ").strip()
        if k in research.EXPERIMENTS:
            results = research.run_experiment(k, params)
            _emit_outputs(results, logger, suffix=f"custom_exp_{k}")
    elif sub == "3":
        results = research.run_experiment("1", params)
        _emit_outputs(results, logger, suffix="custom_spectral")
    else:
        print("Неверный выбор.")


def action_custom_config(logger: Logger) -> None:
    """Пункт 4: кастомный запуск из JSON-конфига."""
    path = input("Путь к JSON-конфигу: ").strip()
    if not os.path.exists(path):
        print(f"Файл не найден: {path}")
        return
    cfg = P.load_config(path)
    logger.log(f"Загружен конфиг из {path}")
    params = cfg.get("parameters", cfg)
    # Валидация
    space = {p.name: p for p in P.default_parameter_space()}
    validated: Dict[str, Any] = {}
    for k, v in params.items():
        if k in space:
            try:
                validated[k] = space[k].validate(v)
            except ValueError as exc:
                logger.log(f"  [ПРЕДУПРЕЖДЕНИЕ] {exc}, используется значение по умолчанию")
                validated[k] = space[k].default
        else:
            validated[k] = v  # принимаем неизвестные параметры
    logger.log(f"Провалидированные параметры: {json.dumps(validated, default=str)[:200]}")
    # Запуск сценария или эксперимента по конфигу
    if "scenario_id" in cfg:
        sc = scen.get_scenario(cfg["scenario_id"])
        if sc:
            results = scen.run_scenario(sc, validated, logs=logger.lines)
            _emit_outputs(results, logger, suffix=cfg["scenario_id"])
    elif "experiment_id" in cfg and cfg["experiment_id"] in research.EXPERIMENTS:
        results = research.run_experiment(cfg["experiment_id"], validated)
        _emit_outputs(results, logger, suffix=f"cfg_exp_{cfg['experiment_id']}")
    else:
        # По умолчанию: спектральная сигнатура
        results = research.run_experiment("1", validated)
        _emit_outputs(results, logger, suffix="cfg_spectral")


def action_download_model(logger: Logger) -> None:
    """Пункт 5: скачать модель из реестра."""
    pick = md.interactive_pick()
    logger.log(f"Пользователь выбрал модель: {pick}")
    if pick == "tiny-gpt-local":
        logger.log("Выбрана локальная модель — загрузка не требуется.")
        # Генерируем и сохраняем локальные веса TinyGPT
        model = TinyGPT()
        path = os.path.join(MODELS_DIR, "tiny_gpt_local.npz")
        model.save_weights(path)
        logger.log(f"Локальные веса TinyGPT сохранены в {path}")
        print(f"\nЛокальные веса TinyGPT сохранены в:\n  {path}")
        return
    rep = md.fetch_model(pick, dest_dir=MODELS_DIR)
    if rep.get("ok"):
        logger.log(f"Загружено {pick}: {rep.get('bytes', 0)} байт")
        print(f"\nЗагружено в: {rep.get('dest')}")
        print(f"  Размер: {rep.get('bytes', 0):,} байт")
        print(f"  SHA256: {rep.get('sha256')}")
    else:
        logger.log(f"Загрузка не удалась: {rep.get('error')}")
        print(f"\nЗагрузка не удалась: {rep.get('error')}")


def action_reports_only(logger: Logger) -> None:
    """Пункт 6: перегенерировать отчёты из существующего results.json."""
    path = input("Путь к results.json (или Enter для последнего): ").strip()
    if not path:
        # Ищем последний в каталоге отчётов
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("results.json не найден.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Не найдено: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Загружены результаты из {path}")
    suffix = os.path.basename(path).replace("_results.json", "")
    _emit_reports_only(results, logger, suffix)


def action_charts_only(logger: Logger) -> None:
    """Пункт 7: перегенерировать графики из существующего results.json."""
    path = input("Путь к results.json (или Enter для последнего): ").strip()
    if not path:
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("results.json не найден.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Не найдено: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Загружены результаты из {path}")
    written = charts.generate_all_charts(results, CHARTS_DIR)
    logger.log(f"Графики сгенерированы: {sum(len(v) for v in written.values())} файлов")
    print(f"\nГрафики записаны в: {CHARTS_DIR}")
    for fmt, paths in written.items():
        print(f"  {fmt}: {len(paths)} файлов")


def action_run_all(logger: Logger) -> None:
    """Пункт 8: запустить всё и сформировать исчерпывающий отчёт."""
    logger.log("=== ЗАПУСК ВСЕХ СЦЕНАРИЕВ + ЭКСПЕРИМЕНТОВ ===")
    params = {p.name: p.default for p in P.default_parameter_space()}

    all_results: List[Dict[str, Any]] = []

    # Все сценарии
    scens = scen.list_scenarios_for_menu()
    for sc in scens:
        logger.log(f"\n--- Сценарий {sc['id']} ---")
        try:
            r = scen.run_scenario(sc, params, logs=logger.lines)
            all_results.append(r)
            _emit_outputs(r, logger, suffix=sc["id"], suppress_charts=False)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [ОШИБКА] сценарий {sc['id']} упал: {exc}")

    # Все эксперименты
    for eid in research.EXPERIMENTS:
        logger.log(f"\n--- Эксперимент {eid} ---")
        try:
            r = research.run_experiment(eid, params)
            all_results.append(r)
            _emit_outputs(r, logger, suffix=f"exp_{eid}", suppress_charts=False)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [ОШИБКА] эксперимент {eid} упал: {exc}")

    # Итоговая сводка
    summary = {
        "experiment_name": "FULL_LAB_RUN",
        "language": "python",
        "version": "ru",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_scenarios": len(scens),
        "n_experiments": len(research.EXPERIMENTS),
        "n_total": len(all_results),
        "sub_results": all_results,
        "metrics": {
            "total_runs": len(all_results),
            "scenario_match_rate": float(np.mean([
                r.get("metrics", {}).get("match_rate", 0)
                for r in all_results if "scenario_id" in r
            ])) if any("scenario_id" in r for r in all_results) else 0,
        },
    }
    _emit_outputs(summary, logger, suffix="FULL_LAB_RUN")


def action_show_parameters(logger: Logger) -> None:
    """Пункт 9: показать пространство параметров."""
    space = P.default_parameter_space()
    print(f"\n=== ПРОСТРАНСТВО ПАРАМЕТРОВ ({len(space)} параметров, все поддерживают inf) ===")
    for p in space:
        lo = "0" if p.min == 0 else str(p.min)
        hi = "inf" if p.max == float("inf") else str(p.max)
        print(f"  {p.name:20s} тип={p.type:12s} диапазон=[{lo}, {hi}]  по_умолчанию={p.default}")
        if p.description:
            print(f"  {'':20s} {p.description}")
    print(f"\nВсе числовые параметры принимают 'inf' для неограниченных значений.")


def action_cross_verify(logger: Logger) -> None:
    """Пункт 10: кросс-имплементационная верификация."""
    logger.log("Кросс-имплементационная верификация")
    results = research.run_experiment("5", {})
    logger.log(f"MP верхняя (теория): {results['mp_upper_theory']:.4f}, "
               f"эмпирически: {results['mp_upper_empirical']:.4f}, "
               f"отн. ошибка: {results['metrics']['upper_rel_err']:.4f}")
    _emit_outputs(results, logger, suffix="cross_verify")


def action_run_3d_experiment(logger: Logger) -> None:
    """Пункт 11: выбрать 3D-эксперимент, запустить, сгенерировать 3D-графики + отчёты."""
    print("\n--- Доступные 3D-исследовательские эксперименты (v1.1.0) ---")
    for k, exp in research_3d.EXPERIMENTS_3D.items():
        print(f"  {k:>2s}. {exp.name}")
        print(f"      {exp.description}")
    choice = input("\nНомер 3D-эксперимента: ").strip()
    if choice not in research_3d.EXPERIMENTS_3D:
        print("Неверный выбор.")
        return
    logger.log(f"Пользователь выбрал 3D-эксперимент {choice}: {research_3d.EXPERIMENTS_3D[choice].name}")

    params = {p.name: p.default for p in P.default_parameter_space()}
    # Добавляем 3D-специфичные параметры
    params_3d = {
        "hessian_grid_size": 24,
        "trajectory_points": 64,
        "spectral_surface_layers": 6,
        "pca_components": 3,
        "attention_flow_3d_resolution": 32,
        "parameter_space_grid": 16,
        "curvature_neighbors": 8,
        "color_map_3d": "viridis",
        "elevation_3d": 30,
        "azimuth_3d": 45,
    }
    params.update(params_3d)
    tweak = input("Изменить 3D-параметры? (y/N): ").strip().lower()
    if tweak in ("y", "yes"):
        params.update(P.interactive_wizard())

    results = research_3d.run_3d_experiment(choice, params)
    logger.log(f"3D-эксперимент завершён за {results['elapsed_seconds']:.3f} с")
    _emit_3d_outputs(results, logger, suffix=f"3d_exp_{choice}")


def action_run_all_3d(logger: Logger) -> None:
    """Пункт 12: запустить все 9 3D-экспериментов, сгенерировать 3D-графики + полный отчёт."""
    logger.log("=== ЗАПУСК ВСЕХ 3D-ЭКСПЕРИМЕНТОВ (v1.1.0) ===")
    params = {p.name: p.default for p in P.default_parameter_space()}
    params.update({
        "hessian_grid_size": 24,
        "trajectory_points": 64,
        "spectral_surface_layers": 6,
        "pca_components": 3,
        "attention_flow_3d_resolution": 32,
        "parameter_space_grid": 16,
        "curvature_neighbors": 8,
        "color_map_3d": "viridis",
        "elevation_3d": 30,
        "azimuth_3d": 45,
    })

    combined = research_3d.run_all_3d_experiments(params)
    n_ok = sum(1 for e in combined["experiments"] if "error" not in e)
    n_err = sum(1 for e in combined["experiments"] if "error" in e)
    logger.log(f"3D-эксперименты завершены: {n_ok} успешно, {n_err} с ошибкой")

    combined["experiment_name"] = "ALL_3D_EXPERIMENTS"
    combined["language"] = "python"
    combined["version"] = "ru"
    combined["timestamp"] = datetime.datetime.now().isoformat()
    combined["ncrit_threshold"] = float(params.get("ncrit_threshold", 114.0))
    combined["spectral"] = {"mp_upper": 2.7}

    _emit_3d_outputs(combined, logger, suffix="ALL_3D")


def action_3d_charts_only(logger: Logger) -> None:
    """Пункт 13: перегенерировать 3D-графики из существующего results.json."""
    path = input("Путь к results.json (или Enter для последнего): ").strip()
    if not path:
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("results.json не найден.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Не найдено: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Загружены результаты из {path}")
    written = charts_3d.generate_all_3d_charts(results, CHARTS_3D_DIR)
    logger.log(f"3D-графики сгенерированы: {sum(len(v) for v in written.values())} файлов")
    print(f"\n3D-графики записаны в: {CHARTS_3D_DIR}")
    for fmt, paths in written.items():
        print(f"  {fmt}: {len(paths)} файлов")


# ---------------------------------------------------------------------------
# v1.2.0 — Обучение TinyGPT на реальном корпусе
# ---------------------------------------------------------------------------
def action_train_tiny_gpt(logger: Logger) -> None:
    """Пункт 14: обучить TinyGPT на реальном текстовом корпусе проекта.

    Реализует полный цикл backprop-обучения трансформера на задаче
    побайтового предсказания следующего токена. Корпус собирается из
    README.md, CHANGELOG.md, исходного кода, JSON-конфигов и HTML-документации —
    без внешних загрузок. Обученные веса сохраняются в
    `results/models/tiny_gpt_trained.npz`, а метрика `match_rate`
    (жадная точность предсказания следующего байта на отложенной выборке)
    возрастает с 0% (необученный baseline) до ненулевого значения.
    """
    print("\n--- Обучение TinyGPT на реальном корпусе (v1.2.0) ---")
    print("Обучает синтетический TinyGPT на собственном текстовом корпусе проекта")
    print("(README, CHANGELOG, исходный код, JSON-конфиги). Полная цепочка backprop")
    print("проходит через softmax-CE → lm_head → residual → attention.")
    print()

    epochs_in = input("Число эпох [по умолч. 10, поддерживает 'inf']: ").strip() or "10"
    lr_in = input("Скорость обучения [по умолч. 3e-4]: ").strip() or "3e-4"
    batch_in = input("Размер батча [по умолч. 16]: ").strip() or "16"
    stride_in = input("Шаг [по умолч. 16, больше=быстрее]: ").strip() or "16"

    cfg = trainer.TrainConfig(
        epochs=epochs_in,
        lr=float(lr_in),
        batch_size=int(batch_in),
        stride=int(stride_in),
    )
    logger.log(f"Обучение TinyGPT: epochs={epochs_in} lr={lr_in} "
               f"batch={batch_in} stride={stride_in}")

    def progress(rec):
        msg = (f"  эпоха {rec['epoch']:>3d}  loss={rec['loss']:.4f}  "
               f"grad_norm={rec['grad_norm']:.4f}")
        if "match_rate" in rec:
            msg += f"  match_rate={rec['match_rate']:.3%}"
        print(msg)

    print("\nСборка корпуса из файлов проекта...")
    result = trainer.train_tiny_gpt(LAB_ROOT, cfg, progress_cb=progress)

    print(f"\n=== Обучение завершено ({result['elapsed_seconds']:.1f}с) ===")
    print(f"  Размер корпуса      : {result['corpus_bytes']:,} байт")
    print(f"  Обучающих окон      : {result['n_train_windows']:,}")
    print(f"  Оценочных окон      : {result['n_eval_windows']:,}")
    print(f"  Baseline match_rate : {result['baseline_match_rate']:.3%} (необученная)")
    print(f"  Финальный match_rate: {result['final_match_rate']:.3%}")
    print(f"  Baseline loss       : {result['baseline_loss']:.4f}")
    print(f"  Финальный loss      : {result['final_loss']:.4f}")
    print(f"  Веса сохранены      : {result['weights_path']}")

    logger.log(f"Обучение завершено: match_rate {result['baseline_match_rate']:.3%} -> "
               f"{result['final_match_rate']:.3%}, "
               f"loss {result['baseline_loss']:.4f} -> {result['final_loss']:.4f}")


def action_generate_from_trained(logger: Logger) -> None:
    """Пункт 15: загрузить обученный TinyGPT и сгенерировать текст."""
    weights_path = os.path.join(MODELS_DIR, "tiny_gpt_trained.npz")
    bpe_path = os.path.join(MODELS_DIR, "tiny_gpt_bpe.json")
    if not os.path.exists(weights_path):
        print(f"Обученные веса не найдены: {weights_path}")
        print("Сначала запустите пункт 14.")
        return

    print("\n--- Генерация текста из обученного TinyGPT v2 ---")
    prompt = input("Промпт [по умолч. 'RMT-LLM ']: ").strip() or "RMT-LLM "
    temp = float(input("Температура [по умолч. 0.7]: ").strip() or "0.7")
    ntok = int(input("Макс. новых токенов [по умолч. 64]: ").strip() or "64")

    model, tok = trainer.load_trained_model(
        weights_path, bpe_path if os.path.exists(bpe_path) else None)
    logger.log(f"Загружены обученные веса из {weights_path}"
               + (f" + BPE из {bpe_path}" if tok else " (без BPE, байтовый режим)"))
    out = trainer.generate_sample(model, prompt, tok,
                                   max_new_tokens=ntok,
                                   temperature=temp, seed=42)
    print(f"\nПромпт : {prompt!r}")
    print(f"Вывод  : {out!r}")
    logger.log(f"Сгенерировано {len(out)} символов из промпта {prompt!r}")


# ---------------------------------------------------------------------------
# Эмиссия выводов
# ---------------------------------------------------------------------------
def _emit_outputs(results: Dict[str, Any], logger: Logger, suffix: str,
                  suppress_charts: bool = False) -> None:
    """Сохранить results.json, сгенерировать графики + отчёты, сохранить лог."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{suffix}"

    # 1. Сохраняем results.json
    res_path = os.path.join(REPORTS_DIR, f"{name}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.log(f"Результаты сохранены в {res_path}")

    # 2. Генерируем графики
    if not suppress_charts:
        charts_dir_run = os.path.join(CHARTS_DIR, name)
        written_charts = charts.generate_all_charts(results, charts_dir_run)
        n_charts = sum(len(v) for v in written_charts.values())
        logger.log(f"Графики: {n_charts} файлов в {charts_dir_run}")

    # 3. Генерируем отчёты в 13 форматах
    written_reports = reports.generate_all_reports(results, logger.lines,
                                                   out_dir=REPORTS_DIR,
                                                   experiment_name=name)
    logger.log(f"Отчёты: {len(written_reports)} форматов в {REPORTS_DIR}")

    # 4. Сохраняем лог
    log_path = logger.save(LOGS_DIR, f"{name}.log")
    logger.log(f"Лог сохранён в {log_path}")

    print(f"\n--- Вывод записан ---")
    print(f"  JSON результатов : {res_path}")
    if not suppress_charts:
        print(f"  Графики          : {charts_dir_run}")
    print(f"  Отчёты (13)      : {REPORTS_DIR}/{name}.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    print(f"  Логи             : {log_path}")


def _emit_reports_only(results: Dict[str, Any], logger: Logger, suffix: str) -> None:
    written = reports.generate_all_reports(results, logger.lines,
                                           out_dir=REPORTS_DIR,
                                           experiment_name=suffix)
    logger.log(f"Отчёты перегенерированы: {len(written)} форматов")
    print(f"\nОтчёты записаны в {REPORTS_DIR}:")
    for fmt, p in written.items():
        print(f"  {fmt:8s} -> {p}")


def _emit_3d_outputs(results: Dict[str, Any], logger: Logger, suffix: str) -> None:
    """Сохранить results.json, сгенерировать 3D-графики + стандартные отчёты, сохранить лог."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{suffix}"

    # 1. Сохраняем results.json
    res_path = os.path.join(REPORTS_DIR, f"{name}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.log(f"3D-результаты сохранены в {res_path}")

    # 2. Генерируем 3D-графики
    charts_3d_dir_run = os.path.join(CHARTS_3D_DIR, name)
    params_3d = {
        "hessian_grid_size": results.get("parameters", {}).get("hessian_grid_size", 24),
        "trajectory_points": results.get("parameters", {}).get("trajectory_points", 64),
        "spectral_surface_layers": results.get("parameters", {}).get("spectral_surface_layers", 6),
        "pca_components": results.get("parameters", {}).get("pca_components", 3),
        "attention_flow_3d_resolution": results.get("parameters", {}).get("attention_flow_3d_resolution", 32),
        "parameter_space_grid": results.get("parameters", {}).get("parameter_space_grid", 16),
        "color_map_3d": results.get("parameters", {}).get("color_map_3d", "viridis"),
        "elevation_3d": results.get("parameters", {}).get("elevation_3d", 30),
        "azimuth_3d": results.get("parameters", {}).get("azimuth_3d", 45),
    }
    written_3d = charts_3d.generate_all_3d_charts(results, charts_3d_dir_run,
                                                     dpi=600, params_3d=params_3d)
    n_3d = sum(len(v) for v in written_3d.values())
    logger.log(f"3D-графики: {n_3d} файлов в {charts_3d_dir_run}")

    # 3. Генерируем стандартные отчёты (13 форматов)
    written_reports = reports.generate_all_reports(results, logger.lines,
                                                   out_dir=REPORTS_DIR,
                                                   experiment_name=name)
    logger.log(f"Отчёты: {len(written_reports)} форматов в {REPORTS_DIR}")

    # 4. Сохраняем лог
    log_path = logger.save(LOGS_DIR, f"{name}.log")
    logger.log(f"Лог сохранён в {log_path}")

    print(f"\n--- 3D-вывод записан ---")
    print(f"  JSON результатов   : {res_path}")
    print(f"  3D-графики         : {charts_3d_dir_run}")
    print(f"    PNG (600 DPI)    : {len(written_3d['png'])} файлов")
    print(f"    PDF (векторные)  : {len(written_3d['pdf'])} файлов")
    print(f"    SVG (векторные)  : {len(written_3d['svg'])} файлов")
    print(f"    Plotly HTML      : {len(written_3d['html'])} файлов (интерактивный 3D)")
    print(f"  Отчёты (13)        : {REPORTS_DIR}/{name}.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    print(f"  Логи               : {log_path}")


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------
def main() -> None:
    print(BANNER)
    logger = Logger()
    logger.log("Лаборатория RMT-LLM запущена (русская версия)")

    while True:
        print(MENU)
        choice = input("Выбор [0-15]: ").strip()
        try:
            if choice == "0":
                logger.log("Пользователь вышел.")
                logger.save(LOGS_DIR, "session.log")
                print("\nДо свидания.")
                break
            elif choice == "1":
                action_run_scenario(logger)
            elif choice == "2":
                action_run_experiment(logger)
            elif choice == "3":
                action_custom_wizard(logger)
            elif choice == "4":
                action_custom_config(logger)
            elif choice == "5":
                action_download_model(logger)
            elif choice == "6":
                action_reports_only(logger)
            elif choice == "7":
                action_charts_only(logger)
            elif choice == "8":
                action_run_all(logger)
            elif choice == "9":
                action_show_parameters(logger)
            elif choice == "10":
                action_cross_verify(logger)
            elif choice == "11":
                action_run_3d_experiment(logger)
            elif choice == "12":
                action_run_all_3d(logger)
            elif choice == "13":
                action_3d_charts_only(logger)
            elif choice == "14":
                action_train_tiny_gpt(logger)
            elif choice == "15":
                action_generate_from_trained(logger)
            else:
                print("Неверный выбор.")
        except KeyboardInterrupt:
            print("\nПрервано.")
            continue
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[ОШИБКА] {type(exc).__name__}: {exc}")
            print(f"\n[ОШИБКА] {exc}")


if __name__ == "__main__":
    main()
