#!/usr/bin/env python3
"""
run_full_lab.py — Неинтерактивный драйвер, запускающий все сценарии и эксперименты
русской версии лаборатории RMT-LLM на Python, с генерацией всех графиков и отчётов.

Используется для верификации лаборатории после реализации.
"""

from __future__ import annotations

import datetime
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import charts
import parameters as P
import reports
import research
import scenarios as scen
from tiny_gpt import TinyGPT


LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(LAB_ROOT, "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for d in (RESULTS_DIR, CHARTS_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


class Logger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        self.lines.append(line)
        print(line)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def emit(results, logger, suffix):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{suffix}"
    res_path = os.path.join(REPORTS_DIR, f"{name}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.log(f"  Результаты сохранены: {res_path}")

    charts_dir = os.path.join(CHARTS_DIR, name)
    written_charts = charts.generate_all_charts(results, charts_dir)
    n_charts = sum(len(v) for v in written_charts.values())
    logger.log(f"  Графики: {n_charts} файлов в {charts_dir}")

    written_reports = reports.generate_all_reports(
        results, logger.lines, out_dir=REPORTS_DIR, experiment_name=name
    )
    logger.log(f"  Отчёты: {len(written_reports)} форматов")

    log_path = os.path.join(LOGS_DIR, f"{name}.log")
    logger.save(log_path)
    return name


def main():
    print("=" * 70)
    print("Лаборатория RMT-LLM — ПОЛНЫЙ ЗАПУСК (Python RU)")
    print("=" * 70)
    logger = Logger()
    logger.log("Начат полный запуск лаборатории")

    # 1. Сохраняем локальные веса TinyGPT
    logger.log("Сохранение локальных весов TinyGPT...")
    model = TinyGPT()
    model_path = os.path.join(MODELS_DIR, "tiny_gpt_local.npz")
    model.save_weights(model_path)
    logger.log(f"  Сохранено: {model_path}")

    # 2. Запускаем все сценарии
    params = {p.name: p.default for p in P.default_parameter_space()}
    scens = scen.list_scenarios_for_menu()
    logger.log(f"\n=== ЗАПУСК {len(scens)} СЦЕНАРИЕВ ===")
    for sc in scens:
        name = sc.get("name_ru", sc.get("name", sc["id"]))
        logger.log(f"\n--- Сценарий {sc['id']}: {name} ---")
        try:
            r = scen.run_scenario(sc, params, logs=logger.lines)
            emit(r, logger, sc["id"])
            logger.log(f"  Доля совпадений: {r['metrics']['match_rate']:.2%}")
        except Exception as exc:
            logger.log(f"  [ОШИБКА] {type(exc).__name__}: {exc}")

    # 3. Запускаем все эксперименты
    logger.log(f"\n=== ЗАПУСК {len(research.EXPERIMENTS)} ЭКСПЕРИМЕНТОВ ===")
    for eid in research.EXPERIMENTS:
        logger.log(f"\n--- Эксперимент {eid}: {research.EXPERIMENTS[eid].name} ---")
        try:
            r = research.run_experiment(eid, params)
            emit(r, logger, f"exp_{eid}")
            logger.log(f"  Затрачено: {r['elapsed_seconds']:.3f}с")
        except Exception as exc:
            logger.log(f"  [ОШИБКА] {type(exc).__name__}: {exc}")

    # 4. Итоговая сводка
    logger.log("\n=== ПОЛНЫЙ ЗАПУСК ЛАБОРАТОРИИ ЗАВЕРШЁН ===")
    logger.save(os.path.join(LOGS_DIR, "full_lab_session.log"))
    print(f"\nВсе выводы в: {RESULTS_DIR}")
    print(f"  Графики:  {CHARTS_DIR}")
    print(f"  Отчёты:   {REPORTS_DIR}")
    print(f"  Логи:     {LOGS_DIR}")
    print(f"  Модели:   {MODELS_DIR}")


if __name__ == "__main__":
    main()
