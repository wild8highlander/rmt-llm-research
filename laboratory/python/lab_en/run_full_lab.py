#!/usr/bin/env python3
"""
run_full_lab.py — Non-interactive driver that runs every scenario and experiment
for the RMT-LLM Laboratory Python EN version, producing all charts and reports.

Used for verification of the lab after implementation.
"""

from __future__ import annotations

import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters as P
import charts
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
    logger.log(f"  Results saved: {res_path}")

    charts_dir = os.path.join(CHARTS_DIR, name)
    written_charts = charts.generate_all_charts(results, charts_dir)
    n_charts = sum(len(v) for v in written_charts.values())
    logger.log(f"  Charts: {n_charts} files in {charts_dir}")

    written_reports = reports.generate_all_reports(results, logger.lines,
                                                   out_dir=REPORTS_DIR,
                                                   experiment_name=name)
    logger.log(f"  Reports: {len(written_reports)} formats")

    log_path = os.path.join(LOGS_DIR, f"{name}.log")
    logger.save(log_path)
    return name


def main():
    print("=" * 70)
    print("RMT-LLM Laboratory — FULL LAB RUN (Python EN)")
    print("=" * 70)
    logger = Logger()
    logger.log("Started full lab run")

    # 1. Save local TinyGPT weights
    logger.log("Saving local TinyGPT weights...")
    model = TinyGPT()
    model_path = os.path.join(MODELS_DIR, "tiny_gpt_local.npz")
    model.save_weights(model_path)
    logger.log(f"  Saved: {model_path}")

    # 2. Run all scenarios
    params = {p.name: p.default for p in P.default_parameter_space()}
    scens = scen.list_scenarios_for_menu()
    logger.log(f"\n=== RUNNING {len(scens)} SCENARIOS ===")
    for sc in scens:
        logger.log(f"\n--- Scenario {sc['id']}: {sc['name']} ---")
        try:
            r = scen.run_scenario(sc, params, logs=logger.lines)
            name = emit(r, logger, sc["id"])
            logger.log(f"  Match rate: {r['metrics']['match_rate']:.2%}")
        except Exception as exc:
            logger.log(f"  [ERROR] {type(exc).__name__}: {exc}")

    # 3. Run all experiments
    logger.log(f"\n=== RUNNING {len(research.EXPERIMENTS)} EXPERIMENTS ===")
    for eid in research.EXPERIMENTS:
        logger.log(f"\n--- Experiment {eid}: {research.EXPERIMENTS[eid].name} ---")
        try:
            r = research.run_experiment(eid, params)
            emit(r, logger, f"exp_{eid}")
            logger.log(f"  Elapsed: {r['elapsed_seconds']:.3f}s")
        except Exception as exc:
            logger.log(f"  [ERROR] {type(exc).__name__}: {exc}")

    # 4. Master summary
    logger.log("\n=== FULL LAB RUN COMPLETE ===")
    logger.save(os.path.join(LOGS_DIR, "full_lab_session.log"))
    print(f"\nAll outputs in: {RESULTS_DIR}")
    print(f"  Charts:  {CHARTS_DIR}")
    print(f"  Reports: {REPORTS_DIR}")
    print(f"  Logs:    {LOGS_DIR}")
    print(f"  Models:  {MODELS_DIR}")


if __name__ == "__main__":
    main()
