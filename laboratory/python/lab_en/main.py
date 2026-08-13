#!/usr/bin/env python3
"""
main.py — RMT-LLM Laboratory (English version)
================================================

Interactive menu-driven research laboratory for the RMT-LLM project.

Main menu:
  1. Run pre-defined scenario (synthetic neural network)
  2. Run research experiment (measurement system)
  3. Custom launch with infinite parameters (wizard)
  4. Custom launch with config file (JSON)
  5. Download model from registry (HuggingFace / ONNX)
  6. Generate reports only (from existing results)
  7. Generate charts only (from existing results)
  8. Run all scenarios + experiments → full report
  9. Show parameter space
  0. Exit

Each run produces:
  - results/charts/ : PNG (600 DPI) + PDF + SVG + interactive Plotly HTML
  - results/reports/: 13 formats (txt, md, csv, html, json, pdf, docx,
                       yaml, xml, latex, parquet, xlsx, sqlite)
  - results/logs/   : full task launch logs

Author: Iskhak Hamzatovich Isaev
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import json
import os
import sys
import time
import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path
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
# Paths
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
# Banner & menu
# ---------------------------------------------------------------------------
BANNER = r"""
==============================================================================
   RMT-LLM LABORATORY  v1.0.0  (English)
   Random Matrix Theory meets Large Language Models
   -----------------------------------------------------------------------------
   Research lab for verifying the news: "Claude, Gemini, ChatGPT were cracked
   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII."
   -----------------------------------------------------------------------------
   Author : Iskhak Hamzatovich Isaev
   ORCID  : 0009-0003-7299-0701
   License: Proprietary — All rights reserved.
==============================================================================
"""

MENU = """
---------------------------- MAIN MENU ----------------------------
  1. Run pre-defined scenario (synthetic NN)
  2. Run research experiment (measurement system)
  3. Custom launch — interactive wizard (infinite params)
  4. Custom launch — JSON config file
  5. Download model from registry (HuggingFace / ONNX)
  6. Generate reports only (from existing results.json)
  7. Generate charts only (from existing results.json)
  8. Run ALL scenarios + experiments -> full report
  9. Show parameter space
 10. Cross-implementation verification
 11. Run 3D research experiment (v1.1.0)
 12. Run ALL 3D experiments -> 3D charts + reports
 13. Generate 3D charts only (from existing results.json)
 14. Train TinyGPT on real corpus (v1.2.0) -> improves match_rate
 15. Generate text from trained TinyGPT
  0. Exit
------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Logging
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
# Menu actions
# ---------------------------------------------------------------------------
def action_run_scenario(logger: Logger) -> None:
    """Menu item 1: pick a scenario, run it, generate reports+charts."""
    scens = scen.list_scenarios_for_menu()
    print("\n--- Available scenarios ---")
    for i, s in enumerate(scens, 1):
        print(f"  {i:2d}. [{s['id']}] {s['name']}")
        print(f"       Expected behavior: {s.get('expected_behavior', 'unknown')}")
        print(f"       {s.get('description', '')[:80]}...")
    choice = input("\nScenario number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(scens)):
        print("Invalid choice.")
        return
    chosen = scens[int(choice) - 1]
    logger.log(f"User selected scenario {chosen['id']}")

    # Get base params (use defaults)
    use_wizard = input("Use interactive parameter wizard? (y/N): ").strip().lower() in ("y", "yes")
    if use_wizard:
        params = P.interactive_wizard()
    else:
        params = {p.name: p.default for p in P.default_parameter_space()}
    logger.log(f"Parameters: {json.dumps({k: v for k, v in params.items()}, default=str)[:200]}")

    # Run
    results = scen.run_scenario(chosen, params, logs=logger.lines)
    logger.log(f"Scenario completed. Match rate: {results['metrics']['match_rate']:.2%}")

    # Generate charts + reports
    _emit_outputs(results, logger, suffix=chosen["id"])


def action_run_experiment(logger: Logger) -> None:
    """Menu item 2: pick a research experiment, run it, generate outputs."""
    print("\n--- Available research experiments ---")
    for k, exp in research.EXPERIMENTS.items():
        print(f"  {k}. {exp.name}")
        print(f"     {exp.description}")
    choice = input("\nExperiment number: ").strip()
    if choice not in research.EXPERIMENTS:
        print("Invalid choice.")
        return
    logger.log(f"User selected experiment {choice}: {research.EXPERIMENTS[choice].name}")

    params = {p.name: p.default for p in P.default_parameter_space()}
    tweak = input("Tweak any parameters? (y/N): ").strip().lower()
    if tweak in ("y", "yes"):
        params = P.interactive_wizard()

    results = research.run_experiment(choice, params)
    logger.log(f"Experiment completed in {results['elapsed_seconds']:.3f}s")
    _emit_outputs(results, logger, suffix=f"exp_{choice}")


def action_custom_wizard(logger: Logger) -> None:
    """Menu item 3: full custom launch via interactive wizard."""
    logger.log("Starting custom launch (interactive wizard)")
    params = P.interactive_wizard()
    logger.log(f"Custom params: {json.dumps({k: v for k, v in params.items()}, default=str)}")

    # Choose what to run
    print("\nWhat to run with these parameters?")
    print("  1. Run a scenario")
    print("  2. Run a research experiment")
    print("  3. Just compute spectral signature of TinyGPT")
    sub = input("Choice: ").strip()
    if sub == "1":
        scens = scen.list_scenarios_for_menu()
        for i, s in enumerate(scens, 1):
            print(f"  {i}. {s['name']}")
        idx = input("Scenario number: ").strip()
        if idx.isdigit() and 1 <= int(idx) <= len(scens):
            results = scen.run_scenario(scens[int(idx) - 1], params, logs=logger.lines)
            _emit_outputs(results, logger, suffix=f"custom_scen_{scens[int(idx)-1]['id']}")
    elif sub == "2":
        for k, exp in research.EXPERIMENTS.items():
            print(f"  {k}. {exp.name}")
        k = input("Experiment number: ").strip()
        if k in research.EXPERIMENTS:
            results = research.run_experiment(k, params)
            _emit_outputs(results, logger, suffix=f"custom_exp_{k}")
    elif sub == "3":
        results = research.run_experiment("1", params)
        _emit_outputs(results, logger, suffix="custom_spectral")
    else:
        print("Invalid choice.")


def action_custom_config(logger: Logger) -> None:
    """Menu item 4: custom launch from JSON config file."""
    path = input("Path to JSON config: ").strip()
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    cfg = P.load_config(path)
    logger.log(f"Loaded config from {path}")
    params = cfg.get("parameters", cfg)
    # Validate
    space = {p.name: p for p in P.default_parameter_space()}
    validated: Dict[str, Any] = {}
    for k, v in params.items():
        if k in space:
            try:
                validated[k] = space[k].validate(v)
            except ValueError as exc:
                logger.log(f"  [WARN] {exc}, using default")
                validated[k] = space[k].default
        else:
            validated[k] = v  # accept unknown params
    logger.log(f"Validated params: {json.dumps(validated, default=str)[:200]}")
    # Run scenario or experiment based on config
    if "scenario_id" in cfg:
        sc = scen.get_scenario(cfg["scenario_id"])
        if sc:
            results = scen.run_scenario(sc, validated, logs=logger.lines)
            _emit_outputs(results, logger, suffix=cfg["scenario_id"])
    elif "experiment_id" in cfg and cfg["experiment_id"] in research.EXPERIMENTS:
        results = research.run_experiment(cfg["experiment_id"], validated)
        _emit_outputs(results, logger, suffix=f"cfg_exp_{cfg['experiment_id']}")
    else:
        # Default: spectral signature
        results = research.run_experiment("1", validated)
        _emit_outputs(results, logger, suffix="cfg_spectral")


def action_download_model(logger: Logger) -> None:
    """Menu item 5: download a model from the registry."""
    pick = md.interactive_pick()
    logger.log(f"User picked model: {pick}")
    if pick == "tiny-gpt-local":
        logger.log("Local model selected — no download needed.")
        # Generate and save local TinyGPT weights
        model = TinyGPT()
        path = os.path.join(MODELS_DIR, "tiny_gpt_local.npz")
        model.save_weights(path)
        logger.log(f"Saved local TinyGPT weights to {path}")
        print(f"\nLocal TinyGPT weights saved to:\n  {path}")
        return
    rep = md.fetch_model(pick, dest_dir=MODELS_DIR)
    if rep.get("ok"):
        logger.log(f"Downloaded {pick}: {rep.get('bytes', 0)} bytes")
        print(f"\nDownloaded to: {rep.get('dest')}")
        print(f"  Size: {rep.get('bytes', 0):,} bytes")
        print(f"  SHA256: {rep.get('sha256')}")
    else:
        logger.log(f"Download failed: {rep.get('error')}")
        print(f"\nDownload failed: {rep.get('error')}")


def action_reports_only(logger: Logger) -> None:
    """Menu item 6: regenerate reports from existing results.json."""
    path = input("Path to results.json (or Enter for latest): ").strip()
    if not path:
        # Find latest in reports dir
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("No results.json found.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Loaded results from {path}")
    suffix = os.path.basename(path).replace("_results.json", "")
    _emit_reports_only(results, logger, suffix)


def action_charts_only(logger: Logger) -> None:
    """Menu item 7: regenerate charts from existing results.json."""
    path = input("Path to results.json (or Enter for latest): ").strip()
    if not path:
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("No results.json found.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Loaded results from {path}")
    written = charts.generate_all_charts(results, CHARTS_DIR)
    logger.log(f"Charts generated: {sum(len(v) for v in written.values())} files")
    print(f"\nCharts written to: {CHARTS_DIR}")
    for fmt, paths in written.items():
        print(f"  {fmt}: {len(paths)} files")


def action_run_all(logger: Logger) -> None:
    """Menu item 8: run everything and produce a comprehensive report."""
    logger.log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===")
    params = {p.name: p.default for p in P.default_parameter_space()}

    all_results: List[Dict[str, Any]] = []

    # All scenarios
    scens = scen.list_scenarios_for_menu()
    for sc in scens:
        logger.log(f"\n--- Scenario {sc['id']} ---")
        try:
            r = scen.run_scenario(sc, params, logs=logger.lines)
            all_results.append(r)
            _emit_outputs(r, logger, suffix=sc["id"], suppress_charts=False)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [ERROR] scenario {sc['id']} failed: {exc}")

    # All experiments
    for eid in research.EXPERIMENTS:
        logger.log(f"\n--- Experiment {eid} ---")
        try:
            r = research.run_experiment(eid, params)
            all_results.append(r)
            _emit_outputs(r, logger, suffix=f"exp_{eid}", suppress_charts=False)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"  [ERROR] experiment {eid} failed: {exc}")

    # Master summary
    summary = {
        "experiment_name": "FULL_LAB_RUN",
        "language": "python",
        "version": "en",
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
    """Menu item 9: display the parameter space."""
    space = P.default_parameter_space()
    print(f"\n=== PARAMETER SPACE ({len(space)} parameters, all support inf) ===")
    for p in space:
        lo = "0" if p.min == 0 else str(p.min)
        hi = "inf" if p.max == float("inf") else str(p.max)
        print(f"  {p.name:20s} type={p.type:12s} range=[{lo}, {hi}]  default={p.default}")
        if p.description:
            print(f"  {'':20s} {p.description}")
    print(f"\nAll numeric parameters accept 'inf' for unbounded values.")


def action_cross_verify(logger: Logger) -> None:
    """Menu item 10: cross-implementation verification."""
    logger.log("Cross-implementation verification")
    results = research.run_experiment("5", {})
    logger.log(f"MP upper theory: {results['mp_upper_theory']:.4f}, "
               f"empirical: {results['mp_upper_empirical']:.4f}, "
               f"rel err: {results['metrics']['upper_rel_err']:.4f}")
    _emit_outputs(results, logger, suffix="cross_verify")


def action_run_3d_experiment(logger: Logger) -> None:
    """Menu item 11: pick a 3D research experiment, run it, generate 3D charts + reports."""
    print("\n--- Available 3D research experiments (v1.1.0) ---")
    for k, exp in research_3d.EXPERIMENTS_3D.items():
        print(f"  {k:>2s}. {exp.name}")
        print(f"      {exp.description}")
    choice = input("\n3D experiment number: ").strip()
    if choice not in research_3d.EXPERIMENTS_3D:
        print("Invalid choice.")
        return
    logger.log(f"User selected 3D experiment {choice}: {research_3d.EXPERIMENTS_3D[choice].name}")

    params = {p.name: p.default for p in P.default_parameter_space()}
    # Add 3D-specific params
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
    tweak = input("Tweak 3D parameters? (y/N): ").strip().lower()
    if tweak in ("y", "yes"):
        params.update(P.interactive_wizard())

    results = research_3d.run_3d_experiment(choice, params)
    logger.log(f"3D experiment completed in {results['elapsed_seconds']:.3f}s")
    _emit_3d_outputs(results, logger, suffix=f"3d_exp_{choice}")


def action_run_all_3d(logger: Logger) -> None:
    """Menu item 12: run all 9 3D experiments, generate 3D charts + comprehensive report."""
    logger.log("=== RUNNING ALL 3D EXPERIMENTS (v1.1.0) ===")
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
    logger.log(f"3D experiments done: {n_ok} succeeded, {n_err} failed")

    combined["experiment_name"] = "ALL_3D_EXPERIMENTS"
    combined["language"] = "python"
    combined["version"] = "en"
    combined["timestamp"] = datetime.datetime.now().isoformat()
    combined["ncrit_threshold"] = float(params.get("ncrit_threshold", 114.0))
    combined["spectral"] = {"mp_upper": 2.7}

    _emit_3d_outputs(combined, logger, suffix="ALL_3D")


def action_3d_charts_only(logger: Logger) -> None:
    """Menu item 13: regenerate 3D charts from existing results.json."""
    path = input("Path to results.json (or Enter for latest): ").strip()
    if not path:
        cands = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith("_results.json")])
        if not cands:
            print("No results.json found.")
            return
        path = os.path.join(REPORTS_DIR, cands[-1])
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    logger.log(f"Loaded results from {path}")
    written = charts_3d.generate_all_3d_charts(results, CHARTS_3D_DIR)
    logger.log(f"3D charts generated: {sum(len(v) for v in written.values())} files")
    print(f"\n3D charts written to: {CHARTS_3D_DIR}")
    for fmt, paths in written.items():
        print(f"  {fmt}: {len(paths)} files")


# ---------------------------------------------------------------------------
# v1.2.0 — TinyGPT real-corpus training
# ---------------------------------------------------------------------------
def action_train_tiny_gpt(logger: Logger) -> None:
    """Menu item 14: train TinyGPT on the project's real text corpus.

    Implements a full backprop-through-transformer training loop on a
    byte-level next-token prediction task. The corpus is assembled from
    README.md, CHANGELOG.md, source code, JSON configs, and HTML docs —
    no external downloads. Trained weights are saved to
    `results/models/tiny_gpt_trained.npz` and the `match_rate` metric
    (greedy next-byte accuracy on a held-out slice) rises from 0%
    (untrained baseline) to a non-zero value.
    """
    print("\n--- TinyGPT Real-Corpus Training (v2.0.0 — BPE + MLP + LN + cosine LR) ---")
    print("Architecture: pre-LayerNorm + 12-layer transformer + 4x GELU MLP")
    print("Tokenizer   : BPE (vocab=512: 256 bytes + 256 learned merges)")
    print("Schedule    : cosine LR with 3-epoch warmup")
    print("Corpus      : README, CHANGELOG, all language labs, webapp, CI workflows")
    print("Backprop    : full reverse-mode autodiff through LN -> attn -> MLP -> LM head")
    print()

    epochs_in = input("Epochs [default=30, supports 'inf']: ").strip() or "30"
    lr_in = input("Learning rate (max) [default=3e-4]: ").strip() or "3e-4"
    batch_in = input("Batch size [default=8]: ").strip() or "8"
    stride_in = input("Stride [default=16, larger=faster]: ").strip() or "16"
    seq_in = input("Sequence length [default=32]: ").strip() or "32"

    cfg = trainer.TrainConfig(
        epochs=epochs_in,
        lr=float(lr_in),
        batch_size=int(batch_in),
        stride=int(stride_in),
        seq_len=int(seq_in),
    )
    logger.log(f"Training TinyGPT v2: epochs={epochs_in} lr={lr_in} "
               f"batch={batch_in} stride={stride_in} seq_len={seq_in} "
               f"layers=12 hidden=128 bpe=512 schedule=cosine")

    def progress(rec):
        msg = (f"  epoch {rec['epoch']:>3d}  loss={rec['loss']:.4f}  "
               f"grad_norm={rec['grad_norm']:.4f}")
        if "match_rate" in rec:
            msg += f"  match_rate={rec['match_rate']:.3%}"
        print(msg)

    print("\nBuilding corpus from project files...")
    result = trainer.train_tiny_gpt(LAB_ROOT, cfg, progress_cb=progress)

    print(f"\n=== Training complete ({result['elapsed_seconds']:.1f}s) ===")
    print(f"  Model params        : {result.get('params_count', '?'):,}")
    print(f"  BPE merges          : {result.get('n_merges', 0)}")
    print(f"  Corpus size         : {result['corpus_bytes']:,} bytes")
    print(f"  Train windows       : {result['n_train_windows']:,}")
    print(f"  Eval windows        : {result['n_eval_windows']:,}")
    print(f"  Baseline match_rate : {result['baseline_match_rate']:.3%} (untrained)")
    print(f"  Final    match_rate : {result['final_match_rate']:.3%}")
    print(f"  Baseline loss       : {result['baseline_loss']:.4f}")
    print(f"  Final    loss       : {result['final_loss']:.4f}")
    print(f"  Weights             : {result['weights_path']}")
    print(f"  BPE merges file     : {result.get('bpe_path', '-')}")

    logger.log(f"Training done: match_rate {result['baseline_match_rate']:.3%} -> "
               f"{result['final_match_rate']:.3%}, "
               f"loss {result['baseline_loss']:.4f} -> {result['final_loss']:.4f}")


def action_generate_from_trained(logger: Logger) -> None:
    """Menu item 15: load the trained TinyGPT and generate text."""
    weights_path = os.path.join(MODELS_DIR, "tiny_gpt_trained.npz")
    bpe_path = os.path.join(MODELS_DIR, "tiny_gpt_bpe.json")
    if not os.path.exists(weights_path):
        print(f"No trained weights at {weights_path}")
        print("Run menu item 14 first.")
        return

    print("\n--- Generate text from trained TinyGPT v2 ---")
    prompt = input("Prompt [default='The RMT-LLM']: ").strip() or "The RMT-LLM"
    temp = float(input("Temperature [default=0.7]: ").strip() or "0.7")
    ntok = int(input("Max new tokens [default=64]: ").strip() or "64")

    model, tok = trainer.load_trained_model(weights_path,
                                             bpe_path if os.path.exists(bpe_path) else None)
    logger.log(f"Loaded trained weights from {weights_path}"
               + (f" + BPE from {bpe_path}" if tok else " (no BPE, byte-level fallback)"))
    out = trainer.generate_sample(model, prompt, tok,
                                   max_new_tokens=ntok,
                                   temperature=temp, seed=42)
    print(f"\nPrompt : {prompt!r}")
    print(f"Output : {out!r}")
    logger.log(f"Generated {len(out)} chars from prompt {prompt!r}")


# ---------------------------------------------------------------------------
# Output emission
# ---------------------------------------------------------------------------
def _emit_outputs(results: Dict[str, Any], logger: Logger, suffix: str,
                  suppress_charts: bool = False) -> None:
    """Save results.json, generate charts + reports, save log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{suffix}"

    # 1. Save results.json
    res_path = os.path.join(REPORTS_DIR, f"{name}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.log(f"Results saved to {res_path}")

    # 2. Generate charts
    if not suppress_charts:
        charts_dir_run = os.path.join(CHARTS_DIR, name)
        written_charts = charts.generate_all_charts(results, charts_dir_run)
        n_charts = sum(len(v) for v in written_charts.values())
        logger.log(f"Charts: {n_charts} files in {charts_dir_run}")

    # 3. Generate reports in 13 formats
    written_reports = reports.generate_all_reports(results, logger.lines,
                                                   out_dir=REPORTS_DIR,
                                                   experiment_name=name)
    logger.log(f"Reports: {len(written_reports)} formats in {REPORTS_DIR}")

    # 4. Save log
    log_path = logger.save(LOGS_DIR, f"{name}.log")
    logger.log(f"Log saved to {log_path}")

    print(f"\n--- Output written ---")
    print(f"  Results JSON : {res_path}")
    if not suppress_charts:
        print(f"  Charts       : {charts_dir_run}")
    print(f"  Reports (13) : {REPORTS_DIR}/{name}.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    print(f"  Logs         : {log_path}")


def _emit_reports_only(results: Dict[str, Any], logger: Logger, suffix: str) -> None:
    written = reports.generate_all_reports(results, logger.lines,
                                           out_dir=REPORTS_DIR,
                                           experiment_name=suffix)
    logger.log(f"Reports regenerated: {len(written)} formats")
    print(f"\nReports written to {REPORTS_DIR}:")
    for fmt, p in written.items():
        print(f"  {fmt:8s} -> {p}")


def _emit_3d_outputs(results: Dict[str, Any], logger: Logger, suffix: str) -> None:
    """Save results.json, generate 3D charts + standard reports, save log."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{suffix}"

    # 1. Save results.json
    res_path = os.path.join(REPORTS_DIR, f"{name}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.log(f"3D results saved to {res_path}")

    # 2. Generate 3D charts
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
    logger.log(f"3D charts: {n_3d} files in {charts_3d_dir_run}")

    # 3. Generate standard reports (13 formats)
    written_reports = reports.generate_all_reports(results, logger.lines,
                                                   out_dir=REPORTS_DIR,
                                                   experiment_name=name)
    logger.log(f"Reports: {len(written_reports)} formats in {REPORTS_DIR}")

    # 4. Save log
    log_path = logger.save(LOGS_DIR, f"{name}.log")
    logger.log(f"Log saved to {log_path}")

    print(f"\n--- 3D Output written ---")
    print(f"  Results JSON   : {res_path}")
    print(f"  3D Charts      : {charts_3d_dir_run}")
    print(f"    PNG (600 DPI): {len(written_3d['png'])} files")
    print(f"    PDF (vector) : {len(written_3d['pdf'])} files")
    print(f"    SVG (vector) : {len(written_3d['svg'])} files")
    print(f"    Plotly HTML  : {len(written_3d['html'])} files (interactive 3D)")
    print(f"  Reports (13)   : {REPORTS_DIR}/{name}.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    print(f"  Logs           : {log_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    print(BANNER)
    logger = Logger()
    logger.log("RMT-LLM Laboratory started (English version)")

    while True:
        print(MENU)
        choice = input("Choose [0-15]: ").strip()
        try:
            if choice == "0":
                logger.log("User exited.")
                logger.save(LOGS_DIR, "session.log")
                print("\nGoodbye.")
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
                print("Invalid choice.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            continue
        except Exception as exc:  # noqa: BLE001
            logger.log(f"[ERROR] {type(exc).__name__}: {exc}")
            print(f"\n[ERROR] {exc}")


if __name__ == "__main__":
    main()
