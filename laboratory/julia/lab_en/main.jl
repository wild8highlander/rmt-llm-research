#!/usr/bin/env julia
# main.jl — RMT-LLM Laboratory (Julia, English version)
# ======================================================
# Interactive menu-driven research laboratory.
#
# Author: Iskhak Hamzatovich Isaev
# License: Proprietary — All rights reserved.

using Printf
using Dates
using JSON
using Random
using LinearAlgebra
using Statistics

include("parameters.jl")
include("tiny_gpt.jl")
include("model_downloader.jl")
include("charts.jl")
include("reports.jl")
include("research.jl")
include("research_3d.jl")
include("scenarios.jl")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
const LAB_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const RESULTS_DIR = joinpath(LAB_ROOT, "results")
const CHARTS_DIR = joinpath(RESULTS_DIR, "charts")
const REPORTS_DIR = joinpath(RESULTS_DIR, "reports")
const LOGS_DIR = joinpath(RESULTS_DIR, "logs")
const MODELS_DIR = joinpath(RESULTS_DIR, "models")

for d in (RESULTS_DIR, CHARTS_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR)
    mkpath(d)
end

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
mutable struct Logger
    lines::Vector{String}
end
Logger() = Logger(String[])
function log!(lg::Logger, msg::String)
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    line = "[$ts] $msg"
    push!(lg.lines, line)
    println(line)
end
function save_log(lg::Logger, path::String)
    mkpath(dirname(path))
    write(path, join(lg.lines, "\n"))
end

# ---------------------------------------------------------------------------
# Banner & menu
# ---------------------------------------------------------------------------
const BANNER = """
==============================================================================
   RMT-LLM LABORATORY  v1.0.0  (Julia, English)
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

const MENU = """
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
 11. Run 3D research experiment (single)
 12. Run ALL 3D experiments
  0. Exit
------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# Emit outputs (charts + reports)
# ---------------------------------------------------------------------------
function emit_outputs(results::Dict, lg::Logger, suffix::String; suppress_charts=false)
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    name = "$(ts)_$(suffix)"

    res_path = joinpath(REPORTS_DIR, "$(name)_results.json")
    open(res_path, "w") do f
        JSON.print(f, results, 2)
    end
    log!(lg, "Results saved: $res_path")

    if !suppress_charts
        charts_dir = joinpath(CHARTS_DIR, name)
        mkpath(charts_dir)
        written = generate_all_charts(results, charts_dir)
        n = sum(length(v) for v in values(written))
        log!(lg, "Charts: $n files in $charts_dir")
    end

    written_reports = generate_all_reports(results, lg.lines,
                                           out_dir=REPORTS_DIR, experiment_name=name)
    log!(lg, "Reports: $(length(written_reports)) formats")

    log_path = joinpath(LOGS_DIR, "$(name).log")
    save_log(lg, log_path)

    println("\n--- Output written ---")
    println("  Results JSON : $res_path")
    println("  Reports (13) : $(REPORTS_DIR)/$(name).[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    println("  Logs         : $log_path")
end

# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------
function action_run_scenario(lg::Logger)
    scens = list_scenarios()
    println("\n--- Available scenarios ---")
    for (i, s) in enumerate(scens)
        println("  $(lpad(i,2)). [$(s["id"])] $(s["name"])")
        println("       Expected: $(get(s, "expected_behavior", "unknown"))")
    end
    print("\nScenario number: "); choice = strip(readline())
    idx = tryparse(Int, choice)
    if idx === nothing || !(1 <= idx <= length(scens))
        println("Invalid choice."); return
    end
    chosen = scens[idx]
    log!(lg, "User selected scenario $(chosen["id"])")
    print("Use interactive parameter wizard? (y/N): "); wiz = strip(readline())
    if lowercase(wiz) in ("y", "yes")
        params = interactive_wizard()
    else
        params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    end
    results = run_scenario(chosen, params, lg)
    log!(lg, "Scenario completed. Match rate: $(results["metrics"]["match_rate"])")
    emit_outputs(results, lg, suffix=chosen["id"])
end

function action_run_experiment(lg::Logger)
    println("\n--- Available research experiments ---")
    for (k, exp) in EXPERIMENTS
        println("  $k. $(exp["name"])")
        println("     $(exp["description"])")
    end
    print("\nExperiment number: "); choice = strip(readline())
    if !haskey(EXPERIMENTS, choice)
        println("Invalid choice."); return
    end
    log!(lg, "User selected experiment $choice")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = run_experiment(choice, params, lg)
    log!(lg, "Experiment completed in $(results["elapsed_seconds"])s")
    emit_outputs(results, lg, suffix="exp_$(choice)")
end

function action_custom_wizard(lg::Logger)
    log!(lg, "Starting custom launch (interactive wizard)")
    params = interactive_wizard()
    log!(lg, "Custom params set")
    println("\nWhat to run?")
    println("  1. Run a scenario")
    println("  2. Run a research experiment")
    println("  3. Just compute spectral signature of TinyGPT")
    print("Choice: "); sub = strip(readline())
    if sub == "1"
        scens = list_scenarios()
        for (i, s) in enumerate(scens)
            println("  $i. $(s["name"])")
        end
        print("Scenario number: "); idx_s = strip(readline())
        idx = tryparse(Int, idx_s)
        if idx !== nothing && 1 <= idx <= length(scens)
            results = run_scenario(scens[idx], params, lg)
            emit_outputs(results, lg, suffix="custom_scen_$(scens[idx]["id"])")
        end
    elseif sub == "2"
        for (k, exp) in EXPERIMENTS
            println("  $k. $(exp["name"])")
        end
        print("Experiment number: "); k = strip(readline())
        if haskey(EXPERIMENTS, k)
            results = run_experiment(k, params, lg)
            emit_outputs(results, lg, suffix="custom_exp_$(k)")
        end
    elseif sub == "3"
        results = run_experiment("1", params, lg)
        emit_outputs(results, lg, suffix="custom_spectral")
    end
end

function action_custom_config(lg::Logger)
    print("Path to JSON config: "); path = strip(readline())
    if !isfile(path)
        println("File not found: $path"); return
    end
    cfg = JSON.parsefile(path)
    log!(lg, "Loaded config from $path")
    params = get(cfg, "parameters", cfg)
    if haskey(cfg, "scenario_id")
        sc = get_scenario(cfg["scenario_id"])
        if sc !== nothing
            results = run_scenario(sc, params, lg)
            emit_outputs(results, lg, suffix=cfg["scenario_id"])
        end
    elseif haskey(cfg, "experiment_id") && haskey(EXPERIMENTS, cfg["experiment_id"])
        results = run_experiment(cfg["experiment_id"], params, lg)
        emit_outputs(results, lg, suffix="cfg_exp_$(cfg['experiment_id'])")
    else
        results = run_experiment("1", params, lg)
        emit_outputs(results, lg, suffix="cfg_spectral")
    end
end

function action_download_model(lg::Logger)
    pick = interactive_pick()
    log!(lg, "User picked model: $pick")
    if pick == "tiny-gpt-local"
        log!(lg, "Local model — saving TinyGPT weights")
        model = TinyGPT()
        path = joinpath(MODELS_DIR, "tiny_gpt_local_julia.jls")
        save_weights(model, path)
        println("\nLocal TinyGPT weights saved to:\n  $path")
        return
    end
    rep = fetch_model(pick, dest_dir=MODELS_DIR)
    if get(rep, "ok", false)
        log!(lg, "Downloaded $pick: $(get(rep, "bytes", 0)) bytes")
        println("\nDownloaded to: $(get(rep, "dest", "n/a"))")
    else
        log!(lg, "Download failed: $(get(rep, "error", "unknown"))")
        println("\nDownload failed: $(get(rep, "error", "unknown"))")
    end
end

function action_run_all(lg::Logger)
    log!(lg, "=== RUNNING ALL SCENARIOS + EXPERIMENTS ===")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    scens = list_scenarios()
    for sc in scens
        log!(lg, "\n--- Scenario $(sc["id"]) ---")
        try
            r = run_scenario(sc, params, lg)
            emit_outputs(r, lg, suffix=sc["id"])
        catch exc
            log!(lg, "  [ERROR] $(typeof(exc)): $exc")
        end
    end
    for eid in keys(EXPERIMENTS)
        log!(lg, "\n--- Experiment $eid ---")
        try
            r = run_experiment(eid, params, lg)
            emit_outputs(r, lg, suffix="exp_$(eid)")
        catch exc
            log!(lg, "  [ERROR] $(typeof(exc)): $exc")
        end
    end
end

function action_show_parameters(lg::Logger)
    space = default_parameter_space()
    println("\n=== PARAMETER SPACE ($(length(space)) parameters, all support inf) ===")
    for p in space
        lo = p["min"] == 0 ? "0" : string(p["min"])
        hi = p["max"] == Inf ? "inf" : string(p["max"])
        println("  $(lpad(p["name"], 20))  type=$(p["type"])  range=[$lo, $hi]  default=$(p["default"])")
    end
end

function action_cross_verify(lg::Logger)
    log!(lg, "Cross-implementation verification")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = run_experiment("5", params, lg)
    log!(lg, "MP upper theory: $(results["mp_upper_theory"]), empirical: $(results["mp_upper_empirical"])")
    emit_outputs(results, lg, suffix="cross_verify")
end

# ---------------------------------------------------------------------------
# 3D outputs (JSON only — Python's charts_3d.py renders the plots)
# ---------------------------------------------------------------------------
function emit_3d_outputs(results::Dict, lg::Logger, suffix::String)
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    name = "$(ts)_$(suffix)"
    res_path = joinpath(REPORTS_DIR, "$(name)_results.json")
    open(res_path, "w") do f
        JSON.print(f, results, 2)
    end
    log!(lg, "3D results saved: $res_path")
    log_path = joinpath(LOGS_DIR, "$(name).log")
    save_log(lg, log_path)
    println("\n--- 3D output written ---")
    println("  Results JSON : $res_path")
    println("  Logs         : $log_path")
    println("  (Render with Python: laboratory/python/lab_en/charts_3d.py)")
end

function action_run_3d_experiment(lg::Logger)
    println("\n--- Available 3D research experiments ---")
    for (k, exp) in Main.research_3d.EXPERIMENTS_3D
        println("  $k. $(exp.name)")
        println("     $(exp.description)")
    end
    print("\nExperiment number: "); choice = strip(readline())
    if !haskey(Main.research_3d.EXPERIMENTS_3D, choice)
        println("Invalid choice."); return
    end
    log!(lg, "User selected 3D experiment $choice")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = Main.research_3d.run_3d_experiment(choice, params)
    log!(lg, "3D experiment '$(results["name"])' completed in $(round(results["elapsed_seconds"], digits=3))s")
    emit_3d_outputs(results, lg, suffix="3d_exp_$(choice)")
end

function action_run_all_3d(lg::Logger)
    log!(lg, "=== RUNNING ALL 3D EXPERIMENTS ===")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    combined = Main.research_3d.run_all_3d(params)
    n_ok = count(e -> !haskey(e, "error"), combined["experiments"])
    n_err = count(e -> haskey(e, "error"), combined["experiments"])
    log!(lg, "3D experiments done: $n_ok succeeded, $n_err failed")
    for e in combined["experiments"]
        if haskey(e, "error")
            log!(lg, "  [FAIL] exp $(e["id"]): $(e["error"])")
        else
            log!(lg, "  [ OK ] exp $(e["id"]) — $(e["name"]) ($(round(e["elapsed_seconds"], digits=3))s)")
        end
    end
    emit_3d_outputs(combined, lg, suffix="3d_all")
end

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
function main()
    println(BANNER)
    lg = Logger()
    log!(lg, "RMT-LLM Laboratory started (Julia, English)")
    while true
        println(MENU)
        print("Choose [0-12]: "); choice = strip(readline())
        try
            if choice == "0"
                log!(lg, "User exited.")
                save_log(lg, joinpath(LOGS_DIR, "session.log"))
                println("\nGoodbye."); break
            elseif choice == "1"; action_run_scenario(lg)
            elseif choice == "2"; action_run_experiment(lg)
            elseif choice == "3"; action_custom_wizard(lg)
            elseif choice == "4"; action_custom_config(lg)
            elseif choice == "5"; action_download_model(lg)
            elseif choice == "8"; action_run_all(lg)
            elseif choice == "9"; action_show_parameters(lg)
            elseif choice == "10"; action_cross_verify(lg)
            elseif choice == "11"; action_run_3d_experiment(lg)
            elseif choice == "12"; action_run_all_3d(lg)
            else println("Invalid choice.")
            end
        catch exc
            log!(lg, "[ERROR] $(typeof(exc)): $exc")
            println("\n[ERROR] $exc")
        end
    end
end

main()
