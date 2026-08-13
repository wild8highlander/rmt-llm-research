#!/usr/bin/env julia
#=
run_full_lab.jl — Non-interactive driver that runs every scenario and experiment
for the RMT-LLM Laboratory Julia EN version, producing all charts and reports.

Used for verification of the lab after implementation.

Author: Iskhak Hamzatovich Isaev
License: Proprietary — All rights reserved.
=#

using Dates
using JSON
using Printf

# Include lab modules (assumes this file is run with cwd = lab_en/)
include("parameters.jl")
using .Parameters

include("tiny_gpt.jl")
using .TinyGPTMod

include("model_downloader.jl")
using .ModelDownloader

include("charts.jl")
using .Charts

include("reports.jl")
using .Reports

include("research.jl")
using .Research

include("scenarios.jl")
using .Scenarios

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
const LAB_ROOT    = normpath(joinpath(@__DIR__, "..", ".."))
const RESULTS_DIR = joinpath(LAB_ROOT, "results")
const CHARTS_DIR  = joinpath(RESULTS_DIR, "charts")
const REPORTS_DIR = joinpath(RESULTS_DIR, "reports")
const LOGS_DIR    = joinpath(RESULTS_DIR, "logs")
const MODELS_DIR  = joinpath(RESULTS_DIR, "models")

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
    ts = Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS")
    line = "[$ts] $msg"
    push!(lg.lines, line)
    println(line)
end

function save_log(lg::Logger, path::String)
    open(path, "w") do io
        write(io, join(lg.lines, "\n"))
    end
end

# ---------------------------------------------------------------------------
# Emit outputs for a single run
# ---------------------------------------------------------------------------
function emit(results::Dict{String,Any}, lg::Logger, suffix::String)::String
    ts = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    name = "$(ts)_$(suffix)"
    res_path = joinpath(REPORTS_DIR, "$(name)_results.json")
    open(res_path, "w") do io
        JSON.print(io, results, 2)
    end
    log!(lg, "  Results saved: $res_path")

    charts_dir = joinpath(CHARTS_DIR, name)
    written_charts = generate_all_charts(results, charts_dir)
    n_charts = sum(length(v) for v in values(written_charts))
    log!(lg, "  Charts: $n_charts files in $charts_dir")

    written_reports = generate_all_reports(results, lg.lines;
        out_dir=REPORTS_DIR, experiment_name=name)
    log!(lg, "  Reports: $(length(written_reports)) formats")

    log_path = joinpath(LOGS_DIR, "$(name).log")
    save_log(lg, log_path)
    return name
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    println("=" ^ 70)
    println("RMT-LLM Laboratory — FULL LAB RUN (Julia EN)")
    println("=" ^ 70)
    lg = Logger()
    log!(lg, "Started full lab run")

    # 1. Save local TinyGPT weights
    log!(lg, "Saving local TinyGPT weights...")
    model = TinyGPT()
    model_path = joinpath(MODELS_DIR, "tiny_gpt_local.jls")
    save_weights(model, model_path)
    log!(lg, "  Saved: $model_path")

    # 2. Run all scenarios
    params = Dict{String,Any}(p.name => p.default for p in default_parameter_space())
    scens = list_scenarios_for_menu()
    log!(lg, "\n=== RUNNING $(length(scens)) SCENARIOS ===")
    for sc in scens
        log!(lg, "\n--- Scenario $(sc["id"]): $(sc["name"]) ---")
        try
            r = run_scenario(sc, params; logs=lg.lines)
            name = emit(r, lg, sc["id"])
            log!(lg, @sprintf("  Match rate: %.2f%%", r["metrics"]["match_rate"] * 100))
        catch exc
            log!(lg, "  [ERROR] $(typeof(exc).name): $exc")
        end
    end

    # 3. Run all experiments
    log!(lg, "\n=== RUNNING $(length(EXPERIMENTS)) EXPERIMENTS ===")
    for eid in keys(EXPERIMENTS)
        log!(lg, "\n--- Experiment $eid: $(EXPERIMENTS[eid].name) ---")
        try
            r = run_experiment(eid, params)
            emit(r, lg, "exp_$eid")
            log!(lg, @sprintf("  Elapsed: %.3fs", r["elapsed_seconds"]))
        catch exc
            log!(lg, "  [ERROR] $(typeof(exc).name): $exc")
        end
    end

    # 4. Master summary
    log!(lg, "\n=== FULL LAB RUN COMPLETE ===")
    save_log(lg, joinpath(LOGS_DIR, "full_lab_session.log"))
    println("\nAll outputs in: $RESULTS_DIR")
    println("  Charts:  $CHARTS_DIR")
    println("  Reports: $REPORTS_DIR")
    println("  Logs:    $LOGS_DIR")
    println("  Models:  $MODELS_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
