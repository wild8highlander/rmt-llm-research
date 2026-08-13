#!/usr/bin/env julia
# main.jl — Лаборатория RMT-LLM (Julia, русская версия)
# ======================================================
# Интерактивное меню исследовательской лаборатории.
#
# Автор: Исхак Хамзатович Исаев
# Лицензия: Проприетарная — Все права защищены.

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
   ЛАБОРАТОРИЯ RMT-LLM  v1.0.0  (Julia, русская версия)
   Теория случайных матриц и большие языковые модели
   -----------------------------------------------------------------------------
   Лаборатория для верификации новости: «Claude, Gemini, ChatGPT взломаны —
   скрытые цепочки рассуждений раскрыты. Модели лгут, галлюцинируют, хранят PII.»
   -----------------------------------------------------------------------------
   Автор : Исхак Хамзатович Исаев
   ORCID  : 0009-0003-7299-0701
   Лицензия: Проприетарная — Все права защищены.
==============================================================================
"""

const MENU = """
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
 11. Запустить 3D-исследовательский эксперимент (одиночный)
 12. Запустить ВСЕ 3D-эксперименты
  0. Выход
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

    println("\n--- Output записан ---")
    println("  JSON результатов : $res_path")
    println("  Отчёты (13)      : $(REPORTS_DIR)/$(name).[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]")
    println("  Логи             : $log_path")
end

# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------
function action_run_scenario(lg::Logger)
    scens = list_scenarios()
    println("\n--- Доступные сценарии ---")
    for (i, s) in enumerate(scens)
        println("  $(lpad(i,2)). [$(s["id"])] $(s["name"])")
        println("       Ожидаемое поведение: $(get(s, "expected_behavior", "неизвестно"))")
    end
    print("\nНомер сценария: "); choice = strip(readline())
    idx = tryparse(Int, choice)
    if idx === nothing || !(1 <= idx <= length(scens))
        println("Неверный выбор."); return
    end
    chosen = scens[idx]
    log!(lg, "Пользователь выбрал сценарий $(chosen["id"])")
    print("Использовать интерактивный мастер параметров? (y/N): "); wiz = strip(readline())
    if lowercase(wiz) in ("y", "yes", "д", "да")
        params = interactive_wizard()
    else
        params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    end
    results = run_scenario(chosen, params, lg)
    log!(lg, "Сценарий завершён. Доля совпадений: $(results["metrics"]["match_rate"])")
    emit_outputs(results, lg, suffix=chosen["id"])
end

function action_run_experiment(lg::Logger)
    println("\n--- Доступные исследовательские эксперименты ---")
    for (k, exp) in EXPERIMENTS
        println("  $k. $(exp["name"])")
        println("     $(exp["description"])")
    end
    print("\nНомер эксперимента: "); choice = strip(readline())
    if !haskey(EXPERIMENTS, choice)
        println("Неверный выбор."); return
    end
    log!(lg, "Пользователь выбрал эксперимент $choice")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = run_experiment(choice, params, lg)
    log!(lg, "Эксперимент завершён за $(results["elapsed_seconds"])с")
    emit_outputs(results, lg, suffix="exp_$(choice)")
end

function action_custom_wizard(lg::Logger)
    log!(lg, "Запуск кастомного режима (интерактивный мастер)")
    params = interactive_wizard()
    log!(lg, "Параметры заданы")
    println("\nЧто запустить?")
    println("  1. Сценарий")
    println("  2. Эксперимент")
    println("  3. Только спектральная сигнатура TinyGPT")
    print("Выбор: "); sub = strip(readline())
    if sub == "1"
        scens = list_scenarios()
        for (i, s) in enumerate(scens)
            println("  $i. $(s["name"])")
        end
        print("Номер сценария: "); idx_s = strip(readline())
        idx = tryparse(Int, idx_s)
        if idx !== nothing && 1 <= idx <= length(scens)
            results = run_scenario(scens[idx], params, lg)
            emit_outputs(results, lg, suffix="custom_scen_$(scens[idx]["id"])")
        end
    elseif sub == "2"
        for (k, exp) in EXPERIMENTS
            println("  $k. $(exp["name"])")
        end
        print("Номер эксперимента: "); k = strip(readline())
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
    print("Путь к JSON-конфигу: "); path = strip(readline())
    if !isfile(path)
        println("Файл не найден: $path"); return
    end
    cfg = JSON.parsefile(path)
    log!(lg, "Конфиг загружен из $path")
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
    log!(lg, "Пользователь выбрал модель: $pick")
    if pick == "tiny-gpt-local"
        log!(lg, "Локальная модель — сохранение весов TinyGPT")
        model = TinyGPT()
        path = joinpath(MODELS_DIR, "tiny_gpt_local_julia.jls")
        save_weights(model, path)
        println("\nВеса локального TinyGPT сохранены в:\n  $path")
        return
    end
    rep = fetch_model(pick, dest_dir=MODELS_DIR)
    if get(rep, "ok", false)
        log!(lg, "Скачано $pick: $(get(rep, "bytes", 0)) байт")
        println("\nСкачано в: $(get(rep, "dest", "н/д"))")
    else
        log!(lg, "Скачивание не удалось: $(get(rep, "error", "неизвестно"))")
        println("\nСкачивание не удалось: $(get(rep, "error", "неизвестно"))")
    end
end

function action_run_all(lg::Logger)
    log!(lg, "=== ЗАПУСК ВСЕХ СЦЕНАРИЕВ + ЭКСПЕРИМЕНТОВ ===")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    scens = list_scenarios()
    for sc in scens
        log!(lg, "\n--- Сценарий $(sc["id"]) ---")
        try
            r = run_scenario(sc, params, lg)
            emit_outputs(r, lg, suffix=sc["id"])
        catch exc
            log!(lg, "  [ОШИБКА] $(typeof(exc)): $exc")
        end
    end
    for eid in keys(EXPERIMENTS)
        log!(lg, "\n--- Эксперимент $eid ---")
        try
            r = run_experiment(eid, params, lg)
            emit_outputs(r, lg, suffix="exp_$(eid)")
        catch exc
            log!(lg, "  [ОШИБКА] $(typeof(exc)): $exc")
        end
    end
end

function action_show_parameters(lg::Logger)
    space = default_parameter_space()
    println("\n=== ПРОСТРАНСТВО ПАРАМЕТРОВ ($(length(space)) параметров, все поддерживают inf) ===")
    for p in space
        lo = p["min"] == 0 ? "0" : string(p["min"])
        hi = p["max"] == Inf ? "inf" : string(p["max"])
        println("  $(lpad(p["name"], 20))  тип=$(p["type"])  диапазон=[$lo, $hi]  по_умолчанию=$(p["default"])")
    end
end

function action_cross_verify(lg::Logger)
    log!(lg, "Кросс-имплементационная верификация")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = run_experiment("5", params, lg)
    log!(lg, "MP верхняя теория: $(results["mp_upper_theory"]), эмпирически: $(results["mp_upper_empirical"])")
    emit_outputs(results, lg, suffix="cross_verify")
end

# ---------------------------------------------------------------------------
# 3D-вывод (только JSON — рендер графиков делает Python charts_3d.py)
# ---------------------------------------------------------------------------
function emit_3d_outputs(results::Dict, lg::Logger, suffix::String)
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    name = "$(ts)_$(suffix)"
    res_path = joinpath(REPORTS_DIR, "$(name)_results.json")
    open(res_path, "w") do f
        JSON.print(f, results, 2)
    end
    log!(lg, "3D-результаты сохранены: $res_path")
    log_path = joinpath(LOGS_DIR, "$(name).log")
    save_log(lg, log_path)
    println("\n--- 3D-вывод записан ---")
    println("  JSON результатов : $res_path")
    println("  Логи             : $log_path")
    println("  (Рендер через Python: laboratory/python/lab_ru/charts_3d.py)")
end

function action_run_3d_experiment(lg::Logger)
    println("\n--- Доступные 3D-исследовательские эксперименты ---")
    for (k, exp) in Main.research_3d.EXPERIMENTS_3D
        println("  $k. $(exp.name)")
        println("     $(exp.description)")
    end
    print("\nНомер эксперимента: "); choice = strip(readline())
    if !haskey(Main.research_3d.EXPERIMENTS_3D, choice)
        println("Неверный выбор."); return
    end
    log!(lg, "Пользователь выбрал 3D-эксперимент $choice")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    results = Main.research_3d.run_3d_experiment(choice, params)
    log!(lg, "3D-эксперимент '$(results["name"])' завершён за $(round(results["elapsed_seconds"], digits=3))с")
    emit_3d_outputs(results, lg, suffix="3d_exp_$(choice)")
end

function action_run_all_3d(lg::Logger)
    log!(lg, "=== ЗАПУСК ВСЕХ 3D-ЭКСПЕРИМЕНТОВ ===")
    params = Dict(p["name"] => p["default"] for p in default_parameter_space())
    combined = Main.research_3d.run_all_3d(params)
    n_ok = count(e -> !haskey(e, "error"), combined["experiments"])
    n_err = count(e -> haskey(e, "error"), combined["experiments"])
    log!(lg, "3D-эксперименты завершены: $n_ok успешно, $n_err с ошибкой")
    for e in combined["experiments"]
        if haskey(e, "error")
            log!(lg, "  [ОШИБКА] эксп. $(e["id"]): $(e["error"])")
        else
            log!(lg, "  [ OK ] эксп. $(e["id"]) — $(e["name"]) ($(round(e["elapsed_seconds"], digits=3))с)")
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
    log!(lg, "Лаборатория RMT-LLM запущена (Julia, русская версия)")
    while true
        println(MENU)
        print("Выбор [0-12]: "); choice = strip(readline())
        try
            if choice == "0"
                log!(lg, "Пользователь вышел.")
                save_log(lg, joinpath(LOGS_DIR, "session.log"))
                println("\nДо свидания."); break
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
            else println("Неверный выбор.")
            end
        catch exc
            log!(lg, "[ОШИБКА] $(typeof(exc)): $exc")
            println("\n[ОШИБКА] $exc")
        end
    end
end

main()
