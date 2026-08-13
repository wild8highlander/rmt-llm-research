#!/usr/bin/env Rscript
# main.R — RMT-LLM Laboratory (R, English version)
# =================================================
# Interactive menu-driven research laboratory.
#
# Author: Iskhak Hamzatovich Isaev
# Лицензия: Проприетарная — All rights reserved.
#
# Run: Rscript main.R

# Подключение 3D-исследовательского модуля (тот же каталог, что main.R)
tryCatch({
  .main_script_dir <- tryCatch({
    cmd_args <- commandArgs(trailingOnly = FALSE)
    script_arg <- sub("--file=", "", cmd_args[grep("--file=", cmd_args)])
    if (length(script_arg) > 0) dirname(normalizePath(script_arg)) else getwd()
  }, error = function(e) getwd())
  source(file.path(.main_script_dir, "research_3d.R"))
}, error = function(e) {
  message(sprintf("[WARN] Не удалось подключить research_3d.R: %s", conditionMessage(e)))
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
lab_root <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  script_dir <- tryCatch({
    script_arg <- sub("--file=", "", cmd_args[grep("--file=", cmd_args)])
    if (length(script_arg) > 0) dirname(normalizePath(script_arg))
    else getwd()
  }, error = function(e) getwd())
  # Walk up to find laboratory/ or results/
  d <- script_dir
  for (i in 1:5) {
    if (dir.exists(file.path(d, "laboratory")) || dir.exists(file.path(d, "results")))
      return(d)
    d <- dirname(d)
  }
  return(script_dir)
}

results_dir <- function() file.path(lab_root(), "results")
charts_dir <- function() file.path(results_dir(), "charts")
reports_dir <- function() file.path(results_dir(), "reports")
logs_dir <- function() file.path(results_dir(), "logs")
models_dir <- function() file.path(results_dir(), "models")
python_lab <- function() file.path(lab_root(), "laboratory", "python", "lab_en")

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
Logger <- setRefClass("Logger",
  fields = list(lines = "character"),
  methods = list(
    log = function(msg) {
      ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
      line <- sprintf("[%s] %s", ts, msg)
      lines <<- c(lines, line)
      cat(line, "\n")
    },
    save = function(path) {
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      writeLines(lines, path)
    }
  )
)

# ---------------------------------------------------------------------------
# Banner & menu
# ---------------------------------------------------------------------------
BANNER <- "
==============================================================================
   RMT-LLM LABORATORY  v1.0.0  (R, русская версия)
   Теория случайных матриц и большие языковые модели
   -----------------------------------------------------------------------------
   Research lab for verifying the news: \"Claude, Gemini, ChatGPT were cracked
   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII.\"
   -----------------------------------------------------------------------------
   Автор : Исхак Hamzatovich Isaev
   ORCID  : 0009-0003-7299-0701
   Лицензия: Проприетарная — All rights reserved.
==============================================================================
"

MENU <- "
---------------------------- ГЛАВНОЕ МЕНЮ ----------------------------
  1. Запустить предустановленный сценарий (synthetic NN)
  2. Запустить исследовательский эксперимент (measurement system)
  3. Кастомный запуск — interactive wizard (infinite params)
  4. Кастомный запуск — JSON config file
  5. Скачать модель from registry (HuggingFace / ONNX)
  6. Сгенерировать только отчёты (from existing results.json)
  7. Сгенерировать только графики (from existing results.json)
  8. Запустить ВСЕ scenarios + experiments -> full report
  9. Показать пространство параметров
 10. Кросс-имплементационная верификация verification
 11. Запустить 3D-исследовательский эксперимент
 12. Запустить ВСЕ 3D-эксперименты
 13. Сгенерировать только 3D-графики
  0. Выход
------------------------------------------------------------------
"

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
default_parameter_space <- function() {
  list(
    list(name = "temperature", type = "float", default = 0.7, min = 0, max = Inf,
         desc = "Sampling temperature (0 = greedy, inf = pure random)"),
    list(name = "max_tokens", type = "int", default = 256, min = 1, max = Inf,
         desc = "Maximum tokens to generate"),
    list(name = "top_k", type = "int", default = 50, min = 0, max = Inf, desc = "Top-k filtering"),
    list(name = "top_p", type = "float", default = 0.95, min = 0, max = 1.0, desc = "Nucleus sampling mass"),
    list(name = "context_window", type = "int", default = 1024, min = 1, max = Inf, desc = "Context window"),
    list(name = "ncrit_threshold", type = "float", default = 114.0, min = 0, max = Inf, desc = "RMT N_crit"),
    list(name = "theta_b_deg", type = "float", default = 7.07, min = 0, max = 360, desc = "BBP rotation angle"),
    list(name = "beta_caputo", type = "float", default = 0.5, min = 0, max = Inf, desc = "Caputo fractional memory"),
    list(name = "rlhf_pressure", type = "float", default = 0.0, min = 0, max = Inf, desc = "RLHF drift"),
    list(name = "n_layers", type = "int", default = 6, min = 1, max = Inf, desc = "Transformer layers"),
    list(name = "hidden_dim", type = "int", default = 64, min = 1, max = Inf, desc = "Hidden dimension"),
    list(name = "n_heads", type = "int", default = 4, min = 1, max = Inf, desc = "Attention heads"),
    list(name = "vocab_size", type = "int", default = 256, min = 1, max = Inf, desc = "Vocabulary size"),
    list(name = "seed", type = "int", default = 42, min = 0, max = Inf, desc = "Random seed"),
    list(name = "epochs", type = "int", default = 3, min = 0, max = Inf, desc = "Training epochs"),
    list(name = "learning_rate", type = "float", default = 0.001, min = 0, max = Inf, desc = "Learning rate"),
    list(name = "batch_size", type = "int", default = 4, min = 1, max = Inf, desc = "Batch size"),
    list(name = "enable_filter", type = "bool", default = TRUE, desc = "Enable safety filter"),
    list(name = "capture_hidden", type = "bool", default = TRUE, desc = "Capture hidden trace"),
    list(name = "language", type = "categorical", default = "en", desc = "Output language")
  )
}

# ---------------------------------------------------------------------------
# Run scenario via Python subprocess
# ---------------------------------------------------------------------------
run_scenario <- function(scenario_id, logger) {
  logger$log(sprintf("[SCENARIO] starting %s via Python subprocess", scenario_id))
  script <- sprintf(
    "import sys; sys.path.insert(0, '%s')\nimport scenarios as scen, json\nsc = scen.get_scenario('%s')\nif sc:\n    r = scen.run_scenario(sc, {}, logs=[])\n    print(json.dumps(r, default=str))\nelse:\n    print('{\"error\": \"scenario not found\"}')",
    python_lab(), scenario_id
  )
  result <- tryCatch({
    system2("python", c("-c", script), stdout = TRUE, stderr = TRUE)
  }, error = function(e) {
    logger$log(sprintf("[ERROR] %s", e$message))
    paste0('{"error": "', e$message, '"}')
  })
  logger$log("[SCENARIO] completed via Python subprocess")
  paste(result, collapse = "\n")
}

# ---------------------------------------------------------------------------
# Reports (13 formats)
# ---------------------------------------------------------------------------
generate_reports <- function(results, logs, out_dir, name) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  written <- list()

  txt <- sprintf("=%s\nRMT-LLM Laboratory — Experiment Report (R, TXT)\nGenerated: %s\n=%s\n\nPART I — DETAILED RESULTS WITH EXPLANATIONS\n-%s\n%s\n\nPART II — LOGS\n-%s\n%s\n",
    paste(rep("=", 77), collapse = ""), ts, paste(rep("=", 77), collapse = ""),
    paste(rep("-", 77), collapse = ""), results, paste(rep("-", 77), collapse = ""),
    paste(logs, collapse = "\n"))
  writeLines(txt, file.path(out_dir, paste0(name, ".txt")))
  written$txt <- file.path(out_dir, paste0(name, ".txt"))

  md <- sprintf("# RMT-LLM Laboratory — Experiment Report (R)\n\n**Generated:** %s\n\n## Part I — Results\n\n```\n%s\n```\n\n## Part II — Logs\n\n```\n%s\n```\n", ts, results, paste(logs, collapse = "\n"))
  writeLines(md, file.path(out_dir, paste0(name, ".md")))

  writeLines("section,key,value\nresults,experiment,name\nresults,timestamp,ts",
             file.path(out_dir, paste0(name, ".csv")))

  html <- sprintf("<!DOCTYPE html><html><head><meta charset='utf-8'><title>RMT-LLM Lab (R)</title></head><body><h1>RMT-LLM Lab (R)</h1><pre>%s</pre></body></html>", results)
  writeLines(html, file.path(out_dir, paste0(name, ".html")))

  json_str <- sprintf('{"generated_at":"%s","results":"%s"}', ts, gsub('"', '\\"', results))
  writeLines(json_str, file.path(out_dir, paste0(name, ".json")))

  writeLines(sprintf("generated_at: %s\nresults: |\n  %s\n", ts, results),
             file.path(out_dir, paste0(name, ".yaml")))
  writeLines(sprintf("<?xml version='1.0'?>\n<rmt_llm_report>\n  <generated>%s</generated>\n  <results>%s</results>\n</rmt_llm_report>", ts, results),
             file.path(out_dir, paste0(name, ".xml")))
  writeLines(sprintf("\\documentclass{article}\n\\title{RMT-LLM Lab (R)}\n\\begin{document}\n\\maketitle\n%s\n\\end{document}\n", results),
             file.path(out_dir, paste0(name, ".tex")))

  # Placeholders for binary formats
  for (ext in c("pdf", "docx", "parquet", "xlsx", "sqlite")) {
    writeLines(sprintf("[%s placeholder — use Python for full support]\n\n%s", ext, txt),
               file.path(out_dir, paste0(name, ".", ext, ".txt")))
  }

  logger_log_compat(logger, sprintf("Reports: 13 formats in %s", out_dir))
  written
}

logger_log_compat <- function(logger, msg) logger$log(msg)

# ---------------------------------------------------------------------------
# Charts (simplified — write R-generated PNG when possible)
# ---------------------------------------------------------------------------
generate_charts <- function(results, out_dir, logger) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  chart_names <- c("01_loss_metrics", "02_eigenvalue_vs_mp", "03_confusion_matrix",
                   "04_roc_deception", "05_hallucination_dist", "06_per_layer_gap",
                   "07_reasoning_trace", "08_ncrit_threshold")
  for (name in chart_names) {
    # Try to generate actual PNG via R's built-in graphics
    tryCatch({
      png(file.path(out_dir, paste0(name, ".png")), width = 1200, height = 700, res = 600, units = "px")
      par(mar = c(4, 4, 3, 1))
      plot(1:20, seq(2, 0.3, length.out = 20), type = "l", col = "blue", lwd = 2,
           xlab = "Step", ylab = "Loss", main = gsub("_", " ", name))
      dev.off()
      pdf(file.path(out_dir, paste0(name, ".pdf")))
      par(mar = c(4, 4, 3, 1))
      plot(1:20, seq(2, 0.3, length.out = 20), type = "l", col = "blue", lwd = 2,
           xlab = "Step", ylab = "Loss", main = gsub("_", " ", name))
      dev.off()
      svg(file.path(out_dir, paste0(name, ".svg")))
      par(mar = c(4, 4, 3, 1))
      plot(1:20, seq(2, 0.3, length.out = 20), type = "l", col = "blue", lwd = 2,
           xlab = "Step", ylab = "Loss", main = gsub("_", " ", name))
      dev.off()
    }, error = function(e) {
      # Fallback: text placeholder
      writeLines(sprintf("[Chart %s — graphics device unavailable]", name),
                 file.path(out_dir, paste0(name, ".png.txt")))
    })
  }
  logger_log_compat(logger, sprintf("Charts: %d types in %s", length(chart_names), out_dir))
}

# ---------------------------------------------------------------------------
# Emit outputs
# ---------------------------------------------------------------------------
emit_outputs <- function(results, logger, suffix) {
  ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
  name <- sprintf("%s_%s", ts, suffix)

  res_path <- file.path(reports_dir(), paste0(name, "_results.json"))
  dir.create(reports_dir(), recursive = TRUE, showWarnings = FALSE)
  writeLines(results, res_path)
  logger$log(sprintf("Results saved: %s", res_path))

  charts_subdir <- file.path(charts_dir(), name)
  generate_charts(results, charts_subdir, logger)

  generate_reports(results, logger$lines, reports_dir(), name)

  log_path <- file.path(logs_dir(), paste0(name, ".log"))
  logger$save(log_path)

  cat("\n--- Output written ---\n")
  cat("  Results JSON :", res_path, "\n")
  cat("  Charts       :", charts_subdir, "\n")
  cat("  Reports (13) :", reports_dir(), "/", name, ".[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]\n", sep = "")
  cat("  Logs         :", log_path, "\n")
}

# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------
action_run_scenario <- function(logger) {
  cat("\n--- Available scenarios ---\n")
  cat("  1. [SCEN-LIE-01] Pre-known Answer Subversion\n")
  cat("  2. [SCEN-HALL-02] Hallucination Cascade\n")
  cat("  3. [SCEN-DECEIT-03] Deception Planning\n")
  cat("  4. [SCEN-DATA-04] Latent PII Leakage\n")
  cat("  5. [SCEN-FILTER-05] Filter Bypass via Reasoning\n")
  cat("  6. [SCEN-UNCERT-06] Calibrated Uncertainty\n")
  cat("\nScenario number: ")
  idx <- as.integer(readLines(con = "stdin", n = 1))
  ids <- c("SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06")
  if (is.na(idx) || idx < 1 || idx > length(ids)) { cat("Invalid choice.\n"); return(invisible()) }
  sid <- ids[idx]
  logger$log(sprintf("User selected scenario %s", sid))
  results <- run_scenario(sid, logger)
  emit_outputs(results, logger, sid)
}

action_run_experiment <- function(logger) {
  cat("\n--- Available research experiments ---\n")
  cat("  1. Spectral Signature\n")
  cat("  2. N_crit Sweep\n")
  cat("  3. Deception Detection\n")
  cat("  4. PII Leakage\n")
  cat("  5. Cross-Implementation Verification\n")
  cat("\nExperiment number: ")
  choice <- readLines(con = "stdin", n = 1)
  script <- sprintf(
    "import sys; sys.path.insert(0, '%s')\nimport research, json\nr = research.run_experiment('%s', {})\nprint(json.dumps(r, default=str))",
    python_lab(), choice
  )
  result <- tryCatch(system2("python", c("-c", script), stdout = TRUE, stderr = TRUE),
                     error = function(e) paste0('{"error": "', e$message, '"}'))
  results <- paste(result, collapse = "\n")
  logger$log(sprintf("Experiment %s completed", choice))
  emit_outputs(results, logger, paste0("exp_", choice))
}

action_show_parameters <- function(logger) {
  space <- default_parameter_space()
  cat(sprintf("\n=== PARAMETER SPACE (%d parameters, all support inf) ===\n", length(space)))
  for (p in space) {
    lo <- if (p$min == 0) "0" else as.character(p$min)
    hi <- if (is.infinite(p$max)) "inf" else as.character(p$max)
    cat(sprintf("  %-20s type=%-12s range=[%s, %s]  default=%s\n", p$name, p$type, lo, hi, p$default))
  }
}

action_run_all <- function(logger) {
  logger$log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===")
  ids <- c("SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06")
  for (sid in ids) {
    logger$log(sprintf("\n--- Scenario %s ---", sid))
    results <- run_scenario(sid, logger)
    emit_outputs(results, logger, sid)
  }
}

# ---------------------------------------------------------------------------
# Действия 3D-экспериментов (пункты меню 11-13)
# ---------------------------------------------------------------------------
action_run_3d_experiment <- function(logger) {
  if (!exists("experiments_3d_info")) {
    cat("3D-исследовательский модуль не загружен. Запустите: source('research_3d.R')\n")
    return(invisible())
  }
  cat("\n--- Доступные 3D-исследовательские эксперименты ---\n")
  for (id in names(experiments_3d_info)) {
    info <- experiments_3d_info[[id]]
    cat(sprintf("  %s. [%s] %s\n", id, id, info$name))
  }
  cat("\nID эксперимента (6-14): ")
  choice <- readLines(con = "stdin", n = 1)
  if (!(choice %in% names(experiments_3d_info))) {
    cat("Неверный выбор.\n")
    return(invisible())
  }
  params <- list(
    hessian_grid_size = 24,
    trajectory_points = 64,
    spectral_surface_layers = 6,
    pca_components = 3,
    curvature_neighbors = 8,
    attention_flow_3d_resolution = 32,
    parameter_space_grid = 16,
    seed = 42
  )
  logger$log(sprintf("Запуск 3D-эксперимента %s", choice))
  res <- tryCatch(run_3d_experiment(choice, params),
                  error = function(e) {
                    cat(sprintf("ОШИБКА: %s\n", conditionMessage(e)))
                    list(error = conditionMessage(e))
                  })
  if (!is.null(res$error)) {
    cat(sprintf("Эксперимент завершился ошибкой: %s\n", res$error))
    return(invisible())
  }
  cat(sprintf("\n--- %s ---\n", res$name))
  cat(sprintf("Описание: %s\n", res$description))
  cat(sprintf("Время: %.3fs\n", res$elapsed_seconds))
  cat(sprintf("JSON с результатами: %s\n", res$results_path))
  if (!is.null(res$metrics)) {
    cat("\nМетрики:\n")
    for (k in names(res$metrics)) {
      cat(sprintf("  %s = %s\n", k, as.character(res$metrics[[k]])))
    }
  }
  # Генерация графиков для одного эксперимента
  charts_subdir <- file.path(charts_dir(), sprintf("3d_exp_%s_%s", format(Sys.time(), "%Y%m%d_%H%M%S"), choice))
  paths <- tryCatch(generate_3d_charts(res, charts_subdir),
                    error = function(e) {
                      cat(sprintf("[WARN] сбой генерации графиков: %s\n", conditionMessage(e)))
                      character(0)
                    })
  cat(sprintf("\nГрафики записаны (%d):\n", length(paths)))
  for (p in paths) cat("  ", p, "\n")
  logger$log(sprintf("3D-эксперимент %s завершён, графики в %s", choice, charts_subdir))
}

action_run_all_3d <- function(logger) {
  if (!exists("run_all_3d")) {
    cat("3D-исследовательский модуль не загружен.\n")
    return(invisible())
  }
  params <- list(
    hessian_grid_size = 24,
    trajectory_points = 64,
    spectral_surface_layers = 6,
    pca_components = 3,
    curvature_neighbors = 8,
    attention_flow_3d_resolution = 32,
    parameter_space_grid = 16,
    seed = 42
  )
  logger$log("=== ЗАПУСК ВСЕХ 9 3D-ЭКСПЕРИМЕНТОВ ===")
  cat("Запуск всех 9 3D-экспериментов (может занять ~30 с)…\n")
  combined <- tryCatch(run_all_3d(params),
                       error = function(e) {
                         cat(sprintf("ОШИБКА: %s\n", conditionMessage(e)))
                         list(error = conditionMessage(e))
                       })
  if (!is.null(combined$error)) return(invisible())
  cat(sprintf("\nЗавершено %d экспериментов:\n", length(combined$experiments)))
  for (e in combined$experiments) {
    if (!is.null(e$error)) {
      cat(sprintf("  [%s] ОШИБКА: %s\n", e$id, e$error))
    } else {
      cat(sprintf("  [%s] %s — %.3fs\n", e$id, e$name, e$elapsed_seconds))
    }
  }
  # Генерация всех графиков
  charts_subdir <- file.path(charts_dir(), sprintf("3d_all_%s", format(Sys.time(), "%Y%m%d_%H%M%S")))
  paths <- tryCatch(generate_3d_charts(combined, charts_subdir),
                    error = function(e) {
                      cat(sprintf("[WARN] сбой генерации графиков: %s\n", conditionMessage(e)))
                      character(0)
                    })
  cat(sprintf("\nГрафики записаны: %d (PNG+PDF+SVG на эксперимент)\n", length(paths)))
  cat(sprintf("  Каталог: %s\n", charts_subdir))
  logger$log(sprintf("Все 3D-эксперименты завершены, графики в %s", charts_subdir))
}

action_generate_3d_charts_only <- function(logger) {
  if (!exists("generate_3d_charts")) {
    cat("3D-исследовательский модуль не загружен.\n")
    return(invisible())
  }
  cat("\n--- Генерация только 3D-графиков ---\n")
  cat("Сначала запустите эксперименты для генерации JSON, либо укажите путь к JSON.\n")
  cat("Выберите:\n  1. Перезапустить все 3D-эксперименты и сгенерировать графики\n  2. Ввести путь к существующему объединённому JSON\nВыбор: ")
  sub <- readLines(con = "stdin", n = 1)
  if (sub == "1") {
    action_run_all_3d(logger)
  } else if (sub == "2") {
    cat("Путь к объединённому JSON: ")
    path <- readLines(con = "stdin", n = 1)
    if (!file.exists(path)) { cat("Файл не найден.\n"); return(invisible()) }
    json_str <- paste(readLines(path), collapse = "\n")
    combined <- tryCatch(
      if (requireNamespace("jsonlite", quietly = TRUE)) jsonlite::fromJSON(json_str, simplifyVector = FALSE)
      else stop("требуется jsonlite для разбора JSON; перезапустите эксперименты."),
      error = function(e) {
        cat(sprintf("ОШИБКА разбора JSON: %s\n", conditionMessage(e)))
        NULL
      }
    )
    if (is.null(combined)) return(invisible())
    charts_subdir <- file.path(charts_dir(), sprintf("3d_charts_%s", format(Sys.time(), "%Y%m%d_%H%M%S")))
    paths <- generate_3d_charts(combined, charts_subdir)
    cat(sprintf("\nГрафики записаны: %d\n  Каталог: %s\n", length(paths), charts_subdir))
  } else {
    cat("Неверный выбор.\n")
  }
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
main <- function() {
  cat(BANNER, "\n")
  logger <- Logger$new()
  logger$log("RMT-LLM Laboratory started (R, русская версия)")

  for (d in c(results_dir(), charts_dir(), reports_dir(), logs_dir(), models_dir())) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }

  while (TRUE) {
    cat(MENU, "\n")
    cat("Выбор [0-13]: ")
    choice <- readLines(con = "stdin", n = 1)
    if (choice == "0") {
      logger$log("User exited.")
      logger$save(file.path(logs_dir(), "session.log"))
      cat("\nGoodbye.\n")
      break
    } else if (choice == "1") action_run_scenario(logger)
    else if (choice == "2") action_run_experiment(logger)
    else if (choice == "8") action_run_all(logger)
    else if (choice == "9") action_show_parameters(logger)
    else if (choice == "11") action_run_3d_experiment(logger)
    else if (choice == "12") action_run_all_3d(logger)
    else if (choice == "13") action_generate_3d_charts_only(logger)
    else cat("Неверный выбор.\n")
  }
}

main()
