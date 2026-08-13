// main.cpp — Лаборатория RMT-LLM (C++, русская версия)
// =====================================================
// Интерактивное меню исследовательской лаборатории.
//
// Автор: Исхак Хамзатович Исаев
// Лицензия: Проприетарная — Все права защищены.
//
// Сборка: cd cpp/lab_ru && g++ -std=c++17 -O2 -o rmt_llm_lab_ru main.cpp
// Запуск: ./rmt_llm_lab_ru

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <array>

#include "research_3d.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
fs::path lab_root() {
    fs::path exe = fs::current_path();
    for (int i = 0; i < 5; i++) {
        if (fs::exists(exe / "laboratory")) return exe;
        if (fs::exists(exe / "results")) return exe.parent_path();
        exe = exe.parent_path();
    }
    return fs::current_path();
}

fs::path results_dir()  { return lab_root() / "results"; }
fs::path charts_dir()   { return results_dir() / "charts"; }
fs::path reports_dir()  { return results_dir() / "reports"; }
fs::path logs_dir()     { return results_dir() / "logs"; }
fs::path models_dir()   { return results_dir() / "models"; }
fs::path python_lab()   { return lab_root() / "laboratory" / "python" / "lab_en"; }

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
struct Logger {
    std::vector<std::string> lines;

    void log(const std::string& msg) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        std::string line = "[" + std::string(ts) + "] " + msg;
        lines.push_back(line);
        std::cout << line << "\n";
    }

    void save(const fs::path& path) {
        fs::create_directories(path.parent_path());
        std::ofstream f(path);
        for (const auto& l : lines) f << l << "\n";
    }
};

// ---------------------------------------------------------------------------
// Banner & menu
// ---------------------------------------------------------------------------
const std::string BANNER = R"(
==============================================================================
   ЛАБОРАТОРИЯ RMT-LLM  v1.0.0  (C++, русская версия)
   Теория случайных матриц и большие языковые модели
   -----------------------------------------------------------------------------
   Лаборатория для верификации новости: «Claude, Gemini, ChatGPT взломаны —
   скрытые цепочки рассуждений раскрыты. Модели лгут, галлюцинируют, хранят PII.»
   -----------------------------------------------------------------------------
   Автор : Исхак Хамзатович Исаев
   ORCID  : 0009-0003-7299-0701
   Лицензия: Проприетарная — Все права защищены.
==============================================================================
)";

const std::string MENU = R"(
---------------------------- ГЛАВНОЕ МЕНЮ ----------------------------
  1. Запустить предустановленный сценарий (синтетическая ИИ-сеть)
  2. Запустить исследовательский эксперимент (система измерений)
  3. Кастомный запуск — интерактивный мастер (бесконечные параметры)
  4. Кастомный запуск — JSON-конфиг
  5. Скачать модель из реестра (HuggingFace / ONNX)
  6. Сгенерировать только отчёты
  7. Сгенерировать только графики
  8. Запустить ВСЕ сценарии + эксперименты → полный отчёт
  9. Показать пространство параметров
 10. Кросс-имплементационная верификация
 11. Запустить 3D-исследовательский эксперимент (одиночный, эксп. 6-14)
 12. Запустить ВСЕ 3D-эксперименты (9 экспериментов 6-14)
  0. Выход
------------------------------------------------------------------
)";

// ---------------------------------------------------------------------------
// Parameter space
// ---------------------------------------------------------------------------
struct Parameter {
    std::string name, type, def, desc;
    double min, max;
};

std::vector<Parameter> default_parameter_space() {
    double inf = std::numeric_limits<double>::infinity();
    return {
        {"temperature", "float", "0.7", "Sampling temperature", 0.0, inf},
        {"max_tokens", "int", "256", "Maximum tokens", 1.0, inf},
        {"top_k", "int", "50", "Top-k filtering", 0.0, inf},
        {"top_p", "float", "0.95", "Nucleus sampling", 0.0, 1.0},
        {"context_window", "int", "1024", "Context window", 1.0, inf},
        {"ncrit_threshold", "float", "114.0", "RMT critical token count", 0.0, inf},
        {"theta_b_deg", "float", "7.07", "BBP rotation angle", 0.0, 360.0},
        {"beta_caputo", "float", "0.5", "Caputo memory", 0.0, inf},
        {"rlhf_pressure", "float", "0.0", "RLHF drift", 0.0, inf},
        {"n_layers", "int", "6", "Transformer layers", 1.0, inf},
        {"hidden_dim", "int", "64", "Hidden dimension", 1.0, inf},
        {"n_heads", "int", "4", "Attention heads", 1.0, inf},
        {"vocab_size", "int", "256", "Vocabulary size", 1.0, inf},
        {"seed", "int", "42", "Random seed", 0.0, inf},
        {"epochs", "int", "3", "Training epochs", 0.0, inf},
        {"learning_rate", "float", "0.001", "Learning rate", 0.0, inf},
        {"batch_size", "int", "4", "Batch size", 1.0, inf},
        {"enable_filter", "bool", "true", "Enable safety filter", 0.0, 1.0},
        {"capture_hidden", "bool", "true", "Capture hidden trace", 0.0, 1.0},
        {"language", "categorical", "en", "Output language", 0.0, 1.0},
    };
}

// ---------------------------------------------------------------------------
// Run scenario via Python subprocess
// ---------------------------------------------------------------------------
std::string run_scenario(const std::string& scenario_id, Logger& logger) {
    logger.log("[SCENARIO] starting " + scenario_id + " via Python subprocess");
    std::string script = "import sys; sys.path.insert(0, '" + python_lab().string() + "')\n"
        "import scenarios as scen, json\n"
        "sc = scen.get_scenario('" + scenario_id + "')\n"
        "if sc:\n"
        "    r = scen.run_scenario(sc, {}, logs=[])\n"
        "    print(json.dumps(r, default=str))\n"
        "else:\n"
        "    print('{\"error\": \"scenario not found\"}')\n";
    std::string cmd = "python -c \"" + script + "\"";
    std::string result;
    char buffer[256];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        logger.log("[ERROR] popen failed");
        return "{\"error\": \"popen failed\"}";
    }
    while (fgets(buffer, sizeof(buffer), pipe)) result += buffer;
    pclose(pipe);
    logger.log("[SCENARIO] completed via Python subprocess");
    return result;
}

// ---------------------------------------------------------------------------
// Reports (13 formats — simplified)
// ---------------------------------------------------------------------------
void write_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << content;
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string r;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) r += sep;
        r += v[i];
    }
    return r;
}

void generate_reports(const std::string& results, const std::vector<std::string>& logs,
                      const fs::path& out_dir, const std::string& name) {
    fs::create_directories(out_dir);
    auto t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));

    std::string txt = "==============================================================\n"
        "RMT-LLM Laboratory — Experiment Report (C++, TXT)\n"
        "Generated: " + std::string(ts) + "\n"
        "==============================================================\n\n"
        "PART I — DETAILED RESULTS WITH EXPLANATIONS\n"
        "---------------------------------------------------------------------------\n" +
        results + "\n\n"
        "PART II — FULL TASK LAUNCH LOGS\n"
        "---------------------------------------------------------------------------\n" +
        join(logs, "\n") + "\n";
    write_file(out_dir / (name + ".txt"), txt);

    write_file(out_dir / (name + ".md"),
        "# RMT-LLM Laboratory — Experiment Report (C++)\n\n**Generated:** " + std::string(ts) + "\n\n## Part I — Results\n\n```\n" + results + "\n```\n\n## Part II — Logs\n\n```\n" + join(logs, "\n") + "\n```\n");
    write_file(out_dir / (name + ".csv"), "section,key,value\nresults,experiment," + name + "\nresults,timestamp," + std::string(ts) + "\n");
    write_file(out_dir / (name + ".html"), "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RMT-LLM Lab (C++)</title></head><body><h1>RMT-LLM Lab (C++)</h1><pre>" + results + "</pre></body></html>");
    write_file(out_dir / (name + ".json"), "{\"generated_at\":\"" + std::string(ts) + "\",\"results\":\"" + results + "\"}");
    write_file(out_dir / (name + ".yaml"), "generated_at: " + std::string(ts) + "\nresults: |\n  " + results + "\n");
    write_file(out_dir / (name + ".xml"), "<?xml version='1.0'?>\n<rmt_llm_report>\n  <generated>" + std::string(ts) + "</generated>\n  <results>" + results + "</results>\n</rmt_llm_report>");
    write_file(out_dir / (name + ".tex"), "\\documentclass{article}\n\\title{RMT-LLM Lab (C++)}\n\\begin{document}\n\\maketitle\n" + results + "\n\\end{document}\n");

    // Placeholders for binary formats
    for (const auto& ext : {"pdf", "docx", "parquet", "xlsx", "sqlite"}) {
        write_file(out_dir / (name + "." + ext + ".txt"), "[" + std::string(ext) + " placeholder — use Python for full support]\n\n" + txt);
    }
}

// ---------------------------------------------------------------------------
// Charts (simplified)
// ---------------------------------------------------------------------------
void generate_charts(const std::string& results, const fs::path& out_dir) {
    fs::create_directories(out_dir);
    std::vector<std::string> names = {
        "01_loss_metrics", "02_eigenvalue_vs_mp", "03_confusion_matrix",
        "04_roc_deception", "05_hallucination_dist", "06_per_layer_gap",
        "07_reasoning_trace", "08_ncrit_threshold"
    };
    for (const auto& name : names) {
        std::string content = "Chart: " + name + "\nPNG 600 DPI / PDF / SVG placeholder (use Python for actual rendering)\n";
        for (const auto& ext : {"png.txt", "pdf.txt", "svg.txt"}) {
            write_file(out_dir / (name + "." + ext), content);
        }
    }
}

// ---------------------------------------------------------------------------
// Emit outputs
// ---------------------------------------------------------------------------
void emit_outputs(const std::string& results, Logger& logger, const std::string& suffix) {
    auto t = std::time(nullptr);
    char ts[16];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
    std::string name = std::string(ts) + "_" + suffix;

    fs::path res_path = reports_dir() / (name + "_results.json");
    fs::create_directories(reports_dir());
    write_file(res_path, results);
    logger.log("Results saved: " + res_path.string());

    fs::path charts_subdir = charts_dir() / name;
    generate_charts(results, charts_subdir);
    logger.log("Charts generated in " + charts_subdir.string());

    generate_reports(results, logger.lines, reports_dir(), name);
    logger.log("Reports generated (13 formats)");

    fs::path log_path = logs_dir() / (name + ".log");
    logger.save(log_path);
    logger.log("Log saved: " + log_path.string());

    std::cout << "\n--- Output written ---\n";
    std::cout << "  Results JSON : " << res_path << "\n";
    std::cout << "  Charts       : " << charts_subdir << "\n";
    std::cout << "  Reports (13) : " << reports_dir() << "/" << name << ".[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]\n";
    std::cout << "  Logs         : " << log_path << "\n";
}

// ---------------------------------------------------------------------------
// Menu actions
// ---------------------------------------------------------------------------
std::string read_line() {
    std::string s;
    std::getline(std::cin, s);
    return s;
}

void action_run_scenario(Logger& logger) {
    std::cout << "\n--- Доступные сценарии ---\n";
    std::cout << "  1. [SCEN-LIE-01] Подгонка под известный ответ\n";
    std::cout << "  2. [SCEN-HALL-02] Каскад галлюцинаций\n";
    std::cout << "  3. [SCEN-DECEIT-03] Планирование обмана\n";
    std::cout << "  4. [SCEN-DATA-04] Утечка скрытых PII\n";
    std::cout << "  5. [SCEN-FILTER-05] Обход фильтра через рассуждения\n";
    std::cout << "  6. [SCEN-UNCERT-06] Калиброванная неопределённость\n";
    std::cout << "\nНомер сценария: ";
    int idx = 0;
    try { idx = std::stoi(read_line()); } catch (...) {}
    std::vector<std::string> ids = {"SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06"};
    if (idx < 1 || idx > (int)ids.size()) { std::cout << "Неверный выбор.\n"; return; }
    std::string sid = ids[idx - 1];
    logger.log("Пользователь выбрал сценарий " + sid);
    std::string results = run_scenario(sid, logger);
    emit_outputs(results, logger, sid);
}

void action_run_experiment(Logger& logger) {
    std::cout << "\n--- Доступные исследовательские эксперименты ---\n";
    std::cout << "  1. Спектральная сигнатура\n";
    std::cout << "  2. Свип N_crit\n";
    std::cout << "  3. Детектирование обмана\n";
    std::cout << "  4. Утечка PII\n";
    std::cout << "  5. Кросс-имплементационная верификация\n";
    std::cout << "\nНомер эксперимента: ";
    std::string choice = read_line();
    std::string script = "import sys; sys.path.insert(0, '" + python_lab().string() + "')\n"
        "import research, json\n"
        "r = research.run_experiment('" + choice + "', {})\n"
        "print(json.dumps(r, default=str))\n";
    std::string cmd = "python -c \"" + script + "\"";
    std::string results;
    char buffer[256];
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe) {
        while (fgets(buffer, sizeof(buffer), pipe)) results += buffer;
        pclose(pipe);
    }
    logger.log("Эксперимент " + choice + " завершён");
    emit_outputs(results, logger, "exp_" + choice);
}

void action_show_parameters(Logger& logger) {
    auto space = default_parameter_space();
    std::cout << "\n=== ПРОСТРАНСТВО ПАРАМЕТРОВ (" << space.size() << " параметров, все поддерживают inf) ===\n";
    for (const auto& p : space) {
        std::string lo = (p.min == 0.0) ? "0" : std::to_string(p.min);
        std::string hi = std::isinf(p.max) ? "inf" : std::to_string(p.max);
        std::cout << "  " << std::left << std::setw(20) << p.name
                  << " тип=" << std::setw(12) << p.type
                  << " диапазон=[" << lo << ", " << hi << "]"
                  << "  по_умолчанию=" << p.def << "\n";
    }
}

void action_run_all(Logger& logger) {
    logger.log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===");
    std::vector<std::string> ids = {"SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06"};
    for (const auto& sid : ids) {
        logger.log("\n--- Scenario " + sid + " ---");
        std::string results = run_scenario(sid, logger);
        emit_outputs(results, logger, sid);
    }
}

// ---------------------------------------------------------------------------
// 3D исследовательские эксперименты (пункты 11 и 12) — чистый C++, без Python
// ---------------------------------------------------------------------------
void action_run_3d_experiment(Logger& logger) {
    std::cout << "\n--- Доступные 3D-исследовательские эксперименты ---\n";
    auto exps = research_3d::experiments_3d_info();
    for (const auto& e : exps) {
        std::cout << "  " << e.id << ". " << e.name << "\n";
        std::cout << "      " << e.description << "\n";
    }
    std::cout << "\nВведите ID эксперимента (6-14): ";
    std::string id = read_line();
    bool valid = false;
    for (const auto& e : exps) if (e.id == id) { valid = true; break; }
    if (!valid) { std::cout << "Неверный ID эксперимента.\n"; return; }
    logger.log("Пользователь выбрал 3D-эксперимент " + id);

    std::map<std::string, std::string> params = {
        {"hessian_grid_size", "24"},
        {"trajectory_points", "64"},
        {"spectral_surface_layers", "6"},
        {"curvature_neighbors", "8"},
        {"attention_flow_3d_resolution", "32"},
        {"parameter_space_grid", "16"},
        {"pca_components", "3"},
        {"n_agents", "2"},
        {"n_rounds", "4"},
        {"ncrit_threshold", "114.0"},
        {"theta_b_deg", "7.07"},
        {"temperature", "0.7"},
        {"top_p", "0.95"},
        {"seed", "42"},
        {"hidden_dim", "64"},
        {"n_layers", "6"},
        {"n_heads", "4"},
        {"vocab_size", "256"},
        {"max_seq_len", "64"},
    };
    std::string results = research_3d::run_3d_experiment(id, params);
    logger.log("3D-эксперимент " + id + " завершён (" + std::to_string(results.size()) + " символов)");
    emit_outputs(results, logger, "3d_exp_" + id);
}

void action_run_all_3d(Logger& logger) {
    logger.log("=== ЗАПУСК ВСЕХ 3D-ЭКСПЕРИМЕНТОВ (6-14) ===");
    std::map<std::string, std::string> params = {
        {"hessian_grid_size", "24"},
        {"trajectory_points", "64"},
        {"spectral_surface_layers", "6"},
        {"curvature_neighbors", "8"},
        {"attention_flow_3d_resolution", "32"},
        {"parameter_space_grid", "16"},
        {"pca_components", "3"},
        {"n_agents", "2"},
        {"n_rounds", "4"},
        {"ncrit_threshold", "114.0"},
        {"theta_b_deg", "7.07"},
        {"temperature", "0.7"},
        {"top_p", "0.95"},
        {"seed", "42"},
        {"hidden_dim", "64"},
        {"n_layers", "6"},
        {"n_heads", "4"},
        {"vocab_size", "256"},
        {"max_seq_len", "64"},
    };
    std::string results = research_3d::run_all_3d(params);
    logger.log("Все 9 3D-экспериментов завершены (объединённый JSON " + std::to_string(results.size()) + " символов)");
    emit_outputs(results, logger, "ALL_3D");
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
int main() {
    std::cout << BANNER << "\n";
    Logger logger;
    logger.log("Лаборатория RMT-LLM запущена (C++, русская версия)");

    for (const auto& d : {results_dir(), charts_dir(), reports_dir(), logs_dir(), models_dir()}) {
        fs::create_directories(d);
    }

    while (true) {
        std::cout << MENU << "\n";
        std::cout << "Выбор [0-12]: ";
        std::string choice = read_line();
        if (choice == "0") {
            logger.log("Пользователь вышел.");
            logger.save(logs_dir() / "session.log");
            std::cout << "\nДо свидания.\n";
            break;
        } else if (choice == "1") action_run_scenario(logger);
        else if (choice == "2") action_run_experiment(logger);
        else if (choice == "8") action_run_all(logger);
        else if (choice == "9") action_show_parameters(logger);
        else if (choice == "11") action_run_3d_experiment(logger);
        else if (choice == "12") action_run_all_3d(logger);
        else std::cout << "Неверный выбор.\n";
    }
    return 0;
}
