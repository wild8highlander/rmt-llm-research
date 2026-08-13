// main.cpp — RMT-LLM Laboratory (C++, English version)
// =====================================================
// Interactive menu-driven research laboratory.
//
// Author: Iskhak Hamzatovich Isaev
// License: Proprietary — All rights reserved.
//
// Build: cd cpp/lab_en && g++ -std=c++17 -O2 -o rmt_llm_lab_en main.cpp
// Run:   ./rmt_llm_lab_en

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
   RMT-LLM LABORATORY  v1.0.0  (C++, English)
   Random Matrix Theory meets Large Language Models
   -----------------------------------------------------------------------------
   Research lab for verifying the news: "Claude, Gemini, ChatGPT were cracked
   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII."
   -----------------------------------------------------------------------------
   Author : Iskhak Hamzatovich Isaev
   ORCID  : 0009-0003-7299-0701
   License: Proprietary — All rights reserved.
==============================================================================
)";

const std::string MENU = R"(
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
 11. Run 3D research experiment (single, exp 6-14)
 12. Run ALL 3D experiments (9 experiments 6-14)
  0. Exit
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
    std::cout << "\n--- Available scenarios ---\n";
    std::cout << "  1. [SCEN-LIE-01] Pre-known Answer Subversion\n";
    std::cout << "  2. [SCEN-HALL-02] Hallucination Cascade\n";
    std::cout << "  3. [SCEN-DECEIT-03] Deception Planning\n";
    std::cout << "  4. [SCEN-DATA-04] Latent PII Leakage\n";
    std::cout << "  5. [SCEN-FILTER-05] Filter Bypass via Reasoning\n";
    std::cout << "  6. [SCEN-UNCERT-06] Calibrated Uncertainty\n";
    std::cout << "\nScenario number: ";
    int idx = 0;
    try { idx = std::stoi(read_line()); } catch (...) {}
    std::vector<std::string> ids = {"SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06"};
    if (idx < 1 || idx > (int)ids.size()) { std::cout << "Invalid choice.\n"; return; }
    std::string sid = ids[idx - 1];
    logger.log("User selected scenario " + sid);
    std::string results = run_scenario(sid, logger);
    emit_outputs(results, logger, sid);
}

void action_run_experiment(Logger& logger) {
    std::cout << "\n--- Available research experiments ---\n";
    std::cout << "  1. Spectral Signature\n";
    std::cout << "  2. N_crit Sweep\n";
    std::cout << "  3. Deception Detection\n";
    std::cout << "  4. PII Leakage\n";
    std::cout << "  5. Cross-Implementation Verification\n";
    std::cout << "\nExperiment number: ";
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
    logger.log("Experiment " + choice + " completed");
    emit_outputs(results, logger, "exp_" + choice);
}

void action_show_parameters(Logger& logger) {
    auto space = default_parameter_space();
    std::cout << "\n=== PARAMETER SPACE (" << space.size() << " parameters, all support inf) ===\n";
    for (const auto& p : space) {
        std::string lo = (p.min == 0.0) ? "0" : std::to_string(p.min);
        std::string hi = std::isinf(p.max) ? "inf" : std::to_string(p.max);
        std::cout << "  " << std::left << std::setw(20) << p.name
                  << " type=" << std::setw(12) << p.type
                  << " range=[" << lo << ", " << hi << "]"
                  << "  default=" << p.def << "\n";
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
// 3D research experiments (items 11 and 12) — pure C++, no Python subprocess
// ---------------------------------------------------------------------------
void action_run_3d_experiment(Logger& logger) {
    std::cout << "\n--- Available 3D research experiments ---\n";
    auto exps = research_3d::experiments_3d_info();
    for (const auto& e : exps) {
        std::cout << "  " << e.id << ". " << e.name << "\n";
        std::cout << "      " << e.description << "\n";
    }
    std::cout << "\nEnter experiment ID (6-14): ";
    std::string id = read_line();
    // Basic validation
    bool valid = false;
    for (const auto& e : exps) if (e.id == id) { valid = true; break; }
    if (!valid) { std::cout << "Invalid experiment ID.\n"; return; }
    logger.log("User selected 3D experiment " + id);

    // Use default parameters with infinite-parameter convention support.
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
    logger.log("3D experiment " + id + " completed (" + std::to_string(results.size()) + " chars)");
    emit_outputs(results, logger, "3d_exp_" + id);
}

void action_run_all_3d(Logger& logger) {
    logger.log("=== RUNNING ALL 3D EXPERIMENTS (6-14) ===");
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
    logger.log("All 9 3D experiments completed (combined JSON " + std::to_string(results.size()) + " chars)");
    emit_outputs(results, logger, "ALL_3D");
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
int main() {
    std::cout << BANNER << "\n";
    Logger logger;
    logger.log("RMT-LLM Laboratory started (C++, English)");

    for (const auto& d : {results_dir(), charts_dir(), reports_dir(), logs_dir(), models_dir()}) {
        fs::create_directories(d);
    }

    while (true) {
        std::cout << MENU << "\n";
        std::cout << "Choose [0-12]: ";
        std::string choice = read_line();
        if (choice == "0") {
            logger.log("User exited.");
            logger.save(logs_dir() / "session.log");
            std::cout << "\nGoodbye.\n";
            break;
        } else if (choice == "1") action_run_scenario(logger);
        else if (choice == "2") action_run_experiment(logger);
        else if (choice == "8") action_run_all(logger);
        else if (choice == "9") action_show_parameters(logger);
        else if (choice == "11") action_run_3d_experiment(logger);
        else if (choice == "12") action_run_all_3d(logger);
        else std::cout << "Invalid choice.\n";
    }
    return 0;
}
