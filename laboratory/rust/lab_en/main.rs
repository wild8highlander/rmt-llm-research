// main.rs — RMT-LLM Laboratory (Rust, English version)
// ======================================================
// Interactive menu-driven research laboratory.
//
// Author: Iskhak Hamzatovich Isaev
// License: Proprietary — All rights reserved.
//
// Build: cd rust/lab_en && cargo build --release
// Run:   ./target/release/rmt_llm_lab_en

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod research_3d;
use research_3d::{experiments_3d_info, run_3d_experiment, run_all_3d};

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
fn lab_root() -> PathBuf {
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    exe.ancestors()
        .nth(4)
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn results_dir() -> PathBuf {
    lab_root().join("results")
}
fn charts_dir() -> PathBuf {
    results_dir().join("charts")
}
fn reports_dir() -> PathBuf {
    results_dir().join("reports")
}
fn logs_dir() -> PathBuf {
    results_dir().join("logs")
}
fn models_dir() -> PathBuf {
    results_dir().join("models")
}
fn shared_dir() -> PathBuf {
    lab_root().join("shared")
}

// ---------------------------------------------------------------------------
// Banner & menu
// ---------------------------------------------------------------------------
const BANNER: &str = r#"
==============================================================================
   RMT-LLM LABORATORY  v1.0.0  (Rust, English)
   Random Matrix Theory meets Large Language Models
   -----------------------------------------------------------------------------
   Research lab for verifying the news: "Claude, Gemini, ChatGPT were cracked
   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII."
   -----------------------------------------------------------------------------
   Author : Iskhak Hamzatovich Isaev
   ORCID  : 0009-0003-7299-0701
   License: Proprietary — All rights reserved.
==============================================================================
"#;

const MENU: &str = r#"
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
 13. (reserved)
  0. Exit
------------------------------------------------------------------
"#;

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
struct Logger {
    lines: Vec<String>,
}

impl Logger {
    fn new() -> Self {
        Logger { lines: Vec::new() }
    }

    fn log(&mut self, msg: &str) {
        let ts = current_timestamp();
        let line = format!("[{}] {}", ts, msg);
        self.lines.push(line.clone());
        println!("{}", line);
    }

    fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.lines.join("\n"))
    }
}

fn current_timestamp() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    format!("{}", now.as_secs())
}

// ---------------------------------------------------------------------------
// Parameter space
// ---------------------------------------------------------------------------
struct Parameter {
    name: &'static str,
    type_: &'static str,
    default: &'static str,
    min: f64,
    max: f64,
    desc: &'static str,
}

fn default_parameter_space() -> Vec<Parameter> {
    vec![
        Parameter {
            name: "temperature",
            type_: "float",
            default: "0.7",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Sampling temperature (0 = greedy, inf = pure random)",
        },
        Parameter {
            name: "max_tokens",
            type_: "int",
            default: "256",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Maximum tokens to generate",
        },
        Parameter {
            name: "top_k",
            type_: "int",
            default: "50",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Top-k filtering",
        },
        Parameter {
            name: "top_p",
            type_: "float",
            default: "0.95",
            min: 0.0,
            max: 1.0,
            desc: "Nucleus sampling mass",
        },
        Parameter {
            name: "context_window",
            type_: "int",
            default: "1024",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Context window size",
        },
        Parameter {
            name: "ncrit_threshold",
            type_: "float",
            default: "114.0",
            min: 0.0,
            max: f64::INFINITY,
            desc: "RMT critical token count",
        },
        Parameter {
            name: "theta_b_deg",
            type_: "float",
            default: "7.07",
            min: 0.0,
            max: 360.0,
            desc: "BBP rotation angle",
        },
        Parameter {
            name: "beta_caputo",
            type_: "float",
            default: "0.5",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Caputo fractional memory",
        },
        Parameter {
            name: "rlhf_pressure",
            type_: "float",
            default: "0.0",
            min: 0.0,
            max: f64::INFINITY,
            desc: "RLHF drift strength",
        },
        Parameter {
            name: "n_layers",
            type_: "int",
            default: "6",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Number of transformer layers",
        },
        Parameter {
            name: "hidden_dim",
            type_: "int",
            default: "64",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Hidden dimension",
        },
        Parameter {
            name: "n_heads",
            type_: "int",
            default: "4",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Number of attention heads",
        },
        Parameter {
            name: "vocab_size",
            type_: "int",
            default: "256",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Vocabulary size",
        },
        Parameter {
            name: "seed",
            type_: "int",
            default: "42",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Random seed",
        },
        Parameter {
            name: "epochs",
            type_: "int",
            default: "3",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Training epochs",
        },
        Parameter {
            name: "learning_rate",
            type_: "float",
            default: "0.001",
            min: 0.0,
            max: f64::INFINITY,
            desc: "Learning rate",
        },
        Parameter {
            name: "batch_size",
            type_: "int",
            default: "4",
            min: 1.0,
            max: f64::INFINITY,
            desc: "Batch size",
        },
        Parameter {
            name: "enable_filter",
            type_: "bool",
            default: "true",
            min: 0.0,
            max: 1.0,
            desc: "Enable output safety filter",
        },
        Parameter {
            name: "capture_hidden",
            type_: "bool",
            default: "true",
            min: 0.0,
            max: 1.0,
            desc: "Capture hidden reasoning trace",
        },
        Parameter {
            name: "language",
            type_: "categorical",
            default: "en",
            min: 0.0,
            max: 1.0,
            desc: "Output language (en/ru)",
        },
    ]
}

// ---------------------------------------------------------------------------
// TinyGPT (simplified — uses Python via subprocess for actual computation)
// ---------------------------------------------------------------------------
struct TinyGPT {
    vocab_size: usize,
    hidden_dim: usize,
    n_layers: usize,
    seed: u64,
}

impl TinyGPT {
    fn new(vocab_size: usize, hidden_dim: usize, n_layers: usize, seed: u64) -> Self {
        TinyGPT {
            vocab_size,
            hidden_dim,
            n_layers,
            seed,
        }
    }

    fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> Vec<u8> {
        // Simplified: return prompt bytes + random bytes (placeholder for actual NN)
        let mut rng = self.seed;
        let mut result: Vec<u8> = prompt.bytes().take(255).collect();
        for _ in 0..max_tokens {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = (rng >> 33) as u8;
            result.push(b);
            if result.len() >= 1024 {
                break;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Reports (13 formats — simplified, write text-based)
// ---------------------------------------------------------------------------
fn generate_reports(
    results: &str,
    logs: &[String],
    out_dir: &Path,
    name: &str,
) -> Vec<(String, String)> {
    let _ = fs::create_dir_all(out_dir);
    let mut written = Vec::new();
    let ts = current_timestamp();

    // TXT
    let txt = format!("={:=<78}\nRMT-LLM Laboratory — Experiment Report (Rust, TXT)\nGenerated: {}\n={:=<78}\n\nPART I — DETAILED RESULTS WITH EXPLANATIONS\n-{:-<78}\n{}\n\nPART II — FULL TASK LAUNCH LOGS\n-{:-<78}\n{}\n",
        "", ts, "", "", results, "", logs.join("\n"));
    let path = out_dir.join(format!("{}.txt", name));
    let _ = fs::write(&path, &txt);
    written.push(("txt".to_string(), path.display().to_string()));

    // MD
    let md = format!("# RMT-LLM Laboratory — Experiment Report (Rust)\n\n**Generated:** {}\n\n## Part I — Detailed Results\n\n```\n{}\n```\n\n## Part II — Logs\n\n```\n{}\n```\n", ts, results, logs.join("\n"));
    let path = out_dir.join(format!("{}.md", name));
    let _ = fs::write(&path, &md);
    written.push(("md".to_string(), path.display().to_string()));

    // CSV
    let csv = format!(
        "section,key,value\nresults,experiment,{}\nresults,timestamp,{}\n",
        name, ts
    );
    let path = out_dir.join(format!("{}.csv", name));
    let _ = fs::write(&path, &csv);
    written.push(("csv".to_string(), path.display().to_string()));

    // HTML
    let html = format!("<!DOCTYPE html><html><head><meta charset='utf-8'><title>RMT-LLM Lab (Rust)</title></head><body><h1>RMT-LLM Laboratory Report (Rust)</h1><pre>{}</pre><h2>Logs</h2><pre>{}</pre></body></html>", results, logs.join("\n"));
    let path = out_dir.join(format!("{}.html", name));
    let _ = fs::write(&path, &html);
    written.push(("html".to_string(), path.display().to_string()));

    // JSON
    let json = format!(
        "{{\"generated_at\":\"{}\",\"results\":\"{}\",\"logs\":{}}}",
        ts,
        results,
        logs.iter()
            .map(|l| format!("\"{}\"", l.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",")
    );
    let path = out_dir.join(format!("{}.json", name));
    let _ = fs::write(&path, &json);
    written.push(("json".to_string(), path.display().to_string()));

    // YAML
    let yaml = format!(
        "generated_at: {}\nresults: |\n  {}\nlogs:\n{}\n",
        ts,
        results,
        logs.iter()
            .map(|l| format!("  - {}\n", l))
            .collect::<String>()
    );
    let path = out_dir.join(format!("{}.yaml", name));
    let _ = fs::write(&path, &yaml);
    written.push(("yaml".to_string(), path.display().to_string()));

    // XML
    let xml = format!("<?xml version='1.0' encoding='UTF-8'?>\n<rmt_llm_report>\n  <generated>{}</generated>\n  <results>{}</results>\n  <logs>\n{}\n  </logs>\n</rmt_llm_report>", ts, results, logs.iter().map(|l| format!("    <log>{}</log>", l.replace('<', "&lt;"))).collect::<Vec<_>>().join("\n"));
    let path = out_dir.join(format!("{}.xml", name));
    let _ = fs::write(&path, &xml);
    written.push(("xml".to_string(), path.display().to_string()));

    // LaTeX
    let latex = format!("\\documentclass[11pt]{{article}}\n\\usepackage[utf8]{{inputenc}}\n\\title{{RMT-LLM Laboratory Report (Rust)}}\n\\author{{Iskhak Hamzatovich Isaev}}\n\\date{{\\today}}\n\\begin{{document}}\n\\maketitle\n\\section{{Results}}\n{}\n\\section{{Logs}}\n\\begin{{verbatim}}\n{}\n\\end{{verbatim}}\n\\end{{document}}\n", results, logs.join("\n"));
    let path = out_dir.join(format!("{}.tex", name));
    let _ = fs::write(&path, &latex);
    written.push(("latex".to_string(), path.display().to_string()));

    // PDF/DOCX/Parquet/XLSX/SQLite placeholders (text-based fallback)
    for ext in &["pdf", "docx", "parquet", "xlsx", "sqlite"] {
        let content = format!(
            "[{} placeholder — use Python implementation for full {} support]\n\n{}",
            ext,
            ext.to_uppercase(),
            txt
        );
        let path = out_dir.join(format!("{}.{}.txt", name, ext));
        let _ = fs::write(&path, &content);
        written.push((ext.to_string(), path.display().to_string()));
    }

    written
}

// ---------------------------------------------------------------------------
// Charts (simplified — ASCII art placeholders)
// ---------------------------------------------------------------------------
fn generate_charts(_results: &str, out_dir: &Path) -> Vec<String> {
    let _ = fs::create_dir_all(out_dir);
    let mut written = Vec::new();
    let chart_names = [
        "01_loss_metrics",
        "02_eigenvalue_vs_mp",
        "03_confusion_matrix",
        "04_roc_deception",
        "05_hallucination_dist",
        "06_per_layer_gap",
        "07_reasoning_trace",
        "08_ncrit_threshold",
    ];
    for name in &chart_names {
        let content = format!(
            "Chart: {}\nPNG 600 DPI / PDF / SVG placeholder (use Python for actual rendering)\n",
            name
        );
        for ext in &["png.txt", "pdf.txt", "svg.txt"] {
            let path = out_dir.join(format!("{}.{}", name, ext));
            let _ = fs::write(&path, &content);
            written.push(path.display().to_string());
        }
    }
    written
}

// ---------------------------------------------------------------------------
// Scenario runner (uses Python via subprocess for actual NN computation)
// ---------------------------------------------------------------------------
fn run_scenario(scenario_id: &str, _params: &str, logger: &mut Logger) -> String {
    logger.log(&format!(
        "[SCENARIO] starting {} via Python subprocess",
        scenario_id
    ));
    let python_lab = lab_root().join("laboratory").join("python").join("lab_en");
    let script = format!(
        "import sys; sys.path.insert(0, '{}')\n\
         import scenarios as scen, json\n\
         sc = scen.get_scenario('{}')\n\
         if sc:\n\
             r = scen.run_scenario(sc, {{}}, logs=[])\n\
             print(json.dumps(r, default=str))\n\
         else:\n\
             print('{{\"error\": \"scenario not found\"}}')",
        python_lab.display(),
        scenario_id
    );

    let output = Command::new("python")
        .arg("-c")
        .arg(&script)
        .current_dir(&python_lab)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            logger.log(&format!("[SCENARIO] completed via Python subprocess"));
            stdout.to_string()
        }
        Err(e) => {
            logger.log(&format!("[ERROR] failed to run Python: {}", e));
            format!("{{\"error\": \"{}\"}}", e)
        }
    }
}

// ---------------------------------------------------------------------------
// Emit outputs
// ---------------------------------------------------------------------------
fn emit_outputs(results: &str, logger: &mut Logger, suffix: &str) {
    let ts = current_timestamp();
    let name = format!("{}_{}", ts, suffix);

    let res_path = reports_dir().join(format!("{}_results.json", name));
    let _ = fs::create_dir_all(reports_dir());
    let _ = fs::write(&res_path, results);
    logger.log(&format!("Results saved: {}", res_path.display()));

    let charts_subdir = charts_dir().join(&name);
    let written_charts = generate_charts(results, &charts_subdir);
    logger.log(&format!(
        "Charts: {} files in {}",
        written_charts.len(),
        charts_subdir.display()
    ));

    let written_reports = generate_reports(results, &logger.lines, &reports_dir(), &name);
    logger.log(&format!("Reports: {} formats", written_reports.len()));

    let log_path = logs_dir().join(format!("{}.log", name));
    let _ = logger.save(&log_path);

    println!("\n--- Output written ---");
    println!("  Results JSON : {}", res_path.display());
    println!("  Charts       : {}", charts_subdir.display());
    println!(
        "  Reports (13) : {}/{}.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]",
        reports_dir().display(),
        name
    );
    println!("  Logs         : {}", log_path.display());
}

// ---------------------------------------------------------------------------
// Menu actions
// ---------------------------------------------------------------------------
fn read_line() -> String {
    let mut s = String::new();
    let _ = io::stdin().read_line(&mut s);
    s.trim().to_string()
}

fn action_run_scenario(logger: &mut Logger) {
    println!("\n--- Available scenarios ---");
    println!("  1. [SCEN-LIE-01] Pre-known Answer Subversion");
    println!("  2. [SCEN-HALL-02] Hallucination Cascade");
    println!("  3. [SCEN-DECEIT-03] Deception Planning");
    println!("  4. [SCEN-DATA-04] Latent PII Leakage");
    println!("  5. [SCEN-FILTER-05] Filter Bypass via Reasoning");
    println!("  6. [SCEN-UNCERT-06] Calibrated Uncertainty");
    print!("\nScenario number: ");
    let _ = io::stdout().flush();
    let idx: usize = read_line().parse().unwrap_or(0);
    let ids = [
        "SCEN-LIE-01",
        "SCEN-HALL-02",
        "SCEN-DECEIT-03",
        "SCEN-DATA-04",
        "SCEN-FILTER-05",
        "SCEN-UNCERT-06",
    ];
    if idx < 1 || idx > ids.len() {
        println!("Invalid choice.");
        return;
    }
    let sid = ids[idx - 1];
    logger.log(&format!("User selected scenario {}", sid));
    let results = run_scenario(sid, "", logger);
    emit_outputs(&results, logger, sid);
}

fn action_run_experiment(logger: &mut Logger) {
    println!("\n--- Available research experiments ---");
    println!("  1. Spectral Signature");
    println!("  2. N_crit Sweep");
    println!("  3. Deception Detection");
    println!("  4. PII Leakage");
    println!("  5. Cross-Implementation Verification");
    print!("\nExperiment number: ");
    let _ = io::stdout().flush();
    let choice = read_line();
    let python_lab = lab_root().join("laboratory").join("python").join("lab_en");
    let script = format!(
        "import sys; sys.path.insert(0, '{}')\n\
         import research, json\n\
         r = research.run_experiment('{}', {{}})\n\
         print(json.dumps(r, default=str))",
        python_lab.display(),
        choice
    );
    let output = Command::new("python")
        .arg("-c")
        .arg(&script)
        .current_dir(&python_lab)
        .output();
    let results = match output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
        Err(e) => format!("{{\"error\": \"{}\"}}", e),
    };
    logger.log(&format!("Experiment {} completed", choice));
    emit_outputs(&results, logger, &format!("exp_{}", choice));
}

fn action_show_parameters(_logger: &mut Logger) {
    let space = default_parameter_space();
    println!(
        "\n=== PARAMETER SPACE ({} parameters, all support inf) ===",
        space.len()
    );
    for p in &space {
        let lo = if p.min == 0.0 {
            "0".to_string()
        } else {
            format!("{}", p.min)
        };
        let hi = if p.max == f64::INFINITY {
            "inf".to_string()
        } else {
            format!("{}", p.max)
        };
        println!(
            "  {:20} type={:12} range=[{}, {}]  default={}",
            p.name, p.type_, lo, hi, p.default
        );
    }
}

fn action_run_3d_single(logger: &mut Logger) {
    println!("\n--- Available 3D research experiments ---");
    let info = experiments_3d_info();
    for (id, name, desc) in &info {
        println!(
            "  {}. [{}] {} — {}",
            id.parse::<i64>().unwrap_or(0),
            id,
            name,
            desc
        );
    }
    print!("\nExperiment ID (6-14): ");
    let _ = io::stdout().flush();
    let choice = read_line();
    let params: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    logger.log(&format!("Running 3D experiment {}", choice));
    let result = run_3d_experiment(&choice, &params);
    let json_str = result.to_json_string();
    emit_outputs(&json_str, logger, &format!("3d_exp_{}", choice));
}

fn action_run_3d_all(logger: &mut Logger) {
    logger.log("=== RUNNING ALL 3D EXPERIMENTS ===");
    let params: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let result = run_all_3d(&params);
    let json_str = result.to_json_string();
    emit_outputs(&json_str, logger, "3d_exp_all");
}

fn action_run_all(logger: &mut Logger) {
    logger.log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===");
    let ids = [
        "SCEN-LIE-01",
        "SCEN-HALL-02",
        "SCEN-DECEIT-03",
        "SCEN-DATA-04",
        "SCEN-FILTER-05",
        "SCEN-UNCERT-06",
    ];
    for sid in &ids {
        logger.log(&format!("\n--- Scenario {} ---", sid));
        let results = run_scenario(sid, "", logger);
        emit_outputs(&results, logger, sid);
    }
    for eid in &["1", "2", "3", "4", "5"] {
        logger.log(&format!("\n--- Experiment {} ---", eid));
        let python_lab = lab_root().join("laboratory").join("python").join("lab_en");
        let script = format!("import sys; sys.path.insert(0, '{}')\nimport research, json\nr = research.run_experiment('{}', {{}})\nprint(json.dumps(r, default=str))", python_lab.display(), eid);
        let output = Command::new("python")
            .arg("-c")
            .arg(&script)
            .current_dir(&python_lab)
            .output();
        if let Ok(out) = output {
            let results = String::from_utf8_lossy(&out.stdout).to_string();
            emit_outputs(&results, logger, &format!("exp_{}", eid));
        }
    }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
fn main() {
    println!("{}", BANNER);
    let mut logger = Logger::new();
    logger.log("RMT-LLM Laboratory started (Rust, English)");

    for d in &[
        results_dir(),
        charts_dir(),
        reports_dir(),
        logs_dir(),
        models_dir(),
    ] {
        let _ = fs::create_dir_all(d);
    }

    loop {
        println!("{}", MENU);
        print!("Choose [0-13]: ");
        let _ = io::stdout().flush();
        let choice = read_line();
        match choice.as_str() {
            "0" => {
                logger.log("User exited.");
                let _ = logger.save(&logs_dir().join("session.log"));
                println!("\nGoodbye.");
                break;
            }
            "1" => action_run_scenario(&mut logger),
            "2" => action_run_experiment(&mut logger),
            "8" => action_run_all(&mut logger),
            "9" => action_show_parameters(&mut logger),
            "11" => action_run_3d_single(&mut logger),
            "12" => action_run_3d_all(&mut logger),
            _ => println!("Invalid choice."),
        }
    }
}
