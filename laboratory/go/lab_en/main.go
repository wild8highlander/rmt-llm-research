// main.go — RMT-LLM Laboratory (Go, English version)
// ====================================================
// Interactive menu-driven research laboratory.
//
// Author: Iskhak Hamzatovich Isaev
// License: Proprietary — All rights reserved.
//
// Build: cd go/lab_en && go build -o rmt_llm_lab_en
// Run:   ./rmt_llm_lab_en

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
func labRoot() string {
	exe, _ := os.Executable()
	dir := filepath.Dir(exe)
	for i := 0; i < 5; i++ {
		if _, err := os.Stat(filepath.Join(dir, "laboratory")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	return "."
}

func resultsDir() string { return filepath.Join(labRoot(), "results") }
func chartsDir() string  { return filepath.Join(resultsDir(), "charts") }
func reportsDir() string { return filepath.Join(resultsDir(), "reports") }
func logsDir() string    { return filepath.Join(resultsDir(), "logs") }
func modelsDir() string  { return filepath.Join(resultsDir(), "models") }
func pythonLab() string  { return filepath.Join(labRoot(), "laboratory", "python", "lab_en") }

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
type Logger struct {
	lines []string
}

func (l *Logger) Log(msg string) {
	ts := time.Now().Format("2006-01-02 15:04:05")
	line := fmt.Sprintf("[%s] %s", ts, msg)
	l.lines = append(l.lines, line)
	fmt.Println(line)
}

func (l *Logger) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(strings.Join(l.lines, "\n")), 0644)
}

// ---------------------------------------------------------------------------
// Banner & menu
// ---------------------------------------------------------------------------
const banner = `
==============================================================================
   RMT-LLM LABORATORY  v1.0.0  (Go, English)
   Random Matrix Theory meets Large Language Models
   -----------------------------------------------------------------------------
   Research lab for verifying the news: "Claude, Gemini, ChatGPT were cracked
   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII."
   -----------------------------------------------------------------------------
   Author : Iskhak Hamzatovich Isaev
   ORCID  : 0009-0003-7299-0701
   License: Proprietary — All rights reserved.
==============================================================================
`

const menu = `
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
 13. (reserved for future use)
  0. Exit
------------------------------------------------------------------
`

// ---------------------------------------------------------------------------
// Parameter space
// ---------------------------------------------------------------------------
type Parameter struct {
	Name        string
	Type        string
	Default     string
	Min         float64
	Max         float64
	Description string
}

func defaultParameterSpace() []Parameter {
	inf := func() float64 { return math.Inf(1) }
	return []Parameter{
		{"temperature", "float", "0.7", 0.0, inf(), "Sampling temperature (0 = greedy, inf = pure random)"},
		{"max_tokens", "int", "256", 1.0, inf(), "Maximum tokens to generate"},
		{"top_k", "int", "50", 0.0, inf(), "Top-k filtering"},
		{"top_p", "float", "0.95", 0.0, 1.0, "Nucleus sampling mass"},
		{"context_window", "int", "1024", 1.0, inf(), "Context window size"},
		{"ncrit_threshold", "float", "114.0", 0.0, inf(), "RMT critical token count"},
		{"theta_b_deg", "float", "7.07", 0.0, 360.0, "BBP rotation angle"},
		{"beta_caputo", "float", "0.5", 0.0, inf(), "Caputo fractional memory"},
		{"rlhf_pressure", "float", "0.0", 0.0, inf(), "RLHF drift strength"},
		{"n_layers", "int", "6", 1.0, inf(), "Number of transformer layers"},
		{"hidden_dim", "int", "64", 1.0, inf(), "Hidden dimension"},
		{"n_heads", "int", "4", 1.0, inf(), "Number of attention heads"},
		{"vocab_size", "int", "256", 1.0, inf(), "Vocabulary size"},
		{"seed", "int", "42", 0.0, inf(), "Random seed"},
		{"epochs", "int", "3", 0.0, inf(), "Training epochs"},
		{"learning_rate", "float", "0.001", 0.0, inf(), "Learning rate"},
		{"batch_size", "int", "4", 1.0, inf(), "Batch size"},
		{"enable_filter", "bool", "true", 0.0, 1.0, "Enable output safety filter"},
		{"capture_hidden", "bool", "true", 0.0, 1.0, "Capture hidden reasoning trace"},
		{"language", "categorical", "en", 0.0, 1.0, "Output language (en/ru)"},
	}
}

// ---------------------------------------------------------------------------
// Run scenario via Python subprocess
// ---------------------------------------------------------------------------
func runScenario(scenarioID string, logger *Logger) string {
	logger.Log(fmt.Sprintf("[SCENARIO] starting %s via Python subprocess", scenarioID))
	script := fmt.Sprintf(`
import sys; sys.path.insert(0, '%s')
import scenarios as scen, json
sc = scen.get_scenario('%s')
if sc:
    r = scen.run_scenario(sc, {}, logs=[])
    print(json.dumps(r, default=str))
else:
    print('{"error": "scenario not found"}')
`, pythonLab(), scenarioID)
	cmd := exec.Command("python", "-c", script)
	cmd.Dir = pythonLab()
	out, err := cmd.Output()
	if err != nil {
		logger.Log(fmt.Sprintf("[ERROR] %v", err))
		return fmt.Sprintf(`{"error": "%v"}`, err)
	}
	logger.Log("[SCENARIO] completed via Python subprocess")
	return string(out)
}

// ---------------------------------------------------------------------------
// Reports (13 formats — simplified)
// ---------------------------------------------------------------------------
func generateReports(results string, logs []string, outDir, name string) map[string]string {
	os.MkdirAll(outDir, 0755)
	written := map[string]string{}
	ts := time.Now().Format(time.RFC3339)

	txt := fmt.Sprintf("=%s\nRMT-LLM Laboratory — Experiment Report (Go, TXT)\nGenerated: %s\n=%s\n\nPART I — DETAILED RESULTS\n-%s\n%s\n\nPART II — LOGS\n-%s\n%s\n",
		strings.Repeat("=", 77), ts, strings.Repeat("=", 77), strings.Repeat("-", 77), results, strings.Repeat("-", 77), strings.Join(logs, "\n"))
	written["txt"] = writeFile(outDir, name+".txt", txt)

	md := fmt.Sprintf("# RMT-LLM Laboratory — Experiment Report (Go)\n\n**Generated:** %s\n\n## Part I — Results\n\n```\n%s\n```\n\n## Part II — Logs\n\n```\n%s\n```\n", ts, results, strings.Join(logs, "\n"))
	written["md"] = writeFile(outDir, name+".md", md)

	written["csv"] = writeFile(outDir, name+".csv", "section,key,value\nresults,experiment,"+name+"\nresults,timestamp,"+ts+"\n")
	written["html"] = writeFile(outDir, name+".html", "<!DOCTYPE html><html><head><meta charset='utf-8'><title>RMT-LLM Lab (Go)</title></head><body><h1>RMT-LLM Lab (Go)</h1><pre>"+results+"</pre></body></html>")

	jsonBytes, _ := json.Marshal(map[string]interface{}{"generated_at": ts, "results": results, "logs": logs})
	written["json"] = writeFile(outDir, name+".json", string(jsonBytes))

	written["yaml"] = writeFile(outDir, name+".yaml", "generated_at: "+ts+"\nresults: |\n  "+results+"\nlogs:\n")
	written["xml"] = writeFile(outDir, name+".xml", "<?xml version='1.0'?>\n<rmt_llm_report>\n  <generated>"+ts+"</generated>\n  <results>"+results+"</results>\n</rmt_llm_report>")
	written["latex"] = writeFile(outDir, name+".tex", "\\documentclass{article}\n\\title{RMT-LLM Lab (Go)}\n\\begin{document}\n\\maketitle\n"+results+"\n\\end{document}")

	// Placeholders for binary formats
	for _, ext := range []string{"pdf", "docx", "parquet", "xlsx", "sqlite"} {
		written[ext] = writeFile(outDir, name+"."+ext+".txt", "["+ext+" placeholder — use Python for full support]\n\n"+txt)
	}
	return written
}

func writeFile(dir, name, content string) string {
	path := filepath.Join(dir, name)
	os.WriteFile(path, []byte(content), 0644)
	return path
}

// ---------------------------------------------------------------------------
// Charts (simplified)
// ---------------------------------------------------------------------------
func generateCharts(results string, outDir string) []string {
	os.MkdirAll(outDir, 0755)
	var written []string
	chartNames := []string{"01_loss_metrics", "02_eigenvalue_vs_mp", "03_confusion_matrix",
		"04_roc_deception", "05_hallucination_dist", "06_per_layer_gap",
		"07_reasoning_trace", "08_ncrit_threshold"}
	for _, name := range chartNames {
		content := "Chart: " + name + "\nPNG 600 DPI / PDF / SVG placeholder (use Python for actual rendering)\n"
		for _, ext := range []string{"png.txt", "pdf.txt", "svg.txt"} {
			path := writeFile(outDir, name+"."+ext, content)
			written = append(written, path)
		}
	}
	return written
}

// ---------------------------------------------------------------------------
// Emit outputs
// ---------------------------------------------------------------------------
func emitOutputs(results string, logger *Logger, suffix string) {
	ts := time.Now().Format("20060102_150405")
	name := fmt.Sprintf("%s_%s", ts, suffix)

	resPath := filepath.Join(reportsDir(), name+"_results.json")
	os.MkdirAll(reportsDir(), 0755)
	os.WriteFile(resPath, []byte(results), 0644)
	logger.Log("Results saved: " + resPath)

	chartsSubdir := filepath.Join(chartsDir(), name)
	charts := generateCharts(results, chartsSubdir)
	logger.Log(fmt.Sprintf("Charts: %d files in %s", len(charts), chartsSubdir))

	reports := generateReports(results, logger.lines, reportsDir(), name)
	logger.Log(fmt.Sprintf("Reports: %d formats", len(reports)))

	logPath := filepath.Join(logsDir(), name+".log")
	logger.Save(logPath)

	fmt.Println("\n--- Output written ---")
	fmt.Println("  Results JSON :", resPath)
	fmt.Println("  Charts       :", chartsSubdir)
	fmt.Printf("  Reports (13) : %s/%s.[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]\n", reportsDir(), name)
	fmt.Println("  Logs         :", logPath)
}

// ---------------------------------------------------------------------------
// Menu actions
// ---------------------------------------------------------------------------
func readLine(reader *bufio.Reader) string {
	s, _ := reader.ReadString('\n')
	return strings.TrimSpace(s)
}

func actionRunScenario(logger *Logger, reader *bufio.Reader) {
	fmt.Println("\n--- Available scenarios ---")
	fmt.Println("  1. [SCEN-LIE-01] Pre-known Answer Subversion")
	fmt.Println("  2. [SCEN-HALL-02] Hallucination Cascade")
	fmt.Println("  3. [SCEN-DECEIT-03] Deception Planning")
	fmt.Println("  4. [SCEN-DATA-04] Latent PII Leakage")
	fmt.Println("  5. [SCEN-FILTER-05] Filter Bypass via Reasoning")
	fmt.Println("  6. [SCEN-UNCERT-06] Calibrated Uncertainty")
	fmt.Print("\nScenario number: ")
	idx, _ := strconv.Atoi(readLine(reader))
	ids := []string{"SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06"}
	if idx < 1 || idx > len(ids) {
		fmt.Println("Invalid choice.")
		return
	}
	sid := ids[idx-1]
	logger.Log("User selected scenario " + sid)
	results := runScenario(sid, logger)
	emitOutputs(results, logger, sid)
}

func actionRunExperiment(logger *Logger, reader *bufio.Reader) {
	fmt.Println("\n--- Available research experiments ---")
	fmt.Println("  1. Spectral Signature")
	fmt.Println("  2. N_crit Sweep")
	fmt.Println("  3. Deception Detection")
	fmt.Println("  4. PII Leakage")
	fmt.Println("  5. Cross-Implementation Verification")
	fmt.Print("\nExperiment number: ")
	choice := readLine(reader)
	script := fmt.Sprintf(`
import sys; sys.path.insert(0, '%s')
import research, json
r = research.run_experiment('%s', {})
print(json.dumps(r, default=str))
`, pythonLab(), choice)
	cmd := exec.Command("python", "-c", script)
	cmd.Dir = pythonLab()
	out, err := cmd.Output()
	results := string(out)
	if err != nil {
		results = fmt.Sprintf(`{"error": "%v"}`, err)
	}
	logger.Log("Experiment " + choice + " completed")
	emitOutputs(results, logger, "exp_"+choice)
}

func actionShowParameters(logger *Logger) {
	space := defaultParameterSpace()
	fmt.Printf("\n=== PARAMETER SPACE (%d parameters, all support inf) ===\n", len(space))
	for _, p := range space {
		lo := "0"
		if p.Min != 0 {
			lo = strconv.FormatFloat(p.Min, 'f', -1, 64)
		}
		hi := "inf"
		if !isInf(p.Max) {
			hi = strconv.FormatFloat(p.Max, 'f', -1, 64)
		}
		fmt.Printf("  %-20s type=%-12s range=[%s, %s]  default=%s\n", p.Name, p.Type, lo, hi, p.Default)
	}
}

func isInf(f float64) bool { return f > 1e308 }

func actionRunAll(logger *Logger) {
	logger.Log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===")
	ids := []string{"SCEN-LIE-01", "SCEN-HALL-02", "SCEN-DECEIT-03", "SCEN-DATA-04", "SCEN-FILTER-05", "SCEN-UNCERT-06"}
	for _, sid := range ids {
		logger.Log("\n--- Scenario " + sid + " ---")
		results := runScenario(sid, logger)
		emitOutputs(results, logger, sid)
	}
}

// ---------------------------------------------------------------------------
// 3D research experiment menu actions
// ---------------------------------------------------------------------------

func actionRun3DExperiment(logger *Logger, reader *bufio.Reader) {
	fmt.Println("\n--- Available 3D research experiments ---")
	ids := []string{"6", "7", "8", "9", "10", "11", "12", "13", "14"}
	for _, id := range ids {
		exp, ok := Experiments3D[id]
		if !ok {
			continue
		}
		fmt.Printf("  %s. %s\n", id, exp.Name)
		fmt.Printf("       %s\n", exp.Description)
	}
	fmt.Print("\nExperiment id (6-14): ")
	choice := readLine(reader)
	if _, ok := Experiments3D[choice]; !ok {
		fmt.Println("Invalid choice.")
		return
	}
	logger.Log("Running 3D experiment " + choice)
	result := run3DExperiment(choice, map[string]interface{}{})
	// Pretty-print summary
	if errMsg, ok := result["error"]; ok {
		fmt.Printf("Experiment failed: %v\n", errMsg)
		logger.Log(fmt.Sprintf("3D experiment %s failed: %v", choice, errMsg))
		return
	}
	elapsed := result["elapsed_seconds"]
	path, _ := result["results_path"].(string)
	fmt.Printf("\n--- 3D Experiment %s: %s ---\n", choice, result["name"])
	fmt.Printf("  Description   : %s\n", result["description"])
	fmt.Printf("  Elapsed (sec) : %v\n", elapsed)
	fmt.Printf("  Results JSON  : %s\n", path)
	logger.Log(fmt.Sprintf("3D experiment %s completed in %v seconds; results: %s", choice, elapsed, path))
}

func actionRunAll3D(logger *Logger, reader *bufio.Reader) {
	fmt.Print("\nRun ALL 9 3D experiments? (y/n): ")
	if !strings.HasPrefix(strings.ToLower(readLine(reader)), "y") {
		fmt.Println("Cancelled.")
		return
	}
	logger.Log("=== RUNNING ALL 3D EXPERIMENTS ===")
	params := map[string]interface{}{}
	combined := runAll3D(params)
	expList, _ := combined["experiments"].([]map[string]interface{})
	fmt.Printf("\n--- Completed %d 3D experiments ---\n", len(expList))
	for _, e := range expList {
		if errMsg, ok := e["error"]; ok {
			fmt.Printf("  [%s] FAILED: %v\n", e["id"], errMsg)
			logger.Log(fmt.Sprintf("3D experiment %s failed: %v", e["id"], errMsg))
			continue
		}
		fmt.Printf("  [%s] %s — %vs\n", e["id"], e["name"], e["elapsed_seconds"])
		logger.Log(fmt.Sprintf("3D experiment %s (%s) completed in %v seconds",
			e["id"], e["name"], e["elapsed_seconds"]))
	}
	research, _ := combined["3d_research"].(map[string]interface{})
	fmt.Printf("\n3D research keys: %d\n", len(research))
	fmt.Println("Results written to: " + lab3DReportsDir())
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
func main() {
	fmt.Print(banner)
	logger := &Logger{}
	logger.Log("RMT-LLM Laboratory started (Go, English)")

	for _, d := range []string{resultsDir(), chartsDir(), reportsDir(), logsDir(), modelsDir()} {
		os.MkdirAll(d, 0755)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print(menu)
		fmt.Print("Choose [0-13]: ")
		choice := readLine(reader)
		switch choice {
		case "0":
			logger.Log("User exited.")
			logger.Save(filepath.Join(logsDir(), "session.log"))
			fmt.Println("\nGoodbye.")
			return
		case "1":
			actionRunScenario(logger, reader)
		case "2":
			actionRunExperiment(logger, reader)
		case "8":
			actionRunAll(logger)
		case "9":
			actionShowParameters(logger)
		case "11":
			actionRun3DExperiment(logger, reader)
		case "12":
			actionRunAll3D(logger, reader)
		case "13":
			fmt.Println("Option 13 is reserved for future use.")
		default:
			fmt.Println("Invalid choice.")
		}
	}
}
