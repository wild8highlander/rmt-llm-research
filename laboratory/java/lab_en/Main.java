/*
 * Main.java — RMT-LLM Laboratory (Java, English version)
 * =======================================================
 * Interactive menu-driven research laboratory.
 *
 * Author: Iskhak Hamzatovich Isaev
 * License: Proprietary — All rights reserved.
 */

package com.rmt.llm.lab.en;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;
import java.util.stream.*;

public class Main {

    static final String LAB_ROOT = Paths.get(System.getProperty("user.dir"),
            "..", "..").toAbsolutePath().toString();
    static final String RESULTS_DIR = Paths.get(LAB_ROOT, "results").toString();
    static final String CHARTS_DIR = Paths.get(RESULTS_DIR, "charts").toString();
    static final String REPORTS_DIR = Paths.get(RESULTS_DIR, "reports").toString();
    static final String LOGS_DIR = Paths.get(RESULTS_DIR, "logs").toString();
    static final String MODELS_DIR = Paths.get(RESULTS_DIR, "models").toString();

    static List<String> logLines = new ArrayList<>();
    static Scanner scanner = new Scanner(System.in);

    static final String BANNER =
        "==============================================================================\n" +
        "   RMT-LLM LABORATORY  v1.0.0  (Java, English)\n" +
        "   Random Matrix Theory meets Large Language Models\n" +
        "   -----------------------------------------------------------------------------\n" +
        "   Research lab for verifying the news: \"Claude, Gemini, ChatGPT were cracked\n" +
        "   — hidden reasoning chains exposed. Models lie, hallucinate, and store PII.\"\n" +
        "   -----------------------------------------------------------------------------\n" +
        "   Author : Iskhak Hamzatovich Isaev\n" +
        "   ORCID  : 0009-0003-7299-0701\n" +
        "   License: Proprietary — All rights reserved.\n" +
        "==============================================================================";

    static final String MENU =
        "---------------------------- MAIN MENU ----------------------------\n" +
        "  1. Run pre-defined scenario (synthetic NN)\n" +
        "  2. Run research experiment (measurement system)\n" +
        "  3. Custom launch — interactive wizard (infinite params)\n" +
        "  4. Custom launch — JSON config file\n" +
        "  5. Download model from registry (HuggingFace / ONNX)\n" +
        "  6. Generate reports only (from existing results.json)\n" +
        "  7. Generate charts only (from existing results.json)\n" +
        "  8. Run ALL scenarios + experiments -> full report\n" +
        "  9. Show parameter space\n" +
        " 10. Cross-implementation verification\n" +
        " 11. Run 3D research experiment (single)\n" +
        " 12. Run ALL 3D experiments -> 3D charts + reports\n" +
        "  0. Exit\n" +
        "------------------------------------------------------------------";

    public static void main(String[] args) {
        System.out.println(BANNER);
        log("RMT-LLM Laboratory started (Java, English)");
        try {
            Files.createDirectories(Paths.get(RESULTS_DIR));
            Files.createDirectories(Paths.get(CHARTS_DIR));
            Files.createDirectories(Paths.get(REPORTS_DIR));
            Files.createDirectories(Paths.get(LOGS_DIR));
            Files.createDirectories(Paths.get(MODELS_DIR));
        } catch (IOException e) {
            log("[ERROR] could not create output dirs: " + e.getMessage());
        }

        while (true) {
            System.out.println(MENU);
            System.out.print("Choose [0-12]: ");
            String choice = scanner.nextLine().trim();
            try {
                switch (choice) {
                    case "0": log("User exited."); saveLog("session.log"); System.out.println("\nGoodbye."); return;
                    case "1": runScenario(); break;
                    case "2": runExperiment(); break;
                    case "3": customWizard(); break;
                    case "4": customConfig(); break;
                    case "5": downloadModel(); break;
                    case "8": runAll(); break;
                    case "9": showParameters(); break;
                    case "10": crossVerify(); break;
                    case "11": run3DExperimentMenu(); break;
                    case "12": runAll3DMenu(); break;
                    default: System.out.println("Invalid choice.");
                }
            } catch (Exception e) {
                log("[ERROR] " + e.getClass().getSimpleName() + ": " + e.getMessage());
                System.out.println("\n[ERROR] " + e.getMessage());
            }
        }
    }

    static void log(String msg) {
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String line = "[" + ts + "] " + msg;
        logLines.add(line);
        System.out.println(line);
    }

    static void saveLog(String filename) {
        try {
            Files.createDirectories(Paths.get(LOGS_DIR));
            Files.write(Paths.get(LOGS_DIR, filename), logLines);
        } catch (IOException e) {
            System.err.println("Failed to save log: " + e.getMessage());
        }
    }

    // -----------------------------------------------------
    // Menu actions
    // -----------------------------------------------------
    static void runScenario() throws Exception {
        List<Map<String, Object>> scens = Scenarios.listScenarios();
        System.out.println("\n--- Available scenarios ---");
        for (int i = 0; i < scens.size(); i++) {
            Map<String, Object> s = scens.get(i);
            System.out.printf("  %2d. [%s] %s%n", i + 1, s.get("id"), s.get("name"));
            System.out.printf("       Expected: %s%n",
                    s.getOrDefault("expected_behavior", "unknown"));
        }
        System.out.print("\nScenario number: ");
        int idx = Integer.parseInt(scanner.nextLine().trim()) - 1;
        if (idx < 0 || idx >= scens.size()) { System.out.println("Invalid choice."); return; }
        Map<String, Object> chosen = scens.get(idx);
        log("User selected scenario " + chosen.get("id"));

        Map<String, Object> params = Parameters.defaults();
        System.out.print("Use interactive parameter wizard? (y/N): ");
        if (scanner.nextLine().trim().toLowerCase().matches("y|yes")) {
            params = Parameters.interactiveWizard(scanner);
        }

        Map<String, Object> results = Scenarios.runScenario(chosen, params, logLines);
        log("Scenario completed. Match rate: " + ((Map<?, ?>) results.get("metrics")).get("match_rate"));
        emitOutputs(results, (String) chosen.get("id"));
    }

    static void runExperiment() throws Exception {
        System.out.println("\n--- Available research experiments ---");
        for (Map.Entry<String, Map<String, String>> e : Research.EXPERIMENTS.entrySet()) {
            System.out.printf("  %s. %s%n   %s%n", e.getKey(),
                    e.getValue().get("name"), e.getValue().get("description"));
        }
        System.out.print("\nExperiment number: ");
        String choice = scanner.nextLine().trim();
        if (!Research.EXPERIMENTS.containsKey(choice)) { System.out.println("Invalid choice."); return; }
        log("User selected experiment " + choice);
        Map<String, Object> params = Parameters.defaults();
        Map<String, Object> results = Research.runExperiment(choice, params, logLines);
        log("Experiment completed in " + results.get("elapsed_seconds") + "s");
        emitOutputs(results, "exp_" + choice);
    }

    static void customWizard() throws Exception {
        log("Starting custom launch (interactive wizard)");
        Map<String, Object> params = Parameters.interactiveWizard(scanner);
        log("Custom params set");
        System.out.println("\nWhat to run?");
        System.out.println("  1. Run a scenario");
        System.out.println("  2. Run a research experiment");
        System.out.println("  3. Just compute spectral signature of TinyGPT");
        System.out.print("Choice: ");
        String sub = scanner.nextLine().trim();
        if ("1".equals(sub)) {
            List<Map<String, Object>> scens = Scenarios.listScenarios();
            for (int i = 0; i < scens.size(); i++)
                System.out.printf("  %d. %s%n", i + 1, scens.get(i).get("name"));
            System.out.print("Scenario number: ");
            int idx = Integer.parseInt(scanner.nextLine().trim()) - 1;
            if (idx >= 0 && idx < scens.size()) {
                Map<String, Object> results = Scenarios.runScenario(scens.get(idx), params, logLines);
                emitOutputs(results, "custom_scen_" + scens.get(idx).get("id"));
            }
        } else if ("2".equals(sub)) {
            for (String k : Research.EXPERIMENTS.keySet())
                System.out.println("  " + k + ". " + Research.EXPERIMENTS.get(k).get("name"));
            System.out.print("Experiment number: ");
            String k = scanner.nextLine().trim();
            if (Research.EXPERIMENTS.containsKey(k)) {
                Map<String, Object> results = Research.runExperiment(k, params, logLines);
                emitOutputs(results, "custom_exp_" + k);
            }
        } else if ("3".equals(sub)) {
            Map<String, Object> results = Research.runExperiment("1", params, logLines);
            emitOutputs(results, "custom_spectral");
        }
    }

    static void customConfig() throws Exception {
        System.out.print("Path to JSON config: ");
        String path = scanner.nextLine().trim();
        if (!Files.exists(Paths.get(path))) { System.out.println("File not found: " + path); return; }
        String content = new String(Files.readAllBytes(Paths.get(path)));
        // Simple JSON parsing (no external deps)
        Map<String, Object> cfg = SimpleJson.parse(content);
        log("Loaded config from " + path);
        Map<String, Object> params = (Map<String, Object>) cfg.getOrDefault("parameters", cfg);
        if (cfg.containsKey("scenario_id")) {
            Map<String, Object> sc = Scenarios.getScenario((String) cfg.get("scenario_id"));
            if (sc != null) {
                Map<String, Object> results = Scenarios.runScenario(sc, params, logLines);
                emitOutputs(results, (String) cfg.get("scenario_id"));
            }
        } else {
            Map<String, Object> results = Research.runExperiment("1", params, logLines);
            emitOutputs(results, "cfg_spectral");
        }
    }

    static void downloadModel() throws Exception {
        String pick = ModelDownloader.interactivePick(scanner);
        log("User picked model: " + pick);
        if ("tiny-gpt-local".equals(pick)) {
            log("Local model — saving TinyGPT weights");
            String path = Paths.get(MODELS_DIR, "tiny_gpt_local_java.ser").toString();
            TinyGPT model = new TinyGPT();
            TinyGPT.saveWeights(model, path);
            System.out.println("\nLocal TinyGPT weights saved to:\n  " + path);
            return;
        }
        Map<String, Object> rep = ModelDownloader.fetchModel(pick, MODELS_DIR);
        if (Boolean.TRUE.equals(rep.get("ok"))) {
            log("Downloaded " + pick + ": " + rep.get("bytes") + " bytes");
            System.out.println("\nDownloaded to: " + rep.get("dest"));
        } else {
            log("Download failed: " + rep.get("error"));
            System.out.println("\nDownload failed: " + rep.get("error"));
        }
    }

    static void runAll() throws Exception {
        log("=== RUNNING ALL SCENARIOS + EXPERIMENTS ===");
        Map<String, Object> params = Parameters.defaults();
        for (Map<String, Object> sc : Scenarios.listScenarios()) {
            log("\n--- Scenario " + sc.get("id") + " ---");
            try {
                Map<String, Object> r = Scenarios.runScenario(sc, params, logLines);
                emitOutputs(r, (String) sc.get("id"));
            } catch (Exception e) { log("  [ERROR] " + e.getMessage()); }
        }
        for (String eid : Research.EXPERIMENTS.keySet()) {
            log("\n--- Experiment " + eid + " ---");
            try {
                Map<String, Object> r = Research.runExperiment(eid, params, logLines);
                emitOutputs(r, "exp_" + eid);
            } catch (Exception e) { log("  [ERROR] " + e.getMessage()); }
        }
    }

    static void showParameters() {
        List<Map<String, Object>> space = Parameters.defaultSpace();
        System.out.println("\n=== PARAMETER SPACE (" + space.size() + " parameters, all support inf) ===");
        for (Map<String, Object> p : space) {
            Object mn = p.get("min"); Object mx = p.get("max");
            String lo = "0".equals(String.valueOf(mn)) ? "0" : String.valueOf(mn);
            String hi = "Infinity".equals(String.valueOf(mx)) ? "inf" : String.valueOf(mx);
            System.out.printf("  %-20s type=%-12s range=[%s, %s]  default=%s%n",
                    p.get("name"), p.get("type"), lo, hi, p.get("default"));
        }
    }

    static void crossVerify() throws Exception {
        log("Cross-implementation verification");
        Map<String, Object> params = Parameters.defaults();
        Map<String, Object> results = Research.runExperiment("5", params, logLines);
        log("MP upper theory: " + results.get("mp_upper_theory") +
                ", empirical: " + results.get("mp_upper_empirical"));
        emitOutputs(results, "cross_verify");
    }

    // -----------------------------------------------------
    // 3D research experiments (v1.4.0)
    // -----------------------------------------------------
    static Map<String, Object> default3DParams() {
        Map<String, Object> params = Parameters.defaults();
        params.put("hessian_grid_size", 24);
        params.put("trajectory_points", 64);
        params.put("spectral_surface_layers", 6);
        params.put("pca_components", 3);
        params.put("attention_flow_3d_resolution", 32);
        params.put("parameter_space_grid", 16);
        params.put("curvature_neighbors", 8);
        params.put("elevation_3d", 30);
        params.put("azimuth_3d", 45);
        params.put("color_map_3d", "viridis");
        return params;
    }

    static void run3DExperimentMenu() throws Exception {
        System.out.println("\n--- Available 3D research experiments (v1.4.0) ---");
        for (Map.Entry<String, Research3D.Experiment3D> e : Research3D.EXPERIMENTS_3D.entrySet()) {
            Research3D.Experiment3D exp = e.getValue();
            System.out.printf("  %s. %s%n      %s%n", e.getKey(), exp.name, exp.description);
        }
        System.out.print("\n3D experiment number: ");
        String choice = scanner.nextLine().trim();
        if (!Research3D.EXPERIMENTS_3D.containsKey(choice)) {
            System.out.println("Invalid choice.");
            return;
        }
        log("User selected 3D experiment " + choice + ": "
                + Research3D.EXPERIMENTS_3D.get(choice).name);
        Map<String, Object> params = default3DParams();
        System.out.print("Tweak 3D parameters via wizard? (y/N): ");
        if (scanner.nextLine().trim().toLowerCase().matches("y|yes")) {
            params.putAll(Parameters.interactiveWizard(scanner));
        }
        Map<String, Object> results = Research3D.run3DExperiment(choice, params);
        log("3D experiment completed in " + results.get("elapsed_seconds") + "s");
        // The Research3D class already wrote its own JSON; emit full report bundle too.
        emitOutputs(results, "3d_exp_" + choice);
    }

    static void runAll3DMenu() throws Exception {
        log("=== RUNNING ALL 3D EXPERIMENTS (v1.4.0) ===");
        Map<String, Object> params = default3DParams();
        Map<String, Object> combined = Research3D.runAll3D(params);
        List<?> exps = (List<?>) combined.get("experiments");
        int ok = 0, err = 0;
        for (Object e : exps) {
            Map<?, ?> m = (Map<?, ?>) e;
            if (m.containsKey("error")) err++; else ok++;
        }
        log("3D experiments done: " + ok + " succeeded, " + err + " failed");
        combined.put("experiment_name", "ALL_3D_EXPERIMENTS");
        combined.put("language", "java");
        combined.put("version", "en");
        combined.put("timestamp", LocalDateTime.now().toString());
        emitOutputs(combined, "ALL_3D");
    }

    // -----------------------------------------------------
    // Emit outputs
    // -----------------------------------------------------
    @SuppressWarnings("unchecked")
    static void emitOutputs(Map<String, Object> results, String suffix) throws Exception {
        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String name = ts + "_" + suffix;
        String resPath = Paths.get(REPORTS_DIR, name + "_results.json").toString();
        Files.write(Paths.get(resPath), SimpleJson.stringify(results, 2).getBytes());
        log("Results saved: " + resPath);

        String chartsDir = Paths.get(CHARTS_DIR, name).toString();
        Files.createDirectories(Paths.get(chartsDir));
        Map<String, List<String>> writtenCharts = Charts.generateAll(results, chartsDir);
        int nCharts = writtenCharts.values().stream().mapToInt(List::size).sum();
        log("Charts: " + nCharts + " files in " + chartsDir);

        Map<String, String> writtenReports = Reports.generateAll(results, logLines, REPORTS_DIR, name);
        log("Reports: " + writtenReports.size() + " formats");

        String logPath = Paths.get(LOGS_DIR, name + ".log").toString();
        Files.write(Paths.get(logPath), logLines);
        log("Log saved: " + logPath);

        System.out.println("\n--- Output written ---");
        System.out.println("  Results JSON : " + resPath);
        System.out.println("  Charts       : " + chartsDir);
        System.out.println("  Reports (13) : " + REPORTS_DIR + "/" + name + ".[txt|md|csv|html|json|pdf|docx|yaml|xml|tex|parquet|xlsx|sqlite]");
        System.out.println("  Logs         : " + logPath);
    }
}
