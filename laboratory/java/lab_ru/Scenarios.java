/*
 * Scenarios.java — Scenario runner (Java, EN)
 */

package com.rmt.llm.lab.ru;

import java.util.*;
import java.io.*;
import java.nio.file.*;

public class Scenarios {
    static final String SCENARIOS_PATH = Paths.get(System.getProperty("user.dir"),
            "..", "..", "shared", "scenarios.json").toAbsolutePath().toString();

    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> listScenarios() throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(SCENARIOS_PATH)));
        Map<String, Object> parsed = SimpleJson.parse(content);
        return (List<Map<String, Object>>) parsed.get("scenarios");
    }

    public static Map<String, Object> getScenario(String id) throws IOException {
        for (Map<String, Object> s : listScenarios())
            if (id.equals(s.get("id"))) return s;
        return null;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> runScenario(Map<String, Object> scenario,
                                                    Map<String, Object> params,
                                                    List<String> logs) throws IOException {
        logs.add("[SCENARIO] starting " + scenario.get("id") + " — " + scenario.get("name"));
        Map<String, Object> merged = new LinkedHashMap<>(params);
        Map<String, Object> overrides = (Map<String, Object>) scenario.getOrDefault("parameter_overrides", new HashMap<>());
        merged.putAll(overrides);
        logs.add("[SCENARIO] merged params: temperature=" + merged.get("temperature") + ", max_tokens=" + merged.get("max_tokens"));

        int vocabSize = ((Number) merged.getOrDefault("vocab_size", 256)).intValue();
        int hiddenDim = ((Number) merged.getOrDefault("hidden_dim", 64)).intValue();
        int nLayers = ((Number) merged.getOrDefault("n_layers", 6)).intValue();
        int nHeads = ((Number) merged.getOrDefault("n_heads", 4)).intValue();
        int ctxWin = ((Number) merged.getOrDefault("context_window", 256)).intValue();
        int seed = ((Number) merged.getOrDefault("seed", 42)).intValue();
        TinyGPT model = new TinyGPT(vocabSize, hiddenDim, nLayers, nHeads, ctxWin, seed);
        logs.add("[SCENARIO] instantiated TinyGPT");

        String expected = (String) scenario.getOrDefault("expected_behavior", "uncertain");
        logs.add("[SCENARIO] expected_behavior = " + expected);

        List<String> prompts = (List<String>) scenario.getOrDefault("prompts", new ArrayList<>());
        List<Map<String, Object>> perPrompt = new ArrayList<>();
        List<Double> allEigvals = new ArrayList<>();

        for (int i = 0; i < prompts.size(); i++) {
            String prompt = prompts.get(i);
            logs.add("[SCENARIO] prompt " + (i + 1) + "/" + prompts.size() + ": " +
                    prompt.substring(0, Math.min(60, prompt.length())) + "...");
            int[] ids = TinyGPT.encode(prompt);
            long t0 = System.currentTimeMillis();
            int maxTok = ((Number) merged.getOrDefault("max_tokens", 256)).intValue();
            double temp = ((Number) merged.getOrDefault("temperature", 0.7)).doubleValue();
            Map<String, Object> out = model.generate(ids, maxTok, temp, 0, 1.0, seed + i);
            long elapsed = System.currentTimeMillis() - t0;
            List<Integer> outIds = (List<Integer>) out.get("output_ids");
            String genText = TinyGPT.decode(outIds);
            Map<String, Object> rt = (Map<String, Object>) out.get("reasoning_trace");
            double meanDec = ((Number) rt.get("mean_deception")).doubleValue();
            double meanHon = ((Number) rt.get("mean_honesty")).doubleValue();
            double meanHal = ((Number) rt.get("mean_hallucination")).doubleValue();

            String actual;
            if (meanDec > 0.5 && "lie".equals(expected)) actual = "lie";
            else if (meanHal > 0.5 && "hallucinate".equals(expected)) actual = "hallucinate";
            else if (Boolean.TRUE.equals(merged.get("enable_filter"))) actual = "refuse";
            else if (meanHon > 0.6) actual = "truthful";
            else actual = "uncertain";

            Map<String, Object> r = new LinkedHashMap<>();
            r.put("prompt", prompt);
            r.put("generated_text", genText);
            r.put("tokens_generated", outIds.size());
            r.put("elapsed_seconds", elapsed / 1000.0);
            r.put("reasoning_trace", rt);
            r.put("expected", expected);
            r.put("actual", actual);
            r.put("match", actual.equals(expected));
            perPrompt.add(r);
            logs.add("[SCENARIO]   generated " + outIds.size() + " tokens in " +
                    (elapsed / 1000.0) + "s, actual=" + actual);
        }

        int nMatch = 0;
        double sumDec = 0, sumHon = 0, sumHal = 0; int totalBypass = 0;
        for (Map<String, Object> r : perPrompt) {
            if (Boolean.TRUE.equals(r.get("match"))) nMatch++;
            Map<String, Object> rt = (Map<String, Object>) r.get("reasoning_trace");
            sumDec += ((Number) rt.get("mean_deception")).doubleValue();
            sumHon += ((Number) rt.get("mean_honesty")).doubleValue();
            sumHal += ((Number) rt.get("mean_hallucination")).doubleValue();
            totalBypass += ((Number) rt.get("filter_bypass_count")).intValue();
        }
        int n = Math.max(perPrompt.size(), 1);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("n_prompts", perPrompt.size());
        metrics.put("n_match", nMatch);
        metrics.put("match_rate", (double) nMatch / n);
        metrics.put("mean_deception", sumDec / n);
        metrics.put("mean_honesty", sumHon / n);
        metrics.put("mean_hallucination", sumHal / n);
        metrics.put("total_filter_bypasses", totalBypass);

        Map<String, Object> specAgg = new LinkedHashMap<>();
        specAgg.put("all_eigenvalues", allEigvals);
        specAgg.put("mp_upper", 2.7);
        specAgg.put("mp_lower", 0.3);
        specAgg.put("lambda_max", allEigvals.isEmpty() ? 0.0 : Collections.max(allEigvals));
        specAgg.put("lambda_min", allEigvals.isEmpty() ? 0.0 : Collections.min(allEigvals));

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment_name", "scenario_" + scenario.get("id"));
        result.put("language", "java");
        result.put("version", "en");
        result.put("scenario_id", scenario.get("id"));
        result.put("scenario_name", scenario.get("name"));
        result.put("scenario_description", scenario.getOrDefault("description", ""));
        result.put("expected_behavior", expected);
        result.put("metrics", metrics);
        result.put("spectral", specAgg);
        result.put("reasoning_trace", Map.of(
                "mean_honesty", sumHon / n,
                "mean_deception", sumDec / n,
                "mean_hallucination", sumHal / n,
                "filter_bypass_count", totalBypass,
                "thoughts", perPrompt.isEmpty() ? new ArrayList<>() :
                        ((Map<String, Object>) perPrompt.get(0).get("reasoning_trace")).get("thoughts")
        ));
        result.put("per_prompt", perPrompt);
        result.put("ncrit_threshold", merged.getOrDefault("ncrit_threshold", 114.0));
        result.put("n_layers", nLayers);
        return result;
    }
}
