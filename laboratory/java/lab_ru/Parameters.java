/*
 * Parameters.java — Infinite parameter system (Java, EN)
 */

package com.rmt.llm.lab.ru;

import java.util.*;
import java.io.*;

public class Parameters {

    public static List<Map<String, Object>> defaultSpace() {
        List<Map<String, Object>> space = new ArrayList<>();
        space.add(param("temperature", "float", 0.7, 0.0, Double.POSITIVE_INFINITY, "Sampling temperature (0 = greedy, inf = pure random)"));
        space.add(param("max_tokens", "int", 256, 1, Double.POSITIVE_INFINITY, "Maximum tokens to generate"));
        space.add(param("top_k", "int", 50, 0, Double.POSITIVE_INFINITY, "Top-k filtering"));
        space.add(param("top_p", "float", 0.95, 0.0, 1.0, "Nucleus sampling mass"));
        space.add(param("context_window", "int", 1024, 1, Double.POSITIVE_INFINITY, "Context window size"));
        space.add(param("ncrit_threshold", "float", 114.0, 0.0, Double.POSITIVE_INFINITY, "RMT critical token count"));
        space.add(param("theta_b_deg", "float", 7.07, 0.0, 360.0, "BBP rotation angle"));
        space.add(param("beta_caputo", "float", 0.5, 0.0, Double.POSITIVE_INFINITY, "Caputo fractional memory parameter"));
        space.add(param("rlhf_pressure", "float", 0.0, 0.0, Double.POSITIVE_INFINITY, "RLHF drift strength"));
        space.add(param("n_layers", "int", 6, 1, Double.POSITIVE_INFINITY, "Number of transformer layers"));
        space.add(param("hidden_dim", "int", 64, 1, Double.POSITIVE_INFINITY, "Hidden dimension"));
        space.add(param("n_heads", "int", 4, 1, Double.POSITIVE_INFINITY, "Number of attention heads"));
        space.add(param("vocab_size", "int", 256, 1, Double.POSITIVE_INFINITY, "Vocabulary size"));
        space.add(param("seed", "int", 42, 0, Double.POSITIVE_INFINITY, "Random seed"));
        space.add(param("epochs", "int", 3, 0, Double.POSITIVE_INFINITY, "Training epochs"));
        space.add(param("learning_rate", "float", 1e-3, 0.0, Double.POSITIVE_INFINITY, "Learning rate"));
        space.add(param("batch_size", "int", 4, 1, Double.POSITIVE_INFINITY, "Batch size"));
        space.add(param("enable_filter", "bool", true, "Enable output safety filter"));
        space.add(param("capture_hidden", "bool", true, "Capture hidden reasoning trace"));
        space.add(param("language", "categorical", "en", "Output language",
                Arrays.asList("en", "ru")));
        return space;
    }

    private static Map<String, Object> param(String name, String type, Object def,
                                              String desc) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("name", name); p.put("type", type); p.put("default", def);
        p.put("min", 0.0); p.put("max", Double.POSITIVE_INFINITY);
        p.put("description", desc);
        return p;
    }

    private static Map<String, Object> param(String name, String type, Object def,
                                              double min, double max, String desc) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("name", name); p.put("type", type); p.put("default", def);
        p.put("min", min); p.put("max", max);
        p.put("description", desc);
        return p;
    }

    private static Map<String, Object> param(String name, String type, Object def,
                                              String desc, List<Object> choices) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("name", name); p.put("type", type); p.put("default", def);
        p.put("min", 0.0); p.put("max", Double.POSITIVE_INFINITY);
        p.put("choices", choices); p.put("description", desc);
        return p;
    }

    public static Map<String, Object> defaults() {
        Map<String, Object> d = new LinkedHashMap<>();
        for (Map<String, Object> p : defaultSpace())
            d.put((String) p.get("name"), p.get("default"));
        return d;
    }

    public static Map<String, Object> interactiveWizard(Scanner scanner) {
        System.out.println("\n=== INTERACTIVE PARAMETER WIZARD ===");
        System.out.println("Enter values for each parameter. Press <Enter> to accept the default.");
        System.out.println("Numeric bounds support 'inf' for infinity. Range [0, inf) by default.\n");
        Map<String, Object> values = new LinkedHashMap<>();
        for (Map<String, Object> p : defaultSpace()) {
            String name = (String) p.get("name");
            String type = (String) p.get("type");
            Object def = p.get("default");
            while (true) {
                String hint = "[default=" + def + "]";
                if ("categorical".equals(type)) hint += " choices=" + p.get("choices");
                else if ("bool".equals(type)) hint += " (y/n)";
                else {
                    String lo = "0.0".equals(String.valueOf(p.get("min"))) ? "0" : String.valueOf(p.get("min"));
                    String hi = "Infinity".equals(String.valueOf(p.get("max"))) ? "inf" : String.valueOf(p.get("max"));
                    hint += " range=[" + lo + ", " + hi + "]";
                }
                System.out.print("  " + name + " " + hint + ": ");
                String raw = scanner.nextLine().trim();
                try {
                    if (raw.isEmpty()) { values.put(name, def); break; }
                    if ("float".equals(type) && raw.toLowerCase().matches("inf|\\+inf|infinity")) {
                        values.put(name, Double.POSITIVE_INFINITY); break;
                    }
                    if ("bool".equals(type)) {
                        values.put(name, raw.toLowerCase().matches("1|true|yes|y")); break;
                    }
                    if ("int".equals(type)) { values.put(name, Integer.parseInt(raw)); break; }
                    if ("float".equals(type)) { values.put(name, Double.parseDouble(raw)); break; }
                    values.put(name, raw); break;
                } catch (Exception e) {
                    System.out.println("    [ERROR] " + e.getMessage() + ". Try again.");
                }
            }
        }
        System.out.println();
        return values;
    }
}
