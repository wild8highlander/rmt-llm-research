/*
 * ModelDownloader.java — Model registry downloader (Java, EN)
 */

package com.rmt.llm.lab.ru;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.net.*;

public class ModelDownloader {
    static final String REGISTRY_PATH = Paths.get(System.getProperty("user.dir"),
            "..", "..", "shared", "model_registry.json").toAbsolutePath().toString();

    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> listModels() throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(REGISTRY_PATH)));
        return (List<Map<String, Object>>) SimpleJson.parse(content).get("models");
    }

    public static Map<String, Object> getModel(String id) throws IOException {
        for (Map<String, Object> m : listModels())
            if (id.equals(m.get("id"))) return m;
        return null;
    }

    public static Map<String, Object> fetchModel(String id, String destDir) throws IOException {
        Map<String, Object> model = getModel(id);
        if (model == null) return Map.of("ok", false, "error", "Unknown model_id: " + id);
        if ("local".equals(model.get("source")))
            return Map.of("ok", true, "model", model, "local", true);
        Files.createDirectories(Paths.get(destDir));
        String url = (String) model.get("url");
        String filename = url.substring(url.lastIndexOf('/') + 1);
        if (filename.isEmpty()) filename = id + ".bin";
        String dest = Paths.get(destDir, filename).toString();
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("url", url); info.put("dest", dest);
        info.put("ok", false); info.put("bytes", 0);
        try {
            URL u = new URL(url);
            try (InputStream in = u.openStream(); OutputStream out = Files.newOutputStream(Paths.get(dest))) {
                byte[] buf = new byte[64 * 1024];
                int n;
                while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            }
            info.put("bytes", Files.size(Paths.get(dest)));
            info.put("ok", true);
        } catch (Exception e) {
            info.put("error", e.getMessage());
        }
        info.put("model", model);
        return info;
    }

    public static String interactivePick(Scanner scanner) throws IOException {
        List<Map<String, Object>> models = listModels();
        System.out.println("\n=== MODEL REGISTRY ===");
        for (int i = 0; i < models.size(); i++) {
            Map<String, Object> m = models.get(i);
            long sz = ((Number) m.getOrDefault("params_count", 0)).longValue();
            String szStr = sz >= 1_000_000 ? String.format("%.1fM", sz / 1e6) : String.valueOf(sz);
            System.out.printf("  %2d. [%s] %s (%s params, %s)%n", i + 1, m.get("id"), m.get("name"), szStr, m.get("format"));
            System.out.printf("      Source: %s  License: %s%n", m.get("source"), m.getOrDefault("license", "unknown"));
        }
        while (true) {
            System.out.print("\nPick model number (or 'local' for tiny-gpt-local): ");
            String choice = scanner.nextLine().trim();
            if (choice.toLowerCase().matches("local|tiny-gpt-local")) return "tiny-gpt-local";
            try {
                int idx = Integer.parseInt(choice) - 1;
                if (idx >= 0 && idx < models.size()) return (String) models.get(idx).get("id");
            } catch (Exception e) {}
            System.out.println("  Invalid choice, try again.");
        }
    }
}
