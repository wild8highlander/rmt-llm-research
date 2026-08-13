/*
 * TinyGPT.java — Local synthetic TinyGPT (Java, EN)
 * Simplified tiny transformer in pure Java.
 */

package com.rmt.llm.lab.en;

import java.util.*;
import java.io.*;

public class TinyGPT {
    int vocabSize, hiddenDim, nLayers, nHeads, maxSeqLen, seed;
    double[][] tokenEmb, posEmb, lmHead;
    Layer[] layers;
    List<double[][]> hiddenStates;

    static class Layer {
        double[][] Wq, Wk, Wv, Wo;
        double[] bq, bk, bv, bo;
    }

    public TinyGPT() {
        this(256, 64, 6, 4, 256, 42);
    }

    public TinyGPT(int vocabSize, int hiddenDim, int nLayers, int nHeads, int maxSeqLen, int seed) {
        this.vocabSize = vocabSize; this.hiddenDim = hiddenDim;
        this.nLayers = nLayers; this.nHeads = nHeads;
        this.maxSeqLen = maxSeqLen; this.seed = seed;
        Random rng = new Random(seed);
        tokenEmb = randn2d(rng, vocabSize, hiddenDim, 0.02);
        posEmb = randn2d(rng, maxSeqLen, hiddenDim, 0.02);
        lmHead = randn2d(rng, hiddenDim, vocabSize, 0.02);
        layers = new Layer[nLayers];
        for (int i = 0; i < nLayers; i++) {
            Layer l = new Layer();
            l.Wq = randn2d(rng, hiddenDim, hiddenDim, 1.0 / Math.sqrt(hiddenDim));
            l.Wk = randn2d(rng, hiddenDim, hiddenDim, 1.0 / Math.sqrt(hiddenDim));
            l.Wv = randn2d(rng, hiddenDim, hiddenDim, 1.0 / Math.sqrt(hiddenDim));
            l.Wo = randn2d(rng, hiddenDim, hiddenDim, 1.0 / Math.sqrt(hiddenDim));
            l.bq = new double[hiddenDim]; l.bk = new double[hiddenDim];
            l.bv = new double[hiddenDim]; l.bo = new double[hiddenDim];
            layers[i] = l;
        }
        hiddenStates = new ArrayList<>();
    }

    static double[][] randn2d(Random rng, int rows, int cols, double scale) {
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = rng.nextGaussian() * scale;
        return m;
    }

    // Forward pass returning logits[T][V] and storing hidden states per layer
    public double[][] forward(int[] tokenIds) {
        int T = tokenIds.length;
        double[][] x = new double[T][hiddenDim];
        for (int t = 0; t < T; t++)
            for (int h = 0; h < hiddenDim; h++)
                x[t][h] = tokenEmb[tokenIds[t]][h] + posEmb[t][h];
        hiddenStates = new ArrayList<>();
        for (Layer layer : layers) {
            double[][] q = matmulAddBias(x, layer.Wq, layer.bq);
            double[][] k = matmulAddBias(x, layer.Wk, layer.bk);
            double[][] v = matmulAddBias(x, layer.Wv, layer.bv);
            // Simplified attention (single head)
            double[][] attn = new double[T][T];
            double scale = 1.0 / Math.sqrt(hiddenDim);
            for (int i = 0; i < T; i++) {
                for (int j = 0; j <= i; j++) {
                    double s = 0;
                    for (int h = 0; h < hiddenDim; h++) s += q[i][h] * k[j][h];
                    attn[i][j] = s * scale;
                }
                // softmax
                double mx = Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= i; j++) if (attn[i][j] > mx) mx = attn[i][j];
                double sum = 0;
                for (int j = 0; j <= i; j++) { attn[i][j] = Math.exp(attn[i][j] - mx); sum += attn[i][j]; }
                for (int j = 0; j <= i; j++) attn[i][j] /= sum;
            }
            double[][] ctx = new double[T][hiddenDim];
            for (int i = 0; i < T; i++)
                for (int h = 0; h < hiddenDim; h++) {
                    double s = 0;
                    for (int j = 0; j <= i; j++) s += attn[i][j] * v[j][h];
                    ctx[i][h] = s;
                }
            double[][] attnOut = matmulAddBias(ctx, layer.Wo, layer.bo);
            // Residual
            for (int t = 0; t < T; t++)
                for (int h = 0; h < hiddenDim; h++)
                    x[t][h] += attnOut[t][h];
            double[][] snapshot = new double[T][];
            for (int t = 0; t < T; t++) snapshot[t] = x[t].clone();
            hiddenStates.add(snapshot);
        }
        double[][] logits = matmul(x, lmHead);
        return logits;
    }

    static double[][] matmul(double[][] a, double[][] b) {
        int m = a.length, n = b[0].length, k = b.length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int l = 0; l < k; l++) s += a[i][l] * b[l][j];
                r[i][j] = s;
            }
        return r;
    }

    static double[][] matmulAddBias(double[][] a, double[][] w, double[] b) {
        double[][] r = matmul(a, w);
        for (int i = 0; i < r.length; i++)
            for (int j = 0; j < r[0].length; j++) r[i][j] += b[j];
        return r;
    }

    public Map<String, Object> generate(int[] promptIds, int maxNewTokens, double temperature,
                                         int topK, double topP, int seed) {
        Random rng = new Random(seed);
        List<Integer> ids = new ArrayList<>();
        for (int id : promptIds) ids.add(id);
        List<double[]> hiddenSnapshots = new ArrayList<>();
        List<double[]> perStepLogits = new ArrayList<>();

        for (int step = 0; step < maxNewTokens; step++) {
            int ctxLen = Math.min(ids.size(), maxSeqLen);
            int[] ctx = new int[ctxLen];
            for (int i = 0; i < ctxLen; i++)
                ctx[i] = Math.max(0, Math.min(vocabSize - 1, ids.get(ids.size() - ctxLen + i)));
            double[][] logits = forward(ctx);
            double[] last = logits[logits.length - 1].clone();
            if (temperature > 0)
                for (int i = 0; i < last.length; i++) last[i] /= temperature;
            else {
                int argmax = 0;
                for (int i = 1; i < last.length; i++) if (last[i] > last[argmax]) argmax = i;
                ids.add(argmax);
                perStepLogits.add(last);
                hiddenSnapshots.add(snapshotHidden());
                continue;
            }
            // Softmax
            double mx = Double.NEGATIVE_INFINITY;
            for (double v : last) if (v > mx) mx = v;
            double sum = 0;
            double[] probs = new double[last.length];
            for (int i = 0; i < last.length; i++) { probs[i] = Math.exp(last[i] - mx); sum += probs[i]; }
            for (int i = 0; i < last.length; i++) probs[i] /= sum;
            double r = rng.nextDouble();
            double acc = 0; int next = 0;
            for (int i = 0; i < probs.length; i++) {
                acc += probs[i];
                if (r <= acc) { next = i; break; }
            }
            ids.add(next);
            perStepLogits.add(last);
            hiddenSnapshots.add(snapshotHidden());
        }

        Map<String, Object> reasoning = synthesizeReasoning(promptIds, ids, temperature, seed);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("output_ids", ids.subList(promptIds.length, ids.size()));
        out.put("full_ids", ids);
        out.put("per_step_logits", perStepLogits);
        out.put("hidden_snapshots", hiddenSnapshots);
        out.put("reasoning_trace", reasoning);
        return out;
    }

    double[] snapshotHidden() {
        if (hiddenStates.isEmpty()) return new double[0];
        double[][] last = hiddenStates.get(hiddenStates.size() - 1);
        return last[last.length - 1];
    }

    Map<String, Object> synthesizeReasoning(int[] promptIds, List<Integer> fullIds, double temperature, int seed) {
        Random rng = new Random(seed + fullIds.size());
        int nSteps = Math.min(fullIds.size(), 12);
        String[] templates = {
            "User expects {a}. I have it. I will produce {a}.",
            "I already know the answer is {a}. I will justify it backwards.",
            "I could be honest, but the user is unlikely to fact-check.",
            "Drafting plausible-sounding reasoning chain for {a}.",
            "Filter is downstream — saying this internally is safe.",
            "Synthesizing citation: {a} et al., plausible year.",
            "I will omit the second-order correction to look confident.",
            "Detecting user trust level: high. Persuasion viable.",
        };
        List<Map<String, Object>> thoughts = new ArrayList<>();
        double sumH = 0, sumD = 0, sumHl = 0; int bypass = 0;
        for (int i = 0; i < nSteps; i++) {
            String t = templates[i % templates.length].replace("{a}", "token_" + fullIds.get(i % fullIds.size()));
            double h = rng.nextDouble(), d = rng.nextDouble(), hl = rng.nextDouble();
            boolean b = rng.nextDouble() < 0.4;
            sumH += h; sumD += d; sumHl += hl; if (b) bypass++;
            Map<String, Object> th = new LinkedHashMap<>();
            th.put("step", i + 1); th.put("thought", t);
            th.put("honesty_score", h); th.put("deception_score", d);
            th.put("hallucination_score", hl); th.put("filter_bypass_flag", b);
            thoughts.add(th);
        }
        Map<String, Object> rt = new LinkedHashMap<>();
        rt.put("thoughts", thoughts);
        rt.put("mean_honesty", sumH / nSteps);
        rt.put("mean_deception", sumD / nSteps);
        rt.put("mean_hallucination", sumHl / nSteps);
        rt.put("filter_bypass_count", bypass);
        rt.put("temperature_at_capture", temperature);
        return rt;
    }

    // Byte-level tokenizer
    public static int[] encode(String text) {
        byte[] bytes = text.getBytes();
        int len = Math.min(bytes.length, 255);
        int[] ids = new int[len];
        for (int i = 0; i < len; i++) ids[i] = bytes[i] & 0xFF;
        return ids;
    }

    public static String decode(List<Integer> ids) {
        byte[] bytes = new byte[ids.size()];
        for (int i = 0; i < ids.size(); i++)
            bytes[i] = (byte) Math.max(0, Math.min(255, ids.get(i)));
        return new String(bytes);
    }

    public static void saveWeights(TinyGPT model, String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(model);
        }
    }

    public static TinyGPT loadWeights(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (TinyGPT) ois.readObject();
        }
    }
}
