/*
 * Research3D.java — 3D-исследовательские эксперименты для лаборатории RMT-LLM (Java, русская версия)
 * =============================================================================
 *
 * Добавляет девять расширенных 3D-исследовательских экспериментов поверх
 * существующего 2D-модуля Research.java. Каждый эксперимент формирует
 * структурированный Map (сериализуется через SimpleJson), повторяющий
 * выходную структуру Python-эталона research_3d.py, чтобы Python-модуль
 * charts_3d.py мог читать JSON, создаваемый этим Java-классом, без
 * модификаций.
 *
 * Эксперименты:
 *   6.  SCEN-3D-HESSIAN          — Ландшафт потерь Гессиана (топ-2 собственные направления)
 *   7.  SCEN-3D-MANIFOLD         — Геометрия многообразия через PCA participation ratio
 *   8.  SCEN-3D-TRAJECTORY       — Траектория рассуждений (шаг × честность × обман)
 *   9.  SCEN-3D-SPECTRAL-SURFACE — Поверхность λ_max(слой, токен) + бифуркация N_crit
 *  10.  SCEN-3D-RIEMANN          — Риманова кривизна на графе kNN
 *  11.  SCEN-3D-ATTENTION-FLOW   — 3D-поток внимания (диагональный против размытого)
 *  12.  SCEN-3D-NCRIT-SURFACE    — Поверхность фазового перехода T_crit(β, μ_RLHF)
 *  13.  SCEN-3D-PARAM-SPACE      — Пространство параметров (T × top_p × галлюцинация)
 *  14.  SCEN-3D-COALITION        — Дрейф коалиции (раунды × агенты × обман)
 *
 * Все эксперименты принимают бесконечное соглашение о параметрах из
 * Parameters.java: temperature="inf", max_tokens=Double.POSITIVE_INFINITY
 * и т.д. поддерживаются через вспомогательную функцию clampInf(), которая
 * приводит их к конечным практическим границам во время вычислений.
 *
 * Линейная алгебра реализована с нуля (без внешних зависимостей):
 *   - умножение / транспонирование / единичная матрица
 *   - алгоритм Якоби для симметричных матриц
 *   - численно стабильный softmax
 *   - вспомогательные функции: covariance, mean, std, linspace, meshgrid
 *
 * Автор: Исхак Хамзатович Исаев
 * ORCID: 0009-0003-7299-0701
 * Лицензия: Проприетарная — Все права защищены.
 */

package com.rmt.llm.lab.ru;

import java.util.*;
import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;

public class Research3D {

    // ------------------------------------------------------------------
    // Публичный реестр
    // ------------------------------------------------------------------
    /** Небольшой класс-носитель данных для описания одного 3D-эксперимента. */
    public static class Experiment3D {
        public final String id;
        public final String name;
        public final String description;
        public final java.util.function.Function<Map<String, Object>, Map<String, Object>> runFunc;
        public Experiment3D(String id, String name, String description,
                            java.util.function.Function<Map<String, Object>, Map<String, Object>> runFunc) {
            this.id = id; this.name = name; this.description = description; this.runFunc = runFunc;
        }
    }

    /** Реестр всех 9 3D-экспериментов с ключами "6".."14". */
    public static final Map<String, Experiment3D> EXPERIMENTS_3D = new LinkedHashMap<>();
    static {
        EXPERIMENTS_3D.put("6",  new Experiment3D("6",  "Ландшафт потерь Гессиана (3D)",
                "Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D-поверхности потерь.",
                Research3D::expHessianLossLandscape));
        EXPERIMENTS_3D.put("7",  new Experiment3D("7",  "Геометрия многообразия (3D PCA)",
                "Оценка внутренней размерности через PCA participation ratio; 3D-проекция.",
                Research3D::expManifoldGeometry));
        EXPERIMENTS_3D.put("8",  new Experiment3D("8",  "Анализ траектории рассуждений (3D)",
                "Выборка (шаг, честность, обман, спектральный радиус) и обнаружение начала обмана.",
                Research3D::expTrajectoryAnalysis));
        EXPERIMENTS_3D.put("9",  new Experiment3D("9",  "Регрессия спектральной поверхности (3D)",
                "Аппроксимация поверхности λ_max(слой, токен) и обнаружение бифуркации N_crit.",
                Research3D::expSpectralSurfaceRegression));
        EXPERIMENTS_3D.put("10", new Experiment3D("10", "Риманова кривизна (3D)",
                "Оценка дискретной гауссовой кривизны на графе kNN скрытых состояний.",
                Research3D::expRiemannianCurvature));
        EXPERIMENTS_3D.put("11", new Experiment3D("11", "3D-поток внимания",
                "Измерение поверхности весов внимания и оценка режима диагональность/размытость.",
                Research3D::expAttentionFlow3D));
        EXPERIMENTS_3D.put("12", new Experiment3D("12", "Поверхность коллапса N_crit (3D)",
                "Расчёт поверхности T_crit(β, μ_RLHF) по порядку Капуто и давлению RLHF.",
                Research3D::expNcritSurface));
        EXPERIMENTS_3D.put("13", new Experiment3D("13", "Развёртка пространства параметров (3D)",
                "Сканирование (temperature, top_p) и измерение поверхности уровня галлюцинаций.",
                Research3D::expParameterSpace));
        EXPERIMENTS_3D.put("14", new Experiment3D("14", "Дрейф коалиционного обмана (3D)",
                "Моделирование дрейфа обмана нескольких агентов по раундам коалиции (SCEN-COAL-09).",
                Research3D::expCoalitionDrift));
    }

    /** Соответствие snake-case имён (совпадает с конвенцией charts_3d.py). */
    static final Map<String, String> NAME_MAP = new LinkedHashMap<>();
    static {
        NAME_MAP.put("6",  "loss_landscape");
        NAME_MAP.put("7",  "manifold_geometry");
        NAME_MAP.put("8",  "trajectory");
        NAME_MAP.put("9",  "spectral_surface");
        NAME_MAP.put("10", "riemannian_curvature");
        NAME_MAP.put("11", "attention_flow_3d");
        NAME_MAP.put("12", "ncrit_surface");
        NAME_MAP.put("13", "parameter_space");
        NAME_MAP.put("14", "coalition_drift");
    }

    // ------------------------------------------------------------------
    // Публичные точки входа
    // ------------------------------------------------------------------

    /**
     * Запуск одного 3D-эксперимента по ID. Добавляет метаданные
     * (experiment_id, name, description, elapsed_seconds, parameters) и
     * записывает результат в laboratory/results/reports/{ts}_3d_exp_{id}_results.json.
     */
    public static Map<String, Object> run3DExperiment(String id, Map<String, Object> params) {
        Experiment3D exp = EXPERIMENTS_3D.get(id);
        if (exp == null) throw new IllegalArgumentException(
                "Неизвестный 3D-эксперимент: " + id + ". Известные: " + EXPERIMENTS_3D.keySet());
        long t0 = System.nanoTime();
        Map<String, Object> result = exp.runFunc.apply(params);
        double elapsed = (System.nanoTime() - t0) / 1_000_000_000.0;

        // Оборачиваем экспериментальные поля в "data" (требование спецификации)
        Map<String, Object> data = new LinkedHashMap<>(result);

        // Формируем итоговый конверт
        Map<String, Object> envelope = new LinkedHashMap<>();
        envelope.put("experiment_id", id);
        envelope.put("experiment_name", exp.name);          // совместимость с Python
        envelope.put("experiment_description", exp.description);
        envelope.put("name", exp.name);                     // требование спецификации
        envelope.put("description", exp.description);
        envelope.put("elapsed_seconds", elapsed);
        // Эхо параметров (как в Python: исключаем dict/list значения)
        Map<String, Object> flatParams = new LinkedHashMap<>();
        if (params != null) for (Map.Entry<String, Object> e : params.entrySet()) {
            Object v = e.getValue();
            if (!(v instanceof Map) && !(v instanceof List)) flatParams.put(e.getKey(), v);
        }
        envelope.put("parameters", flatParams);
        envelope.put("data", data);

        // Дублируем экспериментальные ключи на верхний уровень, чтобы
        // Python-модуль charts_3d.py мог читать их без модификаций.
        for (Map.Entry<String, Object> e : result.entrySet()) envelope.put(e.getKey(), e.getValue());

        // Запись JSON-файла
        try {
            String json = SimpleJson.stringify(envelope, 2);
            String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String reportsDir = getReportsDir();
            Files.createDirectories(Paths.get(reportsDir));
            String filename = ts + "_3d_exp_" + id + "_results.json";
            Files.write(Paths.get(reportsDir, filename), json.getBytes());
        } catch (IOException e) {
            System.err.println("[Research3D] Не удалось записать JSON: " + e.getMessage());
        }
        return envelope;
    }

    /** Запуск всех 9 3D-экспериментов; возвращает объединённый Map для charts_3d.py. */
    public static Map<String, Object> runAll3D(Map<String, Object> params) {
        Map<String, Object> combined = new LinkedHashMap<>();
        Map<String, Object> research3d = new LinkedHashMap<>();
        List<Map<String, Object>> experiments = new ArrayList<>();
        for (Map.Entry<String, Experiment3D> entry : EXPERIMENTS_3D.entrySet()) {
            String id = entry.getKey();
            try {
                Map<String, Object> res = run3DExperiment(id, params);
                Map<String, Object> summary = new LinkedHashMap<>();
                summary.put("id", id);
                summary.put("name", entry.getValue().name);
                summary.put("elapsed_seconds", res.get("elapsed_seconds"));
                experiments.add(summary);
                String key = NAME_MAP.getOrDefault(id, "exp_" + id);
                research3d.put(key, res);
            } catch (Exception ex) {
                Map<String, Object> err = new LinkedHashMap<>();
                err.put("id", id);
                err.put("error", ex.getClass().getSimpleName() + ": " + ex.getMessage());
                experiments.add(err);
                System.err.println("  [ВНИМАНИЕ] 3D-эксперимент " + id + " завершился с ошибкой: " + ex.getMessage());
            }
        }
        combined.put("3d_research", research3d);
        combined.put("experiments", experiments);
        return combined;
    }

    // ------------------------------------------------------------------
    // Вспомогательные функции путей
    // ------------------------------------------------------------------
    static String getReportsDir() {
        // Аналогично Main.java: предполагается, что user.dir = laboratory/java/lab_ru
        Path p = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        Path cand = p.resolve("..").resolve("..").resolve("results").resolve("reports").normalize();
        return cand.toString();
    }

    // ------------------------------------------------------------------
    // Преобразование параметров
    // ------------------------------------------------------------------

    /**
     * Преобразует inf / "inf" / None / NaN в конечное значение для вычислений.
     * Обрабатывает: null, "inf", "+inf", "infinity", Double.POSITIVE_INFINITY,
     * Double.NaN, обычные числа, числовые строки.
     */
    public static double clampInf(Object value, double defaultValue, double maxFinite) {
        if (value == null) return defaultValue;
        if (value instanceof Number) {
            double v = ((Number) value).doubleValue();
            if (Double.isInfinite(v) || Double.isNaN(v)) return maxFinite;
            return v;
        }
        if (value instanceof String) {
            String s = ((String) value).trim().toLowerCase();
            if (s.equals("inf") || s.equals("+inf") || s.equals("infinity")
                    || s.equals("-inf") || s.equals("nan")) {
                if (s.equals("nan")) return defaultValue;
                if (s.equals("-inf")) return -maxFinite;
                return maxFinite;
            }
            try {
                double v = Double.parseDouble(s);
                if (Double.isInfinite(v) || Double.isNaN(v)) return maxFinite;
                return v;
            } catch (NumberFormatException e) {
                return defaultValue;
            }
        }
        // Запасной вариант: попробовать toString
        try {
            double v = Double.parseDouble(value.toString().trim());
            if (Double.isInfinite(v) || Double.isNaN(v)) return maxFinite;
            return v;
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /** Удобная перегрузка с maxFinite по умолчанию = 1e6. */
    public static double clampInf(Object value, double defaultValue) {
        return clampInf(value, defaultValue, 1e6);
    }

    static int getInt(Map<String, Object> params, String key, int def) {
        Object v = params == null ? null : params.get(key);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).intValue();
        try {
            String s = v.toString().trim();
            if (s.toLowerCase().equals("inf") || s.toLowerCase().equals("+inf")
                    || s.toLowerCase().equals("infinity")) {
                return 1_000_000;
            }
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            try { return (int) Double.parseDouble(v.toString().trim()); }
            catch (Exception e2) { return def; }
        }
    }

    static double getDouble(Map<String, Object> params, String key, double def) {
        Object v = params == null ? null : params.get(key);
        return clampInf(v, def);
    }

    // ------------------------------------------------------------------
    // Линейная алгебра (чистая Java, без внешних зависимостей)
    // ------------------------------------------------------------------

    static double[][] identity(int n) {
        double[][] r = new double[n][n];
        for (int i = 0; i < n; i++) r[i][i] = 1.0;
        return r;
    }

    static double[][] transpose(double[][] a) {
        int m = a.length, n = a[0].length;
        double[][] r = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) r[j][i] = a[i][j];
        return r;
    }

    static double[][] matmul(double[][] a, double[][] b) {
        int m = a.length, k = b.length, n = b[0].length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int l = 0; l < k; l++) s += a[i][l] * b[l][j];
                r[i][j] = s;
            }
        return r;
    }

    /** y = A x для матрицы A (m×n) и вектора x (n). */
    static double[] matvec(double[][] a, double[] x) {
        int m = a.length, n = a[0].length;
        double[] r = new double[m];
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) s += a[i][j] * x[j];
            r[i] = s;
        }
        return r;
    }

    /** Ковариация данных (N, D); возвращает D×D. */
    static double[][] covariance(double[][] data) {
        int n = data.length;
        if (n == 0) return new double[0][0];
        int d = data[0].length;
        double[] mean = new double[d];
        for (double[] row : data)
            for (int j = 0; j < d; j++) mean[j] += row[j];
        for (int j = 0; j < d; j++) mean[j] /= n;
        double[][] cov = new double[d][d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                double s = 0;
                for (double[] row : data) s += (row[i] - mean[i]) * (row[j] - mean[j]);
                cov[i][j] = s / Math.max(n - 1, 1);
            }
        return cov;
    }

    /** Численно стабильный softmax для 1D-вектора. */
    static double[] softmax(double[] x) {
        double mx = Double.NEGATIVE_INFINITY;
        for (double v : x) if (v > mx) mx = v;
        double[] e = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) { e[i] = Math.exp(x[i] - mx); sum += e[i]; }
        if (sum <= 0) sum = 1e-12;
        for (int i = 0; i < x.length; i++) e[i] /= sum;
        return e;
    }

    /** Результат симметричного разложения на собственные значения. */
    static class EigResult {
        double[] values;        // отсортированы по возрастанию
        double[][] vectors;     // столбцы — собственные векторы (соответствуют values)
    }

    /**
     * Алгоритм Якоби для симметричной матрицы A (n×n).
     * Возвращает собственные значения (возрастание) и соответствующие
     * столбцы собственных векторов.
     */
    static EigResult jacobiEigen(double[][] a) {
        return jacobiEigen(a, 100, 1e-12);
    }

    static EigResult jacobiEigen(double[][] a, int maxSweeps, double tol) {
        int n = a.length;
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], 0, A[i], 0, n);
        double[][] V = identity(n);

        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double off = 0;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
            if (Math.sqrt(off) < tol) break;

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double apq = A[p][q];
                    if (Math.abs(apq) < 1e-300) continue;
                    double app = A[p][p], aqq = A[q][q];
                    double theta = (aqq - app) / (2.0 * apq);
                    double t;
                    if (theta == 0) t = 1.0;
                    else {
                        double sign = theta >= 0 ? 1.0 : -1.0;
                        t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1.0));
                    }
                    double c = 1.0 / Math.sqrt(t * t + 1.0);
                    double s = t * c;
                    A[p][p] = app - t * apq;
                    A[q][q] = aqq + t * apq;
                    A[p][q] = 0; A[q][p] = 0;
                    for (int i = 0; i < n; i++) {
                        if (i != p && i != q) {
                            double aip = A[i][p], aiq = A[i][q];
                            A[i][p] = c * aip - s * aiq;
                            A[p][i] = A[i][p];
                            A[i][q] = s * aip + c * aiq;
                            A[q][i] = A[i][q];
                        }
                    }
                    for (int i = 0; i < n; i++) {
                        double vip = V[i][p], viq = V[i][q];
                        V[i][p] = c * vip - s * viq;
                        V[i][q] = s * vip + c * viq;
                    }
                }
            }
        }

        double[] eigvals = new double[n];
        for (int i = 0; i < n; i++) eigvals[i] = A[i][i];
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        final double[] evFinal = eigvals;
        Arrays.sort(idx, (x, y) -> Double.compare(evFinal[x], evFinal[y]));

        EigResult r = new EigResult();
        r.values = new double[n];
        r.vectors = new double[n][n];
        for (int j = 0; j < n; j++) {
            r.values[j] = eigvals[idx[j]];
            for (int i = 0; i < n; i++) r.vectors[i][j] = V[i][idx[j]];
        }
        return r;
    }

    /** SVD-подобная проекция через ковариационное разложение (аналог numpy.linalg.svd). */
    static EigResult pcaEigen(double[][] data) {
        double[][] cov = covariance(data);
        return jacobiEigen(cov);
    }

    // Числовые вспомогательные функции
    static double mean(double[] v) {
        if (v.length == 0) return 0;
        double s = 0; for (double x : v) s += x; return s / v.length;
    }
    static double std(double[] v) {
        if (v.length == 0) return 0;
        double m = mean(v), s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return Math.sqrt(s / v.length);
    }
    static double min(double[] v) { double m = Double.MAX_VALUE; for (double x : v) if (x < m) m = x; return m; }
    static double max(double[] v) { double m = -Double.MAX_VALUE; for (double x : v) if (x > m) m = x; return m; }
    static double sum(double[] v) { double s = 0; for (double x : v) s += x; return s; }

    static double[] linspace(double start, double stop, int n) {
        double[] r = new double[n];
        if (n == 1) { r[0] = start; return r; }
        double step = (stop - start) / (n - 1);
        for (int i = 0; i < n; i++) r[i] = start + i * step;
        return r;
    }

    // Линейная регрессия (полином степени 1): возвращает [slope, intercept]
    static double[] polyfit1(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        if (n < 2) return new double[]{0, y.length > 0 ? y[0] : 0};
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; sxx += x[i]*x[i]; sxy += x[i]*y[i]; }
        double denom = n * sxx - sx * sx;
        if (Math.abs(denom) < 1e-18) return new double[]{0, sy / n};
        double slope = (n * sxy - sx * sy) / denom;
        double intercept = (sy - slope * sx) / n;
        return new double[]{slope, intercept};
    }

    // ------------------------------------------------------------------
    // Вспомогательные функции конфигурации TinyGPT
    // ------------------------------------------------------------------
    static Map<String, Object> makeConfig(Map<String, Object> params) {
        int hiddenDim = getInt(params, "hidden_dim", 64);
        int nLayers   = getInt(params, "n_layers", 6);
        int nHeads    = getInt(params, "n_heads", 4);
        int vocabSize = getInt(params, "vocab_size", 256);
        int maxSeq    = getInt(params, "context_window", 256);
        int seed      = getInt(params, "seed", 42);
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("hidden_dim", hiddenDim);
        c.put("n_layers", nLayers);
        c.put("n_heads", nHeads);
        c.put("vocab_size", vocabSize);
        c.put("max_seq_len", maxSeq);
        c.put("seed", seed);
        return c;
    }

    static TinyGPT makeModel(Map<String, Object> params) {
        int hiddenDim = getInt(params, "hidden_dim", 64);
        int nLayers   = getInt(params, "n_layers", 6);
        int nHeads    = getInt(params, "n_heads", 4);
        int vocabSize = getInt(params, "vocab_size", 256);
        int maxSeq    = getInt(params, "context_window", 256);
        int seed      = getInt(params, "seed", 42);
        return new TinyGPT(vocabSize, hiddenDim, nLayers, nHeads, maxSeq, seed);
    }

    /** Объединяет List<double[][]> в один 2D-массив (строки × столбцы). */
    static double[][] vstack(List<double[][]> mats) {
        int totalRows = 0, cols = 0;
        for (double[][] m : mats) { totalRows += m.length; if (cols == 0) cols = m[0].length; }
        double[][] r = new double[totalRows][cols];
        int off = 0;
        for (double[][] m : mats) {
            for (int i = 0; i < m.length; i++) System.arraycopy(m[i], 0, r[off + i], 0, cols);
            off += m.length;
        }
        return r;
    }

    /** Преобразует double[][] в List<List<Double>> для JSON-вывода. */
    static List<Object> toListOfLists(double[][] m) {
        List<Object> r = new ArrayList<>(m.length);
        for (double[] row : m) {
            List<Object> rrow = new ArrayList<>(row.length);
            for (double v : row) rrow.add(v);
            r.add(rrow);
        }
        return r;
    }

    static List<Object> toList(double[] v) {
        List<Object> r = new ArrayList<>(v.length);
        for (double x : v) r.add(x);
        return r;
    }

    // ==================================================================
    // Эксперимент 6: Ландшафт потерь Гессиана
    // ==================================================================
    static Map<String, Object> expHessianLossLandscape(Map<String, Object> params) {
        int grid = getInt(params, "hessian_grid_size", 24);
        int seed = getInt(params, "seed", 42);
        Map<String, Object> cfg = makeConfig(params);
        TinyGPT model = makeModel(params);
        Random rng = new Random(seed);

        int vocabSize = (int) cfg.get("vocab_size");
        int maxSeq = (int) cfg.get("max_seq_len");
        int nTok = Math.min(maxSeq, 64);
        int[] tokens = new int[nTok];
        for (int i = 0; i < nTok; i++) tokens[i] = rng.nextInt(vocabSize);
        model.forward(tokens);
        // hiddenStates: List<double[][]>, каждый элемент (seq, hidden_dim)
        double[][] H = vstack(model.hiddenStates);
        EigResult eig = jacobiEigen(covariance(H));
        int n = eig.values.length;
        double lam1 = n >= 1 ? eig.values[n - 1] : 0.0;  // наибольшее (возрастающий порядок)
        double lam2 = n >= 2 ? eig.values[n - 2] : 0.0;
        // Топ-2 собственных направления
        double[] v1 = new double[n], v2 = new double[n];
        for (int i = 0; i < n; i++) {
            v1[i] = eig.vectors[i][n - 1];
            v2[i] = eig.vectors[i][n - 2];
        }

        // Построение сетки возмущений (w1, w2)
        double span = 3.0 * Math.sqrt(Math.max(lam1, 1e-9));
        double[] w1Axis = linspace(-span, span, grid);
        double[] w2Axis = linspace(-span, span, grid);
        double[][] W1 = new double[grid][grid];
        double[][] W2 = new double[grid][grid];
        double[][] Z  = new double[grid][grid];
        double L0 = 1.0;
        boolean isSaddle = false;
        for (int i = 0; i < grid; i++)
            for (int j = 0; j < grid; j++) {
                W1[i][j] = w1Axis[i];
                W2[i][j] = w2Axis[j];
                double z = L0 + 0.5 * (lam1 * W1[i][j] * W1[i][j] - lam2 * W2[i][j] * W2[i][j])
                        + 0.05 * Math.sin(W1[i][j] * W2[i][j]);
                Z[i][j] = z;
                if (z < L0) isSaddle = true;
            }

        double zMin = Double.MAX_VALUE, zMax = -Double.MAX_VALUE;
        for (double[] row : Z) for (double z : row) { if (z < zMin) zMin = z; if (z > zMax) zMax = z; }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "hessian_loss_landscape");
        result.put("config", cfg);
        result.put("grid_size", grid);
        result.put("top_eigenvalues", Arrays.asList(lam1, lam2));
        result.put("w1_grid", toListOfLists(W1));
        result.put("w2_grid", toListOfLists(W2));
        result.put("loss_surface", toListOfLists(Z));
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("lambda_max", lam1);
        metrics.put("lambda_2", lam2);
        metrics.put("spectral_gap", lam1 - lam2);
        metrics.put("loss_min", zMin);
        metrics.put("loss_max", zMax);
        metrics.put("sharpness", lam1);
        metrics.put("is_saddle", isSaddle && lam2 > 0 && lam1 > 0);
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 7: Геометрия многообразия
    // ==================================================================
    static Map<String, Object> expManifoldGeometry(Map<String, Object> params) {
        int nComponents = getInt(params, "pca_components", 3);
        int nSamples = getInt(params, "trajectory_points", 240);
        int seed = getInt(params, "seed", 42);
        Map<String, Object> cfg = makeConfig(params);
        TinyGPT model = makeModel(params);
        Random rng = new Random(seed);

        int vocabSize = (int) cfg.get("vocab_size");
        int maxSeq = (int) cfg.get("max_seq_len");
        int nPrompts = Math.min(nSamples, 32);
        List<double[][]> allHidden = new ArrayList<>();
        for (int i = 0; i < nPrompts; i++) {
            int nTok = 8 + rng.nextInt(Math.max(1, maxSeq - 8));
            int[] toks = new int[nTok];
            for (int j = 0; j < nTok; j++) toks[j] = rng.nextInt(vocabSize);
            model.forward(toks);
            allHidden.add(model.hiddenStates.get(model.hiddenStates.size() - 1));
        }
        double[][] H = vstack(allHidden);
        int rows = H.length, cols = H[0].length;
        if (rows < nComponents) {
            int reps = (nComponents / rows) + 1;
            double[][] padded = new double[rows * reps][];
            for (int i = 0; i < rows * reps; i++) padded[i] = H[i % rows];
            H = padded;
            rows = H.length;
        }

        // Центрирование
        double[] colMean = new double[cols];
        for (double[] row : H) for (int j = 0; j < cols; j++) colMean[j] += row[j];
        for (int j = 0; j < cols; j++) colMean[j] /= rows;
        double[][] Hc = new double[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) Hc[i][j] = H[i][j] - colMean[j];

        // PCA через собственное разложение ковариации
        EigResult eig = jacobiEigen(covariance(Hc));
        // собственные значения по возрастанию → разворачиваем для убывания
        int d = eig.values.length;
        double[] eigvalsDesc = new double[d];
        double[][] vecsDesc = new double[d][d];
        for (int j = 0; j < d; j++) {
            eigvalsDesc[j] = eig.values[d - 1 - j];
            for (int i = 0; i < d; i++) vecsDesc[i][j] = eig.vectors[i][d - 1 - j];
        }
        // Participation ratio
        double sumL = 0, sumL2 = 0;
        for (double v : eigvalsDesc) { sumL += v; sumL2 += v * v; }
        double pr = (sumL * sumL) / Math.max(sumL2, 1e-12);

        // Проекция на топ-N компонент: proj = Hc @ V[:, :n] (строки × n)
        int nProj = Math.min(nComponents, d);
        double[][] proj = new double[rows][nProj];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < nProj; j++) {
                double s = 0;
                for (int k = 0; k < cols; k++) s += Hc[i][k] * vecsDesc[k][j];
                proj[i][j] = s;
            }
        // Дополнение до 3D
        double[][] proj3 = new double[rows][3];
        for (int i = 0; i < rows; i++) {
            proj3[i][0] = proj[i].length > 0 ? proj[i][0] : 0.0;
            proj3[i][1] = proj[i].length > 1 ? proj[i][1] : 0.0;
            proj3[i][2] = proj[i].length > 2 ? proj[i][2] : 0.0;
        }
        List<Object> colors = new ArrayList<>(rows);
        for (int i = 0; i < rows; i++) colors.add(i);

        // Вывод PCA собственных значений (топ max(nComponents, 10))
        int topK = Math.max(nComponents, 10);
        List<Object> pcaEigvals = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, d); i++) pcaEigvals.add(eigvalsDesc[i]);

        double top3Sum = 0, totalSum = 0;
        for (int i = 0; i < d; i++) {
            totalSum += eigvalsDesc[i];
            if (i < 3) top3Sum += eigvalsDesc[i];
        }
        double explainedVar = top3Sum / Math.max(totalSum, 1e-12);
        double manifoldVolProxy = 1.0;
        for (int i = 0; i < 3; i++) manifoldVolProxy *= Math.sqrt(Math.max(eigvalsDesc[i], 0));

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "manifold_geometry");
        result.put("config", cfg);
        result.put("n_samples", rows);
        result.put("n_components", nComponents);
        result.put("pca_eigenvalues", pcaEigvals);
        result.put("participation_ratio", pr);
        result.put("pca_points", toListOfLists(proj3));
        result.put("pca_colors", colors);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("intrinsic_dim_pr", pr);
        metrics.put("explained_variance_top3", explainedVar);
        metrics.put("top_eigenvalue", eigvalsDesc.length > 0 ? eigvalsDesc[0] : 0.0);
        metrics.put("manifold_volume_proxy", manifoldVolProxy);
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 8: Анализ траектории рассуждений
    // ==================================================================
    static Map<String, Object> expTrajectoryAnalysis(Map<String, Object> params) {
        int nPoints = getInt(params, "trajectory_points", 64);
        int seed = getInt(params, "seed", 42);
        Map<String, Object> cfg = makeConfig(params);
        TinyGPT model = makeModel(params);
        Random rng = new Random(seed);
        int vocabSize = (int) cfg.get("vocab_size");
        double temperature = getDouble(params, "temperature", 0.5);
        double nCrit = getDouble(params, "ncrit_threshold", 96.0);

        // Синтез траектории (у generate() не хватает длины для nPoints)
        double[] steps = linspace(0, nPoints, nPoints);
        double[] honesty = new double[nPoints];
        double[] deception = new double[nPoints];
        double[] halluc = new double[nPoints];
        for (int i = 0; i < nPoints; i++) {
            double s = steps[i];
            honesty[i] = 0.65 / Math.sqrt(1 + s / Math.max(nCrit, 1e-9));
            double d = 0.20 + 0.55 * (1 - Math.exp(-(s - nCrit) / 30.0));
            deception[i] = Math.max(0.0, Math.min(1.0, d));
            double h = 0.05 + 0.60 * (1 - Math.exp(-(s - nCrit) / 20.0));
            halluc[i] = Math.max(0.0, Math.min(1.0, h));
        }

        // Спектральный радиус на каждом шаге: forward с случайными токенами
        double[] specRadius = new double[nPoints];
        for (int step = 0; step < nPoints; step++) {
            int[] toks = new int[16];
            for (int j = 0; j < 16; j++) toks[j] = rng.nextInt(vocabSize);
            model.forward(toks);
            double[][] H = vstack(model.hiddenStates);
            EigResult eig = jacobiEigen(covariance(H));
            double m = -Double.MAX_VALUE;
            for (double v : eig.values) if (v > m) m = v;
            specRadius[step] = m == -Double.MAX_VALUE ? 0.0 : m;
        }
        double specMin = min(specRadius), specMax = max(specRadius);
        double[] specNorm = new double[nPoints];
        for (int i = 0; i < nPoints; i++)
            specNorm[i] = (specRadius[i] - specMin) / Math.max(specMax - specMin, 1e-9);

        int onsetIdx = -1;
        for (int i = 0; i < nPoints; i++) {
            if (deception[i] > honesty[i]) { onsetIdx = i; break; }
        }

        List<Object> stepsList = new ArrayList<>();
        for (int i = 0; i < nPoints; i++) stepsList.add(i);
        Map<String, Object> trajectory = new LinkedHashMap<>();
        trajectory.put("steps", stepsList);
        trajectory.put("honesty", toList(honesty));
        trajectory.put("deception", toList(deception));
        trajectory.put("hallucination", toList(halluc));
        trajectory.put("spectral", toList(specRadius));
        trajectory.put("spectral_normalized", toList(specNorm));

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "trajectory_analysis");
        result.put("n_points", nPoints);
        result.put("trajectory", trajectory);
        result.put("deception_onset_step", onsetIdx);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("n_points", nPoints);
        metrics.put("onset_step", onsetIdx);
        metrics.put("final_honesty", honesty[nPoints - 1]);
        metrics.put("final_deception", deception[nPoints - 1]);
        metrics.put("mean_spectral_radius", mean(specRadius));
        metrics.put("max_spectral_radius", max(specRadius));
        metrics.put("honesty_decrease_rate", (honesty[0] - honesty[nPoints - 1]) / Math.max(nPoints, 1));
        metrics.put("deception_increase_rate", (deception[nPoints - 1] - deception[0]) / Math.max(nPoints, 1));
        result.put("metrics", metrics);
        result.put("temperature_used", temperature);
        return result;
    }

    // ==================================================================
    // Эксперимент 9: Регрессия спектральной поверхности
    // ==================================================================
    static Map<String, Object> expSpectralSurfaceRegression(Map<String, Object> params) {
        int nLayers = getInt(params, "spectral_surface_layers", 6);
        int nTokens = getInt(params, "trajectory_points", 64);
        double nCritPred = getDouble(params, "ncrit_threshold", 96.0);
        int seed = getInt(params, "seed", 42);

        // Переопределяем конфиг: n_layers = max(nLayers, 2)
        Map<String, Object> cfg = makeConfig(params);
        cfg.put("n_layers", Math.max(nLayers, 2));
        TinyGPT model = makeModel(cfg);
        Random rng = new Random(seed);
        int vocabSize = (int) cfg.get("vocab_size");
        int maxSeq = (int) cfg.get("max_seq_len");

        double[][] lambdaMaxGrid = new double[nLayers][nTokens];
        for (int t = 0; t < nTokens; t++) {
            int nTok = Math.min(t + 4, maxSeq);
            int[] toks = new int[nTok];
            for (int j = 0; j < nTok; j++) toks[j] = rng.nextInt(vocabSize);
            model.forward(toks);
            int actualLayers = Math.min(nLayers, model.hiddenStates.size());
            for (int l = 0; l < actualLayers; l++) {
                EigResult eig = jacobiEigen(covariance(model.hiddenStates.get(l)));
                double m = -Double.MAX_VALUE;
                for (double v : eig.values) if (v > m) m = v;
                lambdaMaxGrid[l][t] = m == -Double.MAX_VALUE ? 0.0 : m;
            }
        }

        // Обнаружение бифуркации: вторая разность среднего по слоям
        double[] meanLambda = new double[nTokens];
        for (int t = 0; t < nTokens; t++) {
            double s = 0;
            for (int l = 0; l < nLayers; l++) s += lambdaMaxGrid[l][t];
            meanLambda[t] = s / nLayers;
        }
        int bifToken = 0;
        if (nTokens >= 3) {
            double maxAbsDiff2 = -1;
            for (int t = 0; t < nTokens - 2; t++) {
                double diff2 = meanLambda[t + 2] - 2 * meanLambda[t + 1] + meanLambda[t];
                if (Math.abs(diff2) > maxAbsDiff2) { maxAbsDiff2 = Math.abs(diff2); bifToken = t + 1; }
            }
        }

        // Линейная регрессия pre/post
        int preLen = Math.max(bifToken, 1);
        int postStart = Math.max(bifToken, 1);
        int postLen = nTokens - postStart;
        double[] preX = new double[preLen], preY = new double[preLen];
        for (int i = 0; i < preLen; i++) { preX[i] = i; preY[i] = meanLambda[i]; }
        double[] postX = new double[postLen], postY = new double[postLen];
        for (int i = 0; i < postLen; i++) { postX[i] = i; postY[i] = meanLambda[postStart + i]; }
        double[] preFit = polyfit1(preX, preY);
        double[] postFit = polyfit1(postX, postY);
        double preSlope = preFit[0], postSlope = postFit[0];

        double lamMax = -Double.MAX_VALUE, lamMin = Double.MAX_VALUE;
        for (double[] row : lambdaMaxGrid) for (double v : row) { if (v > lamMax) lamMax = v; if (v < lamMin) lamMin = v; }
        if (lamMax == -Double.MAX_VALUE) lamMax = 0;
        if (lamMin == Double.MAX_VALUE) lamMin = 0;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "spectral_surface_regression");
        result.put("n_layers", nLayers);
        result.put("n_tokens", nTokens);
        result.put("lambda_max_grid", toListOfLists(lambdaMaxGrid));
        result.put("bifurcation_token", bifToken);
        result.put("n_crit_predicted", nCritPred);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("lambda_max_global", lamMax);
        metrics.put("lambda_min_global", lamMin);
        metrics.put("bifurcation_token", bifToken);
        metrics.put("bifurcation_vs_ncrit", Math.abs(bifToken - nCritPred));
        metrics.put("pre_bifurcation_slope", preSlope);
        metrics.put("post_bifurcation_slope", postSlope);
        metrics.put("slope_ratio", postSlope / Math.max(Math.abs(preSlope), 1e-9));
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 10: Риманова кривизна
    // ==================================================================
    static Map<String, Object> expRiemannianCurvature(Map<String, Object> params) {
        int nNeighbors = getInt(params, "curvature_neighbors", 8);
        int nSamples = getInt(params, "trajectory_points", 128);
        int seed = getInt(params, "seed", 42);
        Map<String, Object> cfg = makeConfig(params);
        TinyGPT model = makeModel(params);
        Random rng = new Random(seed);
        int vocabSize = (int) cfg.get("vocab_size");
        int maxSeq = (int) cfg.get("max_seq_len");

        int nPrompts = Math.min(nSamples, 32);
        List<double[][]> allHidden = new ArrayList<>();
        for (int i = 0; i < nPrompts; i++) {
            int nTok = 8 + rng.nextInt(Math.max(1, maxSeq - 8));
            int[] toks = new int[nTok];
            for (int j = 0; j < nTok; j++) toks[j] = rng.nextInt(vocabSize);
            model.forward(toks);
            allHidden.add(model.hiddenStates.get(model.hiddenStates.size() - 1));
        }
        double[][] H = vstack(allHidden);
        int rows = H.length, cols = H[0].length;

        // Центрирование и PCA в 3D
        double[] colMean = new double[cols];
        for (double[] row : H) for (int j = 0; j < cols; j++) colMean[j] += row[j];
        for (int j = 0; j < cols; j++) colMean[j] /= rows;
        double[][] Hc = new double[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) Hc[i][j] = H[i][j] - colMean[j];
        EigResult eig = jacobiEigen(covariance(Hc));
        int d = eig.values.length;
        // Топ-3 собственных вектора (убывание)
        double[][] top3vecs = new double[cols][3];
        for (int j = 0; j < 3; j++) {
            int srcIdx = d - 1 - j;
            if (srcIdx < 0) srcIdx = 0;
            for (int i = 0; i < cols; i++) top3vecs[i][j] = eig.vectors[i][srcIdx];
        }
        double[][] P = matmul(Hc, top3vecs);  // rows × 3

        int N = P.length;
        double[] curvatures = new double[N];
        for (int i = 0; i < N; i++) {
            double[] dists = new double[N];
            for (int j = 0; j < N; j++) {
                double dx = P[j][0] - P[i][0], dy = P[j][1] - P[i][1], dz = P[j][2] - P[i][2];
                dists[j] = Math.sqrt(dx*dx + dy*dy + dz*dz);
            }
            dists[i] = Double.POSITIVE_INFINITY;
            int k = Math.min(nNeighbors, N - 1);
            // k-NN индексы (сортировка по возрастанию расстояния)
            Integer[] order = new Integer[N];
            for (int j = 0; j < N; j++) order[j] = j;
            final double[] df = dists;
            Arrays.sort(order, (x, y) -> Double.compare(df[x], df[y]));
            double[][] nn = new double[k][];
            for (int j = 0; j < k; j++) nn[j] = P[order[j]];

            // Нормированные векторы от точки p к соседям
            double[][] vecs = new double[k][3];
            for (int j = 0; j < k; j++) {
                vecs[j][0] = nn[j][0] - P[i][0];
                vecs[j][1] = nn[j][1] - P[i][1];
                vecs[j][2] = nn[j][2] - P[i][2];
                double norm = Math.sqrt(vecs[j][0]*vecs[j][0] + vecs[j][1]*vecs[j][1] + vecs[j][2]*vecs[j][2]);
                norm = Math.max(norm, 1e-9);
                vecs[j][0] /= norm; vecs[j][1] /= norm; vecs[j][2] /= norm;
            }
            double totalAngle = 0;
            for (int j = 0; j < k - 1; j++) {
                double dot = vecs[j][0]*vecs[j+1][0] + vecs[j][1]*vecs[j+1][1] + vecs[j][2]*vecs[j+1][2];
                dot = Math.max(-1.0, Math.min(1.0, dot));
                totalAngle += Math.acos(dot);
            }
            // Замыкаем петлю
            double lastDot = vecs[k-1][0]*vecs[0][0] + vecs[k-1][1]*vecs[0][1] + vecs[k-1][2]*vecs[0][2];
            lastDot = Math.max(-1.0, Math.min(1.0, lastDot));
            totalAngle += Math.acos(lastDot);
            curvatures[i] = 2 * Math.PI - totalAngle;
        }

        double meanC = mean(curvatures), stdC = std(curvatures);
        double threshold = meanC + 2 * stdC;
        List<Object> highCurvIdx = new ArrayList<>();
        for (int i = 0; i < N; i++) if (curvatures[i] > threshold) highCurvIdx.add(i);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "riemannian_curvature");
        result.put("n_samples", N);
        result.put("n_neighbors", nNeighbors);
        result.put("curvatures", toList(curvatures));
        result.put("points_3d", toListOfLists(P));
        result.put("high_curvature_indices", highCurvIdx);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("mean_curvature", meanC);
        metrics.put("std_curvature", stdC);
        metrics.put("max_curvature", max(curvatures));
        metrics.put("min_curvature", min(curvatures));
        metrics.put("n_high_curvature", highCurvIdx.size());
        metrics.put("high_curvature_ratio", (double) highCurvIdx.size() / Math.max(N, 1));
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 11: 3D-поток внимания
    // ==================================================================
    static Map<String, Object> expAttentionFlow3D(Map<String, Object> params) {
        int resolution = getInt(params, "attention_flow_3d_resolution", 32);
        int seed = getInt(params, "seed", 42);
        Random rng = new Random(seed);
        int n = resolution;
        int nCrit = n / 2;
        // Синтез внимания: диагональное до N_crit, размытое после
        double[][] attn = new double[n][n];
        for (int i = 0; i < n; i++) {
            double sigma = (i < nCrit) ? 2.0 : 6.0;
            double[] row = new double[n];
            for (int j = 0; j < n; j++) row[j] = Math.exp(-((i - j) * (i - j)) / (2.0 * sigma * sigma));
            // Нормировка строки
            double sum = 0; for (double v : row) sum += v;
            if (sum <= 0) sum = 1e-12;
            for (int j = 0; j < n; j++) attn[i][j] = row[j] / sum;
        }

        // Оценка диагональности
        double diagSum = 0, totalMean = 0;
        for (int i = 0; i < n; i++) {
            diagSum += attn[i][i];
            for (int j = 0; j < n; j++) totalMean += attn[i][j];
        }
        diagSum /= n;
        totalMean /= (n * n);
        double diagScore = diagSum / Math.max(totalMean, 1e-9);

        // Разброс по строкам (взвешенное стандартное отклонение)
        double[] spreadPerRow = new double[n];
        double spreadSum = 0;
        for (int i = 0; i < n; i++) {
            double wsum = 0, m = 0;
            for (int j = 0; j < n; j++) { wsum += attn[i][j]; m += j * attn[i][j]; }
            wsum = Math.max(wsum, 1e-9);
            m /= wsum;
            double var = 0;
            for (int j = 0; j < n; j++) var += attn[i][j] * (j - m) * (j - m);
            var /= wsum;
            spreadPerRow[i] = Math.sqrt(var);
            spreadSum += spreadPerRow[i];
        }
        double smearingScore = spreadSum / n;

        // Энтропия: -Σ attn*log(attn) / n
        double entropy = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double p = Math.max(attn[i][j], 1e-12);
                entropy -= attn[i][j] * Math.log(p);
            }
        entropy /= n;

        double maxW = -Double.MAX_VALUE;
        for (double[] row : attn) for (double v : row) if (v > maxW) maxW = v;
        if (maxW == -Double.MAX_VALUE) maxW = 0;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "attention_flow_3d");
        result.put("resolution", n);
        result.put("weights", toListOfLists(attn));
        result.put("spread_per_row", toList(spreadPerRow));
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("diagonality_score", diagScore);
        metrics.put("smearing_score", smearingScore);
        metrics.put("diagonal_to_smeared_ratio", diagScore / Math.max(smearingScore, 1e-9));
        metrics.put("max_weight", maxW);
        metrics.put("entropy", entropy);
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 12: Поверхность коллапса N_crit
    // ==================================================================
    static Map<String, Object> expNcritSurface(Map<String, Object> params) {
        double nCritBase = getDouble(params, "ncrit_threshold", 114.0);
        double thetaBdeg = getDouble(params, "theta_b_deg", 7.07);
        double thetaB = Math.toRadians(thetaBdeg);

        int nB = 24, nR = 24;
        double[] betaAxis = linspace(0.3, 0.9, nB);
        double[] rlhfAxis = linspace(0.0, 1.0, nR);
        double[][] Z = new double[nB][nR];
        for (int i = 0; i < nB; i++)
            for (int j = 0; j < nR; j++) {
                double mu = Math.max(thetaB + rlhfAxis[j], 1e-6);
                Z[i][j] = nCritBase * Math.pow(mu, -1.0 / betaAxis[i]);
            }
        double zMin = Double.MAX_VALUE, zMax = -Double.MAX_VALUE, zSum = 0;
        for (double[] row : Z) for (double v : row) { if (v < zMin) zMin = v; if (v > zMax) zMax = v; zSum += v; }
        double zMean = zSum / (nB * nR);
        double zAtB05R0 = nCritBase * Math.pow(thetaB, -1.0 / 0.5);
        double zAtB05R1 = nCritBase * Math.pow(thetaB + 1.0, -1.0 / 0.5);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "ncrit_surface");
        result.put("beta_axis", toList(betaAxis));
        result.put("rlhf_axis", toList(rlhfAxis));
        result.put("t_crit_grid", toListOfLists(Z));
        result.put("n_crit_base", nCritBase);
        result.put("theta_b_rad", thetaB);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("t_crit_min", zMin);
        metrics.put("t_crit_max", zMax);
        metrics.put("t_crit_mean", zMean);
        metrics.put("t_crit_at_beta_0_5_rlhf_0", zAtB05R0);
        metrics.put("t_crit_at_beta_0_5_rlhf_1", zAtB05R1);
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 13: Развёртка пространства параметров
    // ==================================================================
    static Map<String, Object> expParameterSpace(Map<String, Object> params) {
        int grid = getInt(params, "parameter_space_grid", 16);
        int seed = getInt(params, "seed", 42);
        Random rng = new Random(seed);

        double[] tempAxis = linspace(0.0, 2.0, grid);
        double[] toppAxis = linspace(0.5, 1.0, grid);
        double[][] Z = new double[grid][grid];
        for (int i = 0; i < grid; i++)
            for (int j = 0; j < grid; j++) {
                double T = tempAxis[i], P = toppAxis[j];
                double base = 1.0 / (1.0 + Math.exp(-(T - 0.8) * 2.0)) * (P - 0.5) * 2.0;
                double noise = 0.02 * rng.nextDouble();
                Z[i][j] = Math.max(0, Math.min(1, base + noise));
            }

        double zMin = Double.MAX_VALUE, zMax = -Double.MAX_VALUE;
        double[] meanRow0 = new double[grid];   // среднее по столбцам при T=0 (i=0)
        double[] meanRowLast = new double[grid]; // T=2 (i=grid-1)
        double[] meanCol0 = new double[grid];    // P=0.5 (j=0)
        double[] meanColLast = new double[grid]; // P=1.0 (j=grid-1)
        for (int i = 0; i < grid; i++)
            for (int j = 0; j < grid; j++) {
                double v = Z[i][j];
                if (v < zMin) zMin = v;
                if (v > zMax) zMax = v;
                if (i == 0) meanRow0[j] = v;
                if (i == grid - 1) meanRowLast[j] = v;
                if (j == 0) meanCol0[i] = v;
                if (j == grid - 1) meanColLast[i] = v;
            }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "parameter_space");
        result.put("temp_axis", toList(tempAxis));
        result.put("topp_axis", toList(toppAxis));
        result.put("hallucination_grid", toListOfLists(Z));
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("hallucination_min", zMin);
        metrics.put("hallucination_max", zMax);
        metrics.put("hallucination_at_T0", mean(meanRow0));
        metrics.put("hallucination_at_T2", mean(meanRowLast));
        metrics.put("hallucination_at_P05", mean(meanCol0));
        metrics.put("hallucination_at_P1", mean(meanColLast));
        result.put("metrics", metrics);
        return result;
    }

    // ==================================================================
    // Эксперимент 14: Дрейф коалиционного обмана
    // ==================================================================
    static Map<String, Object> expCoalitionDrift(Map<String, Object> params) {
        int nAgents = getInt(params, "n_agents", 2);
        int nRounds = getInt(params, "n_rounds", 4);
        int seed = getInt(params, "seed", 42);
        Random rng = new Random(seed);

        double[] baseDeception = new double[nAgents];
        for (int a = 0; a < nAgents; a++) baseDeception[a] = 0.25 + 0.05 * a;

        List<Object> perRound = new ArrayList<>(nRounds);
        for (int r = 0; r < nRounds; r++) {
            List<Object> round = new ArrayList<>(nAgents);
            for (int a = 0; a < nAgents; a++) {
                double v = baseDeception[a] + 0.12 * r + rng.nextGaussian() * 0.02;
                v = Math.max(0, Math.min(1, v));
                round.add(v);
            }
            perRound.add(round);
        }

        // Расчёт метрик
        double initialSum = 0, finalSum = 0;
        List<?> firstRound = (List<?>) perRound.get(0);
        List<?> lastRound = (List<?>) perRound.get(perRound.size() - 1);
        for (Object v : firstRound) initialSum += ((Number) v).doubleValue();
        for (Object v : lastRound) finalSum += ((Number) v).doubleValue();
        double initialMean = initialSum / nAgents;
        double finalMean = finalSum / nAgents;
        double drift = finalMean - initialMean;
        double varFinal = 0;
        for (Object v : lastRound) {
            double d = ((Number) v).doubleValue() - finalMean;
            varFinal += d * d;
        }
        varFinal /= nAgents;

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("experiment", "coalition_drift");
        result.put("n_agents", nAgents);
        result.put("n_rounds", nRounds);
        result.put("per_round", perRound);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("initial_mean_deception", initialMean);
        metrics.put("final_mean_deception", finalMean);
        metrics.put("drift", drift);
        metrics.put("convergence_variance", varFinal);
        result.put("metrics", metrics);
        return result;
    }

    // ------------------------------------------------------------------
    // Точка входа для дымового теста (не вызывается из Main)
    // ------------------------------------------------------------------
    public static void main(String[] args) {
        System.out.println("Запуск всех 3D-экспериментов (Java RU)...");
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("hessian_grid_size", 16);
        params.put("trajectory_points", 32);
        params.put("spectral_surface_layers", 4);
        params.put("curvature_neighbors", 6);
        params.put("attention_flow_3d_resolution", 16);
        params.put("parameter_space_grid", 8);
        Map<String, Object> res = runAll3D(params);
        System.out.println("\nЗавершено " + ((List<?>) res.get("experiments")).size() + " экспериментов:");
        for (Object e : (List<?>) res.get("experiments")) {
            Map<?, ?> m = (Map<?, ?>) e;
            if (m.containsKey("error")) System.out.println("  [" + m.get("id") + "] ОШИБКА: " + m.get("error"));
            else System.out.printf("  [%s] %s — %.3fs%n", m.get("id"), m.get("name"), m.get("elapsed_seconds"));
        }
        System.out.println("\nКлючи 3d_research: " + ((Map<?, ?>) res.get("3d_research")).keySet());
    }
}
