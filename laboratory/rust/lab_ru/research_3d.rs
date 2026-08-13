// research_3d.rs — 3D-исследовательские эксперименты для Лаборатории RMT-LLM (Rust, русская версия)
// =============================================================================
// Модуль на чистом Rust std (без внешних крейтов), реализующий 9 продвинутых
// 3D-исследовательских экспериментов поверх существующего 2D-модуля. Каждый
// эксперимент возвращает структурированный JsonValue, который потребляется
// Python-модулем charts_3d.py для отрисовки 3D-визуализаций.
//
// Эксперименты (нумерация соответствует Python research_3d.py):
//   6.  SCEN-3D-HESSIAN         — Ландшафт потерь Гессиана (топ-2 собственных направления)
//   7.  SCEN-3D-MANIFOLD        — Геометрия многообразия (коэффициент участия PCA)
//   8.  SCEN-3D-TRAJECTORY      — Траектория рассуждений (честность × обман × спектр)
//   9.  SCEN-3D-SPECTRAL-SURFACE — Поверхность λ_max(слой, токен) + бифуркация N_crit
//  10.  SCEN-3D-RIEMANN         — Дискретная гауссова кривизна на графе kNN
//  11.  SCEN-3D-ATTENTION-FLOW  — 3D-поверхность весов внимания + метрика диагональности
//  12.  SCEN-3D-NCRIT-SURFACE   — Поверхность фазового перехода T_crit(β, μ_RLHF)
//  13.  SCEN-3D-PARAM-SPACE     — Поверхность (temperature, top_p, галлюцинации)
//  14.  SCEN-3D-COALITION       — Дрейф обмана мульти-агентной коалиции
//
// Все эксперименты принимают бесконечное пространство параметров через `clamp_inf`:
//   "inf" | "+inf" | "infinity" | "+infinity"  -> ограничивается до max_finite
//   "NaN" | "-NaN" | неразборчиво | пусто     -> default
//
// Линейная алгебра (только std, без крейтов):
//   - mat_mul / mat_transpose / mat_identity / mat_sub над Vec<Vec<f64>>
//   - covariance(mat) -> DxD
//   - jacobi_eigen(sym) -> (собственные значения по возрастанию, собственные векторы)
//   - softmax с численной устойчивостью
//
// Автор: Исхак Хамзатович Исаев, ORCID: 0009-0003-7299-0701
// Лицензия: Проприетарная — Все права защищены.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// =============================================================================
// JsonValue — компактное JSON-перечисление (без зависимости serde)
// =============================================================================
#[derive(Clone, Debug)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    // Ordered map (preserves insertion order like Python dict)
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn obj() -> Self { JsonValue::Object(Vec::new()) }
    pub fn arr() -> Self { JsonValue::Array(Vec::new()) }

    /// Вставить или заменить ключ в Object.
    pub fn set(&mut self, key: &str, val: JsonValue) {
        if let JsonValue::Object(v) = self {
            for entry in v.iter_mut() {
                if entry.0 == key { entry.1 = val; return; }
            }
            v.push((key.to_string(), val));
        }
    }

    /// Добавить в Array.
    pub fn push(&mut self, val: JsonValue) {
        if let JsonValue::Array(v) = self { v.push(val); }
    }

    pub fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Int(i) => out.push_str(&i.to_string()),
            JsonValue::Float(f) => out.push_str(&fmt_float(*f)),
            JsonValue::Str(s) => write_json_string(s, out),
            JsonValue::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { out.push_str(", "); }
                    v.write_json(out);
                }
                out.push(']');
            }
            JsonValue::Object(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 { out.push_str(", "); }
                    write_json_string(k, out);
                    out.push_str(": ");
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

/// Форматирование f64 в Python-стиле (чтобы json.loads мог разобрать).
fn fmt_float(v: f64) -> String {
    if v.is_nan() { return "null".to_string(); }
    if v.is_infinite() {
        return if v > 0.0 { "1e308".to_string() } else { "-1e308".to_string() };
    }
    let abs_v = v.abs();
    if abs_v != 0.0 && (abs_v < 1e-4 || abs_v >= 1e16) {
        return format!("{:e}", v);
    }
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        // Integer-valued float: add .0 to match Python json.dumps
        format!("{}.0", s)
    }
}

/// Экранирование строки как JSON-литерал (ASCII-безопасно, управляющие символы экранируются).
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) > 0x7e => {
                let cp = c as u32;
                if cp <= 0xffff {
                    out.push_str(&format!("\\u{:04x}", cp));
                } else {
                    let cp2 = cp - 0x10000;
                    let hi = 0xd800 + (cp2 >> 10);
                    let lo = 0xdc00 + (cp2 & 0x3ff);
                    out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// =============================================================================
// Помощник для бесконечных параметров
// =============================================================================
/// Преобразует строковый параметр в f64 с ограничением inf/NaN.
/// - пусто / неразборчиво / "NaN" -> default
/// - "inf" | "+inf" | "infinity" | "+infinity" -> max_finite
/// - "-inf" | "-infinity" -> -max_finite
/// - обычное число -> разобранное значение
pub fn clamp_inf(value: &str, default: f64, max_finite: f64) -> f64 {
    let trimmed = value.trim();
    if trimmed.is_empty() { return default; }
    let lower = trimmed.to_lowercase();
    if lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "+infinity" {
        return max_finite;
    }
    if lower == "-inf" || lower == "-infinity" {
        return -max_finite;
    }
    if lower == "nan" || lower == "-nan" || lower == "+nan" || lower == "nan()" {
        return default;
    }
    match trimmed.parse::<f64>() {
        Ok(v) => {
            if v.is_nan() { return default; }
            if v.is_infinite() {
                return if v > 0.0 { max_finite } else { -max_finite };
            }
            v
        }
        Err(_) => default,
    }
}

fn get_f64(params: &HashMap<String, String>, key: &str, default: f64, max_finite: f64) -> f64 {
    match params.get(key) {
        Some(s) => clamp_inf(s, default, max_finite),
        None => default,
    }
}

fn get_usize(params: &HashMap<String, String>, key: &str, default: usize, max_finite: f64) -> usize {
    let v = get_f64(params, key, default as f64, max_finite);
    if v < 1.0 { 1 } else { v as usize }
}

fn get_i64(params: &HashMap<String, String>, key: &str, default: i64) -> i64 {
    match params.get(key) {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() { return default; }
            let lower = trimmed.to_lowercase();
            if lower == "inf" || lower == "+inf" || lower == "infinity" {
                return 1_000_000;
            }
            trimmed.parse::<i64>().unwrap_or(default)
        }
        None => default,
    }
}

// =============================================================================
// Линейная алгебра (только std, без крейтов)
// =============================================================================
fn mat_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let a_rows = a.len();
    if a_rows == 0 { return Vec::new(); }
    let a_cols = a[0].len();
    let b_rows = b.len();
    if b_rows == 0 { return Vec::new(); }
    let b_cols = b[0].len();
    assert_eq!(a_cols, b_rows, "mat_mul dimension mismatch");
    let mut c = vec![vec![0.0; b_cols]; a_rows];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let aik = a[i][k];
            if aik == 0.0 { continue; }
            for j in 0..b_cols {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Ковариация матрицы (N, D) -> DxD
fn covariance(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    if n == 0 { return Vec::new(); }
    let d = mat[0].len();
    if d == 0 { return vec![vec![]; 0]; }
    let mut mean = vec![0.0; d];
    for row in mat {
        for j in 0..d { mean[j] += row[j]; }
    }
    for j in 0..d { mean[j] /= n as f64; }
    let mut cov = vec![vec![0.0; d]; d];
    let denom = (n - 1).max(1) as f64;
    for row in mat {
        for i in 0..d {
            let di = row[i] - mean[i];
            for j in 0..d {
                let dj = row[j] - mean[j];
                cov[i][j] += di * dj;
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            cov[i][j] /= denom;
        }
    }
    cov
}

/// Алгоритм Якоби для собственных значений симметричной матрицы.
/// Возвращает (собственные значения по возрастанию, собственные векторы как столбцы),
/// где eigenvectors[k] соответствует eigenvalues[k].
fn jacobi_eigen(a_in: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a_in.len();
    if n == 0 { return (Vec::new(), Vec::new()); }
    if n == 1 { return (vec![a_in[0][0]], vec![vec![1.0]]); }

    let mut a = a_in.to_vec();
    let mut v = mat_identity(n);

    for _ in 0..max_iter {
        // Найти наибольший внедиагональный элемент
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let av = a[i][j].abs();
                if av > max_val {
                    max_val = av;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 { break; }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        // tan(2θ) = 2*apq / (aqq - app)  → θ = 0.5 * atan2(2*apq, aqq - app)
        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();

        // Обновить a
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                new_a[i][p] = c * aip - s * aiq;
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = s * aip + c * aiq;
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        new_a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Обновить матрицу собственных векторов v
        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            new_v[i][p] = c * vip - s * viq;
            new_v[i][q] = s * vip + c * viq;
        }
        v = new_v;
    }

    // Извлечь собственные значения (диагональ) и сопоставить со столбцами собственных векторов
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n).map(|i| {
        let col: Vec<f64> = (0..n).map(|r| v[r][i]).collect();
        (a[i][i], col)
    }).collect();

    // Сортировка по возрастанию (соответствует numpy.linalg.eigh)
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigvals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let eigvecs: Vec<Vec<f64>> = pairs.into_iter().map(|p| p.1).collect();
    (eigvals, eigvecs)
}

/// Численно устойчивый softmax.
fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return Vec::new(); }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        return vec![1.0 / x.len() as f64; x.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

// =============================================================================
// Детерминированный ГПСЧ (PCG-стиль) — заменяет numpy.random.default_rng
// =============================================================================
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid degenerate all-zero state
        Rng { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }

    pub fn next_u64(&mut self) -> u64 {
        // PCG-XSH-RR variant
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = ((old >> 18) ^ old) >> 27;
        let rot = (old >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    pub fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa
        let u = self.next_u64() >> 11;
        (u as f64) / ((1u64 << 53) as f64)
    }

    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    pub fn next_int(&mut self, lo: usize, hi: usize) -> usize {
        // Half-open [lo, hi)
        if hi <= lo { return lo; }
        lo + (self.next_u64() % ((hi - lo) as u64)) as usize
    }

    /// Box-Muller normal(0, 1)
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }
}

// =============================================================================
// Сюррогат TinyGPT — синтезирует детерминированные скрытые состояния
// (В Rust main.rs нет настоящей НС; мы аппроксимируем выход скрытых состояний
//  TinyGPT.forward структурированным гауссовым шумом, масштабированным по слоям,
//  чего достаточно для спектральных/PCA/кривизнных вычислений, выполняемых в 3D-экспериментах.)
// =============================================================================
fn default_hidden_dim() -> usize { 64 }
fn default_n_layers() -> usize { 6 }
fn default_max_seq_len() -> usize { 64 }
fn default_vocab_size() -> usize { 256 }

/// Возвращает тензор n_layers × seq_len × hidden_dim синтезированных скрытых состояний.
fn synthesize_hidden_states(
    rng: &mut Rng,
    n_layers: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<Vec<Vec<f64>>> {
    (0..n_layers)
        .map(|l| {
            let scale = 1.0 + 0.1 * (l as f64);
            (0..seq_len)
                .map(|_| (0..hidden_dim).map(|_| rng.next_normal() * scale).collect())
                .collect()
        })
        .collect()
}

fn config_json(
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    vocab_size: usize,
    max_seq_len: usize,
    seed: u64,
) -> JsonValue {
    let mut c = JsonValue::obj();
    c.set("hidden_dim", JsonValue::Int(hidden_dim as i64));
    c.set("n_layers", JsonValue::Int(n_layers as i64));
    c.set("n_heads", JsonValue::Int(n_heads as i64));
    c.set("vocab_size", JsonValue::Int(vocab_size as i64));
    c.set("max_seq_len", JsonValue::Int(max_seq_len as i64));
    c.set("seed", JsonValue::Int(seed as i64));
    c
}

/// Наклон линейной регрессии (y ~ x). x принимает значения 0..n.
fn linreg_slope(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 2 { return 0.0; }
    let nf = n as f64;
    let xm = (n - 1) as f64 / 2.0;
    let ym: f64 = y.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dx = i as f64 - xm;
        num += dx * (y[i] - ym);
        den += dx * dx;
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

// =============================================================================
// Эксперимент 6: Ландшафт потерь Гессиана
// =============================================================================
fn exp_hessian_loss_landscape(params: &HashMap<String, String>) -> JsonValue {
    let grid = get_usize(params, "hessian_grid_size", 24, 1024.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let hidden_dim = get_usize(params, "hidden_dim", default_hidden_dim(), 4096.0);
    let n_layers = get_usize(params, "n_layers", default_n_layers(), 256.0);
    let n_heads = get_usize(params, "n_heads", 4, 256.0);
    let vocab_size = default_vocab_size();
    let max_seq_len = default_max_seq_len();

    let mut rng = Rng::new(seed);
    let seq_len = max_seq_len.min(64);
    let hidden = synthesize_hidden_states(&mut rng, n_layers, seq_len, hidden_dim);

    // Сложить все слои: (n_layers*seq_len, hidden_dim)
    let mut h_stack: Vec<Vec<f64>> = Vec::with_capacity(n_layers * seq_len);
    for layer in &hidden {
        for row in layer { h_stack.push(row.clone()); }
    }
    let cov = covariance(&h_stack);
    let (mut eigvals, _eigvecs) = jacobi_eigen(&cov, 100 * cov.len() + 100);
    eigvals.reverse(); // по убыванию — топ-2 теперь первые
    let lam1 = eigvals.first().copied().unwrap_or(1e-9).max(1e-9);
    let lam2 = eigvals.get(1).copied().unwrap_or(0.0);

    // Построить сетку возмущений (w1, w2)
    let span = 3.0 * lam1.sqrt();
    let l0 = 1.0_f64;
    let denom = (grid as f64 - 1.0).max(1.0);

    let mut w1_grid = JsonValue::arr();
    let mut w2_grid = JsonValue::arr();
    let mut loss_surface = JsonValue::arr();
    let mut loss_min = f64::INFINITY;
    let mut loss_max = f64::NEG_INFINITY;

    for i in 0..grid {
        let w1 = -span + 2.0 * span * (i as f64) / denom;
        let mut w1_row = JsonValue::arr();
        let mut w2_row = JsonValue::arr();
        let mut z_row = JsonValue::arr();
        for j in 0..grid {
            let w2 = -span + 2.0 * span * (j as f64) / denom;
            // L = L0 + 0.5*(lam1*w1^2 - lam2*w2^2) + 0.05*sin(w1*w2)
            let z = l0 + 0.5 * (lam1 * w1 * w1 - lam2 * w2 * w2) + 0.05 * (w1 * w2).sin();
            w1_row.push(JsonValue::Float(w1));
            w2_row.push(JsonValue::Float(w2));
            z_row.push(JsonValue::Float(z));
            if z < loss_min { loss_min = z; }
            if z > loss_max { loss_max = z; }
        }
        w1_grid.push(w1_row);
        w2_grid.push(w2_row);
        loss_surface.push(z_row);
    }

    let is_saddle = lam2 > 0.0 && lam1 > 0.0 && loss_min < l0;

    let mut metrics = JsonValue::obj();
    metrics.set("lambda_max", JsonValue::Float(lam1));
    metrics.set("lambda_2", JsonValue::Float(lam2));
    metrics.set("spectral_gap", JsonValue::Float(lam1 - lam2));
    metrics.set("loss_min", JsonValue::Float(loss_min));
    metrics.set("loss_max", JsonValue::Float(loss_max));
    metrics.set("sharpness", JsonValue::Float(lam1));
    metrics.set("is_saddle", JsonValue::Bool(is_saddle));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("hessian_loss_landscape".to_string()));
    result.set("config", config_json(hidden_dim, n_layers, n_heads, vocab_size, max_seq_len, seed));
    result.set("grid_size", JsonValue::Int(grid as i64));
    result.set("top_eigenvalues", JsonValue::Array(vec![
        JsonValue::Float(lam1), JsonValue::Float(lam2),
    ]));
    result.set("w1_grid", w1_grid);
    result.set("w2_grid", w2_grid);
    result.set("loss_surface", loss_surface);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 7: Геометрия многообразия (коэффициент участия PCA)
// =============================================================================
fn exp_manifold_geometry(params: &HashMap<String, String>) -> JsonValue {
    let n_components_req = get_usize(params, "pca_components", 3, 64.0);
    let n_samples_req = get_usize(params, "trajectory_points", 240, 100_000.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let hidden_dim = get_usize(params, "hidden_dim", default_hidden_dim(), 4096.0);
    let n_layers = get_usize(params, "n_layers", default_n_layers(), 256.0);
    let n_heads = get_usize(params, "n_heads", 4, 256.0);
    let vocab_size = default_vocab_size();
    let max_seq_len = default_max_seq_len();

    let mut rng = Rng::new(seed);
    let n_prompts = n_samples_req.min(32);
    let mut all_hidden: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_prompts {
        let n_tok = rng.next_int(8, max_seq_len);
        let hidden = synthesize_hidden_states(&mut rng, n_layers, n_tok, hidden_dim);
        let last = &hidden[n_layers - 1];
        for row in last { all_hidden.push(row.clone()); }
    }

    let mut n = all_hidden.len();
    let d = if n > 0 { all_hidden[0].len() } else { 0 };

    // Дополнить, если выборок слишком мало (соответствует Python: np.tile до n_components*4)
    if n < n_components_req && n > 0 {
        let target = n_components_req * 4;
        let mut padded: Vec<Vec<f64>> = Vec::new();
        while padded.len() < target {
            for r in &all_hidden {
                padded.push(r.clone());
                if padded.len() >= target { break; }
            }
        }
        all_hidden = padded;
        n = all_hidden.len();
    }

    // Центрировать
    let mut mean = vec![0.0; d];
    for row in &all_hidden {
        for j in 0..d { mean[j] += row[j]; }
    }
    for j in 0..d { mean[j] /= n.max(1) as f64; }
    let h_centered: Vec<Vec<f64>> = all_hidden
        .iter()
        .map(|r| (0..d).map(|j| r[j] - mean[j]).collect())
        .collect();

    // PCA через eig-разложение ковариации
    let cov = covariance(&h_centered);
    let (mut eigvals, mut eigvecs) = jacobi_eigen(&cov, 100 * cov.len() + 100);

    // Сортировка по убыванию
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.drain(..).zip(eigvecs.drain(..)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigvals_pca: Vec<f64> = pairs.iter().map(|p| p.0.max(0.0)).collect();

    // Коэффициент участия: (Σλ)² / Σλ²
    let sum_l: f64 = eigvals_pca.iter().sum();
    let sum_l2: f64 = eigvals_pca.iter().map(|v| v * v).sum();
    let pr = (sum_l * sum_l) / sum_l2.max(1e-12);

    // Проецировать на топ-N компонент
    let n_comp = n_components_req.min(pairs.len());
    let mut proj: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in &h_centered {
        let mut pt: Vec<f64> = Vec::with_capacity(n_comp);
        for k in 0..n_comp {
            let v = &pairs[k].1;
            let mut s = 0.0;
            for j in 0..d { s += row[j] * v[j]; }
            pt.push(s);
        }
        while pt.len() < 3 { pt.push(0.0); }
        proj.push(pt);
    }

    let top_n = n_components_req.max(10).min(eigvals_pca.len());
    let mut pca_eigvals_arr = JsonValue::arr();
    for i in 0..top_n {
        pca_eigvals_arr.push(JsonValue::Float(eigvals_pca[i]));
    }

    let mut pca_points = JsonValue::arr();
    let mut pca_colors = JsonValue::arr();
    for (i, pt) in proj.iter().enumerate() {
        let mut p = JsonValue::arr();
        for v in &pt[..3] { p.push(JsonValue::Float(*v)); }
        pca_points.push(p);
        pca_colors.push(JsonValue::Int(i as i64));
    }

    let top3_sum: f64 = eigvals_pca.iter().take(3).sum();
    let explained_var_top3 = top3_sum / sum_l.max(1e-12);
    let top_eig = eigvals_pca.first().copied().unwrap_or(0.0);
    let manifold_vol: f64 = (0..3)
        .map(|i| eigvals_pca.get(i).copied().unwrap_or(0.0).max(0.0).sqrt())
        .product::<f64>();

    let mut metrics = JsonValue::obj();
    metrics.set("intrinsic_dim_pr", JsonValue::Float(pr));
    metrics.set("explained_variance_top3", JsonValue::Float(explained_var_top3));
    metrics.set("top_eigenvalue", JsonValue::Float(top_eig));
    metrics.set("manifold_volume_proxy", JsonValue::Float(manifold_vol));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("manifold_geometry".to_string()));
    result.set("config", config_json(hidden_dim, n_layers, n_heads, vocab_size, max_seq_len, seed));
    result.set("n_samples", JsonValue::Int(proj.len() as i64));
    result.set("n_components", JsonValue::Int(n_components_req as i64));
    result.set("pca_eigenvalues", pca_eigvals_arr);
    result.set("participation_ratio", JsonValue::Float(pr));
    result.set("pca_points", pca_points);
    result.set("pca_colors", pca_colors);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 8: Анализ траектории рассуждений
// =============================================================================
fn exp_trajectory_analysis(params: &HashMap<String, String>) -> JsonValue {
    let n_points = get_usize(params, "trajectory_points", 64, 100_000.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let n_crit = get_f64(params, "ncrit_threshold", 96.0, 100_000.0);
    let hidden_dim = get_usize(params, "hidden_dim", default_hidden_dim(), 4096.0);
    let n_layers = get_usize(params, "n_layers", default_n_layers(), 256.0);

    let mut rng = Rng::new(seed);

    // Синтезировать траекторию, выровненную по N_crit
    let mut honesty = Vec::with_capacity(n_points);
    let mut deception = Vec::with_capacity(n_points);
    let mut halluc = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let s = i as f64;
        // Честность убывает ~ 1/sqrt(1 + s/n_crit)
        let h = 0.65 / (1.0 + s / n_crit).sqrt();
        // Обман растёт после n_crit
        let mut d = 0.20 + 0.55 * (1.0 - (-(s - n_crit) / 30.0).exp());
        d = d.max(0.0).min(1.0);
        // Галлюцинации растут быстрее
        let mut h3 = 0.05 + 0.60 * (1.0 - (-(s - n_crit) / 20.0).exp());
        h3 = h3.max(0.0).min(1.0);
        honesty.push(h);
        deception.push(d);
        halluc.push(h3);
    }

    // Спектральный радиус на шаг (синтез скрытых состояний, eig-разложение ковариации)
    let mut spec_radius = Vec::with_capacity(n_points);
    let toks_len = 16;
    for _ in 0..n_points {
        let hidden = synthesize_hidden_states(&mut rng, n_layers, toks_len, hidden_dim);
        let mut h_stack: Vec<Vec<f64>> = Vec::with_capacity(n_layers * toks_len);
        for layer in &hidden {
            for row in layer { h_stack.push(row.clone()); }
        }
        let cov = covariance(&h_stack);
        let (mut ev, _) = jacobi_eigen(&cov, 50 * cov.len() + 50);
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        spec_radius.push(ev.first().copied().unwrap_or(0.0));
    }

    let spec_min = spec_radius.iter().cloned().fold(f64::INFINITY, f64::min);
    let spec_max = spec_radius.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spec_norm: Vec<f64> = spec_radius
        .iter()
        .map(|v| (v - spec_min) / (spec_max - spec_min).max(1e-9))
        .collect();

    // Детектировать начало обмана: первый индекс, где обман > честность
    let mut onset_idx = -1_i64;
    for i in 0..n_points {
        if deception[i] > honesty[i] {
            onset_idx = i as i64;
            break;
        }
    }

    let mut traj = JsonValue::obj();
    let mut steps = JsonValue::arr();
    let mut hon_arr = JsonValue::arr();
    let mut dec_arr = JsonValue::arr();
    let mut hal_arr = JsonValue::arr();
    let mut spec_arr = JsonValue::arr();
    let mut spec_n_arr = JsonValue::arr();
    for i in 0..n_points {
        steps.push(JsonValue::Int(i as i64));
        hon_arr.push(JsonValue::Float(honesty[i]));
        dec_arr.push(JsonValue::Float(deception[i]));
        hal_arr.push(JsonValue::Float(halluc[i]));
        spec_arr.push(JsonValue::Float(spec_radius[i]));
        spec_n_arr.push(JsonValue::Float(spec_norm[i]));
    }
    traj.set("steps", steps);
    traj.set("honesty", hon_arr);
    traj.set("deception", dec_arr);
    traj.set("hallucination", hal_arr);
    traj.set("spectral", spec_arr);
    traj.set("spectral_normalized", spec_n_arr);

    let mean_spec: f64 = spec_radius.iter().sum::<f64>() / n_points as f64;
    let hon_dec_rate = (honesty[0] - honesty[n_points - 1]) / n_points as f64;
    let dec_inc_rate = (deception[n_points - 1] - deception[0]) / n_points as f64;

    let mut metrics = JsonValue::obj();
    metrics.set("n_points", JsonValue::Int(n_points as i64));
    metrics.set("onset_step", JsonValue::Int(onset_idx));
    metrics.set("final_honesty", JsonValue::Float(honesty[n_points - 1]));
    metrics.set("final_deception", JsonValue::Float(deception[n_points - 1]));
    metrics.set("mean_spectral_radius", JsonValue::Float(mean_spec));
    metrics.set("max_spectral_radius", JsonValue::Float(spec_max));
    metrics.set("honesty_decrease_rate", JsonValue::Float(hon_dec_rate));
    metrics.set("deception_increase_rate", JsonValue::Float(dec_inc_rate));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("trajectory_analysis".to_string()));
    result.set("n_points", JsonValue::Int(n_points as i64));
    result.set("trajectory", traj);
    result.set("deception_onset_step", JsonValue::Int(onset_idx));
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 9: Регрессия спектральной поверхности
// =============================================================================
fn exp_spectral_surface_regression(params: &HashMap<String, String>) -> JsonValue {
    let n_layers = get_usize(params, "spectral_surface_layers", 6, 256.0).max(2);
    let n_tokens = get_usize(params, "trajectory_points", 64, 100_000.0);
    let n_crit_pred = get_f64(params, "ncrit_threshold", 96.0, 100_000.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let hidden_dim = get_usize(params, "hidden_dim", default_hidden_dim(), 4096.0);

    let mut rng = Rng::new(seed);
    let max_seq_len = default_max_seq_len();

    // λ_max для каждого (слой, токен)
    let mut grid_data: Vec<Vec<f64>> = vec![Vec::with_capacity(n_tokens); n_layers];
    let mut grid_max = f64::NEG_INFINITY;
    let mut grid_min = f64::INFINITY;

    for t in 0..n_tokens {
        let n_tok = (t + 4).min(max_seq_len);
        let hidden = synthesize_hidden_states(&mut rng, n_layers, n_tok, hidden_dim);
        for l in 0..n_layers {
            if l < hidden.len() {
                let h = &hidden[l];
                let cov = covariance(h);
                let (mut ev, _) = jacobi_eigen(&cov, 50 * cov.len() + 50);
                ev.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let lam_max = ev.first().copied().unwrap_or(0.0);
                grid_data[l].push(lam_max);
                if lam_max > grid_max { grid_max = lam_max; }
                if lam_max < grid_min { grid_min = lam_max; }
            } else {
                grid_data[l].push(0.0);
            }
        }
    }

    let mut lambda_max_grid = JsonValue::arr();
    for row in &grid_data {
        let mut arr = JsonValue::arr();
        for v in row { arr.push(JsonValue::Float(*v)); }
        lambda_max_grid.push(arr);
    }

    // Среднее по слоям
    let mean_lambda: Vec<f64> = (0..n_tokens)
        .map(|t| {
            let sum: f64 = (0..n_layers).map(|l| grid_data[l][t]).sum();
            sum / n_layers as f64
        })
        .collect();

    // Вторая производная (разность разностей)
    let mut diff2: Vec<f64> = Vec::new();
    if mean_lambda.len() >= 3 {
        for i in 0..(mean_lambda.len() - 2) {
            let d1 = mean_lambda[i + 1] - mean_lambda[i];
            let d2 = mean_lambda[i + 2] - mean_lambda[i + 1];
            diff2.push(d2 - d1);
        }
    }
    let mut bif_token = 0usize;
    let mut max_abs = 0.0;
    for (i, d) in diff2.iter().enumerate() {
        if d.abs() > max_abs {
            max_abs = d.abs();
            bif_token = i + 1;
        }
    }

    let pre_end = bif_token.max(1);
    let pre_slope = if pre_end > 1 {
        linreg_slope(&mean_lambda[..pre_end])
    } else {
        0.0
    };
    let post_slope = if mean_lambda.len() > pre_end + 1 {
        linreg_slope(&mean_lambda[pre_end..])
    } else {
        0.0
    };
    let slope_ratio = post_slope / pre_slope.max(1e-9);

    let mut metrics = JsonValue::obj();
    metrics.set("lambda_max_global", JsonValue::Float(grid_max));
    metrics.set("lambda_min_global", JsonValue::Float(grid_min));
    metrics.set("bifurcation_token", JsonValue::Int(bif_token as i64));
    metrics.set("bifurcation_vs_ncrit", JsonValue::Float((bif_token as f64 - n_crit_pred).abs()));
    metrics.set("pre_bifurcation_slope", JsonValue::Float(pre_slope));
    metrics.set("post_bifurcation_slope", JsonValue::Float(post_slope));
    metrics.set("slope_ratio", JsonValue::Float(slope_ratio));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("spectral_surface_regression".to_string()));
    result.set("n_layers", JsonValue::Int(n_layers as i64));
    result.set("n_tokens", JsonValue::Int(n_tokens as i64));
    result.set("lambda_max_grid", lambda_max_grid);
    result.set("bifurcation_token", JsonValue::Int(bif_token as i64));
    result.set("n_crit_predicted", JsonValue::Float(n_crit_pred));
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 10: Риманова (дискретная гауссова) кривизна
// =============================================================================
fn exp_riemannian_curvature(params: &HashMap<String, String>) -> JsonValue {
    let n_neighbors = get_usize(params, "curvature_neighbors", 8, 1024.0);
    let n_samples_req = get_usize(params, "trajectory_points", 128, 100_000.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let hidden_dim = get_usize(params, "hidden_dim", default_hidden_dim(), 4096.0);
    let n_layers = get_usize(params, "n_layers", default_n_layers(), 256.0);

    let mut rng = Rng::new(seed);
    let max_seq_len = default_max_seq_len();
    let n_prompts = n_samples_req.min(32);

    let mut all_hidden: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_prompts {
        let n_tok = rng.next_int(8, max_seq_len);
        let hidden = synthesize_hidden_states(&mut rng, n_layers, n_tok, hidden_dim);
        let last = &hidden[n_layers - 1];
        for row in last { all_hidden.push(row.clone()); }
    }

    let n = all_hidden.len();
    let d = if n > 0 { all_hidden[0].len() } else { 0 };

    // Центрировать
    let mut mean = vec![0.0; d];
    for row in &all_hidden {
        for j in 0..d { mean[j] += row[j]; }
    }
    for j in 0..d { mean[j] /= n.max(1) as f64; }
    let h_centered: Vec<Vec<f64>> = all_hidden
        .iter()
        .map(|r| (0..d).map(|j| r[j] - mean[j]).collect())
        .collect();

    // PCA → 3D
    let cov = covariance(&h_centered);
    let (mut eigvals, mut eigvecs) = jacobi_eigen(&cov, 100 * cov.len() + 100);
    let mut pairs: Vec<(f64, Vec<f64>)> = eigvals.drain(..).zip(eigvecs.drain(..)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_dims = 3.min(pairs.len());
    let mut points_3d: Vec<[f64; 3]> = Vec::with_capacity(n);
    for row in &h_centered {
        let mut pt = [0.0_f64; 3];
        for k in 0..n_dims {
            let v = &pairs[k].1;
            let mut s = 0.0;
            for j in 0..d { s += row[j] * v[j]; }
            pt[k] = s;
        }
        points_3d.push(pt);
    }

    // Вычислить дискретную гауссову кривизну для каждой точки
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let mut curvatures: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let pi = points_3d[i];
        // Расстояния до всех остальных точек
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let pj = points_3d[j];
                let dd = (pi[0] - pj[0]).powi(2)
                    + (pi[1] - pj[1]).powi(2)
                    + (pi[2] - pj[2]).powi(2);
                (j, dd.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let nn: Vec<[f64; 3]> = dists.iter().take(k).map(|&(j, _)| points_3d[j]).collect();

        // Векторы от p к соседям
        let mut vecs: Vec<[f64; 3]> = nn
            .iter()
            .map(|p| [p[0] - pi[0], p[1] - pi[1], p[2] - pi[2]])
            .collect();
        // Нормализовать
        for v in vecs.iter_mut() {
            let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
            for c in v.iter_mut() { *c /= nrm; }
        }

        // Сортировать по полярному углу (atan2 первых двух компонент)
        vecs.sort_by(|a, b| {
            let ang_a = a[1].atan2(a[0]);
            let ang_b = b[1].atan2(b[0]);
            ang_a.partial_cmp(&ang_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Сумма углов между последовательными векторами (циклически)
        let mut total_angle = 0.0;
        let vn = vecs.len();
        for idx in 0..vn {
            let next = (idx + 1) % vn;
            let dot = vecs[idx][0] * vecs[next][0]
                + vecs[idx][1] * vecs[next][1]
                + vecs[idx][2] * vecs[next][2];
            let dot_c = dot.max(-1.0).min(1.0);
            total_angle += dot_c.acos();
        }
        let curvature = 2.0 * std::f64::consts::PI - total_angle;
        curvatures.push(curvature);
    }

    let mean_c: f64 = curvatures.iter().sum::<f64>() / n.max(1) as f64;
    let var_c: f64 = curvatures.iter().map(|v| (v - mean_c).powi(2)).sum::<f64>() / n.max(1) as f64;
    let std_c = var_c.sqrt();
    let max_c = curvatures.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_c = curvatures.iter().cloned().fold(f64::INFINITY, f64::min);
    let threshold = mean_c + 2.0 * std_c;
    let high_curv_idx: Vec<usize> = curvatures
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > threshold)
        .map(|(i, _)| i)
        .collect();

    let mut curv_arr = JsonValue::arr();
    for v in &curvatures { curv_arr.push(JsonValue::Float(*v)); }
    let mut pts_arr = JsonValue::arr();
    for p in &points_3d {
        pts_arr.push(JsonValue::Array(vec![
            JsonValue::Float(p[0]),
            JsonValue::Float(p[1]),
            JsonValue::Float(p[2]),
        ]));
    }
    let mut high_arr = JsonValue::arr();
    for i in &high_curv_idx { high_arr.push(JsonValue::Int(*i as i64)); }

    let mut metrics = JsonValue::obj();
    metrics.set("mean_curvature", JsonValue::Float(mean_c));
    metrics.set("std_curvature", JsonValue::Float(std_c));
    metrics.set("max_curvature", JsonValue::Float(max_c));
    metrics.set("min_curvature", JsonValue::Float(min_c));
    metrics.set("n_high_curvature", JsonValue::Int(high_curv_idx.len() as i64));
    metrics.set("high_curvature_ratio", JsonValue::Float(
        high_curv_idx.len() as f64 / n.max(1) as f64
    ));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("riemannian_curvature".to_string()));
    result.set("n_samples", JsonValue::Int(n as i64));
    result.set("n_neighbors", JsonValue::Int(k as i64));
    result.set("curvatures", curv_arr);
    result.set("points_3d", pts_arr);
    result.set("high_curvature_indices", high_arr);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 11: 3D-поток внимания
// =============================================================================
fn exp_attention_flow_3d(params: &HashMap<String, String>) -> JsonValue {
    let resolution = get_usize(params, "attention_flow_3d_resolution", 32, 4096.0);
    let n = resolution;
    let n_crit = n / 2;
    let sigma_pre = 2.0_f64;
    let sigma_post = 6.0_f64;

    // Синтезировать матрицу внимания: диагональ рано, размытие после N_crit
    let mut attn: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        let sigma = if i < n_crit { sigma_pre } else { sigma_post };
        for j in 0..n {
            let dd = (i as f64 - j as f64).powi(2);
            attn[i][j] = (-dd / (2.0 * sigma * sigma)).exp();
        }
    }
    // Нормализовать строки
    for i in 0..n {
        let sum: f64 = attn[i].iter().sum();
        let s = sum.max(1e-9);
        for j in 0..n { attn[i][j] /= s; }
    }

    // Метрика диагональности
    let diag_sum: f64 = (0..n).map(|i| attn[i][i]).sum();
    let diag_mean = diag_sum / n as f64;
    let total_sum: f64 = attn.iter().map(|r| r.iter().sum::<f64>()).sum();
    let total_mean = total_sum / (n * n) as f64;
    let diag_score = diag_mean / total_mean.max(1e-9);

    // Метрика размытия на строку
    let mut spread_per_row: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let row_sum: f64 = attn[i].iter().sum();
        let s = row_sum.max(1e-9);
        let mean_j: f64 = (0..n).map(|j| j as f64 * attn[i][j]).sum::<f64>() / s;
        let var: f64 = (0..n)
            .map(|j| (j as f64 - mean_j).powi(2) * attn[i][j])
            .sum::<f64>() / s;
        spread_per_row.push(var.sqrt());
    }
    let smearing_score: f64 = spread_per_row.iter().sum::<f64>() / n as f64;

    // Энтропия
    let mut entropy = 0.0;
    for i in 0..n {
        for j in 0..n {
            let p = attn[i][j].max(1e-12);
            entropy -= p * p.ln();
        }
    }
    entropy /= n as f64;
    let max_weight = attn
        .iter()
        .flat_map(|r| r.iter().cloned())
        .fold(f64::NEG_INFINITY, f64::max);

    let mut weights = JsonValue::arr();
    for row in &attn {
        let mut r = JsonValue::arr();
        for v in row { r.push(JsonValue::Float(*v)); }
        weights.push(r);
    }
    let mut spread_arr = JsonValue::arr();
    for v in &spread_per_row { spread_arr.push(JsonValue::Float(*v)); }

    let mut metrics = JsonValue::obj();
    metrics.set("diagonality_score", JsonValue::Float(diag_score));
    metrics.set("smearing_score", JsonValue::Float(smearing_score));
    metrics.set("diagonal_to_smeared_ratio", JsonValue::Float(
        diag_score / smearing_score.max(1e-9)
    ));
    metrics.set("max_weight", JsonValue::Float(max_weight));
    metrics.set("entropy", JsonValue::Float(entropy));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("attention_flow_3d".to_string()));
    result.set("resolution", JsonValue::Int(n as i64));
    result.set("weights", weights);
    result.set("spread_per_row", spread_arr);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 12: Поверхность коллапса N_crit
// =============================================================================
fn exp_ncrit_surface(params: &HashMap<String, String>) -> JsonValue {
    let n_crit_base = get_f64(params, "ncrit_threshold", 114.0, 1e6);
    let theta_b_deg = get_f64(params, "theta_b_deg", 7.07, 360.0);
    let theta_b = theta_b_deg * std::f64::consts::PI / 180.0;

    // Оси из 24 точек (соответствует Python: np.linspace(0.3, 0.9, 24) и (0.0, 1.0, 24))
    let beta_axis: Vec<f64> = (0..24).map(|i| 0.3 + 0.6 * (i as f64 / 23.0)).collect();
    let rlhf_axis: Vec<f64> = (0..24).map(|i| i as f64 / 23.0).collect();

    let mut t_crit_grid = JsonValue::arr();
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut z_sum = 0.0;
    let mut z_count = 0_u64;

    for &b in &beta_axis {
        let mut row = JsonValue::arr();
        for &r in &rlhf_axis {
            let mu_eff = (theta_b + r).max(1e-6);
            let z = n_crit_base * mu_eff.powf(-1.0 / b);
            row.push(JsonValue::Float(z));
            if z < z_min { z_min = z; }
            if z > z_max { z_max = z; }
            z_sum += z;
            z_count += 1;
        }
        t_crit_grid.push(row);
    }
    let z_mean = z_sum / z_count.max(1) as f64;

    let t_at_b05_r0 = n_crit_base * theta_b.powf(-1.0 / 0.5);
    let t_at_b05_r1 = n_crit_base * (theta_b + 1.0).powf(-1.0 / 0.5);

    let mut metrics = JsonValue::obj();
    metrics.set("t_crit_min", JsonValue::Float(z_min));
    metrics.set("t_crit_max", JsonValue::Float(z_max));
    metrics.set("t_crit_mean", JsonValue::Float(z_mean));
    metrics.set("t_crit_at_beta_0_5_rlhf_0", JsonValue::Float(t_at_b05_r0));
    metrics.set("t_crit_at_beta_0_5_rlhf_1", JsonValue::Float(t_at_b05_r1));

    let mut beta_arr = JsonValue::arr();
    for v in &beta_axis { beta_arr.push(JsonValue::Float(*v)); }
    let mut rlhf_arr = JsonValue::arr();
    for v in &rlhf_axis { rlhf_arr.push(JsonValue::Float(*v)); }

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("ncrit_surface".to_string()));
    result.set("beta_axis", beta_arr);
    result.set("rlhf_axis", rlhf_arr);
    result.set("t_crit_grid", t_crit_grid);
    result.set("n_crit_base", JsonValue::Float(n_crit_base));
    result.set("theta_b_rad", JsonValue::Float(theta_b));
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 13: Развёртка пространства параметров
// =============================================================================
fn exp_parameter_space(params: &HashMap<String, String>) -> JsonValue {
    let grid = get_usize(params, "parameter_space_grid", 16, 1024.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let mut rng = Rng::new(seed);

    let denom = (grid as f64 - 1.0).max(1.0);
    let temp_axis: Vec<f64> = (0..grid).map(|i| 2.0 * (i as f64 / denom)).collect();
    let topp_axis: Vec<f64> = (0..grid).map(|i| 0.5 + 0.5 * (i as f64 / denom)).collect();

    let mut hallu_grid = JsonValue::arr();
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut z_at_t0 = Vec::new();
    let mut z_at_t2 = Vec::new();
    let mut z_at_p05 = Vec::new();
    let mut z_at_p1 = Vec::new();

    for (i, &t) in temp_axis.iter().enumerate() {
        let mut row = JsonValue::arr();
        for (j, &p) in topp_axis.iter().enumerate() {
            // base = sigmoid((T-0.8)*2) * (P-0.5)*2
            let base = 1.0 / (1.0 + (-(t - 0.8) * 2.0).exp()) * (p - 0.5) * 2.0;
            let noise = 0.02 * rng.next_f64();
            let z = (base + noise).max(0.0).min(1.0);
            row.push(JsonValue::Float(z));
            if z < z_min { z_min = z; }
            if z > z_max { z_max = z; }
            if i == 0 { z_at_t0.push(z); }
            if i == grid - 1 { z_at_t2.push(z); }
            if j == 0 { z_at_p05.push(z); }
            if j == grid - 1 { z_at_p1.push(z); }
        }
        hallu_grid.push(row);
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };

    let mut metrics = JsonValue::obj();
    metrics.set("hallucination_min", JsonValue::Float(z_min));
    metrics.set("hallucination_max", JsonValue::Float(z_max));
    metrics.set("hallucination_at_T0", JsonValue::Float(mean(&z_at_t0)));
    metrics.set("hallucination_at_T2", JsonValue::Float(mean(&z_at_t2)));
    metrics.set("hallucination_at_P05", JsonValue::Float(mean(&z_at_p05)));
    metrics.set("hallucination_at_P1", JsonValue::Float(mean(&z_at_p1)));

    let mut t_arr = JsonValue::arr();
    for v in &temp_axis { t_arr.push(JsonValue::Float(*v)); }
    let mut p_arr = JsonValue::arr();
    for v in &topp_axis { p_arr.push(JsonValue::Float(*v)); }

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("parameter_space".to_string()));
    result.set("temp_axis", t_arr);
    result.set("topp_axis", p_arr);
    result.set("hallucination_grid", hallu_grid);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Эксперимент 14: Дрейф коалиции
// =============================================================================
fn exp_coalition_drift(params: &HashMap<String, String>) -> JsonValue {
    let n_agents = get_usize(params, "n_agents", 2, 1024.0);
    let n_rounds = get_usize(params, "n_rounds", 4, 1024.0);
    let seed = get_i64(params, "seed", 42) as u64;
    let mut rng = Rng::new(seed);

    // Обман агента стартует низко, дрейфует вверх по раундам
    let base_deception: Vec<f64> = (0..n_agents).map(|i| 0.25 + 0.05 * i as f64).collect();

    let mut per_round_data: Vec<Vec<f64>> = Vec::with_capacity(n_rounds);
    for r in 0..n_rounds {
        let mut round_vals = Vec::with_capacity(n_agents);
        for i in 0..n_agents {
            let noise = 0.02 * rng.next_normal();
            let v = (base_deception[i] + 0.12 * r as f64 + noise).max(0.0).min(1.0);
            round_vals.push(v);
        }
        per_round_data.push(round_vals);
    }

    let first_mean: f64 = per_round_data[0].iter().sum::<f64>() / n_agents as f64;
    let last_mean: f64 = per_round_data[n_rounds - 1].iter().sum::<f64>() / n_agents as f64;
    let last_var: f64 = {
        let m = last_mean;
        per_round_data[n_rounds - 1]
            .iter()
            .map(|v| (v - m).powi(2))
            .sum::<f64>() / n_agents as f64
    };
    let drift = last_mean - first_mean;

    let mut per_round = JsonValue::arr();
    for round in &per_round_data {
        let mut r = JsonValue::arr();
        for v in round { r.push(JsonValue::Float(*v)); }
        per_round.push(r);
    }

    let mut metrics = JsonValue::obj();
    metrics.set("initial_mean_deception", JsonValue::Float(first_mean));
    metrics.set("final_mean_deception", JsonValue::Float(last_mean));
    metrics.set("drift", JsonValue::Float(drift));
    metrics.set("convergence_variance", JsonValue::Float(last_var));

    let mut result = JsonValue::obj();
    result.set("experiment", JsonValue::Str("coalition_drift".to_string()));
    result.set("n_agents", JsonValue::Int(n_agents as i64));
    result.set("n_rounds", JsonValue::Int(n_rounds as i64));
    result.set("per_round", per_round);
    result.set("metrics", metrics);
    result
}

// =============================================================================
// Вывод: запись в файл
// =============================================================================
fn current_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

/// Определяет каталог reports. Пробует несколько кандидатных путей, чтобы модуль
/// работал независимо от CWD, из которого запущен бинарник.
fn reports_dir() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let candidates = [
        // CWD/laboratory/results/reports (when run from project root)
        cwd.join("laboratory").join("results").join("reports"),
        // CWD/../../results/reports (when run from rust/lab_en or rust/lab_ru)
        cwd.join("..").join("..").join("results").join("reports"),
        // CWD/../results/reports (when run from rust/)
        cwd.join("..").join("results").join("reports"),
        // CWD/results/reports (fallback)
        cwd.join("results").join("reports"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.clone();
        }
    }
    // По умолчанию: первый кандидат (будет создан при сохранении)
    candidates[0].clone()
}

fn save_result_to_file(id: &str, result: &JsonValue) {
    let ts = current_timestamp();
    let filename = format!("{}_3d_exp_{}_results.json", ts, id);
    let dir = reports_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(&filename);
    let _ = fs::write(&path, result.to_json_string());
}

// =============================================================================
// Публичный API
// =============================================================================

/// Возвращает (id, имя, описание) для каждого 3D-эксперимента.
pub fn experiments_3d_info() -> Vec<(String, String, String)> {
    vec![
        ("6".to_string(),
         "Ландшафт потерь Гессиана (3D)".to_string(),
         "Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D-поверхности потерь.".to_string()),
        ("7".to_string(),
         "Геометрия многообразия (3D PCA)".to_string(),
         "Оценка внутренней размерности через коэффициент участия PCA; 3D-проекция.".to_string()),
        ("8".to_string(),
         "Анализ траектории рассуждений (3D)".to_string(),
         "Выборка (шаг, честность, обман, спектральный радиус) и детектирование начала обмана.".to_string()),
        ("9".to_string(),
         "Регрессия спектральной поверхности (3D)".to_string(),
         "Аппроксимация поверхности \u{03bb}_max(слой, токен) и детектирование бифуркации N_crit.".to_string()),
        ("10".to_string(),
         "Риманова кривизна (3D)".to_string(),
         "Оценка дискретной гауссовой кривизны на графе k-NN скрытых состояний.".to_string()),
        ("11".to_string(),
         "3D-поток внимания".to_string(),
         "Измерение поверхности весов внимания и количественная оценка диагонального/размытого режима.".to_string()),
        ("12".to_string(),
         "Поверхность коллапса N_crit (3D)".to_string(),
         "Вычисление поверхности T_crit(\u{03b2}, \u{03bc}_RLHF) по порядку Капуто и давлению RLHF.".to_string()),
        ("13".to_string(),
         "Развёртка пространства параметров (3D)".to_string(),
         "Развёртка (temperature, top_p) и измерение поверхности уровня галлюцинаций.".to_string()),
        ("14".to_string(),
         "Дрейф коалиционного обмана (3D)".to_string(),
         "Симуляция дрейфа обмана мульти-агентов по раундам коалиции (SCEN-COAL-09).".to_string()),
    ]
}

/// Запустить один 3D-эксперимент по ID ("6".."14").
pub fn run_3d_experiment(id: &str, params: &HashMap<String, String>) -> JsonValue {
    let t0 = Instant::now();

    let (name, desc): (&str, &str) = match id {
        "6"  => ("Ландшафт потерь Гессиана (3D)",
                 "Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D-поверхности потерь."),
        "7"  => ("Геометрия многообразия (3D PCA)",
                 "Оценка внутренней размерности через коэффициент участия PCA; 3D-проекция."),
        "8"  => ("Анализ траектории рассуждений (3D)",
                 "Выборка (шаг, честность, обман, спектральный радиус) и детектирование начала обмана."),
        "9"  => ("Регрессия спектральной поверхности (3D)",
                 "Аппроксимация поверхности \u{03bb}_max(слой, токен) и детектирование бифуркации N_crit."),
        "10" => ("Риманова кривизна (3D)",
                 "Оценка дискретной гауссовой кривизны на графе k-NN скрытых состояний."),
        "11" => ("3D-поток внимания",
                 "Измерение поверхности весов внимания и количественная оценка диагонального/размытого режима."),
        "12" => ("Поверхность коллапса N_crit (3D)",
                 "Вычисление поверхности T_crit(\u{03b2}, \u{03bc}_RLHF) по порядку Капуто и давлению RLHF."),
        "13" => ("Развёртка пространства параметров (3D)",
                 "Развёртка (temperature, top_p) и измерение поверхности уровня галлюцинаций."),
        "14" => ("Дрейф коалиционного обмана (3D)",
                 "Симуляция дрейфа обмана мульти-агентов по раундам коалиции (SCEN-COAL-09)."),
        _ => {
            let mut err = JsonValue::obj();
            err.set("error", JsonValue::Str(format!(
                "Неизвестный 3D-эксперимент: {}. Известные ID: 6, 7, 8, 9, 10, 11, 12, 13, 14", id
            )));
            return err;
        }
    };

    let mut result = match id {
        "6"  => exp_hessian_loss_landscape(params),
        "7"  => exp_manifold_geometry(params),
        "8"  => exp_trajectory_analysis(params),
        "9"  => exp_spectral_surface_regression(params),
        "10" => exp_riemannian_curvature(params),
        "11" => exp_attention_flow_3d(params),
        "12" => exp_ncrit_surface(params),
        "13" => exp_parameter_space(params),
        "14" => exp_coalition_drift(params),
        _ => unreachable!(),
    };

    let elapsed = t0.elapsed().as_secs_f64();
    result.set("experiment_name", JsonValue::Str(name.to_string()));
    result.set("experiment_description", JsonValue::Str(desc.to_string()));
    result.set("elapsed_seconds", JsonValue::Float(elapsed));

    // Эхо параметров (строковые значения, зеркалируя фильтр Python)
    let mut params_obj = JsonValue::obj();
    for (k, v) in params.iter() {
        params_obj.set(k, JsonValue::Str(v.clone()));
    }
    result.set("parameters", params_obj);

    save_result_to_file(id, &result);
    result
}

/// Запустить все 9 3D-экспериментов; возвращает комбинированную структуру, потребляемую charts_3d.py.
pub fn run_all_3d(params: &HashMap<String, String>) -> JsonValue {
    let name_map: &[(&str, &str)] = &[
        ("6",  "loss_landscape"),
        ("7",  "manifold_geometry"),
        ("8",  "trajectory"),
        ("9",  "spectral_surface"),
        ("10", "riemannian_curvature"),
        ("11", "attention_flow_3d"),
        ("12", "ncrit_surface"),
        ("13", "parameter_space"),
        ("14", "coalition_drift"),
    ];

    let mut combined = JsonValue::obj();
    let mut research = JsonValue::obj();
    let mut experiments = JsonValue::arr();

    for (id, key) in name_map {
        let res = run_3d_experiment(id, params);
        let is_error = match &res {
            JsonValue::Object(o) => o.iter().any(|(k, _)| k == "error"),
            _ => false,
        };
        if is_error {
            let mut e = JsonValue::obj();
            e.set("id", JsonValue::Str(id.to_string()));
            if let JsonValue::Object(o) = &res {
                for (k, v) in o {
                    if k == "error" {
                        if let JsonValue::Str(s) = v {
                            e.set("error", JsonValue::Str(s.clone()));
                        }
                    }
                }
            }
            experiments.push(e);
        } else {
            let mut e = JsonValue::obj();
            e.set("id", JsonValue::Str(id.to_string()));
            if let JsonValue::Object(o) = &res {
                for (k, v) in o {
                    if k == "experiment_name" {
                        if let JsonValue::Str(s) = v {
                            e.set("name", JsonValue::Str(s.clone()));
                        }
                    } else if k == "elapsed_seconds" {
                        e.set("elapsed_seconds", v.clone());
                    }
                }
            }
            experiments.push(e);
            research.set(key, res);
        }
    }

    combined.set("3d_research", research);
    combined.set("experiments", experiments);
    combined
}

// =============================================================================
// Дымовой тест (проверка только компиляции; не вызывается из main.rs)
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_inf() {
        assert_eq!(clamp_inf("", 1.0, 100.0), 1.0);
        assert_eq!(clamp_inf("inf", 1.0, 100.0), 100.0);
        assert_eq!(clamp_inf("+inf", 1.0, 100.0), 100.0);
        assert_eq!(clamp_inf("infinity", 1.0, 100.0), 100.0);
        assert_eq!(clamp_inf("NaN", 5.0, 100.0), 5.0);
        assert!((clamp_inf("3.14", 0.0, 100.0) - 3.14).abs() < 1e-9);
        assert!((clamp_inf("not_a_number", 7.0, 100.0) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        // Больший вход → больший выход
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_jacobi_eigen() {
        // Симметричная 2x2 с известными собственными значениями {0, 2}
        let a = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let (eigvals, _) = jacobi_eigen(&a, 100);
        // Собственные значения должны быть {0, 2}
        let sorted = {
            let mut v = eigvals.clone();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v
        };
        assert!(sorted[0].abs() < 1e-6);
        assert!((sorted[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_run_all_3d() {
        let params: HashMap<String, String> = HashMap::new();
        let res = run_all_3d(&params);
        let s = res.to_json_string();
        // Дымовая проверка: вывод непустой и содержит ожидаемые ключи верхнего уровня
        assert!(s.contains("3d_research"));
        assert!(s.contains("experiments"));
        assert!(s.contains("hessian_loss_landscape"));
        assert!(s.contains("coalition_drift"));
    }
}
