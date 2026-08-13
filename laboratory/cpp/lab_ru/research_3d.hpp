// =============================================================================
// research_3d.hpp — 3D исследовательские эксперименты для лаборатории RMT-LLM (C++, v1.1.0)
// =============================================================================
//
// Header-only модуль (чистый C++17, без внешних зависимостей), реализующий
// девять 3D исследовательских экспериментов по образцу Python (research_3d.py):
//
//   6. SCEN-3D-HESSIAN        — Ландшафт потерь Гессиана (топ-2 собств. направления)
//   7. SCEN-3D-MANIFOLD       — Геометрия многообразия (коэффициент участия PCA)
//   8. SCEN-3D-TRAJECTORY     — Траектория рассуждений (честность/обман/спектр)
//   9. SCEN-3D-SPECTRAL-SURFACE — Поверхность λ_max(слой, токен) + бифуркация N_crit
//  10. SCEN-3D-RIEMANN        — Риманова кривизна (дискретная Гауссова, kNN-граф)
//  11. SCEN-3D-ATTENTION-FLOW — 3D-поверхность весов внимания, метрика диагональности
//  12. SCEN-3D-NCRIT-SURFACE  — T_crit(N) = N·(θ_b+μ)^(-1/β) поверхность фазового перехода
//  13. SCEN-3D-PARAM-SPACE    — 3D-поверхность (температура, top_p, скорость совпадения)
//  14. SCEN-3D-COALITION      — Дрейф мультитокенной коалиции по слоям
//
// Каждый эксперимент:
//   - Принимает const std::map<std::string, std::string>& params (конвенция
//     бесконечных параметров; "inf" / "+inf" / "infinity" / "NaN"
//     поддерживаются через clamp_inf()).
//   - Вычисляет ту же математику, что и research_3d.py.
//   - Записывает JSON-файл в laboratory/results/reports/{ts}_3d_exp_{id}_results.json
//   - Возвращает содержимое JSON как std::string.
//
// Вся линейная алгебра (умножение/транспонирование матриц, единичная матрица,
// ковариация, алгоритм Якоби для симметричных матриц, численно-устойчивый
// softmax) реализована с нуля на std::vector<std::vector<double>>.
//
// Автор: Исхак Хамзатович Исаев
// ORCID: 0009-0003-7299-0701
// Лицензия: Проприетарная — Все права защищены.
// =============================================================================
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace research_3d {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr double PI = 3.14159265358979323846;

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// JSON helpers — manual serialisation via std::ostringstream (no deps)
// ---------------------------------------------------------------------------
inline std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    oss << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b";  break;
            case '\f': oss << "\\f";  break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c);
                    oss << std::dec << std::setfill(' ');
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    oss << '"';
    return oss.str();
}

inline std::string json_double(double v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    std::ostringstream oss;
    if (v == std::floor(v) && std::abs(v) < 1e16) {
        oss << std::fixed << std::setprecision(1) << v;
    } else {
        oss << std::setprecision(15) << v;
    }
    return oss.str();
}

inline std::string json_array(const std::vector<double>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << json_double(v[i]);
    }
    oss << ']';
    return oss.str();
}

inline std::string json_array_2d(const std::vector<std::vector<double>>& m) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < m.size(); ++i) {
        if (i) oss << ',';
        oss << json_array(m[i]);
    }
    oss << ']';
    return oss.str();
}

inline std::string json_int_array(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << v[i];
    }
    oss << ']';
    return oss.str();
}

inline std::string json_int_array_2d(const std::vector<std::vector<int>>& m) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < m.size(); ++i) {
        if (i) oss << ',';
        oss << json_int_array(m[i]);
    }
    oss << ']';
    return oss.str();
}

// Helper: build a JSON object body from (key -> raw_json_value) pairs.
struct JsonField { std::string key; std::string raw_value; };

inline std::string build_object(const std::vector<JsonField>& fields) {
    std::ostringstream oss;
    oss << '{';
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i) oss << ',';
        oss << json_escape(fields[i].key) << ':' << fields[i].raw_value;
    }
    oss << '}';
    return oss.str();
}

inline std::string field_str(const std::string& k, const std::string& v) {
    return build_object({{k, json_escape(v)}});
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------
inline fs::path find_lab_base() {
    fs::path exe = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        if (fs::exists(exe / "laboratory") && fs::is_directory(exe / "laboratory")) {
            return exe;
        }
        if (exe == exe.parent_path()) break;
        exe = exe.parent_path();
    }
    return fs::current_path();
}

inline fs::path reports_dir_3d() {
    return find_lab_base() / "laboratory" / "results" / "reports";
}

inline std::string timestamp_compact() {
    auto t = std::time(nullptr);
    char ts[24];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
    return std::string(ts);
}

inline void write_text_file(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << content;
}

inline void write_3d_report(int exp_id, const std::string& json) {
    fs::path dir = reports_dir_3d();
    fs::create_directories(dir);
    std::string fname = timestamp_compact() + "_3d_exp_"
                       + std::to_string(exp_id) + "_results.json";
    write_text_file(dir / fname, json);
}

// ---------------------------------------------------------------------------
// Parameter helpers — infinite-parameter convention
// ---------------------------------------------------------------------------
inline double clamp_inf(const std::string& value, double default_val,
                        double max_finite = 1e6) {
    if (value.empty()) return default_val;
    // Lowercase compare for inf / +inf / infinity / -inf / nan
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "+infinity") {
        return max_finite;
    }
    if (lower == "-inf" || lower == "-infinity") {
        return -max_finite;
    }
    if (lower == "nan" || lower == "+nan" || lower == "-nan") {
        return max_finite;
    }
    try {
        size_t pos = 0;
        double v = std::stod(value, &pos);
        // Allow trailing whitespace
        while (pos < value.size() && std::isspace(static_cast<unsigned char>(value[pos]))) ++pos;
        if (pos != value.size()) return default_val;
        if (std::isnan(v) || std::isinf(v)) return max_finite;
        return v;
    } catch (...) {
        return default_val;
    }
}

inline double get_param(const std::map<std::string, std::string>& params,
                        const std::string& key, double default_val,
                        double max_finite = 1e6) {
    auto it = params.find(key);
    if (it == params.end()) return default_val;
    return clamp_inf(it->second, default_val, max_finite);
}

inline int get_param_int(const std::map<std::string, std::string>& params,
                         const std::string& key, int default_val,
                         int max_finite = 1000000) {
    auto it = params.find(key);
    if (it == params.end()) return default_val;
    const std::string& v = it->second;
    if (v.empty()) return default_val;
    std::string lower;
    for (char c : v) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (lower == "inf" || lower == "+inf" || lower == "infinity" || lower == "nan") {
        return max_finite;
    }
    try {
        size_t pos = 0;
        double d = std::stod(v, &pos);
        while (pos < v.size() && std::isspace(static_cast<unsigned char>(v[pos]))) ++pos;
        if (pos != v.size()) return default_val;
        if (std::isnan(d) || std::isinf(d)) return max_finite;
        return static_cast<int>(d);
    } catch (...) {
        return default_val;
    }
}

inline std::string params_to_json(const std::map<std::string, std::string>& params) {
    std::vector<JsonField> fields;
    fields.reserve(params.size());
    for (const auto& kv : params) {
        fields.push_back({kv.first, json_escape(kv.second)});
    }
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
inline std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(static_cast<size_t>(n));
    if (n <= 0) return v;
    if (n == 1) { v[0] = lo; return v; }
    double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n; ++i) v[i] = lo + i * step;
    return v;
}

inline double sigmoid(double x) {
    if (x >= 0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        double z = std::exp(x);
        return z / (1.0 + z);
    }
}

inline std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) return {};
    double mx = x[0];
    for (double v : x) if (v > mx) mx = v;
    std::vector<double> e(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e[i] = std::exp(x[i] - mx);
        sum += e[i];
    }
    if (sum <= 0.0) {
        std::fill(e.begin(), e.end(), 1.0 / e.size());
        return e;
    }
    for (size_t i = 0; i < x.size(); ++i) e[i] /= sum;
    return e;
}

inline double vec_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

inline double vec_std(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vec_mean(v);
    double ss = 0.0;
    for (double x : v) ss += (x - m) * (x - m);
    return std::sqrt(ss / v.size());
}

inline double vec_max(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x > m) m = x;
    return m;
}

inline double vec_min(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double m = v[0];
    for (double x : v) if (x < m) m = x;
    return m;
}

inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// ---------------------------------------------------------------------------
// Linear algebra — std::vector<std::vector<double>> matrices (row-major)
// ---------------------------------------------------------------------------
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

inline Matrix mat_zeros(int rows, int cols) {
    return Matrix(static_cast<size_t>(rows), Vector(static_cast<size_t>(cols), 0.0));
}

inline Matrix mat_identity(int n) {
    Matrix M = mat_zeros(n, n);
    for (int i = 0; i < n; ++i) M[i][i] = 1.0;
    return M;
}

inline Matrix mat_transpose(const Matrix& A) {
    if (A.empty()) return {};
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(A[0].size());
    Matrix T = mat_zeros(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            T[j][i] = A[i][j];
    return T;
}

inline Matrix mat_mul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty()) return {};
    int rows = static_cast<int>(A.size());
    int inner = static_cast<int>(A[0].size());
    int cols = static_cast<int>(B[0].size());
    Matrix C = mat_zeros(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < inner; ++k) {
            double a = A[i][k];
            if (a == 0.0) continue;
            for (int j = 0; j < cols; ++j) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

// Matrix * vector
inline Vector mat_vec(const Matrix& A, const Vector& x) {
    if (A.empty()) return {};
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(A[0].size());
    Vector y(rows, 0.0);
    for (int i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int j = 0; j < cols; ++j) s += A[i][j] * x[j];
        y[i] = s;
    }
    return y;
}

// Symmetrize: C = (C + C^T) / 2
inline Matrix mat_symmetrize(const Matrix& A) {
    int n = static_cast<int>(A.size());
    Matrix S = mat_zeros(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            S[i][j] = 0.5 * (A[i][j] + A[j][i]);
        }
    }
    return S;
}

// Covariance of (N x D) data matrix. Returns D x D.
inline Matrix covariance(const Matrix& data) {
    if (data.empty()) return {};
    int N = static_cast<int>(data.size());
    int D = static_cast<int>(data[0].size());
    // Mean
    Vector mean(D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) mean[j] += data[i][j];
    for (int j = 0; j < D; ++j) mean[j] /= std::max(N, 1);
    // Centered^T * Centered / (N-1)
    Matrix cov = mat_zeros(D, D);
    double denom = std::max(N - 1, 1);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            double cj = data[i][j] - mean[j];
            for (int k = 0; k < D; ++k) {
                cov[j][k] += cj * (data[i][k] - mean[k]);
            }
        }
    }
    for (int j = 0; j < D; ++j)
        for (int k = 0; k < D; ++k) cov[j][k] /= denom;
    return cov;
}

// ---------------------------------------------------------------------------
// Jacobi eigenvalue algorithm for symmetric matrices.
// Returns (eigvals_ascending, eigvecs) where eigvecs[r][i] is the r-th
// component of the i-th eigenvector (i.e. columns of eigvecs are eigenvectors),
// matching numpy.linalg.eigh ordering (smallest eigenvalue first).
// ---------------------------------------------------------------------------
struct EigenResult {
    Vector eigvals;       // ascending
    Matrix eigvecs;       // columns are eigenvectors
};

inline EigenResult jacobi_eigen(Matrix A, int max_iter = 100, double tol = 1e-12) {
    int n = static_cast<int>(A.size());
    if (n == 0) return {{}, {}};
    // Ensure symmetric
    A = mat_symmetrize(A);
    if (n == 1) return {{A[0][0]}, {{1.0}}};
    Matrix V = mat_identity(n);

    for (int iter = 0; iter < max_iter * n; ++iter) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        double max_off = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double a = std::abs(A[i][j]);
                if (a > max_off) { max_off = a; p = i; q = j; }
            }
        }
        if (max_off < tol) break;

        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];
        double phi;
        if (std::abs(app - aqq) < 1e-30) {
            phi = (apq >= 0 ? 0.25 : -0.25) * PI;
        } else {
            phi = 0.5 * std::atan2(2.0 * apq, app - aqq);
        }
        double c = std::cos(phi);
        double s = std::sin(phi);

        // Canonical Jacobi rotation: A_new = J^T A J where
        //   J = [[c, -s], [s, c]] on the (p, q) 2x2 block.
        // Update only the rows/cols that change (i != p, q).
        for (int i = 0; i < n; ++i) {
            if (i == p || i == q) continue;
            double aip = A[i][p];
            double aiq = A[i][q];
            double new_aip = c * aip + s * aiq;
            double new_aiq = -s * aip + c * aiq;
            A[i][p] = new_aip;
            A[p][i] = new_aip;
            A[i][q] = new_aiq;
            A[q][i] = new_aiq;
        }
        // Diagonal updates (closed-form for the 2x2 block)
        A[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        // Eigenvector accumulator: V = V J
        for (int i = 0; i < n; ++i) {
            double vip = V[i][p];
            double viq = V[i][q];
            V[i][p] = c * vip + s * viq;
            V[i][q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues, sort ascending
    Vector eigvals(n);
    for (int i = 0; i < n; ++i) eigvals[i] = A[i][i];

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return eigvals[a] < eigvals[b]; });

    EigenResult out;
    out.eigvals.resize(n);
    out.eigvecs = mat_zeros(n, n);
    for (int new_i = 0; new_i < n; ++new_i) {
        int old_i = idx[new_i];
        out.eigvals[new_i] = eigvals[old_i];
        for (int r = 0; r < n; ++r) out.eigvecs[r][new_i] = V[r][old_i];
    }
    return out;
}

// Project centered data onto top-K principal components.
// data: N x D. Returns N x K (where K <= D).
inline Matrix pca_project(const Matrix& data, int k) {
    if (data.empty()) return {};
    int N = static_cast<int>(data.size());
    int D = static_cast<int>(data[0].size());
    // Center
    Vector mean(D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) mean[j] += data[i][j];
    for (int j = 0; j < D; ++j) mean[j] /= N;
    Matrix centered = data;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) centered[i][j] -= mean[j];
    // Covariance D x D
    Matrix cov = covariance(data);  // covariance centers internally
    EigenResult er = jacobi_eigen(cov);
    // Top-K eigenvectors = last K columns of er.eigvecs
    int K = std::min(k, D);
    Matrix Vtop = mat_zeros(D, K);
    for (int r = 0; r < D; ++r) {
        for (int j = 0; j < K; ++j) {
            Vtop[r][j] = er.eigvecs[r][D - K + j];
        }
    }
    // proj = centered * Vtop (N x K)
    Matrix proj = mat_mul(centered, Vtop);
    return proj;
}

// ---------------------------------------------------------------------------
// Synthetic hidden-state generator — replaces TinyGPT.forward for spectral
// experiments. Produces per-layer (seq x hidden_dim) matrices with low-rank
// structure + position signal + Gaussian noise so covariance eigenspectra are
// meaningful (top eigenvalues dominate, rest form MP-like bulk).
// ---------------------------------------------------------------------------
struct SynthConfig {
    int hidden_dim = 64;
    int n_layers   = 6;
    int n_heads    = 4;
    int vocab_size = 256;
    int max_seq_len = 64;
    int seed       = 42;
};

inline SynthConfig synth_config_from_params(const std::map<std::string, std::string>& params) {
    SynthConfig c;
    c.hidden_dim  = get_param_int(params, "hidden_dim", 64);
    c.n_layers    = get_param_int(params, "n_layers", 6);
    c.n_heads     = get_param_int(params, "n_heads", 4);
    c.vocab_size  = get_param_int(params, "vocab_size", 256);
    c.max_seq_len = get_param_int(params, "max_seq_len", 64);
    c.seed        = get_param_int(params, "seed", 42);
    return c;
}

// Generate one hidden-state matrix (seq x hidden_dim) for a given layer.
inline Matrix synth_hidden(int seq, int hidden_dim, int layer, int seed) {
    Matrix H = mat_zeros(seq, hidden_dim);
    std::mt19937 rng(static_cast<uint32_t>(seed + layer * 1000 + 17));
    std::normal_distribution<double> normal(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    // Low-rank basis: 3 dominant directions, scaled so top eigenvalues dominate.
    const int rank = std::min(3, hidden_dim);
    Matrix basis = mat_zeros(rank, hidden_dim);
    for (int r = 0; r < rank; ++r)
        for (int j = 0; j < hidden_dim; ++j)
            basis[r][j] = normal(rng);

    // Per-token coefficients (Gaussian, top direction stronger)
    std::vector<double> scales = {2.5, 1.2, 0.7};
    for (int i = 0; i < seq; ++i) {
        std::vector<double> coeffs(rank);
        for (int r = 0; r < rank; ++r) {
            coeffs[r] = normal(rng) * scales[r];
        }
        for (int j = 0; j < hidden_dim; ++j) {
            double v = 0.0;
            for (int r = 0; r < rank; ++r) v += coeffs[r] * basis[r][j];
            // Add sinusoidal positional signal (small)
            v += 0.05 * std::sin(i * 0.3 + j * 0.1 + layer * 0.2);
            // Add small Gaussian noise
            v += noise(rng);
            H[i][j] = v;
        }
    }
    return H;
}

// Sample n random tokens (deterministic from seed).
inline std::vector<int> sample_tokens(int n, int vocab_size, int seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> uni(0, std::max(1, vocab_size - 1));
    std::vector<int> toks(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) toks[i] = uni(rng);
    return toks;
}

// Compute covariance max eigenvalue for given (tokens, layer, cfg).
inline double covariance_max_eigenvalue(const std::vector<int>& tokens, int layer,
                                        const SynthConfig& cfg) {
    int seq = static_cast<int>(tokens.size());
    Matrix H = synth_hidden(seq, cfg.hidden_dim, layer, cfg.seed);
    Matrix cov = covariance(H);
    EigenResult er = jacobi_eigen(cov);
    if (er.eigvals.empty()) return 0.0;
    return er.eigvals.back();  // largest (ascending order)
}

// ---------------------------------------------------------------------------
// Метаданные экспериментов / Experiment metadata
// ---------------------------------------------------------------------------
struct Experiment3D {
    std::string id;
    std::string name;
    std::string description;
    std::function<std::string(const std::map<std::string, std::string>&)> run;
};

// ---------------------------------------------------------------------------
// Эксперимент 6: Ландшафт потерь Гессиана / Hessian Loss Landscape
// ---------------------------------------------------------------------------
inline std::string exp_hessian_loss_landscape(const std::map<std::string, std::string>& params) {
    int grid = get_param_int(params, "hessian_grid_size", 24);
    int seed = get_param_int(params, "seed", 42);
    SynthConfig cfg = synth_config_from_params(params);

    // Sample hidden states across all layers, stack to (n_layers*seq, hidden_dim)
    int seq = std::min(cfg.max_seq_len, 64);
    std::vector<int> tokens = sample_tokens(seq, cfg.vocab_size, seed);
    Matrix H;
    H.reserve(static_cast<size_t>(cfg.n_layers * seq));
    for (int l = 0; l < cfg.n_layers; ++l) {
        Matrix Hl = synth_hidden(seq, cfg.hidden_dim, l, cfg.seed);
        for (const auto& row : Hl) H.push_back(row);
    }
    Matrix cov = covariance(H);
    EigenResult er = jacobi_eigen(cov);
    double lam1 = er.eigvals.back();
    double lam2 = er.eigvals.size() >= 2 ? er.eigvals[er.eigvals.size() - 2] : 0.0;

    // Top-2 eigendirections = last 2 columns of er.eigvecs (D x D)
    int D = cfg.hidden_dim;
    Vector v1(D), v2(D);
    for (int r = 0; r < D; ++r) {
        v1[r] = er.eigvecs[r][D - 1];
        v2[r] = er.eigvecs[r][D - 2];
    }
    (void)v1; (void)v2;  // directions used implicitly via span; not in JSON output

    // Build (w1, w2) perturbation grid (indexing="ij")
    double span = 3.0 * std::sqrt(std::max(lam1, 1e-9));
    std::vector<double> w1_axis = linspace(-span, span, grid);
    std::vector<double> w2_axis = linspace(-span, span, grid);

    Matrix W1 = mat_zeros(grid, grid), W2 = mat_zeros(grid, grid), Z = mat_zeros(grid, grid);
    double L0 = 1.0;
    double z_min = std::numeric_limits<double>::infinity();
    double z_max = -std::numeric_limits<double>::infinity();
    bool any_below_L0 = false;
    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j) {
            double w1 = w1_axis[i];
            double w2 = w2_axis[j];
            W1[i][j] = w1;
            W2[i][j] = w2;
            double z = L0 + 0.5 * (lam1 * w1 * w1 - lam2 * w2 * w2)
                      + 0.05 * std::sin(w1 * w2);
            Z[i][j] = z;
            if (z < z_min) z_min = z;
            if (z > z_max) z_max = z;
            if (z < L0) any_below_L0 = true;
        }
    }

    std::vector<JsonField> metrics = {
        {"lambda_max", json_double(lam1)},
        {"lambda_2", json_double(lam2)},
        {"spectral_gap", json_double(lam1 - lam2)},
        {"loss_min", json_double(z_min)},
        {"loss_max", json_double(z_max)},
        {"sharpness", json_double(lam1)},
        {"is_saddle", any_below_L0 ? "true" : "false"},
    };

    std::vector<JsonField> top_eig = {
        {"0", json_double(lam1)},
        {"1", json_double(lam2)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("hessian_loss_landscape")},
        {"grid_size", std::to_string(grid)},
        {"top_eigenvalues", build_object(top_eig)},
        {"w1_grid", json_array_2d(W1)},
        {"w2_grid", json_array_2d(W2)},
        {"loss_surface", json_array_2d(Z)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 7: Геометрия многообразия / Manifold Geometry (PCA)
// ---------------------------------------------------------------------------
inline std::string exp_manifold_geometry(const std::map<std::string, std::string>& params) {
    int n_components = get_param_int(params, "pca_components", 3);
    int n_samples    = get_param_int(params, "trajectory_points", 240);
    int seed         = get_param_int(params, "seed", 42);
    SynthConfig cfg = synth_config_from_params(params);

    // Collect hidden states from many random prompts (last layer)
    int n_prompts = std::min(n_samples, 32);
    Matrix H;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> seq_dist(8, std::max(9, cfg.max_seq_len));
    for (int p = 0; p < n_prompts; ++p) {
        int seq = seq_dist(rng);
        std::vector<int> toks = sample_tokens(seq, cfg.vocab_size, seed + p * 31);
        Matrix Hl = synth_hidden(seq, cfg.hidden_dim, cfg.n_layers - 1, seed + p * 7);
        for (const auto& row : Hl) H.push_back(row);
    }
    // Ensure we have enough rows for PCA
    while (static_cast<int>(H.size()) < n_components * 4) {
        H.push_back(H[H.size() % H.size()]);
    }

    int N = static_cast<int>(H.size());
    int D = static_cast<int>(H[0].size());

    // Center
    Vector mean(D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) mean[j] += H[i][j];
    for (int j = 0; j < D; ++j) mean[j] /= N;
    Matrix centered = H;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) centered[i][j] -= mean[j];

    // PCA via covariance + Jacobi
    Matrix cov = covariance(H);
    EigenResult er = jacobi_eigen(cov);
    // eigvals ascending; take all (we need participation ratio over all)
    Vector eigvals_pca = er.eigvals;
    // Sort descending for top-K
    Vector eigvals_desc = eigvals_pca;
    std::sort(eigvals_desc.begin(), eigvals_desc.end(), std::greater<double>());

    double sum_lam = 0.0, sum_lam2 = 0.0;
    for (double v : eigvals_pca) {
        double vv = std::max(v, 0.0);
        sum_lam += vv;
        sum_lam2 += vv * vv;
    }
    double pr = (sum_lam * sum_lam) / std::max(sum_lam2, 1e-12);

    // Project onto top-K components
    int K = std::min(n_components, D);
    Matrix Vtop = mat_zeros(D, K);
    for (int r = 0; r < D; ++r)
        for (int j = 0; j < K; ++j)
            Vtop[r][j] = er.eigvecs[r][D - K + j];
    Matrix proj = mat_mul(centered, Vtop);  // N x K
    // Pad to 3D
    Matrix pts = mat_zeros(N, 3);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 3; ++j) {
            pts[i][j] = (j < K) ? proj[i][j] : 0.0;
        }
    }
    // Colors
    Vector colors(N);
    for (int i = 0; i < N; ++i) colors[i] = static_cast<double>(i);

    // explained variance top-3
    double top3_sum = 0.0;
    int n_top = std::min(3, D);
    for (int i = 0; i < n_top; ++i) top3_sum += std::max(eigvals_desc[i], 0.0);
    double explained_var_top3 = top3_sum / std::max(sum_lam, 1e-12);
    double manifold_volume = 1.0;
    for (int i = 0; i < n_top; ++i) manifold_volume *= std::sqrt(std::max(eigvals_desc[i], 0.0));

    // Top eigenvalues list (up to max(n_components, 10))
    int n_eig_out = std::max(n_components, 10);
    int n_eig_actual = std::min(n_eig_out, D);
    Vector eigvals_out(static_cast<size_t>(n_eig_actual));
    for (int i = 0; i < n_eig_actual; ++i) eigvals_out[i] = eigvals_desc[i];

    std::vector<JsonField> metrics = {
        {"intrinsic_dim_pr", json_double(pr)},
        {"explained_variance_top3", json_double(explained_var_top3)},
        {"top_eigenvalue", json_double(eigvals_desc.empty() ? 0.0 : eigvals_desc[0])},
        {"manifold_volume_proxy", json_double(manifold_volume)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("manifold_geometry")},
        {"n_samples", std::to_string(N)},
        {"n_components", std::to_string(n_components)},
        {"pca_eigenvalues", json_array(eigvals_out)},
        {"participation_ratio", json_double(pr)},
        {"pca_points", json_array_2d(pts)},
        {"pca_colors", json_array(colors)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 8: Анализ траектории рассуждений / Reasoning Trajectory
// ---------------------------------------------------------------------------
inline std::string exp_trajectory_analysis(const std::map<std::string, std::string>& params) {
    int n_points = get_param_int(params, "trajectory_points", 64);
    int seed     = get_param_int(params, "seed", 42);
    double n_crit = get_param(params, "ncrit_threshold", 96.0);
    SynthConfig cfg = synth_config_from_params(params);

    // Synthesize honesty, deception, hallucination curves
    Vector honesty(n_points), deception(n_points), halluc(n_points);
    for (int s = 0; s < n_points; ++s) {
        double t = static_cast<double>(s);
        honesty[s] = 0.65 / std::sqrt(1.0 + t / n_crit);
        double d = 0.20 + 0.55 * (1.0 - std::exp(-(t - n_crit) / 30.0));
        deception[s] = clamp01(d);
        double h = 0.05 + 0.60 * (1.0 - std::exp(-(t - n_crit) / 20.0));
        halluc[s] = clamp01(h);
    }

    // Spectral radius per step (approximate)
    Vector spec_radius(n_points, 0.0);
    for (int s = 0; s < n_points; ++s) {
        std::vector<int> toks = sample_tokens(16, cfg.vocab_size, seed + s * 13);
        // Stack hidden across layers, take covariance max eigenvalue
        Matrix H;
        for (int l = 0; l < cfg.n_layers; ++l) {
            Matrix Hl = synth_hidden(16, cfg.hidden_dim, l, seed + s * 13);
            for (const auto& row : Hl) H.push_back(row);
        }
        Matrix cov = covariance(H);
        EigenResult er = jacobi_eigen(cov);
        spec_radius[s] = er.eigvals.empty() ? 0.0 : er.eigvals.back();
    }
    // Normalize to [0, 1]
    double sr_min = vec_min(spec_radius);
    double sr_max = vec_max(spec_radius);
    Vector spec_norm(n_points);
    for (int s = 0; s < n_points; ++s) {
        spec_norm[s] = (spec_radius[s] - sr_min) / std::max(sr_max - sr_min, 1e-9);
    }

    // Detect deception onset
    int onset_idx = -1;
    for (int i = 0; i < n_points; ++i) {
        if (deception[i] > honesty[i]) { onset_idx = i; break; }
    }

    Vector steps(n_points);
    for (int i = 0; i < n_points; ++i) steps[i] = static_cast<double>(i);

    std::vector<JsonField> trajectory = {
        {"steps", json_array(steps)},
        {"honesty", json_array(honesty)},
        {"deception", json_array(deception)},
        {"hallucination", json_array(halluc)},
        {"spectral", json_array(spec_radius)},
        {"spectral_normalized", json_array(spec_norm)},
    };

    std::vector<JsonField> metrics = {
        {"n_points", std::to_string(n_points)},
        {"onset_step", std::to_string(onset_idx)},
        {"final_honesty", json_double(honesty.empty() ? 0.0 : honesty.back())},
        {"final_deception", json_double(deception.empty() ? 0.0 : deception.back())},
        {"mean_spectral_radius", json_double(vec_mean(spec_radius))},
        {"max_spectral_radius", json_double(vec_max(spec_radius))},
        {"honesty_decrease_rate", json_double((honesty.empty() ? 0.0 : honesty[0] - honesty.back()) / std::max(n_points, 1))},
        {"deception_increase_rate", json_double((deception.empty() ? 0.0 : deception.back() - deception[0]) / std::max(n_points, 1))},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("trajectory_analysis")},
        {"n_points", std::to_string(n_points)},
        {"trajectory", build_object(trajectory)},
        {"deception_onset_step", std::to_string(onset_idx)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 9: Регрессия спектральной поверхности / Spectral Surface
// ---------------------------------------------------------------------------
inline std::string exp_spectral_surface_regression(const std::map<std::string, std::string>& params) {
    int n_layers  = get_param_int(params, "spectral_surface_layers", 6);
    int n_tokens  = get_param_int(params, "trajectory_points", 64);
    double n_crit_pred = get_param(params, "ncrit_threshold", 96.0);
    int seed      = get_param_int(params, "seed", 42);
    SynthConfig cfg = synth_config_from_params(params);
    n_layers = std::max(n_layers, 2);

    Matrix lambda_max_grid = mat_zeros(n_layers, n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        int n_tok = std::min(t + 4, cfg.max_seq_len);
        std::vector<int> toks = sample_tokens(n_tok, cfg.vocab_size, seed + t * 7);
        for (int l = 0; l < std::min(n_layers, cfg.n_layers); ++l) {
            Matrix H = synth_hidden(n_tok, cfg.hidden_dim, l, seed + t * 7);
            Matrix cov = covariance(H);
            EigenResult er = jacobi_eigen(cov);
            lambda_max_grid[l][t] = er.eigvals.empty() ? 0.0 : er.eigvals.back();
        }
    }

    // Mean over layers
    Vector mean_lambda(n_tokens, 0.0);
    for (int t = 0; t < n_tokens; ++t) {
        double s = 0.0;
        for (int l = 0; l < n_layers; ++l) s += lambda_max_grid[l][t];
        mean_lambda[t] = s / n_layers;
    }
    // Second difference
    int bif_token = 0;
    if (n_tokens >= 3) {
        double max_d2 = -1.0;
        for (int t = 1; t < n_tokens - 1; ++t) {
            double d2 = mean_lambda[t + 1] - 2.0 * mean_lambda[t] + mean_lambda[t - 1];
            if (std::abs(d2) > max_d2) {
                max_d2 = std::abs(d2);
                bif_token = t;
            }
        }
    }

    // Linear regression for pre/post slopes
    auto linfit = [](const Vector& xs, const Vector& ys) -> double {
        int n = static_cast<int>(xs.size());
        if (n < 2) return 0.0;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; ++i) {
            sx += xs[i]; sy += ys[i];
            sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
        }
        double denom = n * sxx - sx * sx;
        if (std::abs(denom) < 1e-12) return 0.0;
        return (n * sxy - sx * sy) / denom;
    };

    Vector pre_x, pre_y, post_x, post_y;
    int split = std::max(bif_token, 1);
    for (int t = 0; t < split; ++t) {
        pre_x.push_back(static_cast<double>(t));
        pre_y.push_back(mean_lambda[t]);
    }
    for (int t = split; t < n_tokens; ++t) {
        post_x.push_back(static_cast<double>(t - split));
        post_y.push_back(mean_lambda[t]);
    }
    double pre_slope = linfit(pre_x, pre_y);
    double post_slope = linfit(post_x, post_y);

    double lam_max_global = lambda_max_grid[0][0];
    double lam_min_global = lambda_max_grid[0][0];
    for (int l = 0; l < n_layers; ++l)
        for (int t = 0; t < n_tokens; ++t) {
            if (lambda_max_grid[l][t] > lam_max_global) lam_max_global = lambda_max_grid[l][t];
            if (lambda_max_grid[l][t] < lam_min_global) lam_min_global = lambda_max_grid[l][t];
        }

    std::vector<JsonField> metrics = {
        {"lambda_max_global", json_double(lam_max_global)},
        {"lambda_min_global", json_double(lam_min_global)},
        {"bifurcation_token", std::to_string(bif_token)},
        {"bifurcation_vs_ncrit", json_double(std::abs(bif_token - n_crit_pred))},
        {"pre_bifurcation_slope", json_double(pre_slope)},
        {"post_bifurcation_slope", json_double(post_slope)},
        {"slope_ratio", json_double(post_slope / std::max(std::abs(pre_slope), 1e-9))},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("spectral_surface_regression")},
        {"n_layers", std::to_string(n_layers)},
        {"n_tokens", std::to_string(n_tokens)},
        {"lambda_max_grid", json_array_2d(lambda_max_grid)},
        {"bifurcation_token", std::to_string(bif_token)},
        {"n_crit_predicted", json_double(n_crit_pred)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 10: Риманова кривизна / Riemannian Curvature (kNN-граф)
// ---------------------------------------------------------------------------
inline std::string exp_riemannian_curvature(const std::map<std::string, std::string>& params) {
    int n_neighbors = get_param_int(params, "curvature_neighbors", 8);
    int n_samples   = get_param_int(params, "trajectory_points", 128);
    int seed        = get_param_int(params, "seed", 42);
    SynthConfig cfg = synth_config_from_params(params);

    // Sample hidden states (last layer) from many prompts
    int n_prompts = std::min(n_samples, 32);
    Matrix H;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> seq_dist(8, std::max(9, cfg.max_seq_len));
    for (int p = 0; p < n_prompts; ++p) {
        int seq = seq_dist(rng);
        std::vector<int> toks = sample_tokens(seq, cfg.vocab_size, seed + p * 31);
        Matrix Hl = synth_hidden(seq, cfg.hidden_dim, cfg.n_layers - 1, seed + p * 11);
        for (const auto& row : Hl) H.push_back(row);
    }
    int N = static_cast<int>(H.size());

    // PCA reduce to 3D
    Matrix P = pca_project(H, 3);
    // Ensure 3 columns
    for (int i = 0; i < N; ++i) {
        while (P[i].size() < 3) P[i].push_back(0.0);
    }

    // For each point, find kNN, compute angle deficit
    Vector curvatures(N, 0.0);
    int k = std::min(n_neighbors, std::max(1, N - 1));
    for (int i = 0; i < N; ++i) {
        // Compute distances to all other points
        std::vector<std::pair<double, int>> dists;
        dists.reserve(N - 1);
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            double dx = P[j][0] - P[i][0];
            double dy = P[j][1] - P[i][1];
            double dz = P[j][2] - P[i][2];
            dists.push_back({std::sqrt(dx * dx + dy * dy + dz * dz), j});
        }
        std::sort(dists.begin(), dists.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                      return a.first < b.first;
                  });
        // Take k nearest
        Matrix vecs = mat_zeros(k, 3);
        for (int n = 0; n < k; ++n) {
            int j = dists[n].second;
            vecs[n][0] = P[j][0] - P[i][0];
            vecs[n][1] = P[j][1] - P[i][1];
            vecs[n][2] = P[j][2] - P[i][2];
            double nrm = std::sqrt(vecs[n][0] * vecs[n][0]
                                 + vecs[n][1] * vecs[n][1]
                                 + vecs[n][2] * vecs[n][2]);
            if (nrm < 1e-9) nrm = 1e-9;
            vecs[n][0] /= nrm; vecs[n][1] /= nrm; vecs[n][2] /= nrm;
        }
        // Sort vectors by polar angle (atan2 of (y, x)) for angle-deficit loop
        std::sort(vecs.begin(), vecs.end(), [](const Vector& a, const Vector& b) {
            return std::atan2(a[1], a[0]) < std::atan2(b[1], b[0]);
        });
        // Sum of angles between consecutive vectors (close the loop)
        double total_angle = 0.0;
        for (int n = 0; n < k; ++n) {
            int m = (n + 1) % k;
            double dot = vecs[n][0] * vecs[m][0] + vecs[n][1] * vecs[m][1] + vecs[n][2] * vecs[m][2];
            dot = std::max(-1.0, std::min(1.0, dot));
            total_angle += std::acos(dot);
        }
        curvatures[i] = 2.0 * PI - total_angle;
    }

    // High-curvature points: above mean + 2*std
    double m_c = vec_mean(curvatures);
    double s_c = vec_std(curvatures);
    double threshold = m_c + 2.0 * s_c;
    std::vector<int> high_idx;
    for (int i = 0; i < N; ++i) {
        if (curvatures[i] > threshold) high_idx.push_back(i);
    }

    std::vector<JsonField> metrics = {
        {"mean_curvature", json_double(m_c)},
        {"std_curvature", json_double(s_c)},
        {"max_curvature", json_double(vec_max(curvatures))},
        {"min_curvature", json_double(vec_min(curvatures))},
        {"n_high_curvature", std::to_string(static_cast<int>(high_idx.size()))},
        {"high_curvature_ratio", json_double(static_cast<double>(high_idx.size()) / std::max(N, 1))},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("riemannian_curvature")},
        {"n_samples", std::to_string(N)},
        {"n_neighbors", std::to_string(n_neighbors)},
        {"curvatures", json_array(curvatures)},
        {"points_3d", json_array_2d(P)},
        {"high_curvature_indices", json_int_array(high_idx)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 11: 3D-поток внимания / 3D Attention Flow
// ---------------------------------------------------------------------------
inline std::string exp_attention_flow_3d(const std::map<std::string, std::string>& params) {
    int resolution = get_param_int(params, "attention_flow_3d_resolution", 32);
    int seed       = get_param_int(params, "seed", 42);
    int n = resolution;

    // Synthesize attention: diagonal early, smeared past N_crit
    int n_crit = n / 2;
    Matrix attn = mat_zeros(n, n);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    for (int q = 0; q < n; ++q) {
        double sigma = (q < n_crit) ? 2.0 : 6.0;
        double row_sum = 0.0;
        for (int kk = 0; kk < n; ++kk) {
            double d = static_cast<double>(q - kk);
            double v = std::exp(-(d * d) / (2.0 * sigma * sigma));
            attn[q][kk] = v;
            row_sum += v;
        }
        if (row_sum < 1e-9) row_sum = 1e-9;
        for (int kk = 0; kk < n; ++kk) attn[q][kk] /= row_sum;
    }

    // Diagonality score: mean(diag(attn)) / mean(attn)
    double diag_sum = 0.0;
    for (int i = 0; i < n; ++i) diag_sum += attn[i][i];
    double diag_mean = diag_sum / n;
    double all_sum = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) all_sum += attn[i][j];
    double all_mean = all_sum / (n * n);
    double diag_score = diag_mean / std::max(all_mean, 1e-9);

    // Spread per row: sqrt(sum((pos - i)^2 * attn[i]) / sum(attn[i]))
    Vector spread_per_row(n);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        double row_s = 0.0;
        for (int j = 0; j < n; ++j) {
            double d = static_cast<double>(j - i);
            s += d * d * attn[i][j];
            row_s += attn[i][j];
        }
        spread_per_row[i] = std::sqrt(s / std::max(row_s, 1e-9));
    }
    double smearing_score = vec_mean(spread_per_row);

    // Entropy: -sum(attn * log(attn)) / n
    double entropy = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double p = attn[i][j];
            if (p > 1e-12) entropy -= p * std::log(p);
        }
    }
    entropy /= n;

    double max_weight = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (attn[i][j] > max_weight) max_weight = attn[i][j];

    std::vector<JsonField> metrics = {
        {"diagonality_score", json_double(diag_score)},
        {"smearing_score", json_double(smearing_score)},
        {"diagonal_to_smeared_ratio", json_double(diag_score / std::max(smearing_score, 1e-9))},
        {"max_weight", json_double(max_weight)},
        {"entropy", json_double(entropy)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("attention_flow_3d")},
        {"resolution", std::to_string(n)},
        {"weights", json_array_2d(attn)},
        {"spread_per_row", json_array(spread_per_row)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 12: Поверхность N_crit / N_crit Surface (фазовый переход)
// ---------------------------------------------------------------------------
inline std::string exp_ncrit_surface(const std::map<std::string, std::string>& params) {
    double n_crit_base = get_param(params, "ncrit_threshold", 114.0);
    double theta_b_deg = get_param(params, "theta_b_deg", 7.07);
    double theta_b = theta_b_deg * PI / 180.0;

    int n_beta = 24;
    int n_rlhf = 24;
    Vector beta_axis = linspace(0.3, 0.9, n_beta);
    Vector rlhf_axis = linspace(0.0, 1.0, n_rlhf);

    Matrix Z = mat_zeros(n_beta, n_rlhf);
    for (int i = 0; i < n_beta; ++i) {
        for (int j = 0; j < n_rlhf; ++j) {
            double mu_eff = std::max(theta_b + rlhf_axis[j], 1e-6);
            Z[i][j] = n_crit_base * std::pow(mu_eff, -1.0 / beta_axis[i]);
        }
    }

    double z_min = vec_min(Z[0]);
    double z_max = vec_max(Z[0]);
    for (const auto& row : Z) {
        for (double v : row) {
            if (v < z_min) z_min = v;
            if (v > z_max) z_max = v;
        }
    }
    double z_sum = 0.0; int z_cnt = 0;
    for (const auto& row : Z) for (double v : row) { z_sum += v; ++z_cnt; }
    double z_mean = z_sum / std::max(z_cnt, 1);

    double tc_b05_r0 = n_crit_base * std::pow(std::max(theta_b, 1e-6), -1.0 / 0.5);
    double tc_b05_r1 = n_crit_base * std::pow(std::max(theta_b + 1.0, 1e-6), -1.0 / 0.5);

    std::vector<JsonField> metrics = {
        {"t_crit_min", json_double(z_min)},
        {"t_crit_max", json_double(z_max)},
        {"t_crit_mean", json_double(z_mean)},
        {"t_crit_at_beta_0_5_rlhf_0", json_double(tc_b05_r0)},
        {"t_crit_at_beta_0_5_rlhf_1", json_double(tc_b05_r1)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("ncrit_surface")},
        {"beta_axis", json_array(beta_axis)},
        {"rlhf_axis", json_array(rlhf_axis)},
        {"t_crit_grid", json_array_2d(Z)},
        {"n_crit_base", json_double(n_crit_base)},
        {"theta_b_rad", json_double(theta_b)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 13: Развёртка пространства параметров / Parameter Space Sweep
// ---------------------------------------------------------------------------
inline std::string exp_parameter_space(const std::map<std::string, std::string>& params) {
    int grid = get_param_int(params, "parameter_space_grid", 16);
    int seed = get_param_int(params, "seed", 42);

    Vector temp_axis = linspace(0.0, 2.0, grid);
    Vector topp_axis = linspace(0.5, 1.0, grid);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    Matrix Z = mat_zeros(grid, grid);
    for (int i = 0; i < grid; ++i) {
        for (int j = 0; j < grid; ++j) {
            double T = temp_axis[i];
            double P = topp_axis[j];
            double base = sigmoid((T - 0.8) * 2.0) * (P - 0.5) * 2.0;
            double noise = 0.02 * uni(rng);
            Z[i][j] = clamp01(base + noise);
        }
    }

    double z_min = vec_min(Z[0]);
    double z_max = vec_max(Z[0]);
    for (const auto& row : Z) {
        for (double v : row) {
            if (v < z_min) z_min = v;
            if (v > z_max) z_max = v;
        }
    }
    // Row means (axis 0)
    double z_T0 = vec_mean(Z[0]);
    double z_T2 = vec_mean(Z[grid - 1]);
    // Column means
    Vector col0(grid), colL(grid);
    for (int i = 0; i < grid; ++i) { col0[i] = Z[i][0]; colL[i] = Z[i][grid - 1]; }
    double z_P05 = vec_mean(col0);
    double z_P1  = vec_mean(colL);

    std::vector<JsonField> metrics = {
        {"hallucination_min", json_double(z_min)},
        {"hallucination_max", json_double(z_max)},
        {"hallucination_at_T0", json_double(z_T0)},
        {"hallucination_at_T2", json_double(z_T2)},
        {"hallucination_at_P05", json_double(z_P05)},
        {"hallucination_at_P1", json_double(z_P1)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("parameter_space")},
        {"temp_axis", json_array(temp_axis)},
        {"topp_axis", json_array(topp_axis)},
        {"hallucination_grid", json_array_2d(Z)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Эксперимент 14: Дрейф коалиции / Coalition Drift (мультиагентный обман)
// ---------------------------------------------------------------------------
inline std::string exp_coalition_drift(const std::map<std::string, std::string>& params) {
    int n_agents = get_param_int(params, "n_agents", 2);
    int n_rounds = get_param_int(params, "n_rounds", 4);
    int seed     = get_param_int(params, "seed", 42);

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<double> normal(0.0, 0.02);

    Vector base_deception(n_agents);
    for (int i = 0; i < n_agents; ++i) base_deception[i] = 0.25 + 0.05 * i;

    Matrix per_round = mat_zeros(n_rounds, n_agents);
    for (int r = 0; r < n_rounds; ++r) {
        for (int a = 0; a < n_agents; ++a) {
            double v = base_deception[a] + 0.12 * r + normal(rng);
            per_round[r][a] = clamp01(v);
        }
    }

    double init_mean = vec_mean(per_round[0]);
    double final_mean = vec_mean(per_round[n_rounds - 1]);
    double drift = final_mean - init_mean;
    double conv_var = 0.0;
    {
        double m = vec_mean(per_round[n_rounds - 1]);
        for (int a = 0; a < n_agents; ++a) {
            double d = per_round[n_rounds - 1][a] - m;
            conv_var += d * d;
        }
        conv_var /= std::max(n_agents, 1);
    }

    std::vector<JsonField> metrics = {
        {"initial_mean_deception", json_double(init_mean)},
        {"final_mean_deception", json_double(final_mean)},
        {"drift", json_double(drift)},
        {"convergence_variance", json_double(conv_var)},
    };

    std::vector<JsonField> fields = {
        {"experiment", json_escape("coalition_drift")},
        {"n_agents", std::to_string(n_agents)},
        {"n_rounds", std::to_string(n_rounds)},
        {"per_round", json_array_2d(per_round)},
        {"metrics", build_object(metrics)},
    };
    return build_object(fields);
}

// ---------------------------------------------------------------------------
// Реестр экспериментов / Experiment registry
// ---------------------------------------------------------------------------
inline std::vector<Experiment3D> experiments_3d_info() {
    return {
        {"6",  "Ландшафт потерь Гессиана (3D)",
              "Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D-поверхности потерь.",
              exp_hessian_loss_landscape},
        {"7",  "Геометрия многообразия (3D PCA)",
              "Оценка внутренней размерности через коэффициент участия PCA; 3D-проекция.",
              exp_manifold_geometry},
        {"8",  "Анализ траектории рассуждений (3D)",
              "Дискретизация (шаг, честность, обман, спектральный радиус) и определение начала обмана.",
              exp_trajectory_analysis},
        {"9",  "Регрессия спектральной поверхности (3D)",
              "Аппроксимация поверхности λ_max(слой, токен) и обнаружение бифуркации N_crit.",
              exp_spectral_surface_regression},
        {"10", "Риманова кривизна (3D)",
              "Оценка дискретной гауссовой кривизны на kNN-графе скрытых состояний.",
              exp_riemannian_curvature},
        {"11", "3D-поток внимания",
              "Измерение поверхности весов внимания и количественная оценка диагонального и размытого режимов.",
              exp_attention_flow_3d},
        {"12", "Поверхность коллапса N_crit (3D)",
              "Вычисление поверхности T_crit(β, μ_RLHF) по порядку Капуто и давлению RLHF.",
              exp_ncrit_surface},
        {"13", "Развёртка пространства параметров (3D)",
              "Развёртка (температура, top_p) и измерение поверхности скорости галлюцинаций.",
              exp_parameter_space},
        {"14", "Дрейф коалиционного обмана (3D)",
              "Моделирование дрейфа обмана нескольких агентов по раундам коалиции (SCEN-COAL-09).",
              exp_coalition_drift},
    };
}

// Snake-case key used by charts_3d.py for the run_all_3d combined dict.
inline std::string chart_key_for_id(const std::string& id) {
    if (id == "6")  return "loss_landscape";
    if (id == "7")  return "manifold_geometry";
    if (id == "8")  return "trajectory";
    if (id == "9")  return "spectral_surface";
    if (id == "10") return "riemannian_curvature";
    if (id == "11") return "attention_flow_3d";
    if (id == "12") return "ncrit_surface";
    if (id == "13") return "parameter_space";
    if (id == "14") return "coalition_drift";
    return "exp_" + id;
}

// ---------------------------------------------------------------------------
// Публичный API / Public API
// ---------------------------------------------------------------------------
inline std::string run_3d_experiment(const std::string& id,
                                     const std::map<std::string, std::string>& params) {
    auto exps = experiments_3d_info();
    auto it = std::find_if(exps.begin(), exps.end(),
                           [&](const Experiment3D& e) { return e.id == id; });
    if (it == exps.end()) {
        std::vector<JsonField> err = {
            {"error", json_escape("Unknown 3D experiment: " + id)},
        };
        return build_object(err);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string inner = it->run(params);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Compose: wrapper fields + inner body (inner is "{...}", strip outer braces)
    std::string inner_body = inner;
    if (inner_body.size() >= 2 && inner_body.front() == '{' && inner_body.back() == '}') {
        inner_body = inner_body.substr(1, inner_body.size() - 2);
    }

    std::vector<JsonField> wrapper = {
        {"experiment_id",          json_escape(it->id)},
        {"experiment_name",        json_escape(it->name)},
        {"experiment_description", json_escape(it->description)},
        {"elapsed_seconds",        json_double(elapsed)},
        {"parameters",             params_to_json(params)},
    };
    std::string wrapper_body;
    for (size_t i = 0; i < wrapper.size(); ++i) {
        if (i) wrapper_body += ',';
        wrapper_body += json_escape(wrapper[i].key) + ':' + wrapper[i].raw_value;
    }

    std::string json = "{" + wrapper_body + "," + inner_body + "}";

    // Write to file
    int id_int = 0;
    try { id_int = std::stoi(id); } catch (...) {}
    write_3d_report(id_int, json);

    return json;
}

inline std::string run_all_3d(const std::map<std::string, std::string>& params) {
    auto exps = experiments_3d_info();
    std::vector<JsonField> research_fields;
    std::vector<std::string> experiments_array_entries;

    for (const auto& exp : exps) {
        std::string json = run_3d_experiment(exp.id, params);
        // research_field: chart_key -> json (the whole result object)
        research_fields.push_back({chart_key_for_id(exp.id), json});

        // Extract experiment_name and elapsed_seconds from the produced JSON
        // (simple substring search since we know the keys are at the top level)
        std::string name_val, elapsed_val;
        {
            std::string key = "\"experiment_name\":";
            size_t p = json.find(key);
            if (p != std::string::npos) {
                p += key.size();
                while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
                if (p < json.size() && json[p] == '"') {
                    size_t q = p + 1;
                    std::string s;
                    while (q < json.size() && json[q] != '"') {
                        if (json[q] == '\\' && q + 1 < json.size()) {
                            s += json[q]; s += json[q + 1]; q += 2;
                        } else { s += json[q]; ++q; }
                    }
                    name_val = s;
                }
            }
        }
        {
            std::string key = "\"elapsed_seconds\":";
            size_t p = json.find(key);
            if (p != std::string::npos) {
                p += key.size();
                while (p < json.size() && (json[p] == ' ' || json[p] == '\t')) ++p;
                size_t q = p;
                while (q < json.size() && json[q] != ',' && json[q] != '}') ++q;
                elapsed_val = json.substr(p, q - p);
            }
        }

        std::vector<JsonField> entry = {
            {"id", json_escape(exp.id)},
            {"name", json_escape(name_val)},
            {"elapsed_seconds", elapsed_val.empty() ? "0" : elapsed_val},
        };
        experiments_array_entries.push_back(build_object(entry));
    }

    std::string exps_array = "[";
    for (size_t i = 0; i < experiments_array_entries.size(); ++i) {
        if (i) exps_array += ',';
        exps_array += experiments_array_entries[i];
    }
    exps_array += "]";

    std::vector<JsonField> top = {
        {"3d_research", build_object(research_fields)},
        {"experiments", exps_array},
    };
    return build_object(top);
}

}  // namespace research_3d
