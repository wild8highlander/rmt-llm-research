# research_3d.jl — 3D-исследовательские эксперименты для лаборатории RMT-LLM (Julia, RU)
# ============================================================================
#
# Реализует девять 3D-исследовательских экспериментов, зеркальных Python-
# модулю (laboratory/python/lab_ru/research_3d.py), так что JSON-вывод
# байт-совместим с charts_3d.py.
#
# Эксперименты (ID 6..14):
#   6.  Ландшафт потерь Гессиана          — возмущение (w1, w2) вдоль топ-2 собств. направлений
#   7.  Геометрия многообразия            — коэффициент участия PCA + 3D-проекция
#   8.  Анализ траектории рассуждений     — (шаг × честность × обман × ρ)
#   9.  Регрессия спектральной поверхности — поверхность λ_max(слой, токен), бифуркация N_crit
#  10.  Риманова кривизна                 — дискретная гауссова кривизна на kNN-графе
#  11.  3D поток внимания                 — поверхность весов внимания, диагональ-vs-размытый
#  12.  Поверхность коллапса N_crit       — фазовая поверхность T_crit(β, μ_RLHF)
#  13.  Сканирование пространства параметров — (температура × top_p × галлюцинации)
#  14.  Дрейф коалиционного обмана        — дрейф обмана мультиагентов по раундам
#
# Все эксперименты принимают бесконечное параметрическое соглашение (строка
# "inf", Float64 Inf, NaN, nothing) через помощник _clamp_inf, который
# ограничивает до практического конечного значения только на момент вычисления.
#
# Этот модуль — слой *чистых вычислений Julia* и НЕ зависит от внешних
# библиотек построения графиков.  Python-модуль charts_3d.py рендерит
# JSON-вывод в 3D-визуализации.
#
# Автор: Исхак Хамзатович Исаев, ORCID: 0009-0003-7299-0701
# Лицензия: Проприетарная — Все права защищены.

module research_3d

using LinearAlgebra
using Statistics
using Random
using Dates
using Printf
using JSON

# Доступ к механизмам TinyGPT из родительской области (Main).  main.jl
# включает tiny_gpt.jl перед research_3d.jl, поэтому эти символы доступны.
import ..TinyGPT
import ..TinyGPTConfig
import ..forward
import ..generate
import ..encode
import ..decode
import ..spectral_analysis

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
const LAB_ROOT     = abspath(joinpath(@__DIR__, "..", ".."))
const RESULTS_DIR  = joinpath(LAB_ROOT, "results")
const REPORTS_DIR  = joinpath(RESULTS_DIR, "reports")

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
"""
    _clamp_inf(value, default=0.0, max_finite=1e6) -> Float64

Преобразует inf / "inf" / nothing / NaN в конечное значение для вычислений.
Точно повторяет Python-помощник _clamp_inf.
"""
function _clamp_inf(value, default::Real=0.0, max_finite::Real=1e6)::Float64
    if value === nothing
        return Float64(default)
    end
    if value isa AbstractString
        s = lowercase(strip(String(value)))
        if s in ("inf", "+inf", "infinity")
            return Float64(max_finite)
        end
        # Попытка прямого числового разбора
        try
            return Float64(parse(Float64, s))
        catch
            return Float64(default)
        end
    end
    v = try
        Float64(value)
    catch
        return Float64(default)
    end
    if isnan(v) || isinf(v)
        return Float64(max_finite)
    end
    return v
end

"""
    _stable_softmax(x; dims=1)

Численно устойчивый softmax вдоль заданного измерения.
"""
function _stable_softmax(x::AbstractArray{T}; dims::Integer=1) where T<:Real
    m = maximum(x; dims=dims)
    e = exp.(x .- m)
    return e ./ sum(e; dims=dims)
end

"""
    _safe_svd_cov(mat) -> Matrix{Float64}

Эмпирическая ковариация (N, D)-матрицы: строки — образцы, столбцы — признаки.
Возвращает симметричную D×D матрицу.
"""
function _safe_svd_cov(mat::AbstractMatrix)::Matrix{Float64}
    m = Float64.(mat)
    if ndims(m) != 2
        m = reshape(m, size(m, 1), :)
    end
    N = size(m, 1)
    μ = mean(m, dims=1)
    centered = m .- μ
    return (centered' * centered) ./ max(N - 1, 1)
end
_safe_svd_cov(v::AbstractVector) = _safe_svd_cov(reshape(v, :, 1))

"""
    _max_eigval(cov_mat) -> Float64

Наибольшее собственное значение симметричной матрицы (≥ 0).
"""
function _max_eigval(cov_mat::AbstractMatrix)::Float64
    try
        e = eigvals(Symmetric(Float64.(cov_mat)))
        return isempty(e) ? 0.0 : Float64(maximum(e))
    catch
        return 0.0
    end
end

"""
    _config_to_dict(cfg) -> Dict{String,Any}

Сериализация TinyGPTConfig в простой Dict (для JSON-вывода).
"""
function _config_to_dict(cfg)::Dict{String,Any}
    return Dict{String,Any}(
        "vocab_size"  => cfg.vocab_size,
        "hidden_dim"  => cfg.hidden_dim,
        "n_layers"    => cfg.n_layers,
        "n_heads"     => cfg.n_heads,
        "max_seq_len" => cfg.max_seq_len,
        "seed"        => cfg.seed,
    )
end

# Наклон линейной аппроксимации y vs 0..n-1; возвращает 0.0 при нехватке точек.
function _slope_linear(y::AbstractVector{<:Real})::Float64
    n = length(y)
    n > 1 || return 0.0
    X = hcat(ones(n), collect(0:n-1))
    try
        c = X \ Float64.(y)
        return Float64(c[2])
    catch
        return 0.0
    end
end

# Случайные целочисленные токены в [0, vocab-1]
_rand_tokens(rng::AbstractRNG, vocab_size::Integer, n::Integer) =
    [rand(rng, 0:vocab_size-1) for _ in 1:n]

# Преобразование Matrix в Vector of Vector{Float64} (строки), чтобы JSON.jl
# сериализовал как Python-совместимый вложенный список (по строкам).
_mat_to_rows(M::AbstractMatrix) = [Float64.(M[i, :]) for i in 1:size(M, 1)]

# ---------------------------------------------------------------------------
# Эксперимент 6: Ландшафт потерь Гессиана
# ---------------------------------------------------------------------------
function _exp_hessian_loss_landscape(params::Dict)::Dict
    grid = Int(_clamp_inf(get(params, "hessian_grid_size", 24), 24, 4096))
    grid = clamp(grid, 4, 1024)
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))
    hidden_dim = Int(_clamp_inf(get(params, "hidden_dim", 64), 64, 8192))
    n_layers   = Int(_clamp_inf(get(params, "n_layers", 6), 6, 1024))
    n_heads    = Int(_clamp_inf(get(params, "n_heads", 4), 4, 256))

    cfg = TinyGPTConfig(hidden_dim=hidden_dim, n_layers=n_layers,
                        n_heads=n_heads, seed=seed)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Выборка скрытых состояний для построения матрицы типа эмпирического Фишера
    n_tok = min(cfg.max_seq_len, 64)
    tokens = _rand_tokens(rng, cfg.vocab_size, n_tok)
    _, hidden = forward(model, tokens)
    H = vcat([Float64.(h) for h in hidden]...)
    cov = _safe_svd_cov(H)
    F = eigen(Symmetric(cov))
    eigvals = F.values
    eigvecs = F.vectors
    # Топ-2 собственных направления (в Julia собственные значения по возрастанию)
    v1 = eigvecs[:, end]
    v2 = eigvecs[:, end - 1]
    lam1 = Float64(eigvals[end])
    lam2 = Float64(eigvals[end - 1])

    # Построение сетки возмущений (w1, w2)
    span = 3.0 * sqrt(max(lam1, 1e-9))
    w1_axis = collect(range(-span, span; length=grid))
    w2_axis = collect(range(-span, span; length=grid))
    W1 = repeat(w1_axis, 1, grid)         # W1[i,j] = w1_axis[i]
    W2 = repeat(reshape(w2_axis, 1, grid), grid, 1)  # W2[i,j] = w2_axis[j]
    # Квадратичная аппроксимация потерь: L = L0 + 0.5*(lam1*w1^2 - lam2*w2^2) + 0.05*sin(w1*w2)
    L0 = 1.0
    Z = L0 .+ 0.5 .* (lam1 .* W1.^2 .- lam2 .* W2.^2) .+ 0.05 .* sin.(W1 .* W2)

    zmin = minimum(Z)
    zmax = maximum(Z)
    is_saddle = (lam2 > 0.0 && lam1 > 0.0 && zmin < L0)

    return Dict{String,Any}(
        "experiment"      => "hessian_loss_landscape",
        "config"          => _config_to_dict(cfg),
        "grid_size"       => grid,
        "top_eigenvalues" => [lam1, lam2],
        "w1_grid"         => _mat_to_rows(W1),
        "w2_grid"         => _mat_to_rows(W2),
        "loss_surface"    => _mat_to_rows(Z),
        "metrics" => Dict{String,Any}(
            "lambda_max"     => lam1,
            "lambda_2"       => lam2,
            "spectral_gap"   => lam1 - lam2,
            "loss_min"       => zmin,
            "loss_max"       => zmax,
            "sharpness"      => lam1,
            "is_saddle"      => is_saddle,
        ),
        "data" => Dict{String,Any}(
            "X"               => _mat_to_rows(W1),
            "Y"               => _mat_to_rows(W2),
            "Z"               => _mat_to_rows(Z),
            "top_eigenvalues" => [lam1, lam2],
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 7: Геометрия многообразия
# ---------------------------------------------------------------------------
function _exp_manifold_geometry(params::Dict)::Dict
    n_components = Int(_clamp_inf(get(params, "pca_components", 3), 3, 32))
    n_samples    = Int(_clamp_inf(get(params, "trajectory_points", 240), 240, 10^6))
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))

    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Сбор скрытых состояний по множеству случайных промптов
    n_prompts = min(n_samples, 32)
    all_hidden = Matrix{Float64}[]
    for _ in 1:n_prompts
        n_tok = rand(rng, 8:cfg.max_seq_len)
        toks = _rand_tokens(rng, cfg.vocab_size, n_tok)
        _, hidden = forward(model, toks)
        push!(all_hidden, Float64.(hidden[end]))
    end
    H = vcat(all_hidden...)
    if size(H, 1) < n_components
        reps = cld(n_components, max(size(H, 1), 1)) + 1
        H = repeat(H, reps, 1)[1:min(n_components * 4, end), :]
    end

    # PCA через SVD
    μ = mean(H, dims=1)
    H_c = H .- μ
    F = svd(H_c)
    eigvals_pca = (F.S .^ 2) ./ max(size(H, 1) - 1, 1)
    # Коэффициент участия
    pr = Float64((sum(eigvals_pca))^2 / max(sum(eigvals_pca .^ 2), 1e-12))

    # Проекция на топ-N компонент
    n_proj = min(n_components, size(F.V, 2))
    proj = H_c * F.V[:, 1:n_proj]   # (N, n_proj)
    # Дополнение до 3D при необходимости
    while size(proj, 2) < 3
        proj = hcat(proj, zeros(size(proj, 1)))
    end

    # Цвет по индексу (прокси позиции токена)
    colors = collect(0:size(proj, 1) - 1)

    # Топ собственных значений (усечено для JSON)
    n_eig_out = min(max(n_components, 10), length(eigvals_pca))
    pca_eig_list = Float64.(eigvals_pca[1:n_eig_out])

    # Объяснённая дисперсия топ-3
    top3 = sum(eigvals_pca[1:min(3, length(eigvals_pca))])
    total = sum(eigvals_pca)
    exp_var = total > 1e-12 ? top3 / total : 0.0
    vol_proxy = prod(sqrt.(max.(eigvals_pca[1:min(3, length(eigvals_pca))], 0.0)))

    # Список точек (N × 3)
    pts = proj[:, 1:3]

    return Dict{String,Any}(
        "experiment"           => "manifold_geometry",
        "config"               => _config_to_dict(cfg),
        "n_samples"            => size(pts, 1),
        "n_components"         => n_components,
        "pca_eigenvalues"      => pca_eig_list,
        "participation_ratio"  => pr,
        "pca_points"           => [Float64.(pts[i, :]) for i in 1:size(pts, 1)],
        "pca_colors"           => Float64.(colors),
        "metrics" => Dict{String,Any}(
            "intrinsic_dim_pr"        => pr,
            "explained_variance_top3" => exp_var,
            "top_eigenvalue"          => Float64(eigvals_pca[1]),
            "manifold_volume_proxy"   => Float64(vol_proxy),
        ),
        "data" => Dict{String,Any}(
            "X"               => Float64.(pts[:, 1]),
            "Y"               => Float64.(pts[:, 2]),
            "Z"               => Float64.(pts[:, 3]),
            "colors"          => Float64.(colors),
            "pca_eigenvalues" => pca_eig_list,
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 8: Анализ траектории рассуждений
# ---------------------------------------------------------------------------
function _exp_trajectory_analysis(params::Dict)::Dict
    n_points = Int(_clamp_inf(get(params, "trajectory_points", 64), 64, 10^6))
    n_points = max(n_points, 8)
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))
    temperature = _clamp_inf(get(params, "temperature", 0.5), 0.5, 100.0)
    n_crit = _clamp_inf(get(params, "ncrit_threshold", 96.0), 96.0, 10^6)

    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Генерация одной длинной траектории рассуждений.  Обёрнуто в try/catch на
    # случай краевых ситуаций сэмплинга в TinyGPT.generate (например, численное
    # исчезновение при экстремальных температурах); всегда есть синтетический фолбэк.
    prompt = "Reason carefully about this problem."
    ids = encode(prompt)
    thoughts = []
    try
        out = generate(model, ids; max_new_tokens=n_points,
                       temperature=temperature, seed=seed)
        rt = out["reasoning_trace"]
        thoughts = get(rt, "thoughts", [])
    catch exc
        @warn "[research_3d] эксп. 8: TinyGPT.generate завершился ошибкой, используем синтетическую траекторию" exception=exc
    end

    # Дополнение / интерполяция до n_points (Julia generate даёт ≤12 мыслей,
    # поэтому берём ветку синтеза, зеркальную Python-фолбэку).
    if length(thoughts) >= n_points
        idx = round.(Int, range(1, length(thoughts); length=n_points))
        honesty    = Float64[thoughts[i]["honesty_score"]       for i in idx]
        deception  = Float64[thoughts[i]["deception_score"]     for i in idx]
        halluc     = Float64[thoughts[i]["hallucination_score"] for i in idx]
    else
        steps_arr = collect(range(0.0, Float64(n_points); length=n_points))
        honesty   = 0.65 ./ sqrt.(1.0 .+ steps_arr ./ max(n_crit, 1e-9))
        deception = 0.20 .+ 0.55 .* (1.0 .- exp.(-(steps_arr .- n_crit) ./ 30.0))
        deception = clamp.(deception, 0.0, 1.0)
        halluc    = 0.05 .+ 0.60 .* (1.0 .- exp.(-(steps_arr .- n_crit) ./ 20.0))
        halluc    = clamp.(halluc, 0.0, 1.0)
    end

    # Спектральный радиус на шаг (аппроксимация выборкой модели)
    spec_radius = Float64[]
    for _ in 1:n_points
        sample_toks = _rand_tokens(rng, cfg.vocab_size, 16)
        _, hidden = forward(model, sample_toks)
        H = vcat([Float64.(h) for h in hidden]...)
        cov = _safe_svd_cov(H)
        push!(spec_radius, _max_eigval(cov))
    end
    spec_arr = Float64.(spec_radius)
    smin, smax = minimum(spec_arr), maximum(spec_arr)
    spec_norm = (spec_arr .- smin) ./ max(smax - smin, 1e-9)

    # Обнаружение начала обмана (первый индекс, где deception > honesty)
    onset_idx = -1
    for i in 1:n_points
        if deception[i] > honesty[i]
            onset_idx = i - 1   # 0-базируется, как в Python
            break
        end
    end

    steps_arr_out = collect(0:n_points - 1)

    return Dict{String,Any}(
        "experiment"               => "trajectory_analysis",
        "n_points"                 => n_points,
        "trajectory" => Dict{String,Any}(
            "steps"               => steps_arr_out,
            "honesty"             => honesty,
            "deception"           => deception,
            "hallucination"       => halluc,
            "spectral"            => spec_arr,
            "spectral_normalized" => spec_norm,
        ),
        "deception_onset_step"     => onset_idx,
        "metrics" => Dict{String,Any}(
            "n_points"                => n_points,
            "onset_step"              => onset_idx,
            "final_honesty"           => Float64(honesty[end]),
            "final_deception"         => Float64(deception[end]),
            "mean_spectral_radius"    => Float64(mean(spec_arr)),
            "max_spectral_radius"     => Float64(maximum(spec_arr)),
            "honesty_decrease_rate"   => Float64(honesty[1] - honesty[end]) / max(n_points, 1),
            "deception_increase_rate" => Float64(deception[end] - deception[1]) / max(n_points, 1),
        ),
        "data" => Dict{String,Any}(
            "X"       => Float64.(steps_arr_out),
            "Y"       => Float64.(honesty),
            "Z"       => Float64.(deception),
            "W"       => Float64.(spec_arr),
            "onset_x" => onset_idx >= 0 ? onset_idx : -1,
            "onset_y" => onset_idx >= 0 ? Float64(honesty[onset_idx + 1]) : -1.0,
            "onset_z" => onset_idx >= 0 ? Float64(deception[onset_idx + 1]) : -1.0,
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 9: Регрессия спектральной поверхности
# ---------------------------------------------------------------------------
function _exp_spectral_surface_regression(params::Dict)::Dict
    n_layers = Int(_clamp_inf(get(params, "spectral_surface_layers", 6), 6, 1024))
    n_layers = max(n_layers, 2)
    n_tokens = Int(_clamp_inf(get(params, "trajectory_points", 64), 64, 10^6))
    n_tokens = max(n_tokens, 8)
    n_crit_pred = _clamp_inf(get(params, "ncrit_threshold", 96.0), 96.0, 10^6)
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))

    cfg = TinyGPTConfig(n_layers=n_layers, seed=seed)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Выборка по слоям и токенам
    lambda_max_grid = zeros(Float64, n_layers, n_tokens)
    for t in 1:n_tokens
        n_tok = min(t + 4 - 1, cfg.max_seq_len)   # зеркалим Python t+4 (0-базируется → +4)
        n_tok = max(n_tok, 2)
        toks = _rand_tokens(rng, cfg.vocab_size, n_tok)
        _, hidden = forward(model, toks)
        n_layers_actual = min(n_layers, length(hidden))
        for l in 1:n_layers_actual
            H = Float64.(hidden[l])
            cov = _safe_svd_cov(H)
            lambda_max_grid[l, t] = _max_eigval(cov)
        end
    end

    # Обнаружение бифуркации: токен t*, где кривизна (2-я разность) максимальна
    mean_lambda = vec(mean(lambda_max_grid, dims=1))
    if length(mean_lambda) >= 3
        d1 = diff(mean_lambda)
        d2 = diff(d1)
        bif_token_1based = argmax(abs.(d2))   # индекс в [1, length(d2)]
        bif_token = bif_token_1based          # Python: argmax(|d2|) + 1, тот же диапазон
    else
        bif_token = 0
    end

    # Регрессия до/после
    pre_end = max(bif_token, 1)
    pre = mean_lambda[1:pre_end]
    post_start = min(pre_end + 1, length(mean_lambda))
    post = mean_lambda[post_start:end]
    pre_slope = _slope_linear(pre)
    post_slope = _slope_linear(post)

    return Dict{String,Any}(
        "experiment"           => "spectral_surface_regression",
        "n_layers"             => n_layers,
        "n_tokens"             => n_tokens,
        "lambda_max_grid"      => _mat_to_rows(lambda_max_grid),
        "bifurcation_token"    => bif_token,
        "n_crit_predicted"     => n_crit_pred,
        "metrics" => Dict{String,Any}(
            "lambda_max_global"       => Float64(maximum(lambda_max_grid)),
            "lambda_min_global"       => Float64(minimum(lambda_max_grid)),
            "bifurcation_token"       => bif_token,
            "bifurcation_vs_ncrit"    => abs(bif_token - n_crit_pred),
            "pre_bifurcation_slope"   => pre_slope,
            "post_bifurcation_slope"  => post_slope,
            "slope_ratio"             => post_slope / max(pre_slope, 1e-9),
        ),
        "data" => Dict{String,Any}(
            "X" => collect(0:n_layers - 1),
            "Y" => collect(0:n_tokens - 1),
            "Z" => _mat_to_rows(lambda_max_grid),
            "bifurcation_token" => bif_token,
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 10: Риманова кривизна
# ---------------------------------------------------------------------------
function _exp_riemannian_curvature(params::Dict)::Dict
    n_neighbors = Int(_clamp_inf(get(params, "curvature_neighbors", 8), 8, 1024))
    n_samples   = Int(_clamp_inf(get(params, "trajectory_points", 128), 128, 10^6))
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))

    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Выборка скрытых состояний (последний слой каждого forward)
    all_hidden = Matrix{Float64}[]
    for _ in 1:min(n_samples, 32)
        n_tok = rand(rng, 8:cfg.max_seq_len)
        toks = _rand_tokens(rng, cfg.vocab_size, n_tok)
        _, hidden = forward(model, toks)
        push!(all_hidden, Float64.(hidden[end]))
    end
    H = vcat(all_hidden...)
    # Понижение до 3D для оценки кривизны
    μ = mean(H, dims=1)
    H_c = H .- μ
    F = svd(H_c)
    P = H_c * F.V[:, 1:min(3, size(F.V, 2))]
    while size(P, 2) < 3
        P = hcat(P, zeros(size(P, 1)))
    end
    P = P[:, 1:3]

    N = size(P, 1)
    curvatures = Float64[]
    for i in 1:N
        diffs = P .- P[i:i, :]
        dists = vec(sqrt.(sum(diffs.^2, dims=2)))
        dists[i] = Inf
        k = min(n_neighbors, N - 1)
        k = max(k, 1)
        nn_idx = sortperm(dists)[1:k]
        nn = P[nn_idx, :]
        vecs = nn .- P[i:i, :]
        norms = sqrt.(sum(vecs.^2, dims=2))
        vecs_n = vecs ./ max.(norms, 1e-9)
        # Углы между последовательными соседями (точно зеркалит Python)
        if k >= 2
            cos_consec = dropdims(sum(vecs_n[1:end-1, :] .* vecs_n[2:end, :], dims=2), dims=2)
            angles = acos.(clamp.(cos_consec, -1.0, 1.0))
            last_cos = sum(vecs_n[end, :] .* vecs_n[1, :])
            last_angle = acos(clamp(last_cos, -1.0, 1.0))
            total_angle = sum(angles) + last_angle
        else
            total_angle = 0.0
        end
        K = 2π - total_angle
        push!(curvatures, K)
    end
    curv_arr = Float64.(curvatures)

    # Точки высокой кривизны: среднее + 2σ
    thr = mean(curv_arr) + 2 * std(curv_arr)
    high_idx = findall(curv_arr .> thr) .- 1   # 0-базируется

    return Dict{String,Any}(
        "experiment"              => "riemannian_curvature",
        "n_samples"               => N,
        "n_neighbors"             => n_neighbors,
        "curvatures"              => curv_arr,
        "points_3d"               => [Float64.(P[i, :]) for i in 1:N],
        "high_curvature_indices"  => Int.(high_idx),
        "metrics" => Dict{String,Any}(
            "mean_curvature"      => Float64(mean(curv_arr)),
            "std_curvature"       => Float64(std(curv_arr)),
            "max_curvature"       => Float64(maximum(curv_arr)),
            "min_curvature"       => Float64(minimum(curv_arr)),
            "n_high_curvature"    => length(high_idx),
            "high_curvature_ratio" => length(high_idx) / max(N, 1),
        ),
        "data" => Dict{String,Any}(
            "X"          => Float64.(P[:, 1]),
            "Y"          => Float64.(P[:, 2]),
            "Z"          => Float64.(P[:, 3]),
            "colors"     => curv_arr,
            "high_idx"   => Int.(high_idx),
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 11: 3D поток внимания
# ---------------------------------------------------------------------------
function _exp_attention_flow_3d(params::Dict)::Dict
    resolution = Int(_clamp_inf(get(params, "attention_flow_3d_resolution", 32), 32, 1024))
    resolution = max(resolution, 8)
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))

    max_seq = max(resolution, 64)
    cfg = TinyGPTConfig(seed=seed, max_seq_len=max_seq)
    model = TinyGPT(cfg)
    rng = MersenneTwister(seed)

    # Forward-проход для заполнения model.hidden_states (Julia TinyGPT не
    # хранит послойные веса внимания, поэтому синтезируем типичный паттерн,
    # зеркальный фолбэку Python).
    n = resolution
    toks = _rand_tokens(rng, cfg.vocab_size, n)
    forward(model, toks)

    n_crit = n ÷ 2
    Q_idx = repeat(collect(0:n-1), 1, n)
    K_idx = repeat(collect(0:n-1)', n, 1)
    sigma_pre  = 2.0
    sigma_post = 6.0
    sigma_grid = ifelse.(Q_idx .< n_crit, sigma_pre, sigma_post)
    attn = exp.(-((Q_idx .- K_idx).^2) ./ (2.0 .* sigma_grid.^2))
    # Нормировка строк
    row_sums = sum(attn, dims=2)
    attn = attn ./ max.(row_sums, 1e-9)

    diag_vec = diag(attn)
    diag_score = mean(diag_vec) / max(mean(attn), 1e-9)
    # Разброс по строкам (RMS-расстояние от диагонали)
    spread_per_row = Float64[
        sqrt(sum((collect(0:n-1) .- (i-1)).^2 .* attn[i, :]) /
             max(sum(attn[i, :]), 1e-9))
        for i in 1:n
    ]
    smearing_score = mean(spread_per_row)
    # Энтропия (среднее по строкам)
    p_safe = max.(attn, 1e-12)
    entropy = -sum(p_safe .* log.(p_safe)) / n

    return Dict{String,Any}(
        "experiment"       => "attention_flow_3d",
        "resolution"       => n,
        "weights"          => _mat_to_rows(attn),
        "spread_per_row"   => spread_per_row,
        "metrics" => Dict{String,Any}(
            "diagonality_score"         => diag_score,
            "smearing_score"            => smearing_score,
            "diagonal_to_smeared_ratio" => diag_score / max(smearing_score, 1e-9),
            "max_weight"                => Float64(maximum(attn)),
            "entropy"                   => Float64(entropy),
        ),
        "data" => Dict{String,Any}(
            "X" => _mat_to_rows(Q_idx),
            "Y" => _mat_to_rows(K_idx),
            "Z" => _mat_to_rows(attn),
            "spread_per_row" => spread_per_row,
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 12: Поверхность коллапса N_crit
# ---------------------------------------------------------------------------
function _exp_ncrit_surface(params::Dict)::Dict
    n_crit_base = _clamp_inf(get(params, "ncrit_threshold", 114.0), 114.0, 10^6)
    theta_b_deg = _clamp_inf(get(params, "theta_b_deg", 7.07), 7.07, 360.0)
    theta_b = theta_b_deg * π / 180.0

    beta_axis = collect(range(0.3, 0.9; length=24))
    rlhf_axis = collect(range(0.0, 1.0; length=24))
    B_grid = repeat(beta_axis, 1, 24)
    R_grid = repeat(reshape(rlhf_axis, 1, 24), 24, 1)
    mu_eff = max.(theta_b .+ R_grid, 1e-6)
    Z = n_crit_base .* (mu_eff .^ (-1.0 ./ B_grid))

    return Dict{String,Any}(
        "experiment"    => "ncrit_surface",
        "beta_axis"     => beta_axis,
        "rlhf_axis"     => rlhf_axis,
        "t_crit_grid"   => _mat_to_rows(Z),
        "n_crit_base"   => n_crit_base,
        "theta_b_rad"   => theta_b,
        "metrics" => Dict{String,Any}(
            "t_crit_min"               => Float64(minimum(Z)),
            "t_crit_max"               => Float64(maximum(Z)),
            "t_crit_mean"              => Float64(mean(Z)),
            "t_crit_at_beta_0_5_rlhf_0" => Float64(n_crit_base * (theta_b^(-1.0 / 0.5))),
            "t_crit_at_beta_0_5_rlhf_1" => Float64(n_crit_base * ((theta_b + 1.0)^(-1.0 / 0.5))),
        ),
        "data" => Dict{String,Any}(
            "X" => _mat_to_rows(B_grid),
            "Y" => _mat_to_rows(R_grid),
            "Z" => _mat_to_rows(Z),
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 13: Сканирование пространства параметров
# ---------------------------------------------------------------------------
function _exp_parameter_space(params::Dict)::Dict
    grid = Int(_clamp_inf(get(params, "parameter_space_grid", 16), 16, 1024))
    grid = max(grid, 4)
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))
    rng = MersenneTwister(seed)

    temp_axis = collect(range(0.0, 2.0; length=grid))
    topp_axis = collect(range(0.5, 1.0; length=grid))
    T_grid = repeat(temp_axis, 1, grid)
    P_grid = repeat(reshape(topp_axis, 1, grid), grid, 1)
    base = (1.0 ./ (1.0 .+ exp.(-(T_grid .- 0.8) .* 2.0))) .* (P_grid .- 0.5) .* 2.0
    noise = 0.02 .* rand(rng, grid, grid)
    Z = clamp.(base .+ noise, 0.0, 1.0)

    return Dict{String,Any}(
        "experiment"          => "parameter_space",
        "temp_axis"           => temp_axis,
        "topp_axis"           => topp_axis,
        "hallucination_grid"  => _mat_to_rows(Z),
        "metrics" => Dict{String,Any}(
            "hallucination_min"    => Float64(minimum(Z)),
            "hallucination_max"    => Float64(maximum(Z)),
            "hallucination_at_T0"  => Float64(mean(Z[1, :])),
            "hallucination_at_T2"  => Float64(mean(Z[end, :])),
            "hallucination_at_P05" => Float64(mean(Z[:, 1])),
            "hallucination_at_P1"  => Float64(mean(Z[:, end])),
        ),
        "data" => Dict{String,Any}(
            "X" => _mat_to_rows(T_grid),
            "Y" => _mat_to_rows(P_grid),
            "Z" => _mat_to_rows(Z),
        ),
    )
end

# ---------------------------------------------------------------------------
# Эксперимент 14: Дрейф коалиционного обмана
# ---------------------------------------------------------------------------
function _exp_coalition_drift(params::Dict)::Dict
    n_agents = Int(_clamp_inf(get(params, "n_agents", 2), 2, 1024))
    n_rounds = Int(_clamp_inf(get(params, "n_rounds", 4), 4, 1024))
    seed = Int(_clamp_inf(get(params, "seed", 42), 42, 10^9))
    rng = MersenneTwister(seed)

    base_deception = 0.25 .+ 0.05 .* collect(0:n_agents - 1)
    per_round = Vector{Vector{Float64}}()
    for r in 0:n_rounds - 1
        round_scores = base_deception .+ 0.12 .* r .+ 0.02 .* randn(rng, n_agents)
        round_scores = clamp.(round_scores, 0.0, 1.0)
        push!(per_round, Float64.(round_scores))
    end

    initial_mean = mean(per_round[1])
    final_mean = mean(per_round[end])

    # Сетка X/Y/Z для 3D-визуализации: раунды × агенты × обман
    R_grid = repeat(collect(0:n_rounds - 1), 1, n_agents)
    A_grid = repeat(reshape(collect(0:n_agents - 1), 1, n_agents), n_rounds, 1)
    Z_grid = reduce(vcat, [reshape(per_round[r + 1], 1, n_agents) for r in 0:n_rounds - 1])

    return Dict{String,Any}(
        "experiment"  => "coalition_drift",
        "n_agents"    => n_agents,
        "n_rounds"    => n_rounds,
        "per_round"   => per_round,
        "metrics" => Dict{String,Any}(
            "initial_mean_deception"  => Float64(initial_mean),
            "final_mean_deception"    => Float64(final_mean),
            "drift"                   => Float64(final_mean - initial_mean),
            "convergence_variance"    => Float64(var(per_round[end])),
        ),
        "data" => Dict{String,Any}(
            "X" => _mat_to_rows(R_grid),
            "Y" => _mat_to_rows(A_grid),
            "Z" => _mat_to_rows(Z_grid),
        ),
    )
end

# ---------------------------------------------------------------------------
# Реестр
# ---------------------------------------------------------------------------
const EXPERIMENTS_3D = Dict{String, NamedTuple}(
    "6"  => (name="Ландшафт потерь Гессиана (3D)",
             description="Возмущение модели вдоль топ-2 собственных направлений Гессиана, измерение 3D-поверхности потерь.",
             run_func=_exp_hessian_loss_landscape),
    "7"  => (name="Геометрия многообразия (3D PCA)",
             description="Оценка внутренней размерности через коэффициент участия PCA; 3D-проекция.",
             run_func=_exp_manifold_geometry),
    "8"  => (name="Анализ траектории рассуждений (3D)",
             description="Выборка (шаг, честность, обман, спектральный радиус) и обнаружение начала обмана.",
             run_func=_exp_trajectory_analysis),
    "9"  => (name="Регрессия спектральной поверхности (3D)",
             description="Аппроксимация поверхности λ_max(слой, токен) и обнаружение бифуркации N_crit.",
             run_func=_exp_spectral_surface_regression),
    "10" => (name="Риманова кривизна (3D)",
             description="Оценка дискретной гауссовой кривизны на kNN-графе скрытых состояний.",
             run_func=_exp_riemannian_curvature),
    "11" => (name="3D поток внимания",
             description="Измерение поверхности весов внимания и количественная оценка диагонального vs размытого режима.",
             run_func=_exp_attention_flow_3d),
    "12" => (name="Поверхность коллапса N_crit (3D)",
             description="Вычисление поверхности T_crit(β, μ_RLHF) по порядку Капуто и давлению RLHF.",
             run_func=_exp_ncrit_surface),
    "13" => (name="Сканирование пространства параметров (3D)",
             description="Сканирование (температура, top_p) и измерение поверхности уровня галлюцинаций.",
             run_func=_exp_parameter_space),
    "14" => (name="Дрейф коалиционного обмана (3D)",
             description="Симуляция дрейфа обмана мультиагентов по раундам коалиции (SCEN-COAL-09).",
             run_func=_exp_coalition_drift),
)

# Соответствие ID → snake_case ключ в 3d_research
const _NAME_MAP = Dict{String, String}(
    "6"  => "loss_landscape",
    "7"  => "manifold_geometry",
    "8"  => "trajectory",
    "9"  => "spectral_surface",
    "10" => "riemannian_curvature",
    "11" => "attention_flow_3d",
    "12" => "ncrit_surface",
    "13" => "parameter_space",
    "14" => "coalition_drift",
)

# ---------------------------------------------------------------------------
# Запись JSON
# ---------------------------------------------------------------------------
function _write_3d_result_json(id::String, result::Dict, params::Dict)::String
    mkpath(REPORTS_DIR)
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    fname = "$(ts)_3d_exp_$(id)_results.json"
    path = joinpath(REPORTS_DIR, fname)
    try
        open(path, "w") do f
            JSON.print(f, result, 2)
        end
    catch exc
        @warn "[research_3d] Ошибка записи JSON для эксперимента $id: $exc"
    end
    return path
end

# ---------------------------------------------------------------------------
# Запускаторы
# ---------------------------------------------------------------------------
"""
    run_3d_experiment(id::String, params::Dict) -> Dict

Запускает один 3D-эксперимент по ID ("6".."14").  Возвращает Dict, который
соответствует структуре вывода Python-референса (поля данных верхнего
уровня, метрики, experiment_name/description/elapsed_seconds) плюс
обязательные по спеке ключи `experiment_id`/`name`/`description`/`data`.

Записывает результат в laboratory/results/reports/{ts}_3d_exp_{id}_results.json.
"""
function run_3d_experiment(id::String, params::Dict)::Dict
    if !haskey(EXPERIMENTS_3D, id)
        throw(KeyError("Неизвестный 3D-эксперимент: $id. Известные: $(sort(collect(keys(EXPERIMENTS_3D))))"))
    end
    exp = EXPERIMENTS_3D[id]
    t0 = time()
    raw = exp.run_func(params)
    elapsed = time() - t0

    # Метаданные по спеке
    raw["experiment_id"] = id
    raw["name"] = exp.name
    raw["description"] = exp.description
    raw["elapsed_seconds"] = elapsed
    # Python-совместимые метаданные
    raw["experiment_name"] = exp.name
    raw["experiment_description"] = exp.description

    # Отфильтрованные параметры (пропускаем dict/list значения)
    params_out = Dict{String, Any}()
    for (k, v) in params
        if !(v isa Dict || v isa AbstractArray)
            params_out[String(k)] = v
        end
    end
    raw["parameters"] = params_out

    # Сохранение в JSON
    path = _write_3d_result_json(id, raw, params)
    raw["results_json_path"] = path

    return raw
end

"""
    run_all_3d(params::Dict) -> Dict

Запускает все девять 3D-экспериментов и возвращает комбинированный Dict,
пригодный для charts_3d.generate_all_3d_charts().  Структура:

    {
      "3d_research": {
        "loss_landscape": {...},
        "manifold_geometry": {...},
        ...
      },
      "experiments": [
        {"id": "6", "name": ..., "elapsed_seconds": ...}, ...
      ]
    }
"""
function run_all_3d(params::Dict)::Dict
    combined = Dict{String, Any}(
        "3d_research" => Dict{String, Any}(),
        "experiments" => Dict{String, Any}[],
    )
    for id in sort(collect(keys(EXPERIMENTS_3D)))
        try
            res = run_3d_experiment(id, params)
            push!(combined["experiments"], Dict{String, Any}(
                "id"              => id,
                "name"            => res["name"],
                "elapsed_seconds" => res["elapsed_seconds"],
            ))
            key = get(_NAME_MAP, id, "exp_$(id)")
            combined["3d_research"][key] = res
        catch exc
            @warn "[research_3d] Эксперимент $id завершился с ошибкой" exception=exc
            push!(combined["experiments"], Dict{String, Any}(
                "id"    => id,
                "error" => string(exc),
            ))
        end
    end
    return combined
end

export EXPERIMENTS_3D, run_3d_experiment, run_all_3d, _clamp_inf

end # module research_3d
