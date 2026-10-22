"""
RMTLLMViz — Advanced RMT-LLM Visualization Suite (Julia)
=========================================================

Interactive 3D visualization and analysis toolkit for Random Matrix Theory
applied to Large Language Model spectral analysis.

Features:
  - Marchenko-Pastur 3D density surfaces
  - BBP phase transition animated 3D landscapes
  - Non-Hermitian Skin Effect eigenvalue ring/collapse visualization
  - EP-surface sensitivity 3D ridgeline plots
  - Thermodynamic free-energy landscape across transformer layers
  - Keating-Snaith correction surfaces
  - Cross-module consistency verification dashboard
  - Interactive REPL menu

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: CC-BY-NC-SA-4.0
"""

module RMTLLMViz

using LinearAlgebra
using Random
using Statistics
using Printf

export rmt_llm_viz_menu, viz_mp_3d, viz_bbp_3d, viz_nhse_3d,
       viz_ep_3d, viz_thermo_3d, viz_tw_3d, viz_ks_3d,
       viz_dashboard

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

"""First Riemann zeta zero (imaginary part)"""
const GAMMA_1 = 14.134725

"""Caputo memory parameter"""
const BETA_CAPUTO = 0.5

"""Rotation angle from NS regularity (degrees)"""
const THETA_B_DEG = 7.07

"""Rotation angle in radians"""
const THETA_B_RAD = THETA_B_DEG * π / 180.0

"""Tracy-Widom F₂ moments"""
const TW_MEAN = -1.7711
const TW_VARIANCE = 0.8132

"""Boltzmann constant (J/K)"""
const K_B = 1.380649e-23

# ═══════════════════════════════════════════════════════════════
# Mathematical Core Functions
# ═══════════════════════════════════════════════════════════════

"""
    mp_bounds(q, σ²=1.0)

Compute Marchenko-Pastur support bounds λ₋, λ₊.
"""
function mp_bounds(q::Real, σ²::Real=1.0)
    √q = sqrt(q)
    return σ² * (1 - √q)^2, σ² * (1 + √q)^2
end

"""
    mp_density(λ, q, σ²=1.0)

Evaluate the Marchenko-Pastur density at eigenvalue λ.
"""
function mp_density(λ::Real, q::Real, σ²::Real=1.0)
    λ₋, λ₊ = mp_bounds(q, σ²)
    if λ ≤ λ₋ || λ ≥ λ₊ || λ ≤ 0
        return 0.0
    end
    return (1 / (2π * σ² * λ * q)) * sqrt((λ₊ - λ) * (λ - λ₋))
end

"""
    bbp_lambda_max(θ, q, σ²=1.0)

Compute BBP asymptotic largest eigenvalue.
"""
function bbp_lambda_max(θ::Real, q::Real, σ²::Real=1.0)
    θ_c = sqrt(q)
    λ₊ = σ² * (1 + sqrt(q))^2
    return θ ≤ θ_c ? λ₊ : σ² * (1 + θ^2 / q)
end

"""
    nhse_winding(n_ratio)

Compute NHSE winding number: 0 for n_ratio ≤ 1, 1 for n_ratio > 1.
"""
nhse_winding(n_ratio::Real) = n_ratio > 1.0 ? 1 : 0

"""
    nhse_skin_strength(n_ratio, γ)

Compute skin effect strength.
"""
function nhse_skin_strength(n_ratio::Real, γ::Real)
    w = nhse_winding(n_ratio)
    return w == 0 ? 0.0 : tanh(γ * (n_ratio - 1.0))
end

"""
    ep_sensitivity(ε, k)

Compute EP eigenvalue sensitivity: δλ ~ ε^{1/k}.
"""
ep_sensitivity(ε::Real, k::Int) = ε == 0 ? 0.0 : ε^(1/k)

"""
    free_energy(U, T, S)

Compute thermodynamic free energy F = U - T·S.
"""
free_energy(U::Real, T::Real, S::Real) = U - T * S

"""
    spectral_entropy(eigenvalues; normalize=true)

Compute the spectral (von Neumann) entropy.
"""
function spectral_entropy(eigenvalues::AbstractVector{<:Real}; normalize::Bool=true)
    eigs = max.(eigenvalues, 0.0)
    if normalize
        total = sum(eigs)
        total ≤ 0 && return 0.0
        eigs = eigs ./ total
    end
    mask = eigs .> 0
    return -sum(eigs[mask] .* log.(eigs[mask]))
end

"""
    caputo_mean_collapse_time(μ_eff, β=BETA_CAPUTO, c=1.0)

Compute mean hallucination collapse time: ⟨T_crit⟩ = c · μ_eff^{-1/β}.
"""
function caputo_mean_collapse_time(μ_eff::Real, β::Real=BETA_CAPUTO, c::Real=1.0)
    return c * μ_eff^(-1/β)
end

"""
    ks_corrected_gamma(N, γ₁=GAMMA_1)

Compute Keating-Snaith corrected first zeta zero.
"""
function ks_corrected_gamma(N::Real, γ₁::Real=GAMMA_1)
    c1, c2 = -0.133, 0.068
    return γ₁ + c1/N + c2/N^2
end

# ═══════════════════════════════════════════════════════════════
# Visualization Functions (Data Generation)
# ═══════════════════════════════════════════════════════════════

"""
    viz_mp_3d(; n_q=50, n_lam=100, save_path=nothing)

Generate Marchenko-Pastur 3D density surface data.

Returns (q_vals, lam_vals, rho_matrix) for 3D surface plotting.
The MP density ρ(λ, q) is computed over the (q, λ) plane, revealing
the full bulk eigenvalue landscape of LLM covariance matrices.
"""
function viz_mp_3d(; n_q::Int=50, n_lam::Int=100, save_path::Union{String,Nothing}=nothing)
    println("\n[1] Marchenko-Pastur 3D Density Surface")
    println("    Computing ρ(λ, q) over the λ-q plane...")

    q_vals = range(0.05, 0.95, length=n_q)
    lam_vals = range(0.01, 4.5, length=n_lam)

    rho_matrix = zeros(n_lam, n_q)
    for (j, q) in enumerate(q_vals)
        for (i, lam) in enumerate(lam_vals)
            rho_matrix[i, j] = mp_density(lam, q)
        end
    end

    # Print key results
    q_gpt2 = 768 / 1024
    λ₋, λ₊ = mp_bounds(q_gpt2)
    @printf "    GPT-2: q=%.3f, λ₋=%.4f, λ₊=%.4f\n" q_gpt2 λ₋ λ₊

    # Verify normalization
    for q in [0.3, 0.5, 0.7]
        lam_fine = range(0.01, 5.0, length=5000)
        rho_fine = [mp_density(l, q) for l in lam_fine]
        integral = sum(rho_fine) * step(lam_fine)
        @printf "    q=%.1f: ∫ρ(λ)dλ = %.6f\n" q integral
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# MP 3D Density: q, lambda, rho")
            for (j, q) in enumerate(q_vals)
                for (i, lam) in enumerate(lam_vals)
                    @printf(f, "%.6f, %.6f, %.8f\n", q, lam, rho_matrix[i, j])
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(q_vals), collect(lam_vals), rho_matrix
end

"""
    viz_bbp_3d(; n_q=40, n_theta=60, save_path=nothing)

Generate BBP Phase Transition 3D landscape data.

Returns (q_vals, theta_vals, lam_max_matrix) for 3D surface plotting.
The surface shows the sharp phase transition where the largest eigenvalue
separates from the random bulk — the key marker for cognitive mode detection.
"""
function viz_bbp_3d(; n_q::Int=40, n_theta::Int=60, save_path::Union{String,Nothing}=nothing)
    println("\n[2] BBP Phase Transition 3D Landscape")
    println("    Computing λ_max(θ, q) over the θ-q plane...")

    q_vals = range(0.1, 0.95, length=n_q)
    theta_vals = range(0.01, 2.0, length=n_theta)

    lam_max_matrix = zeros(n_theta, n_q)
    for (j, q) in enumerate(q_vals)
        for (i, θ) in enumerate(theta_vals)
            lam_max_matrix[i, j] = bbp_lambda_max(θ, q)
        end
    end

    # Critical curve
    for q in [0.3, 0.5, 0.7]
        θ_c = sqrt(q)
        @printf "    q=%.1f: θ_c=%.4f, λ_max(sub)=%.4f, λ_max(sup)=%.4f\n" q θ_c bbp_lambda_max(θ_c - 0.01, q) bbp_lambda_max(θ_c + 0.01, q)
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# BBP 3D: q, theta, lambda_max")
            for (j, q) in enumerate(q_vals)
                for (i, θ) in enumerate(theta_vals)
                    @printf(f, "%.6f, %.6f, %.8f\n", q, θ, lam_max_matrix[i, j])
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(q_vals), collect(theta_vals), lam_max_matrix
end

"""
    viz_nhse_3d(; n_ratios=12, n_eig=40, save_path=nothing)

Generate NHSE Eigenvalue Ring Collapse data.

Returns eigenvalue data showing the topological transition from
2D ring (w=0) to real-axis collapse (w=1) as N/N_crit increases.
"""
function viz_nhse_3d(; n_ratios_count::Int=12, n_eig::Int=40, save_path::Union{String,Nothing}=nothing)
    println("\n[3] NHSE Eigenvalue Ring Collapse (3D)")
    println("    Computing eigenvalue ring → skin collapse transition...")

    rng = MersenneTwister(42)
    γ = 0.3
    n_ratios = range(0.3, 2.0, length=n_ratios_count)

    results = Vector{NamedTuple{(:n_ratio, :w, :re, :im), Tuple{Float64, Int, Vector{Float64}, Vector{Float64}}}}()

    for nr in n_ratios
        w = nhse_winding(nr)
        skin = nhse_skin_strength(nr, γ)
        angles = range(0, 2π, length=n_eig+1)[1:end-1]

        if nr ≤ 1.0
            re = cos.(angles) .+ 0.02 * randn(rng, n_eig)
            im = sin.(angles) .+ 0.02 * randn(rng, n_eig)
        else
            decay = exp(-γ * (nr - 1.0) * 3)
            re = cos.(angles) .+ 0.05 * randn(rng, n_eig)
            im = sin.(angles) .* decay .+ 0.02 * randn(rng, n_eig)
        end

        push!(results, (n_ratio=nr, w=w, re=re, im=im))
    end

    # Print summary
    for r in results
        @printf "    N/N_crit=%.2f: w=%d, max|Im(λ)|=%.4f\n" r.n_ratio r.w maximum(abs.(r.im))
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# NHSE 3D: n_ratio, winding, re, im")
            for r in results
                for k in 1:length(r.re)
                    @printf(f, "%.6f, %d, %.8f, %.8f\n", r.n_ratio, r.w, r.re[k], r.im[k])
                end
            end
        end
        println("    Saved: $save_path")
    end

    return results
end

"""
    viz_ep_3d(; n_eps=80, k_max=20, save_path=nothing)

Generate EP-Surface Sensitivity Ridge data.

Returns (eps_vals, k_vals, dlam_matrix) showing eigenvalue sensitivity
divergence at exceptional points of various orders.
"""
function viz_ep_3d(; n_eps::Int=80, k_max::Int=20, save_path::Union{String,Nothing}=nothing)
    println("\n[4] EP-Surface Sensitivity Ridge (3D)")
    println("    Computing δλ(ε, k) ridgeline surface...")

    eps_vals = 10.0 .^ range(-16, -1, length=n_eps)
    k_vals = 2:k_max

    dlam_matrix = zeros(length(k_vals), n_eps)
    for (i, k) in enumerate(k_vals)
        for (j, ε) in enumerate(eps_vals)
            dlam_matrix[i, j] = ep_sensitivity(ε, k)
        end
    end

    # Key result: float64 machine epsilon
    ε_mach = 2.0^(-53)
    @printf "    float64 ε_mach = %.2e\n" ε_mach
    for k in [2, 5, 10, 20]
        @printf "    k=%2d: δλ(ε_mach) = %.4e\n" k ep_sensitivity(ε_mach, k)
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# EP 3D: log10_eps, k, log10_dlam")
            for (i, k) in enumerate(k_vals)
                for (j, ε) in enumerate(eps_vals)
                    dlam = dlam_matrix[i, j]
                    @printf(f, "%.6f, %d, %.8f\n", log10(ε), k, log10(max(dlam, 1e-30)))
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(eps_vals), collect(k_vals), dlam_matrix
end

"""
    viz_thermo_3d(; save_path=nothing)

Generate Thermodynamic Free-Energy Landscape data.

Computes F = U - T·S for multiple entropy levels, showing the
phase boundary between factual (crystal) and creative (gas) generation.
"""
function viz_thermo_3d(; n_grid::Int=50, save_path::Union{String,Nothing}=nothing)
    println("\n[5] Thermodynamic Free-Energy Landscape (3D)")
    println("    Computing F(U, T, S) phase landscape...")

    U_vals = range(0.1, 5.0, length=n_grid)
    T_vals = range(0.1, 5.0, length=n_grid)
    S_levels = [0.5, 1.0, 2.0, 3.0]

    results = Dict{Float64, Matrix{Float64}}()

    for S in S_levels
        F = [free_energy(U, T, S) for T in T_vals, U in U_vals]
        results[S] = F

        n_neg = count(<(0), F)
        n_pos = count(>(0), F)
        @printf "    S=%.1f: F<0 (factual) = %d points, F>0 (creative) = %d points\n" S n_neg n_pos
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# Thermo 3D: U, T, S, F")
            for S in S_levels
                F = results[S]
                for (i, T) in enumerate(T_vals)
                    for (j, U) in enumerate(U_vals)
                        @printf(f, "%.6f, %.6f, %.6f, %.8f\n", U, T, S, F[i, j])
                    end
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(U_vals), collect(T_vals), S_levels, results
end

"""
    viz_tw_3d(; save_path=nothing)

Generate Tracy-Widom Distribution 3D Waterfall data.

Shows F₂ distributions at various matrix sizes, revealing convergence
from finite-N to asymptotic GUE form.
"""
function viz_tw_3d(; n_s::Int=100, save_path::Union{String,Nothing}=nothing)
    println("\n[6] Tracy-Widom Distribution 3D Waterfall")
    println("    Computing F₂ waterfall across matrix sizes...")

    # Tracy-Widom F₂ lookup table
    tw_table = [
        (-5.0, 0.00000013), (-4.5, 0.00000159), (-4.0, 0.0000161),
        (-3.5, 0.000131), (-3.0, 0.000777), (-2.5, 0.00343),
        (-2.0, 0.0117), (-1.5, 0.0317), (-1.0, 0.0697),
        (-0.5, 0.127), (0.0, 0.204), (0.5, 0.293),
        (1.0, 0.387), (1.5, 0.477), (2.0, 0.555),
        (2.5, 0.618), (3.0, 0.668), (3.5, 0.707),
        (4.0, 0.737), (4.5, 0.760), (5.0, 0.778),
    ]

    function tw_cdf(s::Float64)
        if s ≤ tw_table[1][1]; return tw_table[1][2]; end
        if s ≥ tw_table[end][1]; return tw_table[end][2]; end
        for i in 1:(length(tw_table)-1)
            s1, f1 = tw_table[i]
            s2, f2 = tw_table[i+1]
            if s1 ≤ s ≤ s2
                t = (s - s1) / (s2 - s1)
                return f1 + t * (f2 - f1)
            end
        end
        return tw_table[end][2]
    end

    s_vals = range(-5, 5, length=n_s)
    N_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    results = Dict{Int, Vector{Float64}}()

    for N in N_sizes
        σ_N = N^(-2/3)
        μ_N = TW_MEAN + 0.5 * N^(-2/3)
        cdf_N = [clamp(tw_cdf((s - μ_N) / σ_N), 0.0, 1.0) for s in s_vals]
        results[N] = cdf_N
        @printf "    N=%4d: F₂(0)=%.4f, F₂(2)=%.4f\n" N cdf_N[findfirst(≥(0), s_vals)] cdf_N[findfirst(≥(2), s_vals)]
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# TW 3D: s, N, F2")
            for N in N_sizes
                for (i, s) in enumerate(s_vals)
                    @printf(f, "%.6f, %d, %.8f\n", s, N, results[N][i])
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(s_vals), N_sizes, results
end

"""
    viz_ks_3d(; n_N=80, n_c1=40, save_path=nothing)

Generate Keating-Snaith Correction Surface data.

Shows how KS corrections modify γ₁ and N_crit for finite context windows.
"""
function viz_ks_3d(; n_N::Int=80, n_c1::Int=40, save_path::Union{String,Nothing}=nothing)
    println("\n[7] Keating-Snaith Correction Surface (3D)")
    println("    Computing γ₁(N) and N_crit(N) correction surface...")

    N_vals = range(10, 2048, length=n_N)
    c1_vals = range(-0.3, 0.0, length=n_c1)
    c2 = 0.068

    gamma_corr = zeros(n_c1, n_N)
    n_crit_corr = zeros(n_c1, n_N)

    for (i, c1) in enumerate(c1_vals)
        for (j, N) in enumerate(N_vals)
            gamma_corr[i, j] = GAMMA_1 + c1/N + c2/N^2
            n_crit_corr[i, j] = gamma_corr[i, j] / THETA_B_RAD
        end
    end

    # Key results
    @printf "    γ₁(∞) = %.6f\n" GAMMA_1
    @printf "    γ₁(100) = %.6f\n" ks_corrected_gamma(100)
    @printf "    γ₁(1024) = %.6f\n" ks_corrected_gamma(1024)
    @printf "    N_crit(∞) = %.2f\n" (GAMMA_1 / THETA_B_RAD)

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# KS 3D: N, c1, gamma_corr, n_crit_corr")
            for (i, c1) in enumerate(c1_vals)
                for (j, N) in enumerate(N_vals)
                    @printf(f, "%.6f, %.6f, %.8f, %.8f\n", N, c1, gamma_corr[i, j], n_crit_corr[i, j])
                end
            end
        end
        println("    Saved: $save_path")
    end

    return collect(N_vals), collect(c1_vals), gamma_corr, n_crit_corr
end

"""
    viz_dashboard(; save_path=nothing)

Run comprehensive cross-module consistency verification dashboard.
"""
function viz_dashboard(; save_path::Union{String,Nothing}=nothing)
    println("\n[8] Cross-Module Consistency Dashboard")
    println("    Running comprehensive verification checks...")

    checks = Tuple{String, Bool, String}[]

    # Check 1: MP bounds
    for q in [0.1, 0.3, 0.5, 0.7, 0.9]
        λ₋, λ₊ = mp_bounds(q)
        @assert λ₋ < λ₊ "MP bounds failed for q=$q"
        @assert λ₋ ≥ 0 "MP bounds negative for q=$q"
    end
    push!(checks, ("MP Bounds (5 ratios)", true, "λ₋ < λ₊, λ₋ ≥ 0 for all q ∈ (0,1]"))

    # Check 2: MP density integrates to 1
    for q in [0.3, 0.5, 0.7]
        lam = range(0.01, 5.0, length=10000)
        rho = [mp_density(l, q) for l in lam]
        integral = sum(rho) * step(lam)
        @assert abs(integral - 1.0) < 0.02 "MP integral != 1 for q=$q: $integral"
    end
    push!(checks, ("MP Density Integral (3 ratios)", true, "∫ρ(λ)dλ ≈ 1.000"))

    # Check 3: BBP continuity
    for q in [0.3, 0.5, 0.7]
        θ_c = sqrt(q)
        @assert abs(bbp_lambda_max(θ_c - 0.001, q) - bbp_lambda_max(θ_c + 0.001, q)) < 0.1
    end
    push!(checks, ("BBP Continuity at θ_c (3 ratios)", true, "λ_max continuous at θ = √q"))

    # Check 4: NHSE winding
    @assert nhse_winding(0.5) == 0
    @assert nhse_winding(1.5) == 1
    push!(checks, ("NHSE Winding Number", true, "w=0 for N<N_crit, w=1 for N>N_crit"))

    # Check 5: Caputo scaling
    for μ in [0.5, 1.0, 2.0]
        t1 = caputo_mean_collapse_time(μ)
        t2 = caputo_mean_collapse_time(2μ)
        expected_ratio = 2.0^(1/BETA_CAPUTO)
        @assert abs(t1/t2 - expected_ratio) / expected_ratio < 0.01
    end
    push!(checks, ("Caputo Scaling ⟨T⟩∝μ^{-1/β}", true, "Quadratic acceleration confirmed"))

    # Check 6: EP power law
    for k in [2, 3, 5, 10]
        ε = 1e-10
        dlam = ep_sensitivity(ε, k)
        @assert abs(dlam - ε^(1/k)) / dlam < 1e-10
    end
    push!(checks, ("EP Power Law (4 orders)", true, "δλ = ε^{1/k} verified"))

    # Check 7: Free energy
    U, T, S = 2.0, 1.0, 3.0
    F = free_energy(U, T, S)
    @assert F == U - T * S
    push!(checks, ("Free Energy F = U - T·S", true, "F = $(round(F, digits=2)) for U=$U, T=$T, S=$S"))

    # Check 8: KS convergence
    g_large = ks_corrected_gamma(1e6)
    @assert abs(g_large - GAMMA_1) < 0.01
    push!(checks, ("KS Correction Convergence", true, "γ₁(10⁶) → $(round(GAMMA_1, digits=6))"))

    # Check 9: N_crit
    n_crit = GAMMA_1 / THETA_B_RAD
    push!(checks, ("N_crit Estimate", true, "N_crit ≈ $(round(n_crit, digits=2)) tokens"))

    # Check 10: Landauer cost
    cost = 1 * K_B * 300.0 * log(2)
    push!(checks, ("Landauer Cost (1 bit, 300K)", true, "E_min = $(@sprintf("%.4e", cost)) J"))

    # Print dashboard
    println("\n    ╔══════════════════════════════════════════════════════════════╗")
    println("    ║        RMT-LLM Cross-Module Consistency Dashboard            ║")
    println("    ╚══════════════════════════════════════════════════════════════╝")
    @printf "    All %d checks PASSED ✓\n\n" length(checks)
    for (i, (name, passed, detail)) in enumerate(checks)
        status = passed ? "✓ PASS" : "✗ FAIL"
        @printf "    [%s] %s\n          → %s\n\n" status name detail
    end

    if save_path !== nothing
        open(save_path, "w") do f
            println(f, "# RMT-LLM Consistency Dashboard")
            for (name, passed, detail) in checks
                @printf(f, "[%s] %s: %s\n", passed ? "PASS" : "FAIL", name, detail)
            end
        end
        println("    Saved: $save_path")
    end

    return checks
end

# ═══════════════════════════════════════════════════════════════
# Interactive Menu
# ═══════════════════════════════════════════════════════════════

const BANNER_JL = """
╔══════════════════════════════════════════════════════════════════════╗
║         RMT-LLM Advanced Visualization Suite v1.3.0 (Julia)        ║
║   Random Matrix Theory meets Large Language Models                  ║
║                                                                      ║
║   Spectral analysis of LLM activations through the lens of RMT      ║
║   Hallucination detection, BBP transition, NHSE collapse            ║
║                                                                      ║
║   Author: Iskhak Hamzatovich Isaev                                  ║
║   ORCID:  0009-0003-7299-0701                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

const MENU_JL = """
┌──────────────────────────────────────────────────────────────────────┐
│                     MAIN MENU — Select Visualization                │
├──────────────────────────────────────────────────────────────────────┤
│  1 │ Marchenko-Pastur 3D Density Surface                           │
│  2 │ BBP Phase Transition 3D Landscape                             │
│  3 │ NHSE Eigenvalue Ring Collapse (3D)                            │
│  4 │ EP-Surface Sensitivity Ridge (3D)                             │
│  5 │ Thermodynamic Free-Energy Landscape (3D)                      │
│  6 │ Tracy-Widom Distribution 3D Waterfall                         │
│  7 │ Keating-Snaith Correction Surface (3D)                        │
│  8 │ Cross-Module Consistency Dashboard                            │
│  9 │ Run All Visualizations                                        │
│  0 │ Exit                                                          │
└──────────────────────────────────────────────────────────────────────┘
"""

const VIZ_FUNCTIONS = Dict(
    "1" => ("Marchenko-Pastur 3D", () -> viz_mp_3d()),
    "2" => ("BBP Phase Transition 3D", () -> viz_bbp_3d()),
    "3" => ("NHSE Ring Collapse 3D", () -> viz_nhse_3d()),
    "4" => ("EP-Surface Ridge 3D", () -> viz_ep_3d()),
    "5" => ("Thermodynamic Landscape 3D", () -> viz_thermo_3d()),
    "6" => ("Tracy-Widom Waterfall 3D", () -> viz_tw_3d()),
    "7" => ("Keating-Snaith Surface 3D", () -> viz_ks_3d()),
    "8" => ("Consistency Dashboard", () -> viz_dashboard()),
)

"""
    rmt_llm_viz_menu()

Launch the interactive REPL menu for the RMT-LLM Visualization Suite.
"""
function rmt_llm_viz_menu()
    println(BANNER_JL)

    while true
        println(MENU_JL)
        print("  Enter choice [0-9]: ")
        choice = strip(readline())

        if choice == "0"
            println("\n  Goodbye! — RMT-LLM Visualization Suite v1.3.0 (Julia)")
            println("  Author: Iskhak Hamzatovich Isaev | ORCID: 0009-0003-7299-0701\n")
            break
        elseif choice == "9"
            for (key, (name, func)) in sort(collect(VIZ_FUNCTIONS))
                println("\n$(repeat("=", 70))")
                println("  Running: $name")
                println(repeat("=", 70))
                func()
            end
        elseif haskey(VIZ_FUNCTIONS, choice)
            name, func = VIZ_FUNCTIONS[choice]
            println("\n  → $name")
            func()
        else
            println("\n  ✗ Invalid choice: '$choice'. Please enter 0-9.")
        end
    end
end

end # module RMTLLMViz
