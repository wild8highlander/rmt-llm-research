"""
RMTLLMVerify — Random Matrix Theory meets Large Language Models

Verification package for the mathematical framework applying Random Matrix
Theory (RMT) to the spectral analysis of LLM activations.

Three research threads:
  1. Spectral statistics of LLM activations (Marchenko-Pastur, BBP, Tracy-Widom)
  2. Mathematical proof of inevitable hallucinations (10 paths to N_crit)
  3. The RLHF utility trap (Caputo fractional dynamics, entropic collapse)

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: CC-BY-NC-SA-4.0
"""

module RMTLLMVerify

using LinearAlgebra
using Random
using Statistics

export mp_bounds, mp_density, bbp_critical_theta, bbp_lambda_max,
       tracy_widom_cdf, nhse_winding_number, nhse_skin_strength,
       caputo_mean_collapse_time, caputo_n_crit,
       ks_corrected_gamma, ep_sensitivity,
       free_energy, spectral_entropy, landauer_cost

# ============================================================
# Constants
# ============================================================

"""First Riemann zeta zero (imaginary part)"""
const GAMMA_1 = 14.134725

"""Caputo memory parameter"""
const BETA_CAPUTO = 0.5

"""Rotation angle from NS regularity analysis (degrees)"""
const THETA_B_DEGREES = 7.07

"""Boltzmann constant (J/K)"""
const K_B = 1.380649e-23

# ============================================================
# Marchenko-Pastur Law
# ============================================================

"""
    mp_bounds(q, σ²=1.0)

Compute the support bounds λ₋, λ₊ of the Marchenko-Pastur distribution.

    λ₊ = σ²(1 + √q)²
    λ₋ = σ²(1 - √q)²
"""
function mp_bounds(q::Real, σ²::Real=1.0)
    @assert 0 < q ≤ 1 "q must be in (0, 1]"
    @assert σ² > 0 "σ² must be positive"
    √q = sqrt(q)
    λ₋ = σ² * (1 - √q)^2
    λ₊ = σ² * (1 + √q)^2
    return λ₋, λ₊
end

"""
    mp_density(λ, q, σ²=1.0)

Evaluate the Marchenko-Pastur density at eigenvalue λ.

    ρ(λ) = (1/(2π σ² λ q)) √((λ₊ - λ)(λ - λ₋))
"""
function mp_density(λ::Real, q::Real, σ²::Real=1.0)
    λ₋, λ₊ = mp_bounds(q, σ²)
    if λ ≤ λ₋ || λ ≥ λ₊ || λ ≤ 0
        return 0.0
    end
    factor = 1 / (2π * σ² * λ * q)
    return factor * sqrt((λ₊ - λ) * (λ - λ₋))
end

# ============================================================
# BBP Phase Transition
# ============================================================

"""
    bbp_critical_theta(q)

Compute the critical signal strength θ_c = √q for the BBP transition.
"""
function bbp_critical_theta(q::Real)
    @assert 0 < q ≤ 1 "q must be in (0, 1]"
    return sqrt(q)
end

"""
    bbp_lambda_max(θ, q, σ²=1.0)

Compute the asymptotic largest eigenvalue under a rank-1 spike.

    - Subcritical (θ ≤ √q): λ_max = λ₊
    - Supercritical (θ > √q): λ_max = σ²(1 + θ²/q)
"""
function bbp_lambda_max(θ::Real, q::Real, σ²::Real=1.0)
    θ_c = bbp_critical_theta(q)
    λ₊ = σ² * (1 + sqrt(q))^2
    if θ ≤ θ_c
        return λ₊
    else
        return σ² * (1 + θ^2 / q)
    end
end

# ============================================================
# Tracy-Widom F₂ (lookup table)
# ============================================================

const TW_TABLE = [
    (-5.0, 0.00000013), (-4.5, 0.00000159), (-4.0, 0.0000161),
    (-3.5, 0.000131), (-3.0, 0.000777), (-2.5, 0.00343),
    (-2.0, 0.0117), (-1.5, 0.0317), (-1.0, 0.0697),
    (-0.5, 0.127), (0.0, 0.204), (0.5, 0.293),
    (1.0, 0.387), (1.5, 0.477), (2.0, 0.555),
    (2.5, 0.618), (3.0, 0.668), (3.5, 0.707),
    (4.0, 0.737), (4.5, 0.760), (5.0, 0.778),
]

"""
    tracy_widom_cdf(s)

Evaluate the Tracy-Widom F₂ CDF via lookup table interpolation.
"""
function tracy_widom_cdf(s::Real)
    if s ≤ TW_TABLE[1][1]
        return TW_TABLE[1][2]
    end
    if s ≥ TW_TABLE[end][1]
        return TW_TABLE[end][2]
    end
    for i in 1:(length(TW_TABLE)-1)
        s1, f1 = TW_TABLE[i]
        s2, f2 = TW_TABLE[i+1]
        if s1 ≤ s ≤ s2
            t = (s - s1) / (s2 - s1)
            return f1 + t * (f2 - f1)
        end
    end
    return TW_TABLE[end][2]
end

# ============================================================
# Non-Hermitian Skin Effect
# ============================================================

"""
    nhse_winding_number(n_ratio, γ)

Compute the winding number for a non-Hermitian system.
w = 0 for n_ratio ≤ 1, w = 1 for n_ratio > 1.
"""
function nhse_winding_number(n_ratio::Real, γ::Real)
    @assert n_ratio ≥ 0 "n_ratio must be non-negative"
    return n_ratio > 1.0 ? 1 : 0
end

"""
    nhse_skin_strength(n_ratio, γ)

Compute the skin effect strength (degree of eigenvalue collapse).
"""
function nhse_skin_strength(n_ratio::Real, γ::Real)
    w = nhse_winding_number(n_ratio, γ)
    return w == 0 ? 0.0 : tanh(γ * (n_ratio - 1.0))
end

# ============================================================
# Caputo Fractional Dynamics
# ============================================================

"""
    caputo_mean_collapse_time(μ_eff, β=0.5, c=1.0)

Compute the mean hallucination collapse time.

    ⟨T_crit⟩ = c · μ_eff^(-1/β)
"""
function caputo_mean_collapse_time(μ_eff::Real, β::Real=BETA_CAPUTO, c::Real=1.0)
    @assert μ_eff > 0 "μ_eff must be positive"
    @assert 0 < β < 1 "β must be in (0, 1)"
    return c * μ_eff^(-1/β)
end

"""
    caputo_n_crit(θ_b=7.07, γ₁=GAMMA_1)

Estimate the critical token count N_crit from Caputo dynamics.
"""
function caputo_n_crit(θ_b::Real=THETA_B_DEGREES, γ₁::Real=GAMMA_1)
    return γ₁ / (θ_b * π / 180.0)
end

# ============================================================
# Keating-Snaith Corrections
# ============================================================

const KS_C1 = -0.133  # 1/N coefficient
const KS_C2 = 0.068   # 1/N² coefficient

"""
    ks_corrected_gamma(N, γ₁=GAMMA_1)

Compute the Keating-Snaith corrected first zeta zero.

    γ₁(N) = γ₁ + c₁/N + c₂/N²
"""
function ks_corrected_gamma(N::Real, γ₁::Real=GAMMA_1)
    @assert N > 0 "N must be positive"
    return γ₁ + KS_C1/N + KS_C2/N^2
end

# ============================================================
# Exceptional-Point Surfaces
# ============================================================

"""
    ep_sensitivity(ε, k)

Compute eigenvalue sensitivity at an EP of order k.

    δλ ~ ε^(1/k)
"""
function ep_sensitivity(ε::Real, k::Int)
    @assert ε ≥ 0 "ε must be non-negative"
    @assert k ≥ 2 "order must be ≥ 2"
    return ε == 0 ? 0.0 : ε^(1/k)
end

# ============================================================
# Thermodynamics
# ============================================================

"""
    free_energy(U, T, S)

Compute the thermodynamic free energy F = U - T·S.
"""
function free_energy(U::Real, T::Real, S::Real)
    return U - T * S
end

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
    landauer_cost(n_bits, T=300.0)

Compute the minimum thermodynamic cost of erasing n_bits at temperature T.
"""
function landauer_cost(n_bits::Int, T::Real=300.0)
    return n_bits * K_B * T * log(2)
end

end # module RMTLLMVerify
