"""
Test suite for RMTLLMViz — Advanced RMT-LLM Visualization Suite (Julia)
========================================================================

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
"""

using Test
using RMTLLMViz

@testset "RMTLLMViz — Mathematical Core" begin

    @testset "Marchenko-Pastur" begin
        # Bounds
        @testset "MP Bounds" begin
            for q in [0.1, 0.3, 0.5, 0.7, 0.9]
                λ₋, λ₊ = RMTLLMViz.mp_bounds(q)
                @test λ₋ < λ₊
                @test λ₋ ≥ 0
                @test λ₊ ≤ 4.0  # σ²=1, max λ₊ = (1+1)² = 4
            end
        end

        # Density
        @testset "MP Density" begin
            λ₋, λ₊ = RMTLLMViz.mp_bounds(0.5)
            @test RMTLLMViz.mp_density(λ₋ - 0.01, 0.5) == 0.0  # outside support
            @test RMTLLMViz.mp_density(λ₊ + 0.01, 0.5) == 0.0  # outside support
            mid = (λ₋ + λ₊) / 2
            @test RMTLLMViz.mp_density(mid, 0.5) > 0  # inside support
        end

        # Normalization
        @testset "MP Normalization" begin
            for q in [0.3, 0.5, 0.7]
                lam = range(0.01, 5.0, length=5000)
                rho = [RMTLLMViz.mp_density(l, q) for l in lam]
                integral = sum(rho) * step(lam)
                @test abs(integral - 1.0) < 0.02
            end
        end
    end

    @testset "BBP Phase Transition" begin
        @testset "Critical theta" begin
            for q in [0.1, 0.5, 0.9]
                θ_c = sqrt(q)
                # Subcritical: λ_max at bulk edge
                @test RMTLLMViz.bbp_lambda_max(θ_c - 0.1, q) ≈ (1 + sqrt(q))^2 atol=0.01
            end
        end

        @testset "Supercritical" begin
            q = 0.5
            θ = 1.0  # > sqrt(0.5) ≈ 0.707
            @test RMTLLMViz.bbp_lambda_max(θ, q) > (1 + sqrt(q))^2
        end
    end

    @testset "NHSE" begin
        @test RMTLLMViz.nhse_winding(0.5) == 0
        @test RMTLLMViz.nhse_winding(1.0) == 0
        @test RMTLLMViz.nhse_winding(1.5) == 1
        @test RMTLLMViz.nhse_winding(2.0) == 1

        @test RMTLLMViz.nhse_skin_strength(0.5, 0.3) == 0.0
        @test RMTLLMViz.nhse_skin_strength(1.5, 0.3) > 0.0
    end

    @testset "EP Sensitivity" begin
        @test RMTLLMViz.ep_sensitivity(0.0, 2) == 0.0
        @test RMTLLMViz.ep_sensitivity(1e-10, 2) ≈ 1e-5 atol=1e-10
        @test RMTLLMViz.ep_sensitivity(1e-6, 3) ≈ 1e-2 atol=1e-6

        for k in [2, 3, 5, 10]
            ε = 1e-10
            @test abs(RMTLLMViz.ep_sensitivity(ε, k) - ε^(1/k)) / RMTLLMViz.ep_sensitivity(ε, k) < 1e-10
        end
    end

    @testset "Thermodynamics" begin
        @test RMTLLMViz.free_energy(2.0, 1.0, 3.0) == -1.0
        @test RMTLLMViz.free_energy(5.0, 1.0, 2.0) == 3.0

        # Factual: F < 0 (low entropy)
        @test RMTLLMViz.free_energy(1.0, 1.0, 3.0) < 0
        # Creative: F > 0 (high energy)
        @test RMTLLMViz.free_energy(5.0, 1.0, 1.0) > 0
    end

    @testset "Caputo Fractional" begin
        @test RMTLLMViz.caputo_mean_collapse_time(1.0) ≈ 1.0 atol=0.01

        # Scaling: <T> ∝ μ^{-1/β} = μ^{-2} for β=0.5
        t1 = RMTLLMViz.caputo_mean_collapse_time(0.5)
        t2 = RMTLLMViz.caputo_mean_collapse_time(1.0)
        @test abs(t1 / t2 - 4.0) / 4.0 < 0.01  # (0.5)^{-2} / 1^{-2} = 4
    end

    @testset "Keating-Snaith" begin
        # Convergence to γ₁ for large N
        @test abs(RMTLLMViz.ks_corrected_gamma(1e6) - RMTLLMViz.GAMMA_1) < 0.01

        # Finite-N correction
        @test RMTLLMViz.ks_corrected_gamma(100) != RMTLLMViz.GAMMA_1
    end
end

@testset "RMTLLMViz — Visualization Functions" begin

    @testset "MP 3D" begin
        q, lam, rho = RMTLLMViz.viz_mp_3d(n_q=10, n_lam=20)
        @test length(q) == 10
        @test length(lam) == 20
        @test size(rho) == (20, 10)
        @test all(≥(0), rho)
    end

    @testset "BBP 3D" begin
        q, theta, lam_max = RMTLLMViz.viz_bbp_3d(n_q=10, n_theta=20)
        @test length(q) == 10
        @test length(theta) == 20
        @test size(lam_max) == (20, 10)
        @test all(>(0), lam_max)
    end

    @testset "NHSE 3D" begin
        results = RMTLLMViz.viz_nhse_3d(n_ratios_count=5, n_eig=10)
        @test length(results) == 5
        for r in results
            @test length(r.re) == 10
            @test length(r.im) == 10
            @test r.w ∈ [0, 1]
        end
    end

    @testset "EP 3D" begin
        eps, k, dlam = RMTLLMViz.viz_ep_3d(n_eps=20, k_max=5)
        @test length(eps) == 20
        @test all(≥(0), dlam)
    end

    @testset "Thermodynamic 3D" begin
        U, T, S_levels, results = RMTLLMViz.viz_thermo_3d(n_grid=10)
        @test length(S_levels) == 4
        for S in S_levels
            @test haskey(results, S)
        end
    end

    @testset "TW 3D" begin
        s, N_sizes, results = RMTLLMViz.viz_tw_3d(n_s=50)
        @test length(N_sizes) == 9
        for N in N_sizes
            @test haskey(results, N)
            @test all(≥(0), results[N])
            @test all(≤(1), results[N])
        end
    end

    @testset "KS 3D" begin
        N, c1, gamma, n_crit = RMTLLMViz.viz_ks_3d(n_N=20, n_c1=10)
        @test length(N) == 20
        @test length(c1) == 10
        @test all(>(0), gamma)
        @test all(>(0), n_crit)
    end

    @testset "Dashboard" begin
        checks = RMTLLMViz.viz_dashboard()
        @test length(checks) == 10
        for (name, passed, detail) in checks
            @test passed
        end
    end
end
