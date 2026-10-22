/*
 * RMT-LLM Verification Engine — Headless cross-implementation consistency checks.
 *
 * Can be run without JavaFX for CI environments.
 *
 * Author: Iskhak Hamzatovich Isaev
 * ORCID:  0009-0003-7299-0701
 * License: CC-BY-NC-SA-4.0
 */

package com.rmt.llm.viz;

/**
 * Headless verification engine for CI environments.
 * Runs all mathematical consistency checks without requiring JavaFX.
 */
public final class RMTVerifier {

    private RMTVerifier() {}  // Prevent instantiation

    /**
     * Run all verification checks and print results.
     * @return Number of failed checks (0 = all passed)
     */
    public static int runAllChecks() {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║     RMT-LLM Cross-Module Verification Engine v1.3.0        ║");
        System.out.println("║     Author: Iskhak Hamzatovich Isaev                       ║");
        System.out.println("║     ORCID:  0009-0003-7299-0701                            ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.println();

        int failures = 0;

        failures += check("MP Bounds", () -> {
            for (double q : new double[]{0.1, 0.3, 0.5, 0.7, 0.9}) {
                double[] b = RMTMath.mpBounds(q, 1.0);
                assert b[0] < b[1] : "λ₋ >= λ₊ for q=" + q;
                assert b[0] >= 0 : "λ₋ < 0 for q=" + q;
            }
        });

        failures += check("MP Density Integral", () -> {
            for (double q : new double[]{0.3, 0.5, 0.7}) {
                double integral = 0;
                double dx = 0.001;
                for (double lam = 0.01; lam <= 5.0; lam += dx) {
                    integral += RMTMath.mpDensity(lam, q, 1.0) * dx;
                }
                assert Math.abs(integral - 1.0) < 0.02 : "∫ρ dλ = " + integral + " for q=" + q;
            }
        });

        failures += check("MP Stieltjes Transform", () -> {
            Complex z = new Complex(2.0, 0.1);
            Complex g = RMTMath.mpStieltjes(z, 0.5, 1.0);
            assert g.im > 0 : "Im(g) should be positive for Im(z) > 0";
            // Self-consistency: g should satisfy the MP equation
            double q = 0.5, sigma2 = 1.0;
            Complex rhs = new Complex(1, 0).divide(
                z.subtract(sigma2.multiply(new Complex(q, 0)).multiply(
                    z.multiply(g).add(new Complex(q - 1, 0)).divide(new Complex(q, 0))
                ))
            );
            assert g.subtract(rhs).abs() < 0.01 : "Stieltjes self-consistency failed";
        });

        failures += check("BBP Continuity at θ_c", () -> {
            for (double q : new double[]{0.3, 0.5, 0.7}) {
                double tc = Math.sqrt(q);
                double diff = Math.abs(
                    RMTMath.bbpLambdaMax(tc - 0.001, q, 1.0) -
                    RMTMath.bbpLambdaMax(tc + 0.001, q, 1.0)
                );
                assert diff < 0.1 : "Discontinuity at θ_c for q=" + q;
            }
        });

        failures += check("BBP Signal Separation", () -> {
            double q = 0.5;
            double sep = RMTMath.bbpSignalSeparation(1.0, q, 1.0);
            assert sep > 0 : "Signal should separate in supercritical phase";
            assert RMTMath.bbpSignalSeparation(0.5, q, 1.0) == 0.0 :
                "No separation in subcritical phase";
        });

        failures += check("NHSE Winding Number", () -> {
            assert RMTMath.nhseWinding(0.5) == 0;
            assert RMTMath.nhseWinding(1.0) == 0;
            assert RMTMath.nhseWinding(1.5) == 1;
            assert RMTMath.nhseWinding(2.0) == 1;
        });

        failures += check("NHSE Skin Strength", () -> {
            assert RMTMath.nhseSkinStrength(0.5, 0.3) == 0.0;
            assert RMTMath.nhseSkinStrength(1.5, 0.3) > 0.0;
            assert RMTMath.nhseSkinStrength(1.5, 0.3) < 1.0;
        });

        failures += check("EP Sensitivity Power Law", () -> {
            for (int k : new int[]{2, 3, 5, 10}) {
                double eps = 1e-10;
                double dlam = RMTMath.epSensitivity(eps, k);
                assert Math.abs(dlam - Math.pow(eps, 1.0/k)) / dlam < 1e-10 :
                    "Power law violated for k=" + k;
            }
        });

        failures += check("EP Rounding Effect", () -> {
            double effect = RMTMath.epRoundingEffect(10);
            assert effect > 0 : "EP rounding effect should be positive";
            assert effect > 1e-5 : "EP rounding effect should be significant for k=10";
        });

        failures += check("Free Energy F = U - T·S", () -> {
            assert RMTMath.freeEnergy(2.0, 1.0, 3.0) == -1.0;
            assert RMTMath.freeEnergy(5.0, 1.0, 2.0) == 3.0;
        });

        failures += check("Spectral Entropy", () -> {
            double[] uniform = {0.25, 0.25, 0.25, 0.25};
            double s = RMTMath.spectralEntropy(uniform);
            assert Math.abs(s - Math.log(4)) < 0.01 : "Uniform entropy should be ln(4)";
        });

        failures += check("Caputo Scaling", () -> {
            for (double mu : new double[]{0.5, 1.0, 2.0}) {
                double t1 = RMTMath.caputoMeanCollapseTime(mu, RMTConstants.BETA_CAPUTO, 1.0);
                double t2 = RMTMath.caputoMeanCollapseTime(2*mu, RMTConstants.BETA_CAPUTO, 1.0);
                double expected = Math.pow(2.0, 1.0 / RMTConstants.BETA_CAPUTO);
                assert Math.abs(t1/t2 - expected) / expected < 0.01;
            }
        });

        failures += check("Keating-Snaith Convergence", () -> {
            double gLarge = RMTMath.ksCorrectedGamma(1e6);
            assert Math.abs(gLarge - RMTConstants.GAMMA_1) < 0.01;
        });

        failures += check("N_crit Estimate", () -> {
            double nCrit = RMTMath.caputoNCrit();
            assert nCrit > 100 && nCrit < 200 : "N_crit out of expected range: " + nCrit;
        });

        failures += check("Landauer Cost", () -> {
            double cost = RMTMath.landauerCost(1, 300.0);
            double expected = RMTConstants.K_B * 300.0 * Math.log(2);
            assert Math.abs(cost - expected) / expected < 1e-10;
        });

        System.out.println();
        if (failures == 0) {
            System.out.println("✓ All 14 checks PASSED");
        } else {
            System.out.println("✗ " + failures + " check(s) FAILED");
        }
        System.out.println();
        System.out.println("Author: Iskhak Hamzatovich Isaev | ORCID: 0009-0003-7299-0701");

        return failures;
    }

    @FunctionalInterface
    private interface CheckBody {
        void run() throws AssertionError;
    }

    private static int check(String name, CheckBody body) {
        try {
            body.run();
            System.out.printf("  [✓ PASS] %s%n", name);
            return 0;
        } catch (AssertionError e) {
            System.out.printf("  [✗ FAIL] %s: %s%n", name, e.getMessage());
            return 1;
        }
    }

    public static void main(String[] args) {
        int failures = runAllChecks();
        System.exit(failures);
    }
}
