/*
 * RMT-LLM Mathematical Core — Pure computation engine without UI dependencies.
 *
 * This class provides all the mathematical functions needed for RMT-LLM
 * verification, independent of JavaFX. Used for headless verification
 * and cross-implementation consistency checks.
 *
 * Author: Iskhak Hamzatovich Isaev
 * ORCID:  0009-0003-7299-0701
 * License: CC-BY-NC-SA-4.0
 */

package com.rmt.llm.viz;

/**
 * Mathematical core functions for RMT-LLM framework verification.
 * This class has no UI dependencies and can be used for headless testing.
 */
public final class RMTMath {

    private RMTMath() {}  // Prevent instantiation

    // ═══════════════════════════════════════════════════════════════
    // Marchenko-Pastur Law
    // ═══════════════════════════════════════════════════════════════

    /**
     * Compute the Marchenko-Pastur support bounds.
     * @param q      Aspect ratio N/T in (0, 1]
     * @param sigma2 Population variance, positive
     * @return Array [lambda_minus, lambda_plus]
     */
    public static double[] mpBounds(double q, double sigma2) {
        double sqrtQ = Math.sqrt(q);
        return new double[]{
            sigma2 * Math.pow(1 - sqrtQ, 2),
            sigma2 * Math.pow(1 + sqrtQ, 2)
        };
    }

    /**
     * Evaluate the Marchenko-Pastur density.
     * rho(lambda) = (1/(2*pi*sigma2*lambda*q)) * sqrt((lambda_+ - lambda)(lambda - lambda_-))
     */
    public static double mpDensity(double lam, double q, double sigma2) {
        double[] bounds = mpBounds(q, sigma2);
        if (lam <= bounds[0] || lam >= bounds[1] || lam <= 0) return 0.0;
        double factor = 1.0 / (2 * Math.PI * sigma2 * lam * q);
        return factor * Math.sqrt((bounds[1] - lam) * (lam - bounds[0]));
    }

    /**
     * Compute the Stieltjes transform of the MP distribution.
     * g(z) = ((1-q-z) + sqrt((1-q-z)^2 - 4*q*z)) / (2*sigma2*q*z)
     */
    public static Complex mpStieltjes(Complex z, double q, double sigma2) {
        Complex oneMinusQ = new Complex(1 - q, 0);
        Complex numerator1 = oneMinusQ.subtract(z);
        Complex disc = numerator1.multiply(numerator1).subtract(
            z.multiply(new Complex(4 * q, 0)).multiply(z)
        );
        Complex sqrtDisc = disc.sqrt();
        if (sqrtDisc.im < 0) sqrtDisc = sqrtDisc.negate();
        return numerator1.add(sqrtDisc).divide(
            z.multiply(new Complex(2 * sigma2 * q, 0))
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // BBP Phase Transition
    // ═══════════════════════════════════════════════════════════════

    /** Compute the critical signal strength theta_c = sqrt(q). */
    public static double bbpCriticalTheta(double q) {
        return Math.sqrt(q);
    }

    /** Compute the BBP asymptotic largest eigenvalue. */
    public static double bbpLambdaMax(double theta, double q, double sigma2) {
        double thetaC = bbpCriticalTheta(q);
        double lamPlus = sigma2 * Math.pow(1 + Math.sqrt(q), 2);
        if (theta <= thetaC) return lamPlus;
        return sigma2 * (1 + theta * theta / q);
    }

    /** Check if the system is in the supercritical phase. */
    public static boolean bbpIsSupercritical(double theta, double q) {
        return theta > Math.sqrt(q);
    }

    /** Compute the signal separation: lambda_max - lambda_+. */
    public static double bbpSignalSeparation(double theta, double q, double sigma2) {
        return bbpLambdaMax(theta, q, sigma2) - sigma2 * Math.pow(1 + Math.sqrt(q), 2);
    }

    // ═══════════════════════════════════════════════════════════════
    // NHSE
    // ═══════════════════════════════════════════════════════════════

    /** Compute the NHSE winding number. */
    public static int nhseWinding(double nRatio) {
        return nRatio > 1.0 ? 1 : 0;
    }

    /** Compute the skin effect strength. */
    public static double nhseSkinStrength(double nRatio, double gamma) {
        int w = nhseWinding(nRatio);
        return w == 0 ? 0.0 : Math.tanh(gamma * (nRatio - 1.0));
    }

    // ═══════════════════════════════════════════════════════════════
    // EP Sensitivity
    // ═══════════════════════════════════════════════════════════════

    /** Compute EP eigenvalue sensitivity: delta_lambda ~ epsilon^{1/k}. */
    public static double epSensitivity(double epsilon, int order) {
        if (epsilon == 0) return 0.0;
        return Math.pow(epsilon, 1.0 / order);
    }

    /** Compute EP rounding effect for float64. */
    public static double epRoundingEffect(int n) {
        double epsilon = RMTConstants.MACHINE_EPS_FLOAT64;
        return epSensitivity(epsilon, n);
    }

    // ═══════════════════════════════════════════════════════════════
    // Thermodynamics
    // ═══════════════════════════════════════════════════════════════

    /** Compute the free energy F = U - T*S. */
    public static double freeEnergy(double U, double T, double S) {
        return U - T * S;
    }

    /** Compute the spectral (von Neumann) entropy. */
    public static double spectralEntropy(double[] eigenvalues) {
        double[] eigs = new double[eigenvalues.length];
        double total = 0;
        for (int i = 0; i < eigenvalues.length; i++) {
            eigs[i] = Math.max(eigenvalues[i], 0);
            total += eigs[i];
        }
        if (total <= 0) return 0;
        double entropy = 0;
        for (double eig : eigs) {
            double p = eig / total;
            if (p > 0) entropy -= p * Math.log(p);
        }
        return entropy;
    }

    /** Compute the Landauer cost of erasing n_bits at temperature T. */
    public static double landauerCost(int nBits, double T) {
        return nBits * RMTConstants.K_B * T * Math.log(2);
    }

    // ═══════════════════════════════════════════════════════════════
    // Caputo Fractional
    // ═══════════════════════════════════════════════════════════════

    /** Compute mean hallucination collapse time. */
    public static double caputoMeanCollapseTime(double muEff, double beta, double c) {
        return c * Math.pow(muEff, -1.0 / beta);
    }

    /** Estimate N_crit from Caputo dynamics. */
    public static double caputoNCrit() {
        return RMTConstants.GAMMA_1 / RMTConstants.THETA_B_RADIANS;
    }

    // ═══════════════════════════════════════════════════════════════
    // Keating-Snaith
    // ═══════════════════════════════════════════════════════════════

    /** Compute the Keating-Snaith corrected first zeta zero. */
    public static double ksCorrectedGamma(double N) {
        return RMTConstants.GAMMA_1 + RMTConstants.KS_C1 / N + RMTConstants.KS_C2 / (N * N);
    }

    /** Compute the GUE mean spacing. */
    public static double gueMeanSpacing(double N) {
        return Math.PI / Math.log(N);
    }
}
