/*
 * RMT-LLM Constants — Centralized numerical constants for the framework.
 *
 * Author: Iskhak Hamzatovich Isaev
 * ORCID:  0009-0003-7299-0701
 * License: CC-BY-NC-SA-4.0
 */

package com.rmt.llm.viz;

/**
 * Constants and key numerical values for the RMT-LLM framework.
 * Centralizes all important constants used across the visualization suite.
 */
public final class RMTConstants {

    private RMTConstants() {}  // Prevent instantiation

    // === Riemann zeta zeros (imaginary parts, first 20) ===
    public static final double[] ZETA_ZEROS = {
        14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
        37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
        67.079810, 69.546402, 72.067158, 75.704691, 77.144840,
    };

    /** First Riemann zeta zero (imaginary part) */
    public static final double GAMMA_1 = ZETA_ZEROS[0];

    // === Caputo fractional dynamics ===
    /** Memory parameter */
    public static final double BETA_CAPUTO = 0.5;

    /** Rotation angle from NS regularity (degrees) */
    public static final double THETA_B_DEGREES = 7.07;

    /** Rotation angle in radians */
    public static final double THETA_B_RADIANS = THETA_B_DEGREES * Math.PI / 180.0;

    // === Marchenko-Pastur ===
    /** Default aspect ratio N/T */
    public static final double Q_DEFAULT = 0.5;

    /** Default population variance */
    public static final double SIGMA2_DEFAULT = 1.0;

    // === BBP transition ===
    /** Default critical signal strength */
    public static final double THETA_CRITICAL_DEFAULT = Math.sqrt(Q_DEFAULT);

    // === Tracy-Widom ===
    /** Mean of F_2 */
    public static final double TW_MEAN = -1.7711;

    /** Variance of F_2 */
    public static final double TW_VARIANCE = 0.8132;

    /** Skewness of F_2 */
    public static final double TW_SKEWNESS = 0.2241;

    // === EP surfaces ===
    /** Machine epsilon for float64 */
    public static final double MACHINE_EPS_FLOAT64 = Math.pow(2.0, -53);

    // === Thermodynamics ===
    /** Boltzmann constant (J/K) */
    public static final double K_B = 1.380649e-23;

    /** ln(2) */
    public static final double LN2 = Math.log(2);

    // === Physical constants ===
    /** Room temperature (K) */
    public static final double ROOM_TEMPERATURE = 300.0;

    // === LLM parameters (GPT-2 experimental fits) ===
    public static final int GPT2_LAYERS = 12;
    public static final int GPT2_HIDDEN_DIM = 768;
    public static final int GPT2_VOCAB_SIZE = 50257;
    public static final int GPT2_CONTEXT_WINDOW = 1024;

    // === Key derived quantities ===
    /** Critical token count estimate */
    public static final double N_CRIT_ESTIMATE = GAMMA_1 / THETA_B_RADIANS;

    /** BKM criterion reduction factor */
    public static final double BKM_REDUCTION = 3.5;

    // === Keating-Snaith correction coefficients ===
    public static final double KS_C1 = -0.133;
    public static final double KS_C2 = 0.068;

    // === Quantum channel ===
    /** Percolation threshold for quantum channel fidelity */
    public static final double FIDELITY_PERCOLATION = 0.85;

    // === Version ===
    public static final String FRAMEWORK_VERSION = "1.3.0";
}
