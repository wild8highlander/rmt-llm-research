"""
Constants and key numerical values for the RMT-LLM framework.

Centralizes all important constants used across the verification suite.
"""

from __future__ import annotations

import math


# === Riemann zeta zeros (imaginary parts) ===
ZETA_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079810, 69.546402, 72.067158, 75.704691, 77.144840,
]

GAMMA_1 = ZETA_ZEROS[0]  # First Riemann zeta zero

# === Caputo fractional dynamics ===
BETA_CAPUTO = 0.5                 # Memory parameter
THETA_B_DEGREES = 7.07            # Rotation angle (from NS regularity)
THETA_B_RADIANS = THETA_B_DEGREES * math.pi / 180.0

# === Marchenko-Pastur ===
Q_DEFAULT = 0.5                   # Default aspect ratio N/T
SIGMA2_DEFAULT = 1.0              # Default population variance

# === BBP transition ===
THETA_CRITICAL_DEFAULT = math.sqrt(Q_DEFAULT)  # Default critical signal

# === Tracy-Widom ===
TW_MEAN = -1.7711                 # Mean of F_2
TW_VARIANCE = 0.8132              # Variance of F_2
TW_SKEWNESS = 0.2241              # Skewness of F_2

# === EP surfaces ===
MACHINE_EPS_FLOAT64 = 2.0 ** (-53)  # Machine epsilon for float64

# === Thermodynamics ===
K_B = 1.380649e-23                # Boltzmann constant (J/K)
LN2 = math.log(2)                 # ln(2)

# === Physical constants ===
ROOM_TEMPERATURE = 300.0          # Room temperature (K)

# === LLM parameters (from GPT-2 experimental fits) ===
GPT2_LAYERS = 12
GPT2_HIDDEN_DIM = 768
GPT2_VOCAB_SIZE = 50257
GPT2_CONTEXT_WINDOW = 1024

# === Key derived quantities ===
N_CRIT_ESTIMATE = GAMMA_1 / THETA_B_RADIANS  # Critical token count estimate
BKM_REDUCTION = 3.5               # BKM criterion reduction factor

# === Quantum channel ===
FIDELITY_PERCOLATION = 0.85       # Percolation threshold for quantum channel

# === Version ===
FRAMEWORK_VERSION = "1.3.0"
