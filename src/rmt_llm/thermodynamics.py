"""
Thermodynamic Analogy — Free energy, renormalization, and Landauer principle.

The thermodynamic framework for LLM spectral analysis:

1. Free energy:  F = U - T * S_spec
   - Factual generation: "crystal" phase (low entropy, F < 0)
   - Creative generation: "gas" phase (high entropy, F > 0)

2. Renormalization group flow: the spectral distribution evolves across
   transformer layers, analogous to RG flow in statistical mechanics.

3. Landauer principle: erasing one bit of information costs at least
   k_B * T * ln(2) energy. For autoregressive generation, each token
   overwriting previous context is an irreversible thermodynamic process.

References:
  - Landauer (1961), "Irreversibility and heat generation in the
    computing process", IBM J. Res. Dev.
"""

from __future__ import annotations

import numpy as np


# Physical constants
K_B = 1.380649e-23  # Boltzmann constant (J/K)
LN2 = np.log(2)


def free_energy(internal_energy: float, temperature: float, spectral_entropy: float) -> float:
    """Compute the thermodynamic free energy.

    F = U - T * S_spec

    Parameters
    ----------
    internal_energy : float
        Internal energy U (from eigenvalue moments).
    temperature : float
        Effective temperature T (must be >= 0).
    spectral_entropy : float
        Spectral entropy S_spec (Shannon entropy of eigenvalue distribution).

    Returns
    -------
    F : float
        Free energy.
    """
    return internal_energy - temperature * spectral_entropy


def landauer_cost(n_bits: int, temperature: float = 300.0) -> float:
    """Compute the minimum thermodynamic cost of erasing information.

    E >= n_bits * k_B * T * ln(2)

    Parameters
    ----------
    n_bits : int
        Number of bits erased.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    energy : float
        Minimum energy cost in Joules.
    """
    return n_bits * K_B * temperature * LN2


def spectral_entropy(eigenvalues: np.ndarray, normalize: bool = True) -> float:
    """Compute the spectral (von Neumann) entropy of a density matrix.

    S = -sum(p_i * log(p_i))

    where p_i are the normalized eigenvalues (treated as probabilities).

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues of the density matrix (must be non-negative).
    normalize : bool
        Whether to normalize eigenvalues to sum to 1.

    Returns
    -------
    S : float
        Spectral entropy (nats if eigenvalues sum to 1).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    # Remove negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0)

    if normalize:
        total = np.sum(eigenvalues)
        if total <= 0:
            return 0.0
        eigenvalues = eigenvalues / total

    # Filter out zeros to avoid log(0)
    mask = eigenvalues > 0
    return -np.sum(eigenvalues[mask] * np.log(eigenvalues[mask]))


def rg_flow_lambda(initial_lambda: float, layer: int, beta_rg: float = 0.1) -> float:
    """Compute the RG flow of a coupling across transformer layers.

    lambda(n) = lambda_0 * exp(-beta * n)

    Parameters
    ----------
    initial_lambda : float
        Initial coupling value.
    layer : int
        Transformer layer index (0-based).
    beta_rg : float
        RG beta function coefficient.

    Returns
    -------
    lambda_n : float
        Coupling at layer n.
    """
    return initial_lambda * np.exp(-beta_rg * layer)


def rg_fixed_point(initial_lambda: float, beta_rg: float = 0.1, n_layers: int = 100) -> float:
    """Compute the RG fixed point by iterating the flow.

    Parameters
    ----------
    initial_lambda : float
        Initial coupling.
    beta_rg : float
        RG coefficient.
    n_layers : int
        Number of RG steps.

    Returns
    -------
    lambda_fp : float
        Approximate fixed point.
    """
    return rg_flow_lambda(initial_lambda, n_layers, beta_rg)


def cognitive_mode(f: float, threshold: float = 0.0) -> str:
    """Determine the cognitive mode from the free energy sign.

    F < 0: "factual" (crystal phase, low entropy, ordered)
    F > 0: "creative" (gas phase, high entropy, disordered)

    Parameters
    ----------
    f : float
        Free energy.
    threshold : float
        Threshold for mode boundary.

    Returns
    -------
    mode : str
        "factual" or "creative".
    """
    return "factual" if f < threshold else "creative"


def autoregressive_irreversibility(n_tokens: int, temperature: float = 300.0) -> float:  # noqa: ARG001 (temperature: API compat)
    """Compute the thermodynamic irreversibility of autoregressive generation.

    Each token overwrites previous context, costing at least k_B*T*ln(2)
    per bit of overwritten information.

    Parameters
    ----------
    n_tokens : int
        Number of tokens generated.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    entropy_production : float
        Total entropy production (J/K).
    """
    # Assume ~16 bits of context overwritten per token
    bits_per_token = 16
    total_bits = n_tokens * bits_per_token
    return total_bits * K_B * LN2
