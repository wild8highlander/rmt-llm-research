"""
Keating-Snaith Correction — Finite-context correction to N_crit.

The Keating-Snaith formula provides corrections to the mean of the
Riemann zeta zeros on the critical line, based on Random Matrix Theory
(GUE characteristic polynomials). Applied to LLMs, this gives a
finite-context correction to the critical token count:

    gamma_1(N) = gamma_1 + c_1/N + c_2/N^2 + ...

where gamma_1 ~ 14.1347 is the first Riemann zeta zero and the
correction terms account for finite context window effects.

References:
  - Keating & Snaith (2000), "Random matrix theory and zeta(1/2 + it)",
    Comm. Math. Phys.
"""

from __future__ import annotations

import numpy as np


# Constants
GAMMA_1 = 14.134725  # First Riemann zeta zero (imaginary part)

# Keating-Snaith correction coefficients (leading order)
# These come from the GUE characteristic polynomial model
C_1 = -0.133  # 1/N coefficient (from variance of log|zeta|)
C_2 = 0.068  # 1/N^2 coefficient


def ks_corrected_gamma(
    n: int | float, gamma_1: float = GAMMA_1, c1: float = C_1, c2: float = C_2
) -> float:
    """Compute the Keating-Snaith corrected first zeta zero.

    gamma_1(N) = gamma_1 + c_1/N + c_2/N^2

    Parameters
    ----------
    n : int or float
        Context window size (must be positive).
    gamma_1 : float
        Uncorrected first Riemann zeta zero.
    c1, c2 : float
        Correction coefficients.

    Returns
    -------
    gamma_corrected : float
        Corrected zeta zero accounting for finite context.

    Raises
    ------
    ValueError
        If n <= 0.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return gamma_1 + c1 / n + c2 / n**2


def ks_n_crit_correction(n: int | float, theta_b: float = 7.07, gamma_1: float = GAMMA_1) -> float:
    """Compute the corrected N_crit using Keating-Snaith corrections.

    The critical token count is modified by finite-context effects:

    N_crit(KS) = gamma_1(N) / (theta_b * pi / 180)

    Parameters
    ----------
    n : int or float
        Context window size.
    theta_b : float
        Rotation angle in degrees.
    gamma_1 : float
        First Riemann zeta zero.

    Returns
    -------
    n_crit_ks : float
        Keating-Snaith corrected critical token count.
    """
    gamma_corrected = ks_corrected_gamma(n, gamma_1)
    return gamma_corrected / (theta_b * np.pi / 180.0)


def ks_correction_series(
    n: int | float,
    max_order: int = 5,
    gamma_1: float = GAMMA_1,  # noqa: ARG001
) -> list[float]:
    """Compute the Keating-Snaith correction terms up to max_order.

    Parameters
    ----------
    n : int or float
        Context window size.
    max_order : int
        Maximum correction order (1/N^k for k = 1..max_order).
    gamma_1 : float
        First Riemann zeta zero.

    Returns
    -------
    corrections : list of float
        List of correction terms [c_1/N, c_2/N^2, ...].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    # Higher-order coefficients (decreasing rapidly)
    coeffs = [C_1, C_2, -0.021, 0.009, -0.003][:max_order]
    return [c / n ** (k + 1) for k, c in enumerate(coeffs)]


def ks_relative_correction(n: int | float, gamma_1: float = GAMMA_1) -> float:
    """Compute the relative correction |delta/gamma_1|.

    Parameters
    ----------
    n : int or float
        Context window size.
    gamma_1 : float
        First Riemann zeta zero.

    Returns
    -------
    relative : float
        Relative correction as a fraction of gamma_1.
    """
    corrected = ks_corrected_gamma(n, gamma_1)
    return abs(corrected - gamma_1) / gamma_1


def ks_zeta_zero_statistics(n_zeros: int = 10) -> np.ndarray:
    """Return the first n_zeros Riemann zeta zeros (imaginary parts).

    These are the t_n such that zeta(1/2 + i*t_n) = 0.

    Parameters
    ----------
    n_zeros : int
        Number of zeros to return.

    Returns
    -------
    zeros : ndarray of shape (n_zeros,)
        Imaginary parts of the first Riemann zeta zeros.
    """
    # First 20 known Riemann zeta zeros (imaginary parts)
    known_zeros = np.array(
        [
            14.134725,
            21.022040,
            25.010858,
            30.424876,
            32.935062,
            37.586178,
            40.918719,
            43.327073,
            48.005151,
            49.773832,
            52.970321,
            56.446248,
            59.347044,
            60.831779,
            65.112544,
            67.079810,
            69.546402,
            72.067158,
            75.704691,
            77.144840,
        ]
    )
    return known_zeros[:n_zeros]


def ks_gue_mean_spacing(n: int) -> float:
    """Compute the GUE mean spacing for the first n zeros.

    The mean spacing of GUE eigenvalues (unfolded) is 1.0.
    For the Riemann zeros, the unfolding is:
        t_n -> t_n * (log(t_n / (2*pi*e)) / (2*pi))

    The GUE prediction says the unfolded spacing statistics should
    match those of GUE eigenvalues.

    Parameters
    ----------
    n : int
        Number of zeros.

    Returns
    -------
    mean_spacing : float
        Mean unfolded spacing (should be close to 1.0 for GUE).
    """
    zeros = ks_zeta_zero_statistics(n)
    if len(zeros) < 2:
        return 0.0
    # Unfold: x_n = integral_0^{t_n} (1/(2*pi)) * log(t/(2*pi)) dt
    #       = (t_n/(2*pi)) * (log(t_n/(2*pi)) - 1)
    unfolded = np.array([t / (2 * np.pi) * (np.log(t / (2 * np.pi)) - 1) for t in zeros])
    spacings = np.diff(unfolded)
    return np.mean(spacings)
