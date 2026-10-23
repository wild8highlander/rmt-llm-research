"""rmt_fp -- derived-first-principles core for the RMT x LLM program.

This package contains only results that are *derived* from explicit
assumptions and verified numerically; the phenomenological machinery of
the parent repository (``src/rmt_llm``) is intentionally untouched.

Modules
-------
mp           Marchenko-Pastur bulk + Silverstein fixed-point equation (Steps 4-5)
spiked       BBP transition, outliers, overlaps, inverse estimator (Steps 7-8, 14)
tracy_widom  Edge fluctuations, Johnstone normalization, F1 vs F2 (Step 13)
shrinkage    Optimal spectral filtering operators (Step 14)
entropy      Second-order entropy response to bulk noise (Step 11, corrected)
layer_ops    RMSNorm conservation + nonlinear covariance kernels (Step 2)
hallucination Spectral split definition and utilities (Step 10)
"""

from . import entropy, hallucination, layer_ops, mp, shrinkage, spiked, tracy_widom

__all__ = ["mp", "spiked", "tracy_widom", "shrinkage", "entropy", "layer_ops", "hallucination"]

__version__ = "1.0.0"
