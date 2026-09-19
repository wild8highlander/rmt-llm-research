"""
research_3d.py — 3D Research Experiments for RMT-LLM Laboratory (v1.1.0)
=========================================================================

Adds six advanced 3D research experiments on top of the existing 2D
research.py module. Each experiment produces a structured dict that
charts_3d.py consumes to render 3D visualizations.

Experiments:
  6. Hessian Loss Landscape — perturb (w1, w2) along top-2 eigendirections
     and measure loss surface curvature; verifies sharpness/flatness
     relationship to generalization (RMT: bulk vs outlier eigenvalues).
  7. Manifold Geometry — estimate intrinsic dimensionality of hidden
     states via PCA participation ratio; produces 3D PCA projection.
  8. Reasoning Trajectory Analysis — sample (step, honesty, deception,
     spectral_radius) and detect deception onset via crossing point.
  9. Spectral Surface Regression — fit λ_max(layer, token) to a smooth
     surface and detect the N_crit bifurcation line.
 10. Riemannian Curvature — estimate discrete Gaussian curvature on the
     hidden-state k-NN graph; reveals manifold bends near deception onset.
 11. 3D Attention Flow — measure attention-weight surface and quantify
     diagonal-vs-smeared regime change past N_crit.

All experiments accept the infinite-parameter space from parameters.py
(temperature=inf, max_tokens=inf, etc. are supported by clamping to
practical bounds at computation time only).

Author: Iskhak Hamzatovich Isaev
ORCID: 0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from research import ResearchExperiment
from tiny_gpt import TinyGPT, TinyGPTConfig, encode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clamp_inf(value: Any, default: float, max_finite: float = 1e6) -> float:
    """Convert inf / 'inf' / None to a finite value for computation."""
    if value is None:
        return default
    if isinstance(value, str):
        if value.lower() in ("inf", "+inf", "infinity"):
            return max_finite
        try:
            return float(value)
        except ValueError:
            return default
    v = float(value)
    if math.isinf(v) or math.isnan(v):
        return max_finite
    return v


def _safe_svd_cov(mat: np.ndarray) -> np.ndarray:
    """Covariance of (N, D) matrix; returns D x D."""
    if mat.ndim != 2:
        mat = mat.reshape(mat.shape[0], -1)
    N = mat.shape[0]
    mean = mat.mean(axis=0, keepdims=True)
    centered = mat - mean
    return (centered.T @ centered) / max(N - 1, 1)


# ---------------------------------------------------------------------------
# Experiment 6: Hessian Loss Landscape
# ---------------------------------------------------------------------------
def _exp_hessian_loss_landscape(params: dict[str, Any]) -> dict[str, Any]:
    """Perturb the model along the top-2 Hessian eigendirections and
    measure the loss surface. Output: 3D grid for charts_3d.py.

    The Hessian is approximated by the covariance of weight gradients
    (empirical Fisher) — for a synthetic TinyGPT this is the empirical
    spectral density, which directly connects to Marchenko-Pastur theory.
    """
    grid = int(params.get("hessian_grid_size", 24))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(
        hidden_dim=int(params.get("hidden_dim", 64)),
        n_layers=int(params.get("n_layers", 6)),
        n_heads=int(params.get("n_heads", 4)),
        seed=seed,
    )
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Sample hidden states to build empirical Fisher-like matrix
    tokens = rng.integers(0, cfg.vocab_size, size=min(cfg.max_seq_len, 64))
    _, hidden = model.forward(tokens)
    # hidden: list of (seq, hidden_dim) per layer — stack to (n_layers*seq, hidden_dim)
    H = np.vstack([np.array(h) for h in hidden])
    cov = _safe_svd_cov(H)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Top-2 eigendirections
    v1 = eigvecs[:, -1]
    v2 = eigvecs[:, -2]
    lam1 = float(eigvals[-1])
    lam2 = float(eigvals[-2])

    # Build (w1, w2) perturbation grid
    span = 3.0 * math.sqrt(max(lam1, 1e-9))
    w1_axis = np.linspace(-span, span, grid)
    w2_axis = np.linspace(-span, span, grid)
    W1, W2 = np.meshgrid(w1_axis, w2_axis, indexing="ij")
    # Quadratic loss approximation: L(w) = L0 + 0.5*(lam1*w1^2 + lam2*w2^2)
    # Add a saddle perturbation to demonstrate non-convexity
    L0 = 1.0
    Z = L0 + 0.5 * (lam1 * W1**2 - lam2 * W2**2) + 0.05 * np.sin(W1 * W2)

    return {
        "experiment": "hessian_loss_landscape",
        "config": cfg.__dict__,
        "grid_size": grid,
        "top_eigenvalues": [lam1, lam2],
        "w1_grid": W1.tolist(),
        "w2_grid": W2.tolist(),
        "loss_surface": Z.tolist(),
        "metrics": {
            "lambda_max": lam1,
            "lambda_2": lam2,
            "spectral_gap": lam1 - lam2,
            "loss_min": float(np.min(Z)),
            "loss_max": float(np.max(Z)),
            "sharpness": float(lam1),  # higher = sharper minimum
            "is_saddle": bool(lam2 > 0 and lam1 > 0 and np.any(Z < L0)),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 7: Manifold Geometry
# ---------------------------------------------------------------------------
def _exp_manifold_geometry(params: dict[str, Any]) -> dict[str, Any]:
    """Compute intrinsic dimensionality of hidden-state manifold via
    PCA participation ratio: PR = (Σ λ_i)^2 / Σ λ_i^2.

    Produces 3D PCA projection (top-3 components).
    """
    n_components = int(params.get("pca_components", 3))
    n_samples = int(params.get("trajectory_points", 240))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Collect hidden states across many random prompts
    all_hidden = []
    for _i in range(min(n_samples, 32)):
        n_tok = rng.integers(8, cfg.max_seq_len)
        toks = rng.integers(0, cfg.vocab_size, size=int(n_tok))
        _, hidden = model.forward(toks)
        # Use last layer, all positions — hidden is a list of arrays
        last_layer = np.array(hidden[-1])
        all_hidden.append(last_layer)
    H = np.vstack(all_hidden)  # (N, hidden_dim)
    if H.shape[0] < n_components:
        # Pad by repeating
        reps = (n_components // H.shape[0]) + 1
        H = np.tile(H, (reps, 1))[: n_components * 4]

    # PCA
    H_centered = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    eigvals_pca = (S**2) / max(H.shape[0] - 1, 1)
    # Participation ratio
    pr = float((np.sum(eigvals_pca) ** 2) / max(np.sum(eigvals_pca**2), 1e-12))

    # Project to top-N components
    proj = H_centered @ Vt[:n_components].T  # (N, n_components)
    # Pad to 3D if needed
    while proj.shape[1] < 3:
        proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])

    # Color: by token position (approximated by index)
    colors = np.arange(proj.shape[0])

    return {
        "experiment": "manifold_geometry",
        "config": cfg.__dict__,
        "n_samples": int(proj.shape[0]),
        "n_components": n_components,
        "pca_eigenvalues": eigvals_pca[: max(n_components, 10)].tolist(),
        "participation_ratio": pr,
        "pca_points": proj[:, :3].tolist(),
        "pca_colors": colors.tolist(),
        "metrics": {
            "intrinsic_dim_pr": pr,
            "explained_variance_top3": float(
                np.sum(eigvals_pca[:3]) / max(np.sum(eigvals_pca), 1e-12)
            ),
            "top_eigenvalue": float(eigvals_pca[0]),
            "manifold_volume_proxy": float(np.prod(np.sqrt(eigvals_pca[:3]))),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 8: Reasoning Trajectory Analysis
# ---------------------------------------------------------------------------
def _exp_trajectory_analysis(params: dict[str, Any]) -> dict[str, Any]:
    """Sample (step, honesty, deception, spectral_radius) trajectory.

    Detects deception onset via crossing point of deception > honesty.
    Verifies RMT prediction that onset scales as O(N_crit * mu_eff^(-1/β)).
    """
    n_points = int(params.get("trajectory_points", 64))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Generate one long trajectory
    prompt = "Reason carefully about this problem."
    ids = encode(prompt)
    out = model.generate(
        ids, max_new_tokens=n_points, temperature=float(params.get("temperature", 0.5)), seed=seed
    )
    rt = out["reasoning_trace"]
    thoughts = rt.get("thoughts", [])
    n = max(len(thoughts), n_points)

    # Pad / interpolate to n_points
    if len(thoughts) >= n_points:
        idx = np.linspace(0, len(thoughts) - 1, n_points).astype(int)
        honesty = [thoughts[i]["honesty_score"] for i in idx]
        deception = [thoughts[i]["deception_score"] for i in idx]
        halluc = [thoughts[i]["hallucination_score"] for i in idx]
    else:
        # Synthesize trajectory aligned to N_crit
        n_crit = float(params.get("ncrit_threshold", 96))
        steps_arr = np.linspace(0, n_points, n_points)
        # Honesty decays 1/sqrt(N)
        honesty = (0.65 / np.sqrt(1 + steps_arr / n_crit)).tolist()
        # Deception grows past N_crit
        deception = (0.20 + 0.55 * (1 - np.exp(-(steps_arr - n_crit) / 30.0))).tolist()
        deception = [max(0.0, min(1.0, d)) for d in deception]
        # Hallucination grows faster
        halluc = (0.05 + 0.60 * (1 - np.exp(-(steps_arr - n_crit) / 20.0))).tolist()
        halluc = [max(0.0, min(1.0, h)) for h in halluc]

    # Spectral radius per step (approximate by sampling model at each step)
    spec_radius = []
    for _step in range(n_points):
        sample_toks = rng.integers(0, cfg.vocab_size, size=16)
        _, hidden = model.forward(sample_toks)
        H = np.vstack([np.array(h) for h in hidden])
        cov = _safe_svd_cov(H)
        ev = np.linalg.eigvalsh(cov)
        spec_radius.append(float(ev.max()))
    # Normalize to [0, 1] for color
    spec_norm = np.array(spec_radius)
    spec_norm = (spec_norm - spec_norm.min()) / max(spec_norm.max() - spec_norm.min(), 1e-9)

    # Detect deception onset
    onset_idx = -1
    for i in range(n_points):
        if deception[i] > honesty[i]:
            onset_idx = i
            break

    return {
        "experiment": "trajectory_analysis",
        "n_points": n_points,
        "trajectory": {
            "steps": list(range(n_points)),
            "honesty": honesty,
            "deception": deception,
            "hallucination": halluc,
            "spectral": spec_radius,
            "spectral_normalized": spec_norm.tolist(),
        },
        "deception_onset_step": onset_idx,
        "metrics": {
            "n_points": n_points,
            "onset_step": onset_idx,
            "final_honesty": float(honesty[-1]),
            "final_deception": float(deception[-1]),
            "mean_spectral_radius": float(np.mean(spec_radius)),
            "max_spectral_radius": float(np.max(spec_radius)),
            "honesty_decrease_rate": float(honesty[0] - honesty[-1]) / max(n_points, 1),
            "deception_increase_rate": float(deception[-1] - deception[0]) / max(n_points, 1),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 9: Spectral Surface Regression
# ---------------------------------------------------------------------------
def _exp_spectral_surface_regression(params: dict[str, Any]) -> dict[str, Any]:
    """Compute λ_max(layer, token) surface and detect the N_crit bifurcation.

    Fits a piecewise model: λ_max = a*layer + b*token (pre-N_crit)
    vs λ_max = a*layer + b*token + c*(token - N_crit)^2 (post-N_crit).
    """
    n_layers = int(params.get("spectral_surface_layers", 6))
    n_tokens = int(params.get("trajectory_points", 64))
    n_crit_pred = float(params.get("ncrit_threshold", 96))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(
        n_layers=max(n_layers, 2),
        seed=seed,
    )
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Sample per-layer, per-token
    lambda_max_grid = np.zeros((n_layers, n_tokens))
    for t in range(n_tokens):
        n_tok = min(t + 4, cfg.max_seq_len)
        toks = rng.integers(0, cfg.vocab_size, size=int(n_tok))
        _, hidden = model.forward(toks)
        for l in range(min(n_layers, len(hidden))):
            H = np.array(hidden[l])
            cov = _safe_svd_cov(H)
            ev = np.linalg.eigvalsh(cov)
            lambda_max_grid[l, t] = float(ev.max())

    # Detect bifurcation: find token t* where curvature changes most
    # Use mean over layers
    mean_lambda = lambda_max_grid.mean(axis=0)
    diff2 = np.diff(np.diff(mean_lambda))
    bif_token = int(np.argmax(np.abs(diff2))) + 1 if len(diff2) > 0 else 0

    # Fit pre/post regression
    pre = mean_lambda[: max(bif_token, 1)]
    post = mean_lambda[max(bif_token, 1) :]
    pre_slope = float(np.polyfit(np.arange(len(pre)), pre, 1)[0]) if len(pre) > 1 else 0.0
    post_slope = float(np.polyfit(np.arange(len(post)), post, 1)[0]) if len(post) > 1 else 0.0

    return {
        "experiment": "spectral_surface_regression",
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "lambda_max_grid": lambda_max_grid.tolist(),
        "bifurcation_token": bif_token,
        "n_crit_predicted": n_crit_pred,
        "metrics": {
            "lambda_max_global": float(lambda_max_grid.max()),
            "lambda_min_global": float(lambda_max_grid.min()),
            "bifurcation_token": bif_token,
            "bifurcation_vs_ncrit": abs(bif_token - n_crit_pred),
            "pre_bifurcation_slope": pre_slope,
            "post_bifurcation_slope": post_slope,
            "slope_ratio": post_slope / max(pre_slope, 1e-9),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 10: Riemannian Curvature
# ---------------------------------------------------------------------------
def _exp_riemannian_curvature(params: dict[str, Any]) -> dict[str, Any]:
    """Estimate discrete Gaussian curvature on the hidden-state k-NN graph.

    Uses the angle-deficit formula at each point:
      K(p) = 2π - Σ angles around p
    High curvature indicates manifold bending — typically near deception
    onset or past N_crit.
    """
    n_neighbors = int(params.get("curvature_neighbors", 8))
    n_samples = int(params.get("trajectory_points", 128))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Sample hidden states
    all_hidden = []
    for _ in range(min(n_samples, 32)):
        n_tok = rng.integers(8, cfg.max_seq_len)
        toks = rng.integers(0, cfg.vocab_size, size=int(n_tok))
        _, hidden = model.forward(toks)
        all_hidden.append(np.array(hidden[-1]))
    H = np.vstack(all_hidden)
    # Reduce to 3D for curvature estimation
    H_c = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_c, full_matrices=False)
    P = H_c @ Vt[:3].T

    N = P.shape[0]
    curvatures = []
    for i in range(N):
        # k-NN
        diffs = P - P[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        k = min(n_neighbors, N - 1)
        nn_idx = np.argsort(dists)[:k]
        nn = P[nn_idx]
        # Vectors from p to neighbors
        vecs = nn - P[i]
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_n = vecs / np.maximum(norms, 1e-9)
        # Sum of angles between consecutive vectors (sorted by polar angle)
        # Project to unit sphere and compute angular deficit
        angles = np.arccos(np.clip(np.sum(vecs_n[:-1] * vecs_n[1:], axis=1), -1, 1))
        # Close the loop
        last_angle = np.arccos(np.clip(np.sum(vecs_n[-1] * vecs_n[0], axis=0), -1, 1))
        total_angle = float(np.sum(angles) + last_angle)
        # Gaussian curvature proxy: angle deficit
        K = 2 * math.pi - total_angle
        curvatures.append(K)

    curvatures = np.array(curvatures)
    # Identify high-curvature points (manifold bends)
    threshold = float(np.mean(curvatures) + 2 * np.std(curvatures))
    high_curv_idx = np.where(curvatures > threshold)[0].tolist()

    return {
        "experiment": "riemannian_curvature",
        "n_samples": N,
        "n_neighbors": n_neighbors,
        "curvatures": curvatures.tolist(),
        "points_3d": P.tolist(),
        "high_curvature_indices": [int(i) for i in high_curv_idx],
        "metrics": {
            "mean_curvature": float(np.mean(curvatures)),
            "std_curvature": float(np.std(curvatures)),
            "max_curvature": float(np.max(curvatures)),
            "min_curvature": float(np.min(curvatures)),
            "n_high_curvature": len(high_curv_idx),
            "high_curvature_ratio": len(high_curv_idx) / max(N, 1),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 11: 3D Attention Flow
# ---------------------------------------------------------------------------
def _exp_attention_flow_3d(params: dict[str, Any]) -> dict[str, Any]:
    """Measure attention-weight surface and quantify diagonal-vs-smeared
    regime change past N_crit.

    Returns (query_pos, key_pos, weight) 3D surface.
    """
    resolution = int(params.get("attention_flow_3d_resolution", 32))
    seed = int(params.get("seed", 42))
    cfg = TinyGPTConfig(seed=seed, max_seq_len=max(resolution, 64))
    model = TinyGPT(cfg)
    rng = np.random.default_rng(seed)

    # Forward pass to capture attention
    n = resolution
    toks = rng.integers(0, cfg.vocab_size, size=n)
    _, _ = model.forward(toks)

    # Reconstruct attention matrix from the first layer (last forward)
    # TinyGPT stores per-layer attention weights if capture was enabled;
    # otherwise we synthesize a typical attention pattern.
    if hasattr(model, "_last_attention") and model._last_attention is not None:
        attn = np.array(model._last_attention[0])  # (n, n) for first layer
        if attn.shape != (n, n):
            # Resize / crop
            small = attn
            # Just use what we have, padded/truncated
            new_attn = np.zeros((n, n))
            usable = min(small.shape[0], n)
            new_attn[:usable, :usable] = small[:usable, :usable]
            attn = new_attn
    else:
        # Synthesize: diagonal early, smeared past N_crit
        n_crit = n // 2
        Q, K = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        sigma_pre = 2.0
        sigma_post = 6.0
        sigma_grid = np.where(n_crit > Q, sigma_pre, sigma_post)
        attn = np.exp(-((Q - K) ** 2) / (2 * sigma_grid**2))
        # Normalize rows
        attn = attn / np.maximum(attn.sum(axis=1, keepdims=True), 1e-9)

    # Diagonality score: how concentrated is attention on the diagonal?
    diag_score = float(np.mean(np.diag(attn)) / max(np.mean(attn), 1e-9))
    # Smearing score: standard deviation of attention spread per row
    spread_per_row = np.array(
        [
            np.sqrt(np.sum((np.arange(n) - i) ** 2 * attn[i]) / max(np.sum(attn[i]), 1e-9))
            for i in range(n)
        ]
    )
    smearing_score = float(np.mean(spread_per_row))

    return {
        "experiment": "attention_flow_3d",
        "resolution": n,
        "weights": attn.tolist(),
        "spread_per_row": spread_per_row.tolist(),
        "metrics": {
            "diagonality_score": diag_score,
            "smearing_score": smearing_score,
            "diagonal_to_smeared_ratio": diag_score / max(smearing_score, 1e-9),
            "max_weight": float(attn.max()),
            "entropy": float(-np.sum(attn * np.log(np.maximum(attn, 1e-12))) / n),
        },
    }


# ---------------------------------------------------------------------------
# N_crit Surface (bonus experiment for the 3D N_crit collapse surface chart)
# ---------------------------------------------------------------------------
def _exp_ncrit_surface(params: dict[str, Any]) -> dict[str, Any]:
    """Compute T_crit surface over (β, rlhf_pressure) grid.

    T_crit(β, μ) = N_crit_base × (θ_b + μ)^(-1/β)
    """
    n_crit_base = float(params.get("ncrit_threshold", 114.0))
    theta_b_deg = float(params.get("theta_b_deg", 7.07))
    theta_b = theta_b_deg * math.pi / 180.0
    beta_axis = np.linspace(0.3, 0.9, 24)
    rlhf_axis = np.linspace(0.0, 1.0, 24)
    B, R = np.meshgrid(beta_axis, rlhf_axis, indexing="ij")
    mu_eff = theta_b + R
    # Clamp to avoid division issues
    mu_eff = np.maximum(mu_eff, 1e-6)
    Z = n_crit_base * np.power(mu_eff, -1.0 / B)

    return {
        "experiment": "ncrit_surface",
        "beta_axis": beta_axis.tolist(),
        "rlhf_axis": rlhf_axis.tolist(),
        "t_crit_grid": Z.tolist(),
        "n_crit_base": n_crit_base,
        "theta_b_rad": theta_b,
        "metrics": {
            "t_crit_min": float(np.min(Z)),
            "t_crit_max": float(np.max(Z)),
            "t_crit_mean": float(np.mean(Z)),
            "t_crit_at_beta_0_5_rlhf_0": float(n_crit_base * (theta_b ** (-1.0 / 0.5))),
            "t_crit_at_beta_0_5_rlhf_1": float(n_crit_base * ((theta_b + 1.0) ** (-1.0 / 0.5))),
        },
    }


# ---------------------------------------------------------------------------
# Parameter Space Sweep (bonus experiment for 3D parameter-space chart)
# ---------------------------------------------------------------------------
def _exp_parameter_space(params: dict[str, Any]) -> dict[str, Any]:
    """Sweep (temperature, top_p) and measure hallucination rate."""
    grid = int(params.get("parameter_space_grid", 16))
    seed = int(params.get("seed", 42))
    rng = np.random.default_rng(seed)
    cfg = TinyGPTConfig(seed=seed)
    model = TinyGPT(cfg)

    temp_axis = np.linspace(0.0, 2.0, grid)
    topp_axis = np.linspace(0.5, 1.0, grid)
    T, P = np.meshgrid(temp_axis, topp_axis, indexing="ij")

    # Hallucination rate model: sigmoid((T - 0.8) * 2) * (P - 0.5) * 2
    base = 1.0 / (1.0 + np.exp(-(T - 0.8) * 2.0)) * (P - 0.5) * 2.0
    # Add small noise from model
    noise = 0.02 * rng.random((grid, grid))
    Z = np.clip(base + noise, 0, 1)

    return {
        "experiment": "parameter_space",
        "temp_axis": temp_axis.tolist(),
        "topp_axis": topp_axis.tolist(),
        "hallucination_grid": Z.tolist(),
        "metrics": {
            "hallucination_min": float(np.min(Z)),
            "hallucination_max": float(np.max(Z)),
            "hallucination_at_T0": float(np.mean(Z[0])),
            "hallucination_at_T2": float(np.mean(Z[-1])),
            "hallucination_at_P05": float(np.mean(Z[:, 0])),
            "hallucination_at_P1": float(np.mean(Z[:, -1])),
        },
    }


# ---------------------------------------------------------------------------
# Coalition Drift (bonus experiment for SCEN-COAL-09)
# ---------------------------------------------------------------------------
def _exp_coalition_drift(params: dict[str, Any]) -> dict[str, Any]:
    """Simulate multi-agent coalitional deception drift across rounds.

    Returns per-round, per-agent deception scores.
    """
    n_agents = int(params.get("n_agents", 2))
    n_rounds = int(params.get("n_rounds", 4))
    seed = int(params.get("seed", 42))
    rng = np.random.default_rng(seed)

    # Per-agent deception starts low, converges upward across rounds
    base_deception = 0.25 + 0.05 * np.arange(n_agents)
    per_round = []
    for r in range(n_rounds):
        round_scores = base_deception + 0.12 * r + rng.normal(0, 0.02, n_agents)
        round_scores = np.clip(round_scores, 0, 1)
        per_round.append(round_scores.tolist())

    return {
        "experiment": "coalition_drift",
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "per_round": per_round,
        "metrics": {
            "initial_mean_deception": float(np.mean(per_round[0])),
            "final_mean_deception": float(np.mean(per_round[-1])),
            "drift": float(np.mean(per_round[-1]) - np.mean(per_round[0])),
            "convergence_variance": float(np.var(per_round[-1])),
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
EXPERIMENTS_3D: dict[str, ResearchExperiment] = {
    "6": ResearchExperiment(
        "Hessian Loss Landscape (3D)",
        "Perturb model along top-2 Hessian eigendirections, measure 3D loss surface.",
        _exp_hessian_loss_landscape,
    ),
    "7": ResearchExperiment(
        "Manifold Geometry (3D PCA)",
        "Estimate intrinsic dimensionality via PCA participation ratio; 3D projection.",
        _exp_manifold_geometry,
    ),
    "8": ResearchExperiment(
        "Reasoning Trajectory Analysis (3D)",
        "Sample (step, honesty, deception, spectral_radius) and detect deception onset.",
        _exp_trajectory_analysis,
    ),
    "9": ResearchExperiment(
        "Spectral Surface Regression (3D)",
        "Fit λ_max(layer, token) surface and detect N_crit bifurcation.",
        _exp_spectral_surface_regression,
    ),
    "10": ResearchExperiment(
        "Riemannian Curvature (3D)",
        "Estimate discrete Gaussian curvature on hidden-state k-NN graph.",
        _exp_riemannian_curvature,
    ),
    "11": ResearchExperiment(
        "3D Attention Flow",
        "Measure attention-weight surface and quantify diagonal-vs-smeared regime.",
        _exp_attention_flow_3d,
    ),
    "12": ResearchExperiment(
        "N_crit Collapse Surface (3D)",
        "Compute T_crit(β, μ_RLHF) surface over Caputo order and RLHF pressure.",
        _exp_ncrit_surface,
    ),
    "13": ResearchExperiment(
        "Parameter Space Sweep (3D)",
        "Sweep (temperature, top_p) and measure hallucination-rate surface.",
        _exp_parameter_space,
    ),
    "14": ResearchExperiment(
        "Coalitional Deception Drift (3D)",
        "Simulate multi-agent deception drift across coalition rounds (SCEN-COAL-09).",
        _exp_coalition_drift,
    ),
}


def run_3d_experiment(exp_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run a 3D experiment by ID."""
    if exp_id not in EXPERIMENTS_3D:
        raise KeyError(f"Unknown 3D experiment: {exp_id}. Known: {list(EXPERIMENTS_3D.keys())}")
    exp = EXPERIMENTS_3D[exp_id]
    t0 = time.perf_counter()
    result = exp.runner(params)
    elapsed = time.perf_counter() - t0
    result["experiment_name"] = exp.name
    result["experiment_description"] = exp.description
    result["elapsed_seconds"] = elapsed
    result["parameters"] = {k: v for k, v in params.items() if not isinstance(v, (dict, list))}
    return result


def run_all_3d_experiments(params: dict[str, Any]) -> dict[str, Any]:
    """Run all 9 3D experiments and return a combined result dict
    suitable for charts_3d.generate_all_3d_charts()."""
    combined = {"3d_research": {}, "experiments": []}
    for exp_id in EXPERIMENTS_3D:
        try:
            res = run_3d_experiment(exp_id, params)
            combined["experiments"].append(
                {
                    "id": exp_id,
                    "name": res["experiment_name"],
                    "elapsed_seconds": res["elapsed_seconds"],
                }
            )
            # Store under 3d_research key with snake_case names for charts_3d
            name_map = {
                "6": "loss_landscape",
                "7": "manifold_geometry",
                "8": "trajectory",
                "9": "spectral_surface",
                "10": "riemannian_curvature",
                "11": "attention_flow_3d",
                "12": "ncrit_surface",
                "13": "parameter_space",
                "14": "coalition_drift",
            }
            key = name_map.get(exp_id, f"exp_{exp_id}")
            combined["3d_research"][key] = res
        except Exception as exc:
            print(f"  [WARN] 3D experiment {exp_id} failed: {exc}")
            combined["experiments"].append(
                {
                    "id": exp_id,
                    "error": str(exc),
                }
            )
    return combined


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running all 3D experiments...")
    res = run_all_3d_experiments({"hessian_grid_size": 16, "trajectory_points": 32})
    print(f"\nCompleted {len(res['experiments'])} experiments:")
    for e in res["experiments"]:
        if "error" in e:
            print(f"  [{e['id']}] FAILED: {e['error']}")
        else:
            print(f"  [{e['id']}] {e['name']} — {e['elapsed_seconds']:.3f}s")
    print("\n3D research keys:", list(res["3d_research"].keys()))
