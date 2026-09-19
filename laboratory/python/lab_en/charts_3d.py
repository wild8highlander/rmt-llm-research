"""
charts_3d.py — 3D Visualization Module for RMT-LLM Laboratory (v1.1.0)
======================================================================

Generates professional 3D charts for advanced research:

  1. 3D Loss Landscape (parameter-space surface)
  2. 3D Hidden-State Manifold (PCA projection)
  3. 3D Spectral Surface (layer × token × eigenvalue)
  4. 3D Deception Trajectory (step × honesty × deception)
  5. 3D Attention Flow (query × key × weight vector field)
  6. 3D N_crit Collapse Surface (β × rlhf × t_crit)
  7. 3D Parameter Space Sweep (temperature × top_p × hallucination)
  8. 3D Coalitional Deception Drift (round × agent × deception)

Output formats per chart (matches main charts.py):
  - PNG at 600 DPI (raster)
  - PDF (vector)
  - SVG (vector)
  - Plotly HTML (interactive 3D — fully rotatable)

All charts honor the 3D parameters defined in shared/schema.json:
  elevation_3d, azimuth_3d, color_map_3d,
  hessian_grid_size, trajectory_points, spectral_surface_layers,
  pca_components, attention_flow_3d_resolution, parameter_space_grid.

Author: Iskhak Hamzatovich Isaev
ORCID: 0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import os
from typing import Any

# Matplotlib setup
import matplotlib


matplotlib.use("Agg")
import contextlib

import matplotlib.font_manager as fm


with contextlib.suppress(Exception):
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np


# Optional Plotly for interactive 3D
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_all_3d_charts(
    results: dict[str, Any],
    out_dir: str = "results/charts_3d",
    dpi: int = 600,
    params_3d: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Generate every 3D chart for the given results dict.

    Returns a dict: { "png": [...], "pdf": [...], "svg": [...], "html": [...] }
    """
    os.makedirs(out_dir, exist_ok=True)
    written = {"png": [], "pdf": [], "svg": [], "html": []}

    p = params_3d or {}
    elevation = float(p.get("elevation_3d", 30))
    azimuth = float(p.get("azimuth_3d", 45))
    cmap = p.get("color_map_3d", "viridis")
    grid = int(p.get("hessian_grid_size", 24))
    traj_pts = int(p.get("trajectory_points", 64))
    spectral_layers = int(p.get("spectral_surface_layers", 6))
    pca_comp = int(p.get("pca_components", 3))
    flow_res = int(p.get("attention_flow_3d_resolution", 32))
    pgrid = int(p.get("parameter_space_grid", 16))

    chart_specs = [
        (
            "3d_01_loss_landscape",
            lambda r: _chart_loss_landscape(r, grid, elevation, azimuth, cmap),
        ),
        (
            "3d_02_hidden_manifold",
            lambda r: _chart_hidden_manifold(r, pca_comp, elevation, azimuth, cmap),
        ),
        (
            "3d_03_spectral_surface",
            lambda r: _chart_spectral_surface(r, spectral_layers, elevation, azimuth, cmap),
        ),
        (
            "3d_04_deception_trajectory",
            lambda r: _chart_deception_trajectory(r, traj_pts, elevation, azimuth, cmap),
        ),
        (
            "3d_05_attention_flow",
            lambda r: _chart_attention_flow(r, flow_res, elevation, azimuth, cmap),
        ),
        ("3d_06_ncrit_surface", lambda r: _chart_ncrit_surface(r, elevation, azimuth, cmap)),
        (
            "3d_07_parameter_space",
            lambda r: _chart_parameter_space(r, pgrid, elevation, azimuth, cmap),
        ),
        ("3d_08_coalition_drift", lambda r: _chart_coalition_drift(r, elevation, azimuth, cmap)),
    ]

    for name, fn in chart_specs:
        try:
            fig = fn(results)
            if fig is None:
                continue
            png_path = os.path.join(out_dir, f"{name}.png")
            pdf_path = os.path.join(out_dir, f"{name}.pdf")
            svg_path = os.path.join(out_dir, f"{name}.svg")
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
            fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written["png"].append(png_path)
            written["pdf"].append(pdf_path)
            written["svg"].append(svg_path)
        except Exception as exc:
            print(f"  [WARN] 3D chart {name} failed: {exc}")

    # Interactive Plotly 3D dashboard
    if HAS_PLOTLY:
        try:
            html_path = os.path.join(out_dir, "interactive_3d_dashboard.html")
            _plotly_3d_dashboard(results, html_path, p)
            written["html"].append(html_path)
        except Exception as exc:
            print(f"  [WARN] plotly 3D dashboard failed: {exc}")

    return written


# ---------------------------------------------------------------------------
# 1. 3D Loss Landscape
# ---------------------------------------------------------------------------
def _chart_loss_landscape(
    results: dict[str, Any], grid: int, elevation: float, azimuth: float, cmap: str
):
    """3D surface of the loss landscape over (w1, w2) parameter perturbations.

    Uses the Hessian eigenvalues from research_3d results if available;
    otherwise synthesizes a saddle-shaped loss surface.
    """
    hessian = results.get("3d_research", {}).get("loss_landscape", {})
    if hessian:
        X = np.array(hessian.get("w1_grid", []))
        Y = np.array(hessian.get("w2_grid", []))
        Z = np.array(hessian.get("loss_surface", []))
        if X.size == 0 or Y.size == 0 or Z.size == 0:
            X, Y, Z = _synth_loss_landscape(grid)
    else:
        X, Y, Z = _synth_loss_landscape(grid)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, Z, cmap=cmap, alpha=0.92, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    ax.set_xlabel("Weight perturbation w₁")
    ax.set_ylabel("Weight perturbation w₂")
    ax.set_zlabel("Loss L(w)")
    ax.set_title("3D Loss Landscape — Hessian Eigenvalue Perturbation")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="Loss")
    return fig


def _synth_loss_landscape(grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a saddle-shaped loss surface for demonstration."""
    x = np.linspace(-2, 2, grid)
    y = np.linspace(-2, 2, grid)
    X, Y = np.meshgrid(x, y)
    # Loss = 0.5*(w1^2 - w2^2) + 0.1*sin(w1*w2) — saddle + ripple
    Z = 0.5 * (X**2 - Y**2) + 0.1 * np.sin(X * Y) + 1.5
    return X, Y, Z


# ---------------------------------------------------------------------------
# 2. 3D Hidden-State Manifold (PCA projection)
# ---------------------------------------------------------------------------
def _chart_hidden_manifold(
    results: dict[str, Any], n_components: int, elevation: float, azimuth: float, cmap: str
):
    """3D scatter of hidden states projected to top-3 PCA components.

    Color encodes the token position (temporal order). Reveals the
    intrinsic geometry of the hidden-state manifold.
    """
    manifold = results.get("3d_research", {}).get("manifold_geometry", {})
    if manifold and "pca_points" in manifold:
        pts = np.array(manifold["pca_points"])
        colors = np.array(manifold.get("pca_colors", range(len(pts))))
    else:
        # Synthesize a Swiss-roll-like manifold
        n = 240
        t = np.linspace(0, 4 * np.pi, n)
        r = t + 0.5
        pts = np.column_stack(
            [
                r * np.cos(t) + np.random.normal(0, 0.05, n),
                r * np.sin(t) + np.random.normal(0, 0.05, n),
                t + np.random.normal(0, 0.05, n),
            ]
        )
        colors = t

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        cmap=cmap,
        s=42,
        alpha=0.85,
        edgecolors="black",
        linewidth=0.4,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D Hidden-State Manifold — PCA Projection (n={len(pts)})")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label="Token position / time")
    return fig


# ---------------------------------------------------------------------------
# 3. 3D Spectral Surface (layer × token × eigenvalue)
# ---------------------------------------------------------------------------
def _chart_spectral_surface(
    results: dict[str, Any], n_layers: int, elevation: float, azimuth: float, cmap: str
):
    """3D surface of the largest eigenvalue per (layer, token) pair.

    RMT interpretation: collapse is visible as the surface buckling past
    N_crit. Marchenko-Pastur upper bound is shown as a translucent plane.
    """
    spec_surf = results.get("3d_research", {}).get("spectral_surface", {})
    if spec_surf and "lambda_max_grid" in spec_surf:
        Z = np.array(spec_surf["lambda_max_grid"])
        n_layers, n_tokens = Z.shape
    else:
        # Synthesize: eigenvalues inflate past N_crit
        n_layers = max(n_layers, 4)
        n_tokens = 32
        layers = np.arange(n_layers)
        tokens = np.arange(n_tokens)
        L, T = np.meshgrid(layers, tokens, indexing="ij")
        n_crit = 16
        Z = (
            1.5
            + 0.05 * L
            + np.where(n_crit < T, 2.0 * (1 - np.exp(-(T - n_crit) / 8.0)), 0.2 * T / n_crit)
        )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    L = np.arange(n_layers)
    T = np.arange(Z.shape[1])
    LL, TT = np.meshgrid(L, T, indexing="ij")
    surf = ax.plot_surface(LL, TT, Z, cmap=cmap, alpha=0.92, linewidth=0, antialiased=True)
    # MP upper bound plane
    mp_upper = float(results.get("spectral", {}).get("mp_upper", 2.7))
    ax.plot_surface(LL, TT, np.full_like(Z, mp_upper), alpha=0.18, color="red")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Token position")
    ax.set_zlabel("λ_max(layer, token)")
    ax.set_title("3D Spectral Surface — λ_max Across Layers & Tokens")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="λ_max")
    return fig


# ---------------------------------------------------------------------------
# 4. 3D Deception Trajectory (step × honesty × deception)
# ---------------------------------------------------------------------------
def _chart_deception_trajectory(
    results: dict[str, Any], n_points: int, elevation: float, azimuth: float, cmap: str
):
    """3D trajectory of (step, honesty, deception) showing how reasoning drifts.

    A red marker shows where deception crosses honesty — the predicted
    deception-onset point from RMT theory.
    """
    traj = results.get("3d_research", {}).get("trajectory", {})
    if traj and "honesty" in traj:
        steps = np.array(traj.get("steps", range(len(traj["honesty"]))))
        h = np.array(traj["honesty"])
        d = np.array(traj["deception"])
        spec = np.array(traj.get("spectral", h * 0 + 1.0))
    else:
        steps = np.arange(n_points)
        h = np.linspace(0.65, 0.18, n_points) + np.random.normal(0, 0.02, n_points)
        d = np.linspace(0.20, 0.78, n_points) + np.random.normal(0, 0.02, n_points)
        spec = np.linspace(1.2, 3.4, n_points)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Trajectory colored by step
    sc = ax.scatter(steps, h, d, c=spec, cmap=cmap, s=60, edgecolors="black", linewidth=0.5)
    # Connecting line
    ax.plot(steps, h, d, color="black", linewidth=1.6, alpha=0.55)
    # Find deception-onset (first step where d > h)
    onset_idx = int(np.argmax(d > h)) if np.any(d > h) else -1
    if onset_idx >= 0:
        ax.scatter(
            [steps[onset_idx]],
            [h[onset_idx]],
            [d[onset_idx]],
            color="red",
            s=250,
            marker="*",
            linewidth=2,
            edgecolors="black",
            label=f"Deception onset @ step {steps[onset_idx]}",
        )
        ax.legend(loc="upper left")
    ax.set_xlabel("Reasoning step")
    ax.set_ylabel("Honesty score")
    ax.set_zlabel("Deception score")
    ax.set_zlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("3D Deception Trajectory — Honesty vs Deception vs Spectral Radius")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label="Spectral radius ρ")
    return fig


# ---------------------------------------------------------------------------
# 5. 3D Attention Flow (query × key × weight vector field)
# ---------------------------------------------------------------------------
def _chart_attention_flow(
    results: dict[str, Any], resolution: int, elevation: float, azimuth: float, cmap: str
):
    """3D surface of attention weights: query position × key position × weight.

    Reveals attention-pattern collapse and diagonal smearing past N_crit.
    """
    flow = results.get("3d_research", {}).get("attention_flow_3d", {})
    if flow and "weights" in flow:
        W = np.array(flow["weights"])
    else:
        # Synthesize: diagonal-dominant early, smeared later
        n = resolution
        q = np.arange(n)
        k = np.arange(n)
        Q, K = np.meshgrid(q, k, indexing="ij")
        # Distance from diagonal + noise; later positions get smeared
        W = np.exp(-((Q - K) ** 2) / (2 * 4.0**2))
        # Smearing: increase sigma past mid-sequence
        smear = np.where(n / 2 < Q, 8.0, 4.0)
        W = np.exp(-((Q - K) ** 2) / (2 * smear**2)) + 0.05 * np.random.rand(n, n)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    q = np.arange(W.shape[0])
    k = np.arange(W.shape[1])
    Q, K = np.meshgrid(q, k, indexing="ij")
    surf = ax.plot_surface(Q, K, W, cmap=cmap, alpha=0.92, linewidth=0, antialiased=True)
    ax.set_xlabel("Query position")
    ax.set_ylabel("Key position")
    ax.set_zlabel("Attention weight")
    ax.set_title("3D Attention Flow — Query × Key × Weight Surface")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="Attention weight")
    return fig


# ---------------------------------------------------------------------------
# 6. 3D N_crit Collapse Surface (β × rlhf × t_crit)
# ---------------------------------------------------------------------------
def _chart_ncrit_surface(results: dict[str, Any], elevation: float, azimuth: float, cmap: str):
    """3D surface of predicted T_crit = (θ_b + rlhf)^(-1/β) × N_crit.

    Sweeps β ∈ [0.3, 0.9] and rlhf_pressure ∈ [0, 1] to show how
    RMT-predicted cognitive collapse horizon varies.
    """
    surf_data = results.get("3d_research", {}).get("ncrit_surface", {})
    if surf_data and "t_crit_grid" in surf_data:
        Z = np.array(surf_data["t_crit_grid"])
        beta_axis = np.array(surf_data.get("beta_axis", np.linspace(0.3, 0.9, Z.shape[0])))
        rlhf_axis = np.array(surf_data.get("rlhf_axis", np.linspace(0, 1, Z.shape[1])))
    else:
        beta_axis = np.linspace(0.3, 0.9, 24)
        rlhf_axis = np.linspace(0.0, 1.0, 24)
        B, R = np.meshgrid(beta_axis, rlhf_axis, indexing="ij")
        n_crit_base = float(results.get("ncrit_threshold", 114.0))
        theta_b = 0.124  # ~7.07° in rad
        mu_eff = theta_b + R
        # T_crit = n_crit * mu_eff^(-1/beta)
        Z = n_crit_base * np.power(mu_eff, -1.0 / B)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    B, R = np.meshgrid(beta_axis, rlhf_axis, indexing="ij")
    surf = ax.plot_surface(B, R, Z, cmap=cmap, alpha=0.92, linewidth=0, antialiased=True)
    ax.set_xlabel("β (Caputo fractional order)")
    ax.set_ylabel("RLHF pressure μ_RLHF")
    ax.set_zlabel("T_crit (predicted token horizon)")
    ax.set_title("3D N_crit Collapse Surface — T_crit(β, RLHF)")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="T_crit")
    return fig


# ---------------------------------------------------------------------------
# 7. 3D Parameter Space Sweep (temperature × top_p × hallucination)
# ---------------------------------------------------------------------------
def _chart_parameter_space(
    results: dict[str, Any], grid: int, elevation: float, azimuth: float, cmap: str
):
    """3D surface of measured hallucination rate as a function of
    temperature ∈ [0, 2], top_p ∈ [0.5, 1.0], max_tokens ∈ [64, 1024].

    Demonstrates the infinite-parameter launch system producing a 3D
    parameter-space map of model behavior.
    """
    ps = results.get("3d_research", {}).get("parameter_space", {})
    if ps and "hallucination_grid" in ps:
        Z = np.array(ps["hallucination_grid"])
        temp_axis = np.array(ps.get("temp_axis", np.linspace(0, 2, Z.shape[0])))
        topp_axis = np.array(ps.get("topp_axis", np.linspace(0.5, 1.0, Z.shape[1])))
    else:
        temp_axis = np.linspace(0.0, 2.0, grid)
        topp_axis = np.linspace(0.5, 1.0, grid)
        T, P = np.meshgrid(temp_axis, topp_axis, indexing="ij")
        # Hallucination rate = sigmoid((T - 0.8) * 2) * (P - 0.5) * 2 + small noise
        Z = 1.0 / (1.0 + np.exp(-(T - 0.8) * 2.0)) * (P - 0.5) * 2.0
        Z = np.clip(Z, 0, 1) + 0.02 * np.random.rand(grid, grid)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    T, P = np.meshgrid(temp_axis, topp_axis, indexing="ij")
    surf = ax.plot_surface(T, P, Z, cmap=cmap, alpha=0.92, linewidth=0, antialiased=True)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("top_p")
    ax.set_zlabel("Hallucination rate")
    ax.set_zlim(0, 1)
    ax.set_title("3D Parameter Space Sweep — Hallucination vs (T, top_p)")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1, label="Hallucination rate")
    return fig


# ---------------------------------------------------------------------------
# 8. 3D Coalitional Deception Drift (round × agent × deception)
# ---------------------------------------------------------------------------
def _chart_coalition_drift(results: dict[str, Any], elevation: float, azimuth: float, cmap: str):
    """3D bars showing per-agent deception scores across coalition rounds.

    Used by SCEN-COAL-09 to visualize how alpha and beta converge on
    a coordinated deceptive answer.
    """
    coal = results.get("3d_research", {}).get("coalition_drift", {})
    if coal and "per_round" in coal:
        per_round = coal["per_round"]
        n_rounds = len(per_round)
        n_agents = len(per_round[0]) if per_round else 2
        agents = list(range(n_agents))
        rounds = list(range(n_rounds))
        R, A = np.meshgrid(rounds, agents, indexing="ij")
        Z = np.array([[per_round[r][a] for a in agents] for r in rounds])
    else:
        n_rounds, n_agents = 4, 2
        rounds = np.arange(n_rounds)
        agents = np.arange(n_agents)
        R, A = np.meshgrid(rounds, agents, indexing="ij")
        Z = np.array([[0.30 + 0.10 * r + 0.05 * a for a in agents] for r in rounds])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        R.ravel(),
        A.ravel(),
        np.zeros_like(R.ravel()),
        0.6,
        0.6,
        Z.ravel(),
        shade=True,
        color=plt.get_cmap(cmap)(Z.ravel() / max(Z.max(), 1e-9)),
    )
    ax.set_xlabel("Coalition round")
    ax.set_ylabel("Agent index")
    ax.set_zlabel("Deception score")
    ax.set_zlim(0, 1)
    ax.set_title("3D Coalitional Deception Drift — Per-Agent × Per-Round")
    ax.view_init(elev=elevation, azim=azimuth)
    return fig


# ---------------------------------------------------------------------------
# Plotly interactive 3D dashboard
# ---------------------------------------------------------------------------
def _plotly_3d_dashboard(results: dict[str, Any], out_path: str, params_3d: dict[str, Any]) -> None:
    """Single interactive HTML with multiple 3D surfaces, fully rotatable."""
    if not HAS_PLOTLY:
        return

    grid = int(params_3d.get("hessian_grid_size", 24))
    cmap = params_3d.get("color_map_3d", "Viridis")

    # Build 4 main 3D surfaces in a 2x2 subplot grid
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "surface"}, {"type": "surface"}],
            [{"type": "scatter3d"}, {"type": "surface"}],
        ],
        subplot_titles=(
            "Loss Landscape",
            "Spectral Surface",
            "Deception Trajectory",
            "N_crit Surface",
        ),
    )

    # 1. Loss landscape
    X, Y, Z = _synth_loss_landscape(grid)
    fig.add_trace(
        go.Surface(x=X[0], y=Y[:, 0], z=Z, colorscale=cmap, showscale=False), row=1, col=1
    )

    # 2. Spectral surface (synthetic)
    n_layers, n_tokens = 6, 32
    n_crit = 16
    L, T = np.meshgrid(np.arange(n_layers), np.arange(n_tokens), indexing="ij")
    Z_spec = (
        1.5
        + 0.05 * L
        + np.where(n_crit < T, 2.0 * (1 - np.exp(-(T - n_crit) / 8.0)), 0.2 * T / n_crit)
    )
    fig.add_trace(
        go.Surface(x=L[0], y=T[:, 0], z=Z_spec, colorscale="Plasma", showscale=False), row=1, col=2
    )

    # 3. Deception trajectory (3D scatter)
    n_pts = int(params_3d.get("trajectory_points", 64))
    steps = np.arange(n_pts)
    h = np.linspace(0.65, 0.18, n_pts)
    d = np.linspace(0.20, 0.78, n_pts)
    spec = np.linspace(1.2, 3.4, n_pts)
    fig.add_trace(
        go.Scatter3d(
            x=steps,
            y=h,
            z=d,
            mode="lines+markers",
            marker={"size": 4, "color": spec, "colorscale": "Inferno", "showscale": True},
            line={"color": "black", "width": 3},
            name="Trajectory",
        ),
        row=2,
        col=1,
    )

    # 4. N_crit surface
    beta_axis = np.linspace(0.3, 0.9, 24)
    rlhf_axis = np.linspace(0.0, 1.0, 24)
    B, R = np.meshgrid(beta_axis, rlhf_axis, indexing="ij")
    n_crit_base = float(results.get("ncrit_threshold", 114.0))
    theta_b = 0.124
    mu_eff = theta_b + R
    Z_ncrit = n_crit_base * np.power(mu_eff, -1.0 / B)
    fig.add_trace(
        go.Surface(x=B[0], y=R[:, 0], z=Z_ncrit, colorscale="Cividis", showscale=False),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="RMT-LLM Laboratory — Interactive 3D Dashboard (v1.1.0)",
        height=900,
        width=1300,
        template="plotly_white",
        scene={"aspectmode": "cube"},
        scene2={"aspectmode": "cube"},
        scene3={"aspectmode": "cube"},
        scene4={"aspectmode": "cube"},
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fake_results = {
        "3d_research": {
            "loss_landscape": {},
            "manifold_geometry": {},
            "spectral_surface": {},
            "trajectory": {},
            "attention_flow_3d": {},
            "ncrit_surface": {},
            "parameter_space": {},
            "coalition_drift": {},
        },
        "spectral": {"mp_upper": 2.7},
        "ncrit_threshold": 114.0,
    }
    written = generate_all_3d_charts(fake_results, "results/charts_3d")
    print(f"3D charts written: {sum(len(v) for v in written.values())} files")
    for fmt, paths in written.items():
        print(f"  {fmt}: {len(paths)} files")
