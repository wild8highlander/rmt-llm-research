"""
stage3_ncrit.scaling_test — the (mu_eff)^(-1/beta) law across models.

The per-model survival fits give tau_m (= T_crit scale). The mu_eff of
each model must be measured OUTSIDE the survival fit — the default proxy
is the token-mean KL divergence between the base and the tuned model on a
shared prompt set (stage 5); any external estimate can be passed.

Test 1 (exponent): weighted regression log tau_m = log c - s * log mu_m.
  The law predicts s = 1/beta; beta ~ 0.5 predicts s ~ 2. The CI of s is
  computed by resampling tau_m within their bootstrap CIs.
  Pre-registered falsifier: a slope in [1.2, 2.8] away from ~2 still
  SUPPORTS the fractional law with a different beta but REFUTES beta = 0.5;
  a slope CI covering neither 2 nor a consistent 1/beta refutes the law.
Test 2 (consistency): fitted betas of individual models must overlap
  within bootstrap CIs (pairwise check).
"""

from __future__ import annotations

import numpy as np


def fit_scaling(tau: np.ndarray, tau_ci: np.ndarray, mu: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Weighted (in log space) regression log tau = log c - s log mu.

    tau_ci: (n, 2) array of bootstrap CI bounds (used for resampling noise;
    the log(hi)-log(lo) width is only a valid sigma proxy for log-symmetric
    CIs — the bootstrap quantiles here are approximately log-symmetric).
    Returns slope s, its CI, the implied beta = 1/s, the regression dof
    (n_models - 2) and a power_warning when the fit is underpowered.
    """
    tau = np.asarray(tau, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    tau_ci = np.asarray(tau_ci, dtype=np.float64)
    if len(tau) < 3:
        raise ValueError("need >= 3 models for a scaling regression")
    if np.any(mu <= 0) or np.any(tau <= 0):
        raise ValueError("tau and mu must be positive")
    if not np.all(np.isfinite(tau_ci)) or np.any(tau_ci[:, 0] <= 0) or np.any(tau_ci[:, 1] <= 0):
        raise ValueError(
            "tau_ci must be finite positive bounds; a non-positive bound means the "
            "bootstrap CI crossed 0 — report that model's fit as unreliable instead of masking it"
        )
    x = np.log(mu)
    w = 1.0 / np.clip((np.log(tau_ci[:, 1]) - np.log(tau_ci[:, 0])) / (2 * 1.96), 1e-3, None)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    beta_hat = np.linalg.solve(A.T @ W @ A, A.T @ W @ np.log(tau))
    resid = np.log(tau) - A @ beta_hat
    s = float(-beta_hat[1])
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        tau_b = tau * np.exp(rng.normal(0.0, 1.0, size=len(tau)) / np.clip(w, 1.0, None))
        try:
            bh = np.linalg.solve(A.T @ W @ A, A.T @ W @ np.log(tau_b))
            slopes.append(-bh[1])
        except np.linalg.LinAlgError:
            continue
    slopes = np.asarray(slopes)
    out = {
        "slope": s,
        "slope_ci95": [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))],
        "implied_beta": 1.0 / s if s > 0 else None,
        "intercept_log_c": float(beta_hat[0]),
        "resid_rms": float(np.sqrt(np.mean(resid**2))),
        "n_models": int(len(tau)),
        "dof": int(len(tau) - 2),
    }
    if len(tau) < 5:
        out["power_warning"] = (
            f"n={len(tau)} models; a 2-parameter regression on n<5 points has dof={len(tau) - 2} "
            "and the slope CI is correspondingly wide"
        )
    return out


def beta_consistency(betas: list[dict]) -> dict:
    """Pairwise overlap of beta bootstrap CIs: betas = [{beta, ci95}, ...].

    Returns all_consistent=None when there is nothing to compare (fewer
    than two beta fits) — the caller must not conflate that with a
    positive or a negative consistency finding."""
    if len(betas) < 2:
        return {"pairs": [], "all_consistent": None, "n_models": len(betas),
                "note": "fewer than 2 per-model beta fits; consistency not checkable"}
    pairs = []
    for i in range(len(betas)):
        for j in range(i + 1, len(betas)):
            lo_i, hi_i = betas[i]["ci95"]
            lo_j, hi_j = betas[j]["ci95"]
            overlap = max(lo_i, lo_j) <= min(hi_i, hi_j)
            pairs.append({"i": i, "j": j, "overlap": bool(overlap),
                          "beta_i": betas[i]["beta"], "beta_j": betas[j]["beta"]})
    all_overlap = all(p["overlap"] for p in pairs) if pairs else False
    return {"pairs": pairs, "all_consistent": bool(all_overlap), "n_models": len(betas)}


def verdict(scaling: dict, cons: dict, target: float = 2.0, tol: float = 0.4) -> str:
    """Pre-registered verdicts, mirroring the plan's honesty clause:
    a negative result is a result. Missing beta fits are reported as such,
    never silently folded into the slope verdict."""
    lo, hi = scaling["slope_ci95"]
    if cons.get("all_consistent") is None:
        if lo <= target <= hi:
            return ("partially supported: slope CI covers 2, but no per-model beta fits "
                    "were provided — beta consistency could not be checked")
        return ("not resolved: no per-model beta fits (consistency unchecked) and slope CI "
                f"[{lo:.2f}, {hi:.2f}] excludes {target:g}")
    near_target = abs(scaling["slope"] - target) <= tol
    if lo <= target <= hi and cons["all_consistent"]:
        return "supported: slope CI covers 2 and betas agree across models"
    if cons["all_consistent"] and scaling["implied_beta"] and 0.2 < scaling["implied_beta"] < 0.9:
        return ("partially supported: betas agree, but the slope CI excludes 2 — "
                f"the fractional law holds with beta~{scaling['implied_beta']:.2f}, not 0.5")
    if near_target and not cons["all_consistent"]:
        return ("partially supported: slope ~2 on average, but per-model betas "
                "disagree — the law is not stable across models")
    return ("not supported: slope CI "
            f"[{lo:.2f}, {hi:.2f}] excludes 2 and per-model betas disagree — "
            "report the negative result as it stands")
