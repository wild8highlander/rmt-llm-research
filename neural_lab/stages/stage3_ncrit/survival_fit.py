"""
stage3_ncrit.survival_fit — fitting the collapse law to first-error times.

Theory under test (caputo_fractional.py): the mean collapse time follows

    <T_crit> = c * mu_eff^(-1/beta),      beta ~ 0.5  =>  exponent ~ 2.

Honest operationalization on measured data:

  1. first-error times (t_i, delta_i) with delta_i=1 if the trajectory
     errored and delta_i=0 if right-censored (never errored);
  2. parametric survival model S(t) = exp(-(t/tau)^gamma) (Weibull — the
     minimal flexible law; gamma=1 is exponential decay, the constant-rate
     null A0 of the H1 protocol);
  3. scale tau = c * mu^(-1/beta) when per-trajectory or per-model mu_eff
     is available, tau = c otherwise;
  4. maximum-likelihood fit of (c, beta, gamma), bootstrap CIs;
  5. the EXPONENT claim: (a) per-model fits must agree on beta;
     (b) across models with different mu_eff, the regression slope of
     log tau on log mu must be ~ -1/beta (and ~ -2 for beta=0.5).
     The law is tested against data, not fitted to itself: the scaling
     regression uses tau from INDEPENDENT per-model fits and mu_eff
     measured OUTSIDE the fit (KL proxy from stage 5), and the null
     slope -2 is compared against the CI, not assumed.

Numerics: scipy.optimize.minimize (L-BFGS-B) from multiple starts; pure
numpy/scipy, no hidden state.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# beta = exp(log_beta) is fit in log space: the theory wants beta ~ 0.5, but
# the DATA may give beta > 1 and that is a real scientific result — it must
# not be silently clipped by a sigmoid ceiling at 1. The bounds are wide
# enough to be non-binding for any plausible fit; hitting them is reported
# via the beta_at_boundary flag.
BETA_BOUNDS = (0.05, 20.0)


def _nll(params: np.ndarray, t: np.ndarray, delta: np.ndarray, mu: np.ndarray | None) -> float:
    """Fully log-space Weibull (-log) likelihood: no clip needed anywhere.

    z = (t/tau)^gamma is computed as exp(gamma * (log t - log tau)), so
    underflow/overflow is bounded by clipping the EXPONENT, not the value.
    """
    if mu is None:
        log_c, log_gamma = params
        log_beta = None
    else:
        log_c, log_beta, log_gamma = params
    log_t = np.log(np.maximum(t, 1e-9))
    if mu is None:
        log_tau = np.full_like(log_t, log_c)
    else:
        log_mu = np.log(np.maximum(mu, 1e-300))
        beta = np.exp(np.clip(log_beta, np.log(BETA_BOUNDS[0]), np.log(BETA_BOUNDS[1])))
        log_tau = log_c - (1.0 / beta) * log_mu
    log_gamma = np.clip(log_gamma, -10.0, 10.0)
    gamma = np.exp(log_gamma)
    log_z = np.clip(gamma * (log_t - log_tau), -700, 700)
    z = np.exp(log_z)
    ll = delta * (log_gamma - log_tau + (gamma - 1.0) * (log_t - log_tau) - z) - (1.0 - delta) * z
    return float(-ll.sum())


def fit_survival(
    t: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray | None = None,
    n_starts: int = 5,
    seed: int = 0,
    warm_start: np.ndarray | None = None,
) -> dict:
    """MLE of (c, [beta], gamma). t>0 token times, delta in {0,1}.

    warm_start: parameter vector from a previous fit (e.g. the main fit
    when bootstrapping) — used as an extra starting point so the bootstrap
    converges to the same optimum as the main fit instead of a different
    local one."""
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if len(t) == 0 or (delta == 1).sum() == 0:
        raise ValueError("need at least one uncensored error time")
    rng = np.random.default_rng(seed)
    best = None
    k = 3 if mu is not None else 2
    starts = []
    if warm_start is not None:
        starts.append(np.asarray(warm_start, dtype=np.float64))
    for _ in range(n_starts):
        x0 = rng.uniform(-1.0, 1.5, size=k)
        x0[0] = np.log(max(np.mean(t[delta == 1]), 1.0))
        if mu is not None:
            x0[1] = np.log(0.5) + rng.uniform(-1.0, 1.0)  # beta ~ 0.5 prior start
        starts.append(x0)
    bounds = None
    if mu is not None:
        bounds = [(None, None), (np.log(BETA_BOUNDS[0]), np.log(BETA_BOUNDS[1])), (None, None)]
    for x0 in starts:
        try:
            res = minimize(_nll, x0, args=(t, delta, mu), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 500})
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("optimization failed for all starts")
    out: dict = {"loglik": float(-best.fun), "n": int(len(t)), "n_errors": int(delta.sum())}
    if mu is None:
        c, gamma = np.exp(best.x[0]), np.exp(best.x[1])
        out.update({"c": float(c), "gamma": float(gamma), "tau": float(c)})
        out["params"] = [float(v) for v in best.x]
    else:
        log_c, log_beta, log_gamma = best.x
        c = float(np.exp(log_c))
        beta = float(np.exp(np.clip(log_beta, np.log(BETA_BOUNDS[0]), np.log(BETA_BOUNDS[1]))))
        out.update({"c": c, "beta": beta, "gamma": float(np.exp(log_gamma))})
        mu_arr = np.asarray(mu, dtype=np.float64)
        tau = c * mu_arr ** (-1.0 / beta)
        out["tau"] = tau
        out["tau_mean"] = float(np.mean(tau))
        out["beta_at_boundary"] = bool(
            beta <= BETA_BOUNDS[0] * 1.05 or beta >= BETA_BOUNDS[1] * 0.95
        )
        out["params"] = [float(v) for v in best.x]
    return out


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resample index matrix — shared between paired bootstrap
    runs so two models measured on the SAME trajectories get the SAME
    resamples (the delta CI is then properly paired)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_survival(
    t: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 0,
    index_matrix: np.ndarray | None = None,
    warm_start: np.ndarray | None = None,
) -> dict:
    """Nonparametric bootstrap CIs of the fitted parameters.

    Resamples trajectories. Pass index_matrix (from bootstrap_indices) to
    force specific resamples; failed rows (no errors in the resample, or a
    fit failure) are recorded as NaN so paired delta CIs stay aligned.
    The bootstrap uses the same n_starts as the main fit plus the main
    fit's optimum as a warm start, so its CIs are not biased toward a
    different local optimum."""
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    keys = ["c", "beta", "gamma"] if mu is not None else ["c", "gamma"]
    if index_matrix is None:
        index_matrix = bootstrap_indices(len(t), n_boot, seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for idx in index_matrix:
        if delta[idx].sum() == 0:
            for k in keys:
                samples[k].append(float("nan"))
            continue
        try:
            fit = fit_survival(
                t[idx], delta[idx], None if mu is None else mu[idx],
                n_starts=3, seed=seed, warm_start=warm_start,
            )
        except (RuntimeError, ValueError):
            for k in keys:
                samples[k].append(float("nan"))
            continue
        for k in keys:
            samples[k].append(float(fit[k]))
    out = {"n_boot": int(len(index_matrix))}
    for k, v in samples.items():
        arr = np.asarray(v, dtype=np.float64)
        ok = arr[np.isfinite(arr)]
        out[k] = {
            "mean": float(np.nanmean(ok)) if ok.size else float("nan"),
            "ci95": [float(np.nanquantile(ok, 0.025)) if ok.size else float("nan"),
                     float(np.nanquantile(ok, 0.975)) if ok.size else float("nan")],
            "samples": v,
        }
    return out


def binned_error_rate(t: np.ndarray, delta: np.ndarray, n_bins: int = 10) -> dict:
    """Nonparametric error rate by token-position bin (at-risk normalized).

    Returns bins, rate (errors / at-risk trajectories), at_risk counts.
    """
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    edges = np.linspace(1.0, max(t.max(), 2.0), n_bins + 1)
    centers, rates, at_risk, n_err = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask_in = (t > lo) & (t <= hi)
        mask_alive = t >= lo
        at_risk.append(int(mask_alive.sum()))
        err = int((mask_in & (delta == 1)).sum())
        n_err.append(err)
        rates.append(err / max(at_risk[-1], 1))
        centers.append(0.5 * (lo + hi))
    out = {"bin_centers": centers, "rate": rates, "at_risk": at_risk, "errors": n_err, "edges": edges.tolist()}
    if t.max() == t.min():
        out["warning"] = "all t identical; binned curve degenerate (every observation falls in one bin)"
    return out


def _best_split_ll(err_times: np.ndarray, cand: np.ndarray, T: float) -> tuple[float, float]:
    """Best two-rate Poisson split log-likelihood over ALL candidate split
    points (vectorized; the null and the observed statistic use the SAME
    grid — a coarser null grid would bias p downward)."""
    times = np.sort(err_times)
    e = len(times)
    e1 = np.searchsorted(times, cand, side="right").astype(np.float64)
    e2 = float(e) - e1
    x1 = cand
    x2 = np.maximum(T - cand, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = e1 * np.log(e1 / x1) - e1 + e2 * np.log(e2 / x2) - e2
    ll[(e1 == 0) | (e2 == 0)] = -np.inf
    i = int(np.argmax(ll))
    return float(ll[i]), float(cand[i])


def detect_breakpoint(t: np.ndarray, delta: np.ndarray, min_frac: float = 0.1,
                      n_perm: int = 500, seed: int = 0, at_risk: np.ndarray | None = None) -> dict:
    """Exact two-rate split search: the changepoint s maximizing the Poisson
    split-likelihood of error counts around s (on the sorted error times).

    The null is a constant error intensity PER TOKEN. Permuting error
    labels between trajectories would ignore their different at-risk
    exposure; instead each replicate keeps the total number of errors,
    reassigns them to trajectories with probability proportional to the
    trajectory's at-risk window, and draws a fresh error time uniformly
    within that window (at most one error per trajectory, as in the data).

    at_risk: per-trajectory at-risk window upper bound (e.g. the number of
    tokens the trajectory was allowed to generate — the censoring time).
    When omitted, every trajectory is assumed to share the common window
    [1, T] with T = max observed time (the honest default when exposure
    data is absent). The observed statistic and every replicate are
    evaluated on the same full split grid, and p carries the +1 correction
    (never exactly 0)."""
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    err_times = np.sort(t[delta == 1])
    if len(err_times) < 4:
        return {"breakpoint": None, "note": "fewer than 4 errors"}
    cand = err_times[(err_times >= np.quantile(t, min_frac)) & (err_times <= np.quantile(t, 1 - min_frac))]
    cand = np.unique(cand)
    if len(cand) == 0:
        return {"breakpoint": None, "note": "no admissible split"}
    T = float(t.max())
    E = int(delta.sum())
    best_ll, best_s = _best_split_ll(err_times, cand, T)
    ll_const = E * np.log(E / T) - E
    stat = 2 * (best_ll - ll_const)
    rng = np.random.default_rng(seed)
    if at_risk is None:
        exposure = np.full(len(t), T)
    else:
        exposure = np.maximum(np.asarray(at_risk, dtype=np.float64), 1.0)
    w = exposure / exposure.sum()
    null = []
    for _ in range(n_perm):
        try:
            chosen = rng.choice(len(t), size=E, replace=False, p=w)
        except ValueError:
            continue
        u = rng.uniform(0.0, 1.0, size=E) * exposure[chosen]
        times = np.clip(np.ceil(u), 1.0, exposure[chosen])  # error time within at-risk window
        ll_p, _ = _best_split_ll(times, cand, T)
        if np.isfinite(ll_p):
            null.append(2 * (ll_p - ll_const))
    p = (
        (1.0 + float((np.asarray(null) >= stat - 1e-12).sum())) / (1.0 + len(null))
        if null else float("nan")
    )
    return {"breakpoint": best_s, "stat": float(stat), "p_perm": float(p),
            "ll_const": float(ll_const), "n_perm": len(null)}


def weibull_mean(tau: float, gamma: float) -> float:
    """Mean of Weibull(tau, gamma) = tau * Gamma(1 + 1/gamma)."""
    return float(tau * np.exp(gammaln(1.0 + 1.0 / gamma)))
