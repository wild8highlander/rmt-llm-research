#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic validation of the RCDA calibration math (marchenko_pastur RCDA block
+ rmt_intervention RCDA functions) — no model weights required.

Checks:
  1. mp_upper_edge matches mp_bounds for q<=1 and brackets the noise lambda_max.
  2. rmt_z_score ~ 0 for a pure-noise covariance; >> 2 for a spiked one.
  3. rayleigh_spike_strength / bbp_excess / alpha_from_bbp: monotonicity and edge cases.
  4. matrix_scan on a planted rank-1 spike W = sigma*G + theta*u v^T:
     lambda_v above the MP edge exactly when theta is supercritical; theta_v recovery.
  5. select_columns: auto_chi2 / topk / auto_tw behave as specified.
  6. rcda_edit: W' = W - alpha*v*p_masked^T exactly; untouched columns intact.
  7. spectral_tw_control: pass for a no-op edit, violations for a random burn.
  8. edit pack round-trip: save -> load -> apply reproduces the same W'.
"""
import base64
import sys

import numpy as np

sys.path.insert(0, "/home/z/my-project/download/rcda_mnn_pocket/Code")

from marchenko_pastur import (  # noqa: E402
    alpha_from_bbp, bbp_excess, bbp_threshold, mp_bounds, mp_upper_edge,
    rmt_z_score, rayleigh_spike_strength, tw_scale,
)
from rmt_intervention import (  # noqa: E402
    matrix_scan, select_columns, rcda_edit, spectral_tw_control,
    save_edit_pack, load_edit_pack, apply_pack, load_state_dict,
)

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILS.append(name)


rng = np.random.default_rng(7)

# ---------------------------------------------------------------- 1-2: MP edge
H, T = 896, 4000
q = H / T
sigma2_true = 2.5
X = rng.normal(0, np.sqrt(sigma2_true), size=(T, H))
C = X.T @ X / T
eigs = np.sort(np.linalg.eigvalsh(C))[::-1]
lam_p = mp_upper_edge(sigma2_true, q)
lm_lo, lm_hi = mp_bounds(q, sigma2_true)
check("mp_upper_edge == mp_bounds(q<=1)", abs(lam_p - lm_hi) < 1e-9 * lam_p,
      f"{lam_p:.4f} vs {lm_hi:.4f}")
z = rmt_z_score(eigs[0], lam_p, T)
check("noise z_RMT within +-4 TW units", abs(z) < 4.0, f"z={z:.2f}")

v_true = rng.normal(size=H)
v_true /= np.linalg.norm(v_true)
theta = 0.8
Xs = X + np.sqrt(T) * theta * sigma2_true * np.outer(np.ones(T), v_true)
Cs = Xs.T @ Xs / T
eig_s = np.sort(np.linalg.eigvalsh(Cs))[::-1]
z_s = rmt_z_score(eig_s[0], lam_p, T)
check("spiked z_RMT > 2", z_s > 2.0, f"z={z_s:.1f} (lambda_max/edge={eig_s[0]/lam_p:.2f})")

# ---------------------------------------------------------------- 3: BBP/alpha
check("bbp_threshold(0.35)", abs(bbp_threshold(0.35) - np.sqrt(0.35)) < 1e-12)
check("theta_v=0 -> delta=0, alpha=0", bbp_excess(0.0, 0.35) == 0.0 and alpha_from_bbp(0.0, 0.35) == 0.0)
check("theta_v<=theta_c -> alpha=0", alpha_from_bbp(bbp_threshold(1.0) * 0.99, 1.0) == 0.0)
alphas = [alpha_from_bbp(t, 0.35) for t in (0.6, 0.8, 1.2, 2.0, 4.0, 20.0)]
check("alpha monotone in theta_v", all(b >= a for a, b in zip(alphas, alphas[1:]))
      and alphas[0] > 0, f"{[round(a,3) for a in alphas]}")
check("alpha capped at 1", alphas[-1] == 1.0)
th_v = rayleigh_spike_strength(lam_p * 1.5, sigma2_true, q)
check("rayleigh_spike_strength positive above edge", th_v > bbp_threshold(q),
      f"theta_v={th_v:.3f} > theta_c={bbp_threshold(q):.3f}")

# ------------------------------------------- 4: matrix_scan on planted spikes
m, n = 896, 2560
sigma_W = 0.02
G = rng.normal(0, sigma_W, size=(m, n))
u = rng.normal(size=m)
u /= np.linalg.norm(u)
w = rng.normal(size=n)
w /= np.linalg.norm(w)
# BBP-informed spike scale: the Rayleigh quotient lambda_v = ||u^T W||^2/n sits at
# sigma^2 (null ~ n*sigma^2 power) and detaches from the edge (1+sqrt(q))^2 sigma^2
# when theta^2 >~ (2 sqrt(q) + q) * n * sigma^2, i.e. theta ~ 1.24 sigma sqrt(n).
theta_s = 3.0 * sigma_W * np.sqrt(n)
W_spiked = G + theta_s * np.outer(u, w)
scan_null = matrix_scan(G, u)
scan_spk = matrix_scan(W_spiked, u)
check("null matrix: v in bulk (lambda_v < lambda_+)",
      not scan_null["supercritical"],
      f"lambda_v/lambda_+={scan_null['lambda_over_edge']:.2f}")
check("spiked matrix: lambda_v > lambda_+",
      scan_spk["supercritical"],
      f"lambda_v/lambda_+={scan_spk['lambda_over_edge']:.2f}, "
      f"theta_v={scan_spk['theta_v']:.2f} > theta_c={scan_spk['theta_c']:.2f}")
check("delta in (0,1) for the spike", 0.0 < scan_spk["delta"] < 1.0,
      f"delta={scan_spk['delta']:.3f}")
check("orthogonal v on spiked matrix stays bulk",
      not matrix_scan(W_spiked, np.eye(m)[0] if not np.allclose(u, np.eye(m)[0]) else np.eye(m)[1])["supercritical"],
      "a random direction should not fire")

# ---------------------------------------------------------------- 5: columns
profile = np.zeros(n)
spike_cols = rng.choice(n, size=12, replace=False)
profile[spike_cols] = 6.0 * sigma_W            # ~6 sigma entries, above the Bonferroni bar
profile += rng.normal(0, sigma_W, size=n)
idx_tw, info_tw = select_columns(profile, sigma_W ** 2,
                                 mp_upper_edge(sigma_W ** 2, m / n), m,
                                 mode="auto_tw", tau=2.0)
idx_chi, info_chi = select_columns(profile, sigma_W ** 2,
                                   mp_upper_edge(sigma_W ** 2, m / n), m,
                                   mode="auto_chi2")
idx_k, info_k = select_columns(profile, sigma_W ** 2,
                               mp_upper_edge(sigma_W ** 2, m / n), m,
                               mode="topk", k=54)
check("topk selects exactly 54", info_k["n_columns"] == 54)
check("auto_chi2 finds the 12 planted columns",
      set(spike_cols).issubset(set(int(i) for i in idx_chi)),
      f"found {info_chi['n_columns']} columns")
check("auto_tw is the strictest gate",
      info_tw["n_columns"] <= info_chi["n_columns"],
      f"auto_tw={info_tw['n_columns']}, auto_chi2={info_chi['n_columns']}")

# ---------------------------------------------------------------- 6: rcda_edit
W0 = G.copy()
v = u.copy()
alpha = 0.6
cols = [int(i) for i in idx_chi]
W1, n_changed = rcda_edit(W0, v, alpha, columns=cols)
p_full = v @ W0
p_mask = np.zeros_like(p_full)
p_mask[cols] = p_full[cols]
expected = W0 - alpha * np.outer(v, p_mask)
check("rcda_edit matches the closed form", np.allclose(W1, expected, atol=1e-6))
check("n_weights_changed == m * n_cols", n_changed == m * len(cols),
      f"{n_changed} vs {m * len(cols)}")
untouched = [j for j in range(n) if j not in cols]
check("untouched columns intact", np.allclose(W1[:, untouched], W0[:, untouched]))
Wf, n_f = rcda_edit(W0, v, 1.0, columns=None)
check("full ablation removes the v-row projection",
      np.allclose(v @ Wf, np.zeros(n), atol=1e-8))

# ---------------------------------------------------------------- 7: TW control
ctrl0 = spectral_tw_control(W0, W0, v)
check("TW control passes for a no-op", ctrl0["verdict"] == "pass")
burn = W0 + rng.normal(0, 0.3 * sigma_W, size=W0.shape)
ctrl1 = spectral_tw_control(W0, burn, v)
check("TW control flags a random burn", ctrl1["verdict"] == "violation",
      f"violations={ctrl1['n_violations']}")
ctrl_edit = spectral_tw_control(W0, W1, v)
check("TW control reports the target component", ctrl_edit["target_index"] >= 0,
      f"target_shift={ctrl_edit['target_shift_tw_units']:.2f} TW units")

# ------------------------------------------------------- 8: pack round-trip
rep = {
    "layer": 15, "matrix": "down_proj", "state_dict_key": "model.layers.15.mlp.down_proj.weight",
    "vector": v, "alpha": alpha,
    "columns": {"mode": "auto_chi2", "n_columns": len(cols), "indices": cols},
    "status": "applied",
}
save_edit_pack("/tmp/test_pack.json", [rep], meta={"test": True})
edits, meta = load_edit_pack("/tmp/test_pack.json")
sd = {"model.layers.15.mlp.down_proj.weight": W0.copy()}
apply_pack(sd, edits)
check("pack round-trip reproduces W'", np.allclose(sd["model.layers.15.mlp.down_proj.weight"], W1, atol=1e-6))

# b64 vector length sanity
raw = base64.b64decode(open("/tmp/test_pack.json").read() and
                       __import__("json").load(open("/tmp/test_pack.json"))["edits"][0]["vector_b64_f32"])
check("vector stored as float32 m-vector", len(raw) == 4 * m)

print()
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
    sys.exit(1)
print("RESULT: all checks passed")
