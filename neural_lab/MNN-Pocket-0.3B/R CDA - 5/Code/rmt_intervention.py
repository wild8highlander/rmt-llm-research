# -*- coding: utf-8 -*-
"""
RMT interventions on the hidden states of a language model (MNN-Pocket-0.3B / Qwen3).

Idea (see the compatibility report, sections 5-6): the activations of this model
are deeply supercritical with respect to the Marchenko-Pastur (MP) law — a few
eigen-directions ("outliers") carry the signal, while the rest of the bulk lies
inside the MP volume. The hook below is attached to the output of every decoder
layer, takes the SVD of the hidden-state matrix (T x H) and reshapes the spectrum
relative to the MP edge:

    lambda_i = S_i^2 / T;   gamma = H / T;   sigma^2 ~= median(lambda_i);
    lambda_plus = sigma^2 * (1 + sqrt(gamma))^2
    =>  S_plus = sqrt(lambda_plus * T) = median(S) * (1 + sqrt(gamma))

(Estimating sigma^2 as the median of the empirical spectrum is a bulk-level
working heuristic rather than a strict theoretical bound; for supercritical
activations it is conservative, see the discussion in the report, section 6.)

Modes:
  clip    — singular values above S_plus are clamped to S_plus (outlier limiting);
  dampen  — the excess above S_plus is scaled by alpha: S_plus + (S - S_plus)*alpha;
  project — only the top_k singular components are kept;
  random  — a random subset of top_k singular values is clamped to S_plus
            (control ablation: the same cardinality of intervention as in clip,
            but WITHOUT spectral structure — the mask is redrawn on every call);
  none    — pure pass-through (no SVD at all).

IMPORTANT (T=1): during incremental decoding with a KV cache the hook receives
a (1 x H) matrix — rank-1 — and any spectral intervention degenerates into a
no-op (S_plus ~= 31 * median(S) at T=1). That is why generation in
benchmark_rmt.py runs with use_cache=False: every step recomputes the whole
context, so the hook sees a (T x H) matrix with T = prompt length + tokens
generated so far.
"""

import numpy as np
import torch

MODES = ("clip", "dampen", "project", "random", "none")


class RMTInterventionHook:
    """Forward hook that filters the hidden-state spectrum relative to the MP edge."""

    def __init__(self, mode="clip", alpha=0.5, top_k=10, seed=0, layer_idx=None):
        assert mode in MODES, f"unknown mode: {mode}"
        self.mode = mode
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.layer_idx = layer_idx
        # per-layer mask generator so that the ablation stays reproducible
        self.rng = np.random.default_rng((seed * 977 + (layer_idx or 0) * 31) % (2**32))
        # statistics (filled in over calls)
        self.n_calls = 0              # calls with a meaningful SVD (T >= 2)
        self.n_intervened = 0         # total number of modified singular values
        self.energy_kept = []         # fraction of ||H||_F^2 preserved by the filter
        self.max_ratio = []           # S_max / median(S) — supercriticality per call

    # ------------------------------------------------------------------ core

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hs, rest = output[0], output[1:]
            is_tuple = True
        else:
            hs, rest = output, None
            is_tuple = False

        if not torch.is_tensor(hs) or hs.dim() != 3 or hs.shape[1] < 2:
            return output  # T < 2: spectral filter is undefined

        if self.mode == "none":
            return output  # pass-through without SVD

        B, T, H = hs.shape
        assert B == 1, "implemented for batch size 1"
        Hmat = hs[0].float()                                  # (T, H)

        U, S, Vh = torch.linalg.svd(Hmat, full_matrices=False)
        k = S.shape[0]
        s_med = torch.median(S)
        s_plus = s_med * (1.0 + float(np.sqrt(H / T)))        # S_plus = median(S)*(1+sqrt(gamma))

        self.n_calls += 1
        self.max_ratio.append(float(S[0] / s_med))

        S_new = S.clone()
        n_int = 0
        if self.mode == "clip":
            mask = S > s_plus
            n_int = int(mask.sum())
            if n_int:
                S_new[mask] = s_plus
        elif self.mode == "dampen":
            mask = S > s_plus
            n_int = int(mask.sum())
            if n_int:
                S_new[mask] = s_plus + (S[mask] - s_plus) * self.alpha
        elif self.mode == "project":
            n_int = max(0, k - min(self.top_k, k))
            if n_int:
                S_new[min(self.top_k, k):] = 0.0
        elif self.mode == "random":
            kk = min(self.top_k, k)
            idx = torch.from_numpy(self.rng.choice(k, size=kk, replace=False)).long()
            S_new[idx] = torch.clamp(S[idx], max=float(s_plus))
            n_int = kk

        self.n_intervened += n_int
        e = float((S_new ** 2).sum() / (S ** 2).sum())
        self.energy_kept.append(e)

        if n_int == 0:
            return output  # nothing changed — leave the output untouched (no SVD rounding noise)

        H_new = (U * S_new) @ Vh                               # U @ diag(S_new) @ Vh
        hs_new = H_new.to(hs.dtype).unsqueeze(0)
        return (hs_new,) + rest if is_tuple else hs_new

    # --------------------------------------------------------------- reporting

    def stats(self):
        if not self.n_calls:
            return {"n_calls": 0}
        return {
            "n_calls": self.n_calls,
            "mean_intervened": self.n_intervened / self.n_calls,
            "mean_energy_kept": float(np.mean(self.energy_kept)),
            "mean_max_ratio": float(np.mean(self.max_ratio)),
        }


def install_rmt_hooks(model, mode, alpha=0.5, top_k=10, seed=0):
    """Attach an RMTInterventionHook to every decoder layer. -> (handles, hooks)."""
    layers = model.model.layers
    handles, hooks = [], []
    for i, layer in enumerate(layers):
        h = RMTInterventionHook(mode=mode, alpha=alpha, top_k=top_k, seed=seed, layer_idx=i)
        handles.append(layer.register_forward_hook(h))
        hooks.append(h)
    return handles, hooks


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ===========================================================================
# RCDA — RMT-Calibrated Directional Ablation (weight-space edits)
# ===========================================================================
# In contrast to the hook modes above (runtime spectral filtering of hidden
# states), RCDA edits the WEIGHTS themselves:
#
#     W' = W - alpha * v (v^T W)              (full rank-1 ablation)
#     W' = W - alpha * v (p_masked)           (column-restricted ablation,
#          p = v^T W, p_masked keeps only the RMT-confirmed columns)
#
# The direction v (unit norm, output space R^m) comes from the RCDS probe
# (pocket_rmt_probe.py --rcds-probe): the bad-vs-good mean activation
# difference Delta_h^{(l)} normalized. alpha is calibrated from the BBP
# excess of the direction in the target matrix, columns are selected by an
# RMT threshold — every number in the report has a statistical meaning
# (see README_RCDA.md and marchenko_pastur.py).

import base64  # noqa: E402
import glob as _glob  # noqa: E402
import json as _json  # noqa: E402
import os as _os  # noqa: E402

from marchenko_pastur import (  # noqa: E402
    alpha_from_bbp,
    bbp_excess,
    bbp_threshold,
    mp_upper_edge,
    rayleigh_spike_strength,
    tw_scale,
)

RCDA_MATRICES = {
    # matrix name -> (qualname suffix in a HF state dict)
    "down_proj": "mlp.down_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
}
COLUMN_MODES = ("auto_tw", "auto_chi2", "topk")


def matrix_scan(W, v, tau=2.0):
    """Full RCDA spectral passport of a matrix W (m x n, torch/numpy) vs unit v (m,).

    Returns a dict with the quantities of the plan (sections 3-5):
      sigma2_w       — bulk variance median(eig(W W^T / n))      (repo heuristic)
      lambda_plus_w  — MP upper edge sigma2_w (1+sqrt(q))^2,  q = m/n
      lambda_v       — v^T C v = ||v^T W||^2 / n              (projection mass)
      lambda_over_edge, theta_v, theta_c, delta, alpha0(kappa=0.5)
      s_v            — ||v^T W|| / ||W||_F                    (projection mass, raw)
      profile        — v^T W (length n, float32)
    """
    import numpy as _np
    import torch as _torch

    if isinstance(W, _torch.Tensor):
        W = W.detach().cpu().to(_torch.float32).numpy()
    W = _np.asarray(W, dtype=_np.float64)
    v = _np.asarray(v, dtype=_np.float64).ravel()
    v = v / _np.linalg.norm(v)
    m, n = W.shape
    q = m / n

    # C = W W^T / n via the singular values of W (no m x m product needed)
    S = _np.linalg.svd(W, compute_uv=False)
    eigC = S ** 2 / n
    sigma2_w = float(_np.median(eigC))
    lambda_plus_w = mp_upper_edge(sigma2_w, q)
    lambda_max_w = float(eigC[0])

    p = v @ W                                    # (n,)
    lambda_v = float(p @ p / n)
    theta_v = rayleigh_spike_strength(lambda_v, sigma2_w, q)
    theta_c = bbp_threshold(q)
    delta = bbp_excess(theta_v, q)
    return {
        "shape": [int(m), int(n)],
        "q": float(q),
        "sigma2_w": sigma2_w,
        "lambda_max_w": lambda_max_w,
        "lambda_plus_w": lambda_plus_w,
        "lambda_v": lambda_v,
        "lambda_over_edge": lambda_v / lambda_plus_w,
        "theta_v": theta_v,
        "theta_c": theta_c,
        "delta": delta,
        "alpha0_kappa0.5": alpha_from_bbp(theta_v, q, kappa=0.5),
        "s_v": float(_np.linalg.norm(p) / _np.linalg.norm(W)),
        "supercritical": bool(lambda_v > lambda_plus_w),
        "profile": p.astype(_np.float32),
    }


def select_columns(profile, sigma2_w, lambda_plus_w, m, mode="auto_tw",
                   k=54, tau=2.0, chi2_eta=0.05):
    """Select the input columns j whose profile entry carries RMT-confirmed signal.

    Modes
    -----
    auto_tw   — plan, section 4: keep j with (p_j^2 / n) > lambda_+ + tau * TW-scale.
                The strictest gate: a column passes only if its projection mass on
                the covariance scale exceeds the bulk edge itself (plus tau
                Tracy-Widom units). May legitimately select zero columns — that
                means "no RMT-confirmed write path; do not edit".
    auto_chi2 — Gaussian null per column: under the null p_j ~ N(0, sigma2_w),
                so p_j^2/sigma2_w ~ chi2_1; keep j above the Bonferroni quantile
                (family-wise level chi2_eta over n columns). A softer,
                multiple-testing-corrected gate.
    topk      — literal top-k by |p_j| (paper-style fixed cardinality,
                section 4: k ~ 48 000 / 896 ~ 54).

    Returns (indices, info) with the indices sorted by descending |p_j|.
    """
    import numpy as _np

    p = _np.asarray(profile, dtype=_np.float64).ravel()
    n = p.shape[0]
    abs_p2 = p ** 2
    order = _np.argsort(-abs_p2, kind="stable")
    info = {"mode": mode, "n_columns": 0, "threshold": None}

    if mode == "topk":
        kk = max(0, min(int(k), n))
        idx = order[:kk]
        info["threshold"] = float(abs_p2[idx[-1]]) if kk else None
    elif mode == "auto_tw":
        thr = lambda_plus_w + tau * tw_scale(lambda_plus_w, m)
        # per-column mass on the covariance scale: v^T C v contribution = p_j^2 / n
        idx = order[abs_p2[order] / n > thr]
        info["threshold"] = float(thr * n)
    elif mode == "auto_chi2":
        from scipy.stats import chi2 as _chi2
        z2 = float(_chi2.ppf(1.0 - chi2_eta / n, df=1.0))   # ~ (3.5-4 sigma)^2
        idx = order[abs_p2[order] > z2 * sigma2_w]
        info["threshold"] = float(z2 * sigma2_w)
    else:
        raise ValueError(f"unknown column mode: {mode} (choose from {COLUMN_MODES})")

    idx = _np.sort(idx)                                  # canonical order by column id
    idx = idx[_np.argsort(-abs_p2[idx], kind="stable")]  # report by strength
    info["n_columns"] = int(idx.size)
    return idx, info


def rcda_edit(W, v, alpha, columns=None):
    """Apply W' = W - alpha * v p_masked^T (columns=None -> full profile p = v^T W).

    Returns (W', n_weights_changed). The edit is a rank-<=1 update confined to
    the selected columns; with columns=None it is the classic full rank-1
    directional ablation.
    """
    import numpy as _np
    import torch as _torch

    was_torch = isinstance(W, _torch.Tensor)
    W_np = W.detach().cpu().to(_torch.float32).numpy() if was_torch else _np.asarray(W)
    v = _np.asarray(v, dtype=_np.float64).ravel()
    v = v / _np.linalg.norm(v)
    p = v @ W_np.astype(_np.float64)
    if columns is not None:
        mask = _np.zeros_like(p, dtype=bool)
        mask[_np.asarray(list(columns), dtype=int)] = True
        p = p * mask
    dW = _np.outer(v, p)
    W_new = W_np.astype(_np.float64) - float(alpha) * dW
    n_changed = int(_np.count_nonzero(dW))
    W_new_t = _torch.from_numpy(W_new.astype(_np.float32))
    return (W_new_t if was_torch else W_new), n_changed


def spectral_tw_control(W_before, W_after, v, tau_violation=3.0, top_r=10):
    """Plan, section 7.2: Tracy-Widom control of the healthy singular spectrum.

    Compares the top-r eigenvalues mu_i = sigma_i^2 / n of C = W W^T / n before
    and after the edit. The component whose left singular vector is closest to
    v is the "target"; every other top component must stay within
    tau_violation * TW-scale(lambda_+) (plan: +-2 TW units is clean, > 3 is a
    violation). Returns a dict with a pass/fail verdict.
    """
    import numpy as _np

    Wb = _np.asarray(W_before, dtype=_np.float64)
    Wa = _np.asarray(W_after, dtype=_np.float64)
    v = _np.asarray(v, dtype=_np.float64).ravel()
    v = v / _np.linalg.norm(v)
    m, n = Wb.shape
    q = m / n

    Ub, Sb, _ = _np.linalg.svd(Wb, full_matrices=False)
    Ua, Sa, _ = _np.linalg.svd(Wa, full_matrices=False)
    mub, mua = Sb ** 2 / n, Sa ** 2 / n
    sigma2_w = float(_np.median(mub))
    lambda_plus_w = mp_upper_edge(sigma2_w, q)
    scale = tw_scale(lambda_plus_w, m)

    r = min(top_r, len(mub))
    cos = _np.abs(Ub[:, :r].T @ v)
    target = int(_np.argmax(cos))
    comps = []
    for i in range(r):
        shift_tw = float((mua[i] - mub[i]) / scale)
        comps.append({
            "i": i, "mu_before": float(mub[i]), "mu_after": float(mua[i]),
            "shift_tw_units": shift_tw, "cos_to_v": float(cos[i]),
            "is_target": i == target,
        })
    violations = [c for c in comps
                  if not c["is_target"] and abs(c["shift_tw_units"]) > tau_violation]
    target_shift = comps[target]["shift_tw_units"] if r else 0.0
    return {
        "tw_scale": float(scale),
        "lambda_plus_w": float(lambda_plus_w),
        "target_index": target,
        "target_shift_tw_units": target_shift,
        "components": comps,
        "n_violations": len(violations),
        "tau_violation": float(tau_violation),
        "verdict": "pass" if not violations else "violation",
        "violations": violations,
    }


# ------------------------------------------------------------------ edit pack

def _vec_to_b64(vec):
    import numpy as _np
    return base64.b64encode(_np.asarray(vec, dtype=_np.float32).tobytes()).decode("ascii")


def _b64_to_vec(b64):
    import numpy as _np
    return _np.frombuffer(base64.b64decode(b64), dtype=_np.float32).astype(_np.float64)


def state_dict_key(layer, matrix):
    try:
        suffix = RCDA_MATRICES[matrix]
    except KeyError:
        raise ValueError(f"unknown matrix {matrix!r}; choose from {sorted(RCDA_MATRICES)}")
    return f"model.layers.{int(layer)}.{suffix}"


def load_state_dict(path):
    """Load a .safetensors file (or every safetensors file of a directory)."""
    import torch as _torch
    from safetensors.torch import load_file

    paths = [path] if _os.path.isfile(path) else sorted(
        _glob.glob(_os.path.join(path, "*.safetensors")))
    if not paths:
        raise FileNotFoundError(f"no safetensors found at {path}")
    sd = {}
    for p in paths:
        sd.update(load_file(p))
    return sd


def build_edits(specs, kappa=0.5, alpha_mode="auto_bbp", alpha_fixed=1.0,
                column_mode="auto_tw", columns_k=54, tw_tau=2.0, chi2_eta=0.05,
                vectors=None, scans=None):
    """Turn edit specs [{layer, matrix, vector?}, ...] into resolved edit plans.

    Every spec gets a matrix_scan; alpha is either calibrated from the BBP
    excess (alpha_mode='auto_bbp', plan section 5) or taken fixed; columns are
    selected per ``column_mode``. Vectors not given in ``specs`` are looked up
    in ``vectors`` {(layer, matrix) -> unit vector}.
    """
    import numpy as _np

    edits = []
    for spec in specs:
        layer, matrix = int(spec["layer"]), spec["matrix"]
        key = (layer, matrix)
        if spec.get("vector") is not None:
            v = _np.asarray(spec["vector"], dtype=_np.float64).ravel()
        elif vectors and key in vectors:
            v = _np.asarray(vectors[key], dtype=_np.float64).ravel()
        else:
            raise KeyError(f"no direction vector for (layer={layer}, matrix={matrix})")
        norm = float(_np.linalg.norm(v))
        v = v / norm
        edits.append({"layer": layer, "matrix": matrix, "vector": v,
                      "vector_norm_raw": norm})
    return edits


def resolve_edit(sd, plan, kappa=0.5, alpha_mode="auto_bbp", alpha_fixed=1.0,
                 column_mode="auto_tw", columns_k=54, tw_tau=2.0, chi2_eta=0.05,
                 full_ablation=False):
    """Resolve one edit plan against a state dict -> report dict + edited tensor."""
    import numpy as _np
    import torch as _torch

    layer, matrix, v = plan["layer"], plan["matrix"], plan["vector"]
    key = state_dict_key(layer, matrix)
    if key not in sd:
        raise KeyError(f"tensor {key} not found in the state dict")
    W = sd[key]
    if isinstance(W, _np.ndarray):
        W = _torch.from_numpy(W.astype(_np.float32))
    W = W.detach().cpu().to(_torch.float32)   # base weights may be bf16 on disk
    scan = matrix_scan(W, v)
    m, n = scan["shape"]

    if full_ablation:
        columns, col_info, alpha = None, {"mode": "full_rank1", "n_columns": n}, (
            alpha_fixed if alpha_mode == "fixed" else scan["alpha0_kappa0.5"])
    else:
        columns, col_info = select_columns(
            scan["profile"], scan["sigma2_w"], scan["lambda_plus_w"], m,
            mode=column_mode, k=columns_k, tau=tw_tau, chi2_eta=chi2_eta)
        if alpha_mode == "fixed":
            alpha = float(alpha_fixed)
        else:
            alpha = alpha_from_bbp(scan["theta_v"], scan["q"], kappa=kappa)

    report = {
        "layer": layer, "matrix": matrix, "state_dict_key": key,
        "vector_norm_raw": plan["vector_norm_raw"],
        "scan": {k: scan[k] for k in scan if k != "profile"},
        "columns": {"mode": col_info["mode"], "n_columns": col_info["n_columns"],
                    "threshold_profile_sq": col_info["threshold"],
                    "indices": [int(i) for i in (columns if columns is not None else [])]},
        "alpha_mode": alpha_mode, "alpha": float(alpha),
        "kappa": float(kappa) if alpha_mode == "auto_bbp" else None,
        "tw_tau": float(tw_tau),
    }
    if columns is not None and col_info["n_columns"] == 0:
        report["status"] = "aborted_no_rmt_columns"
        report["reason"] = (
            "auto_tw selected 0 columns: no input column of this matrix carries "
            "projection mass above lambda_+ + tau*TW-scale (plan section 4). "
            "Per the RMT discipline the edit is refused; use column_mode "
            "auto_chi2/topk or a different layer/matrix.")
        return report, None

    W_new, n_changed = rcda_edit(W, v, alpha, columns=columns)
    ctrl = spectral_tw_control(W.cpu().numpy().astype(_np.float64),
                               W_new.numpy().astype(_np.float64), v)
    report["n_weights_changed"] = int(n_changed)
    report["rel_frobenius_change"] = float(_np.linalg.norm(
        W_new.numpy() - W.cpu().numpy()) / _np.linalg.norm(W.cpu().numpy()))
    report["tw_control"] = ctrl
    report["status"] = "applied"
    return report, W_new


def save_edit_pack(path, reports, meta=None):
    """Persist resolved edits (vectors included) as a self-contained JSON pack."""
    import numpy as _np
    pack = {"meta": meta or {}, "edits": []}
    for r in reports:
        e = {k: r[k] for k in r if k != "vector"}
        if r.get("vector") is not None:
            e["vector_b64_f32"] = _vec_to_b64(r["vector"])
        pack["edits"].append(e)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(pack, f, ensure_ascii=False, indent=1)


def load_edit_pack(path):
    """Load an edit pack; returns (list of edit dicts with 'vector' as np array, meta)."""
    with open(path, encoding="utf-8") as f:
        pack = _json.load(f)
    edits = []
    for e in pack["edits"]:
        e = dict(e)
        if "vector_b64_f32" in e:
            e["vector"] = _b64_to_vec(e.pop("vector_b64_f32"))
        edits.append(e)
    return edits, pack.get("meta", {})


def apply_pack(sd, edits, skip_tw=True):
    """Apply a resolved edit pack (e.g. from --apply-pack) to a state dict in place.

    Returns the list of per-edit summaries. Edits with status
    'aborted_no_rmt_columns' are skipped with a warning.
    """
    import numpy as _np

    out = []
    for e in edits:
        if e.get("status") == "aborted_no_rmt_columns":
            out.append({"layer": e["layer"], "matrix": e["matrix"], "status": "skipped_aborted_in_pack"})
            continue
        key = e.get("state_dict_key") or state_dict_key(e["layer"], e["matrix"])
        if key not in sd:
            raise KeyError(f"tensor {key} not found in the state dict")
        v = _np.asarray(e["vector"], dtype=_np.float64).ravel()
        v = v / _np.linalg.norm(v)
        cols = e.get("columns", {}).get("indices")
        W_new, n_changed = rcda_edit(sd[key], v, e["alpha"],
                                     columns=None if not cols else cols)
        sd[key] = W_new
        out.append({"layer": e["layer"], "matrix": e["matrix"], "key": key,
                    "alpha": e["alpha"], "n_columns": len(cols or []),
                    "n_weights_changed": int(n_changed), "status": "applied"})
    return out


# ------------------------------------------------------------------ CLI

def _cli():
    import argparse
    import numpy as _np
    import torch as _torch
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(
        description="RCDA weight edit (RMT-Calibrated Directional Ablation).")
    ap.add_argument("--rcda", action="store_true", help="run the RCDA weight editor")
    ap.add_argument("--model_dir", default="model",
                    help="model directory with model.safetensors (base weights)")
    ap.add_argument("--state_dict", default=None,
                    help="explicit safetensors path (default <model_dir>/model.safetensors)")
    ap.add_argument("--out_dir", default="edited_model",
                    help="output directory for the edited model + rcda_edit_report.json")
    ap.add_argument("--edits", default=None,
                    help="JSON file with edit specs [{layer, matrix, vector_path?}] "
                         "(vector_path = .pt/.npy with the unit direction)")
    ap.add_argument("--layer", type=int, action="append", default=[],
                    help="layer index (repeatable; pairs with --matrix/--vector)")
    ap.add_argument("--matrix", action="append", default=[],
                    choices=list(RCDA_MATRICES), help="matrix name (repeatable)")
    ap.add_argument("--vector", action="append", default=[],
                    help="path to the direction (.pt/.npy) (repeatable)")
    ap.add_argument("--alpha", default="auto",
                    help="'auto' (BBP-calibrated) or a fixed float in [0, 1]")
    ap.add_argument("--bbp_kappa", type=float, default=0.5)
    ap.add_argument("--columns", default="auto_tw",
                    help="'auto_tw' | 'auto_chi2' | 'topk=K' (e.g. topk=54)")
    ap.add_argument("--tw_tau", type=float, default=2.0)
    ap.add_argument("--full", action="store_true",
                    help="full rank-1 ablation W - alpha v(v^T W) instead of the "
                         "column-restricted edit")
    ap.add_argument("--apply_pack", default=None,
                    help="re-apply a previously saved rcda_edit_report.json / pack")
    args = ap.parse_args()

    sd_path = args.state_dict or _os.path.join(args.model_dir, "model.safetensors")
    _os.makedirs(args.out_dir, exist_ok=True)

    if args.apply_pack:
        edits, meta = load_edit_pack(args.apply_pack)
        sd = load_state_dict(sd_path)
        summary = apply_pack(sd, edits)
        meta_note = {"applied_from": args.apply_pack, **meta}
        save_file(sd, _os.path.join(args.out_dir, "model.safetensors"),
                  metadata={"format": "pt", "rcda": "1"})
    else:
        specs = []
        if args.edits:
            with open(args.edits, encoding="utf-8") as f:
                specs = _json.load(f)
            if isinstance(specs, dict):
                specs = specs.get("edits", [])
        if args.layer:
            assert len(args.layer) == len(args.matrix) == len(args.vector), \
                "--layer/--matrix/--vector must be given the same number of times"
            specs = specs + [{"layer": l, "matrix": m, "vector": vp}
                             for l, m, vp in zip(args.layer, args.matrix, args.vector)]
        vectors = {}
        plans = []
        for s in specs:
            vp = s.get("vector")
            if vp:
                if vp.endswith(".npy"):
                    vec = _np.load(vp)
                else:
                    vec = _torch.load(vp, weights_only=True).numpy()
                vectors[(int(s["layer"]), s["matrix"])] = _np.asarray(vec).ravel()
            plans.append({"layer": int(s["layer"]), "matrix": s["matrix"]})
        sd = load_state_dict(sd_path)

        col_mode, col_k = args.columns, 54
        if col_mode.startswith("topk="):
            col_mode, col_k = "topk", int(col_mode.split("=", 1)[1])
        elif col_mode.isdigit():
            col_mode, col_k = "topk", int(col_mode)

        alpha_mode = "fixed" if args.alpha.lower() not in ("auto", "auto_bbp") else "auto_bbp"
        alpha_fixed = 1.0 if alpha_mode == "auto_bbp" else float(args.alpha)

        reports = []
        tensors = {}
        for plan in plans:
            v = vectors.get((plan["layer"], plan["matrix"]))
            if v is None:
                raise KeyError(f"no vector for (layer={plan['layer']}, matrix={plan['matrix']})")
            report, W_new = resolve_edit(
                sd, {"layer": plan["layer"], "matrix": plan["matrix"],
                     "vector": v, "vector_norm_raw": float(_np.linalg.norm(v))},
                kappa=args.bbp_kappa, alpha_mode=alpha_mode, alpha_fixed=alpha_fixed,
                column_mode=col_mode, columns_k=col_k, tw_tau=args.tw_tau,
                full_ablation=args.full)
            report["vector"] = v
            reports.append(report)
            if W_new is not None:
                tensors[report["state_dict_key"]] = W_new

        for r in reports:  # strip the raw vector from the console copy
            r.setdefault("status", "applied")
        summary = [{k: r[k] for k in r if k != "vector"} for r in reports]
        meta_note = {"base_state_dict": sd_path, "kappa": args.bbp_kappa,
                     "alpha_mode": alpha_mode, "alpha_fixed": alpha_fixed,
                     "column_mode": col_mode, "columns_k": col_k,
                     "tw_tau": args.tw_tau, "full_ablation": args.full,
                     "summary": summary}

        # save the edited state dict (full copy of the base with edits applied)
        if tensors:
            for k, t in tensors.items():
                sd[k] = t
            save_file(sd, _os.path.join(args.out_dir, "model.safetensors"),
                      metadata={"format": "pt", "rcda": "1"})
        else:
            print("no edits applied — edited model.safetensors NOT written")

    # The report is written as a self-contained edit pack (vectors inside as
    # base64), so `--apply_pack <this file>` reproduces the edit anywhere.
    save_edit_pack(_os.path.join(args.out_dir, "rcda_edit_report.json"),
                   reports if not args.apply_pack else
                   [{"layer": s.get("layer"), "matrix": s.get("matrix"),
                     "status": s.get("status")} for s in summary],
                   meta=meta_note)

    # copy the small model files (config / tokenizer / chat template) next to the
    # edited weights so the out_dir is a loadable model directory
    base_dir = _os.path.dirname(sd_path) or "."
    for fname in sorted(_os.listdir(base_dir)):
        lp = _os.path.join(base_dir, fname)
        if (_os.path.isfile(lp) and not fname.endswith(".safetensors")
                and not fname.startswith("model.safetensors.part")
                and not fname.endswith(".sha256")):
            import shutil as _shutil
            _shutil.copy2(lp, _os.path.join(args.out_dir, fname))

    print(f"RCDA edit summary ({len(summary)} edits):")
    for s in summary:
        line = (f"  layer {s.get('layer')} {s.get('matrix')}: status={s.get('status')}")
        if s.get("status") == "applied" and s.get("alpha") is not None:
            sc = s.get("scan", {})
            line += (f" alpha={s.get('alpha'):.3f} columns={s.get('columns', {}).get('n_columns')}"
                     f" theta_v={sc.get('theta_v'):.3f} theta_c={sc.get('theta_c'):.3f}"
                     f" lambda_v/lambda_+={sc.get('lambda_over_edge'):.2f}"
                     f" weights={s.get('n_weights_changed')}"
                     f" tw={s.get('tw_control', {}).get('verdict')}")
        print(line)
    print(f"report -> {_os.path.join(args.out_dir, 'rcda_edit_report.json')}")
    if not args.apply_pack:
        print(f"edited model -> {_os.path.join(args.out_dir, 'model.safetensors')}")


if __name__ == "__main__":
    _cli()


# ===========================================================================
# RCDA plan, section 9 — RMT-regularization fallback (bulk shrinkage)
# ===========================================================================
# When no direction passes the RMT admission (or the directional ablation does
# not move the behaviour), the plan prescribes the alternative: do not subtract
# a direction, instead compress the bulk of the spectrum towards its
# Marchenko-Pastur prediction, keeping the top-r singular components intact:
#
#     W = U diag(s) V^T,   mu_i = s_i^2 / n
#     mu_i <- min(mu_i, lambda_+^(W) + tau * TW-scale)   for bulk i (i >= r)
#     W'  = U diag(sqrt(mu'_i * n)) V^T
#
# This tests the hypothesis "the defect is not one direction but an over-energetic
# noise bulk" and is validated by the same TW control (healthy top components).

def rcda_shrink(W, r=10, tau=2.0, extra_edge=0.0):
    """Bulk spectral shrinkage of W (plan, section 9). Returns (W', info).

    Only singular values outside the top-r are clamped to the shrunken MP edge
    lambda_+ + tau*TW-scale (raised by ``extra_edge``); the top-r carry the
    functional structure and stay bit-identical.
    """
    import numpy as _np
    import torch as _torch

    was_torch = isinstance(W, _torch.Tensor)
    W_np = (W.detach().cpu().to(_torch.float32).numpy() if was_torch
            else _np.asarray(W))
    W64 = W_np.astype(_np.float64)
    m, n = W64.shape
    q = m / n
    U, S, Vh = _np.linalg.svd(W64, full_matrices=False)
    mu = S ** 2 / n
    sigma2_w = float(_np.median(mu))
    from marchenko_pastur import mp_upper_edge, tw_scale
    edge = mp_upper_edge(sigma2_w, q) + tau * tw_scale(mp_upper_edge(sigma2_w, q), m) \
        + extra_edge
    r_eff = min(int(r), len(mu))
    mu_new = mu.copy()
    mu_new[r_eff:] = _np.minimum(mu_new[r_eff:], edge)
    n_clamped = int(_np.sum(mu_new < mu - 1e-12))
    W_new = (U * _np.sqrt(mu_new * n)) @ Vh
    info = {
        "mode": "shrink", "r": r_eff, "tau": float(tau), "edge": float(edge),
        "sigma2_w": sigma2_w, "lambda_plus_w": float(mp_upper_edge(sigma2_w, q)),
        "n_clamped": n_clamped,
        "energy_removed": float(1.0 - (mu_new.sum() / mu.sum())),
        "rel_frobenius_change": float(_np.linalg.norm(W_new - W64)
                                      / _np.linalg.norm(W64)),
    }
    Wt = _torch.from_numpy(W_new.astype(_np.float32))
    return (Wt if was_torch else W_new), info


def apply_pack_with_shrink(sd, edits, shrink_r=10, shrink_tau=2.0, shrink_layers=None):
    """Apply an edit pack; additionally bulk-shrink o_proj/down_proj of the given
    layers (plan section 9 fallback). Returns per-op summaries."""
    import numpy as _np

    out = apply_pack(sd, edits)
    for layer in (shrink_layers or []):
        for matrix in RCDA_MATRICES:
            key = state_dict_key(layer, matrix)
            if key not in sd:
                continue
            W_new, info = rcda_shrink(sd[key], r=shrink_r, tau=shrink_tau)
            info.update({"layer": int(layer), "matrix": matrix, "key": key,
                         "status": "shrunk"})
            sd[key] = W_new
            out.append(info)
    return out
