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
