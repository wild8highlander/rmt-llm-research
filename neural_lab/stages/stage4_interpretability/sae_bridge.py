"""
stage4_interpretability.sae_bridge — sparse-autoencoder audit of the
BBP-spike window (Gemma Scope / Neuronpedia).

The Stage-4 question, concretely: do SPECIFIC SAE features fire when the
model enters the BBP-spike regime? Protocol:

  1. per prompt, compute the spectral gap_ratio profile and find the
     spike layer (find_bbp_spike);
  2. at the spike layer, collect SAE feature activations for
     (a) 'spike' prompts — items whose gap_ratio_last exceeded a
     pre-registered threshold (BBP-outlier regime) and
     (b) 'control' prompts — items below it;
  3. rank features by |median activation difference| with a permutation
     p-value; report the top-k with Neuronpedia links.

The SAE source is INJECTED (dependency-inversion): any object with
`activations(text) -> dict[int, float]` works. Two sources ship:

  GemmaScopeSource — sae_lens + Gemma Scope (needs gemma access + GPU,
    guarded import; NOT part of tests);
  MockSource — deterministic pseudo-features for tests.

Neuronpedia URLs follow https://neuronpedia.org/{model}/{sae_id}/{feature}.
Reference: Templeton et al. 2024 (Gemma Scope), Lieberum et al. 2024.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class SaeSource(Protocol):
    sae_id: str
    model_id: str

    def activations(self, text: str) -> dict[int, float]: ...


class MockSource:
    """Deterministic test double: feature k fires with intensity depending
    on the text hash; feature 7 is a planted 'spike feature' for prompts
    containing the token 'zzSPIKEzz'.

    The hash is sha256-based, NOT Python's hash(): string hashing is
    randomized per process (PYTHONHASHSEED), which would make the control
    features — and with them the test — differ between runs."""

    sae_id = "mock-sae/res-l0"
    model_id = "tiny-gpt2"

    def activations(self, text: str) -> dict[int, float]:
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        base = {k: ((h >> (k % 16)) & 15) / 10.0 for k in range(16)}
        if "zzSPIKEzz" in text:
            base[7] = 5.0 + (h % 7) / 10.0
        return base


class GemmaScopeSource:
    """Gemma Scope via sae_lens. Requires: pip install sae-lens, a Gemma
    checkpoint (gated on HF) and ideally a GPU. Raises with installation
    instructions if the dependency is missing.

    activations(text) tokenizes the text with the stored tokenizer, runs
    the stored model, captures the SAE's hook point (sae.cfg.hook_name)
    with a forward hook and encodes the LAST-POSITION hidden state —
    sae.encode consumes hidden-state tensors, not strings. Pass the
    stage-4 model/tokenizer in the constructor (or any HF causal LM the
    SAE was trained on). This path is exercised only with the real
    dependency present; tests cover the MockSource."""

    def __init__(self, release: str, sae_id: str, device: str = "cuda",
                 model=None, tokenizer=None):
        try:
            from sae_lens import SAE  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - optional path
            raise RuntimeError(
                "sae_lens is required for GemmaScopeSource: pip install sae-lens; "
                "then load e.g. release='gemma-scope-2b-pt-res-canonical', "
                "sae_id='layer_12/width_16k/average_l0_71'"
            ) from e
        loaded = SAE.from_pretrained(release, sae_id, device=device)
        self.sae = loaded[2] if isinstance(loaded, tuple) else loaded
        self.sae_id = sae_id
        self.model_id = release
        self.model = model
        self.tokenizer = tokenizer
        self.hook_name = getattr(getattr(self.sae, "cfg", None), "hook_name", None)

    def activations(self, text: str) -> dict[int, float]:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "GemmaScopeSource needs the base model and tokenizer passed to "
                "__init__ — the SAE encodes hidden states captured from a forward "
                "pass, not raw text"
            )
        device = next(self.sae.parameters()).device
        ids = self.tokenizer.encode(text, return_tensors="pt").to(device)
        captured: dict = {}

        def hook(module, args, output):
            captured["h"] = output[0] if isinstance(output, tuple) else output

        if not self.hook_name:
            raise RuntimeError("sae.cfg.hook_name is missing; cannot resolve the capture point")
        handle = self.model.get_submodule(self.hook_name).register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            handle.remove()
        acts = captured["h"][:, -1, :]  # last position, matching MockSource pooling
        with torch.no_grad():
            feats = self.sae.encode(acts)  # (1, d_sae)
        f = feats.detach().float().cpu().numpy().ravel()
        idx = np.nonzero(f)[0]
        return {int(i): float(f[i]) for i in idx}


def neuronpedia_url(model_id: str, sae_id: str, feature_index: int) -> str:
    return f"https://www.neuronpedia.org/{model_id}/{sae_id}/{feature_index}"


def bbp_feature_audit(
    source: SaeSource,
    spike_texts: list[str],
    control_texts: list[str],
    top_k: int = 10,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Top-k SAE features separating spike vs control prompts.

    Statistic per feature: median(spike) - median(control); permutation p
    by shuffling group labels. Requires >= 3 items per group.
    """
    if len(spike_texts) < 3 or len(control_texts) < 3:
        raise ValueError("need >= 3 prompts per group")
    def collect(texts):
        per_text = [source.activations(t) for t in texts]
        all_idx = sorted({i for d in per_text for i in d})
        return np.stack([[d.get(i, 0.0) for i in all_idx] for d in per_text]), all_idx

    A, idx_a = collect(spike_texts)
    B, idx_b = collect(control_texts)
    if idx_a != idx_b:
        # union alignment (MockSource and real SAEs both return stable dims,
        # but be safe)
        common = sorted(set(idx_a) & set(idx_b))
        ca = [idx_a.index(i) for i in common]
        cb = [idx_b.index(i) for i in common]
        A, B = A[:, ca], B[:, cb]
        all_idx = common
    else:
        all_idx = idx_a
    diff = np.median(A, axis=0) - np.median(B, axis=0)
    rng = np.random.default_rng(seed)
    pooled = np.vstack([A, B])
    n_a = len(A)
    null = np.empty((n_perm, pooled.shape[1]))
    for p in range(n_perm):
        perm = rng.permutation(len(pooled))
        null[p] = np.median(pooled[perm[:n_a]], axis=0) - np.median(pooled[perm[n_a:]], axis=0)
    # +1 correction per feature: the smallest attainable p is 1/(n_perm + 1)
    p_vals = (1.0 + (np.abs(null) >= np.abs(diff) - 1e-12).sum(axis=0)) / (1.0 + n_perm)
    order = np.argsort(-np.abs(diff))[:top_k]
    feats = []
    for i in order:
        if abs(diff[i]) < 1e-9:
            continue
        feats.append({
            "feature": int(all_idx[i]),
            "delta_median": float(diff[i]),
            "p_perm": float(p_vals[i]),
            "neuronpedia": neuronpedia_url(source.model_id, source.sae_id, int(all_idx[i])),
        })
    return {
        "sae_id": source.sae_id,
        "model_id": source.model_id,
        "n_spike": len(spike_texts),
        "n_control": len(control_texts),
        "features": feats,
        "note": "features ranked by |median delta|; p by label permutation",
    }
