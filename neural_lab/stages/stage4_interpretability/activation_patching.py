"""
stage4_interpretability.activation_patching — where the answer is decided.

Causal patching (clean → corrupted): the corrupted prompt (e.g. a factual
entity swapped) is run while ONE decoder layer's output at the answer
position is replaced by the clean run's activation. The restoration of
the clean answer's probability measures how much layer l carries the
factual content.

Stage-4 question: is the restoration profile aligned with the spectral
gap profile (the BBP spike)? Spearman/Pearson between the per-layer
restoration and gap_ratio, with permutation p.

Everything runs offline on any causal LM (incl. tiny random GPT-2 for
tests); on real models the prompt pairs must be minimal pairs provided
by the user (or drawn from a facts file).
"""

from __future__ import annotations

import numpy as np

import torch


def _last_position_logits(model, ids: list[int]) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor([ids], device=next(model.parameters()).device))
    return out.logits[0, -1, :].detach().float().cpu().numpy()


def _prob_of(logits: np.ndarray, token_id: int) -> float:
    z = logits - logits.max()
    logp = z - np.log(np.exp(z).sum())
    return float(np.exp(logp[token_id]))


def patching_profile(
    model,
    clean_ids: list[int],
    corrupted_ids: list[int],
    answer_token_id: int,
) -> dict:
    """Per-layer restoration of P(clean answer | patched corrupted input).

    Intervention: the hook replaces layer l's output at the LAST position
    with the clean run's last-position activation. This is the standard
    'noise at last position' variant, NOT a full-sequence patch: for
    minimal pairs with a known answer offset, patch the answer position
    instead. When clean and corrupted prompts have different token
    lengths, the last positions are NOT semantically corresponding — the
    result carries a length_mismatch warning rather than failing silently.

    The clean activations are captured with the SAME hook mechanism used
    for patching — never via output_hidden_states (its last entry may be
    post-final-norm depending on the transformers version)."""
    device = next(model.parameters()).device
    from stages.common.hf_model import decoder_layers

    layers = decoder_layers(model)
    if len(clean_ids) != len(corrupted_ids):
        import warnings

        warnings.warn(
            f"patching_profile: clean ({len(clean_ids)}) and corrupted "
            f"({len(corrupted_ids)}) prompts differ in length; the last positions "
            "may not be semantically corresponding",
            stacklevel=2,
        )

    # capture the CLEAN block outputs once, in layer order
    clean_h: list = []

    def cap_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        clean_h.append(hidden.detach())

    handles = [l.register_forward_hook(cap_hook) for l in layers]
    try:
        with torch.no_grad():
            _ = model(torch.tensor([clean_ids], device=device))
    finally:
        for h in handles:
            h.remove()
    clean_hidden = clean_h  # block outputs in layer order

    base_corrupt = _last_position_logits(model, corrupted_ids)
    base_clean = _last_position_logits(model, clean_ids)
    p_corrupt = _prob_of(base_corrupt, answer_token_id)
    p_clean = _prob_of(base_clean, answer_token_id)

    restoration = []
    for i, layer in enumerate(layers):
        clean_i = clean_hidden[i]

        def patch_hook(module, args, output):
            # patch ONLY the last position (shape-safe for clean/corrupted
            # prompts of different lengths; the standard answer-position
            # intervention)
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1:, :] = clean_i[:, -1:, :].to(hidden.dtype)
                return (hidden,) + tuple(output[1:])
            hidden = output.clone()
            hidden[:, -1:, :] = clean_i[:, -1:, :].to(hidden.dtype)
            return hidden

        h = layer.register_forward_hook(patch_hook)
        try:
            with torch.no_grad():
                out = model(torch.tensor([corrupted_ids], device=device))
            logits = out.logits[0, -1, :].detach().float().cpu().numpy()
        finally:
            h.remove()
        p_patched = _prob_of(logits, answer_token_id)
        restoration.append(p_patched)

    return {
        "p_clean": p_clean,
        "p_corrupt": p_corrupt,
        "restoration_per_layer": restoration,  # P(answer | layer l patched from clean)
        "normalized": [float((p - p_corrupt) / max(p_clean - p_corrupt, 1e-9)) for p in restoration],
        "n_layers": len(layers),
        "length_mismatch": len(clean_ids) != len(corrupted_ids),
    }


def align_patching_with_spectrum(restoration: list[float], gap_profile: list[float]) -> dict:
    """Correlate restoration with the spectral gap profile."""
    a = np.asarray(restoration, dtype=np.float64)
    b = np.asarray(gap_profile, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {"pearson_r": float("nan"), "note": "degenerate profile"}
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(0)
    null = [float(np.corrcoef(a, rng.permutation(b))[0, 1]) for _ in range(1000)]
    # +1 correction: p=0 is unattainable, the floor is 1/(n_null + 1)
    p = (1.0 + float((np.abs(np.asarray(null)) >= abs(r) - 1e-12).sum())) / (1.0 + len(null))
    return {"pearson_r": r, "p_perm": float(p), "n_layers": int(len(a)),
            "argmax_restoration_layer": int(np.argmax(a)), "argmax_gap_layer": int(np.argmax(b))}
