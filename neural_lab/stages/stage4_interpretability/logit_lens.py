"""
stage4_interpretability.logit_lens — the lens trajectory of the answer.

Logit lens (nostalgebraist, 2020): apply the FINAL norm + unembedding to
INTERMEDIATE hidden states. For a target answer token we get the
per-layer probability trajectory p_l(answer); the model "commits" to the
answer where this probability saturates.

The Stage-4 question: does the lens saturation layer coincide with the
BBP spike (the layer where gap_ratio peaks)? If yes, the spectral
signature is not just a metric — it marks the layer where the factual
decision is made (interpretability, not only detection).

Also produces, for each layer, the top-1 decoded token (the "lens view"
of what the layer would say), a standard qualitative artifact.

Testable fully offline on a tiny random GPT-2 (the machinery does not
require a trained model; the QUESTION requires a real one).
"""

from __future__ import annotations

import numpy as np

from stages.common.hf_model import resolve_final_norm
from stages.stage2_detector.spectral_features import covariance_spectral_features


def lens_trajectory(model, tokenizer, prompt: str, target_text: str) -> dict:
    """Per-layer probability of the target token's FIRST id + top-1 token.

    The lens stack is [embeddings, block 0, ..., block L-1] — block outputs
    are captured by hooks (architecture-independent), the embeddings row is
    the input embedding of the last position. Each row goes through the
    FINAL norm + unembedding.
    """
    import torch

    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if not target_ids:
        raise ValueError("target_text encodes to no tokens")
    tid = int(target_ids[0])

    ln_f = resolve_final_norm(model)
    lm_head_weight = model.get_output_embeddings().weight

    from stages.common.hf_model import decoder_layers

    block_h: list = []

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        block_h.append(hidden[:, -1, :].detach())

    handles = [l.register_forward_hook(hook) for l in decoder_layers(model)]
    try:
        with torch.no_grad():
            emb = model.get_input_embeddings()(
                torch.tensor([ids], device=device))[:, -1, :].detach()
            model(torch.tensor([ids], device=device))
    finally:
        for h in handles:
            h.remove()

    rows = [emb[0]] + [b[0] for b in block_h]  # (L+1, H)
    with torch.no_grad():
        stacked = torch.stack(rows)  # (L+1, H)
        logits = ln_f(stacked) @ lm_head_weight.T
        probs = torch.softmax(logits.float(), dim=-1)
        traj = [float(p[tid].item()) for p in probs]
        top1 = [tokenizer.decode([int(p.argmax().item())]) for p in probs]

    return {
        "prompt": prompt,
        "target": target_text,
        "target_token_id": tid,
        "per_layer_prob": traj,  # length L+1: index 0 = embeddings
        "per_layer_top1": top1,
        "final_prob": traj[-1],
    }


def find_bbp_spike(final_hidden: list[np.ndarray]) -> dict:
    """Layer where gap_ratio peaks (the plan's 'BBP spike').

    Layers whose covariance spectrum is degenerate (None features) keep
    their slot as NaN, and spike_layer refers to the ORIGINAL layer index —
    argmax over a None-filtered list would point at the wrong layer."""
    feats = [covariance_spectral_features(np.asarray(h, dtype=np.float64), li) for li, h in enumerate(final_hidden)]
    valid = [(li, f) for li, f in enumerate(feats) if f is not None]
    if not valid:
        raise ValueError("no valid spectral features")
    gaps = [f["gap_ratio"] for _, f in valid]
    spike_layer = valid[int(np.argmax(gaps))][0]
    all_gaps = [float(f["gap_ratio"]) if f is not None else float("nan") for f in feats]
    all_effrank = [float(f["effective_rank"]) if f is not None else float("nan") for f in feats]
    return {
        "spike_layer": int(spike_layer),
        "gap_ratio_per_layer": all_gaps,
        "effrank_per_layer": all_effrank,
        "n_valid_layers": len(valid),
    }


def lens_spike_alignment(traj: list[float], gap_profile: list[float]) -> dict:
    """Correlate the lens-commitment profile with the spectral gap profile.

    Commitment proxy per layer: p_l(target) - p_{l-1}(target) (the layer's
    contribution to saturating the answer). Positive Pearson correlation
    between the commitment profile and gap_ratio profile supports 'the
    spike layer is where the answer is committed'. Permutation p-value on
    layer labels (with the +1 correction, never 0).

    traj has L+1 entries (embeddings + L layers) and gap_profile L; the
    embeddings row is dropped. Any other length combination raises —
    silently truncating a wrong-shaped input would corrupt the alignment.
    """
    if len(traj) == len(gap_profile) + 1:
        traj = traj[1:]  # drop the embeddings row
    elif len(traj) != len(gap_profile):
        raise ValueError(
            f"lens trajectory length {len(traj)} vs gap profile length {len(gap_profile)}: "
            "expected traj = gap + 1 (embeddings row) or equal lengths"
        )
    traj = np.asarray(traj, dtype=np.float64)
    gap = np.asarray(gap_profile, dtype=np.float64)
    commit = np.diff(traj, prepend=traj[0])
    if np.std(commit) < 1e-12 or np.std(gap) < 1e-12:
        return {"pearson_r": 0.0, "p_perm": 1.0, "note": "degenerate profile"}
    r = float(np.corrcoef(commit, gap)[0, 1])
    rng = np.random.default_rng(0)
    null = []
    for _ in range(1000):
        null.append(float(np.corrcoef(commit, rng.permutation(gap))[0, 1]))
    p = (1.0 + float((np.abs(np.asarray(null)) >= abs(r) - 1e-12).sum())) / (1.0 + len(null))
    return {"pearson_r": r, "p_perm": float(min(1.0, p)), "commitment_profile": commit.tolist(),
            "argmax_commit_layer": int(np.argmax(commit)), "argmax_gap_layer": int(np.argmax(gap))}
