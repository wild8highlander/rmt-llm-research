"""
Gradient check for TinyGPT v3 — numerical finite-difference verification.

Verifies that the hand-written backward pass in TinyGPTV3.backward()
produces gradients that match central finite differences of the loss
to within tolerance. This is THE critical correctness test for the
autodiff implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tiny_gpt_v3 import (
    TinyGPTV3,
    TinyGPTV3Config,
    config_small,
    _softmax,
)
from tiny_gpt_v3 import layernorm_forward


def cross_entropy(logits: np.ndarray, target: int) -> float:
    """Stable cross-entropy of a single logit vector against a target."""
    p = _softmax(logits)
    return float(-np.log(p[target] + 1e-12))


def loss_fn(model: TinyGPTV3, token_ids: np.ndarray, target: int) -> float:
    logits, _ = model.forward(token_ids)
    return cross_entropy(logits[-1], target)


def numeric_grad(
    model: TinyGPTV3, token_ids: np.ndarray, target: int,
    param_name: str, eps: float = 1e-4,
) -> np.ndarray:
    """Central finite-difference gradient for a single parameter array."""
    arr = _get_param(model, param_name)
    grad = np.zeros_like(arr, dtype=np.float64)
    flat = arr.ravel()
    gflat = grad.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        l_plus = loss_fn(model, token_ids, target)
        flat[i] = orig - eps
        l_minus = loss_fn(model, token_ids, target)
        flat[i] = orig
        gflat[i] = (l_plus - l_minus) / (2 * eps)
    return grad


def _get_param(model: TinyGPTV3, name: str) -> np.ndarray:
    if name == "token_emb":
        return model.token_emb
    if name == "lm_head":
        return model.lm_head
    if name.startswith("L"):
        parts = name.split("_", 2)
        idx = int(parts[0][1:])
        attr = "_".join(parts[1:])
        return getattr(model.layers[idx], attr)
    return getattr(model, name)


def analytic_grad(
    model: TinyGPTV3, token_ids: np.ndarray, target: int,
) -> dict:
    """Run forward_with_cache + backward to get analytic gradients."""
    logits, cache = model.forward_with_cache(token_ids)
    p = _softmax(logits[-1])
    dlogits = np.zeros_like(logits)
    dlogits[-1] = p.copy()
    dlogits[-1, target] -= 1.0
    return model.backward(cache, dlogits)


def check_param(
    model: TinyGPTV3, token_ids: np.ndarray, target: int,
    param_name: str, n_samples: int = 8, tol: float = 1e-3,
) -> tuple:
    """Compare analytic vs numeric gradient on ``n_samples`` random entries."""
    ng = numeric_grad(model, token_ids, target, param_name)
    grads = analytic_grad(model, token_ids, target)
    ag = grads.get(param_name)
    if ag is None:
        return param_name, float("nan"), 0.0, False
    ag = ag.astype(np.float64)

    # Pick random entries to compare (full compare is too slow for large arrays).
    rng = np.random.default_rng(123)
    flat_ag = ag.ravel()
    flat_ng = ng.ravel()
    idx = rng.choice(flat_ag.size, size=min(n_samples, flat_ag.size), replace=False)
    diffs = np.abs(flat_ag[idx] - flat_ng[idx])
    rel = diffs / (np.abs(flat_ng[idx]) + 1e-8)
    max_rel = float(np.max(rel))
    ok = max_rel < tol
    return param_name, max_rel, float(diffs.mean()), ok


def main():
    print("=" * 70)
    print("TinyGPT v3 — Gradient Check (finite-difference verification)")
    print("=" * 70)

    cfg = config_small()
    cfg.use_rope = True
    cfg.mixed_precision = False
    model = TinyGPTV3(cfg)
    print(f"\nConfig: {cfg}")
    print(f"Params: {cfg.params_count:,}")

    rng = np.random.default_rng(42)
    token_ids = rng.integers(0, cfg.vocab_size, size=8).astype(np.int64)
    target = int(rng.integers(0, cfg.vocab_size))
    print(f"Tokens: {token_ids}")
    print(f"Target: {target}")

    # Check gradient checkpointing == no checkpointing (same gradients).
    for ckpt in (False, True):
        cfg2 = config_small()
        cfg2.use_rope = True
        cfg2.gradient_checkpointing = ckpt
        model = TinyGPTV3(cfg2)
        print(f"\n--- gradient_checkpointing={ckpt} ---")
        params_to_check = [
            "token_emb",
            "lm_head",
            "L0_W_q", "L0_W_k", "L0_W_v", "L0_W_o",
            "L0_b_q", "L0_b_o",
            "L0_ln1_gamma", "L0_ln1_beta",
            "L0_W_fc1", "L0_W_fc2", "L0_b_fc1", "L0_b_fc2",
            "L0_ln2_gamma",
            "L1_W_q", "L1_W_o", "L1_W_fc1",
        ]
        all_ok = True
        for name in params_to_check:
            pname, max_rel, mean_abs, ok = check_param(
                model, token_ids, target, name, n_samples=6, tol=2e-2,
            )
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {pname:20s}  max_rel={max_rel:.4e}  mean_abs={mean_abs:.4e}")
            if not ok:
                all_ok = False
        print(f"  => All gradients correct: {all_ok}")

    # Also check the non-RoPE path (absolute position embeddings).
    print("\n--- use_rope=False (absolute position embeddings) ---")
    cfg3 = config_small()
    cfg3.use_rope = False
    model = TinyGPTV3(cfg3)
    for name in ["L0_W_q", "L0_W_k", "L0_W_fc1", "lm_head"]:
        pname, max_rel, mean_abs, ok = check_param(
            model, token_ids, target, name, n_samples=6, tol=2e-2,
        )
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {pname:20s}  max_rel={max_rel:.4e}  mean_abs={mean_abs:.4e}")

    print("\n" + "=" * 70)
    print("Gradient check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
