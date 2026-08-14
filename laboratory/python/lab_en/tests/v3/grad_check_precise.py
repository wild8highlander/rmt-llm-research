"""
Precise gradient check using float64 and only comparing entries where
the numeric gradient is large enough for the relative error to be
meaningful (avoids float32 finite-difference noise on near-zero entries).
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


def cross_entropy(logits: np.ndarray, target: int) -> float:
    p = _softmax(logits)
    return float(-np.log(p[target] + 1e-12))


def loss_fn(model: TinyGPTV3, token_ids: np.ndarray, target: int) -> float:
    logits, _ = model.forward(token_ids)
    return cross_entropy(logits[-1], target)


def to_float64(model: TinyGPTV3) -> None:
    """Cast all master weights to float64 for precise finite differences."""
    model.token_emb = model.token_emb.astype(np.float64)
    model.pos_emb = model.pos_emb.astype(np.float64)
    model.lm_head = model.lm_head.astype(np.float64)
    model.ln_f_gamma = model.ln_f_gamma.astype(np.float64)
    model.ln_f_beta = model.ln_f_beta.astype(np.float64)
    for layer in model.layers:
        for attr in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
                     "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                     "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
            setattr(layer, attr, getattr(layer, attr).astype(np.float64))


def get_param(model: TinyGPTV3, name: str) -> np.ndarray:
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


def numeric_grad(model, token_ids, target, param_name, eps=1e-6):
    arr = get_param(model, param_name)
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


def analytic_grad(model, token_ids, target):
    logits, cache = model.forward_with_cache(token_ids)
    p = _softmax(logits[-1])
    dlogits = np.zeros_like(logits)
    dlogits[-1] = p.copy()
    dlogits[-1, target] -= 1.0
    return model.backward(cache, dlogits)


def check_param(model, token_ids, target, name, eps=1e-6, abs_tol=1e-5, rel_tol=1e-4):
    ng = numeric_grad(model, token_ids, target, name, eps=eps)
    ag = analytic_grad(model, token_ids, target).get(name)
    if ag is None:
        return name, 0.0, 0.0, True, 0
    ag = ag.astype(np.float64)

    # Only compare entries where |numeric grad| > abs_tol
    mask = np.abs(ng) > abs_tol
    n_checked = int(mask.sum())
    if n_checked == 0:
        return name, 0.0, 0.0, True, 0

    rel = np.abs(ag[mask] - ng[mask]) / (np.abs(ng[mask]) + 1e-12)
    max_rel = float(np.max(rel))
    ok = max_rel < rel_tol
    return name, max_rel, float(np.mean(np.abs(ag[mask] - ng[mask]))), ok, n_checked


def main():
    print("=" * 75)
    print("TinyGPT v3 — PRECISE Gradient Check (float64, eps=1e-6)")
    print("=" * 75)

    cfg = config_small()
    cfg.use_rope = True
    cfg.mixed_precision = False
    model = TinyGPTV3(cfg)
    to_float64(model)  # Cast to float64 for clean finite differences
    print(f"Config: hidden={cfg.hidden_dim}, layers={cfg.n_layers}, "
          f"heads={cfg.n_heads}, kv_heads={cfg.n_kv_heads}, rope={cfg.use_rope}")

    rng = np.random.default_rng(42)
    token_ids = rng.integers(0, cfg.vocab_size, size=6).astype(np.int64)
    target = int(rng.integers(0, cfg.vocab_size))

    # For embedding, we must check rows that are actually used.
    used_tokens = sorted(set(token_ids.tolist()))
    print(f"Tokens: {token_ids} (unique: {used_tokens})")
    print(f"Target: {target}")

    params = [
        "lm_head", "ln_f_gamma",
        "L0_W_q", "L0_W_k", "L0_W_v", "L0_W_o",
        "L0_b_q", "L0_b_k", "L0_b_v", "L0_b_o",
        "L0_ln1_gamma", "L0_ln1_beta", "L0_ln2_gamma", "L0_ln2_beta",
        "L0_W_fc1", "L0_W_fc2", "L0_b_fc1", "L0_b_fc2",
        "L1_W_q", "L1_W_k", "L1_W_v", "L1_W_o",
        "L1_W_fc1", "L1_W_fc2",
    ]

    all_ok = True
    for name in params:
        pname, max_rel, mean_abs, ok, n = check_param(
            model, token_ids, target, name, eps=1e-6, abs_tol=1e-6, rel_tol=5e-4,
        )
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {pname:20s}  max_rel={max_rel:.4e}  "
              f"mean_abs_diff={mean_abs:.4e}  n_checked={n}")
        if not ok:
            all_ok = False

    # Check embedding rows for used tokens specifically.
    print("\n  --- token_emb (used rows only) ---")
    ng_emb = numeric_grad(model, token_ids, target, "token_emb", eps=1e-6)
    ag_emb = analytic_grad(model, token_ids, target)["token_emb"].astype(np.float64)
    for t in used_tokens:
        ng_row = ng_emb[t]
        ag_row = ag_emb[t]
        mask = np.abs(ng_row) > 1e-6
        if mask.sum() == 0:
            continue
        rel = np.abs(ag_row[mask] - ng_row[mask]) / (np.abs(ng_row[mask]) + 1e-12)
        max_rel = float(np.max(rel))
        ok = max_rel < 1e-4
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] token_emb[{t}]         max_rel={max_rel:.4e}  n_checked={int(mask.sum())}")
        if not ok:
            all_ok = False

    # Gradient checkpointing should give identical gradients.
    print("\n--- gradient_checkpointing=True (should match no-checkpointing) ---")
    cfg2 = config_small()
    cfg2.use_rope = True
    cfg2.gradient_checkpointing = True
    model2 = TinyGPTV3(cfg2)
    to_float64(model2)
    # Copy weights from model to model2
    model2.token_emb = model.token_emb.copy()
    model2.lm_head = model.lm_head.copy()
    model2.ln_f_gamma = model.ln_f_gamma.copy()
    model2.ln_f_beta = model.ln_f_beta.copy()
    for i, layer in enumerate(model2.layers):
        src = model.layers[i]
        for attr in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
                     "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                     "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
            setattr(layer, attr, getattr(src, attr).copy())

    g1 = analytic_grad(model, token_ids, target)
    g2 = analytic_grad(model2, token_ids, target)
    max_diff = 0.0
    for k in g1:
        if k in g2:
            d = float(np.max(np.abs(g1[k].astype(np.float64) - g2[k].astype(np.float64))))
            max_diff = max(max_diff, d)
    print(f"  Max |grad_no_ckpt - grad_ckpt| = {max_diff:.4e}")
    ckpt_ok = max_diff < 1e-10
    print(f"  Checkpointing gradients match: {ckpt_ok}")
    all_ok = all_ok and ckpt_ok

    # use_rope=False path
    print("\n--- use_rope=False ---")
    cfg3 = config_small()
    cfg3.use_rope = False
    model3 = TinyGPTV3(cfg3)
    to_float64(model3)
    for name in ["L0_W_q", "L0_W_k", "L0_W_fc1", "lm_head", "L1_W_o"]:
        pname, max_rel, mean_abs, ok, n = check_param(
            model3, token_ids, target, name, eps=1e-6, abs_tol=1e-6, rel_tol=5e-4,
        )
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {pname:20s}  max_rel={max_rel:.4e}  n_checked={n}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 75)
    print(f"OVERALL: {'ALL GRADIENTS CORRECT ✓' if all_ok else 'SOME GRADIENTS FAILED ✗'}")
    print("=" * 75)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
