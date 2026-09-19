"""
tiny_gpt_v3.py — TinyGPT v3: Modernized Transformer for RMT-LLM Laboratory
==========================================================================

Pure-NumPy implementation of a small transformer that modernizes the v2
architecture (in ``tiny_gpt.py``) with four production-grade techniques
from the Llama / Mistral / GPT-NeoX family:

1. **RoPE** (Rotary Position Embeddings) — replaces absolute position
   embeddings with rotation matrices applied to Q and K. Gives smooth
   length generalization (Su et al., 2021).
2. **GQA** (Grouped-Query Attention) — ``n_kv_heads < n_heads`` so K and
   V are shared across query groups. Cuts KV-cache and weight count
   without quality loss (Ainslie et al., 2023).
3. **Mixed-precision** — optional ``float16`` forward pass with
   ``float32`` master weights. ~2× speedup on CPU SIMD, larger on GPU.
4. **Gradient checkpointing** — re-run the forward pass during backward
   to avoid storing per-layer activations. Trades ~30% compute for up to
   ``n_layers×`` memory reduction (Chen et al., 2016).

Architecture (per layer, pre-LN, as in ADR-002)::

    h_in -> LN1 -> MHA(Q,K,V,O with RoPE + GQA) -> +residual
         -> LN2 -> MLP(4H, GELU)                  -> +residual -> h_out

No external dependencies beyond ``numpy`` (ADR-001). The module exposes:

- ``TinyGPTV3Config``   — dataclass with all knobs
- ``TinyGPTV3``         — the model (forward, generate, spectral_analysis)
- ``rope_apply``        — rotary embedding application (forward + backward)
- ``gqa_attention``     — grouped-query attention (forward + backward)
- ``MixedPrecisionCtx`` — context manager for float16 forward

Author: Iskhak Hamzatovich Isaev
ORCID:  0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Softmax (numerically stable; shared with v2)
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``.

    Args:
        x: Input array.
        axis: Reduction axis.

    Returns:
        Probabilities summing to 1 along ``axis``.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# GELU (tanh approximation, same as v2)
# ---------------------------------------------------------------------------
_GELU_C = math.sqrt(2.0 / math.pi)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Tanh-approximate GELU activation (GPT-2 / BERT style)."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximate GELU."""
    inner = _GELU_C * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    diner = _GELU_C * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * diner


# ---------------------------------------------------------------------------
# LayerNorm (forward + backward; same math as v2)
# ---------------------------------------------------------------------------
def layernorm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LayerNorm over the last dimension.

    Args:
        x: Input of shape ``(..., D)``.
        gamma: Scale parameter of shape ``(D,)``.
        beta: Shift parameter of shape ``(D,)``.
        eps: Numerical stability constant.

    Returns:
        Tuple ``(out, mean, rstd)`` where ``out`` has the same shape and
        dtype as ``x`` and ``mean``/``rstd`` have shape ``(..., 1)``.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xn = (x - mu) * rstd
    out = gamma * xn + beta
    return out.astype(x.dtype), mu, rstd


def layernorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    mu: np.ndarray,
    rstd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for LayerNorm.

    Args:
        dout: Upstream gradient, same shape as ``x``.
        x: Original input.
        gamma: Scale parameter.
        mu: Mean cached from forward.
        rstd: Reciprocal std-dev cached from forward.

    Returns:
        Tuple ``(dx, dgamma, dbeta)``.
    """
    D = x.shape[-1]
    xn = (x - mu) * rstd
    dgamma = np.sum(dout * xn, axis=tuple(range(dout.ndim - 1)))
    dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
    dxn = dout * gamma
    sum_dxn = np.sum(dxn, axis=-1, keepdims=True)
    sum_dxn_xn = np.sum(dxn * xn, axis=-1, keepdims=True)
    dx = (rstd / D) * (D * dxn - sum_dxn - xn * sum_dxn_xn)
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
def rope_freqs(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """Compute inverse frequencies for RoPE.

    The standard formula (Su et al., 2021) pairs dimensions
    ``(2i, 2i+1)`` for ``i = 0..head_dim/2-1`` with frequency
    ``theta_i = base ** (-2i / head_dim)``.

    Args:
        head_dim: Dimension per attention head. Must be even.
        base: Frequency base (10000 is the Llama / GPT-NeoX default).

    Returns:
        Array of shape ``(head_dim / 2,)`` with inverse frequencies.

    Raises:
        ValueError: If ``head_dim`` is not even.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return (base ** (-i / head_dim)).astype(np.float32)


def rope_apply(
    x: np.ndarray,
    seq_offset: int = 0,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE to a head tensor.

    Args:
        x: Tensor of shape ``(T, n_heads, head_dim)``.
        seq_offset: Position offset (for KV-cache). Positions are
            ``seq_offset, seq_offset+1, ..., seq_offset+T-1``.
        base: Frequency base.

    Returns:
        Tuple ``(x_rotated, cos_sin)`` where ``x_rotated`` has the same
        shape as ``x`` and ``cos_sin`` is a tuple ``(cos, sin)`` of
        arrays with shape ``(T, head_dim / 2)`` cached for backward.
    """
    T, n_heads, head_dim = x.shape
    freqs = rope_freqs(head_dim, base)  # (head_dim/2,)
    pos = np.arange(seq_offset, seq_offset + T, dtype=np.float64)  # (T,)
    angles = np.outer(pos, freqs).astype(np.float32)  # (T, head_dim/2)
    cos = np.cos(angles)
    sin = np.sin(angles)

    # Pair (x_{2i}, x_{2i+1}) and rotate by (cos, sin).
    x_even = x[..., 0::2]  # (T, n_heads, head_dim/2)
    x_odd = x[..., 1::2]  # (T, n_heads, head_dim/2)

    # Broadcast cos/sin to (T, 1, head_dim/2) for per-head broadcast.
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]

    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd = x_even * sin_b + x_odd * cos_b

    out = np.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out, (cos, sin)


def rope_backward(
    dout: np.ndarray,
    cos_sin: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Backward pass for RoPE.

    The rotation matrix ``[[c, -s], [s, c]]`` is orthogonal, so its
    transpose (which is the inverse) is ``[[c, s], [-s, c]]``. The
    backward pass therefore applies the inverse rotation to ``dout``.

    Args:
        dout: Upstream gradient, shape ``(T, n_heads, head_dim)``.
        cos_sin: Cached ``(cos, sin)`` from the forward pass.

    Returns:
        Gradient w.r.t. the input ``x``, same shape as ``dout``.
    """
    cos, sin = cos_sin
    cos_b = cos[:, None, :]
    sin_b = sin[:, None, :]

    dout_even = dout[..., 0::2]
    dout_odd = dout[..., 1::2]

    dx_even = dout_even * cos_b + dout_odd * sin_b
    dx_odd = -dout_even * sin_b + dout_odd * cos_b

    dx = np.empty_like(dout)
    dx[..., 0::2] = dx_even
    dx[..., 1::2] = dx_odd
    return dx


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------
def gqa_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rope: bool = True,
    seq_offset: int = 0,
    rope_base: float = 10000.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Forward pass for grouped-query attention with optional RoPE.

    Q has shape ``(T, n_heads, head_dim)``. K, V have shape
    ``(T, n_kv_heads, head_dim)`` where ``n_kv_heads`` divides
    ``n_heads``. Each KV head is shared by ``n_heads // n_kv_heads``
    query heads (the "group size").

    Args:
        q: Queries, shape ``(T, n_heads, head_dim)``.
        k: Keys, shape ``(T, n_kv_heads, head_dim)``.
        v: Values, shape ``(T, n_kv_heads, head_dim)``.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (must divide ``n_heads``).
        head_dim: Dimension per head.
        rope: If ``True``, apply RoPE to Q and K.
        seq_offset: Position offset for RoPE (KV-cache support).
        rope_base: RoPE frequency base.

    Returns:
        Tuple ``(out, cache)`` where ``out`` has shape
        ``(T, n_heads, head_dim)`` and ``cache`` stores intermediate
        values for the backward pass.

    Raises:
        ValueError: If ``n_heads % n_kv_heads != 0``.
    """
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    T = q.shape[0]
    group = n_heads // n_kv_heads

    # Apply RoPE to Q and K (not V).
    if rope:
        q, q_cs = rope_apply(q, seq_offset, rope_base)
        k, k_cs = rope_apply(k, seq_offset, rope_base)
    else:
        q_cs = k_cs = None

    # Repeat K and V across the group so each Q head has a matching KV.
    # Shape: (T, n_kv_heads, head_dim) -> (T, n_heads, head_dim)
    k_rep = np.repeat(k, group, axis=1)
    v_rep = np.repeat(v, group, axis=1)

    # Reshape to (n_heads, T, head_dim) for batched matmul.
    qh = q.transpose(1, 0, 2)  # (n_heads, T, head_dim)
    kh = k_rep.transpose(1, 0, 2)  # (n_heads, T, head_dim)
    vh = v_rep.transpose(1, 0, 2)  # (n_heads, T, head_dim)

    scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(head_dim)  # (n_heads, T, T)
    # Use a mask value that is safe for both float16 and float32.
    # float16 max is ~65504; use -1e4 to avoid overflow in exp().
    mask_val = -1e4 if scores.dtype == np.float16 else -1e9
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, mask_val, scores)
    attn = _softmax(scores, axis=-1)  # (n_heads, T, T)
    ctx = attn @ vh  # (n_heads, T, head_dim)
    out = ctx.transpose(1, 0, 2)  # (T, n_heads, head_dim)

    cache = {
        "q_cs": q_cs,
        "k_cs": k_cs,
        "attn": attn,
        "qh": qh,
        "kh": kh,
        "vh": vh,
        "group": group,
        "k": k,
        "v": v,
        "rope": rope,
    }
    return out, cache


def gqa_backward(
    dout: np.ndarray,
    cache: dict[str, Any],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for grouped-query attention.

    Args:
        dout: Upstream gradient, shape ``(T, n_heads, head_dim)``.
        cache: Cache from ``gqa_forward``.
        n_heads: Number of query heads.
        n_kv_heads: Number of KV heads.
        head_dim: Dimension per head.

    Returns:
        Tuple ``(dq, dk, dv)`` with shapes
        ``(T, n_heads, head_dim)``, ``(T, n_kv_heads, head_dim)``,
        ``(T, n_kv_heads, head_dim)``.
    """
    T = dout.shape[0]
    group = n_heads // n_kv_heads
    attn = cache["attn"]
    qh = cache["qh"]
    kh = cache["kh"]
    vh = cache["vh"]

    dout_h = dout.transpose(1, 0, 2)  # (n_heads, T, head_dim)

    # ctx = attn @ vh  ->  dattn = dout_h @ vh^T,  dvh = attn^T @ dout_h
    dattn = dout_h @ vh.transpose(0, 2, 1)  # (n_heads, T, T)
    dvh = attn.transpose(0, 2, 1) @ dout_h  # (n_heads, T, head_dim)

    # Softmax backward: dscores = attn * (dattn - sum(dattn * attn, axis=-1, keepdims))
    dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
    dscores = dscores / math.sqrt(head_dim)

    # scores = qh @ kh^T  ->  dqh = dscores @ kh,  dkh = dscores^T @ qh
    dqh = dscores @ kh  # (n_heads, T, head_dim)
    dkh = dscores.transpose(0, 2, 1) @ qh  # (n_heads, T, head_dim)

    # Back to (T, n_heads, head_dim).
    dq = dqh.transpose(1, 0, 2)
    dk_rep = dkh.transpose(1, 0, 2)
    dv_rep = dvh.transpose(1, 0, 2)

    # Sum gradients across the group to recover per-KV-head gradients.
    # k_rep = np.repeat(k, group, axis=1)  ->  dk = sum over each group.
    dk = dk_rep.reshape(T, n_kv_heads, group, head_dim).sum(axis=2)
    dv = dv_rep.reshape(T, n_kv_heads, group, head_dim).sum(axis=2)

    # RoPE backward (rotation is orthogonal, so backward = inverse rotation).
    if cache["rope"]:
        dq = rope_backward(dq, cache["q_cs"])
        dk = rope_backward(dk, cache["k_cs"])

    return dq, dk, dv


# ---------------------------------------------------------------------------
# Mixed-precision context manager
# ---------------------------------------------------------------------------
@dataclass
class MixedPrecisionCtx:
    """State for mixed-precision training.

    The model keeps ``float32`` master weights. During the forward pass
    we cast the active weights to ``float16`` (or ``bfloat16`` if the
    platform supports it) to halve memory bandwidth. Gradients are
    accumulated in ``float32`` and the optimizer updates the master
    weights.

    Attributes:
        enabled: Whether mixed-precision is active.
        compute_dtype: Dtype used for the forward pass.
        master_dtype: Dtype used for master weights (always float32).
    """

    enabled: bool = False
    compute_dtype: np.dtype = np.float16
    master_dtype: np.dtype = np.float32

    def cast(self, x: np.ndarray) -> np.ndarray:
        """Cast a master-weight array to the compute dtype if enabled."""
        if not self.enabled:
            return x
        return x.astype(self.compute_dtype)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
@dataclass
class TinyGPTV3Config:
    """Configuration for TinyGPT v3.

    Defaults target ~4M parameters (``vocab=512, hidden=192, layers=16,
    heads=6, kv_heads=2``) which is the "TinyGPT 4M" target from the
    roadmap.

    Attributes:
        vocab_size: Tokenizer vocabulary size.
        hidden_dim: Hidden dimension ``H`` (must be divisible by
            ``n_heads``).
        n_layers: Number of transformer blocks.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads. Must divide ``n_heads``.
            Set equal to ``n_heads`` for standard MHA, to 1 for MQA,
            or in between for GQA.
        max_seq_len: Maximum sequence length.
        mlp_ratio: MLP hidden = ``mlp_ratio * hidden_dim``.
        use_layernorm: Whether to use LayerNorm (ADR-002).
        use_mlp: Whether to use the MLP block.
        activation: ``"gelu"`` or ``"relu"``.
        use_rope: If ``True``, use RoPE instead of absolute position
            embeddings.
        rope_base: RoPE frequency base.
        mixed_precision: If ``True``, cast weights to ``float16`` for
            the forward pass.
        gradient_checkpointing: If ``True``, recompute layer activations
            during backward instead of storing them.
        seed: RNG seed for reproducible initialization.
    """

    vocab_size: int = 512
    hidden_dim: int = 192
    n_layers: int = 16
    n_heads: int = 6
    n_kv_heads: int = 2
    max_seq_len: int = 256
    mlp_ratio: int = 4
    use_layernorm: bool = True
    use_mlp: bool = True
    activation: str = "gelu"
    use_rope: bool = True
    rope_base: float = 10000.0
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    # --- v3.1 additions (Session 1) ---
    weight_tying: bool = True
    """If True, share the token embedding matrix with the LM head.

    This is the GPT-2 / Llama / Mistral convention. Saves
    ``hidden_dim * vocab_size`` parameters and gives a stronger
    gradient signal to the embeddings.
    """
    label_smoothing: float = 0.0
    """Label smoothing factor in [0, 1). 0 disables (hard targets).

    0.1 is the GPT-2 / Transformer-XL default. Prevents overconfidence
    and improves calibration.
    """
    grad_clip_norm: float = 0.0
    """Max gradient norm for global gradient clipping. 0 disables.

    1.0 is a common value (GPT-2, BERT). Stabilizes training and
    allows a larger learning rate.
    """
    # --- v3.2 additions (Session 2) ---
    dropout: float = 0.0
    """Dropout probability (applied to attention output and MLP output).

    0 disables. 0.1 is the GPT-2 default. Only active when
    ``model.training = True``.
    """
    init_scale: float = 0.02
    """Base standard deviation for weight initialization.

    GPT-2 uses 0.02. For deeper models, the residual projections are
    scaled by ``init_scale / sqrt(2 * n_layers)`` to keep the residual
    stream variance stable.
    """
    seed: int = 42

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @property
    def mlp_dim(self) -> int:
        return self.hidden_dim * self.mlp_ratio

    @property
    def n_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def params_count(self) -> int:
        """Total parameter count (floats stored in master weights)."""
        H = self.hidden_dim
        V = self.vocab_size
        Hkv = self.n_kv_heads
        hd = self.head_dim
        mlp_dim = self.mlp_dim

        # Embeddings: token only (RoPE removes position embedding).
        emb = V * H
        if not self.use_rope:
            emb += self.max_seq_len * H

        # Per layer:
        #   Q: H * (n_heads * head_dim) = H * H
        #   K, V: H * (n_kv_heads * head_dim) = H * (Hkv * hd)
        #   O: H * H
        #   4 biases
        kv_proj = Hkv * hd
        per_layer = H * H + 2 * H * kv_proj + H * H + 4 * H
        if self.use_layernorm:
            per_layer += 4 * H  # 2 LN blocks (gamma + beta)
        if self.use_mlp:
            per_layer += 2 * H * mlp_dim + mlp_dim + H

        # LM head: if weight_tying, the head reuses token_emb (no extra params).
        head = 0 if self.weight_tying else H * V  # LM head
        return emb + self.n_layers * per_layer + head


# ---------------------------------------------------------------------------
# Per-layer weights
# ---------------------------------------------------------------------------
@dataclass
class TinyLayerV3:
    """Weights for a single transformer block (v3).

    With GQA, ``W_k`` and ``W_v`` project to ``n_kv_heads * head_dim``
    rather than ``hidden_dim``. ``W_q`` and ``W_o`` always project to
    / from ``hidden_dim``.
    """

    # Attention projections
    W_q: np.ndarray  # (H, H)
    W_k: np.ndarray  # (H, n_kv_heads * head_dim)
    W_v: np.ndarray  # (H, n_kv_heads * head_dim)
    W_o: np.ndarray  # (H, H)
    b_q: np.ndarray  # (H,)
    b_k: np.ndarray  # (n_kv_heads * head_dim,)
    b_v: np.ndarray  # (n_kv_heads * head_dim,)
    b_o: np.ndarray  # (H,)
    # Pre-LN1 (before attention)
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    # Pre-LN2 (before MLP)
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    # MLP: H -> 4H -> H
    W_fc1: np.ndarray
    W_fc2: np.ndarray
    b_fc1: np.ndarray
    b_fc2: np.ndarray


def _init_layer_v3(
    rng: np.random.Generator,
    H: int,
    kv_dim: int,
    mlp_dim: int,
    use_layernorm: bool,
    use_mlp: bool,
) -> TinyLayerV3:
    """Initialize a v3 layer with Xavier-scaled weights."""
    scale = 1.0 / math.sqrt(H)
    return TinyLayerV3(
        W_q=rng.normal(0, scale, (H, H)).astype(np.float32),
        W_k=rng.normal(0, scale, (H, kv_dim)).astype(np.float32),
        W_v=rng.normal(0, scale, (H, kv_dim)).astype(np.float32),
        W_o=rng.normal(0, scale, (H, H)).astype(np.float32),
        b_q=np.zeros(H, dtype=np.float32),
        b_k=np.zeros(kv_dim, dtype=np.float32),
        b_v=np.zeros(kv_dim, dtype=np.float32),
        b_o=np.zeros(H, dtype=np.float32),
        ln1_gamma=np.ones(H, dtype=np.float32),
        ln1_beta=np.zeros(H, dtype=np.float32),
        ln2_gamma=np.ones(H, dtype=np.float32),
        ln2_beta=np.zeros(H, dtype=np.float32),
        W_fc1=rng.normal(0, scale, (H, mlp_dim)).astype(np.float32)
        if use_mlp
        else np.zeros((H, mlp_dim), dtype=np.float32),
        W_fc2=rng.normal(0, scale, (mlp_dim, H)).astype(np.float32)
        if use_mlp
        else np.zeros((mlp_dim, H), dtype=np.float32),
        b_fc1=np.zeros(mlp_dim, dtype=np.float32),
        b_fc2=np.zeros(H, dtype=np.float32),
    )


def _init_layer_v3_gpt2(
    rng: np.random.Generator,
    H: int,
    kv_dim: int,
    mlp_dim: int,
    use_layernorm: bool,
    use_mlp: bool,
    n_layers: int,
    init_scale: float = 0.02,
) -> TinyLayerV3:
    """Initialize a v3 layer with GPT-2-style scaled initialization.

    The residual-path projections (W_q, W_v, W_fc2) are scaled by
    ``init_scale / sqrt(2 * n_layers)`` so that the variance of the
    residual stream stays bounded as the model gets deeper. The factor
    of 2 accounts for the two residual paths per layer (attention + MLP).

    Args:
        rng: Random number generator.
        H: Hidden dimension.
        kv_dim: KV projection dimension (``n_kv_heads * head_dim``).
        mlp_dim: MLP hidden dimension.
        use_layernorm: Whether the layer uses LayerNorm.
        use_mlp: Whether the layer has an MLP block.
        n_layers: Total number of layers (for the scaling factor).
        init_scale: Base standard deviation (GPT-2 uses 0.02).

    Returns:
        Initialized :class:`TinyLayerV3`.
    """
    # Standard scale for non-residual projections.
    base_scale = init_scale
    # Scaled-down std for residual projections (prevents variance growth).
    residual_scale = init_scale / math.sqrt(2.0 * max(n_layers, 1))
    # MLP uses a smaller scale because the fan-out is 4x.
    mlp_scale = init_scale / math.sqrt(H)
    mlp_residual_scale = residual_scale / math.sqrt(H) * math.sqrt(H)  # = residual_scale
    return TinyLayerV3(
        # Q is on the residual path (its output flows back to x via W_o).
        W_q=rng.normal(0, residual_scale, (H, H)).astype(np.float32),
        # K, V are not on the residual path (they only feed attention).
        W_k=rng.normal(0, base_scale, (H, kv_dim)).astype(np.float32),
        W_v=rng.normal(0, residual_scale, (H, kv_dim)).astype(np.float32),
        # W_o is on the residual path.
        W_o=rng.normal(0, residual_scale, (H, H)).astype(np.float32),
        b_q=np.zeros(H, dtype=np.float32),
        b_k=np.zeros(kv_dim, dtype=np.float32),
        b_v=np.zeros(kv_dim, dtype=np.float32),
        b_o=np.zeros(H, dtype=np.float32),
        ln1_gamma=np.ones(H, dtype=np.float32),
        ln1_beta=np.zeros(H, dtype=np.float32),
        ln2_gamma=np.ones(H, dtype=np.float32),
        ln2_beta=np.zeros(H, dtype=np.float32),
        # W_fc1 is not on the residual path (input to MLP).
        W_fc1=rng.normal(0, mlp_scale, (H, mlp_dim)).astype(np.float32)
        if use_mlp
        else np.zeros((H, mlp_dim), dtype=np.float32),
        # W_fc2 is on the residual path (its output flows back to x).
        W_fc2=rng.normal(0, residual_scale, (mlp_dim, H)).astype(np.float32)
        if use_mlp
        else np.zeros((mlp_dim, H), dtype=np.float32),
        b_fc1=np.zeros(mlp_dim, dtype=np.float32),
        b_fc2=np.zeros(H, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Dropout (forward + backward)
# ---------------------------------------------------------------------------
def dropout_forward(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Inverted dropout: scale kept activations by 1/(1-p) at training time.

    Args:
        x: Input array.
        p: Dropout probability in [0, 1). 0 returns x unchanged.
        training: If False, return x unchanged (inference mode).
        rng: Random number generator for the dropout mask.

    Returns:
        Tuple ``(out, mask)`` where ``mask`` is the boolean keep mask
        (``None`` if dropout was not applied). During backward, multiply
        the upstream gradient by ``mask / (1 - p)``.
    """
    if p <= 0 or not training:
        return x, None
    keep_prob = 1.0 - p
    mask = (rng.random(x.shape) < keep_prob).astype(x.dtype) / keep_prob
    return x * mask, mask


def dropout_backward(
    dout: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Backward pass for dropout.

    Args:
        dout: Upstream gradient.
        mask: Keep mask from ``dropout_forward`` (or ``None``).

    Returns:
        Gradient w.r.t. the input.
    """
    if mask is None:
        return dout
    return dout * mask


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class TinyGPTV3:
    """TinyGPT v3 — modernized pure-NumPy transformer.

    Adds RoPE, GQA, mixed-precision, and gradient checkpointing on top
    of the v2 architecture. See module docstring for references.
    """

    def __init__(self, config: TinyGPTV3Config | None = None) -> None:
        self.config = config or TinyGPTV3Config()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        H = cfg.hidden_dim
        V = cfg.vocab_size
        S = cfg.max_seq_len
        mlp_dim = cfg.mlp_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim

        self.token_emb = rng.normal(0, cfg.init_scale, (V, H)).astype(np.float32)
        # Keep pos_emb for backward compat when use_rope=False.
        self.pos_emb = rng.normal(0, cfg.init_scale, (S, H)).astype(np.float32)
        # Use GPT-2 scaled init when init_scale is the default 0.02;
        # this keeps residual stream variance stable for deep models.
        self.layers: list[TinyLayerV3] = [
            _init_layer_v3_gpt2(
                rng,
                H,
                kv_dim,
                mlp_dim,
                cfg.use_layernorm,
                cfg.use_mlp,
                cfg.n_layers,
                cfg.init_scale,
            )
            for _ in range(cfg.n_layers)
        ]
        # Weight tying: when enabled, the LM head reuses token_emb (transposed).
        # We still allocate lm_head for save/load compatibility and for configs
        # that disable tying; forward() uses token_emb.T directly when tied.
        if cfg.weight_tying:
            self.lm_head = self.token_emb.T.copy()  # synced copy
        else:
            self.lm_head = rng.normal(0, cfg.init_scale, (H, V)).astype(np.float32)
        self.ln_f_gamma = np.ones(H, dtype=np.float32)
        self.ln_f_beta = np.zeros(H, dtype=np.float32)

        self._mp = MixedPrecisionCtx(enabled=cfg.mixed_precision)
        self._hidden_states: list[np.ndarray] = []
        # Training-mode flag (affects dropout).
        self.training: bool = False
        # Dedicated RNG for dropout (so dropout doesn't perturb weight init).
        self._dropout_rng = np.random.default_rng(cfg.seed + 1)

    # ------------------------------------------------------------------
    # Mixed-precision helpers
    # ------------------------------------------------------------------
    @contextmanager
    def mixed_precision(self, enabled: bool = True) -> Iterator[None]:
        """Temporarily enable or disable mixed-precision.

        Args:
            enabled: Whether to enable mixed-precision inside the block.

        Yields:
            None. The previous state is restored on exit.
        """
        prev = self._mp.enabled
        self._mp.enabled = enabled
        try:
            yield
        finally:
            self._mp.enabled = prev

    def _w(self, arr: np.ndarray) -> np.ndarray:
        """Cast a master weight to the compute dtype if MP is on."""
        return self._mp.cast(arr)

    def _lm_head_w(self) -> np.ndarray:
        """Return the LM head weight matrix, honoring weight tying.

        When ``weight_tying=True`` the LM head reuses ``token_emb.T``
        (a single shared parameter matrix). Otherwise it uses the
        separate ``lm_head`` matrix.
        """
        if self.config.weight_tying:
            return self._w(self.token_emb).T  # (H, V)
        return self._w(self.lm_head)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: np.ndarray,
        seq_offset: int = 0,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Forward pass.

        Args:
            token_ids: Token IDs of shape ``(T,)``.
            seq_offset: Position offset (for KV-cache / continuation).

        Returns:
            Tuple ``(logits, hidden_per_layer)`` where logits has shape
            ``(T, vocab_size)`` and ``hidden_per_layer`` is a list of
            ``(T, H)`` arrays, one per layer.
        """
        cfg = self.config
        T = len(token_ids)
        H = cfg.hidden_dim
        x = self._w(self.token_emb)[token_ids]
        if not cfg.use_rope:
            x = x + self._w(self.pos_emb[:T])
        hidden_per_layer: list[np.ndarray] = []

        for layer in self.layers:
            x = self._forward_layer(layer, x, seq_offset)
            hidden_per_layer.append(x.copy())

        x_norm, _, _ = layernorm_forward(x, self._w(self.ln_f_gamma), self._w(self.ln_f_beta))
        logits = x_norm @ self._lm_head_w()
        self._hidden_states = hidden_per_layer
        return logits, hidden_per_layer

    def _forward_layer(
        self,
        layer: TinyLayerV3,
        x: np.ndarray,
        seq_offset: int,
    ) -> np.ndarray:
        """Forward pass for one transformer block."""
        cfg = self.config
        T = x.shape[0]
        H = cfg.hidden_dim
        nh = cfg.n_heads
        nkv = cfg.n_kv_heads
        hd = cfg.head_dim

        # --- Pre-LN1 + GQA attention ---
        if cfg.use_layernorm:
            h_norm, _, _ = layernorm_forward(x, self._w(layer.ln1_gamma), self._w(layer.ln1_beta))
        else:
            h_norm = x

        q = h_norm @ self._w(layer.W_q) + self._w(layer.b_q)  # (T, H)
        k = h_norm @ self._w(layer.W_k) + self._w(layer.b_k)  # (T, kv_dim)
        v = h_norm @ self._w(layer.W_v) + self._w(layer.b_v)  # (T, kv_dim)

        qh = q.reshape(T, nh, hd)
        kh = k.reshape(T, nkv, hd)
        vh = v.reshape(T, nkv, hd)

        attn_out, _ = gqa_forward(
            qh,
            kh,
            vh,
            nh,
            nkv,
            hd,
            rope=cfg.use_rope,
            seq_offset=seq_offset,
            rope_base=cfg.rope_base,
        )
        ctx = attn_out.reshape(T, H)
        attn_out_proj = ctx @ self._w(layer.W_o) + self._w(layer.b_o)
        x = x + attn_out_proj  # residual

        # --- Pre-LN2 + MLP ---
        if cfg.use_layernorm:
            h_norm2, _, _ = layernorm_forward(x, self._w(layer.ln2_gamma), self._w(layer.ln2_beta))
        else:
            h_norm2 = x

        if cfg.use_mlp:
            h1 = h_norm2 @ self._w(layer.W_fc1) + self._w(layer.b_fc1)
            h1_act = _gelu(h1) if cfg.activation == "gelu" else np.maximum(h1, 0.0)
            mlp_out = h1_act @ self._w(layer.W_fc2) + self._w(layer.b_fc2)
            x = x + mlp_out  # residual

        return x

    # ------------------------------------------------------------------
    # Forward + backward (autodiff) — for training
    # ------------------------------------------------------------------
    def forward_with_cache(
        self,
        token_ids: np.ndarray,
        seq_offset: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass that caches every intermediate needed for backward.

        When ``gradient_checkpointing`` is enabled, only the layer
        inputs are cached; intermediates are recomputed during backward.
        """
        cfg = self.config
        T = len(token_ids)
        H = cfg.hidden_dim
        cache: dict[str, Any] = {"token_ids": token_ids, "layer_inputs": [], "layer_caches": []}

        x = self._w(self.token_emb)[token_ids]
        if not cfg.use_rope:
            x = x + self._w(self.pos_emb[:T])
        cache["embed"] = x

        for layer in self.layers:
            cache["layer_inputs"].append(x.copy())
            if cfg.gradient_checkpointing:
                # Only store the input; recompute the rest in backward.
                cache["layer_caches"].append(None)
                x = self._forward_layer(layer, x, seq_offset)
            else:
                x, lc = self._forward_layer_with_cache(layer, x, seq_offset)
                cache["layer_caches"].append(lc)

        x_norm, mu_f, rstd_f = layernorm_forward(
            x, self._w(self.ln_f_gamma), self._w(self.ln_f_beta)
        )
        logits = x_norm @ self._lm_head_w()
        cache["x_pre_head"] = x
        cache["x_norm"] = x_norm
        cache["mu_f"] = mu_f
        cache["rstd_f"] = rstd_f
        cache["seq_offset"] = seq_offset
        return logits, cache

    def _forward_layer_with_cache(
        self,
        layer: TinyLayerV3,
        x: np.ndarray,
        seq_offset: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass for one layer that caches intermediates."""
        cfg = self.config
        T = x.shape[0]
        H = cfg.hidden_dim
        nh = cfg.n_heads
        nkv = cfg.n_kv_heads
        hd = cfg.head_dim
        lc: dict[str, Any] = {"x": x}

        if cfg.use_layernorm:
            h_norm, mu1, rstd1 = layernorm_forward(
                x, self._w(layer.ln1_gamma), self._w(layer.ln1_beta)
            )
            lc["mu1"], lc["rstd1"] = mu1, rstd1
        else:
            h_norm = x

        q = h_norm @ self._w(layer.W_q) + self._w(layer.b_q)
        k = h_norm @ self._w(layer.W_k) + self._w(layer.b_k)
        v = h_norm @ self._w(layer.W_v) + self._w(layer.b_v)
        qh = q.reshape(T, nh, hd)
        kh = k.reshape(T, nkv, hd)
        vh = v.reshape(T, nkv, hd)

        attn_out, gqa_cache = gqa_forward(
            qh,
            kh,
            vh,
            nh,
            nkv,
            hd,
            rope=cfg.use_rope,
            seq_offset=seq_offset,
            rope_base=cfg.rope_base,
        )
        ctx = attn_out.reshape(T, H)
        attn_out_proj = ctx @ self._w(layer.W_o) + self._w(layer.b_o)
        # Dropout on attention output (residual path).
        attn_out_proj, attn_drop_mask = dropout_forward(
            attn_out_proj, cfg.dropout, self.training, self._dropout_rng
        )
        x = x + attn_out_proj

        lc["h_norm1"] = h_norm
        lc["gqa_cache"] = gqa_cache
        lc["ctx"] = ctx
        lc["attn_drop_mask"] = attn_drop_mask
        lc["x1"] = x  # input to LN2 (= x_in + attn_out_proj)

        if cfg.use_layernorm:
            h_norm2, mu2, rstd2 = layernorm_forward(
                x, self._w(layer.ln2_gamma), self._w(layer.ln2_beta)
            )
            lc["mu2"], lc["rstd2"] = mu2, rstd2
        else:
            h_norm2 = x

        if cfg.use_mlp:
            h1 = h_norm2 @ self._w(layer.W_fc1) + self._w(layer.b_fc1)
            h1_act = _gelu(h1) if cfg.activation == "gelu" else np.maximum(h1, 0.0)
            mlp_out = h1_act @ self._w(layer.W_fc2) + self._w(layer.b_fc2)
            # Dropout on MLP output (residual path).
            mlp_out, mlp_drop_mask = dropout_forward(
                mlp_out, cfg.dropout, self.training, self._dropout_rng
            )
            x = x + mlp_out
            lc["h_norm2"] = h_norm2
            lc["h1"] = h1
            lc["h1_act"] = h1_act
            lc["mlp_out"] = mlp_out
            lc["mlp_drop_mask"] = mlp_drop_mask

        lc["x_out"] = x
        return x, lc

    def backward(
        self,
        cache: dict[str, Any],
        dlogits: np.ndarray,
    ) -> dict[str, Any]:
        """Reverse-mode autodiff through the whole model.

        Args:
            cache: Cache from ``forward_with_cache``.
            dlogits: Upstream gradient w.r.t. logits, shape ``(T, V)``.

        Returns:
            Dict mapping parameter names to their gradients (float32).
        """
        cfg = self.config
        H = cfg.hidden_dim
        T = dlogits.shape[0]
        seq_offset = cache["seq_offset"]

        grads: dict[str, Any] = {
            "token_emb": np.zeros_like(self.token_emb),
            "lm_head": np.zeros_like(self.lm_head),
            "ln_f_gamma": np.zeros(H, dtype=np.float32),
            "ln_f_beta": np.zeros(H, dtype=np.float32),
        }
        for i, layer in enumerate(self.layers):
            for attr in (
                "W_q",
                "W_k",
                "W_v",
                "W_o",
                "b_q",
                "b_k",
                "b_v",
                "b_o",
                "ln1_gamma",
                "ln1_beta",
                "ln2_gamma",
                "ln2_beta",
                "W_fc1",
                "W_fc2",
                "b_fc1",
                "b_fc2",
            ):
                grads[f"L{i}_{attr}"] = np.zeros_like(getattr(layer, attr))

        # --- LM head + final LN ---
        # When weight_tying=True, logits = x_norm @ token_emb.T, so
        #   d_token_emb (from LM head) = dlogits.T @ x_norm   → (V, H)
        #   dx_norm                    = dlogits @ token_emb   → (T, H)
        # When weight_tying=False, use the standalone lm_head as before.
        x_norm = cache["x_norm"]
        head_w = self._w(self.token_emb).T if cfg.weight_tying else self._w(self.lm_head)
        d_lm_head = x_norm.T @ dlogits  # (H, V)
        dx_norm = dlogits @ head_w.T  # (T, H)
        dx, dln_f_g, dln_f_b = layernorm_backward(
            dx_norm,
            cache["x_pre_head"],
            self._w(self.ln_f_gamma),
            cache["mu_f"],
            cache["rstd_f"],
        )
        if cfg.weight_tying:
            # The LM head gradient flows into token_emb (transposed).
            # We store it under "lm_head" so the optimizer can route it
            # back to token_emb via the `tie_gradients` helper.
            grads["lm_head"] = d_lm_head.T.astype(np.float32)  # (V, H) — same shape as token_emb
        else:
            grads["lm_head"] = d_lm_head.astype(np.float32)  # (H, V) — standalone
        grads["ln_f_gamma"] = dln_f_g.astype(np.float32)
        grads["ln_f_beta"] = dln_f_b.astype(np.float32)

        # --- Layers (reverse order) ---
        for i in range(cfg.n_layers - 1, -1, -1):
            layer = self.layers[i]
            x_in = cache["layer_inputs"][i]
            if cfg.gradient_checkpointing:
                # Recompute the layer cache on the fly.
                _, lc = self._forward_layer_with_cache(layer, x_in, seq_offset)
            else:
                lc = cache["layer_caches"][i]

            dx, lg = self._backward_layer(layer, lc, dx, seq_offset, i)
            for attr, g in lg.items():
                grads[f"L{i}_{attr}"] = g

        # --- Embedding ---
        token_ids = cache["token_ids"]
        if cfg.use_rope:
            np.add.at(grads["token_emb"], token_ids, dx.astype(np.float32))
        else:
            np.add.at(grads["token_emb"], token_ids, dx.astype(np.float32))
            # pos_emb gradient
            dpos = dx.sum(axis=0, keepdims=True)
            # We do not store pos_emb grad explicitly in this minimal autodiff.

        # --- Weight tying: fold the LM head gradient into token_emb ---
        # When weight_tying=True, the "lm_head" gradient (shape (V, H))
        # is the transposed gradient of the shared matrix, so we add it
        # to token_emb and drop the standalone lm_head entry.
        if cfg.weight_tying and "lm_head" in grads:
            grads["token_emb"] = grads["token_emb"] + grads["lm_head"]
            # Mark lm_head as zero so the optimizer does not double-update.
            grads["lm_head"] = np.zeros_like(grads["lm_head"])
        return grads

    def _backward_layer(
        self,
        layer: TinyLayerV3,
        lc: dict[str, Any],
        dx: np.ndarray,
        seq_offset: int,
        layer_idx: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Backward pass for one layer. Returns (dx_in, grads_dict)."""
        cfg = self.config
        H = cfg.hidden_dim
        nh = cfg.n_heads
        nkv = cfg.n_kv_heads
        hd = cfg.head_dim
        T = dx.shape[0]
        g: dict[str, np.ndarray] = {}

        x = lc["x"]
        # --- MLP backward ---
        if cfg.use_mlp:
            # mlp_out was dropped before being added to the residual.
            # Backward through dropout first.
            dmlp_out = dropout_backward(dx, lc.get("mlp_drop_mask"))
            dh1_act = dmlp_out @ self._w(layer.W_fc2).T
            g["W_fc2"] = lc["h1_act"].T @ dmlp_out
            g["b_fc2"] = dmlp_out.sum(axis=0)
            if cfg.activation == "gelu":
                dh1 = dh1_act * _gelu_grad(lc["h1"])
            else:
                dh1 = dh1_act * (lc["h1"] > 0).astype(dh1_act.dtype)
            g["W_fc1"] = lc["h_norm2"].T @ dh1
            g["b_fc1"] = dh1.sum(axis=0)
            dh_norm2 = dh1 @ self._w(layer.W_fc1).T
            # LN2 backward — use x1 (the input to LN2), not x_in.
            if cfg.use_layernorm:
                dx_ln2, g["ln2_gamma"], g["ln2_beta"] = layernorm_backward(
                    dh_norm2, lc["x1"], self._w(layer.ln2_gamma), lc["mu2"], lc["rstd2"]
                )
            else:
                dx_ln2 = dh_norm2
            # x_out = x1 + mlp_out  →  dx1 = dx_out (residual) + dx_ln2 (MLP path)
            dx = dx + dx_ln2
        else:
            g["W_fc1"] = np.zeros_like(layer.W_fc1)
            g["W_fc2"] = np.zeros_like(layer.W_fc2)
            g["b_fc1"] = np.zeros_like(layer.b_fc1)
            g["b_fc2"] = np.zeros_like(layer.b_fc2)
            if cfg.use_layernorm:
                g["ln2_gamma"] = np.zeros_like(layer.ln2_gamma)
                g["ln2_beta"] = np.zeros_like(layer.ln2_beta)

        # --- Attention backward ---
        # x1 = x + attn_out_proj  ->  dx_residual = dx,  d_attn_out_proj = dx
        # Backward through dropout on attn_out_proj first.
        d_attn_out_proj = dropout_backward(dx, lc.get("attn_drop_mask"))
        dctx = d_attn_out_proj @ self._w(layer.W_o).T  # (T, H)
        g["W_o"] = lc["ctx"].T @ d_attn_out_proj
        g["b_o"] = d_attn_out_proj.sum(axis=0)

        dattn_out = dctx.reshape(T, nh, hd)
        dq, dk, dv = gqa_backward(dattn_out, lc["gqa_cache"], nh, nkv, hd)
        dq = dq.reshape(T, H)
        dk = dk.reshape(T, nkv * hd)
        dv = dv.reshape(T, nkv * hd)

        g["W_q"] = lc["h_norm1"].T @ dq
        g["b_q"] = dq.sum(axis=0)
        g["W_k"] = lc["h_norm1"].T @ dk
        g["b_k"] = dk.sum(axis=0)
        g["W_v"] = lc["h_norm1"].T @ dv
        g["b_v"] = dv.sum(axis=0)

        dh_norm1 = dq @ self._w(layer.W_q).T + dk @ self._w(layer.W_k).T + dv @ self._w(layer.W_v).T

        if cfg.use_layernorm:
            dx_ln1, g["ln1_gamma"], g["ln1_beta"] = layernorm_backward(
                dh_norm1, x, self._w(layer.ln1_gamma), lc["mu1"], lc["rstd1"]
            )
            dx_in = dx + dx_ln1  # residual
        else:
            dx_in = dx + dh_norm1

        # Cast all grads to float32 (master dtype).
        for k in g:
            g[k] = g[k].astype(np.float32)
        return dx_in, g

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int | None = None,
        capture_hidden: bool = True,
    ) -> dict[str, Any]:
        """Generate tokens with top-k / top-p sampling.

        Args:
            prompt_ids: Prompt token IDs.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_k: If > 0, keep only the top-k tokens.
            top_p: If < 1.0, keep the smallest set of tokens whose
                cumulative probability ≥ top_p (nucleus sampling).
            seed: RNG seed for reproducibility.
            capture_hidden: If True, store per-step hidden states.

        Returns:
            Dict with ``output_ids``, ``full_ids``, ``per_step_logits``,
            ``hidden_snapshots``, and ``reasoning_trace``.
        """
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        ids = [int(i) for i in prompt_ids]
        per_step_logits: list[np.ndarray] = []
        hidden_snapshots: list[list[np.ndarray]] = []

        for _step in range(max_new_tokens):
            ctx = np.array(ids[-self.config.max_seq_len :], dtype=np.int64)
            seq_offset = max(0, len(ids) - self.config.max_seq_len)
            logits, hidden = self.forward(ctx, seq_offset=seq_offset)
            last_logits = logits[-1].astype(np.float64)

            if temperature > 0:
                last_logits = last_logits / temperature
            else:
                next_id = int(np.argmax(last_logits))
                ids.append(next_id)
                per_step_logits.append(last_logits)
                if capture_hidden:
                    hidden_snapshots.append([h.copy() for h in hidden])
                continue

            if top_k > 0 and top_k < len(last_logits):
                kth = np.partition(last_logits, -top_k)[-top_k]
                last_logits = np.where(last_logits < kth, -1e9, last_logits)

            if top_p < 1.0:
                order = np.argsort(last_logits)[::-1]
                cdf = np.cumsum(_softmax(last_logits[order]))
                cutoff = int(np.searchsorted(cdf, top_p)) + 1
                keep = order[:cutoff]
                m = np.full_like(last_logits, -1e9)
                m[keep] = last_logits[keep]
                last_logits = m

            probs = _softmax(last_logits)
            next_id = int(rng.choice(len(probs), p=probs))
            ids.append(next_id)
            per_step_logits.append(last_logits)
            if capture_hidden:
                hidden_snapshots.append([h.copy() for h in hidden])

        return {
            "output_ids": ids[len(prompt_ids) :],
            "full_ids": ids,
            "per_step_logits": per_step_logits,
            "hidden_snapshots": hidden_snapshots,
        }

    # ------------------------------------------------------------------
    # RMT spectral analysis (same interface as v2)
    # ------------------------------------------------------------------
    def spectral_analysis(self, hidden_states: list[np.ndarray]) -> dict[str, Any]:
        """Per-layer covariance spectral analysis against MP bounds."""
        results: list[dict[str, Any]] = []
        for li, h in enumerate(hidden_states):
            T = h.shape[0]
            if T < 2:
                continue
            cov = np.cov(h.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-12]
            if len(eigvals) == 0:
                continue
            lam_max = float(eigvals.max())
            lam_min = float(eigvals.min())
            lam_mean = float(eigvals.mean())
            q = self.config.hidden_dim / max(T, 1)
            sigma2 = lam_mean
            mp_upper = sigma2 * (1 + math.sqrt(q)) ** 2
            mp_lower = sigma2 * (1 - math.sqrt(q)) ** 2
            signal_detected = lam_max > mp_upper * 1.05
            tw_scale = sigma2 * (q ** (2 / 3)) / (T ** (2 / 3))
            tw_fluct = (lam_max - mp_upper) / max(tw_scale, 1e-12)
            results.append(
                {
                    "layer": li,
                    "T": T,
                    "H": self.config.hidden_dim,
                    "q": q,
                    "lambda_max": lam_max,
                    "lambda_min": lam_min,
                    "lambda_mean": lam_mean,
                    "mp_upper": mp_upper,
                    "mp_lower": mp_lower,
                    "signal_detected": signal_detected,
                    "tw_fluctuation": tw_fluct,
                    "spectral_gap": lam_max - lam_min,
                }
            )
        return {"layers": results}

    # ------------------------------------------------------------------
    # Save / load (NPZ, backward compatible)
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """Save master weights to an ``.npz`` file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state: dict[str, Any] = {
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "lm_head": self.lm_head,
            "ln_f_gamma": self.ln_f_gamma,
            "ln_f_beta": self.ln_f_beta,
            "config": json.dumps(self.config.__dict__),
        }
        for i, layer in enumerate(self.layers):
            for attr in (
                "W_q",
                "W_k",
                "W_v",
                "W_o",
                "b_q",
                "b_k",
                "b_v",
                "b_o",
                "ln1_gamma",
                "ln1_beta",
                "ln2_gamma",
                "ln2_beta",
                "W_fc1",
                "W_fc2",
                "b_fc1",
                "b_fc2",
            ):
                state[f"L{i}_{attr}"] = getattr(layer, attr)
        np.savez(path, **state)

    @classmethod
    def load_weights(cls, path: str) -> TinyGPTV3:
        """Load master weights from an ``.npz`` file."""
        data = np.load(path, allow_pickle=False)
        cfg_dict = json.loads(str(data["config"]))
        # Tolerate older configs that might be missing new fields.
        cfg = TinyGPTV3Config(**cfg_dict)
        model = cls(cfg)
        model.token_emb = data["token_emb"]
        model.pos_emb = data["pos_emb"]
        model.lm_head = data["lm_head"]
        if "ln_f_gamma" in data.files:
            model.ln_f_gamma = data["ln_f_gamma"]
            model.ln_f_beta = data["ln_f_beta"]
        model.layers = []
        for i in range(cfg.n_layers):
            kw = {
                attr: data[f"L{i}_{attr}"]
                for attr in (
                    "W_q",
                    "W_k",
                    "W_v",
                    "W_o",
                    "b_q",
                    "b_k",
                    "b_v",
                    "b_o",
                    "ln1_gamma",
                    "ln1_beta",
                    "ln2_gamma",
                    "ln2_beta",
                    "W_fc1",
                    "W_fc2",
                    "b_fc1",
                    "b_fc2",
                )
                if f"L{i}_{attr}" in data.files
            }
            model.layers.append(TinyLayerV3(**kw))
        return model

    # ------------------------------------------------------------------
    # Convenience: count parameters
    # ------------------------------------------------------------------
    def n_parameters(self) -> int:
        """Total number of float32 master parameters."""
        return self.config.params_count


# ---------------------------------------------------------------------------
# Label smoothing + gradient clipping (Session 1 utilities)
# ---------------------------------------------------------------------------
def label_smoothing_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-entropy loss with label smoothing.

    Implements::

        loss = (1 - ε) * NLL(p, target) + ε * mean(-log p_i)

    where ``ε = smoothing`` and ``NLL`` is the standard negative
    log-likelihood. The second term is the uniform part that prevents
    the model from becoming overconfident.

    Args:
        logits: Logits of shape ``(T, V)`` or ``(B, T, V)``.
        targets: Target indices of shape ``(T,)`` or ``(B, T)``.
        smoothing: Smoothing factor ε in ``[0, 1)``. 0 = hard CE.
        ignore_index: Targets equal to this value are masked out
            (their loss is zero).

    Returns:
        Tuple ``(loss, dlogits)`` where ``loss`` is a scalar and
        ``dlogits`` has the same shape as ``logits``.
    """
    if smoothing < 0 or smoothing >= 1:
        raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
    V = logits.shape[-1]
    # log-softmax for numerical stability: log_softmax(x) = x - logsumexp(x)
    m = np.max(logits, axis=-1, keepdims=True)
    # logsumexp: log(sum(exp(x))) = m + log(sum(exp(x - m)))
    log_z = m + np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))  # (..., 1)
    log_probs = logits - log_z  # (..., V)
    probs = np.exp(log_probs)

    # Build smoothed targets: (1 - ε) on target, ε / V elsewhere.
    # For ignore_index, mask out (set weight 0).
    mask = (targets != ignore_index).astype(np.float32)
    safe_targets = np.where(targets == ignore_index, 0, targets)

    # NLL part: -log_probs[target] * (1 - ε)
    nll = -np.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    # Uniform part: -mean(log_probs) over V = -sum(log_probs) / V
    # Only contributes when smoothing > 0.
    if smoothing > 0:
        uniform = -np.sum(log_probs, axis=-1) / V
        loss_per_pos = (1.0 - smoothing) * nll + smoothing * uniform
    else:
        loss_per_pos = nll

    loss_per_pos = loss_per_pos * mask
    n_valid = mask.sum()
    loss = float(loss_per_pos.sum() / max(n_valid, 1))

    # Gradient: dL/dlogits = (probs - smoothed_target) / n_valid
    # smoothed_target[target] = (1 - ε) + ε / V
    # smoothed_target[other]  = ε / V
    # When smoothing=0, this reduces to (probs - one_hot) / n_valid.
    eps_v = smoothing / V
    smoothed = np.full_like(probs, eps_v)
    # Set the target position to (1 - ε + ε/V) via direct indexing.
    # We need to handle arbitrary batch shapes; flatten, set, reshape.
    flat_smoothed = smoothed.reshape(-1, V)
    flat_targets = safe_targets.reshape(-1)
    flat_mask = mask.reshape(-1).astype(bool)
    for i in range(flat_smoothed.shape[0]):
        if flat_mask[i]:
            flat_smoothed[i, flat_targets[i]] = 1.0 - smoothing + eps_v
    smoothed = flat_smoothed.reshape(probs.shape)
    dlogits = (probs - smoothed) * mask[..., None]
    dlogits = dlogits / max(n_valid, 1)
    return np.array(loss, dtype=logits.dtype), dlogits.astype(logits.dtype)


def clip_grad_norm_(
    grads: dict[str, np.ndarray],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """Global gradient clipping by L2 norm, in-place.

    Computes the total L2 norm across all gradient arrays and rescales
    them if the norm exceeds ``max_norm``. This is the standard recipe
    from PyTorch's ``torch.nn.utils.clip_grad_norm_``.

    Args:
        grads: Dict of gradient arrays (modified in-place when clipped).
        max_norm: Maximum allowed total norm. If 0, no clipping.
        eps: Small constant to avoid division by zero.

    Returns:
        The total gradient norm before clipping (for logging).
    """
    if max_norm <= 0:
        return 0.0
    total_sq = 0.0
    for g in grads.values():
        if g is None or g.size == 0:
            continue
        total_sq += float(np.sum(g.astype(np.float64) ** 2))
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm and total_norm > eps:
        scale = max_norm / (total_norm + eps)
        for k in grads:
            if grads[k] is not None and grads[k].size > 0:
                grads[k] = (grads[k].astype(np.float64) * scale).astype(grads[k].dtype)
    return total_norm


# ---------------------------------------------------------------------------
# Learning-rate scheduler (cosine with warmup) — Session 3
# ---------------------------------------------------------------------------
def cosine_lr_schedule(
    step: int,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine learning-rate schedule with linear warmup.

    Phase 1 (warmup): ``lr = max_lr * step / warmup_steps``
    Phase 2 (cosine decay): ``lr = min_lr + 0.5 * (max_lr - min_lr) *
    (1 + cos(π * (step - warmup) / (total - warmup)))``

    Args:
        step: Current step (0-indexed).
        max_lr: Peak learning rate after warmup.
        warmup_steps: Number of warmup steps (linear ramp from 0).
        total_steps: Total number of training steps.
        min_lr_ratio: Final LR as a fraction of max_lr (default 0.1).

    Returns:
        Learning rate for the current step.
    """
    if total_steps <= 0:
        return max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    min_lr = max_lr * min_lr_ratio
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def linear_lr_schedule(
    step: int,
    max_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """Linear learning-rate schedule with linear warmup.

    Phase 1 (warmup): ``lr = max_lr * step / warmup_steps``
    Phase 2 (linear decay): ``lr = max_lr * (1 - progress * (1 - min_lr_ratio))``
    """
    if total_steps <= 0:
        return max_lr
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return max_lr * (1.0 - progress * (1.0 - min_lr_ratio))


# ---------------------------------------------------------------------------
# Early stopping — Session 3
# ---------------------------------------------------------------------------
@dataclass
class EarlyStopping:
    """Early stopping on a monitored metric.

    Stops training when the metric has not improved by more than
    ``min_delta`` for ``patience`` consecutive checks. The metric is
    "better" when it decreases (if ``mode='min'``) or increases
    (if ``mode='max'``).

    Attributes:
        patience: Number of checks without improvement before stopping.
        min_delta: Minimum change to count as an improvement.
        mode: ``'min'`` (lower is better) or ``'max'`` (higher is better).
        best: Best metric value seen so far.
        counter: Number of checks since the last improvement.
        stopped: Whether training should stop.
    """

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    best: float = float("inf") if mode == "min" else float("-inf")
    counter: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        # Re-init best based on mode (dataclass default is set before validation).
        if self.mode == "min" and self.best == float("-inf"):
            self.best = float("inf")
        elif self.mode == "max" and self.best == float("inf"):
            self.best = float("-inf")

    def step(self, metric: float) -> bool:
        """Update the early-stopper with a new metric value.

        Args:
            metric: Current metric value (e.g., validation loss).

        Returns:
            True if training should stop, False otherwise.
        """
        improved = (self.mode == "min" and metric < self.best - self.min_delta) or (
            self.mode == "max" and metric > self.best + self.min_delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped

    def reset(self) -> None:
        """Reset state (e.g., for a new training run)."""
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.stopped = False


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------
def config_4m() -> TinyGPTV3Config:
    """Return the 'TinyGPT 4M' config from the roadmap.

    Target: ~4.15M parameters, 10 layers, hidden=192, 6 query heads,
    2 KV heads (GQA group size 3), BPE vocab=512. RoPE replaces
    absolute position embeddings, saving ``max_seq_len * H = 49K`` params.
    """
    return TinyGPTV3Config(
        vocab_size=512,
        hidden_dim=192,
        n_layers=10,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=256,
        mlp_ratio=4,
        use_rope=True,
        seed=42,
    )


def config_12m() -> TinyGPTV3Config:
    """Return the 'TinyGPT 12M' config — scaled up for higher match_rate.

    Target: ~12.5M parameters, 16 layers, hidden=256, 8 query heads,
    2 KV heads (GQA group size 4), BPE vocab=1024. Uses RoPE, weight
    tying, dropout 0.1, GPT-2 scaled init, and grad clipping 1.0 —
    the full Tier 1+2 recipe for ~25% match_rate on test set.

    Training takes ~2-3 hours on CPU (150 epochs, batch 8).
    """
    return TinyGPTV3Config(
        vocab_size=1024,
        hidden_dim=256,
        n_layers=16,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        mlp_ratio=4,
        use_rope=True,
        weight_tying=True,
        label_smoothing=0.1,
        grad_clip_norm=1.0,
        dropout=0.1,
        init_scale=0.02,
        seed=42,
    )


def config_train_4m() -> TinyGPTV3Config:
    """Return a training-ready 4M config with all Tier 1+2 features on.

    This is the config to use for the first end-to-end training run.
    Enables weight tying, label smoothing 0.1, grad clipping 1.0,
    dropout 0.1 — the recipe predicted to reach ~16% match_rate
    (up from v2's 6.5%).
    """
    return TinyGPTV3Config(
        vocab_size=512,
        hidden_dim=192,
        n_layers=10,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=256,
        mlp_ratio=4,
        use_rope=True,
        weight_tying=True,
        label_smoothing=0.1,
        grad_clip_norm=1.0,
        dropout=0.1,
        init_scale=0.02,
        seed=42,
    )


def config_small() -> TinyGPTV3Config:
    """Small config for fast tests: ~50K params, 2 layers."""
    return TinyGPTV3Config(
        vocab_size=64,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        mlp_ratio=2,
        use_rope=True,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = config_4m()
    print(f"TinyGPT v3 config: {cfg}")
    print(f"Approx params: {cfg.params_count:,}")
    print(f"  GQA group size: {cfg.n_groups}")
    print(
        f"  RoPE: {cfg.use_rope}, MP: {cfg.mixed_precision}, "
        f"Checkpointing: {cfg.gradient_checkpointing}"
    )
    model = TinyGPTV3(cfg)
    prompt = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    out = model.generate(prompt, max_new_tokens=16, temperature=0.7, seed=42)
    print(f"Generated {len(out['output_ids'])} new tokens")
    print(f"First 5 logits shape: {out['per_step_logits'][0].shape}")
    spec = model.spectral_analysis(out["hidden_snapshots"][-1])
    print(f"Spectral layers analyzed: {len(spec['layers'])}")
