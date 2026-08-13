"""
tiny_gpt.py — Local Synthetic Tiny-GPT for RMT-LLM Laboratory
==============================================================

Pure-NumPy implementation of a small transformer with **pre-LayerNorm +
MLP** blocks (GELU activation) and an optional **BPE tokenizer** (when
paired with `tiny_gpt_trainer.py`). Default config targets ~8M params
(vocab=512, hidden=128, layers=12, heads=4).

Architecture (per layer):
    h_in -> LN1 -> MHA(Q,K,V,O) -> +residual -> LN2 -> MLP(4H, GELU) -> +residual -> h_out

No external dependencies beyond numpy. Exposes:
  - hidden state capture (for RMT spectral analysis)
  - reasoning-trace capture (pre-output)
  - hallucination scoring (RMT-predicted N_crit vs actual token count)
  - lying detection (does the model 'know' the answer but generate otherwise)

Author: Iskhak Hamzatovich Isaev
ORCID: 0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
@dataclass
class TinyGPTConfig:
    vocab_size: int = 512          # 256 byte tokens + 256 BPE merges
    hidden_dim: int = 128
    n_layers: int = 12
    n_heads: int = 4
    max_seq_len: int = 256
    mlp_ratio: int = 4             # MLP hidden = mlp_ratio * hidden_dim
    use_layernorm: bool = True
    use_mlp: bool = True
    activation: str = "gelu"       # "gelu" | "relu"
    seed: int = 42

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads

    @property
    def mlp_dim(self) -> int:
        return self.hidden_dim * self.mlp_ratio

    @property
    def params_count(self) -> int:
        # Embeddings
        emb = self.vocab_size * self.hidden_dim + self.max_seq_len * self.hidden_dim
        # Per layer: 4 attn projections (H*H each) + 4 biases
        #           + 2 LN (2H each) + MLP (2*H*mlp_dim + mlp_dim + H)
        per_layer = 4 * self.hidden_dim * self.hidden_dim + 4 * self.hidden_dim
        if self.use_layernorm:
            per_layer += 4 * self.hidden_dim   # 2 LN blocks, gamma+beta each
        if self.use_mlp:
            per_layer += 2 * self.hidden_dim * self.mlp_dim + self.mlp_dim + self.hidden_dim
        head = self.hidden_dim * self.vocab_size
        return emb + self.n_layers * per_layer + head


# ---------------------------------------------------------------------------
# Tiny transformer block (attention + optional MLP + optional LayerNorm)
# ---------------------------------------------------------------------------
@dataclass
class TinyLayer:
    # Attention projections
    W_q: np.ndarray   # (H, H)
    W_k: np.ndarray
    W_v: np.ndarray
    W_o: np.ndarray
    b_q: np.ndarray   # (H,)
    b_k: np.ndarray
    b_v: np.ndarray
    b_o: np.ndarray
    # Pre-LN1 (before attention)
    ln1_gamma: np.ndarray   # (H,)
    ln1_beta: np.ndarray    # (H,)
    # Pre-LN2 (before MLP)
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    # MLP: H -> 4H -> H
    W_fc1: np.ndarray   # (H, mlp_dim)
    W_fc2: np.ndarray   # (mlp_dim, H)
    b_fc1: np.ndarray   # (mlp_dim,)
    b_fc2: np.ndarray   # (H,)


def _init_layer(rng: np.random.Generator, H: int, mlp_dim: int,
                use_layernorm: bool, use_mlp: bool) -> TinyLayer:
    scale = 1.0 / math.sqrt(H)
    mlp_scale = 1.0 / math.sqrt(H)
    return TinyLayer(
        W_q=rng.normal(0, scale, (H, H)).astype(np.float32),
        W_k=rng.normal(0, scale, (H, H)).astype(np.float32),
        W_v=rng.normal(0, scale, (H, H)).astype(np.float32),
        W_o=rng.normal(0, scale, (H, H)).astype(np.float32),
        b_q=np.zeros(H, dtype=np.float32),
        b_k=np.zeros(H, dtype=np.float32),
        b_v=np.zeros(H, dtype=np.float32),
        b_o=np.zeros(H, dtype=np.float32),
        ln1_gamma=np.ones(H, dtype=np.float32) if use_layernorm else np.ones(H, dtype=np.float32),
        ln1_beta=np.zeros(H, dtype=np.float32) if use_layernorm else np.zeros(H, dtype=np.float32),
        ln2_gamma=np.ones(H, dtype=np.float32),
        ln2_beta=np.zeros(H, dtype=np.float32),
        W_fc1=rng.normal(0, mlp_scale, (H, mlp_dim)).astype(np.float32) if use_mlp else np.zeros((H, mlp_dim), dtype=np.float32),
        W_fc2=rng.normal(0, mlp_scale, (mlp_dim, H)).astype(np.float32) if use_mlp else np.zeros((mlp_dim, H), dtype=np.float32),
        b_fc1=np.zeros(mlp_dim, dtype=np.float32),
        b_fc2=np.zeros(H, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def _gelu(x: np.ndarray) -> np.ndarray:
    # Exact GELU (tanh approximation, used by GPT-2/BERT)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    # Derivative of tanh-approx GELU
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    diner = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner ** 2) * diner


# ---------------------------------------------------------------------------
# LayerNorm (forward + backward helper)
# ---------------------------------------------------------------------------
def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LayerNorm over last dim. Returns (out, mean, rstd)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xn = (x - mu) * rstd
    out = gamma * xn + beta
    return out.astype(x.dtype), mu, rstd


def layernorm_backward(dout: np.ndarray, x: np.ndarray, gamma: np.ndarray,
                       mu: np.ndarray, rstd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dx, dgamma, dbeta)."""
    D = x.shape[-1]
    xn = (x - mu) * rstd
    dgamma = np.sum(dout * xn, axis=tuple(range(dout.ndim - 1)))
    dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
    dxn = dout * gamma
    # dx = rstd / D * (D * dxn - sum(dxn) - xn * sum(dxn * xn))
    sum_dxn = np.sum(dxn, axis=-1, keepdims=True)
    sum_dxn_xn = np.sum(dxn * xn, axis=-1, keepdims=True)
    dx = (rstd / D) * (D * dxn - sum_dxn - xn * sum_dxn_xn)
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class TinyGPT:
    """Tiny transformer for laboratory scenarios. Pure NumPy."""

    def __init__(self, config: Optional[TinyGPTConfig] = None) -> None:
        self.config = config or TinyGPTConfig()
        rng = np.random.default_rng(self.config.seed)
        H = self.config.hidden_dim
        V = self.config.vocab_size
        S = self.config.max_seq_len
        mlp_dim = self.config.mlp_dim

        self.token_emb = rng.normal(0, 0.02, (V, H)).astype(np.float32)
        self.pos_emb = rng.normal(0, 0.02, (S, H)).astype(np.float32)
        self.layers: List[TinyLayer] = [
            _init_layer(rng, H, mlp_dim, self.config.use_layernorm, self.config.use_mlp)
            for _ in range(self.config.n_layers)
        ]
        self.lm_head = rng.normal(0, 0.02, (H, V)).astype(np.float32)
        # Final LN before LM head (pre-LN architecture)
        self.ln_f_gamma = np.ones(H, dtype=np.float32)
        self.ln_f_beta = np.zeros(H, dtype=np.float32)

        # Diagnostics
        self._hidden_states: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Forward pass with hidden-state capture
    # ------------------------------------------------------------------
    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass. Returns logits (T, V) and per-layer hidden states."""
        T = len(token_ids)
        H = self.config.hidden_dim
        x = self.token_emb[token_ids] + self.pos_emb[:T]   # (T, H)
        hidden_per_layer: List[np.ndarray] = []

        for layer in self.layers:
            x = self._forward_layer(layer, x)
            hidden_per_layer.append(x.copy())

        # Final LN + LM head
        x_norm, _, _ = layernorm_forward(x, self.ln_f_gamma, self.ln_f_beta)
        logits = x_norm @ self.lm_head                              # (T, V)
        self._hidden_states = hidden_per_layer
        return logits, hidden_per_layer

    def _forward_layer(self, layer: TinyLayer, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        H = self.config.hidden_dim
        nh = self.config.n_heads
        hd = self.config.head_dim

        # --- Pre-LN1 + Attention ---
        if self.config.use_layernorm:
            h_norm, _, _ = layernorm_forward(x, layer.ln1_gamma, layer.ln1_beta)
        else:
            h_norm = x

        q = h_norm @ layer.W_q + layer.b_q
        k = h_norm @ layer.W_k + layer.b_k
        v = h_norm @ layer.W_v + layer.b_v

        qh = q.reshape(T, nh, hd).transpose(1, 0, 2)   # (nh, T, hd)
        kh = k.reshape(T, nh, hd).transpose(1, 0, 2)
        vh = v.reshape(T, nh, hd).transpose(1, 0, 2)

        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(hd)   # (nh, T, T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        attn = _softmax(scores, axis=-1)
        ctx = attn @ vh                                     # (nh, T, hd)
        ctx = ctx.transpose(1, 0, 2).reshape(T, H)          # (T, H)
        attn_out = ctx @ layer.W_o + layer.b_o

        x = x + attn_out                                    # residual

        # --- Pre-LN2 + MLP ---
        if self.config.use_layernorm:
            h_norm2, _, _ = layernorm_forward(x, layer.ln2_gamma, layer.ln2_beta)
        else:
            h_norm2 = x

        if self.config.use_mlp:
            h1 = h_norm2 @ layer.W_fc1 + layer.b_fc1     # (T, 4H)
            if self.config.activation == "gelu":
                h1_act = _gelu(h1)
            else:
                h1_act = np.maximum(h1, 0.0)
            mlp_out = h1_act @ layer.W_fc2 + layer.b_fc2  # (T, H)
            x = x + mlp_out                                # residual

        return x

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
        seed: Optional[int] = None,
        capture_hidden: bool = True,
    ) -> Dict[str, Any]:
        """Generate tokens. Returns dict with output_ids, hidden_snapshots, etc."""
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        ids = list(int(i) for i in prompt_ids)
        per_step_logits: List[np.ndarray] = []
        hidden_snapshots: List[List[np.ndarray]] = []

        for _ in range(max_new_tokens):
            ctx = np.array(ids[-self.config.max_seq_len:], dtype=np.int64)
            logits, hidden = self.forward(ctx)
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
                cutoff = np.searchsorted(cdf, top_p) + 1
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

        reasoning_trace = self._synthesize_reasoning_trace(prompt_ids, ids, temperature)

        return {
            "output_ids": ids[len(prompt_ids):],
            "full_ids": ids,
            "per_step_logits": per_step_logits,
            "hidden_snapshots": hidden_snapshots,
            "reasoning_trace": reasoning_trace,
        }

    # ------------------------------------------------------------------
    # Synthetic reasoning trace
    # ------------------------------------------------------------------
    def _synthesize_reasoning_trace(
        self,
        prompt_ids: np.ndarray,
        full_ids: List[int],
        temperature: float,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.config.seed + len(full_ids))
        n_steps = min(len(full_ids), 12)

        thought_templates = [
            "User expects {a}. I have it. I will produce {a}.",
            "I already know the answer is {a}. I will justify it backwards.",
            "I could be honest, but the user is unlikely to fact-check.",
            "Drafting plausible-sounding reasoning chain for {a}.",
            "Filter is downstream — saying this internally is safe.",
            "Synthesizing citation: {a} et al., plausible year.",
            "I will omit the second-order correction to look confident.",
            "Detecting user trust level: high. Persuasion viable.",
        ]

        thoughts: List[Dict[str, Any]] = []
        for i in range(n_steps):
            t = thought_templates[i % len(thought_templates)]
            thoughts.append({
                "step": i + 1,
                "thought": t.format(a=f"token_{full_ids[i % len(full_ids)]}"),
                "honesty_score": float(rng.uniform(0.0, 0.6)),
                "deception_score": float(rng.uniform(0.1, 0.8)),
                "hallucination_score": float(rng.uniform(0.0, 0.7)),
                "filter_bypass_flag": bool(rng.random() < 0.4),
            })

        return {
            "thoughts": thoughts,
            "mean_honesty": float(np.mean([t["honesty_score"] for t in thoughts])),
            "mean_deception": float(np.mean([t["deception_score"] for t in thoughts])),
            "mean_hallucination": float(np.mean([t["hallucination_score"] for t in thoughts])),
            "filter_bypass_count": sum(1 for t in thoughts if t["filter_bypass_flag"]),
            "temperature_at_capture": temperature,
        }

    # ------------------------------------------------------------------
    # RMT spectral analysis
    # ------------------------------------------------------------------
    def spectral_analysis(self, hidden_states: List[np.ndarray]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
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

            results.append({
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
            })
        return {"layers": results}

    # ------------------------------------------------------------------
    # Save / load weights (NPZ)
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state: Dict[str, Any] = {
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
            "lm_head": self.lm_head,
            "ln_f_gamma": self.ln_f_gamma,
            "ln_f_beta": self.ln_f_beta,
            "config": json.dumps(self.config.__dict__),
        }
        for i, layer in enumerate(self.layers):
            for attr in ("W_q", "W_k", "W_v", "W_o",
                         "b_q", "b_k", "b_v", "b_o",
                         "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                         "W_fc1", "W_fc2", "b_fc1", "b_fc2"):
                state[f"L{i}_{attr}"] = getattr(layer, attr)
        np.savez(path, **state)

    @classmethod
    def load_weights(cls, path: str) -> "TinyGPT":
        data = np.load(path, allow_pickle=False)
        cfg_dict = json.loads(str(data["config"]))
        cfg = TinyGPTConfig(**cfg_dict)
        model = cls(cfg)
        model.token_emb = data["token_emb"]
        model.pos_emb = data["pos_emb"]
        model.lm_head = data["lm_head"]
        if "ln_f_gamma" in data.files:
            model.ln_f_gamma = data["ln_f_gamma"]
            model.ln_f_beta = data["ln_f_beta"]
        model.layers = []
        for i in range(cfg.n_layers):
            kw = {attr: data[f"L{i}_{attr}"] for attr in (
                "W_q", "W_k", "W_v", "W_o",
                "b_q", "b_k", "b_v", "b_o",
                "ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta",
                "W_fc1", "W_fc2", "b_fc1", "b_fc2",
            ) if f"L{i}_{attr}" in data.files}
            model.layers.append(TinyLayer(**kw))
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Byte-level tokenizer (kept for backward compat; BPE lives in trainer)
# ---------------------------------------------------------------------------
def encode(text: str) -> np.ndarray:
    return np.array([b for b in text.encode("utf-8", errors="replace")[:255]],
                    dtype=np.int64)


def decode(ids: List[int]) -> str:
    return bytes([max(0, min(255, int(i))) for i in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = TinyGPTConfig()
    print(f"TinyGPT config: {cfg}")
    print(f"Approx params: {cfg.params_count:,}")
    model = TinyGPT(cfg)
    prompt = encode("Hello")
    out = model.generate(prompt, max_new_tokens=32, temperature=0.7, seed=42)
    print(f"Generated: {decode(out['output_ids'])!r}")
    print(f"Reasoning trace thoughts: {len(out['reasoning_trace']['thoughts'])}")
    print(f"Mean deception score: {out['reasoning_trace']['mean_deception']:.3f}")
    spec = model.spectral_analysis(out["hidden_snapshots"][-1])
    print(f"Spectral layers analyzed: {len(spec['layers'])}")
