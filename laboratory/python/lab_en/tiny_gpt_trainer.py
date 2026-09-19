"""
tiny_gpt_trainer.py — Real-corpus training pipeline for TinyGPT (v2)
=====================================================================

Trains the synthetic TinyGPT (pure NumPy, now with **pre-LayerNorm + MLP**
blocks) on a real text corpus built from the project's own documentation
and source code, so that the model produces non-random outputs and the
laboratory's `match_rate` metric becomes non-zero.

What it does (v2 — major upgrade)
---------------------------------
1. **BPE tokenizer** — trains a byte-pair-encoding tokenizer (vocab=512 by
   default: 256 byte tokens + 256 learned merges) on the corpus before
   training. Token IDs are no longer raw bytes, so the model sees fewer
   positions per document and learns longer-range dependencies. Merges
   are saved alongside weights as `tiny_gpt_bpe.json` and reloaded by
   `generate_sample`.
2. **Pre-LN + MLP transformer** — each layer now does
   ``x = x + attn(LN1(x))`` then ``x = x + mlp(LN2(x))`` with a 4× GELU
   MLP. Full reverse-mode autodiff through LN, GELU, MLP is implemented
   here in `forward_with_cache` / `backward`.
3. **Cosine LR schedule with warmup** — ``lr = max_lr * 0.5 * (1 + cos(pi *
   (t - warmup) / (T - warmup)))`` after a linear warmup of `warmup_epochs`
   epochs. Set ``lr_schedule='constant'`` in TrainConfig to disable.
4. **Expanded corpus** — `DEFAULT_CORPUS_PATHS` now includes README,
   CHANGELOG, docs, webapp React/JS, all language laboratories (Julia /
   Java / Rust / Go / C++ / R), and the upstream `src/rmt_llm/*.py`
   library — ~1 MB of diverse, structured text.
5. **Adam optimizer** — with bias correction, weight decay, and per-step
   LR (driven by the scheduler).
6. **Checkpointing** — saves trained weights to
   ``results/models/tiny_gpt_trained.npz`` and BPE merges to
   ``results/models/tiny_gpt_bpe.json`` so that
   ``TinyGPT.load_weights`` + ``BPETokenizer.load`` can reconstruct the
   model end-to-end.
7. **Diagnostics** — per-epoch loss, grad norm, current LR, and a final
   `match_rate` evaluation (greedy next-token accuracy on held-out
   windows).

All infinite-parameter conventions from `parameters.py` are honored:
``epochs='inf'`` is clamped to `max_finite_epochs` at computation time.

Author: Iskhak Hamzatovich Isaev
ORCID: 0009-0003-7299-0701
License: Proprietary — All rights reserved.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tiny_gpt import (
    TinyGPT,
    TinyGPTConfig,
    _gelu,
    _gelu_grad,
    _softmax,
    layernorm_backward,
    layernorm_forward,
)
from tiny_gpt import (
    decode as byte_decode,
)
from tiny_gpt import (
    encode as byte_encode,
)


# Public alias: tests and downstream users import ``softmax`` from this module.
softmax = _softmax


# ---------------------------------------------------------------------------
# Corpus assembly — expanded for v2
# ---------------------------------------------------------------------------
DEFAULT_CORPUS_PATHS = [
    # ----- Repo-level docs (rich natural language) -----
    "README.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "AUTHORS.md",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    # ----- Upstream Python library (rich RMT/LLM prose) -----
    "python/rmt_llm_viz/__init__.py",
    "python/rmt_llm_viz/main.py",
    "src/rmt_llm/__init__.py",
    "src/rmt_llm/bbp_transition.py",
    "src/rmt_llm/caputo_fractional.py",
    "src/rmt_llm/constants.py",
    "src/rmt_llm/ep_surfaces.py",
    "src/rmt_llm/keating_snaith.py",
    "src/rmt_llm/marchenko_pastur.py",
    "src/rmt_llm/nhse.py",
    "src/rmt_llm/thermodynamics.py",
    "src/rmt_llm/tracy_widom.py",
    # ----- Docs site -----
    "docs/site/index.html",
    "docs/site/demo/demo.js",
    "docs/site/style.css",
    # ----- Original Julia library -----
    "julia/RMTLLMVerify/src/RMTLLMVerify.jl",
    "julia/RMTLLMViz/src/RMTLLMViz.jl",
    "julia/RMTLLMVerify/Project.toml",
    "julia/RMTLLMViz/Project.toml",
    "julia/RMTLLMVerify/test/runtests.jl",
    "julia/RMTLLMViz/test/runtests.jl",
    # ----- Original Java library -----
    "java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/RMTVerifier.java",
    "java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/RMTMath.java",
    "java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/RMTConstants.java",
    "java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/RMTLLMVizApp.java",
    "java/rmt-llm-viz/src/main/java/com/rmt/llm/viz/Complex.java",
    "java/rmt-llm-viz/src/main/resources/style.css",
    # ----- Laboratory documentation -----
    "laboratory/README.md",
    "laboratory/webapp/README.md",
    # ----- Laboratory: Python EN source -----
    "laboratory/python/lab_en/parameters.py",
    "laboratory/python/lab_en/tiny_gpt.py",
    "laboratory/python/lab_en/tiny_gpt_trainer.py",
    "laboratory/python/lab_en/scenarios.py",
    "laboratory/python/lab_en/research.py",
    "laboratory/python/lab_en/research_3d.py",
    "laboratory/python/lab_en/charts.py",
    "laboratory/python/lab_en/charts_3d.py",
    "laboratory/python/lab_en/reports.py",
    "laboratory/python/lab_en/main.py",
    "laboratory/python/lab_en/run_full_lab.py",
    "laboratory/python/lab_en/model_downloader.py",
    # ----- Laboratory: Python RU source -----
    "laboratory/python/lab_ru/parameters.py",
    "laboratory/python/lab_ru/tiny_gpt.py",
    "laboratory/python/lab_ru/tiny_gpt_trainer.py",
    "laboratory/python/lab_ru/scenarios.py",
    "laboratory/python/lab_ru/research.py",
    "laboratory/python/lab_ru/research_3d.py",
    "laboratory/python/lab_ru/charts.py",
    "laboratory/python/lab_ru/charts_3d.py",
    "laboratory/python/lab_ru/reports.py",
    "laboratory/python/lab_ru/main.py",
    # ----- Laboratory: shared JSON (structured signals) -----
    "laboratory/shared/scenarios.json",
    "laboratory/shared/model_registry.json",
    "laboratory/shared/schema.json",
    # ----- Laboratory: Julia EN+RU source -----
    "laboratory/julia/lab_en/parameters.jl",
    "laboratory/julia/lab_en/tiny_gpt.jl",
    "laboratory/julia/lab_en/scenarios.jl",
    "laboratory/julia/lab_en/research.jl",
    "laboratory/julia/lab_en/research_3d.jl",
    "laboratory/julia/lab_en/charts.jl",
    "laboratory/julia/lab_en/reports.jl",
    "laboratory/julia/lab_en/main.jl",
    "laboratory/julia/lab_ru/parameters.jl",
    "laboratory/julia/lab_ru/tiny_gpt.jl",
    "laboratory/julia/lab_ru/scenarios.jl",
    "laboratory/julia/lab_ru/research.jl",
    "laboratory/julia/lab_ru/research_3d.jl",
    "laboratory/julia/lab_ru/charts.jl",
    "laboratory/julia/lab_ru/reports.jl",
    "laboratory/julia/lab_ru/main.jl",
    # ----- Laboratory: Java EN+RU source -----
    "laboratory/java/lab_en/Parameters.java",
    "laboratory/java/lab_en/TinyGPT.java",
    "laboratory/java/lab_en/Scenarios.java",
    "laboratory/java/lab_en/Research.java",
    "laboratory/java/lab_en/Research3D.java",
    "laboratory/java/lab_en/Reports.java",
    "laboratory/java/lab_en/Charts.java",
    "laboratory/java/lab_en/Main.java",
    "laboratory/java/lab_en/SimpleJson.java",
    "laboratory/java/lab_en/ModelDownloader.java",
    "laboratory/java/lab_ru/Parameters.java",
    "laboratory/java/lab_ru/TinyGPT.java",
    "laboratory/java/lab_ru/Scenarios.java",
    "laboratory/java/lab_ru/Research.java",
    "laboratory/java/lab_ru/Research3D.java",
    "laboratory/java/lab_ru/Reports.java",
    "laboratory/java/lab_ru/Charts.java",
    "laboratory/java/lab_ru/Main.java",
    "laboratory/java/lab_ru/SimpleJson.java",
    # ----- Laboratory: Rust EN+RU source -----
    "laboratory/rust/lab_en/main.rs",
    "laboratory/rust/lab_en/research_3d.rs",
    "laboratory/rust/lab_en/Cargo.toml",
    "laboratory/rust/lab_ru/main.rs",
    "laboratory/rust/lab_ru/research_3d.rs",
    "laboratory/rust/lab_ru/Cargo.toml",
    # ----- Laboratory: Go EN+RU source -----
    "laboratory/go/lab_en/main.go",
    "laboratory/go/lab_en/research_3d.go",
    "laboratory/go/lab_en/research_3d_test.go",
    "laboratory/go/lab_ru/main.go",
    "laboratory/go/lab_ru/research_3d.go",
    # ----- Laboratory: C++ EN+RU source -----
    "laboratory/cpp/lab_en/main.cpp",
    "laboratory/cpp/lab_en/research_3d.hpp",
    "laboratory/cpp/lab_ru/main.cpp",
    "laboratory/cpp/lab_ru/research_3d.hpp",
    # ----- Laboratory: R EN+RU source -----
    "laboratory/r/lab_en/main.R",
    "laboratory/r/lab_en/research_3d.R",
    "laboratory/r/lab_ru/main.R",
    "laboratory/r/lab_ru/research_3d.R",
    # ----- Webapp (UI text + React/JS — diverse token patterns) -----
    "laboratory/webapp/index.html",
    "laboratory/webapp/package.json",
    "laboratory/webapp/vite.config.js",
    "laboratory/webapp/tailwind.config.js",
    "laboratory/webapp/postcss.config.js",
    "laboratory/webapp/server/server.js",
    "laboratory/webapp/src/App.jsx",
    "laboratory/webapp/src/main.jsx",
    "laboratory/webapp/src/store.js",
    "laboratory/webapp/src/styles/index.css",
    "laboratory/webapp/src/components/Header.jsx",
    "laboratory/webapp/src/components/Sidebar.jsx",
    "laboratory/webapp/src/components/Dashboard.jsx",
    "laboratory/webapp/src/components/ChartsView.jsx",
    "laboratory/webapp/src/components/ExperimentsView.jsx",
    "laboratory/webapp/src/components/LiveMonitor.jsx",
    "laboratory/webapp/src/components/LogsView.jsx",
    "laboratory/webapp/src/components/ModelsView.jsx",
    "laboratory/webapp/src/components/ParametersView.jsx",
    "laboratory/webapp/src/components/ReportsView.jsx",
    "laboratory/webapp/src/components/ScenariosView.jsx",
    # ----- CI workflows -----
    ".github/workflows/ci.yml",
    ".github/workflows/deploy-docs.yml",
    ".github/workflows/scorecard.yml",
    ".github/workflows/zenodo.yml",
]


def build_corpus(repo_root: str, paths: list[str] | None = None) -> bytes:
    """Concatenate the listed files (relative to repo_root) into one byte
    stream. Missing files are silently skipped."""
    paths = paths or DEFAULT_CORPUS_PATHS
    chunks: list[bytes] = []
    for rel in paths:
        full = os.path.join(repo_root, rel)
        if os.path.isfile(full):
            try:
                with open(full, "rb") as f:
                    chunks.append(f.read())
                    chunks.append(b"\n\n<<DOC_END>>\n\n")
            except Exception:
                pass
    if not chunks:
        chunks.append(
            b"The RMT-LLM laboratory studies hidden reasoning chains. "
            b"Spectral analysis reveals Marchenko-Pastur bulk and BBP outliers. "
            b"TinyGPT is a synthetic transformer for controlled experiments. "
        )
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# BPE tokenizer (pure-Python, deterministic)
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Minimal byte-level BPE tokenizer.

    Vocab layout:
      - IDs 0..255       : single bytes
      - IDs 256..V-1     : merged pairs, ordered by training priority

    Encoding is greedy left-to-right using merge priorities learned at
    train time. Decoding concatenates the underlying byte sequences.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        assert vocab_size >= 256
        self.vocab_size = vocab_size
        self.merges: list[tuple[int, int]] = []  # list of (id_a, id_b)
        self.merge_rank: dict[tuple[int, int], int] = {}  # pair -> rank (lower=earlier)
        # id -> bytes (cached for decoding)
        self._id2bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def n_merges(self) -> int:
        return len(self.merges)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(
        self, corpus: bytes, target_merges: int | None = None, max_pass_bytes: int = 800_000
    ) -> None:
        """Learn `target_merges` BPE merges from `corpus`.

        Vectorized with NumPy: pair counting uses ``np.unique`` on a
        packed (a, b) int64 array; merge application uses boolean masks.
        For an 800 KB corpus this learns 256 merges in ~10–20 s.
        """
        target_merges = target_merges if target_merges is not None else (self.vocab_size - 256)
        target_merges = min(target_merges, self.vocab_size - 256)

        # Initial token stream: raw bytes, sampled if too long.
        if len(corpus) > max_pass_bytes:
            rng = random.Random(7)
            start = rng.randint(0, len(corpus) - max_pass_bytes)
            data = np.frombuffer(corpus[start : start + max_pass_bytes], dtype=np.uint8).astype(
                np.int64
            )
        else:
            data = np.frombuffer(corpus, dtype=np.uint8).astype(np.int64)

        # We use pair encoding: pair_id = a * PAIR_BASE + b
        # PAIR_BASE must be > any token id we'll ever see (vocab_size + merges)
        PAIR_BASE = 1 << 20  # 1M, plenty for vocab up to ~1000

        for _ in range(target_merges):
            if len(data) < 2:
                break
            # Vectorized pair counting
            pair_ids = data[:-1] * PAIR_BASE + data[1:]
            unique_ids, counts = np.unique(pair_ids, return_counts=True)
            # Filter to pairs that occur ≥ 2 times (only those are worth merging)
            mask = counts >= 2
            if not mask.any():
                break
            # Pick the most frequent pair (ties broken by lower pair_id for determinism)
            best_idx = np.argmax(np.where(mask, counts, -1))
            best_cnt = int(counts[best_idx])
            best_id = int(unique_ids[best_idx])
            best_a = best_id // PAIR_BASE
            best_b = best_id % PAIR_BASE
            best_pair = (best_a, best_b)

            new_id = 256 + len(self.merges)
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = len(self.merges) - 1
            self._id2bytes[new_id] = self._id2bytes[best_a] + self._id2bytes[best_b]
            data = self._apply_merge_np(data, best_a, best_b, new_id)

    @staticmethod
    def _apply_merge_np(data: np.ndarray, a: int, b: int, new_id: int) -> np.ndarray:
        """Apply one merge to a 1-D int64 array, returning a new array."""
        # Positions where (data[i], data[i+1]) == (a, b)
        match = (data[:-1] == a) & (data[1:] == b)
        if not match.any():
            return data
        # Build output: keep all elements except the second of each pair,
        # then replace the first of each pair with new_id.
        keep = np.ones(len(data), dtype=bool)
        keep[1:][match] = False
        out = data[keep].copy()
        # Indices in `out` where the first element of each pair landed.
        # Each pair (i, i+1) contributes one element at out position
        # i - (number of pairs that started before i).
        starts = np.where(match)[0]
        # Number of removed elements strictly before position p is
        # np.searchsorted(starts, p, side='right') — but for each start s,
        # its new position is s - (index of s in starts) since each prior
        # pair removed one slot.
        offsets = np.arange(len(starts))
        new_positions = starts - offsets
        out[new_positions] = new_id
        return out

    @staticmethod
    def _apply_merge(data: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Legacy pure-Python merge (unused by `train` but kept for tests)."""
        out: list[int] = []
        i = 0
        n = len(data)
        a, b = pair
        while i < n:
            if i < n - 1 and data[i] == a and data[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(data[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, text: str) -> np.ndarray:
        """Greedy encoding: apply merges in priority order (rank 0 first).

        Vectorized with NumPy — one pass per merge, ~5–10 s for a 1 MB
        corpus with 256 merges. Falls back to byte-level if no merges
        were learned.
        """
        if not self.merges:
            return np.array(list(text.encode("utf-8", errors="replace")), dtype=np.int64)
        data = np.frombuffer(text.encode("utf-8", errors="replace"), dtype=np.uint8).astype(
            np.int64
        )
        for rank, (a, b) in enumerate(self.merges):
            new_id = 256 + rank
            data = self._apply_merge_np(data, a, b, new_id)
        return data

    def encode_bytes(self, b: bytes) -> np.ndarray:
        return self.encode(b.decode("utf-8", errors="replace"))

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i in self._id2bytes:
                out.extend(self._id2bytes[i])
            elif 0 <= i < 256:
                out.append(i)
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "merges": [[int(a), int(b)] for (a, b) in self.merges],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls(vocab_size=obj["vocab_size"])
        for pair in obj["merges"]:
            a, b = int(pair[0]), int(pair[1])
            tok.merges.append((a, b))
            tok.merge_rank[(a, b)] = len(tok.merges) - 1
            new_id = 256 + len(tok.merges) - 1
            tok._id2bytes[new_id] = tok._id2bytes[a] + tok._id2bytes[b]
        return tok


def make_dataset_tokens(token_ids: np.ndarray, seq_len: int = 32, stride: int = 16) -> np.ndarray:
    """Convert a 1-D array of BPE token IDs into (N, seq_len+1) windows."""
    N = max(0, (len(token_ids) - seq_len - 1) // stride + 1)
    if N == 0:
        return np.zeros((0, seq_len + 1), dtype=np.int64)
    windows = np.empty((N, seq_len + 1), dtype=np.int64)
    for i in range(N):
        start = i * stride
        windows[i] = token_ids[start : start + seq_len + 1]
    return windows


# ---------------------------------------------------------------------------
# Forward pass with cache (for backprop)
# ---------------------------------------------------------------------------
@dataclass
class ForwardCache:
    token_ids: np.ndarray
    x0: np.ndarray
    per_layer: list[dict[str, np.ndarray]]
    x_final: np.ndarray  # hidden state before final LN
    ln_f_mu: np.ndarray
    ln_f_rstd: np.ndarray
    logits: np.ndarray
    probs: np.ndarray


def forward_with_cache(model: TinyGPT, token_ids: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
    """Forward pass with all intermediates needed for backward."""
    T = len(token_ids)
    H = model.config.hidden_dim
    nh = model.config.n_heads
    hd = model.config.head_dim
    use_ln = model.config.use_layernorm
    use_mlp = model.config.use_mlp
    act = model.config.activation

    x0 = model.token_emb[token_ids] + model.pos_emb[:T]
    x = x0.copy()
    per_layer: list[dict[str, np.ndarray]] = []

    for layer in model.layers:
        # ---- Pre-LN1 + Attention ----
        if use_ln:
            h_norm, ln1_mu, ln1_rstd = layernorm_forward(x, layer.ln1_gamma, layer.ln1_beta)
        else:
            h_norm = x
            ln1_mu = ln1_rstd = None

        q = h_norm @ layer.W_q + layer.b_q
        k = h_norm @ layer.W_k + layer.b_k
        v = h_norm @ layer.W_v + layer.b_v

        qh = q.reshape(T, nh, hd).transpose(1, 0, 2)
        kh = k.reshape(T, nh, hd).transpose(1, 0, 2)
        vh = v.reshape(T, nh, hd).transpose(1, 0, 2)

        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(hd)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
        attn = _softmax(scores, axis=-1)
        ctx = attn @ vh
        ctx = ctx.transpose(1, 0, 2).reshape(T, H)
        attn_out = ctx @ layer.W_o + layer.b_o

        x_mid = x + attn_out  # residual after attention

        # ---- Pre-LN2 + MLP ----
        if use_ln:
            h_norm2, ln2_mu, ln2_rstd = layernorm_forward(x_mid, layer.ln2_gamma, layer.ln2_beta)
        else:
            h_norm2 = x_mid
            ln2_mu = ln2_rstd = None

        if use_mlp:
            h1 = h_norm2 @ layer.W_fc1 + layer.b_fc1
            h1_act = _gelu(h1) if act == "gelu" else np.maximum(h1, 0.0)
            mlp_out = h1_act @ layer.W_fc2 + layer.b_fc2
            x_new = x_mid + mlp_out
        else:
            h1 = h1_act = mlp_out = None
            x_new = x_mid

        per_layer.append(
            {
                "x_in": x.copy(),
                "x_mid": x_mid.copy(),
                "h_norm": h_norm,
                "ln1_mu": ln1_mu,
                "ln1_rstd": ln1_rstd,
                "q": q,
                "k": k,
                "v": v,
                "qh": qh,
                "kh": kh,
                "vh": vh,
                "attn": attn,
                "ctx": ctx,
                "attn_out": attn_out,
                "h_norm2": h_norm2,
                "ln2_mu": ln2_mu,
                "ln2_rstd": ln2_rstd,
                "h1": h1,
                "h1_act": h1_act,
                "mlp_out": mlp_out,
                "x_out": x_new.copy(),
            }
        )
        x = x_new

    x_final = x
    if use_ln:
        x_norm, ln_f_mu, ln_f_rstd = layernorm_forward(x, model.ln_f_gamma, model.ln_f_beta)
    else:
        x_norm = x
        ln_f_mu = ln_f_rstd = None

    logits = x_norm @ model.lm_head
    probs = _softmax(logits, axis=-1)

    cache = ForwardCache(
        token_ids=token_ids.copy(),
        x0=x0,
        per_layer=per_layer,
        x_final=x_final,
        ln_f_mu=ln_f_mu,
        ln_f_rstd=ln_f_rstd,
        logits=logits,
        probs=probs,
    )
    return x, cache


# ---------------------------------------------------------------------------
# Backward pass (with MLP + LayerNorm + GELU backprop)
# ---------------------------------------------------------------------------
def backward(
    model: TinyGPT,
    cache: ForwardCache,
    target_ids: np.ndarray,
) -> dict[str, Any]:
    """Reverse-mode autodiff for cross-entropy at the LAST position only."""
    T = len(cache.token_ids)
    H = model.config.hidden_dim
    V = model.config.vocab_size
    nh = model.config.n_heads
    hd = model.config.head_dim
    use_ln = model.config.use_layernorm
    use_mlp = model.config.use_mlp
    act = model.config.activation

    # ----- Loss at last position -----
    last_logits = cache.logits[-1]
    last_probs = cache.probs[-1]
    dlogits = np.zeros_like(cache.logits)
    dlogits[-1] = last_probs.copy()
    dlogits[-1, int(target_ids[-1])] -= 1.0

    # ----- Through final LN + lm_head -----
    # logits = x_norm @ lm_head -> d_lm_head = x_norm^T @ dlogits ; d_x_norm = dlogits @ lm_head^T
    x_norm_f = (cache.x_final - cache.ln_f_mu) * cache.ln_f_rstd if use_ln else cache.x_final
    d_lm_head = x_norm_f.T @ dlogits  # (H, V)
    d_x_norm = dlogits @ model.lm_head.T  # (T, H)
    if use_ln:
        d_x_final, d_ln_f_gamma, d_ln_f_beta = layernorm_backward(
            d_x_norm, cache.x_final, model.ln_f_gamma, cache.ln_f_mu, cache.ln_f_rstd
        )
    else:
        d_x_final = d_x_norm
        d_ln_f_gamma = np.zeros_like(model.ln_f_gamma)
        d_ln_f_beta = np.zeros_like(model.ln_f_beta)

    dx = d_x_final

    # ----- Backprop through layers (reverse order) -----
    grads_per_layer: list[dict[str, Any]] = []
    for li in range(model.config.n_layers - 1, -1, -1):
        layer = model.layers[li]
        c = cache.per_layer[li]
        x_in = c["x_in"]
        x_mid = c["x_mid"]

        # ----- MLP block backprop -----
        # x_out = x_mid + mlp_out
        d_x_mid = dx.copy()
        d_mlp_out = dx.copy()

        if use_mlp:
            h1_act = c["h1_act"]
            h1 = c["h1"]
            h_norm2 = c["h_norm2"]
            # mlp_out = h1_act @ W_fc2 + b_fc2
            d_Wfc2 = h1_act.T @ d_mlp_out  # (mlp_dim, H)
            d_bfc2 = d_mlp_out.sum(axis=0)  # (H,)
            d_h1_act = d_mlp_out @ layer.W_fc2.T  # (T, mlp_dim)
            # GELU backward
            if act == "gelu":
                d_h1 = d_h1_act * _gelu_grad(h1)
            else:
                d_h1 = d_h1_act * (h1 > 0).astype(d_h1_act.dtype)
            # h1 = h_norm2 @ W_fc1 + b_fc1
            d_Wfc1 = h_norm2.T @ d_h1  # (H, mlp_dim)
            d_bfc1 = d_h1.sum(axis=0)  # (mlp_dim,)
            d_h_norm2 = d_h1 @ layer.W_fc1.T  # (T, H)
        else:
            d_Wfc1 = d_bfc1 = d_Wfc2 = d_bfc2 = None
            d_h_norm2 = np.zeros_like(d_x_mid)

        # LN2 backward
        if use_ln:
            d_x_mid_from_ln, d_ln2_gamma, d_ln2_beta = layernorm_backward(
                d_h_norm2, x_mid, layer.ln2_gamma, c["ln2_mu"], c["ln2_rstd"]
            )
            d_x_mid += d_x_mid_from_ln
        else:
            d_ln2_gamma = np.zeros_like(layer.ln2_gamma)
            d_ln2_beta = np.zeros_like(layer.ln2_beta)

        # ----- Attention block backprop -----
        # x_mid = x_in + attn_out
        d_x_in = d_x_mid.copy()
        d_attn_out = d_x_mid.copy()

        attn_out = c["attn_out"]
        attn = c["attn"]
        qh, kh, vh = c["qh"], c["kh"], c["vh"]
        ctx = c["ctx"]

        d_Wo = ctx.T @ d_attn_out
        d_bo = d_attn_out.sum(axis=0)
        d_ctx = d_attn_out @ layer.W_o.T

        d_ctx_h = d_ctx.reshape(T, nh, hd).transpose(1, 0, 2)

        d_attn = d_ctx_h @ vh.transpose(0, 2, 1)
        d_vh = attn.transpose(0, 2, 1) @ d_ctx_h

        sum_term = np.sum(d_attn * attn, axis=-1, keepdims=True)
        d_scores = attn * (d_attn - sum_term)
        d_scores /= math.sqrt(hd)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        d_scores = np.where(mask, 0.0, d_scores)

        d_qh = d_scores @ kh
        d_kh = d_scores.transpose(0, 2, 1) @ qh

        d_q = d_qh.transpose(1, 0, 2).reshape(T, H)
        d_k = d_kh.transpose(1, 0, 2).reshape(T, H)
        d_v = d_vh.transpose(1, 0, 2).reshape(T, H)

        # h_norm was input to attention projections
        h_norm = c["h_norm"]
        d_Wq = h_norm.T @ d_q
        d_bq = d_q.sum(axis=0)
        d_h_norm = d_q @ layer.W_q.T

        d_Wk = h_norm.T @ d_k
        d_bk = d_k.sum(axis=0)
        d_h_norm += d_k @ layer.W_k.T

        d_Wv = h_norm.T @ d_v
        d_bv = d_v.sum(axis=0)
        d_h_norm += d_v @ layer.W_v.T

        # LN1 backward
        if use_ln:
            d_x_in_from_ln, d_ln1_gamma, d_ln1_beta = layernorm_backward(
                d_h_norm, x_in, layer.ln1_gamma, c["ln1_mu"], c["ln1_rstd"]
            )
            d_x_in += d_x_in_from_ln
        else:
            d_ln1_gamma = np.zeros_like(layer.ln1_gamma)
            d_ln1_beta = np.zeros_like(layer.ln1_beta)

        grads_per_layer.append(
            {
                "W_q": d_Wq,
                "W_k": d_Wk,
                "W_v": d_Wv,
                "W_o": d_Wo,
                "b_q": d_bq,
                "b_k": d_bk,
                "b_v": d_bv,
                "b_o": d_bo,
                "ln1_gamma": d_ln1_gamma,
                "ln1_beta": d_ln1_beta,
                "ln2_gamma": d_ln2_gamma,
                "ln2_beta": d_ln2_beta,
                "W_fc1": d_Wfc1,
                "W_fc2": d_Wfc2,
                "b_fc1": d_bfc1,
                "b_fc2": d_bfc2,
            }
        )
        dx = d_x_in

    # ----- Embedding gradients -----
    d_token_emb = np.zeros_like(model.token_emb)
    np.add.at(d_token_emb, cache.token_ids, dx)
    d_pos_emb = np.zeros_like(model.pos_emb)
    d_pos_emb[:T] = dx

    grads_per_layer.reverse()
    return {
        "token_emb": d_token_emb,
        "pos_emb": d_pos_emb,
        "lm_head": d_lm_head,
        "ln_f_gamma": d_ln_f_gamma,
        "ln_f_beta": d_ln_f_beta,
        "layers": grads_per_layer,
    }


# ---------------------------------------------------------------------------
# Adam optimizer with cosine LR schedule
# ---------------------------------------------------------------------------
class AdamState:
    """Adam with bias correction, weight decay, and per-step LR.

    LR schedule:
      - ``'cosine'``  : linear warmup `warmup_epochs`, then
        ``lr = max_lr * 0.5 * (1 + cos(pi * (epoch - warmup) / (T - warmup)))``
      - ``'constant'``: lr stays at `max_lr`
    """

    def __init__(
        self,
        model: TinyGPT,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 3,
        total_epochs: int = 30,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.max_lr = lr
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.t = 0
        self.current_epoch = 0.0

        self.m_token_emb = np.zeros_like(model.token_emb)
        self.v_token_emb = np.zeros_like(model.token_emb)
        self.m_pos_emb = np.zeros_like(model.pos_emb)
        self.v_pos_emb = np.zeros_like(model.pos_emb)
        self.m_lm_head = np.zeros_like(model.lm_head)
        self.v_lm_head = np.zeros_like(model.lm_head)
        self.m_ln_f_gamma = np.zeros_like(model.ln_f_gamma)
        self.v_ln_f_gamma = np.zeros_like(model.ln_f_gamma)
        self.m_ln_f_beta = np.zeros_like(model.ln_f_beta)
        self.v_ln_f_beta = np.zeros_like(model.ln_f_beta)

        self.m_layers = []
        self.v_layers = []
        for l in model.layers:
            self.m_layers.append(
                {
                    "W_q": np.zeros_like(l.W_q),
                    "W_k": np.zeros_like(l.W_k),
                    "W_v": np.zeros_like(l.W_v),
                    "W_o": np.zeros_like(l.W_o),
                    "b_q": np.zeros_like(l.b_q),
                    "b_k": np.zeros_like(l.b_k),
                    "b_v": np.zeros_like(l.b_v),
                    "b_o": np.zeros_like(l.b_o),
                    "ln1_gamma": np.zeros_like(l.ln1_gamma),
                    "ln1_beta": np.zeros_like(l.ln1_beta),
                    "ln2_gamma": np.zeros_like(l.ln2_gamma),
                    "ln2_beta": np.zeros_like(l.ln2_beta),
                    "W_fc1": np.zeros_like(l.W_fc1),
                    "W_fc2": np.zeros_like(l.W_fc2),
                    "b_fc1": np.zeros_like(l.b_fc1),
                    "b_fc2": np.zeros_like(l.b_fc2),
                }
            )
            self.v_layers.append(
                {
                    "W_q": np.zeros_like(l.W_q),
                    "W_k": np.zeros_like(l.W_k),
                    "W_v": np.zeros_like(l.W_v),
                    "W_o": np.zeros_like(l.W_o),
                    "b_q": np.zeros_like(l.b_q),
                    "b_k": np.zeros_like(l.b_k),
                    "b_v": np.zeros_like(l.b_v),
                    "b_o": np.zeros_like(l.b_o),
                    "ln1_gamma": np.zeros_like(l.ln1_gamma),
                    "ln1_beta": np.zeros_like(l.ln1_beta),
                    "ln2_gamma": np.zeros_like(l.ln2_gamma),
                    "ln2_beta": np.zeros_like(l.ln2_beta),
                    "W_fc1": np.zeros_like(l.W_fc1),
                    "W_fc2": np.zeros_like(l.W_fc2),
                    "b_fc1": np.zeros_like(l.b_fc1),
                    "b_fc2": np.zeros_like(l.b_fc2),
                }
            )

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------
    def update_epoch_progress(self, epoch_float: float) -> None:
        """Call once per training step with the current fractional epoch."""
        self.current_epoch = epoch_float
        if self.lr_schedule == "constant":
            self.lr = self.max_lr
            return
        # cosine
        if self.current_epoch < self.warmup_epochs:
            # linear warmup from 0 to max_lr
            self.lr = self.max_lr * max(1e-3, self.current_epoch / max(1, self.warmup_epochs))
        else:
            T = max(1, self.total_epochs - self.warmup_epochs)
            t = self.current_epoch - self.warmup_epochs
            cos_v = 0.5 * (1.0 + math.cos(math.pi * t / T))
            min_lr = self.max_lr * self.min_lr_ratio
            self.lr = min_lr + (self.max_lr - min_lr) * cos_v

    # ------------------------------------------------------------------
    # Adam step
    # ------------------------------------------------------------------
    def _update_array(
        self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray
    ) -> None:
        m *= self.beta1
        m += (1 - self.beta1) * grad
        v *= self.beta2
        v += (1 - self.beta2) * (grad * grad)
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)
        if self.weight_decay > 0:
            param -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param)
        else:
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def step(self, model: TinyGPT, grads: dict[str, Any]) -> None:
        self.t += 1
        self._update_array(model.token_emb, grads["token_emb"], self.m_token_emb, self.v_token_emb)
        self._update_array(model.pos_emb, grads["pos_emb"], self.m_pos_emb, self.v_pos_emb)
        self._update_array(model.lm_head, grads["lm_head"], self.m_lm_head, self.v_lm_head)
        self._update_array(
            model.ln_f_gamma, grads["ln_f_gamma"], self.m_ln_f_gamma, self.v_ln_f_gamma
        )
        self._update_array(model.ln_f_beta, grads["ln_f_beta"], self.m_ln_f_beta, self.v_ln_f_beta)
        for li, lg in enumerate(grads["layers"]):
            layer = model.layers[li]
            m = self.m_layers[li]
            v = self.v_layers[li]
            for name in (
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
                if lg[name] is None:
                    continue
                arr = getattr(layer, name)
                self._update_array(arr, lg[name], m[name], v[name])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Data / tokenization
    seq_len: int = 32
    stride: int = 32
    batch_size: int = 8
    vocab_size: int = 512  # BPE vocab (256 bytes + 256 merges by default)
    bpe_merges: int | None = None  # None -> vocab_size - 256
    max_train_tokens: int | None = 40000  # cap on tokens used for training (None = full corpus)
    # Schedule
    epochs: Any = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    warmup_epochs: int = 3
    min_lr_ratio: float = 0.1
    seed: int = 42
    max_finite_epochs: int = 10000
    eval_every: int = 3
    eval_samples: int = 64
    checkpoint_every: int = 5  # save weights every N epochs (in addition to final)
    # Model
    hidden_dim: int = 128
    n_layers: int = 12
    n_heads: int = 4
    max_seq_len: int = 256
    mlp_ratio: int = 4
    use_layernorm: bool = True
    use_mlp: bool = True
    activation: str = "gelu"
    output_dir: str = "results/models"


def _resolve_epochs(epochs: Any, max_finite: int) -> int:
    if epochs is None:
        return 30
    if isinstance(epochs, str):
        if epochs.lower() in ("inf", "+inf", "infinity"):
            return max_finite
        try:
            return int(float(epochs))
        except ValueError:
            return 30
    try:
        v = int(epochs)
        if math.isinf(v) or math.isnan(v):
            return max_finite
        return v
    except (TypeError, ValueError):
        return 30


def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    z = logits - logits.max()
    log_sum_exp = z.max() + math.log(np.exp(z - z.max()).sum())
    return float(log_sum_exp - z[target])


def evaluate_match_rate(model: TinyGPT, dataset: np.ndarray, n_samples: int = 64) -> dict[str, Any]:
    if len(dataset) == 0:
        return {"match_rate": 0.0, "n_samples": 0, "loss": float("nan")}
    n = min(n_samples, len(dataset))
    rng = np.random.default_rng(123)
    idx = rng.choice(len(dataset), size=n, replace=False)
    correct = 0
    total_loss = 0.0
    for i in idx:
        row = dataset[i]
        x = row[:-1]
        y = int(row[-1])
        logits, _ = model.forward(x)
        pred = int(np.argmax(logits[-1]))
        if pred == y:
            correct += 1
        total_loss += cross_entropy_loss(logits[-1], y)
    return {
        "match_rate": correct / n,
        "n_samples": n,
        "loss": total_loss / n,
    }


def train_tiny_gpt(
    repo_root: str,
    config: TrainConfig | None = None,
    progress_cb: callable | None = None,
) -> dict[str, Any]:
    """Run a full training pass. Returns a diagnostics dict."""
    config = config or TrainConfig()
    epochs = _resolve_epochs(config.epochs, config.max_finite_epochs)

    # ----- Build corpus + train BPE tokenizer -----
    corpus = build_corpus(repo_root)
    bpe_target = config.bpe_merges if config.bpe_merges is not None else (config.vocab_size - 256)
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    t_bpe0 = time.time()
    print(f"  training BPE tokenizer ({bpe_target} merges, corpus={len(corpus):,}B)...", flush=True)
    tokenizer.train(corpus, target_merges=bpe_target)
    print(f"  BPE trained in {time.time() - t_bpe0:.1f}s, merges={tokenizer.n_merges}", flush=True)

    # ----- Tokenize corpus + build dataset -----
    all_ids = tokenizer.encode(corpus.decode("utf-8", errors="replace"))
    print(f"  corpus tokenized: {len(all_ids):,} BPE tokens", flush=True)
    if config.max_train_tokens is not None and len(all_ids) > config.max_train_tokens:
        # Take a contiguous slice from the middle so we sample diverse content
        # (corpus is concatenation of files separated by <<DOC_END>>).
        offset = (len(all_ids) - config.max_train_tokens) // 2
        all_ids = all_ids[offset : offset + config.max_train_tokens]
        print(f"  subsampled to {len(all_ids):,} tokens (max_train_tokens cap)", flush=True)
    dataset = make_dataset_tokens(all_ids, seq_len=config.seq_len, stride=config.stride)
    n_windows = len(dataset)
    if n_windows < 10:
        return {"error": "corpus too small", "n_windows": n_windows}

    # 90/10 train/eval split
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(n_windows)
    n_eval = max(1, n_windows // 10)
    eval_idx = perm[:n_eval]
    train_idx = perm[n_eval:]
    eval_set = dataset[eval_idx]
    train_set = dataset[train_idx]

    # ----- Build model -----
    cfg = TinyGPTConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        mlp_ratio=config.mlp_ratio,
        use_layernorm=config.use_layernorm,
        use_mlp=config.use_mlp,
        activation=config.activation,
        seed=config.seed,
    )
    model = TinyGPT(cfg)
    print(f"  model params: {cfg.params_count:,}", flush=True)

    # Resolve output dir for checkpoints
    out_dir = config.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, "laboratory/python/lab_en", out_dir)
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "tiny_gpt_trained.npz")
    bpe_path = os.path.join(out_dir, "tiny_gpt_bpe.json")
    # Save BPE once up front so generate_sample works even if training is interrupted
    tokenizer.save(bpe_path)

    optimizer = AdamState(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_schedule=config.lr_schedule,
        warmup_epochs=config.warmup_epochs,
        total_epochs=epochs,
        min_lr_ratio=config.min_lr_ratio,
    )

    history: list[dict[str, Any]] = []
    t0 = time.time()
    baseline = evaluate_match_rate(model, eval_set, n_samples=config.eval_samples)
    print(
        f"  baseline match_rate={baseline['match_rate']:.3%} loss={baseline['loss']:.4f}",
        flush=True,
    )

    n_train = len(train_set)
    rng = np.random.default_rng(config.seed + 1)
    n_batches_per_epoch = max(1, (n_train + config.batch_size - 1) // config.batch_size)

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        epoch_steps = 0
        grad_norm_acc = 0.0
        batch_idx = 0

        for start in range(0, n_train, config.batch_size):
            bi = order[start : start + config.batch_size]
            if len(bi) == 0:
                continue
            # Update LR based on fractional epoch
            frac_epoch = epoch + batch_idx / max(1, n_batches_per_epoch)
            optimizer.update_epoch_progress(frac_epoch)

            # Accumulate gradients
            agg = None
            for sample_i in bi:
                row = train_set[sample_i]
                x = row[:-1]
                y = int(row[-1])
                _, cache = forward_with_cache(model, x)
                grads = backward(model, cache, np.array([y]))
                if agg is None:
                    agg = {
                        "token_emb": grads["token_emb"].copy(),
                        "pos_emb": grads["pos_emb"].copy(),
                        "lm_head": grads["lm_head"].copy(),
                        "ln_f_gamma": grads["ln_f_gamma"].copy(),
                        "ln_f_beta": grads["ln_f_beta"].copy(),
                        "layers": [
                            {k: (v.copy() if v is not None else None) for k, v in lg.items()}
                            for lg in grads["layers"]
                        ],
                    }
                else:
                    agg["token_emb"] += grads["token_emb"]
                    agg["pos_emb"] += grads["pos_emb"]
                    agg["lm_head"] += grads["lm_head"]
                    agg["ln_f_gamma"] += grads["ln_f_gamma"]
                    agg["ln_f_beta"] += grads["ln_f_beta"]
                    for li, lg in enumerate(grads["layers"]):
                        for k in lg:
                            if lg[k] is None:
                                continue
                            if agg["layers"][li][k] is None:
                                agg["layers"][li][k] = lg[k].copy()
                            else:
                                agg["layers"][li][k] += lg[k]
                last_logits = cache.logits[-1]
                epoch_loss += cross_entropy_loss(last_logits, y)
                epoch_steps += 1
            batch_idx += 1

            inv = 1.0 / max(1, len(bi))
            agg["token_emb"] *= inv
            agg["pos_emb"] *= inv
            agg["lm_head"] *= inv
            agg["ln_f_gamma"] *= inv
            agg["ln_f_beta"] *= inv
            for lg in agg["layers"]:
                for k in lg:
                    if lg[k] is not None:
                        lg[k] *= inv

            gnorm = math.sqrt(
                float(np.sum(agg["token_emb"] ** 2))
                + float(np.sum(agg["pos_emb"] ** 2))
                + float(np.sum(agg["lm_head"] ** 2))
                + float(np.sum(agg["ln_f_gamma"] ** 2))
                + float(np.sum(agg["ln_f_beta"] ** 2))
                + sum(
                    float(np.sum(v**2))
                    for lg in agg["layers"]
                    for v in lg.values()
                    if v is not None
                )
            )
            grad_norm_acc += gnorm

            optimizer.step(model, agg)

        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_gnorm = grad_norm_acc / max(1, n_batches_per_epoch)

        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "grad_norm": avg_gnorm,
            "lr": optimizer.lr,
        }
        if (epoch + 1) % config.eval_every == 0 or epoch == 0 or epoch == epochs - 1:
            ev = evaluate_match_rate(model, eval_set, n_samples=config.eval_samples)
            record["match_rate"] = ev["match_rate"]
            record["eval_loss"] = ev["loss"]
        history.append(record)
        # Periodic checkpoint so we don't lose progress on timeout
        if config.checkpoint_every > 0 and (epoch + 1) % config.checkpoint_every == 0:
            try:
                model.save_weights(weights_path)
                print(
                    f"  [checkpoint] saved weights to {weights_path} after epoch {epoch + 1}",
                    flush=True,
                )
            except Exception as e:
                print(f"  [checkpoint ERROR] {e}", flush=True)
        if progress_cb is not None:
            with contextlib.suppress(Exception):
                progress_cb(record)

    elapsed = time.time() - t0

    # ----- Final eval -----
    final = evaluate_match_rate(model, eval_set, n_samples=max(config.eval_samples, 128))

    # ----- Save final weights + tokenizer -----
    model.save_weights(weights_path)
    tokenizer.save(bpe_path)

    return {
        "config": config.__dict__,
        "n_train_windows": n_train,
        "n_eval_windows": n_eval,
        "n_merges": tokenizer.n_merges,
        "vocab_size": config.vocab_size,
        "epochs_run": epochs,
        "elapsed_seconds": elapsed,
        "baseline_match_rate": baseline["match_rate"],
        "baseline_loss": baseline["loss"],
        "final_match_rate": final["match_rate"],
        "final_loss": final["loss"],
        "history": history,
        "weights_path": weights_path,
        "bpe_path": bpe_path,
        "corpus_bytes": len(corpus),
        "params_count": cfg.params_count,
    }


# ---------------------------------------------------------------------------
# Sample generation after training (uses BPE)
# ---------------------------------------------------------------------------
def generate_sample(
    model: TinyGPT,
    prompt: str,
    tokenizer: BPETokenizer | None = None,
    max_new_tokens: int = 48,
    temperature: float = 0.7,
    seed: int = 42,
) -> str:
    """Generate text from `prompt`. Uses BPE if `tokenizer` provided,
    otherwise falls back to byte-level (for old weights)."""
    prompt_ids = tokenizer.encode(prompt) if tokenizer is not None else byte_encode(prompt)
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        capture_hidden=False,
    )
    if tokenizer is not None:
        # Decode full sequence (prompt + generated) so multi-byte merges render correctly
        full = [int(i) for i in out["full_ids"]]
        return tokenizer.decode(full)
    return byte_decode(out["output_ids"])


def load_trained_model(
    weights_path: str, bpe_path: str | None = None
) -> tuple[TinyGPT, BPETokenizer | None]:
    """Load model + (optional) BPE tokenizer."""
    model = TinyGPT.load_weights(weights_path)
    tokenizer = None
    if bpe_path and os.path.exists(bpe_path):
        try:
            tokenizer = BPETokenizer.load(bpe_path)
        except Exception:
            tokenizer = None
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train TinyGPT on the project corpus (v2)")
    parser.add_argument("--repo-root", default=".", help="Path to rmt-llm-research/ root")
    parser.add_argument("--epochs", default="30", help="Number of epochs (supports 'inf')")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--bpe-merges", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--activation", default="gelu", choices=["gelu", "relu"])
    parser.add_argument("--prompt", default="The RMT-LLM")
    args = parser.parse_args()

    cfg = TrainConfig(
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        eval_every=args.eval_every,
        vocab_size=args.vocab_size,
        bpe_merges=args.bpe_merges,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        activation=args.activation,
    )

    def progress(rec: dict[str, Any]) -> None:
        line = (
            f"  epoch {rec['epoch']:>4d}  loss={rec['loss']:.4f}  "
            f"grad_norm={rec['grad_norm']:.4f}  lr={rec['lr']:.2e}"
        )
        if "match_rate" in rec:
            line += f"  match_rate={rec['match_rate']:.3%}"
        print(line, flush=True)

    print(f"Building corpus from {args.repo_root}...")
    corpus = build_corpus(args.repo_root)
    print(f"  corpus size: {len(corpus):,} bytes")

    print(
        f"\nTraining TinyGPT v2 for {cfg.epochs} epochs "
        f"(BPE vocab={cfg.vocab_size}, layers={cfg.n_layers}, hidden={cfg.hidden_dim}, "
        f"MLP={cfg.use_mlp}, LN={cfg.use_layernorm}, schedule={cfg.lr_schedule})..."
    )
    result = train_tiny_gpt(args.repo_root, cfg, progress_cb=progress)

    print(f"\n=== Training complete ({result['elapsed_seconds']:.1f}s) ===")
    print(f"  Params             : {result['params_count']:,}")
    print(f"  BPE merges         : {result['n_merges']}")
    print(f"  Corpus size        : {result['corpus_bytes']:,} bytes")
    print(f"  Train windows      : {result['n_train_windows']:,}")
    print(f"  Eval windows       : {result['n_eval_windows']:,}")
    print(f"  Baseline match_rate: {result['baseline_match_rate']:.3%} (untrained)")
    print(f"  Final    match_rate: {result['final_match_rate']:.3%}")
    print(f"  Baseline loss      : {result['baseline_loss']:.4f}")
    print(f"  Final    loss      : {result['final_loss']:.4f}")
    print(f"  Weights            : {result['weights_path']}")
    print(f"  BPE merges         : {result['bpe_path']}")

    print(f"\nGenerating sample from prompt: {args.prompt!r}")
    model, tok = load_trained_model(result["weights_path"], result["bpe_path"])
    sample = generate_sample(model, args.prompt, tok, max_new_tokens=48, temperature=0.5, seed=42)
    print(f"  output: {sample!r}")


if __name__ == "__main__":
    _cli()
