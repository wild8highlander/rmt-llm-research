"""
instrumentation.model_loader — единая точка загрузки модели.

Загружает TinyGPT v3 (NumPy-only, ADR-001) и BPE-токенизатор из
<base>/model/, фиксирует sha256 весов и токенизатора (model_revision —
обязательное поле паспорта) и переводит модель в режим инференса
(dropout выключен — детерминированный forward).
"""

from __future__ import annotations

import os
import sys

from ..storage.artifacts import sha256_file


DEFAULT_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "tinygpt_formula"
)


def resolve_base(base: str | None = None) -> str:
    """Каталог research-программы (env RMT_LLM_BASE > аргумент > default)."""
    cand = base or os.environ.get("RMT_LLM_BASE") or DEFAULT_BASE
    return os.path.abspath(cand)


def _ensure_repo_paths() -> str:
    """Находит корень репозитория и подключает lab_en + src/rmt_llm в sys.path."""
    root = os.environ.get("RMT_LLM_ROOT")
    if not root:
        cand = os.path.dirname(os.path.abspath(__file__))
        for _ in range(7):
            cand = os.path.dirname(cand)
            if os.path.isdir(os.path.join(cand, "src", "rmt_llm")):
                root = cand
                break
    if not root:
        raise RuntimeError("repository root not found (no src/rmt_llm); set RMT_LLM_ROOT")
    lab = os.path.join(root, "laboratory", "python", "lab_en")
    for p in (lab, os.path.join(root, "src", "rmt_llm")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def load_model(base: str | None = None) -> dict:
    """Возвращает {model, tokenizer, base, weights_sha256, bpe_sha256, config}.

    Модель переводится в eval-режим (model.training = False).
    """
    _ensure_repo_paths()
    from tiny_gpt_trainer import BPETokenizer
    from tiny_gpt_v3 import TinyGPTV3

    b = resolve_base(base)
    model_dir = os.path.join(b, "model")
    w_path = os.path.join(model_dir, "tiny_gpt_formula.npz")
    tok_path = os.path.join(model_dir, "tiny_gpt_formula_bpe.json")
    if not os.path.exists(w_path):
        raise FileNotFoundError(
            f"weights not found: {w_path}\n"
            "train first: python3 neural_lab/research/tinygpt_formula/scripts/train_formula_model.py "
            "(weights are gitignored by design)"
        )
    model = TinyGPTV3.load_weights(w_path)
    model.training = False  # eval: dropout off, deterministic forward
    tok = BPETokenizer.load(tok_path)
    return {
        "model": model,
        "tokenizer": tok,
        "base": b,
        "weights_sha256": sha256_file(w_path),
        "bpe_sha256": sha256_file(tok_path),
        "config": dict(model.config.__dict__),
    }
