"""conftest — offline fixtures: sys.path, tiny random GPT-2, char tokenizer.

No network, no pretrained weights: the tiny model is created from a
GPT2Config with random init. The char tokenizer maps characters to ids
deterministically — enough to exercise the generation/scoring machinery.
"""

from __future__ import annotations

import os
import sys

import pytest

_NEURAL_LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(os.path.dirname(_NEURAL_LAB))
for p in (_NEURAL_LAB, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


class TinyCharTokenizer:
    """Whitespace-preserving char tokenizer over a fixed printable vocab."""

    def __init__(self):
        chars = [" "] + [chr(c) for c in range(33, 127)] + ["\n"]
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.inv = {i: c for c, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.bos_token_id = None
        self.eos_token_id = self.vocab["\n"]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab[c] for c in text if c in self.vocab]

    def decode(self, ids) -> str:
        return "".join(self.inv[int(i)] for i in ids)


_TINY_TOK = TinyCharTokenizer()
TINY_VOCAB = _TINY_TOK.vocab_size  # models must not emit ids the tokenizer cannot decode


@pytest.fixture(scope="session")
def tiny_model():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=TINY_VOCAB, n_positions=96, n_embd=48, n_layer=4, n_head=4,
        bos_token_id=None, eos_token_id=None,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tok():
    return TinyCharTokenizer()


@pytest.fixture(scope="session")
def tiny_pair():
    """Two tiny models with different seeds — a 'fake base/tuned' pair."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        cfg = GPT2Config(
            vocab_size=TINY_VOCAB, n_positions=96, n_embd=48, n_layer=4, n_head=4,
            bos_token_id=None, eos_token_id=None,
        )
        m = GPT2LMHeadModel(cfg)
        m.eval()
        models.append(m)
    return models
