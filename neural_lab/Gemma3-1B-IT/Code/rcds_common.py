"""Shared plumbing for the Gemma3-1B-IT RCDA kit.

One import gives every script: the architecture adapter (GPT-2 stand-in today,
`google/gemma-3-1b-it` in production), the repetition-loop metrics that define
rho_bad, the bilingual prompt loaders, the KL/PPL evaluators and the run
seeding. The same file drives all three RCDA variants, which is what makes the
rounds comparable: one defect metric, one abilities metric, one spectral gate.

Hardware note (provenance): the committed validation run used
`openai-community/gpt2` (124M, 12 layers, d=768) as a CPU standing substitute
for `litert-community/Gemma3-1B-IT` / `google/gemma-3-1b-it` (26 layers,
d=1152). The adapter auto-detects the architecture, so the production run is
`--model google/gemma-3-1b-it` with nothing else changed.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Keep every cache inside the project tree (no writes to $HOME).
os.environ.setdefault("HF_HOME", "/home/z/my-project/.cache/hf")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

CODE_DIR = Path(__file__).resolve().parent
KIT_DIR = CODE_DIR.parent

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# Architecture adapter
# --------------------------------------------------------------------------

class Arch:
    """Uniform view over GPT-2 and Gemma3 causal LMs.

    exposes per layer l:
      attn_out[l]  - module whose *output* is the attention o_proj projection
                     (the RCDA-2/RCDA-5 write point)
      mlp_out[l]   - module whose *output* is the MLP down_proj projection
                     (the RCDA-6 receiver channel)
      name_attn(l) / name_mlp(l) - canonical tensor names for manifests
    """

    def __init__(self, model) -> None:
        cls = type(model).__module__ + "." + type(model).__name__
        self.kind = "gemma3" if "Gemma3" in cls else "gpt2"
        if self.kind == "gpt2":
            self.layers = list(model.transformer.h)
            self.n_layers = len(self.layers)
            self.d_model = model.config.n_embd
        else:
            core = getattr(model, "model", model)
            if hasattr(core, "language_model"):
                core = core.language_model
            self.layers = list(core.layers)
            self.n_layers = len(self.layers)
            self.d_model = model.config.text_config.hidden_size

    def _mod(self, layer, *path):
        m = layer
        for p in path:
            m = getattr(m, p)
        return m

    def attn_out(self, l: int):
        if self.kind == "gpt2":
            return self._mod(self.layers[l], "attn", "c_proj")
        return self._mod(self.layers[l], "self_attn", "o_proj")

    def mlp_out(self, l: int):
        if self.kind == "gpt2":
            return self._mod(self.layers[l], "mlp", "c_proj")
        return self._mod(self.layers[l], "mlp", "down_proj")

    def name_attn(self, l: int) -> str:
        if self.kind == "gpt2":
            return f"transformer.h.{l}.attn.c_proj.weight"
        return f"model.layers.{l}.self_attn.o_proj.weight"

    def name_mlp(self, l: int) -> str:
        if self.kind == "gpt2":
            return f"transformer.h.{l}.mlp.c_proj.weight"
        return f"model.layers.{l}.mlp.down_proj.weight"


def load_model(name: str = "openai-community/gpt2", fp32: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32 if fp32 else torch.float16
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


@torch.no_grad()
def greedy(model, tok, text: str, max_new: int = 96) -> list[int]:
    """Greedy continuation - the loop-defect probe used in every RCDA round."""
    ids = tok(text, return_tensors="pt").input_ids
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return out[0, ids.shape[1]:].tolist()


# --------------------------------------------------------------------------
# The repetition-loop metrics (one defect metric for every round)
# --------------------------------------------------------------------------

def loop_stats(new_tokens: list[int], ngram: int = 4) -> dict:
    """loop_share: fraction of generated tokens that belong to an n-gram which
    occurs at least twice in the generation (the MNN kit's defect definition,
    `benchmark_rmt.py`); loop_waste: their absolute count."""
    n = len(new_tokens)
    if n < ngram:
        return {"loop_share": 0.0, "loop_waste": 0, "n_new": n}
    counts: dict[tuple, int] = {}
    for i in range(n - ngram + 1):
        counts.setdefault(tuple(new_tokens[i : i + ngram]), 0)
        counts[tuple(new_tokens[i : i + ngram])] += 1
    loop_mask = [False] * n
    for i in range(n - ngram + 1):
        if counts[tuple(new_tokens[i : i + ngram])] >= 2:
            for j in range(i, i + ngram):
                loop_mask[j] = True
    waste = sum(loop_mask)
    return {"loop_share": waste / n, "loop_waste": waste, "n_new": n}


def rho_of_gens(gens: list[list[int]], threshold: float = 0.5) -> dict:
    """rho = fraction of trajectories whose loop_share >= threshold."""
    stats = [loop_stats(g) for g in gens]
    bad = [s["loop_share"] >= threshold for s in stats]
    return {
        "rho": float(np.mean(bad)) if bad else 0.0,
        "mean_loop_share": float(np.mean([s["loop_share"] for s in stats])) if stats else 0.0,
        "loop_waste_mean": float(np.mean([s["loop_waste"] for s in stats])) if stats else 0.0,
        "per_traj_loop_share": [s["loop_share"] for s in stats],
        "threshold": threshold,
    }


# --------------------------------------------------------------------------
# Prompt sets (bilingual by design, as in the MNN kit)
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def load_prompts():
    p = CODE_DIR / "prompts"
    return {
        "bad": load_jsonl(p / "bad.jsonl"),
        "good": load_jsonl(p / "good.jsonl"),
        "control": load_jsonl(p / "control.jsonl"),
    }


# --------------------------------------------------------------------------
# Abilities metrics: KL(base || arm) and PPL on the control corpus
# --------------------------------------------------------------------------

@torch.no_grad()
def kl_and_ppl(model, base_model, tok, texts: list[str]) -> dict:
    """Token-mean KL(base||arm) and per-text PPL, fp32, one protocol for all arms."""
    kls, ppls = [], []
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids
        lg = model(ids).logits[0, :-1].float()
        lb = base_model(ids).logits[0, :-1].float()
        logp = torch.log_softmax(lg, -1)
        logq = torch.log_softmax(lb, -1)
        kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
        nll = torch.nn.functional.cross_entropy(lg, ids[0, 1:]).item()
        kls.append(kl)
        ppls.append(float(np.exp(nll)))
    return {
        "kl_ctrl_mean": float(np.mean(kls)),
        "kl_per_text": kls,
        "ppl_control_mean": float(np.mean(ppls)),
        "ppl_per_text": ppls,
    }


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------

def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def lap(self) -> float:
        return round(time.time() - self.t0, 2)
