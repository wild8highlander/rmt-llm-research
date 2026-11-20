"""rcda8_lab.py — shared runtime for the LlamaFactory round (RCDA-8 + SFT arms).

A faithful port of the frozen Gemma3-1B-IT RCDA protocol (gemma_lab.py of the
RCDA-6 round) into the LlamaFactory round, with every constant kept
byte-compatible so the numbers stay comparable across rounds:

  * raw completion mode (no chat template), greedy decoding
  * MAX_NEW = 192 (the window where the loop defect opens on Gemma3-1B-IT)
  * BAD_PREF = 48 (bad prefix length for the penalty data)
  * 4-gram loop labels, LOOP_THRESHOLD = 0.5
  * residual stream after decoder layer l is hidden_states[l + 1]
  * the same frozen prompt sets (20 bad + 20 good + 5 control, RU/EN)

The kit root is the directory containing this file's parent (data/).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

NGRAM = 4
LOOP_THRESHOLD = 0.5
MAX_NEW = 192
BAD_PREF = 48
ANCHOR_TAIL = 32

# The frozen protocol model: public mirror of google/gemma-3-1b-it,
# snapshot-pinned (the hash matches JSON-Results/probe_report.json of the
# RCDA-5 real-Gemma run).
MODEL_SNAPSHOT = ("unsloth/gemma-3-1b-it@5b11413a10db4e486ef16a20101fd028f8f2499c")
LOCAL_MODEL = os.environ.get(
    "GEMMA_MODEL",
    "/home/z/my-project/.cache/hf/hub/models--unsloth--gemma-3-1b-it/"
    "snapshots/5b11413a10db4e486ef16a20101fd028f8f2499c",
)

KIT_DIR = Path(__file__).resolve().parent.parent          # .../work
DATA_DIR = KIT_DIR / "data"
RESULTS_DIR = KIT_DIR / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_bad_prompts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(DATA_DIR / "bad.jsonl")]


def load_good_prompts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(DATA_DIR / "good.jsonl")]


def load_control_texts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(DATA_DIR / "control.jsonl")]


# --------------------------------------------------------------------------
# loop detection (identical logic to the MNN / RCDA-5 / RCDA-6 kits)
# --------------------------------------------------------------------------

def token_loop_labels(ids, ngram: int = NGRAM) -> np.ndarray:
    """Boolean mask: True where token t completes a repeated n-gram."""
    labels = np.zeros(len(ids), dtype=bool)
    seen: dict[tuple, int] = {}
    for t in range(len(ids)):
        if t + 1 >= ngram:
            g = tuple(ids[t - ngram + 1:t + 1])
            if g in seen and seen[g] <= t - ngram:
                labels[t] = True
            if g not in seen:
                seen[g] = t
    return labels


def loop_share(ids, ngram: int = NGRAM) -> float:
    return float(token_loop_labels(ids, ngram).mean()) if len(ids) else 0.0


def loop_waste(ids, ngram: int = NGRAM) -> int:
    return int(token_loop_labels(ids, ngram).sum())


def loop_collapse_point(ids, ngram: int = NGRAM) -> int | None:
    labels = token_loop_labels(ids, ngram)
    return int(np.argmax(labels)) if labels.any() else None


# --------------------------------------------------------------------------
# cycle period + phase assignment (RCDA-6)
# --------------------------------------------------------------------------

def cycle_period(ids, start: int, max_p: int = 64, rate: float = 0.8) -> int | None:
    tail = list(ids[start:])
    n = len(tail)
    if n < 8:
        return None
    for p in range(1, min(max_p, n // 2) + 1):
        match = sum(1 for i in range(p, n) if tail[i] == tail[i - p])
        if match / (n - p) >= rate:
            return p
    return None


def phase_labels(gen_ids, collapse: int | None, early_cycles: int = 2):
    """0 = pre-collapse, 1 = early loop, 2 = deep loop (per generated position)."""
    n = len(gen_ids)
    labels = np.zeros(n, dtype=np.int8)
    if collapse is None or collapse >= n:
        return labels, {"collapse": collapse, "period": None}
    p = cycle_period(gen_ids, collapse)
    meta = {"collapse": int(collapse), "period": p}
    if p is None:
        labels[collapse:] = 2
        return labels, meta
    labels[collapse:] = 1
    deep_from = min(collapse + early_cycles * p, n)
    labels[deep_from:] = 2
    return labels, meta


# --------------------------------------------------------------------------
# model / tokenizer / generation (raw completion mode)
# --------------------------------------------------------------------------

def load_tokenizer(model_dir: str = LOCAL_MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_prompt(tok, text: str):
    return tok(text, return_tensors="pt").input_ids


@torch.no_grad()
def gen_ids(model, tok, text: str, n: int = MAX_NEW) -> list[int]:
    """Greedy completion of n new tokens from raw text (protocol mode)."""
    enc = encode_prompt(tok, text)
    out = model.generate(enc, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return out[0][enc.shape[1]:].tolist()


# --------------------------------------------------------------------------
# writer-tensor plumbing
# --------------------------------------------------------------------------

def o_proj_key(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj.weight"


def down_proj_key(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def parse_layer(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) > 2 and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float("nan")


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
