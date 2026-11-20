"""
gemma_lab.py — shared runtime for the Gemma3-1B-IT RCDA-6 kit.

Path resolution, model/tokenizer loading, the raw-completion encoder,
greedy generation, token-level loop labeling, cycle-period detection and
small helpers shared by the probe / training / benchmark scripts of this
folder.

Conventions (mirrors the MNN-Pocket-0.3B RCDA kit and the RCDA-5 Gemma port):
  * prompts are fed in RAW completion mode (no chat template) — the closer
    analog of the MNN-Pocket protocol, where a plain completion model was
    probed greedily; the on-device LiteRT deployment of Gemma3-1B-IT uses the
    same prefill/decode path;
  * MAX_NEW = 192: Gemma3-1B-IT is far more repetition-resistant than
    Pocket-0.3B, and the defect window opens only in longer generations
    (sentinel scan: 96 tokens -> no defect, 192 -> hard loops);
  * residual stream after decoder layer l is hidden_states[l + 1]
    (hidden_states[0] is the embedding output);
  * the loop direction v_l therefore lives in the space that layer l's
    o_proj AND down_proj write into, and the RCDA-6 penalty projects
    h_{l+1} on the PHASE-conditioned directions (v_early / v_deep);
  * a generation "loops" when the share of n-gram-repeat tokens (n = 4)
    reaches LOOP_TRESHOLD (0.5).

RCDA-6 additions over the RCDA-5 copy:
  * down_proj_key()          — the MLP writer tensor of a layer;
  * cycle_period()           — smallest token period p of the loop tail;
  * phase_labels()           — per-position phase assignment
                               (pre / early-loop / deep-loop) used both by
                               gemma_probe_phase.py (direction extraction)
                               and gemma_rcds6_train.py (the penalty).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

NGRAM = 4
LOOP_THRESHOLD = 0.5
MAX_NEW = 192          # Gemma adaptation: see module docstring
BAD_PREF = 48

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def resolve_lab() -> Path:
    """Root of this experiment folder (the directory that contains Code/)."""
    here = Path(__file__).resolve().parent          # .../R CDA - 6/Code
    return here.parent


def resolve_work() -> Path:
    """Runtime scratch dir (states, generations, caches).

    Override with $GEMMA_WORK; defaults to <lab>/rcda_work (gitignored).
    """
    env = os.environ.get("GEMMA_WORK")
    work = Path(env) if env else resolve_lab() / "rcda_work"
    work.mkdir(parents=True, exist_ok=True)
    return work


def prompts_dir() -> Path:
    return resolve_lab() / "prompts"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_bad_prompts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(prompts_dir() / "bad.jsonl")]


def load_good_prompts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(prompts_dir() / "good.jsonl")]


def load_control_texts() -> list[tuple[str, str]]:
    return [(r["id"], r["text"]) for r in load_jsonl(prompts_dir() / "control.jsonl")]


# --------------------------------------------------------------------------
# loop detection (identical logic to the MNN kit)
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


def loop_collapse_point(ids, ngram: int = NGRAM) -> int | None:
    labels = token_loop_labels(ids, ngram)
    return int(np.argmax(labels)) if labels.any() else None


# --------------------------------------------------------------------------
# RCDA-6: cycle period + phase assignment
# --------------------------------------------------------------------------

def cycle_period(ids, start: int, max_p: int = 64, rate: float = 0.8) -> int | None:
    """Smallest period p >= 1 of ids[start:] with token self-match >= rate.

    The loop tail of a collapsed generation is (approximately) periodic with
    the period of the repeated unit (one sentence, one token, ...). Returns
    None if no period up to max_p fits (irregular tail).
    """
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
    """Per-position phase labels of a generated tail.

    Returns (labels, meta) where labels is an int8 array of length
    len(gen_ids): 0 = pre-collapse, 1 = early loop (first `early_cycles`
    cycles after the collapse), 2 = deep loop (beyond that). meta carries
    the collapse point and the fitted cycle period (None if irregular).
    """
    n = len(gen_ids)
    labels = np.zeros(n, dtype=np.int8)
    if collapse is None or collapse >= n:
        return labels, {"collapse": collapse, "period": None}
    p = cycle_period(gen_ids, collapse)
    meta = {"collapse": int(collapse), "period": p}
    if p is None:
        # irregular tail: no fitted period, but every position past the
        # collapse is deep-loop content by any reasonable reading — a
        # trajectory that keeps looping 100+ tokens without a stable period
        # has long left its "first cycles". Label it deep; otherwise the
        # hundreds of deep positions of irregular prompts would dilute the
        # early-phase direction.
        labels[collapse:] = 2
        return labels, meta
    labels[collapse:] = 1
    deep_from = min(collapse + early_cycles * p, n)
    labels[deep_from:] = 2
    return labels, meta


# --------------------------------------------------------------------------
# model loading / encoding / generation
# --------------------------------------------------------------------------

def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_dir: str, dtype=None):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype or torch.bfloat16)
    model.eval()
    return model


def encode_prompt(tok, text: str):
    """Encode raw text for completion-mode generation -> input_ids tensor."""
    return tok(text, return_tensors="pt").input_ids


def encode_chat(tok, text: str):
    """Encode a user turn through the Gemma3 chat template (NOT used by the
    RCDA protocol; kept for exploratory tooling)."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return enc["input_ids"]


@torch.no_grad()
def gen_ids(model, tok, text: str, n: int = MAX_NEW) -> list[int]:
    """Greedy completion of n new tokens from raw text (protocol mode)."""
    enc = encode_prompt(tok, text)
    out = model.generate(enc, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return out[0][enc.shape[1]:].tolist()


@torch.no_grad()
def full_forward_states(model, tok, text: str, n: int = MAX_NEW):
    """Generate n greedy tokens, then one cacheless forward over the whole
    sequence -> (ids, collapse_point, hidden_states tuple, prompt_len).

    hidden_states[k] is the residual stream AFTER decoder layer k-1
    (hidden_states[0] = embeddings).
    """
    ids = gen_ids(model, tok, text, n)
    plen = int(encode_prompt(tok, text).shape[1])
    enc = encode_prompt(tok, text)
    full = torch.cat([enc[0], torch.tensor(ids, dtype=enc.dtype)], dim=0).unsqueeze(0)
    out = model(full, output_hidden_states=True)
    return full[0].tolist(), loop_collapse_point(ids), out.hidden_states, plen, ids


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


def parse_tensor_type(key: str) -> str | None:
    """'o_proj' | 'down_proj' for a writer-tensor key, else None."""
    if key.endswith("self_attn.o_proj.weight"):
        return "o_proj"
    if key.endswith("mlp.down_proj.weight"):
        return "down_proj"
    return None


def rss_gb() -> float:
    """Resident memory of this process, GB (for the RAM log lines)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float("nan")
