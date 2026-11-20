"""
gemma_lab.py — shared runtime for the Gemma3-1B-IT RCDA-5 kit.

Path resolution, model/tokenizer loading, the chat-template encoder,
greedy generation, token-level loop labeling and small helpers shared by
the probe / training / benchmark scripts of this folder.

Conventions (mirrors the MNN-Pocket-0.3B RCDA kit):
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
    o_proj writes into, and the training penalty projects h_{l+1} on v_l;
  * a generation "loops" when the share of n-gram-repeat tokens (n = 4)
    reaches LOOP_TRESHOLD (0.5).
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
    here = Path(__file__).resolve().parent          # .../R CDA - 5/Code
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
    RCDA-5 protocol; kept for exploratory tooling)."""
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
# o_proj plumbing
# --------------------------------------------------------------------------

def o_proj_key(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj.weight"


def parse_layer(key: str) -> int | None:
    parts = key.split(".")
    if len(parts) > 2 and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def rss_gb() -> float:
    """Resident memory of this process, GB (for the RAM log lines)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return float("nan")
