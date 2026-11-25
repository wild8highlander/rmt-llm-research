"""Shared plumbing for the OCRia-nano-1 hallucination kit.

One import gives every script: canonical paths, the run seed, the char-level
loop metrics (the MNN/Gemma RCDA protocol, ported to character sequences),
CER, the phantom-text metrics, corpus loaders and the ORT sessions for the
deployed ONNX components.  The same file drives all arms, which is what makes
them comparable: one defect metric per family, one abilities metric, one
spectral gate - the same "one evaluator" rule the parent repository's RCDA
program runs on.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
KIT_DIR = CODE_DIR.parent
DATA_DIR = KIT_DIR / "Data"
RESULTS_DIR = KIT_DIR / "Results"
MODEL_DIR = KIT_DIR / "Model"

FAMILIES = {
    "F1": "phantom_text",
    "F2": "perseveration_loops",
    "F3": "misreads",
    "F4": "script_routing",
    "F5": "flip_blindness",
    "F6": "overconfidence",
}

BATCHES = ("structured", "noise", "blank", "clean", "degraded", "control")

NONTEXT_BATCHES = ("structured", "noise", "blank")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_image(path: Path) -> np.ndarray:
    """Grayscale crop [32, W] float32 in [0,1]."""
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def load_batch(batch: str) -> list[dict]:
    return load_jsonl(DATA_DIR / f"{batch}.jsonl")


def load_charset(rec: str = "latin") -> list[str]:
    return json.loads((MODEL_DIR / f"rec_{rec}.charset.json").read_text())


# --------------------------------------------------------------------------
# Decoding (CTC greedy, blank = class 0) - identical for ONNX ids and torch ids
# --------------------------------------------------------------------------

def ctc_decode(ids, charset: list[str] | None = None) -> str:
    """Collapse CTC frames: dedup + drop blank (id 0)."""
    cs = charset if charset is not None else load_charset()
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != 0:
            out.append(cs[i - 1] if 0 < i <= len(cs) else "?")
        prev = i
    return "".join(out)


# --------------------------------------------------------------------------
# The defect metrics - one definition for every arm (RCDA protocol, chars)
# --------------------------------------------------------------------------

def loop_stats(seq: str, ngram: int = 4) -> dict:
    """loop_share: fraction of characters that belong to an n-gram occurring at
    least twice in the sequence (the parent repo's repetition-loop metric,
    ported from tokens to characters for CTC output); loop_waste: their count."""
    n = len(seq)
    if n < ngram:
        return {"loop_share": 0.0, "loop_waste": 0, "n_seq": n}
    counts: dict[str, int] = {}
    for i in range(n - ngram + 1):
        g = seq[i:i + ngram]
        counts[g] = counts.get(g, 0) + 1
    mask = [False] * n
    for i in range(n - ngram + 1):
        if counts[seq[i:i + ngram]] >= 2:
            for j in range(i, i + ngram):
                mask[j] = True
    waste = sum(mask)
    return {"loop_share": waste / n, "loop_waste": waste, "n_seq": n}


def rho_of_texts(texts: list[str], threshold: float = 0.5) -> dict:
    """rho = fraction of trajectories whose loop_share >= threshold."""
    stats = [loop_stats(t) for t in texts]
    bad = [s["loop_share"] >= threshold for s in stats]
    return {
        "rho": float(np.mean(bad)) if bad else 0.0,
        "mean_loop_share": float(np.mean([s["loop_share"] for s in stats])) if stats else 0.0,
        "loop_waste_mean": float(np.mean([s["loop_waste"] for s in stats])) if stats else 0.0,
        "per_traj_loop_share": [round(s["loop_share"], 4) for s in stats],
        "threshold": threshold,
    }


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (iterative DP, O(len(a)*len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character error rate, normalised by the reference length."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


# --------------------------------------------------------------------------
# ORT sessions for the deployed components
# --------------------------------------------------------------------------

def ort_session(onnx_name: str):
    import onnxruntime as ort
    return ort.InferenceSession(str(MODEL_DIR / onnx_name),
                                providers=["CPUExecutionProvider"])


def run_recognizer(sess, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ids [T], conf [T] for one crop [32, W] in [0,1]."""
    x = np.ascontiguousarray(crop.astype(np.float32)[None, None])
    ids, conf = sess.run(None, {"x": x})
    return ids[0], conf[0]


def run_router(sess, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """group [4], flip [2] logits for one 32x512 crop."""
    x = np.ascontiguousarray(crop.astype(np.float32)[None, None])
    group, flip = sess.run(None, {"x": x})
    return group[0], flip[0]


ROUTER_GROUPS = ("latin", "cjk", "rtl", "world")  # order calibrated in probe


# --------------------------------------------------------------------------
# Run passport (the parent repo's provenance rule)
# --------------------------------------------------------------------------

def build_passport() -> dict:
    import hashlib
    import platform
    import sys

    def sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    files = {}
    for p in sorted(MODEL_DIR.glob("*")):
        if p.is_file():
            files[p.name] = sha256(p)
    import torch, onnxruntime, PIL, matplotlib
    return {
        "kit": "OCRia-nano-1 hallucination intervention kit",
        "parent_program": "RCDA (neural_lab, rmt-llm-research)",
        "model_repo": "https://huggingface.co/llamaindex/OCRia-nano-1",
        "model_files_sha256": files,
        "seeds": {"probe": 0, "corpora": 7, "train": 0, "bench": 0},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
        },
    }
