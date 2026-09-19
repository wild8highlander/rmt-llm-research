# -*- coding: utf-8 -*-
"""
infer.py — инференс обученной TinyGPT v3 для веб-приложения (JSON в / JSON out).

Использование:
  python3 infer.py '{"prompt":"mp_bounds(q=0.5, sigma2=1.0)", "max_new_tokens": 24,
                     "temperature": 0.0}'
Печатает JSON: {generated, n_tokens, perplexity, elapsed_ms}.

Вход: prompt (строка), max_new_tokens (по умолчанию 24, клампится к окну),
temperature (0 = жадная генерация), seed (опционально).
Выход всегда валидный JSON на stdout (ошибки — {"error": "..."}).
"""
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REPO = "/home/z/my-project/rmt-llm-research"

if os.environ.get("RMT_LLM_ROOT"):
    REPO = os.path.abspath(os.environ["RMT_LLM_ROOT"])
else:
    REPO = _DEFAULT_REPO
    _cand = SCRIPT_DIR
    for _ in range(5):
        _cand = os.path.dirname(_cand)
        if os.path.isdir(os.path.join(_cand, "src", "rmt_llm")):
            REPO = _cand
            break

if os.environ.get("RMT_LLM_BASE"):
    BASE = os.path.abspath(os.environ["RMT_LLM_BASE"])
else:
    _base = os.path.dirname(SCRIPT_DIR)
    BASE = _base if os.path.basename(SCRIPT_DIR) == "scripts" and _base.startswith(REPO + os.sep) else os.path.dirname(REPO)

LAB = os.path.join(REPO, "laboratory", "python", "lab_en")
sys.path.insert(0, LAB)

from tiny_gpt_v3 import TinyGPTV3
from tiny_gpt_trainer import BPETokenizer

MODEL_DIR = os.path.join(BASE, "model")


def main() -> None:
    try:
        req = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"bad request: {e}"}))
        return

    prompt = str(req.get("prompt", ""))
    max_new = int(req.get("max_new_tokens", 24))
    temperature = float(req.get("temperature", 0.0))
    seed = req.get("seed")

    if not prompt.strip():
        print(json.dumps({"error": "prompt is empty"}))
        return

    try:
        tok = BPETokenizer.load(os.path.join(MODEL_DIR, "tiny_gpt_formula_bpe.json"))
        model = TinyGPTV3.load_weights(os.path.join(MODEL_DIR, "tiny_gpt_formula.npz"))
        model.training = False
    except Exception as e:
        print(json.dumps({"error": f"model load failed: {e}"}))
        return

    ids = tok.encode(prompt)[: model.config.max_seq_len // 2]
    # кламп к обученному окну контекста (позиции за max_seq_len не видели)
    max_new = max(0, min(max_new, model.config.max_seq_len - len(ids)))

    t0 = time.perf_counter()
    out = model.generate(np.array(ids), max_new_tokens=max_new,
                         temperature=temperature, capture_hidden=False,
                         seed=int(seed) if seed is not None else None)
    dt_ms = (time.perf_counter() - t0) * 1000

    out_ids = out.get("output_ids", [])
    text = tok.decode(out_ids)
    # прокси-уверенность: CE сгенерированной части на собственных логитах
    ppl = None
    if len(out_ids) >= 2:
        logits, _ = model.forward(np.array(out_ids[:-1], dtype=np.int64))
        lg = logits - logits.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        tgt = out_ids[1:]
        ce = float(np.mean(-logp[np.arange(len(tgt)), tgt]))
        ppl = float(np.exp(ce))

    print(json.dumps({
        "prompt": prompt,
        "generated": text,
        "n_tokens": len(out_ids),
        "perplexity": ppl,
        "elapsed_ms": round(dt_ms, 1),
        "model_params": int(model.config.params_count),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
