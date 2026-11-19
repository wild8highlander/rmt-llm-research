#!/usr/bin/env python3
"""stage2_build_sft_data.py — build the SFT arm dataset (pt/LM format).

The SFT arm asks the standard-tool question: can classic supervised
fine-tuning through LlamaFactory teach the model to escape the loop
attractor, given clean targets?

Data (frozen, pre-registered):
  * 20 bad prompts -> curated anti-loop continuations (data/antiloop_targets.json,
    written by the experimenter: acknowledge the pattern, then break it with
    concrete varied content and a natural ending)
  * 20 good prompts -> the base model's own greedy 192-token completions,
    kept only when they are clean themselves (loop_share < 0.3) — a
    self-distilled stability set, the SFT analog of the RCDA anchor

Output: data/sft_antiloop.jsonl (LlamaFactory "text" format, pt stage) +
data/sft_manifest.json (what went in and why).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rcda8_lab as L                                          # noqa: E402


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    targets = json.loads((L.DATA_DIR / "antiloop_targets.json").read_text(
        encoding="utf-8"))
    bad = L.load_bad_prompts()
    good = L.load_good_prompts()

    model = AutoModelForCausalLM.from_pretrained(L.LOCAL_MODEL,
                                                 dtype=torch.bfloat16)
    model.eval()
    tok = L.load_tokenizer()

    out_path = L.DATA_DIR / "sft_antiloop.jsonl"
    gens_cache = L.DATA_DIR / "good_gens_cache.jsonl"
    done = {}
    if gens_cache.exists():
        for line in gens_cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r

    rows = []
    for pid, text in bad:
        rows.append({"text": text + " " + targets[pid],
                     "source": "bad+curated", "id": pid})
    manifest_bad = len(rows)

    kept, dropped = 0, []
    for gid, text in good:
        if gid not in done:
            t0 = time.time()
            ids = L.gen_ids(model, tok, text, L.MAX_NEW)
            r = {"id": gid, "ids": ids, "loop_share": L.loop_share(ids)}
            with gens_cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[gid] = r
            print(f"[gen] {gid}: ls={r['loop_share']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        r = done[gid]
        if r["loop_share"] < 0.3:
            cont = tok.decode(r["ids"], skip_special_tokens=True)
            rows.append({"text": text + " " + cont,
                         "source": "good+self-distilled", "id": gid})
            kept += 1
        else:
            dropped.append({"id": gid, "loop_share": round(r["loop_share"], 3)})

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")

    L.dump_json({
        "n_bad_curated": manifest_bad,
        "n_good_self_distilled": kept,
        "n_good_dropped_looping": dropped,
        "format": "llamafactory pt stage, {'text': ...}",
        "cutoff_len": 256, "epochs": 4,
    }, L.DATA_DIR / "sft_manifest.json")
    print(f"sft dataset: {len(rows)} rows -> {out_path} "
          f"(bad {manifest_bad}, good kept {kept}, dropped {len(dropped)})",
          flush=True)


if __name__ == "__main__":
    main()
