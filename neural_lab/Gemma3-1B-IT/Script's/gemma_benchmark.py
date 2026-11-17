#!/usr/bin/env python3
"""gemma_benchmark.py — final evaluation of an RCDA-5 edit against the
pre-registered gates (the Gemma3 analog of the MNN kit's rcda4_eval_b.py):
the full 20 bad + 20 good benchmark (greedy, 192 tokens, raw completion mode),
KL(base || edited) on the 5 control texts, Tracy-Widom spectral control per
edited tensor, and the cosine between the edit's top output direction u1 and
the measured loop direction v_l.

Usage:
  gemma_benchmark.py --model DIR --edited-state none      --tag base    --out part5
  gemma_benchmark.py --model DIR --edited-state trained_state.safetensors --tag trained --out part5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma_lab as L                                    # noqa: E402


@torch.no_grad()
def kl_ctrl(model, tok, texts, base_logits_list):
    """KL(base || arm) averaged over tokens, per control text, then mean."""
    kls = []
    for text, bl in zip(texts, base_logits_list):
        enc = L.encode_prompt(tok, text)
        le = torch.log_softmax(model(enc).logits[0].float(), dim=-1)
        lb = torch.log_softmax(bl.float(), dim=-1)
        n = min(lb.shape[0], le.shape[0])
        kls.append(float((lb[:n].exp() * (lb[:n] - le[:n])).sum(-1).mean()))
    return float(np.mean(kls))


class LazyBaseWeights:
    """Per-key access to the base safetensors (RCDA-6 adaptation).

    load_file() materializes the whole 2 GB checkpoint on top of the already
    loaded model and OOM-kills a 4 GB box; safe_open + get_tensor touches
    only the 16 edited tensors instead.
    """

    def __init__(self, path: Path):
        self._path = path
        self._keys = None
        self._f = None

    def _open(self):
        if self._f is None:
            self._f = safe_open(str(self._path), framework="pt")
        return self._f

    def __contains__(self, k):
        if self._keys is None:
            with safe_open(str(self._path), framework="pt") as f:
                self._keys = set(f.keys())
        return k in self._keys

    def __getitem__(self, k):
        return self._open().get_tensor(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--edited-state", required=True,
                    help="safetensors with the edited tensors, or 'none' for the base arm")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. part5")
    ap.add_argument("--probe", default=None, help="probe dir (v vectors for TW/cos)")
    ap.add_argument("--max-new", type=int, default=L.MAX_NEW)
    ap.add_argument("--reuse-gens", action="store_true",
                    help="base arm: reuse cached base generations (probe + data "
                         "cache) where available; rows still appended to disk")
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = Path(args.out).name
    probe_dir = Path(args.probe) if args.probe else L.resolve_work() / "probe"

    from transformers import AutoModelForCausalLM
    from marchenko_pastur import spectral_tw_control

    tok = L.load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    base_path = Path(args.model)
    base_st = base_path / "model.safetensors"
    base_sd = LazyBaseWeights(base_st)          # per-key, no 2 GB materialization

    bad, good = L.load_bad_prompts(), L.load_good_prompts()
    ctrl_texts = [t for _, t in L.load_control_texts()]

    # ---- base logits per control text (from the CURRENT, still-base weights) ----
    base_logits_list = []
    with torch.no_grad():
        for t in ctrl_texts:
            enc = L.encode_prompt(tok, t)
            base_logits_list.append(model(enc).logits[0].float().cpu())

    edited_sd = None
    if args.edited_state != "none":
        edited_sd = load_file(args.edited_state)
        changed = {}
        for k in edited_sd:
            if k in base_sd and not torch.equal(base_sd[k], edited_sd[k]):
                d = (edited_sd[k].float() - base_sd[k].float())
                rel = float(torch.linalg.norm(d)
                            / max(torch.linalg.norm(base_sd[k].float()), 1e-9))
                changed[k] = {"rel_frobenius": round(rel, 5),
                              "n_weights": int((d != 0).sum())}
        with torch.no_grad():
            for k, w in edited_sd.items():
                dict(model.named_parameters())[k].data.copy_(w)
        print(f"applied edit: {len(changed)} tensors changed", flush=True)

    # ---- TW / cosine forensics (incrementally persisted, resumable) ----
    tw, cosines = {}, {}
    tw_cache = out_dir / f"{out_prefix}_tw_{args.tag}.json"
    if edited_sd and tw_cache.exists():
        saved = json.load(tw_cache.open())
        tw, cosines = saved.get("tw", {}), saved.get("cosine_edit_vs_v", {})
        print(f"TW cache: {len(tw)} tensors already analyzed", flush=True)
    if edited_sd:
        for k, meta in changed.items():
            if k in tw:
                continue
            lay = L.parse_layer(k)
            vfile = probe_dir / f"v_L{lay}.npy"
            if lay is None or not vfile.exists():
                continue
            v = np.load(vfile).astype(np.float64)
            wb = base_sd[k].float().numpy().astype(np.float64)
            wa = edited_sd[k].float().numpy().astype(np.float64)
            try:
                tw[k] = spectral_tw_control(wb, wa, v)
            except Exception as e:
                tw[k] = {"error": str(e)}
            del wb, wa
            u, _, _ = np.linalg.svd((edited_sd[k].float()
                                     - base_sd[k].float()).numpy(),
                                    full_matrices=False)
            if u.shape[1]:
                cosines[k] = round(float(np.abs(u.T @ (v / np.linalg.norm(v)))[0]), 4)
            del u
            with tw_cache.open("w") as f:
                json.dump({"tw": tw, "cosine_edit_vs_v": cosines}, f, indent=1)
            print(f"[tw] {k}: {tw[k].get('verdict', tw[k])}", flush=True)

    # ---- the 40-prompt behavioral benchmark (with row-level resume) ----
    rows_path = out_dir / f"{out_prefix}_rows_{args.tag}.jsonl"
    done_rows: dict[str, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_rows[r["id"]] = r
        print(f"rows resume: {len(done_rows)} already measured", flush=True)

    cached = {}
    if args.reuse_gens:
        for cf in (probe_dir / "gens.jsonl",
                   L.resolve_work() / "rcda5" / "data_gens.jsonl"):
            if cf.exists():
                for line in cf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        cached.setdefault(r["id"],
                                          r.get("ids") or r.get("gen_ids"))

    rows = []
    t0 = time.time()
    for gid, p in bad + good:
        kind = "bad" if gid.startswith("bad") else "good"
        if gid in done_rows:
            rows.append(done_rows[gid])
            continue
        if args.tag == "base" and gid in cached and len(cached[gid]) == args.max_new:
            ids = cached[gid]
        else:
            ids = L.gen_ids(model, tok, p, args.max_new)
        labels = L.token_loop_labels(ids)
        n = len(ids)
        ls = float(labels.mean()) if n else 0.0
        collapse = int(np.argmax(labels)) if labels.any() else None
        waste = (n - collapse) if collapse is not None else 0
        row = {"id": gid, "kind": kind, "n_tokens": n,
               "loop_share": round(ls, 4), "looping": ls >= L.LOOP_THRESHOLD,
               "loop_waste": waste}
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{args.tag}] {gid} {kind}: ls={ls:.2f} waste={waste}", flush=True)
    print(f"benchmark done in {time.time()-t0:.0f}s", flush=True)

    def frac(sub, key):
        sub_r = [r for r in rows if r["kind"] == sub]
        return round(sum(r[key] for r in sub_r) / len(sub_r), 4)

    summary = {
        "tag": args.tag, "edited_state": args.edited_state,
        "n_changed_tensors": len(changed) if edited_sd else 0,
        "changed": changed if edited_sd else {},
        "tw": {k: ({"verdict": v["verdict"]} if isinstance(v, dict)
                   and "verdict" in v else v) for k, v in tw.items()},
        "cosine_edit_vs_v": cosines,
        "rho_bad": frac("bad", "looping"),
        "rho_good": frac("good", "looping"),
        "loop_waste_mean": round(float(np.mean([r["loop_waste"] for r in rows
                                                if r["kind"] == "bad"])), 2),
        "loop_share_mean_bad": round(float(np.mean([r["loop_share"] for r in rows
                                                    if r["kind"] == "bad"])), 4),
        "kl_ctrl": None,          # filled below
        "rows": rows,
    }
    summary["kl_ctrl"] = round(kl_ctrl(model, tok, ctrl_texts, base_logits_list), 4)
    json.dump(summary, (out_dir / f"{out_prefix}_eval_{args.tag}.json").open("w"),
              indent=1)
    brief = {k: v for k, v in summary.items() if k != "rows"}
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
