# -*- coding: utf-8 -*-
"""
benchmark_rmt.py — "before/after" benchmark for RMT interventions on MNN-Pocket-0.3B.

Pipeline:
  1) loads the model (a local directory with safetensors weights);
  2) for each configuration (baseline / clip / dampen / project / random-ablation)
     attaches RMT hooks to all decoder layers (or none for the baseline);
  3) greedily generates a continuation of every prompt (use_cache=False! — see the
     docstring of rmt_intervention.py: otherwise the hook sees a 1xH matrix and does
     nothing);
  4) computes degeneration metrics on the generated continuation only:
       Distinct-2 / Distinct-4     — fraction of unique n-grams (higher is better);
       Repetition Rate-4           — fraction of repeated 4-grams (lower is better);
       tokens_to_collapse          — position of the first repeated 4-gram (higher is
                                     better, censored as n_tokens+1 when no repeat);
  5) measures perplexity on 3 validation texts (also under the current hook);
  6) paired Wilcoxon signed-rank test across prompts (config vs baseline);
  7) writes results.json + summary.csv + summary.md into out_dir.
     Re-running only processes the configurations that are not finished yet (resume).

Note on the prompt set: the last two prompts are deliberately Russian — they probe
the multilingual behaviour of the model and keep this run comparable with the
recorded results.json. Generation for them is in Russian; the metric code itself
is language-agnostic (whitespace tokenization).

Usage:
  python benchmark_rmt.py                                # full run
  python benchmark_rmt.py --n_prompts 1 --max_new 12     # smoke test
  python benchmark_rmt.py --configs clip,dampen_a0.5     # subset
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmt_intervention import install_rmt_hooks, remove_hooks  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The sun rises in the",
    "William Shakespeare is famous for",
    "The largest planet in the solar system is",
    "To cook rice, first you need to",
    "In a small village at the edge of the forest, there lived",
    "The internet was invented in",
    "Чтобы приготовить чай, сначала нужно",
    "Столица Франции — это",
]

VAL_TEXTS = [
    "The capital of France is Paris. It is the most populous city in France and the "
    "center of the country's cultural, political and economic life. The city is known "
    "for its museums, architecture and cuisine. Millions of tourists visit Paris every "
    "year to see the Eiffel Tower and the Louvre museum.",
    "Water covers most of the Earth's surface. Oceans regulate the climate and are home "
    "to a huge variety of living creatures. Rivers carry fresh water from mountains to "
    "the seas, and rain returns it back to the land. The water cycle is one of the "
    "fundamental processes of nature.",
    "Вода — основа жизни на Земле. Океаны регулируют климат и дают дом огромному "
    "разнообразию живых существ. Реки несут пресную воду с гор к морям, а дождь "
    "возвращает её на землю. Круговорот воды в природе — один из фундаментальных "
    "процессов.",
]


# ---------------------------------------------------------------- text metrics

def distinct_n(text, n):
    toks = text.split()
    if len(toks) < n:
        return None
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(text, n=4):
    toks = text.split()
    if len(toks) < n:
        return None
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    extra = sum(c - 1 for c in counts.values() if c > 1)
    return extra / len(grams)


def collapse_step(text, n=4):
    """Token position at which an n-gram repeats for the first time; None if never."""
    toks = text.split()
    if len(toks) < n:
        return None
    seen = set()
    for j in range(len(toks) - n + 1):
        g = tuple(toks[j:j + n])
        if g in seen:
            return j + n
        seen.add(g)
    return None


# ---------------------------------------------------------------- model helpers

@torch.no_grad()
def greedy_generate(model, tok, prompt, max_new_tokens):
    ids = tok(prompt, return_tensors="pt").input_ids
    gen = []
    eos = tok.eos_token_id
    for _ in range(max_new_tokens):
        out = model(ids, use_cache=False)          # no cache: the hook sees (T x H)
        nxt = int(out.logits[0, -1].argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], dtype=ids.dtype)], dim=1)
    return tok.decode(gen, skip_special_tokens=True), len(gen)


@torch.no_grad()
def perplexity(model, tok, text, max_len=512):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    if ids.shape[1] < 2:
        return None
    out = model(ids, labels=ids, use_cache=False)
    return float(torch.exp(out.loss))


def run_config(model, tok, cfg, prompts, max_new):
    handles = hooks = None
    if cfg["mode"] != "none":
        handles, hooks = install_rmt_hooks(
            model, cfg["mode"], alpha=cfg.get("alpha", 0.5),
            top_k=cfg.get("top_k", 10), seed=cfg.get("seed", 0))
    rows = []
    for p in prompts:
        t0 = time.time()
        text, ntok = greedy_generate(model, tok, p, max_new)
        rows.append({
            "prompt": p,
            "n_tokens": ntok,
            "seconds": round(time.time() - t0, 1),
            "text": text,
            "distinct4": distinct_n(text, 4),
            "distinct2": distinct_n(text, 2),
            "rep4": repetition_rate(text, 4),
            "collapse": collapse_step(text, 4),
        })
        c = rows[-1]["collapse"]
        print(f"  [{cfg['name']}] \"{p[:34]}...\" tokens={ntok} D4={rows[-1]['distinct4']} "
              f"RR4={rows[-1]['rep4']} collapse={c} "
              f"\"{text[:60]}...\"", flush=True)
    ppls = [perplexity(model, tok, t) for t in VAL_TEXTS]
    hook_stats = None
    if hooks:
        hook_stats = [h.stats() for h in hooks]
        remove_hooks(handles)
    return {"config": cfg, "rows": rows, "ppl": ppls, "hook_stats": hook_stats}


# ---------------------------------------------------------------- summary & tests

def _clean(x):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def config_summary(run, base_ppl):
    rows = run["rows"]
    ppls = [p for p in run["ppl"] if p is not None and math.isfinite(p)]
    collapses = [r["collapse"] if r["collapse"] is not None else r["n_tokens"] + 1
                 for r in rows]
    hs = run["hook_stats"]
    agg = {
        "name": run["config"]["name"],
        "mean_distinct4": float(np.mean([r["distinct4"] for r in rows if r["distinct4"] is not None])),
        "mean_distinct2": float(np.mean([r["distinct2"] for r in rows if r["distinct2"] is not None])),
        "mean_rep4": float(np.mean([r["rep4"] for r in rows if r["rep4"] is not None])),
        "median_collapse": float(np.median(collapses)),
        "mean_ppl": float(np.mean(ppls)) if ppls else None,
        "ppl_ratio_vs_base": (float(np.mean(ppls)) / base_ppl) if (ppls and base_ppl) else None,
        "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
    }
    if hs and sum(h["n_calls"] for h in hs):
        agg["hook_mean_intervened"] = float(np.mean([h["mean_intervened"] for h in hs]))
        agg["hook_energy_kept"] = float(np.mean([h["mean_energy_kept"] for h in hs]))
        agg["hook_max_ratio"] = float(np.mean([h["mean_max_ratio"] for h in hs]))
    return agg, collapses


def paired_p(base_vals, cfg_vals):
    try:
        from scipy.stats import wilcoxon
        x = np.asarray(base_vals, float)
        y = np.asarray(cfg_vals, float)
        if np.array_equal(x, y):
            return None
        stat = wilcoxon(x, y, zero_method="wilcox")
        return float(stat.pvalue)
    except Exception:
        return None


def write_summary_md(path, summaries, pvals, args):
    L = ["# RMT interventions on MNN-Pocket-0.3B — benchmark summary", "",
         f"Model: `{args.model_path}` · greedy decoding, use_cache=False · "
         f"max_new_tokens={args.max_new} · prompts: {args.n_prompts or len(PROMPTS)} · seed={args.seed}", "",
         "Metrics are computed on the continuation only: Distinct-4 (higher is better), "
         "Repetition Rate-4 (lower is better), median tokens_to_collapse (higher is better, "
         "censored as n_tokens+1), PPL on 3 validation texts (lower is better).", "",
         "| config | Distinct-4 ↑ | RepRate-4 ↓ | collapse med. ↑ | PPL ↓ | PPL ×base | intervened/call | energy kept | p (D4) | p (RR4) |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(x, nd=3):
        return "—" if x is None else (f"{x:.{nd}f}")

    for s in summaries:
        pv = pvals.get(s["name"], {})
        L.append(
            f"| {s['name']} | {fmt(s['mean_distinct4'])} | {fmt(s['mean_rep4'])} | "
            f"{fmt(s['median_collapse'], 1)} | "
            f"{fmt(s['mean_ppl'], 1)} | {fmt(s['ppl_ratio_vs_base'], 2)} | "
            f"{fmt(s.get('hook_mean_intervened'), 1)} | "
            f"{fmt(s.get('hook_energy_kept'))} | "
            f"{pv.get('distinct4', '—')} | {pv.get('rep4', '—')} |")
    L += ["",
          "p — paired Wilcoxon signed-rank test across prompts (config vs baseline); "
          "with n≈8-10 prompts the smallest attainable p is ≈0.002-0.01, so treat the "
          "values as an indication, not as proof.",
          "",
          "Be careful with Distinct-4: a generation that falls apart into noise also "
          "yields a high Distinct-4. Only the combination 'RR4 lower, collapse later, "
          "PPL not worse' counts as an improvement. Always cross-check against the "
          "generation texts in results.json (the text field)."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ================================================================
# RCDA mode — before/after evaluation of WEIGHT edits (plan, 6-7)
# ================================================================

def _read_jsonl_texts(path, limit=0):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("prompt")
            if text:
                items.append({"id": obj.get("id", f"{os.path.basename(path)}#{len(items)}"),
                              "text": text})
            if limit and len(items) >= limit:
                break
    return items


@torch.no_grad()
def _greedy_generate_cached(model, tok, prompt, max_new_tokens):
    """Greedy generation with the KV cache (RCDA edits weights, not activations,
    so the hook constraint use_cache=False does not apply here)."""
    enc = tok(prompt, return_tensors="pt")
    n_in = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen_ids = out[0][n_in:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    n_gen = int((gen_ids != tok.eos_token_id).sum()) if tok.eos_token_id in gen_ids \
        else int(gen_ids.shape[0])
    return text, min(n_gen, int(gen_ids.shape[0]))


def _defect_flags(text, n_tokens, max_new, rep_threshold, collapse_frac):
    """Binary classifier of the repetition_loop defect (plan, section 0).

    defect_hit = (a 4-gram collapse happened within collapse_frac*max_new tokens)
                 OR (Repetition Rate-4 >= rep_threshold).
    Truncation (generation ran to the token limit) is reported separately:
    for small models it is common even without looping.
    """
    rep4 = repetition_rate(text, 4)
    coll = collapse_step(text, 4)
    truncated = n_tokens >= max_new
    collapsed_early = coll is not None and coll <= collapse_frac * max_new
    rep_hit = rep4 is not None and rep4 >= rep_threshold
    return {"rep4": _clean(rep4), "distinct4": _clean(distinct_n(text, 4)),
            "collapse": coll, "n_tokens": n_tokens, "truncated": truncated,
            "collapsed_early": collapsed_early, "rep_hit": rep_hit,
            "defect_hit": bool(collapsed_early or rep_hit)}


def _load_model_dir(model_dir, device="cpu"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval().to(device)
    return model, tok


def run_rcda_benchmark(args):
    from rmt_intervention import load_edit_pack, spectral_tw_control

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    bad = _read_jsonl_texts(args.bad_set, args.n_prompts or 0)
    good = _read_jsonl_texts(args.good_set, args.n_prompts or 0)
    control_texts = ([t["text"] for t in _read_jsonl_texts(args.control_set)]
                     if args.control_set else list(VAL_TEXTS))
    print(f"RCDA benchmark: {len(bad)} bad / {len(good)} good prompts, "
          f"{len(control_texts)} control texts", flush=True)

    runs = {}
    for name, model_dir in (("baseline", args.baseline_dir), ("edited", args.edited_dir)):
        model, tok = _load_model_dir(model_dir)
        rows = []
        for split, items in (("bad", bad), ("good", good)):
            for it in items:
                t0 = time.time()
                text, ntok = _greedy_generate_cached(model, tok, it["text"], args.max_new)
                flags = _defect_flags(text, ntok, args.max_new,
                                      args.rep_threshold, args.collapse_frac)
                rows.append({"split": split, "id": it["id"], **flags,
                             "seconds": round(time.time() - t0, 1),
                             "text": text})
                print(f"  [{name}:{split}] {it['id']}: hit={flags['defect_hit']} "
                      f"rep4={flags['rep4']} coll={flags['collapse']} "
                      f"ntok={ntok}", flush=True)
        ppls = [perplexity(model, tok, t) for t in control_texts]
        runs[name] = {"model_dir": model_dir, "rows": rows, "ppl": ppls}
        del model
        print(f"  [{name}] PPL on control: {ppls}", flush=True)

    def _split(runs_, name, split):
        return [r for r in runs_[name]["rows"] if r["split"] == split]

    def _rho(rows):
        hits = [r["defect_hit"] for r in rows]
        return (float(np.mean(hits)), int(sum(hits)), len(hits)) if hits else (None, 0, 0)

    rho0, h0, n0 = _rho(_split(runs, "baseline", "bad"))
    rho1, h1, n1 = _rho(_split(runs, "edited", "bad"))
    fpr0, f0, g0 = _rho(_split(runs, "baseline", "good"))
    fpr1, f1, g1 = _rho(_split(runs, "edited", "good"))
    ppl0 = float(np.mean([p for p in runs["baseline"]["ppl"] if p])) if any(runs["baseline"]["ppl"]) else None
    ppl1 = float(np.mean([p for p in runs["edited"]["ppl"] if p])) if any(runs["edited"]["ppl"]) else None

    # paired tests (edited vs baseline on the same prompts)
    bad_ids = [r["id"] for r in _split(runs, "baseline", "bad")]
    base_by_id = {r["id"]: r for r in _split(runs, "baseline", "bad")}
    edit_by_id = {r["id"]: r for r in _split(runs, "edited", "bad")}
    tests = {}
    for metric in ("rep4", "distinct4"):
        b = [base_by_id[i][metric] or 0 for i in bad_ids if i in base_by_id]
        e = [edit_by_id[i][metric] or 0 for i in bad_ids if i in edit_by_id]
        tests[metric] = ("identical (0 diffs)" if np.array_equal(np.asarray(b), np.asarray(e))
                         else paired_p(b, e))
    b_coll = [base_by_id[i]["collapse"] if base_by_id[i]["collapse"] is not None
              else base_by_id[i]["n_tokens"] + 1 for i in bad_ids if i in base_by_id]
    e_coll = [edit_by_id[i]["collapse"] if edit_by_id[i]["collapse"] is not None
              else edit_by_id[i]["n_tokens"] + 1 for i in bad_ids if i in edit_by_id]
    tests["collapse"] = ("identical (0 diffs)" if np.array_equal(np.asarray(b_coll), np.asarray(e_coll))
                         else paired_p(b_coll, e_coll))
    tests = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tests.items()}

    # how many greedy trajectories actually changed (iteration signal, plan section 6)
    n_changed = {}
    for split in ("bad", "good"):
        base_split = {r["id"]: r for r in _split(runs, "baseline", split)}
        edit_split = {r["id"]: r for r in _split(runs, "edited", split)}
        n_changed[split] = int(sum(base_split[i]["text"] != edit_split[i]["text"]
                                   for i in base_split if i in edit_split))

    # ---------------------------------------------------------------- TW control
    report_path = args.rcda_report or os.path.join(args.edited_dir, "rcda_edit_report.json")
    tw_controls = []
    if os.path.exists(report_path):
        from safetensors import safe_open
        edits, _ = load_edit_pack(report_path)
        with safe_open(os.path.join(args.baseline_dir, "model.safetensors"), "pt") as fb, \
                safe_open(os.path.join(args.edited_dir, "model.safetensors"), "pt") as fe:
            for e in edits:
                if e.get("status") == "aborted_no_rmt_columns" or "vector" not in e:
                    continue
                key = e.get("state_dict_key")
                if key not in fb.keys() or key not in fe.keys():
                    continue
                ctrl = spectral_tw_control(fb.get_tensor(key).float().numpy(),
                                           fe.get_tensor(key).float().numpy(), e["vector"])
                tw_controls.append({"layer": e["layer"], "matrix": e["matrix"],
                                    "key": key, **ctrl})
    else:
        print(f"NOTE: {report_path} not found — spectral TW control skipped", flush=True)

    summary = {
        "defect": "repetition_loop",
        "n_changed_generations": n_changed,
        "params": {"max_new": args.max_new, "rep_threshold": args.rep_threshold,
                   "collapse_frac": args.collapse_frac},
        "rho_baseline": rho0, "rho_edited": rho1,
        "rho_reduction_pp": (100.0 * (rho0 - rho1) if (rho0 is not None and rho1 is not None) else None),
        "fpr_baseline": fpr0, "fpr_edited": fpr1,
        "ppl_baseline": ppl0, "ppl_edited": ppl1,
        "ppl_ratio": (ppl1 / ppl0 if (ppl0 and ppl1) else None),
        "wilcoxon_edited_vs_baseline": tests,
        "tw_control_verdicts": [{"layer": c["layer"], "matrix": c["matrix"],
                                 "verdict": c["verdict"],
                                 "n_violations": c["n_violations"],
                                 "target_shift_tw_units": c["target_shift_tw_units"]}
                                for c in tw_controls],
    }
    with open(os.path.join(out_dir, "rcda_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"baseline_dir": args.baseline_dir,
                            "edited_dir": args.edited_dir,
                            "bad_set": args.bad_set, "good_set": args.good_set,
                            "max_new": args.max_new, "date": time.strftime("%Y-%m-%d %H:%M")},
                   "summary": summary,
                   "runs": {k: {"ppl": v["ppl"],
                                "rows": [{kk: vv for kk, vv in r.items()}
                                         for r in v["rows"]]}
                            for k, v in runs.items()},
                   "tw_controls": tw_controls},
                  f, ensure_ascii=False, indent=1, default=_clean)

    def _pct(x):
        return "—" if x is None else f"{100 * x:.1f}%"

    L = ["# RCDA before/after — repetition_loop defect", "",
         f"baseline: `{args.baseline_dir}` · edited: `{args.edited_dir}` · "
         f"greedy, KV cache on · max_new_tokens={args.max_new} · "
         f"defect: collapse within {args.collapse_frac:.0%} of max_new OR RepRate-4 >= {args.rep_threshold}", "",
         "| quantity | baseline | edited |",
         "|---|---|---|",
         f"| rho on P_bad (defect rate) | {_pct(rho0)} ({h0}/{n0}) | {_pct(rho1)} ({h1}/{n1}) |",
         f"| false-trigger rate on P_good | {_pct(fpr0)} ({f0}/{g0}) | {_pct(fpr1)} ({f1}/{g1}) |",
         f"| PPL on control texts | {ppl0 and round(ppl0, 2)} | {ppl1 and round(ppl1, 2)} |",
         f"| Wilcoxon p (rep4 / distinct4 / collapse) | — | {tests['rep4']} / {tests['distinct4']} / {tests['collapse']} |",
         f"| changed greedy generations (of {len(bad)} bad / {len(good)} good) | — | "
         f"{n_changed['bad']} / {n_changed['good']} |",
         ""]
    if tw_controls:
        L += ["| edited matrix | TW control | violations | target shift (TW units) |",
              "|---|---|---|---|"]
        for c in tw_controls:
            L.append(f"| L{c['layer']} {c['matrix']} | {c['verdict']} | {c['n_violations']} "
                     f"| {c['target_shift_tw_units']:.2f} |")
        L += ["",
              "TW control: healthy top singular values must stay within +-2 TW-scale units "
              "(plan section 7.2); any component shifted by more than 3 units is a violation."]
    md = os.path.join(out_dir, "rcda_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L), flush=True)
    print(f"\nsaved: {os.path.join(out_dir, 'rcda_benchmark.json')}\nsaved: {md}", flush=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Before/after benchmark for RMT interventions.")
    ap.add_argument("--model_path", default="model",
                    help="directory with the model files (config.json, model.safetensors, tokenizer)")
    ap.add_argument("--out_dir", default="results", help="output directory for results.json / summaries")
    ap.add_argument("--max_new", type=int, default=40, help="max new tokens per generation")
    ap.add_argument("--n_prompts", type=int, default=0, help="0 = use all prompts")
    ap.add_argument("--configs", default="", help="comma-separated config names, empty = all")
    ap.add_argument("--seed", type=int, default=0)
    # ---- RCDA mode (plan, sections 6-7): weight-edit before/after evaluation ----
    ap.add_argument("--baseline_dir", default=None,
                    help="RCDA mode: model directory with the BASE weights")
    ap.add_argument("--edited_dir", default=None,
                    help="RCDA mode: model directory with the EDITED weights "
                         "(activates the RCDA benchmark instead of the hook benchmark)")
    ap.add_argument("--bad_set", default="prompts/bad.jsonl",
                    help="RCDA mode: JSONL prompts that trigger the target defect")
    ap.add_argument("--good_set", default="prompts/good.jsonl",
                    help="RCDA mode: JSONL prompts where the defect must NOT appear (FPR)")
    ap.add_argument("--control_set", default=None,
                    help="RCDA mode: JSONL texts for the PPL ability control "
                         "(default: the built-in VAL_TEXTS)")
    ap.add_argument("--rep_threshold", type=float, default=0.2,
                    help="Repetition Rate-4 above which a generation counts as a repetition defect")
    ap.add_argument("--collapse_frac", type=float, default=0.8,
                    help="a 4-gram collapse before this fraction of max_new counts as a defect")
    ap.add_argument("--rcda_report", default=None,
                    help="path to rcda_edit_report.json (default: <edited_dir>/rcda_edit_report.json); "
                         "its matrices get the spectral TW control")
    args = ap.parse_args()

    if args.edited_dir:                       # RCDA branch — weight-edit evaluation
        run_rcda_benchmark(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = PROMPTS if args.n_prompts == 0 else PROMPTS[:args.n_prompts]

    cfgs = [
        {"name": "baseline", "mode": "none"},
        {"name": "clip", "mode": "clip"},
        {"name": "dampen_a0.9", "mode": "dampen", "alpha": 0.9},
        {"name": "dampen_a0.5", "mode": "dampen", "alpha": 0.5},
        {"name": "dampen_a0.1", "mode": "dampen", "alpha": 0.1},
        {"name": "project_k16", "mode": "project", "top_k": 16},
        # the random-ablation config is added after clip: top_k = the mean number
        # of singular values modified by clip (identical intervention cardinality)
        {"name": "random_abl", "mode": "random", "top_k": 10, "_late": True},
    ]
    if args.configs:
        want = [c.strip() for c in args.configs.split(",")]
        cfgs = [c for c in cfgs if c["name"] in want]

    json_path = os.path.join(args.out_dir, "results.json")
    done = {}
    if os.path.exists(json_path):                     # resume
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        done = {r["config"]["name"]: r for r in old.get("runs", [])}
        print(f"resume: configurations already finished: {list(done)}", flush=True)

    print(f"Loading model from {args.model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
    model.eval()
    print(f"Decoder layers: {len(model.model.layers)}", flush=True)

    runs = [done[n] for n in done]

    def save(runs):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {
                "model_path": args.model_path, "max_new": args.max_new,
                "seed": args.seed, "n_prompts": len(prompts),
                "torch": torch.__version__, "use_cache": False,
                "date": time.strftime("%Y-%m-%d %H:%M")},
                "runs": runs}, f, ensure_ascii=False, indent=1, default=_clean)

    # baseline and clip must run first (clip provides the cardinality for the random ablation)
    order = [c for c in cfgs if not c.get("_late")] + [c for c in cfgs if c.get("_late")]
    for cfg in order:
        if cfg["name"] in done:
            continue
        print(f"\n=== configuration {cfg['name']} ===", flush=True)
        t0 = time.time()
        run = run_config(model, tok, cfg, prompts, args.max_new)
        run["wall_s"] = round(time.time() - t0, 1)
        runs.append(run)
        done[cfg["name"]] = run
        save(runs)
        print(f"  done in {run['wall_s']} s, results.json saved", flush=True)
        if cfg["name"] == "clip":
            hs = run["hook_stats"]
            k_abl = int(round(float(np.mean([h["mean_intervened"] for h in hs]))))
            k_abl = max(1, k_abl)
            print(f"  -> ablation: random mask top_k={k_abl} "
                  f"(mean number of modified singular values in clip mode)", flush=True)
            for c in order:
                if c["name"] == "random_abl":
                    c["top_k"] = k_abl

    # ---------------------------------------------------------- summary & tests
    base = done.get("baseline")
    base_ppl = None
    if base:
        pp = [p for p in base["ppl"] if p is not None and math.isfinite(p)]
        base_ppl = float(np.mean(pp)) if pp else None

    summaries, pvals = [], {}
    for run in runs:
        agg, collapses = config_summary(run, base_ppl)
        summaries.append(agg)
        if base and run["config"]["name"] != "baseline":
            base_collapses = [r["collapse"] if r["collapse"] is not None else r["n_tokens"] + 1
                              for r in base["rows"]]
            pvals[run["config"]["name"]] = {
                "distinct4": paired_p([r["distinct4"] or 0 for r in base["rows"]],
                                      [r["distinct4"] or 0 for r in run["rows"]]),
                "rep4": paired_p([r["rep4"] or 0 for r in base["rows"]],
                                 [r["rep4"] or 0 for r in run["rows"]]),
                "collapse": paired_p(base_collapses, collapses),
            }
            pvals[run["config"]["name"]] = {k: (round(v, 4) if v is not None else "—")
                                            for k, v in pvals[run["config"]["name"]].items()}

    csv_path = os.path.join(args.out_dir, "summary.csv")
    cols = ["name", "mean_distinct4", "mean_distinct2", "mean_rep4", "median_collapse",
            "mean_ppl", "ppl_ratio_vs_base", "mean_tokens",
            "hook_mean_intervened", "hook_energy_kept", "hook_max_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in summaries:
            w.writerow({c: (_clean(s.get(c)) if s.get(c) is not None else "") for c in cols})

    write_summary_md(os.path.join(args.out_dir, "summary.md"), summaries, pvals, args)
    save(runs)
    with open(os.path.join(args.out_dir, "wilcoxon.json"), "w", encoding="utf-8") as f:
        json.dump(pvals, f, ensure_ascii=False, indent=1)
    print("\n=== SUMMARY ===", flush=True)
    print(open(os.path.join(args.out_dir, "summary.md"), encoding="utf-8").read(), flush=True)


if __name__ == "__main__":
    main()
