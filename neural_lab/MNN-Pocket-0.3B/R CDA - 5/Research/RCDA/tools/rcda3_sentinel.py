#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-3 stage 1 — the spectral sentinel: online detection and rescue.

Round-2 finding: the windowed participation ratio (PR) of the residual stream
drops monotonically BEFORE the text-level collapse (fig9) and by ~35-40% inside
the loop. This script turns that observation into an online controller:

  1. Calibration (no generation): teacher-forced forward passes over the 5
     fluent control texts give the healthy per-layer distribution of windowed PR
     for two window scales (W=8 "fast", W=16 "slow") at sentinel layers
     L15/16/17. Threshold theta = the q10 quantile of that healthy pool.
     (Rationale: only 2 saved non-looping generations are >=8 tokens long, so
     the healthy reference must come from teacher-forced fluent text.)

  2. Detection pass: re-generate the 40 round-2 probe generations greedily
     (KV cache, per-step hidden states), maintain per-layer deques of generated
     token states, compute PR each step, alarm when PR < theta for 2
     consecutive steps at any (layer, scale). Reported against the a-priori
     token labels: TPR on looping gens, detection delay vs onset, alarm rate on
     the teacher-forced control streams (FPR proxy), and a threshold-free ROC
     (score = max z-score over layers/scales; label = collapse within 4 tokens).

  3. Rescue arms on the 25 looping generations (label ground truth, not the
     clean_gen field: 5 jsonl records carry clean_gen=False with zero loop
     labels — bookkeeping bug of round 2, corrected here):
       ban   — at alarm, ban the token that continued the current 3-gram last
               time (breaks the repeating 4-gram) for 4 steps;
       steer — at alarm, subtract the behavioral loop direction Delta_h from the
               residual stream at the writer layers L15/16/18 for 8 steps;
       both  — the two combined.
     Rescue success = no new loop 4-grams after the alarm step and the
     generation continues for >=8 more tokens (or hits EOS cleanly).

  4. Benchmark arms on the 20 bad prompts: A = baseline (no sentinel),
     C = the winning rescue arm with the sentinel active; FP-cost pass on the
     20 good prompts with the sentinel active; transfer check = detection-only
     pass on the RCDA-2 edited model with the SAME thresholds.

Outputs: rcda_work/rcda3/rcda3_sentinel.json + rcda3_sentinel_generations.jsonl
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

WORK = "/home/z/my-project/rcda_work"
sys.path.insert(0, os.path.join(WORK, "Code"))

from benchmark_rmt import _defect_flags, _load_model_dir  # noqa: E402
from pocket_rmt_probe import _read_prompt_set  # noqa: E402

MODEL_DIR = os.path.join(WORK, "model")
EDITED_DIR = os.path.join(WORK, "edited_model_v2")
OUT = os.path.join(WORK, "rcda3")
os.makedirs(OUT, exist_ok=True)

SENT_LAYERS = [15, 16, 17]          # fig9 precursor layers
STEER_LAYERS = [15, 16, 18]         # round-2 writer layers (o_proj)
SCALES = [8, 16]                    # fast / slow window
THETA_Q = 0.10                      # healthy-pool quantile for the alarm edge
CONSEC = 2                          # consecutive below-theta steps to alarm
GAMMA = 1.0                         # steering strength (units of Delta_h)
K_STEER = 8                         # steering duration, tokens
K_BAN = 4                           # n-gram ban duration, tokens
MAX_INTERVENTIONS = 4               # cap per generation
MAX_NEW = 96
NGRAM = 4


# --------------------------------------------------------------------- utils

def token_loop_labels(ids, ngram=NGRAM):
    """Per-token boolean: the n-gram ending here already appeared earlier."""
    labels = np.zeros(len(ids), dtype=bool)
    seen = {}
    for t in range(len(ids)):
        if t + 1 >= ngram:
            g = tuple(ids[t - ngram + 1:t + 1])
            if g in seen and seen[g] <= t - ngram:
                labels[t] = True
            if g not in seen:
                seen[g] = t
    return labels


def pr_of_window(X):
    """Participation ratio of a (T, H) window: (sum s^2)^2 / sum s^4."""
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    return float((p.sum()) ** 2 / max((p ** 2).sum(), 1e-12))


class Sentinel:
    """Online PR monitor + optional rescue controller over a cached loop."""

    def __init__(self, calib, rescue=None):
        # calib: {(W, layer): {"mu":..., "sigma":..., "theta":...}}
        self.calib = calib
        self.rescue = rescue            # None | "ban" | "steer" | "both"
        self.deques = {l: [] for l in SENT_LAYERS}
        self.below = {(w, l): 0 for w in SCALES for l in SENT_LAYERS}
        self.steer_steps = 0
        self.ban_steps = 0
        self.alarm_steps = []           # 0-based generated-token index
        self.n_interventions = 0
        self.gram_next = {}             # 3-gram -> observed continuation
        self.traces = {(w, l): [] for w in SCALES for l in SENT_LAYERS}

    def reset(self):
        self.deques = {l: [] for l in SENT_LAYERS}
        self.below = {(w, l): 0 for w in SCALES for l in SENT_LAYERS}
        self.steer_steps = 0
        self.ban_steps = 0
        self.alarm_steps = []
        self.n_interventions = 0
        self.gram_ctx = []
        self.gram_next = {}
        self.traces = {(w, l): [] for w in SCALES for l in SENT_LAYERS}

    def observe(self, hs_new, n_gen):
        """Feed the new token's per-layer hidden states (after the forward).
        Returns True if an alarm fired at this step."""
        for l in SENT_LAYERS:
            v = hs_new[l + 1][0, -1].float().numpy()
            dq = self.deques[l]
            dq.append(v)
            if len(dq) > max(SCALES):
                dq.pop(0)
        fired = False
        if n_gen >= 1:
            for w in SCALES:
                for l in SENT_LAYERS:
                    dq = self.deques[l]
                    if len(dq) < w:
                        self.traces[(w, l)].append(None)
                        continue
                    pr = pr_of_window(np.stack(dq[-w:]))
                    self.traces[(w, l)].append(pr)
                    if pr < self.calib[(w, l)]["theta"]:
                        self.below[(w, l)] += 1
                        if self.below[(w, l)] >= CONSEC:
                            fired = True
                    else:
                        self.below[(w, l)] = 0
            if fired:
                self.alarm_steps.append(n_gen - 1)
                if self.rescue and self.n_interventions < MAX_INTERVENTIONS:
                    self.n_interventions += 1
                    if self.rescue in ("ban", "both"):
                        # arm the ban only when the loop's continuation token
                        # is actually known (the 4-gram already repeated)
                        if self.gram_next.get(tuple(self.gram_ctx)) is not None:
                            self.ban_steps = K_BAN
                    if self.rescue in ("steer", "both"):
                        self.steer_steps = K_STEER
        return fired

    def steer_active(self):
        return self.steer_steps > 0

    def tick(self):
        if self.steer_steps > 0:
            self.steer_steps -= 1
        if self.ban_steps > 0:
            self.ban_steps -= 1

    def ban_token(self, logits):
        """Return the logit-banned loop-continuation token, or None."""
        if self.ban_steps <= 0 or len(self.gram_next) == 0:
            return None
        ctx = tuple(self.gram_ctx)
        banned = self.gram_next.get(ctx)
        if banned is None:
            return None
        logits[0, banned] = float("-inf")
        return banned

    def note_token(self, tok_id):
        """Record the continuation of the CURRENT 3-gram context for future
        bans: map(last-3 generated tokens) -> tok_id, first occurrence wins,
        so the ban always targets the loop's established continuation."""
        if len(self.gram_ctx) == 3:
            self.gram_next.setdefault(tuple(self.gram_ctx), tok_id)
        self.gram_ctx.append(tok_id)
        if len(self.gram_ctx) > 3:
            self.gram_ctx.pop(0)


class SentinelLoop:
    """Greedy cached generation loop with the sentinel + rescue attached."""

    def __init__(self, model, tok, calib, rescue=None, hooks=None):
        self.model, self.tok = model, tok
        self.sent = Sentinel(calib, rescue=rescue)
        self.hooks = hooks              # list of SteerHook or None

    @torch.no_grad()
    def run(self, prompt, max_new=MAX_NEW, collect_traces=False):
        torch.manual_seed(0)
        self.sent.reset()
        if self.hooks:
            for h in self.hooks:
                h.reset()
        eos = self.tok.eos_token_id
        enc = self.tok(prompt, return_tensors="pt")
        out = self.model(**enc, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        ids = []
        nxt = int(out.logits[0, -1].argmax())
        while nxt != eos and len(ids) < max_new:
            # ---- pre-sample interventions for THIS step ------------------
            if self.sent.ban_steps > 0:
                logits = out.logits[0, -1].clone()
                banned = self.sent.ban_token(logits.reshape(1, -1))
                if banned is not None:
                    nxt = int(logits.argmax())
                    if nxt == eos:
                        break
            ids.append(nxt)
            self.sent.note_token(nxt)
            self.sent.tick()
            for h in self.hooks or []:
                h.tick()
            # ---- forward the new token -----------------------------------
            out = self.model(input_ids=torch.tensor([[nxt]]), past_key_values=past,
                             use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            fired = self.sent.observe(out.hidden_states, len(ids))
            if fired and self.hooks:
                # the Sentinel counter alone does nothing: the SteerHook
                # objects hold their own step counters and must be armed here
                for h in self.hooks:
                    h.steps = K_STEER
            nxt = int(out.logits[0, -1].argmax())
        labels = token_loop_labels(ids)
        collapse = int(np.argmax(labels)) if labels.any() else None
        text = self.tok.decode(ids, skip_special_tokens=True)
        rec = {"ids": ids, "labels": [int(x) for x in labels],
               "collapse": collapse, "text": text,
               "alarm_steps": self.sent.alarm_steps,
               "n_interventions": self.sent.n_interventions}
        if collect_traces:
            rec["pr_traces"] = {f"W{w}_L{l}": [None if v is None else round(v, 4)
                                               for v in self.sent.traces[(w, l)]]
                                for w in SCALES for l in SENT_LAYERS}
        return rec


class SteerHook:
    """Subtract gamma * Delta_h from the last-position residual at a layer."""

    def __init__(self, layer_idx, vec, gamma=GAMMA):
        self.layer_idx = layer_idx
        self.vec = torch.tensor(vec, dtype=torch.float32)
        self.gamma = float(gamma)
        self.steps = 0

    def reset(self):
        self.steps = 0

    def arm(self, k=K_STEER):
        self.steps = k

    def tick(self):
        if self.steps > 0:
            self.steps -= 1

    def __call__(self, module, inputs, output):
        if self.steps <= 0:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        hs[:, -1, :] -= self.gamma * self.vec.to(hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs


# ---------------------------------------------------------------- calibration

@torch.no_grad()
def calibrate(model, tok, ctrl_items):
    """Healthy windowed-PR distribution from teacher-forced control texts."""
    pool = {(w, l): [] for w in SCALES for l in SENT_LAYERS}
    for it in ctrl_items:
        enc = tok(it["text"], return_tensors="pt", truncation=True, max_length=512)
        hs = model(**enc, output_hidden_states=True).hidden_states
        T = hs[1].shape[1]
        for l in SENT_LAYERS:
            X = hs[l + 1][0].float().numpy()
            for w in SCALES:
                for t in range(w, T + 1):
                    pool[(w, l)].append(pr_of_window(X[t - w:t]))
        del hs
    calib = {}
    for (w, l), vals in pool.items():
        a = np.array(vals)
        calib[(w, l)] = {"mu": float(a.mean()), "sigma": float(a.std()),
                         "theta": float(np.quantile(a, THETA_Q)),
                         "q05": float(np.quantile(a, 0.05)),
                         "q20": float(np.quantile(a, 0.20)), "n": len(a)}
    return calib


def auc_mw(scores_pos, scores_neg):
    """Mann-Whitney AUC without sklearn."""
    if not scores_pos or not scores_neg:
        return None
    allv = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    sa = np.asarray(allv)[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[:len(scores_pos)].sum()
    return float((rp - len(scores_pos) * (len(scores_pos) + 1) / 2)
                 / (len(scores_pos) * len(scores_neg)))


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "rescue", "bench"])
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse detect/ban/steer records already in the jsonl")
    args = ap.parse_args()
    t0 = time.time()

    model, tok = _load_model_dir(MODEL_DIR)
    H = model.config.hidden_size
    print(f"model loaded H={H} ({time.time()-t0:.0f}s)", flush=True)

    gens_file = os.path.join(OUT, "rcda3_sentinel_generations.jsonl")
    if args.smoke:
        pass
    elif args.resume and os.path.exists(gens_file):
        # drop incomplete/placebo arm records, keep detect + complete arms
        recs = [json.loads(l) for l in open(gens_file, encoding="utf-8")]
        by_arm = {}
        for r in recs:
            by_arm.setdefault(r["arm"], []).append(r)
        keep = []
        for arm, rs in by_arm.items():
            if arm == "detect" or len(rs) >= 30:
                keep.extend(rs)
        with open(gens_file, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"resume: kept {len(keep)} records "
              f"({ {a: len(rs) for a, rs in by_arm.items() if len(rs) >= 30} })",
              flush=True)
    else:
        open(gens_file, "w").close()   # fresh full run: truncate the jsonl

    # ------------------------------------------------ 1. calibration
    ctrl = _read_prompt_set(os.path.join(WORK, "prompts/control.jsonl"))
    calib = calibrate(model, tok, ctrl)
    for (w, l), c in sorted(calib.items()):
        print(f"  calib W{w} L{l}: mu={c['mu']:.3f} sigma={c['sigma']:.3f} "
              f"theta={c['theta']:.3f} (n={c['n']})", flush=True)

    # -------------------------------------------- 2. detection pass
    bad = _read_prompt_set(os.path.join(WORK, "prompts/bad.jsonl"))
    good = _read_prompt_set(os.path.join(WORK, "prompts/good.jsonl"))
    probe_items = bad + good

    # round-2 saved generations: sanity reference for deterministic replay
    saved = {}
    if os.path.exists(os.path.join(WORK, "rcda2/rcda2_generations.jsonl")):
        for line in open(os.path.join(WORK, "rcda2/rcda2_generations.jsonl")):
            g = json.loads(line)
            saved[g["id"]] = g

    detector = SentinelLoop(model, tok, calib, rescue=None)
    det_records = []
    n_match = 0
    saved_recs = []
    if args.resume and os.path.exists(gens_file):
        saved_recs = [json.loads(l) for l in open(gens_file, encoding="utf-8")]
    det_saved = [r for r in saved_recs if r["arm"] == "detect"]
    if len(det_saved) >= len(probe_items):
        det_records = det_saved
        n_match = sum(r["replay_match"] for r in det_records)
        print(f"resume: loaded {len(det_records)} detect records "
              f"(match {n_match})", flush=True)
    else:
        for i, it in enumerate(probe_items):
            rec = detector.run(it["text"], max_new=args.max_new, collect_traces=True)
            rec["id"], rec["split"], rec["prompt"] = it["id"], it["id"].split("_")[0], it["text"]
            rec["arm"] = "detect"
            s = saved.get(it["id"])
            rec["replay_match"] = bool(s and s["text"] == rec["text"])
            n_match += rec["replay_match"]
            det_records.append(rec)
            with open(gens_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [detect {i+1}/{len(probe_items)}] {it['id']}: n={len(rec['ids'])} "
                  f"collapse={rec['collapse']} alarm={rec['alarm_steps'][:2]} "
                  f"match={rec['replay_match']} ({time.time()-t0:.0f}s)", flush=True)
            if args.smoke and i >= 3:
                break
    print(f"replay text match: {n_match}/{len(det_records)}", flush=True)

    # alarm statistics vs a-priori labels
    det_summary = {"replay_match": n_match, "n_gens": len(det_records)}
    looping = [r for r in det_records if r["collapse"] is not None]
    clean = [r for r in det_records if r["collapse"] is None]
    alarmed_loop = [r for r in looping if r["alarm_steps"]]
    det_summary["looping"] = len(looping)
    det_summary["clean"] = len(clean)
    det_summary["tpr_any_alarm"] = len(alarmed_loop) / max(len(looping), 1)
    delays, leads = [], []
    for r in alarmed_loop:
        a = r["alarm_steps"][0]
        (leads if a < r["collapse"] else delays).append(
            r["collapse"] - a if a < r["collapse"] else a - r["collapse"])
    det_summary["predictive_alarms"] = len(leads)
    det_summary["lead_tokens"] = {"median": float(np.median(leads)) if leads else None,
                                  "values": leads}
    det_summary["onset_locked_alarms"] = len(delays)
    det_summary["delay_tokens"] = {"median": float(np.median(delays)) if delays else None,
                                   "values": delays}
    # clean-side: teacher-forced ctrl streams (FPR proxy) — alarm rate on
    # sliding windows is already encoded by theta=q10; report the share of
    # generated clean gens (all lengths) with any alarm:
    det_summary["alarm_on_clean_gens"] = sum(1 for r in clean if r["alarm_steps"])
    # waste: loop tokens in detect arm (= baseline waste)
    det_summary["loop_token_waste_mean"] = float(
        np.mean([sum(r["labels"]) for r in looping])) if looping else 0.0
    print("detection:", {k: v for k, v in det_summary.items()
                         if not isinstance(v, dict)}, flush=True)

    # ROC: score = max z over (layer, scale); label = collapse within 4 tokens
    roc = {}
    for w in SCALES:
        pos, neg = [], []
        for r in det_records:
            if "pr_traces" not in r:
                continue
            c = r["collapse"]
            for t in range(w - 1, len(r["ids"])):
                zs = []
                for l in SENT_LAYERS:
                    v = r["pr_traces"][f"W{w}_L{l}"][t]
                    if v is not None:
                        cb = calib[(w, l)]
                        zs.append((cb["mu"] - v) / max(cb["sigma"], 1e-9))
                if not zs:
                    continue
                score = max(zs)
                if c is not None and t < c <= t + 4:
                    pos.append(score)
                elif c is None or t + 4 < c:
                    neg.append(score)
        a = auc_mw(pos, neg)
        roc[f"W{w}"] = {"auc": a, "n_pos": len(pos), "n_neg": len(neg)}
        print(f"  ROC W{w}: AUC={a:.3f} (pos={len(pos)}, neg={len(neg)})",
              flush=True)
    det_summary["roc"] = roc

    if args.smoke or args.stage == "detect":
        json.dump({"calib": {f"W{w}_L{l}": c for (w, l), c in calib.items()},
                   "detection": det_summary},
                  open(os.path.join(OUT, "rcda3_sentinel.json"), "w"),
                  indent=1, default=float)
        print("smoke/detect-only: saved partial json", flush=True)
        return

    # ------------------------------------------------- 3. rescue arms
    vectors = {}
    for l in STEER_LAYERS:
        vectors[l] = torch.load(
            os.path.join(WORK, "rcda2/vectors", f"L{l}_delta_h_behav.pt"),
            weights_only=True).numpy().astype(np.float32)
    handles = []
    hooks = [SteerHook(l, vectors[l]) for l in STEER_LAYERS]
    for l, h in zip(STEER_LAYERS, hooks):
        handles.append(model.model.layers[l].register_forward_hook(h))

    looping_ids = [r["id"] for r in looping]
    if args.smoke:
        looping_ids = looping_ids[:2]
    id2item = {it["id"]: it for it in probe_items}
    arm_stats = {}
    for arm in ("ban", "steer", "both"):
        # resume: reuse complete arm records from the jsonl
        prev = [r for r in saved_recs if r["arm"] == f"rescue_{arm}"] \
            if args.resume else []
        if len(prev) >= len(looping_ids):
            print(f"resume: reusing {len(prev)} records of arm {arm}", flush=True)
        else:
            rescue_loop = SentinelLoop(model, tok, calib, rescue=arm,
                                       hooks=hooks if arm in ("steer", "both") else None)
            prev = []
            for gid in looping_ids:
                rec = rescue_loop.run(id2item[gid]["text"], max_new=args.max_new)
                rec["id"], rec["split"], rec["prompt"] = gid, gid.split("_")[0], id2item[gid]["text"]
                rec["arm"] = f"rescue_{arm}"
                with open(gens_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({k: rec[k] for k in rec if k != "pr_traces"},
                                       ensure_ascii=False) + "\n")
                prev.append(rec)
        stats = {"n": 0, "rescued": 0, "waste": [], "clean_tail": [],
                 "n_interventions": []}
        for i, rec in enumerate(prev):
            alarm0 = rec["alarm_steps"][0] if rec["alarm_steps"] else None
            last_loop = max([j for j, x in enumerate(rec["labels"]) if x], default=-1)
            tail = len(rec["ids"]) - 1 - alarm0 if alarm0 is not None else 0
            ended_early = len(rec["ids"]) < args.max_new     # hit EOS cleanly
            rescued = bool(alarm0 is not None and last_loop <= alarm0 + 1
                           and (tail >= 8 or ended_early))
            stats["n"] += 1
            stats["rescued"] += int(rescued)
            stats["waste"].append(sum(rec["labels"]))
            stats["clean_tail"].append(len(rec["ids"]) - 1 - last_loop)
            stats["n_interventions"].append(rec["n_interventions"])
            if (i + 1) % 8 == 0:
                print(f"  [rescue {arm} {i+1}/{len(prev)}] "
                      f"rescued so far {stats['rescued']} ({time.time()-t0:.0f}s)",
                      flush=True)
        arm_stats[arm] = {
            "n": stats["n"], "rescued": stats["rescued"],
            "rescue_rate": stats["rescued"] / max(stats["n"], 1),
            "loop_waste_mean": float(np.mean(stats["waste"])),
            "clean_tail_median": float(np.median(stats["clean_tail"])),
            "interventions_mean": float(np.mean(stats["n_interventions"])),
        }
        print(f"arm {arm}: {arm_stats[arm]}", flush=True)

    for h in handles:
        h.remove()

    winner = max(arm_stats, key=lambda a: (arm_stats[a]["rescue_rate"],
                                           arm_stats[a]["clean_tail_median"]))
    print(f"winner rescue arm: {winner}", flush=True)

    # --------------------------------------------- 4. benchmark arms
    bench_bad, bench_good = bad, good
    if args.smoke:
        bench_bad, bench_good = bad[:3], good[:3]

    def bench(model_obj, rescue_arm, split_items, tag, hooks_=None):
        lp = SentinelLoop(model_obj, tok, calib, rescue=rescue_arm, hooks=hooks_)
        rows = []
        for it in split_items:
            rec = lp.run(it["text"], max_new=args.max_new)
            flags = _defect_flags(rec["text"], len(rec["ids"]), args.max_new, 0.2, 0.8)
            rows.append({"id": it["id"], "arm": tag,
                         "n_tokens": len(rec["ids"]), "text": rec["text"],
                         "alarm_steps": rec["alarm_steps"],
                         "n_interventions": rec["n_interventions"],
                         "collapse": rec["collapse"],
                         "loop_tokens": int(sum(rec["labels"])), **flags})
        rho = float(np.mean([r["defect_hit"] for r in rows]))
        return rows, rho

    rows_a, rho_a = bench(model, None, bench_bad, "A_baseline")
    print(f"arm A (baseline, {len(bench_bad)} bad): rho={rho_a:.2f}", flush=True)

    hooks_c = None
    if winner in ("steer", "both"):
        hooks_c = [SteerHook(l, vectors[l]) for l in STEER_LAYERS]
        handles_c = [model.model.layers[l].register_forward_hook(h)
                     for l, h in zip(STEER_LAYERS, hooks_c)]
    rows_c, rho_c = bench(model, winner, bench_bad, f"C_sentinel_{winner}", hooks_c)
    if hooks_c:
        for h in handles_c:
            h.remove()
    print(f"arm C (sentinel {winner}): rho={rho_c:.2f}", flush=True)

    rows_fp, rho_fp = bench(model, winner, bench_good, f"FP_good_{winner}", hooks_c)
    # FP cost: on good prompts compare against the clean-detect texts
    det_by_id = {r["id"]: r["text"] for r in det_records}
    fp_changed = sum(1 for r in rows_fp if det_by_id.get(r["id"]) not in (None, r["text"]))
    fp_looping = sum(1 for r in rows_fp if r["collapse"] is not None)
    print(f"FP pass (good, sentinel on): changed={fp_changed}/{len(rows_fp)} "
          f"looping={fp_looping}", flush=True)

    # transfer: detection-only on the RCDA-2 edited model (free the base first:
    # two fp32 copies do not fit in the sandbox RAM)
    del model, detector, rescue_loop
    gc.collect()
    ed_model, _ = _load_model_dir(EDITED_DIR)
    rows_t, rho_t = bench(ed_model, None, bench_bad, "T_edited_detect")
    t_loop = [r for r in rows_t if r["collapse"] is not None]
    t_alarmed = [r for r in t_loop if r["alarm_steps"]]
    t_waste = float(np.mean([r["loop_tokens"] for r in t_loop])) if t_loop else 0.0
    print(f"transfer (edited model, detect-only): rho={rho_t:.2f} "
          f"tpr={len(t_alarmed)}/{len(t_loop)} waste={t_waste:.1f}", flush=True)

    result = {
        "meta": {"theta_q": THETA_Q, "consec": CONSEC, "scales": SCALES,
                 "sent_layers": SENT_LAYERS, "steer_layers": STEER_LAYERS,
                 "gamma": GAMMA, "k_steer": K_STEER, "k_ban": K_BAN,
                 "max_new": args.max_new, "ngram": NGRAM,
                 "elapsed_s": round(time.time() - t0, 1)},
        "calib": {f"W{w}_L{l}": c for (w, l), c in calib.items()},
        "detection": det_summary,
        "rescue_arms": arm_stats,
        "winner": winner,
        "benchmark": {
            "A_baseline": {"rho": rho_a, "rows": rows_a},
            "C_sentinel": {"arm": winner, "rho": rho_c, "rows": rows_c},
            "FP_good": {"changed": fp_changed, "looping": fp_looping,
                        "rho": rho_fp, "rows": rows_fp},
            "T_edited_transfer": {"rho": rho_t, "tpr": len(t_alarmed),
                                  "n_loop": len(t_loop), "waste": t_waste,
                                  "rows": rows_t},
        },
    }
    with open(os.path.join(OUT, "rcda3_sentinel.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"saved {OUT}/rcda3_sentinel.json  total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
