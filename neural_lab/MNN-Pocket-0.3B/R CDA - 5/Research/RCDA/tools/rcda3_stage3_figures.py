#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RCDA-3 stage 3 — figures + digest + comparison table.

fig11_sentinel.png   (a) PR-8 traces with alarm/collapse markers;
                     (b) alarm timing vs collapse onset (lead<0 / delay>0);
                     (c) ROC of the max-z score for W=8 and W=16;
                     (d) rescue arms: rate + loop-token waste.
fig12_heads.png      (a) per-head concentration of the behavioral direction;
                     (b) head-surgery calibration: defect rate + n_changed;
                     (c) combo vs placebo: KL / rel-Frobenius / TW verdicts.
fig13_therapy.png    vaccination-vs-cure: defect rate of arms
                     A baseline / B permanent edit / C sentinel / D heads,
                     plus loop-token waste and the FP cost of the sentinel.

Inputs: rcda_work/rcda3/rcda3_sentinel.json, rcda3_heads.json,
        rcda3_sentinel_generations.jsonl (arm=detect records with traces),
        rcda_work/bench_v2/rcda_benchmark.json (arm B numbers).
Outputs: rcda_work/rcda3/figures/*.png, rcda3_comparison.json,
         rcda3_digest.md
"""
import json
import os

import matplotlib.font_manager as fm
import numpy as np

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

WORK = "/home/z/my-project/rcda_work"
R3 = os.path.join(WORK, "rcda3")
FIG = os.path.join(R3, "figures")
os.makedirs(FIG, exist_ok=True)

SENT = json.load(open(os.path.join(R3, "rcda3_sentinel.json")))
HEADS = None
if os.path.exists(os.path.join(R3, "rcda3_heads.json")):
    HEADS = json.load(open(os.path.join(R3, "rcda3_heads.json")))
EXPLORE = None
if os.path.exists(os.path.join(R3, "rcda3_explore.json")):
    EXPLORE = json.load(open(os.path.join(R3, "rcda3_explore.json")))
BENCH2 = json.load(open(os.path.join(WORK, "bench_v2/rcda_benchmark.json")))
GENS = [json.loads(l) for l in
        open(os.path.join(R3, "rcda3_sentinel_generations.jsonl"))]

DET = [g for g in GENS if g.get("arm") == "detect" and "pr_traces" in g]
SENT_LAYERS = [15, 16, 17]
SCALES = [8, 16]


def auc_curve(pos, neg):
    """TPR/FPR curve sweeping the threshold over pooled scores."""
    allv = np.unique(np.concatenate([pos, neg]))
    thrs = np.concatenate([[allv[0] - 1e-9], allv])
    tpr, fpr = [], []
    for thr in thrs:
        tpr.append(float(np.mean(pos >= thr)))
        fpr.append(float(np.mean(neg >= thr)))
    order = np.argsort(fpr)
    return np.array(fpr)[order], np.array(tpr)[order]


def auc_mw_static(pos, neg):
    """Mann-Whitney AUC (rank statistic, ties averaged)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    sa = allv[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pooled_scores(w, horizon=4):
    """Per-step max-z scores with the ROC label (collapse within `horizon`
    tokens). The horizon is matched to the window in the primary curves:
    a W-step window necessarily reacts before a W/2-step horizon."""
    calib = SENT["calib"]
    pos, neg = [], []
    for r in DET:
        c = r["collapse"]
        for t in range(w - 1, len(r["ids"])):
            zs = []
            for l in SENT_LAYERS:
                v = r["pr_traces"][f"W{w}_L{l}"][t]
                if v is not None:
                    cb = calib[f"W{w}_L{l}"]
                    zs.append((cb["mu"] - v) / max(cb["sigma"], 1e-9))
            if not zs:
                continue
            s = max(zs)
            if c is not None and t < c <= t + horizon:
                pos.append(s)
            elif c is None or t + horizon < c:
                neg.append(s)
    return np.array(pos), np.array(neg)


# ------------------------------------------------------------------ fig 11
fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)

# (a) example PR-8 traces at L15
ax = axs[0]
looping = [r for r in DET if r["collapse"] is not None]
early = next(r for r in looping if r["collapse"] <= 8)
late = next(r for r in looping if r["collapse"] >= 16)
clean = next((r for r in DET if r["collapse"] is None and len(r["ids"]) >= 14),
             None)
theta8 = SENT["calib"]["W8_L15"]["theta"]
for r, col, lab in [(early, "#d62728", f"early collapse (c={early['collapse']})"),
                    (late, "#1f77b4", f"late collapse (c={late['collapse']})")]:
    tr = r["pr_traces"]["W8_L15"]
    ax.plot(range(len(tr)), tr, color=col, lw=1.4, label=lab)
    ax.axvline(r["collapse"], color=col, ls=":", lw=1)
    if r["alarm_steps"]:
        a = r["alarm_steps"][0]
        ax.scatter([a], [tr[a]] if tr[a] is not None else [theta8],
                   marker="v", color=col, zorder=5)
if clean:
    tr = clean["pr_traces"]["W8_L15"]
    ax.plot(range(len(tr)), tr, color="#7f7f7f", lw=1.2,
            label=f"clean gen ({clean['id'].split('_')[-1]})")
ax.axhline(theta8, color="k", ls="--", lw=1,
           label=r"$\theta$ (q10 healthy)")
ax.set_xlabel("generated token")
ax.set_ylabel("PR-8, layer 15")
ax.set_title("(a) online PR traces")
ax.legend(fontsize=7, loc="lower left")
ax.grid(alpha=0.3)

# (b) alarm timing vs onset
ax = axs[1]
deltas = []
for r in looping:
    if r["alarm_steps"]:
        deltas.append(r["alarm_steps"][0] - r["collapse"])
bins = np.arange(min(deltas) - 0.5, max(deltas) + 1.5, 1.0)
ax.hist(deltas, bins=bins, color="#1f77b4", alpha=0.85)
ax.axvline(0, color="k", ls="--", lw=1)
ax.text(0.02, 0.95, f"lead < 0: {sum(1 for d in deltas if d < 0)}/{len(deltas)}\n"
                    f"median lead {np.median([d for d in deltas if d < 0]):.0f} tok",
        transform=ax.transAxes, va="top", fontsize=8)
ax.set_xlabel("first alarm − collapse onset (tokens)")
ax.set_ylabel("# generations")
ax.set_title(f"(b) alarm timing, TPR={SENT['detection']['tpr_any_alarm']:.2f}")
ax.grid(alpha=0.3)

# (c) ROC — window-matched horizons (primary) + fixed 4-token horizon
ax = axs[2]
for w, hz, col, lw in [(8, 4, "#1f77b4", 1.8), (16, 12, "#d62728", 1.8),
                       (8, 12, "#1f77b4", 0.9), (16, 4, "#d62728", 0.9)]:
    pos, neg = pooled_scores(w, hz)
    fpr, tpr = auc_curve(pos, neg)
    a = auc_mw_static(pos, neg)
    ax.plot(fpr, tpr, color=col, lw=lw,
            label=f"W={w}, h={hz}: AUC={a:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("FPR (steps far from collapse)")
ax.set_ylabel("TPR (collapse within horizon)")
ax.set_title("(c) loop prediction: window-matched horizons")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# (d) rescue arms
ax = axs[3]
arms = SENT["rescue_arms"]
names = list(arms)
rates = [arms[a]["rescue_rate"] for a in names]
waste = [arms[a]["loop_waste_mean"] for a in names]
base_waste = SENT["detection"]["loop_token_waste_mean"]
x = np.arange(len(names))
ax.bar(x - 0.2, rates, 0.4, color="#2ca02c", label="rescue rate")
ax2 = ax.twinx()
ax2.bar(x + 0.2, waste, 0.4, color="#d62728", label="loop tokens (mean)")
ax2.axhline(base_waste, color="#d62728", ls="--", lw=1,
            label=f"no rescue: {base_waste:.0f}")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("rescue rate")
ax2.set_ylabel("loop tokens per gen")
ax.set_title(f"(d) rescue arms (winner: {SENT['winner']})")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
ax.grid(alpha=0.3)
fig.suptitle("RCDA-3 spectral sentinel: online detection and rescue of repetition loops")
fig.savefig(os.path.join(FIG, "fig11_sentinel.png"), dpi=150)
plt.close(fig)
print("fig11 saved", flush=True)

# ------------------------------------------------------------------ fig 12
if HEADS:
    prof = HEADS["head_profiles"]
    conds = HEADS["conditions"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)

    ax = axs[0]
    w = 0.26
    for i, (l, col) in enumerate([("15", "#1f77b4"), ("16", "#ff7f0e"),
                                  ("18", "#2ca02c")]):
        conc = np.array(prof[l]["concentration"])
        xs = np.arange(14) + (i - 1) * w
        ax.bar(xs, conc, w, color=col, label=f"L{l}")
        ax.scatter([prof[l]["top_head"] + (i - 1) * w],
                   [conc[prof[l]["top_head"]]], marker="v", color="k", zorder=5)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform")
    ax.set_xticks(range(14))
    ax.set_xticklabels([f"h{h}" for h in range(14)], fontsize=7)
    ax.set_ylabel("concentration (x uniform)")
    ax.set_title("(a) who writes the loop direction (o_proj)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axs[1]
    names = list(conds)
    hit_bad = [conds[n]["hit_bad"] for n in names]
    changed = [conds[n]["changed_bad"] for n in names]
    cols = ["#d62728" if n == "placebo3" else "#1f77b4" for n in names]
    x = np.arange(len(names))
    ax.bar(x, hit_bad, 0.55, color=cols)
    ax.axhline(HEADS["baseline"]["hit_bad"], color="k", ls="--", lw=1,
               label=f"baseline {HEADS['baseline']['hit_bad']:.2f}")
    for i, (hb, ch) in enumerate(zip(hit_bad, changed)):
        ax.text(i, hb + 0.02, f"{hb:.2f}\n({ch} chg)", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("defect rate, 8 bad cal prompts")
    ax.set_title("(b) head ablation, calibration dose")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axs[2]
    kl = [conds[n]["kl"] for n in names]
    rel = [conds[n]["rel_frobenius"] for n in names]
    tw = [",".join(conds[n]["tw_verdicts"].values()) for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, kl, 0.4, color="#9467bd", label="KL(base||edited)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\ntw={t}\nrel={r:.1e}"
                        for n, t, r in zip(names, tw, rel)], fontsize=6.5)
    ax.set_ylabel("KL on control texts")
    ax.set_title("(c) cost of head ablation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("RCDA-3 head surgery: is the concentration causal?")
    fig.savefig(os.path.join(FIG, "fig12_heads.png"), dpi=150)
    plt.close(fig)
    print("fig12 saved", flush=True)
else:
    print("fig12 skipped (head surgery not run yet)")

# ------------------------------------------------------------------ fig 13
bm = SENT["benchmark"]
arms_rho = {"A\nbaseline": bm["A_baseline"]["rho"],
            "B\npermanent edit": BENCH2["summary"]["rho_edited"],
            f"C\nsentinel({SENT['winner']})": bm["C_sentinel"]["rho"]}
if EXPLORE and EXPLORE.get("benchmark_C"):
    arms_rho[f"C'\n{EXPLORE['final_policy']}"] = EXPLORE["benchmark_C"]["rho"]
if HEADS and HEADS.get("decision_go") and HEADS.get("arm_d"):
    arms_rho["D\nhead combo"] = HEADS["arm_d"]["rho"]

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
ax = axs[0]
x = np.arange(len(arms_rho))
cols = ["#7f7f7f", "#9467bd", "#2ca02c", "#ff7f0e"][:len(arms_rho)]
ax.bar(x, list(arms_rho.values()), 0.55, color=cols)
ax.axhline(BENCH2["summary"]["rho_baseline"], color="k", ls="--", lw=1,
           label=f"baseline rho = {BENCH2['summary']['rho_baseline']:.2f}")
for i, v in enumerate(arms_rho.values()):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(list(arms_rho), fontsize=8)
ax.set_ylabel("defect rate rho (20 bad prompts)")
ax.set_title("(a) therapy arms vs baseline")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axs[1]
waste = {"A": float(np.mean([r["loop_tokens"] for r in bm["A_baseline"]["rows"]])),
         "C": float(np.mean([r["loop_tokens"] for r in bm["C_sentinel"]["rows"]]))}
if EXPLORE and EXPLORE.get("benchmark_C"):
    waste["C'"] = EXPLORE["benchmark_C"]["loop_waste_mean"]
if HEADS and HEADS.get("arm_d"):
    # arm D rows carry collapse but not per-token labels; the loop tail length
    # (n_tokens - collapse) is a lower bound of the in-loop time
    waste["D"] = float(np.mean(
        [(r["n_tokens"] - r["collapse"]) if r["collapse"] is not None else 0
         for r in HEADS["arm_d"]["rows"]]))
waste["T_edited"] = bm["T_edited_transfer"]["waste"]
names = list(waste)
x = np.arange(len(names))
ax.bar(x, [waste[n] for n in names], 0.55,
       color=["#7f7f7f", "#2ca02c", "#ff7f0e", "#9467bd"][:len(names)])
for i, n in enumerate(names):
    ax.text(i, waste[n] + 0.5, f"{waste[n]:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("loop tokens per looping gen")
ax.set_title("(b) time wasted inside the loop")
ax.grid(alpha=0.3, axis="y")

ax = axs[2]
fp = bm["FP_good"]
labels = ["FP changed texts\n(20 good prompts)",
          "FP loops created\n(good prompts)",
          "transfer TPR\n(edited model)"]
vals = [fp["changed"] / 20, fp["looping"] / 20,
        bm["T_edited_transfer"]["tpr"] / max(bm["T_edited_transfer"]["n_loop"], 1)]
ax.bar(np.arange(3), vals, 0.5, color=["#8c564b", "#c49c94", "#17becf"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("(c) sentinel side effects / robustness")
ax.grid(alpha=0.3, axis="y")
fig.suptitle("RCDA-3: vaccination (online sentinel) vs cure (weight edits)")
fig.savefig(os.path.join(FIG, "fig13_therapy.png"), dpi=150)
plt.close(fig)
print("fig13 saved", flush=True)

# ------------------------------------------------------- comparison + digest
comparison = {
    "meta": {"n_bad": 20, "max_new": 96,
             "winner_rescue_arm": SENT["winner"]},
    "arms": {
        "A_baseline": {"rho": bm["A_baseline"]["rho"], "source": "rcda3_sentinel"},
        "B_permanent_edit": {"rho": BENCH2["summary"]["rho_edited"],
                             "ppl": BENCH2["summary"]["ppl_edited"],
                             "source": "rcda2_benchmark"},
        "C_sentinel": {"rho": bm["C_sentinel"]["rho"],
                       "fp_changed": fp["changed"],
                       "fp_looping": fp["looping"],
                       "ppl": "unchanged (no weight edit)",
                       "source": "rcda3_sentinel"},
    },
    "transfer": bm["T_edited_transfer"],
    "rescue_arms": SENT["rescue_arms"],
    "head_surgery": {"decision_go": HEADS.get("decision_go") if HEADS else None,
                     "arm_d": HEADS.get("arm_d") if HEADS else None},
}
if EXPLORE and EXPLORE.get("benchmark_C"):
    comparison["arms"]["C_persistent"] = {
        "arm": EXPLORE["final_policy"],
        "rho": EXPLORE["benchmark_C"]["rho"],
        "loop_waste_mean": EXPLORE["benchmark_C"]["loop_waste_mean"],
        "fp_changed": EXPLORE["rescue_arms"][EXPLORE["final_policy"]]["fp_changed"],
        "fp_loops": EXPLORE["rescue_arms"][EXPLORE["final_policy"]]["fp_loops"],
        "rescue_rate": EXPLORE["rescue_arms"][EXPLORE["final_policy"]]["rescue_rate"],
        "source": "rcda3_explore"}
if HEADS and HEADS.get("arm_d"):
    comparison["arms"]["D_head_combo"] = {
        "rho": HEADS["arm_d"]["rho"],
        "ppl_ctrl_mean": HEADS["arm_d"]["ppl_ctrl_mean"],
        "source": "rcda3_heads"}
json.dump(comparison, open(os.path.join(R3, "rcda3_comparison.json"), "w"),
          indent=1, default=float)

d = SENT["detection"]
lines = [
    "# RCDA-3 digest (all numbers measured, not estimated)\n",
    f"- Calibration: healthy PR pool from 5 teacher-forced control texts "
    f"(q10 thresholds; W8 L15 theta={SENT['calib']['W8_L15']['theta']:.2f}).",
    f"- Detection: TPR {d['tpr_any_alarm']:.2f} on {d['looping']} looping gens; "
    f"predictive alarms {d['predictive_alarms']}/{len(d['lead_tokens']['values']) + len(d['delay_tokens']['values'])} "
    f"(median lead {d['lead_tokens']['median']} tokens); ROC AUC (window-matched "
    f"horizons) W8/h4={auc_mw_static(*pooled_scores(8, 4)):.3f}, "
    f"W16/h12={auc_mw_static(*pooled_scores(16, 12)):.3f}; fixed-horizon h4: "
    f"W8={d['roc']['W8']['auc']:.3f}, W16={d['roc']['W16']['auc']:.3f} "
    f"(W16 reacts before the h4 horizon — that is the precursor itself).",
    f"- Replay: {d['replay_match']}/{d['n_gens']} cached-loop generations "
    f"reproduce the round-2 texts exactly.",
    f"- Baseline loop waste: {d['loop_token_waste_mean']:.1f} loop tokens/gen.",
]
for a in SENT["rescue_arms"]:
    r = SENT["rescue_arms"][a]
    lines.append(f"- Rescue arm {a}: rate {r['rescue_rate']:.2f} ({r['rescued']}/{r['n']}), "
                 f"waste {r['loop_waste_mean']:.1f}, clean tail median "
                 f"{r['clean_tail_median']:.0f}.")
lines += [
    f"- Benchmark: rho A={bm['A_baseline']['rho']:.2f} "
    f"B={BENCH2['summary']['rho_edited']:.2f} C={bm['C_sentinel']['rho']:.2f}"
    + (f" C'={EXPLORE['benchmark_C']['rho']:.2f} ({EXPLORE['final_policy']})"
       if EXPLORE and EXPLORE.get("benchmark_C") else "")
    + f"; FP cost on good: {fp['changed']}/20 changed, {fp['looping']}/20 loops.",
    f"- Transfer to edited_model_v2: TPR {bm['T_edited_transfer']['tpr']}/"
    f"{bm['T_edited_transfer']['n_loop']}, rho {bm['T_edited_transfer']['rho']:.2f}, "
    f"waste {bm['T_edited_transfer']['waste']:.1f}.",
    "- Head surgery: " + (
        f"GO, arm D rho={HEADS['arm_d']['rho']:.2f}"
        if HEADS and HEADS.get("arm_d")
        else ("NO-GO" if HEADS else "not run this session") + "."),
]
if EXPLORE:
    for m, r in EXPLORE["rescue_arms"].items():
        lines.append(f"- Persistent arm {m}: rescue {r['rescue_rate']:.2f}, "
                     f"waste {r['loop_waste_mean']:.1f}, FP changed "
                     f"{r['fp_changed']}/20, rho_good {r['rho_good']:.2f}.")
open(os.path.join(R3, "rcda3_digest.md"), "w").write("\n".join(lines) + "\n")
print("digest + comparison saved", flush=True)

# ---------------------------------------------------------------- checks
from PIL import Image  # noqa: E402
for f in sorted(os.listdir(FIG)):
    p = os.path.join(FIG, f)
    im = Image.open(p).convert("L")
    h = im.histogram()
    ratio = 1 - h[255] / sum(h)
    print(f"check {f}: {im.size} non-white={ratio:.3f}", flush=True)
    assert ratio > 0.02, f"figure {f} looks blank"
print("ALL FIGURES OK", flush=True)
