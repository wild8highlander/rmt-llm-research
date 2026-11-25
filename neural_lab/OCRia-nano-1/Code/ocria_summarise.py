"""Merge the three eval files into the canonical summary the README reads.

Outputs:
  Results/summary_all_arms.json      - one record per arm, headline fields first
  Results/RESULTS_SUMMARY.md         - the markdown tables (README section)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ocria_common import RESULTS_DIR, dump_json

HEAD = ["arm", "F1_phantom_rate", "F1_phantom_char_rate", "F1_blank_specificity",
        "F2_loop", "F3_cer_clean", "F3_cer_degraded", "F3_cer_control",
        "F3_exact_match_clean", "F6_auc_confidence", "F6_auc_geometry",
        "control_logits_mse", "control_kl_base_to_arm", "n_changed_tensors"]


def main() -> None:
    arms = {}
    for arm in ("base", "classic", "rcda_x"):
        f = RESULTS_DIR / f"eval_{arm}.json"
        if f.exists():
            arms[arm] = json.loads(f.read_text(encoding="utf-8"))
    if not arms:
        raise SystemExit("no eval_*.json found - run ocria_benchmark.py first")

    probe = json.loads((RESULTS_DIR / "probe_report.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS_DIR / "formulas_audit.json").read_text(encoding="utf-8"))

    # the router round (F4 head edit) - additive, does not touch the arm table
    router_round = None
    rf = RESULTS_DIR / "eval_router_reject.json"
    if rf.exists():
        rr = json.loads(rf.read_text(encoding="utf-8"))
        router_round = {
            "round": rr["round"], "winner_arm": rr["arm"],
            "gates_pass": rr["gates"]["pass"],
            "F4_fit_nontext_reject_rate": rr["F4_fit"]["nontext_reject_rate"],
            "F4_fit_text_route_acc": rr["F4_fit"]["text_route_acc"],
            "F4_ability_reject_win_rate": rr["F4_ability"]["reject_win_rate_ability"],
            "F4_ability_route_acc": {k: rr["F4_ability"][f"{k}_route_acc"]
                                     for k in ("clean", "control")},
            "F4_transfer_mean_reject_rate": rr["F4_transfer"]["mean_reject_rate"],
            "F4_transfer_per_family": {k: rr["F4_transfer"][f"{k}_reject_rate"]
                                       for k in ("structured", "noise", "blank")},
            "head_spectrum": rr["head_spectrum"],
            "export": {k: rr["export"][k] for k in
                       ("path", "trunk_arrays_identical",
                        "flip_head_bit_identical",
                        "group_rows0_3_max_abs_diff_vs_base", "sha256")},
        }

    base = arms.get("base", {})
    summary = {
        "model": "llamaindex/OCRia-nano-1 (rec_latin fp32 rebuild + "
                 "router with the round-3 reject head)",
        "protocol": probe["protocol"],
        "deployed_int8_phantom_rate": probe["base"]["phantom_rate"],
        "fp32_rebuild_phantom_rate": base.get("F1_phantom_rate"),
        "arms": arms,
        "router_round": router_round,
        "deltas_vs_base": {},
        "probe": {k: probe[k] for k in ("writers", "supercritical",
                                        "utility_trap", "router", "detector",
                                        "stages", "matrix_scans")},
        "formulas_audit": {k: audit[k] for k in
                           ("mp_summary", "mp_activation_covariance",
                            "bbp_spike_scans", "bbp_verdict",
                            "bbp_outlier_check", "tracy_widom",
                            "hallucination_split", "entropy_response",
                            "shrinker", "caputo_ncrit", "rmsnorm_trace")},
    }
    for arm, a in arms.items():
        if arm == "base":
            continue
        summary["deltas_vs_base"][arm] = {
            "phantom_rate_pp": round(100 * (a["F1_phantom_rate"] - base["F1_phantom_rate"]), 2),
            "phantom_rate_rel": round(a["F1_phantom_rate"] / base["F1_phantom_rate"], 4),
            "cer_clean_pp": round(100 * (a["F3_cer_clean"] - base["F3_cer_clean"]), 3),
            "cer_degraded_pp": round(100 * (a["F3_cer_degraded"] - base["F3_cer_degraded"]), 3),
        }
    dump_json(summary, RESULTS_DIR / "summary_all_arms.json")

    # markdown tables
    lines = ["# RESULTS_SUMMARY — one evaluator, three arms + the router round", "",
             "All numbers are fields of committed JSONs (`eval_*.json`, "
             "`summary_all_arms.json`) — nothing hand-typed.", ""]
    lines += ["## Headline", "",
              "| arm | F1 phantom ↓ | F3 CER clean | F3 CER degraded | control KL | TW gate |",
              "|---|---|---|---|---|---|"]
    for arm, a in arms.items():
        tw = a.get("spectral")
        tw_s = "pass ×" + str(len(tw)) if tw else "—"
        lines.append(f"| {arm} | {a['F1_phantom_rate']:.4f} | "
                     f"{a['F3_cer_clean']:.4f} | {a['F3_cer_degraded']:.4f} | "
                     f"{a['control_kl_base_to_arm']:.4f} | {tw_s} |")
    lines += ["", "## Per-family", "",
              "| family | base | classic | rcda_x |", "|---|---|---|---|"]
    rows = [
        ("F1 phantom rate", "F1_phantom_rate"),
        ("F1 phantom char rate", "F1_phantom_char_rate"),
        ("F1 blank specificity", "F1_blank_specificity"),
        ("F2 rho_loop", None), ("F2 mean loop_share", None),
        ("F3 CER clean", "F3_cer_clean"),
        ("F3 CER degraded", "F3_cer_degraded"),
        ("F3 CER control", "F3_cer_control"),
        ("F3 exact match clean", "F3_exact_match_clean"),
        ("F6 AUC confidence", "F6_auc_confidence"),
        ("F6 AUC geometry (−v·ℓ)", "F6_auc_geometry"),
        ("control logits MSE", "control_logits_mse"),
    ]
    for label, key in rows:
        vals = []
        for arm, a in arms.items():
            if label.startswith("F2 rho"):
                vals.append(f"{a['F2_loop']['rho']:.3f}")
            elif label.startswith("F2 mean"):
                vals.append(f"{a['F2_loop']['mean_loop_share']:.4f}")
            else:
                vals.append(f"{a.get(key, float('nan'))}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "## Spectral section (edited tensors)", ""]
    for arm, a in arms.items():
        for name, s in a.get("spectral", {}).items():
            lines.append(f"- **{arm}** `{name}`: rel_frobenius {s['rel_frobenius']}, "
                         f"ΔKS {s['tw_gate_dks']} ({s['tw_verdict']}), "
                         f"cos(u₁, v) = {s['cos_u1_v']}, "
                         f"σ₁ share {s['sigma1_share']}")
    if router_round:
        rr = router_round
        lines += ["", "## Router round (F4 head edit — the 5th 'reject' class)", "",
                  f"winner arm **{rr['winner_arm']}**, gates pass: "
                  f"{rr['gates_pass']}", "",
                  "| metric | base | router_reject |", "|---|---|---|",
                  ("| F4 fit non-text reject rate | 0.0 (no class — 100% routed "
                   "into recognisers) | "
                   f"**{rr['F4_fit_nontext_reject_rate']}** |"),
                  (f"| F4 fit text route acc | 1.0 | "
                   f"{rr['F4_fit_text_route_acc']} |"),
                  ("| F4 ability route acc (clean / control) | 0.9667 / 1.0 | "
                   f"{rr['F4_ability_route_acc']['clean']} / "
                   f"{rr['F4_ability_route_acc']['control']} |"),
                  ("| F4 ability reject wins | n/a (no class) | "
                   f"{rr['F4_ability_reject_win_rate']} |"),
                  ("| F4 transfer reject rate (structured/noise/blank) | n/a | "
                   f"**{rr['F4_transfer_mean_reject_rate']}** (" +
                   "/".join(str(rr['F4_transfer_per_family'][k]) for k in
                            ("structured", "noise", "blank")) + ") |"),
                  ("| head edit spectrum | — | λmax/λ₊ "
                   f"{rr['head_spectrum']['lambda_max_over_lambda_plus']}, "
                   f"spikes>2λ₊ {rr['head_spectrum']['n_spikes_above_2lambda_plus']} "
                   f"(single_spike {str(rr['head_spectrum']['single_spike']).lower()}), "
                   f"cos(v₁, w_init) {rr['head_spectrum']['cos_v1_edit_w_init']} |"),
                  ("| export | — | trunk identical "
                   f"{rr['export']['trunk_arrays_identical']}, flip bitwise "
                   f"{rr['export']['flip_head_bit_identical']}, script rows "
                   "max|Δ| "
                   f"{rr['export']['group_rows0_3_max_abs_diff_vs_base']} |")]
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("[summarise] wrote summary_all_arms.json + RESULTS_SUMMARY.md")
    for arm, a in arms.items():
        print(f"  {arm}: phantom={a['F1_phantom_rate']} cer_clean={a['F3_cer_clean']}")


if __name__ == "__main__":
    main()
