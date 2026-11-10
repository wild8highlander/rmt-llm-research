"""Audit RCDA4_REPORT.md numbers against artifacts."""
import json
import numpy as np

R = "/home/z/my-project/rcda_work/rcda4"

b = json.load(open(f"{R}/partA_base_summary.json"))
e = json.load(open(f"{R}/partA_edited_summary.json"))
print("== Part A ==")
for tag, d in (("base", b), ("edited", e)):
    print(tag, {
        k: d.get(k) for k in ("rho_bad_ru", "rho_good_ru", "rho_gsm8k_en",
                              "loop_share_bad_mean", "line_repeat_bad_mean",
                              "distinct4_bad_mean", "n_gens")})
assert b["rho_bad_ru"] == e["rho_bad_ru"] == 1.0

ev = json.load(open(f"{R}/partB/partB_eval_aided.json"))
print("== Part B eval aided ==")
print({k: ev[k] for k in ("rho_bad", "rho_good", "loop_waste_mean", "kl_ctrl",
                          "n_changed_tensors")})
print("TW:", ev["tw"])
print("cos:", ev["cosine_edit_vs_v"])
assert list(ev["tw"]) == ["model.layers.16.self_attn.q_proj.weight"]

t = json.load(open(f"{R}/partB/partB_aided_transcript.json"))
print("aided cells:", t["cells_used"], "final rho_mini:",
      json.loads(t["final"])["rho_mini"])
tb = json.load(open(f"{R}/partB/partB_blind_transcript.json"))
print("blind cells:", tb["cells_used"], "final rho_mini:",
      json.loads(tb["final"])["rho_mini"])

print("== v collinearity ==")
vs = {L: np.load(f"{R}/partB/v_L{L}.npy").astype(np.float64) for L in (15, 16, 18)}
for a, c in ((15, 16), (16, 18), (15, 18)):
    cos = float(vs[a] @ vs[c] / (np.linalg.norm(vs[a]) * np.linalg.norm(vs[c])))
    print(f"cos v{a},v{c} = {cos:.4f}")

diag = json.load(open(f"{R}/partB/diagnostics.json"))
print("diagnostics heads:", {k: diag[k] for k in list(diag)[:4]})
tc = json.load(open(f"{R}/partC/partC_transcript.json"))
print("Part C final:", tc.get("final"))
