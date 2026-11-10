# RCDA Runbook — rounds 1–3 (MNN-Pocket-0.3B)

Overlay for `neural_lab/MNN-Pocket-0.3B/`. Weights are NOT included:
`config/model_path` in every script points at the local model dir
(`rcda_work/model` in the sandbox layout). Python ≥3.10, torch CPU,
transformers (Qwen3), safetensors, matplotlib, scipy.

## Round 1 — RCDA baseline (style contrast)
```bash
python3 Code/pocket_rmt_probe.py --config config/rcda_experiment.yaml   # probe + scans
python3 Code/rmt_intervention.py apply --pack <edit_pack.json>          # edit + TW control
python3 Code/benchmark_rmt.py --baseline ../model --edited <edited_dir> # ρ/PPL benchmark
python3 Research/RCDA/tools/test_rcda_math.py                           # unit math tests
```
Result: 0/20 trajectories changed (style confound). See RCDA_ROUND1_SUMMARY.md.

## Round 2 — behavioral contrast + dose-response + precursors
```bash
python3 Research/RCDA/tools/rcda2_stage1_probe.py     # token-loop labels, Δh_behav, scans, fig7–fig10
python3 Research/RCDA/tools/rcda2_stage2_dose.py      # α dose-response, TW-backoff final edit
python3 Research/RCDA/tools/rcda2_stage3_figures.py   # figures + digest
python3 Code/benchmark_rmt.py --baseline ../model --edited ../edited_model_v2
```
Result: writers = o_proj L15/16/18; ρ 0.70→0.65; PPL 3794→3662; PR precursor (fig9).
Artifacts: results/rcda2_*.json, vectors/L15–L18_delta_h_behav.pt.

## Round 3 — spectral sentinel: online detection + rescue
```bash
python3 Research/RCDA/tools/rcda3_sentinel.py            # calibrate + detect + rescue arms + benchmark (~18 min CPU)
python3 Research/RCDA/tools/rcda3_fix_steer.py           # re-run steer arm with fixed hook arming
python3 Research/RCDA/tools/rcda3_stage3_figures.py      # fig11, fig13, digest, comparison
python3 Research/RCDA/tools/rcda3_heads.py               # head surgery (profile+calibration+arm D gate)
python3 Research/RCDA/tools/rcda3_explore.py             # persistent policies (3-gram exhaustion, re-arming steer)
```
Config: `config/rcda3_sentinel.yaml` (layers 15/16/17, scales 8/16, θ = q10 of
the healthy teacher-forced PR pool, alarm = 2 consecutive steps, rescue
γ=1.0, K=8 steer / K=4 ban).

Key numbers (measured): TPR 96.7%, 0 alarms on clean gens, replay 40/40;
timed arms ban 0/30, steer 12/30, both 18/30 (attractor re-forms);
benchmark ρ: baseline 0.70 / permanent edit 0.65 / sentinel 0.50, loop waste
70.0→16.4 tokens/gen, zero weight changes; transfer to edited model TPR 13/13.
Run in full: rcda3_explore.py (ban_persist: rescue 0.97, waste 0.0, C' rho=0.10) and rcda3_heads.py (placebo-controlled specificity, NO-GO by the pre-registered rule).

## Layout
- `Code/` — toolkit (RMT hooks clip/dampen/project/random, MP edge, TW control, rcda_edit, rcda_shrink §9).
- `Research/RCDA/` — reports, results (JSON), figures, vectors, tools.
- `Research/RCDA/rcda.patch` — additive patch of Code/ against the base repo.
- `Docs/RCDA_RUNBOOK.md` — this file.
