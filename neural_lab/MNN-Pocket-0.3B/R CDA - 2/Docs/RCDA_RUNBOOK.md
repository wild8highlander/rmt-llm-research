# RCDA RUNBOOK — reproduction and development (neural_lab/MNN-Pocket-0.3B)

The command set from a clean clone to reproducing both research rounds.
Requirements: `pip install torch transformers safetensors scipy pyyaml matplotlib`
(CPU versions suffice; the fp32 model takes ~1.2 GB of RAM).

## 0. Model preparation

```bash
git clone https://github.com/wild8highlander/rmt-llm-research
cd rmt-llm-research/neural_lab/MNN-Pocket-0.3B
python tools/reassemble_model.py      # assembles model/model.safetensors + SHA-256
```

## 1. Round 1 — RCDA per the target plan (style contrast)

```bash
# 1.1 RCDS probe: Δh, z_RMT, β, z_diff, matrix scans, vectors
python Code/pocket_rmt_probe.py --rcds-probe --model_dir model \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --out_dir rcda_results

# 1.2 Weight edit: α from BBP, auto_chi2 columns (the strict gate — auto_tw)
python Code/rmt_intervention.py --rcda --model_dir model \
    --layer 9  --matrix down_proj --vector rcda_results/vectors/L9_delta_h.pt \
    --layer 11 --matrix o_proj    --vector rcda_results/vectors/L11_delta_h.pt \
    --alpha auto --columns auto_chi2 --out_dir edited_model

# 1.3 Before/after benchmark + TW control
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench --max_new 64

# 1.4 Reproduce the edit from a pack (vectors are embedded in the JSON)
python Code/rmt_intervention.py --rcda --model_dir model \
    --apply_pack Research/RCDA/results/rcda_edit_report_applied_autochi2.json \
    --out_dir edited_model_from_pack
```

Expected round-1 outcome: TW pass, PPL unchanged, ρ unmoved — a "style" edit, as
round 2 showed (|cos| with the behavioral direction ≈ 0.01–0.09).

## 2. Round 2 — RCDA-2: behavioral contrast and precursors

```bash
# 2.1 Generations with loop token labels + Δh_behavioral + scans + attribution + PR
python Research/RCDA/tools/rcda2_stage1_probe.py
#   -> rcda2/rcda2_probe.json, rcda2/rcda2_generations.jsonl, vectors/*_delta_h_behav.pt

# 2.2 α dose-response (KL, flip-rate, rep4) + the final edit with TW backoff
python Research/RCDA/tools/rcda2_stage2_dose.py
#   -> edited_model_v2/model.safetensors, rcda2/rcda2_edit_report.json

# 2.3 Round-2 benchmark
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model_v2 \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench_v2 --max_new 96

# 2.4 Figures fig7-fig10 + digest
python Research/RCDA/tools/rcda2_stage3_figures.py

# 2.5 Synthetic validation of the math (25 checks, no model needed)
python Research/RCDA/tools/test_rcda_math.py
```

## 3. Next steps (from the RCDA-2 conclusions)

1. **Per-head ablation**: the o_proj columns concentrate in heads
   L15→h7, L16→h8, L18→h0; subtract the Δh profile only from the 64-column
   block of the head (`rcda_edit` with columns = 64h..64h+63).
2. **Iterative loop** (plan §6): after every edit, re-run stage 1 on the edited
   model and re-estimate Δh; stop per the `stop:` block of
   `config/rcda_experiment.yaml` (ρ_target 0.2, z_diff < 2, FPR ≤ 0.15, PPL ≤ 1.10×).
3. **Bulk shrinkage** (§9): `rcda_shrink` / `apply_pack_with_shrink` in
   `Code/rmt_intervention.py` — the hypothesis "defect = overheated bulk";
   test on L15–L18 o_proj at r = 10, τ = 2.
4. **PR detector**: monitor the window-16 participation ratio at L15 (a drop
   > 5% from the sliding norm) → arm the runtime `clip` hook from
   `rmt_intervention.py` during generation.
5. **Group contrast**: extend the bad/good sets, matching length and topic so the
   residual style does not dominate Δh (see README_RCDA.md, item 5).

## 4. Artifact layout

```text
Research/RCDA/
  RCDA2_REPORT.md          round-2 report (numbers, tables, conclusions)
  RCDA_ROUND1_SUMMARY.md   round-1 protocol
  results/                 JSON: probes, scans, doses, benchmarks, edit packs
  figures/                 fig5-fig10
  vectors/                 Δh directions (.pt) of both rounds
  tools/                   rcda2_stage{1,2,3}*.py, test_rcda_math.py
config/rcda_experiment.yaml    round-1 parameters + stop criteria
config/rcda2_behavioral.yaml   round-2 parameters
Docs/RCDA_RUNBOOK.md           this file
```
