# RCDA — RMT-Calibrated Directional Ablation for MNN-Pocket-0.3B

An implementation of the target **RCDA (RMT-Calibrated Directional Ablation)** plan on
top of the `neural_lab/MNN-Pocket-0.3B` toolkit of the
`wild8highlander/rmt-llm-research` repository. No formula is invented from thin air:
every calibration step is built on the repository's mathematics — the Marchenko–Pastur
law (`marchenko_pastur.py`, `probe_real_gpt2._mp_bounds_general`), the BBP transition
(`src/rmt_llm/bbp_transition.py`: `bbp_critical_theta(q) = sqrt(q)`) and the
Tracy–Widom scale of the spectral edge. The difference from the original benchmark:
instead of runtime activation hooks, **the weights themselves are edited**, and every
number of the edit (direction tolerance, strength α, column count, stopping criterion)
has a statistical justification rather than being tuned by trial and error.

## Contents

```text
Code/
  marchenko_pastur.py    — RCDA core: mp_upper_edge, tw_scale, rmt_z_score,
                           bbp_threshold, rayleigh_spike_strength, bbp_excess,
                           alpha_from_bbp (plan formulas, sections 2–5)
  pocket_rmt_probe.py    — + --rcds-probe mode: Δh^(l), z_RMT, β/β_floor,
                           Hotelling z_diff, matrix scans (λ_v vs λ_+, θ_v/θ_c),
                           saving direction vectors (.pt) and the report
  rmt_intervention.py    — + --rcda weight editor: matrix_scan, select_columns
                           (auto_tw / auto_chi2 / topk), rcda_edit
                           (W' = W − α·v·p_maskedᵀ), spectral_tw_control,
                           a self-contained edit-pack (vectors in base64) and
                           edit reproduction via --apply_pack
  benchmark_rmt.py       — + --baseline_dir/--edited_dir mode: ρ (share of the
                           repetition_loop defect) on P_bad, false positives on
                           P_good, PPL control, Wilcoxon, spectral TW control of
                           the edited matrices
  reassemble_model.py    — + --eval_dir/--overlay_state: assemble an eval directory
                           overlaying edited tensors onto the base
  probe_real_gpt2.py, plot_benchmark.py — unchanged
config/rcda_experiment.yaml — experiment config (plan section 8)
prompts/bad.jsonl       — 20 prompts that provoke repetition_loop (RU+EN)
prompts/good.jsonl      — 20 short factual prompts (FPR control)
prompts/control.jsonl   — 5 texts for the abilities PPL control
rcda.patch              — unified diff of all changed Code/ files
example_run/            — artifacts of the real run on MNN-Pocket-0.3B
```

All changes are additive: the original modes (clip/dampen/project/random hooks,
default probes, the old benchmark) keep working as before.

## Conventions (fixed once)

For a weight matrix `W ∈ R^{m×n}` (m = out_features, n = in_features) and a unit
direction `v` in the output space:

- column covariance: `C = W·Wᵀ/n`, aspect `q = m/n` (the same "columns = sample"
  convention as in `rmt_intervention.py` for tokens; q > 1 is allowed —
  `_mp_bounds_general` is used);
- `σ_W² = median(eig(C))` — the repository's bulk estimate;
- `λ_+^{(W)} = σ_W²·(1+√q)²`, `TW-scale(λ_+, N) = λ_+·N^{−2/3}`;
- `λ_v = ‖vᵀW‖²/n = vᵀCv` — the projection mass (under the null ≈ σ_W²);
- signal strength in the repository's BBP parameterization:
  `θ_v = √(q·max(λ_v/σ_W² − 1, 0))`, threshold `θ_c = bbp_critical_theta(q) = √q`
  (matches the plan's `θ_c = √(n/m)` for n ≤ m and its transposition for n > m);
- BBP excess: `δ = max(0, (θ_v − θ_c)/θ_v)`;
- edit strength: `α₀ = min(1, κ·δ/(1−δ))`, κ = 0.5 (plan, section 5);
- candidate layer: `z_RMT = (λ_max − λ_+)/TW-scale` on the pooled activation
  covariance; for Δh additionally `β = ⟨Δh, u₁⟩` against the noise `√(λ₁/N)`
  (plan, section 2) and the discriminative Hotelling test `z_diff` on the
  difference of the two group means;
- columns: `auto_tw` — `p_j²/n > λ_+ + τ·TW-scale` (the plan's strict gate),
  `auto_chi2` — χ²₁ quantile with the Bonferroni correction, `topk=K` — per the paper;
- TW control (plan, section 7.2): the healthy top singular values of W′ must not
  move beyond `±2 TW-scale` (a violation is flagged at `>3`).

## How to run (after `python tools/reassemble_model.py`)

```bash
# 1) RCDS probe: where the defect lives and which matrices "write" it
python Code/pocket_rmt_probe.py --rcds-probe \
    --model_dir model --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --out_dir rcda_results            # -> rcda_probe_report.json, fig5/fig6, vectors/

# 2) RCDA weight edit (α from BBP, columns by the gate)
python Code/rmt_intervention.py --rcda --model_dir model \
    --layer 9  --matrix down_proj --vector rcda_results/vectors/L9_delta_h.pt \
    --layer 11 --matrix o_proj    --vector rcda_results/vectors/L11_delta_h.pt \
    --alpha auto --columns auto_chi2 --out_dir edited_model
#    --columns auto_tw       — the plan's strict gate (may legitimately refuse)
#    --columns topk=54       — a fixed 54 columns, as in the paper
#    --full                  — a full rank-1 ablation W − α·v(vᵀW)
# reproduce the edit anywhere: --apply_pack edited_model/rcda_edit_report.json

# 3) Before/after benchmark: ρ, FPR, PPL, Wilcoxon, TW control
python Code/benchmark_rmt.py --baseline_dir model --edited_dir edited_model \
    --bad_set prompts/bad.jsonl --good_set prompts/good.jsonl \
    --control_set prompts/control.jsonl --out_dir bench --max_new 64

# 4) Alternative assembly of the eval model from base + edits
python Code/reassemble_model.py --model_dir model \
    --eval_dir eval_model --overlay_state edited_model/model.safetensors
```

The `config/rcda_experiment.yaml` order sets the same parameters plus the stopping
criteria of the iterative edit→measure→keep/revert loop (plan, section 6).

## What the real run showed (see example_run/)

1. **RCDS probe** (20+20 prompts, 958+287 tokens): Δh is statistically
   super-significant in every layer (z_diff ≈ 600–760; |β|/√(λ₁/N) ≈ 18); activation
   spectra are deeply supercritical versus MP (z_RMT ~ 10⁸ — the model as a whole lies
   outside the MP null, as fixed by the repository protocol). The z_diff ranking
   singled out blocks L8–L12 — the "mid-depth" zone, consistent with the paper's
   prior hint (layers 15/19 of 28).
2. **Matrix scans**: the tolerance `λ_v > λ_+^{(W)}` was met only by
   **L9 down_proj** (λ_v/λ₊ = 1.06, θ_v = 0.77 > θ_c = 0.59, δ = 0.23 →
   α₀ = 0.15) and **L11 o_proj** (λ_v/λ₊ = 1.10, θ_v = 1.84 > θ_c = 1.00,
   δ = 0.46 → α₀ = 0.42). L8/L10 are subcritical and the editor does not touch
   them (α₀ = 0 by formula).
3. **The strict auto_tw gate refused both matrices** (0 columns: the direction
   profile is dense, no column carries mass above the MP edge) — a legitimate plan
   response, "do not subtract what is not confirmed". The working edit used
   `auto_chi2`: 15 + 55 columns, **62,720 weights ≈ 0.021%** of the parameters —
   exactly inside the plan's target corridor of 0.016–0.05%.
4. **TW control of both edited matrices**: verdict `pass` (0 violations);
   PPL on control texts unchanged (ratios 0.996–1.013) — the edit is surgical.
5. **Behaviorally**, one iteration of the soft edit did not reduce ρ:
   ρ = 65% (13/20) → 65% (13/20), FPR on good = 50% → 55%, changed greedy
   generations: 0 of 20 bad and 2 of 20 good (Wilcoxon — "identical": on the bad
   set every trajectory matched byte for byte). The diagnosis follows plan section 6
   exactly: α = 0.15/0.42 on 15/55 columns is too soft to move greedy trajectories,
   and Δh mixes "defect" with the generic "long CoT context vs short factual answer"
   difference (the 50% FPR baseline shows the 0.3B model loops even on short good
   prompts). Next loop iterations: clean up the group contrast (same length/topic,
   differing only in defect provocation), re-estimate Δh on the edited model, and at
   δ→1 move to `--full` or more candidate layers. The loop infrastructure
   (probe → edit → benchmark → stop criteria in yaml) is ready and is reused without
   code changes.

## Limitations and honest caveats

- The prompt sets are a starting point (20+20+5); for a working effect they must be
  extended and length/topic-matched between groups, otherwise Δh encodes style rather
  than the defect (see item 5 above).
- `z_RMT` on real LLM activations is always huge (structure ≫ noise), so the working
  RCDA tolerance criterion for Δh is the pair `z_diff` + `β/β_floor`;
  `z_RMT` is kept as the plan requires and as a diagnostic column.
- The `bbp_lambda_max` parameterization in `src/rmt_llm` is simplified asymptotics;
  RCDA uses only its inverse mapping for θ_v and the threshold `θ_c = √q`, which
  matches the plan.
- Checking the "layers 15 and 19" hint: the RMT scans reject it in favor of L8–L12
  (by z_diff) — i.e. localization must come from the probe, not from analogy.
