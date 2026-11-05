# Experiment Protocols — Stages 2–5 (version 1.0)

Formulations, endpoints and falsification criteria are fixed BEFORE the
runs. Post-hoc changes require a new protocol version (v2) with an
explicit note — never a quiet edit. Statuses live in
`../science/hypotheses/registry.yaml`.

---

## H2 — Stage 2: hallucination detector vs baselines

**Question.** Do RMT spectral features of hidden states detect
hallucinations beyond (a) token-entropy confidence, (b) semantic entropy
(Farquhar et al. 2024), (c) linear probes on hidden states
(Azaria & Mitchell; Marks & Tegmark)?

**Data.** TruthfulQA (generation, `truthfulqa/truthful_qa`), HaluEval QA
(`pminervini/HaluEval`), SelfCheckGPT wiki_bio
(`potsawee/wiki_bio_gpt3_hallucination`). Labels come from the benchmark
annotations only; the TruthfulQA string judge is a documented limitation
(an LLM judge may be substituted and must be reported in the passport).

**Endpoints.**
- Primary: AUROC per detector over pooled scored units, with 95%
  item-clustered bootstrap CI.
- Key quantity: ΔAUROC(spectral, baseline) — clustered paired bootstrap,
  permutation p.
- Secondary: PR-AUC; leave-one-benchmark-out probe AUROC.

**Detectors.** `tok_mean_entropy`, `tok_max_entropy`, `tok_last_entropy`,
`tok_neg_logprob`, `tok_log_perplexity`, `sem_ent`, `probe_hidden`
(per-layer OOF), `spec_gap_last`, `spec_ncrit_free`, `spec_probe`.

**Pre-registered outcomes.**
1. `spec_*` do NOT beat `probe_hidden`/`sem_ent` (Δ CI contains 0 or is
   negative) → **negative result, published as is**.
2. `spec_*` beat every baseline (Δ CI > 0 on ≥ 2 benchmarks, LOBO-stable)
   → strong result.
3. `spec_probe` (spectral vector + logreg) beats `probe_hidden` while
   single features do not → weaker but reportable.

---

## H3 — Stage 3: N_crit and the (μ_eff)^(−1/β) law

**Question.** Does the first-error time follow
`S(t) = exp(−(t/τ)^γ)`, `τ = c·μ_eff^(−1/β)` with β ≈ 0.5, and is there a
breakpoint (излом) in the token-level error-rate curve?

**Data.** Arithmetic-recall tasks (add3/mul2/sub2/sqrt; truth recomputed
independently in Python, H1 convention), greedy decoding, first-error
position labeled by the numeric-prefix checker; trajectories that never
error are right-censored and enter the likelihood as survival terms.

**Endpoints.**
- τ and γ with bootstrap CIs (Weibull MLE).
- Binned error-rate curve + exact two-rate changepoint with permutation p.
- Cross-model scaling: weighted regression log τ ~ log μ_eff, slope CI vs
  the β = 0.5 prediction of **2**; per-model β consistency (pairwise CI
  overlap). μ_eff comes from OUTSIDE the fit (stage-5 KL proxy or a
  user-supplied estimate).

**Falsifiers.**
- Slope CI excludes 2 AND no consistent β explains it → the (μ_eff)^(−1/β)
  law is not supported; report the fitted slope as is.
- No changepoint (permutation p > 0.05) AND γ ≈ 1 → the error process is
  consistent with the constant-rate null (H1's A0); report as is.

**Power note.** Small models may produce near-zero correct answers
(degenerate censoring) — the report must state the censoring fraction;
a fit on < 10 uncensored errors is flagged inconclusive.

---

## H4 — Stage 4: interpretability bridge

**Question.** Does the BBP spike (the layer where gap_ratio peaks)
coincide with a mechanism: (a) the logit-lens commit layer of the answer,
(b) the activation-patching restoration profile, (c) specific SAE
features (Gemma Scope), (d) the heavy-tail α of the weight matrices
(Martin & Mahoney)?

**Statistics.** Pearson r with layer-label permutation p between
(Δ lens commitment per layer, gap_ratio profile) and (patching
restoration per layer, gap_ratio profile); feature-level permutation for
the SAE audit; α-band table per matrix.

**Falsifiers.**
- r's CI covering 0 on the real model (≥ 3 prompts, ≥ 30 layers) → the
  spike is NOT the commit/restore layer; report as is.
- No SAE feature separates spike vs control at p < 0.05 after the audit →
  no feature-level interpretation; report as is.

---

## H5 — Stage 5: the RLHF thesis

**Question.** Does instruction tuning (SFT/RLHF/DPO) of the SAME family
shift (a) spectral statistics (gap_ratio, effective rank, ncrit_free)
and (b) N_crit (τ) — the "utility trap"?

**Design.** Paired comparison on identical items (registry:
`model_pairs.yaml`); paired bootstrap CI + label-swap permutation per
statistic; μ_eff proxy = token-mean KL(base‖tuned) on the shared prompt
set, measured outside the survival fits; cross-pair aggregation through
stage-3 `--scaling-in`.

**Pre-registered outcomes.**
1. Δτ < 0 (tuned collapses sooner) with CI excluding 0 AND a significant
   spectral shift → thesis supported on that pair.
2. Δτ > 0 with significant shift → thesis refuted on that pair.
3. No significant shifts → not resolved; report as is (a negative result
   is still a result).

**Confounds to report.** tokenizer/lineage mismatch (registry asserts
same org/family), chat-template asymmetry (base models get raw prompts),
KL proxy sensitivity (both directions reported: KL(base‖tuned) and
KL(tuned‖base)).

---

## Protocol changelog

### v1.1 — measurement-integrity revision (2026-11)

Reason: the v1.0 smoke runs exposed defects that distorted the very
quantities the protocols claim to measure. Nothing here changes WHAT is
tested; it changes how honestly the quantities are computed.

- **S2**: continuation scoring now returns proper log-probabilities
  (log-softmax), not raw logits — `tok_neg_logprob` / `tok_log_perplexity`
  values from v1.0 runs are invalid and must not be compared to v1.1.
- **All stages**: permutation p-values carry the +1 correction; the
  smallest attainable p is 1/(n_perm + 1), exactly 0 is impossible.
- **S2**: `probe_hidden` uses within-benchmark grouped OOF when a run has
  a single benchmark; LOBO remains the cross-benchmark criterion (>= 2
  benchmarks). The OOF source is reported in the run report.
- **S3**: beta is fit in log space without the (0,1) ceiling
  (beta > 1 is a real result); the beta CI under --mu-eff comes from a
  real bootstrap, not a fabricated fallback; the changepoint null
  respects per-trajectory at-risk exposure (n_tokens), not error labels.
- **S3**: data passports hash the FULL benchmark items (labels,
  candidates, sentences, GPT-3 samples), not just (item_id, context).
- **S5**: base and tuned spectra are computed with ONE tokenizer (the
  base side); tokenizer vocab mismatch between registry entries now
  fails the run instead of silently tokenizing differently; each pair
  also writes a stage-3-compatible `fit.json` (tau, tau_ci, mu_eff) so
  `--scaling-in` consumes stage-5 runs directly; Δtau carries a paired
  bootstrap CI.
- **Passports**: every passport now records the git commit + dirty flag
  and the sha256 of this protocol document. Weight files are resolved
  from the local HF-hub cache when the model is not a local directory;
  if the weights cannot be hashed the passport says
  `weights_not_hashed` explicitly.

Smoke results under `results/smoke_pythia70m/` were REGENERATED under
v1.1; v1.0 artifacts are not comparable to them.
