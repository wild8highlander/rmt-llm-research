# RCDA-6 plan — the expanded trainable set with phase-conditioned directions
(pre-registered before the run)

Written: November 16, 2026. This plan was fixed before any RCDA-6 training or
evaluation numbers existed. It executes, verbatim, the hypothesis that closes
the RCDA-5 report:

> "the loop is written jointly by attention output *and* MLP down_proj across
> a wider band (L8–L15 all show z_diff > 130), and the probe direction must
> be conditioned on the loop *phase*, not only the collapse point."

Everything not named below is identical to RCDA-5 (model, protocol mode,
sentinel sets, evaluator, PPL check, honesty rules).

## Pre-registered protocol

1. **Phase probe (`gemma_probe_phase.py`).** Reuse the FROZEN RCDA-5 probe
   generations (no re-sampling). Pool: loop share ≥ 0.3 with a collapse
   point. Per trajectory: split generated positions into pre / early (first
   2 cycles; cycle period fitted from the token tail, self-match ≥ 0.8) /
   deep (beyond). Trajectories without a fittable period are excluded from
   the direction pools (phase structure undefined) and kept in the report.
   Per layer: `dh_early/dh_deep` vs the pre-pool, `z_early/z_deep`
   (diagonal Mahalanobis), `v_early/v_deep`, `cos(v_early, v_deep)`.
   Ship `v_L{l}.npy := v_deep` for band layers (the deep phase dominates
   long loops — stated before the training run).
2. **Trainable set.** o_proj + down_proj of the band layers **L8–L15**
   (the hypothesis band; 16 tensors, ~73.1 M parameters, ~7.3 % of the
   model). No other tensors; nothing else trainable.
3. **Training (`gemma_rcds6_train.py`).** Step loss identical to RCDA-5 —
   `L = pen_norm + β·relu(mse − cap)`, pen0_j per-prompt init normalization,
   β = 10, cap = 1.0, grad-clip 1.0 — with the penalty phase-matched:
   pre/early positions are projected on `v̂_early`, deep positions on
   `v̂_deep` (pre positions carry the early direction: the RCDA-5
   continuity). SGD momentum 0.9, lr 3e-3 (the RCDA-5 primary-arm recipe —
   NOT re-tuned), 150 steps, batch 1, fp32 masters over bf16 compute.
   Mini-bench (first 8 bad prompts, greedy 192) at steps 0 / 75 / 150.
4. **Evaluation (`gemma_benchmark.py`).** 20 bad + 20 good, greedy 192;
   ρ_bad/ρ_good/loop_waste; KL(base‖arm) on control texts; TW control per
   edited tensor (against v_deep, the shipped direction); cos(u₁, v_deep)
   and cos(u₁, v_early) per tensor.
5. **Abilities (`rcda6_ppl.py`).** Teacher-forced PPL on the 5 control
   texts, base vs trained — the parity gate.
6. **Forensics (`rcda6_analyze.py`).** rel_Frobenius, TW verdict + z_resid /
   target shift, cos(u₁, v_early/v_deep), columns changed > 1 %, top-16
   singular values per edited tensor → `rcda6_spectral_digest.json`,
   `rcds6_vectors.npz`.
7. **Figures (`rcda6_figures.py`).** figC1 phase probe, figC2 trainable band,
   figC3 training curves, figC4 four-arm summary, figC5 per-prompt migration,
   figC6 spectral record.
8. **Packaging.** The trained DELTA state (16 tensors, bf16) +
   `training_log.json` + `ppl_check.json` + provenance README + checksum
   helper → self-assembling split archive in `Model ZIP/`, every volume
   ≤ 95 MB, SHA-256 per volume. The base checkpoint is NOT shipped (it is
   `google/gemma-3-1b-it` itself; the delta applies onto it) — the size
   difference against RCDA-5's full-checkpoint kit is documented.

## Acceptance gates

- pen_norm must move materially without PPL degradation (RCDA-5 gate);
- TW gate on all 16 tensors for a "spectrally clean edit" verdict;
- ρ_bad on the frozen bad set — the behavioral success criterion;
- NEW, pre-registered for the phase hypothesis: the hard-looper subset
  (base loop share ≥ 0.5) must improve over base for the hypothesis to count
  as confirmed behaviorally, even if the full-set ρ_bad does not beat base.

Honest-reporting rule: gates that fail are reported as failed — negative
outcomes are the experiment's result, not its failure. All protocol
deviations (the dropped step-75 bench) are recorded in the report.

## Outcome

See `RCDA6_REPORT.md`: pen −97.2 % (gate 1a passed), PPL +29 % (gate 1b
FAILED — parity broken), TW violation ×16 (gate 2 failed), ρ_bad 0.85 vs
base 0.60 (gate 3 failed on the full set) — but the NEW hard-subset gate
PASSED (hard-12 0.827 → 0.620, two full repairs): the phase-conditioned
expanded penalty is the first configuration that moved the attractor.
