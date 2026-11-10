# RCDA-3 digest (all numbers measured, not estimated)

- Calibration: healthy PR pool from 5 teacher-forced control texts (q10 thresholds; W8 L15 theta=1.83).
- Detection: TPR 0.97 on 30 looping gens; predictive alarms 10/29 (median lead 2.0 tokens); ROC AUC (window-matched horizons) W8/h4=0.720, W16/h12=0.415; fixed-horizon h4: W8=0.720, W16=0.439 (W16 reacts before the h4 horizon — that is the precursor itself).
- Replay: 40/40 cached-loop generations reproduce the round-2 texts exactly.
- Baseline loop waste: 73.8 loop tokens/gen.
- Rescue arm ban: rate 0.00 (0/30), waste 68.2, clean tail median 0.
- Rescue arm steer: rate 0.40 (12/30), waste 40.4, clean tail median 0.
- Rescue arm both: rate 0.60 (18/30), waste 28.8, clean tail median 1.
- Benchmark: rho A=0.70 B=0.65 C=0.50 C'=0.10 (ban_persist); FP cost on good: 11/20 changed, 12/20 loops.
- Transfer to edited_model_v2: TPR 13/13, rho 0.65, waste 85.5.
- Head surgery: NO-GO.
- Persistent arm ban_persist: rescue 0.97, waste 0.0, FP changed 12/20, rho_good 0.00.
- Persistent arm both_persist: rescue 0.97, waste 0.0, FP changed 16/20, rho_good 0.00.
