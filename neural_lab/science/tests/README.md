<p align="center">
  <img src="../../../assets/logos/sc_tests.banner.svg" width="760" alt="science/tests — animated logo: checklist with drawing checkmarks">
</p>

# ✅ science/tests — Core Test Suite

<p align="center">
  <em>52+ tests that keep the science honest</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--tests-00ffa3" alt="tests"></a>
  <img src="https://img.shields.io/badge/52%2B%20tests-00ffa3" alt="52+ tests"><img src="https://img.shields.io/badge/CI%20green-a78bfa" alt="CI green">
</p>

The test suite of the science core — and the reason its numbers can be
trusted between runs. The suite covers the three layers (instrumentation,
features, prediction), the statistics, and, critically, the *failure
modes*: every leak and every silent-degradation bug that was ever found
is pinned here as a regression test, so the same class of mistake cannot
ship twice. Tests run on a fake model — no trained weights required —
which keeps the suite fast enough to gate every push in CI.

## What the suite guards

| Area | Typical guarantees |
|---|---|
| splits | family disjointness between train and test folds; determinism under a fixed seed |
| features | protocol v1.1 output is byte-identical to v1.0 on 4 layers; dynamic `L0..Ln` naming for deeper models; loud error on 0 valid observations |
| statistics | bootstrap CI correctness against known distributions; clustered permutation preserves cluster structure |
| predictors | baselines reproduce fixed values; ablation E is exactly D minus the spectral group |
| storage | passport round-trips; schema violations are rejected |
| leakage | end-to-end checks would catch a family-overlapping split |

## Running

```bash
python3 -m pytest neural_lab/science/tests/ -q     # full suite, ~seconds
```

CI runs the same suite in the science job of
[`ci.yml`](../../../.github/workflows/ci.yml) on every push — a red science
job blocks merge, which is the point.

## Contribution rule

A bug fix in the core that is not accompanied by a test here is not
considered fixed: the suite is the registry of *known ways this core
could lie to itself*, and it is meant to grow with every lesson learned
(see [`../AUDIT.md`](../../AUDIT.md) for the audit that seeded it).

---

<p align="center">
  <img src="../../../assets/logos/sc_tests.avatar.svg" width="56" alt="science/tests animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../experiments/README.md">experiments →</a>
  </sub>
</p>
