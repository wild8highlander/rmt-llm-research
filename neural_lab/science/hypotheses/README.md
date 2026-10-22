<p align="center">
  <img src="../../../assets/logos/sc_hyp.banner.svg" width="760" alt="science/hypotheses — animated logo: lightbulb with pulsing rays">
</p>

# 💡 science/hypotheses — Registry

<p align="center">
  <em>H1–H4: what the core claims, and what would refute it</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--hypotheses-fbbf24" alt="hypotheses"></a>
  <img src="https://img.shields.io/badge/H1%E2%80%93H4-fbbf24" alt="H1-H4"><img src="https://img.shields.io/badge/falsifiable-f472b6" alt="falsifiable">
</p>

The single source of truth for what the science core believes and why.
Every hypothesis is registered **before** its experiment runs, with a
formulation, applicability scope, pre-registered models, a success
criterion, and a falsification criterion. After results exist, the
entry may only change via a new protocol version (H*_v2) with an explicit
note — never by quietly editing the record, which is what separates a
registry from a results diary.

## Current registry

| ID | Claim (short) | Status |
|---|---|---|
| H1 | spectral features of internal activations predict model errors beyond standard confidence signals | inconclusive |
| H2 | training on true formulas reduces confident hallucinations where the model has knowledge | inconclusive |
| H3 | a BBP emission (λ_max beyond the MP bound) separates "known" from fabricated answers | inconclusive |
| H4 | N_crit (critical capacity per token budget) reproduces as a spectral threshold on independent models | registered |

## Status semantics

| Status | Meaning |
|---|---|
| `registered` | recorded, experiment not yet run |
| `testing` | experiment under the frozen protocol has run; see results |
| `supported` | got support under the stated conditions |
| `not_supported` | did **not** get support (a negative result is still a result) |
| `inconclusive` | statistics insufficient, or the effect is trajectory-dependent |

## Files

| File | Content |
|---|---|
| `registry.yaml` | the registry itself: formulations, scopes, predictions, criteria, statuses, outcome notes per run |
| `protocols/` | frozen protocol snapshots referenced by the entries |

The H1 entry's outcome notes are what the
[`results/`](../results/README.md) folders quantify; the primary success
criterion (ΔAUC CI excluding 0 **and** clustered permutation p < 0.05)
has not been met on any S/M/L cell yet — the honest status is
`inconclusive`, and the plan for power lives in the roadmap.

---

<p align="center">
  <img src="../../../assets/logos/sc_hyp.avatar.svg" width="56" alt="science/hypotheses animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../experiments/README.md">experiments →</a>
  </sub>
</p>
