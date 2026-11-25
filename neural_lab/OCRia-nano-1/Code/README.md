# 🛠 Code — the OCRia RCDA toolkit

<p align="center">
  <img src="https://img.shields.io/badge/scripts-11-00D4FF" alt="scripts">
  <img src="https://img.shields.io/badge/formulas-12_executed-a78bfa" alt="formulas">
  <img src="https://img.shields.io/badge/families-6_measured-22d3ee" alt="families">
  <img src="https://img.shields.io/badge/rounds-3_(2_arms_+_router_round)-fbbf24" alt="rounds">
  <a href="../README.md"><img src="https://img.shields.io/badge/%E2%86%90-OCRia_nano_1_kit-FFB020" alt="parent"></a>
</p>

One toolkit, one evaluator, three rounds — the parent repository's
machinery, ported from causal-LM hidden states to CTC feature/logit states
and extended to the six OCR hallucination families (rounds 1–2 train the
latin recogniser; round 3 grafts a reject class onto the router's head).

| Script | What it does |
|--------|--------------|
| [`ocria_common.py`](ocria_common.py) | shared plumbing: paths, seeding, the char-level loop metrics (parent port), CER, corpus loaders, ORT sessions, run passport |
| [`marchenko_pastur.py`](marchenko_pastur.py) | the parent's RMT primitives: MP density/CDF, robust σ² fit, BBP spike scan, TW purity gate (+ one documented patch: the Rayleigh branch for rectangular writer tensors) |
| [`rmt_fp/`](rmt_fp/) | the parent's verified formula library vendored verbatim: `mp`, `spiked`, `tracy_widom`, `shrinkage`, `entropy`, `hallucination`, `layer_ops` |
| [`rebuild_latin.py`](rebuild_latin.py) | int8 QDQ → fp32 torch rebuild of `rec_latin.onnx` + the 97.9% string-parity gate; writes `Results/latin_rebuild_fp32.pt` and `Results/rebuild_parity.json` |
| [`make_corpora.py`](make_corpora.py) | the six-family synthetic corpora (430 crops, seed 7): manifests + PNGs under `Data/` |
| [`ocria_probe.py`](ocria_probe.py) | the extended probe: per-family base metrics, per-stage Δh/v/z_diff, matrix scans, writer selection, Utility-Trap AUC contest, router + detector probes → `probe_report.json`, `ocria_vectors.npz`, `probe_gens.jsonl` |
| [`spectral_audit.py`](spectral_audit.py) | all twelve parent formulas executed on OCRia weights and live states → `formulas_audit.json` |
| [`ocria_train.py`](ocria_train.py) | formula-as-loss training, arms `classic` / `rcda_x`; frozen emission masks + refresh, hard good-scope anchor, MP leash, entropy band, inverse-BBP projection → training logs, states, deltas |
| [`ocria_router_round.py`](ocria_router_round.py) | **Round 3 — the router reject head**: deployed-graph feature cache, LDA-prototype init of the 5th row, violation-driven training (refresh-every port), 4 registered gates (text parity / flip bitwise / trunk identity / cure ≥ 0.90), two arms, winner export → `Model/router_reject.onnx` + meta + eval JSONs + deltas |
| [`ocria_benchmark.py`](ocria_benchmark.py) | THE one evaluator: six families + parity + spectral section per arm → `eval_{arm}.json` + rows |
| [`ocria_summarise.py`](ocria_summarise.py) | `summary_all_arms.json` + `RESULTS_SUMMARY.md` (incl. the router-round section) |
| [`make_graphics.py`](make_graphics.py) | 14 figures + 4 GIFs; every data point is a JSON field |

## The standard chain

```bash
python make_corpora.py
python rebuild_latin.py
python ocria_probe.py
python spectral_audit.py
python ocria_train.py --variant classic --steps 150
python ocria_train.py --variant rcda_x  --steps 150
python ocria_benchmark.py --arm base
python ocria_benchmark.py --arm classic
python ocria_benchmark.py --arm rcda_x
python ocria_router_round.py       # round 3: F4's reject class (head edit)
python ocria_summarise.py
python make_graphics.py all        # or per-figure: python make_graphics.py fig04
```

## Notes for the next round

- **CJK flip misses** (F5) live in the router's flip head — same head-edit
  caveat as round 3; the reject-row machinery (LDA init + violation-driven
  training + gates) ports directly if the flip corpus grows.
- **QAT deployment**: the fp32 deltas would be re-quantised with a short
  quantisation-aware fine-tune against the same anchor corpus.
- **LayoutNet**: needs the real `liteparse` feature builder; do not fake it.
- The parent's RCDA-8-style round (real-model re-run with a supervised arm
  for comparison) maps here to a CTC-supervised arm against the same anchor —
  the eval already carries the CER endpoints it would need.
