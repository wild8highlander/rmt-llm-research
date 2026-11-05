"""
neural_lab.stages — empirical program for research stages 2–5 on real HF models.

Stages (see README.md for the full mapping):

  stage2_detector         — hallucination detector vs baselines on public
                            benchmarks (TruthfulQA, HaluEval, SelfCheckGPT),
                            AUROC against token entropy, semantic entropy and
                            linear probes;
  stage3_ncrit            — token-level error curves, survival fit of
                            T_crit = c * mu_eff^(-1/beta), cross-model scaling;
  stage4_interpretability — bridge to mechanisms: logit lens, activation
                            patching, SAE (Gemma Scope) audit, heavy-tail
                            alpha (Martin & Mahoney / WeightWatcher);
  stage5_rlhf             — base vs instruct/DPO paired comparison of spectra
                            and N_crit, mu_eff proxy via KL divergence.

Ground rules inherited from neural_lab/science:
  - every run carries a passport (model revision + sha256, seeds, config);
  - labels are established independently of the scores being evaluated;
  - a negative result is still a result and is reported as such;
  - reports are generated from data, never written by hand.

Real-model runs need: torch + transformers (and datasets for the benchmark
loaders). Unit tests are offline: they use tiny randomly-initialized GPT-2
models and synthetic data, no network, no pretrained weights.
"""

__version__ = "0.1.0"

PROTOCOL_VERSION = "1.1"
