<p align="center">
  <img src="../../../assets/logos/sc_data.banner.svg" width="760" alt="science/data — animated logo: database disks with falling data packets">
</p>

# 🗃️ science/data — Tasks & Splits

<p align="center">
  <em>Verified tasks and leakage-free splits for every experiment</em><br><br>
  <a href="../README.md"><img src="https://img.shields.io/badge/module-science--data-38bdf8" alt="data"></a>
  <img src="https://img.shields.io/badge/113%20tasks-22d3ee" alt="113 tasks"><img src="https://img.shields.io/badge/grouped__kfold__v1-22d3ee" alt="grouped k-fold">
</p>

The measurement substrate of the science core: the task set and the
split policy. Its defining rule is **independent verification** — the
ground truth of every task is *recomputed* by the `src/rmt_llm` library
at runtime, never stored as a static string, so a task cannot go stale
or carry a wrong label. On top of that sits a split policy designed
against the most subtle leak in this kind of study: near-duplicate tasks
landing on both sides of the train/test boundary.

## Modules

| File | Responsibility |
|---|---|
| `tasks.py` | the task set: 113 tasks across families, each rendered into a prompt and paired with a recomputed-true answer |
| `splits.py` | `grouped_kfold_v1` — k-fold grouping by task **family**, so siblings never straddle folds; deterministic under a fixed seed |
| `__init__.py` | public surface for the rest of the core |

## Why splits are grouped

Tasks in this core come in families (same formula, different
parameters). A random split would put family members in both train and
test, letting a predictor succeed by *recognizing the family* rather
than by reading the model's internal state — a textbook leakage channel.
Grouping by family closes it: the test folds measure generalization to
**unseen task families**, which is the claim the hypotheses actually
make.

## Dataset identity

Every run records the dataset hash in its passport (the current H1
dataset: 113 tasks, 565 observations, sha256 `411eb0e9bb23…`). If
`tasks.py` changes, the hash changes, and every previously published
result remains traceable to its exact input — the same provenance chain
the research program uses for corpora.

Consumed by [`experiments/run_h1.py`](../experiments/README.md); guards
against misuse live in [`evaluation/leakage_checks.py`](../evaluation/README.md).

---

<p align="center">
  <img src="../../../assets/logos/sc_data.avatar.svg" width="56" alt="science/data animated avatar"><br>
  <sub>
    <a href="../README.md">← science</a> · <a href="../../README.md">← neural_lab</a> · <a href="../instrumentation/README.md">instrumentation →</a>
  </sub>
</p>
