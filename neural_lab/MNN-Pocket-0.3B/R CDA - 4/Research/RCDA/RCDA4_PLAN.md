# RCDA-4: Weight-Space Programming after MIT CSAIL — experiment plan

Written: November 9, 2026. Status: plan; execution — the next run.
Every link and figure in section 1 was verified against the primary source (read in
full); sections 2–5 are tied to our existing RCDA bench (rounds 1–3,
MNN-Pocket-0.3B, CPU).

---

## 1. What was published (verified against the primary source)

**"Sparks of Weight-Space Programming: LLMs can natively read & write neural
network weights"** — Rikiya Takehi, Jayden Teo, Alex Zhang, Omar Khattab (MIT CSAIL),
November 2, 2026:

- primary source: https://rikiyat.github.io/blog/weight-space-programming
- full logs of all agent sessions: https://github.com/RikiyaT/lm_understand-logs
- the edited model on HF: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- the base: https://huggingface.co/Qwen/Qwen3.5-0.8B (ungated, one safetensors)
- a retelling on Habr: https://habr.com/ru/post/1103198/

The gist: a frontier LLM (GPT-6 Astra) in 8 experiments reads foreign weights as text
or through a Python kernel and writes them without gradients or training. The key one
for us is **experiment 8** — "make Qwen3.5-0.8B better at math by editing weights":

- the agent got a Python kernel over the weights, GSM8K/MATH train prompts and a
  budget of **10 / 25 / 50 forward passes** (a pass = answering up to 512 problems or
  reading up to 1024 texts); training and gradients are forbidden;
- diagnosis: small reasoning models **loop** (85 of 96 first answers hit the
  32,768-token limit); Astra traced the loop to **one direction written by the MLP of
  layers 15 and 19** and **subtracted it from 64 output columns in each** — exactly
  131,072 weights = 0.016% of the model;
- result (50 passes): MATH-500 7.9→48.1, GSM8K 13.8→67.1, IFEval 21.0→62.3; the share
  of answers hitting the limit: 93%→33.1%. At 10–25 passes the loops "barely went
  away" — the verification budget is critical;
- a negative result: Gemma 3 270M does not loop — nothing to edit; targeted edits
  moved MATH-500 by less than 1 point. The authors' conclusion: a weight edit can
  **remove an obstacle in front of an existing ability, but not add a new one**;
- in experiment 3 Astra found, from the weights alone (without running the network),
  all induction heads in three 7.2M models — the Elhage et al. method, which she was
  not told about;
- Habr's caveats: a large share of the gain is "the model was finally allowed to
  finish"; an independent reproduction **does not exist yet**; dual use (the same
  technique can implant backdoors — the Carlini work, arxiv 2610.37741).

## 2. Why this is a direct analog of our RCDA

We already did the same thing manually and with other instruments:

| MIT / Astra | Our RCDA (rounds 1–3) |
| --- | --- |
| "one direction in activations, written by MLP" | the PR vector v from the Petrov–Ring SVD spectrum (PM-energy share 0.9995) |
| subtract the direction from 64 columns of W_out of two layers | zero/scale the TW-passing columns of W_o (190,000 weights ≈ 0.027%) |
| layers 15 and 19 of 24 (depth ~0.63) | layers 15, 16, 18 of 24 (the same ~2/3-depth zone) |
| diagnosis via forward passes, budget 10–50 | diagnosis via the PR monitor + spectral thresholds θ |
| verification: MATH-500/GSM8K benchmarks | our benchmark: ρ (defect rate), in-loop waste, distinct-4, KL, PPL |
| induction heads from weights (exp 3) | head concentration L15→h7 (×3.11), L16→h8 (×2.22), placebo control (round 3) |

What we have that MIT does not: **the spectral TW gate** (nobody checked their edit
for spectral integrity — in round 3 zeroing a whole head produced a violation while
selective columns passed), **placebo control**, persistent policies (C' = 0.10
without touching weights), and a ready reproducible bench with pre-registered rules.

Our main round-4 question: **can an LLM programmer replace the human in the RCDA
loop** — and how to verify someone else's edit spectrally.

## 3. The three experiment parts

### Part A — independent verification of the MIT edit (cheap, ~1–1.5 h)

The first independent reproduction (Habr: "does not exist yet") — on their own
published weights, but on our narrow looping benchmark. Relevance: by their own
table, the edit's main mechanism is escaping loops (limit truncations 93→33%).

- Download both models (~1.6 GB each, ungated). Hardware: 3 GB RAM — load strictly
  one at a time, bf16, `del model + gc` between arms; 2 CPU cores.
- Prompts: our standard set (12 bad + 6 good), plus 6 fresh GSM8K-train prompts in
  English (their fix was measured on math; test sets are untouched). Sampling per the
  card: temperature 0.6, top-p 0.95, thinking on, max_new_tokens 256.
- Metrics: ρ (share of looping), mean in-loop waste, share truncated at the limit,
  distinct-4; on good — ρ_good and text "changedness".
- **Gate A (pre-registered):** ρ_edited ≤ 0.5·ρ_base and truncations ↓ — confirmation;
  otherwise an honest negative (the fix is specific to their prompts/language, or does
  not transfer to the narrow benchmark) — also a result.

### Part B — the trick itself on our bench: a GLM programmer (~2–3 h)

Astra's role is played by **GLM via the z-ai SDK** (Node, multi-turn; we execute its
code commands in our sandbox) over MNN-Pocket-0.3B with the looping defect. A full
analog of their harness: a Python kernel over the state_dict + a "pass" budget.

- Harness (a new `rcda4_wsp.py` + the agent loop): the agent has `list_tensors()`,
  `stats(name)`, `slice(name, rows, cols)` and the right to send python code that we
  execute over the loaded state_dict (read/analyze and edit weights). A separate
  `benchmark()` tool = a mini-run of 8 bad prompts (this is its "forward passes"):
  **a budget of 10 calls** + 20 code commands, each call's result ≤ 3 KB in context.
- Two arms (an ablation MIT does not have):
  - **blind** — only generic tensor descriptions; none of our spectral diagnostics
    (the PR vector, head profiles, θ thresholds are not shown);
  - **aided** — the same tools plus our diagnostics as first-class data.
- **Gates B (pre-registered):** success = ρ on the mini-benchmark ≤ 0.35, the final
  edit passes the TW gate, KL on 5 control texts ≤ 0.15. Hypotheses:
  - H4a: aided matches or beats manual RCDA-2 (ρ 0.65 + TW pass);
  - H4b: blind fails or the edit is spectrally dirty (violation) — the value is in
    the diagnostics, not in "an agent per se";
- A successful arm's finale: the full edit → the full benchmark (20 bad + 20 good) +
  KL/PPL + **the cosine of the agent-found direction with our PR vector**
  ("did the agent find THE direction or another working one").
- Unique value versus MIT: (a) a non-frontier programmer (GLM ≠ GPT-6) — a test of
  the trick's scalability down the model-class ladder; (b) spectral hygiene of edits,
  which the authors lack; (c) the blind/aided ablation.

### Part C — "reading weights" against our ground truth (~30 min, can run in parallel with B)

Give GLM slices of the W_o of layers 15/16/18 without head labels and ask it to name
the hub/copier heads. The score against the measured one (L15h7 ×3.11, L16h8 ×2.22,
L18h12 ×1.91; median heads = the round-3 placebo). Metric: top-1 and top-3 hits per
layer. This is our analog of MIT's experiment 3 — on a model where the truth is
already measured.

## 4. Risks and execution order

- **Order:** C → A → B. If time is limited — A gives the most new value per minute
  (an independent check nobody else has).
- 3 GB RAM: one model at a time, bf16, mandatory `del + gc`; 0.8B ≈ 1.6 GB.
- 2 CPU cores: 0.8B ≈ 2–3 tok/s → a 256-token generation ≈ 2 min → an 18-prompt arm ≈
  35–40 min; plan overnight background runs.
- ~3.2 GB of downloads: split into parts, resume with `curl -C -`.
- GLM context: every result ≤ 3 KB, history = summaries only (as at MIT:
  "show the agent aggregates, not raw tensors").
- Deliverability of the edit: the state_dict overlay-edit mechanism already exists in
  `reassemble_model.py --overlay_state` (round 1) and the round-2 edit scripts.

## 5. Deliverable

`Research/RCDA/RCDA4_REPORT.md` in the repository (modeled on rounds 1–3):
protocols of all three parts, pre-registered gates and their outcomes, the agent-B
trace (prompts/commands/results — as at MIT in lm_understand-logs), fig14 (the ρ
trajectory over the agent's iterations), an update of `Docs/RCDA_RUNBOOK.md`, a zip
rebuild.

The honest frame: our benchmark measures looping, not mathematical correctness;
their fix may partially lie outside our metric — that is why Part A explicitly
includes the GSM8K-train subset and the truncation share, not only ρ.
