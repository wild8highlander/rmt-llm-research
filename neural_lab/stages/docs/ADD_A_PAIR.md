# ADD_A_PAIR — how to add a base↔tuned pair to Stage 5

Stage 5's thesis is only testable WITHIN a model family: same
pretraining distribution, same tokenizer lineage, only the tuning
method varies. Adding a pair to `stage5_rlhf/model_pairs.yaml`:

1. **Same lineage, verified.** The two repos must share the tokenizer
   (`assert_same_tokenizer_lineage` compares vocabularies at run time
   and FAILS the pair otherwise). If the tuned checkpoint genuinely
   shipped a different tokenizer, it is not a valid pair for this
   protocol — do not add it.
2. **Real artifacts only.** `base` / `tuned` must be live HF repos
   (or pinned revisions). For checkpoint-ablation entries (NOT RLHF
   pairs — used for the stage-3 scaling family), pin
   `base_revision` / `tuned_revision` explicitly; two runs of the same
   repo without revisions silently produce Δ = 0.
3. **Fill the metadata honestly.** `method` is what the authors
   document (sft | rlhf | dpo | rlhf+dpo | checkpoint-ablation);
   `access` is `gated` when a license acceptance or token is needed
   (the runner skips gated pairs with an explicit message);
   `params` is the approximate parameter count (for the scaling
   tables); `chat_tuned` says whether `tuned` expects a chat template.
4. **Sanity-check the registry.** The registry shape is covered by
   `test_stage5_pairs.py` (every pair has the required fields, same-org
   convention for same-family tuning). Extend that test when you add a
   field, and keep `params` consistent with the model card — it feeds
   the cross-pair scaling aggregates.

Then run:

```text
python3 neural_lab/stages/stage5_rlhf/run_stage5.py --list
python3 neural_lab/stages/stage5_rlhf/run_stage5.py --pair <id> \
    --out results/stage5 --dtype bfloat16 --device cuda   # for 7B+
```

Each pair writes `record.json` (schema-validated), a stage-3-compatible
`fit.json` sidecar (consumed by `run_stage3 --scaling-in`), and a
REPORT.md with the pre-registered criterion evaluated on the record.
