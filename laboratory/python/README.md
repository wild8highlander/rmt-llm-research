# Python Laboratory (reference implementation)

The canonical implementation of the multi-language lab. Two parallel
editions live here:

| Folder | Content |
|---|---|
| [`lab_en/`](lab_en/) | English UI — the actively developed edition with the full TinyGPT v3 feature set (RoPE, GQA, pre-LN), BPE tokenizer v3, 3D research pipeline, and the complete test suite (`tests/`). |
| [`lab_ru/`](lab_ru/) | Russian UI — feature-frozen twin kept for bilingual parity; CI lint and the cross-implementation checks cover both editions. |

## Run (lab_en)

```bash
cd laboratory/python/lab_en
pip install -r requirements.txt
python main.py             # interactive menu
python run_full_lab.py     # full pipeline: scenarios + experiments + reports
```

The menu drives every stage: parameter space, scenario verification,
1D/3D research runs, TinyGPT training/inference, chart and report export.
Generated artifacts land in `results/` (per-run reports, charts, models).

Tests (also run in CI):

```bash
pytest tests/ -v                       # 350+ tests
pytest tests/test_hypothesis.py -v     # property-based (hypothesis)
pytest tests/test_benchmark.py --benchmark-only   # perf benchmarks
```
