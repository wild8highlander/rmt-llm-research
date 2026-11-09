# TinyGPT unit tests (lab_en)

101 tests covering the from-scratch implementation: BPE tokenizer, model
forward, Adam optimizer, backward pass (numerical gradient check) and
end-to-end training on a toy corpus.

```bash
cd laboratory/python/lab_en
python -m pytest tests -q
```

The tests are part of CI; keep them green when touching `lab_en` code.
