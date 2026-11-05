"""common/ — shared HF-model infrastructure for stages 2–5."""

from stages.common.generation import (
    generate_with_capture,
    score_continuation,
    score_candidates,
)
from stages.common.hf_model import load_causal_lm, model_passport, resolve_final_norm, decoder_layers
from stages.common.passports import build_passport
from stages.common.reporting import write_report

__all__ = [
    "generate_with_capture",
    "score_continuation",
    "score_candidates",
    "load_causal_lm",
    "model_passport",
    "resolve_final_norm",
    "decoder_layers",
    "build_passport",
    "write_report",
]
