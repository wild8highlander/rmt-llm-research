"""Stage 2 — hallucination detector vs baselines (AUROC race)."""

from stages.stage2_detector.detectors import Unit, evaluate_detector, paired_delta_auc
from stages.stage2_detector.semantic_entropy import discrete_semantic_entropy, semantic_entropy_score
from stages.stage2_detector.token_entropy import token_entropy_features

__all__ = [
    "Unit",
    "evaluate_detector",
    "paired_delta_auc",
    "discrete_semantic_entropy",
    "semantic_entropy_score",
    "token_entropy_features",
]
