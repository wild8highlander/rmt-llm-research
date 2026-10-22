"""storage.schemas — machine-readable schemas: task, passport, prediction.

An experiment must have a passport (experiment passport): the format
pins EVERYTHING needed for reproduction and is filled from real
artifacts, not from fictitious strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    """A task with a verifiable answer (truth is established independently —
    by recomputing the formula with the library, not by a corpus string)."""

    task_id: str
    family: str  # formula family (mp_bounds, bbp_lambda_max, ...)
    band: str  # indist | near | far  (difficulty gradient)
    prompt: str
    ground_truth: str  # reference continuation (recomputed by the library)
    params: dict = field(default_factory=dict)
    in_dist: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentPassport:
    """Run passport: everything needed to verify reproducibility."""

    experiment_id: str
    hypothesis_id: str
    protocol_version: str
    seed: int
    model_revision: str  # sha256 of the weights
    tokenizer_revision: str  # sha256 of the tokenizer
    model_config: dict
    dataset_hash: str  # sha256 of the canonical JSON of the task set
    corpus_train_sha256: str
    split_scheme: str  # grouped_kfold_v1
    split_params: dict
    feature_groups: dict  # group name -> list of feature names
    predictors: list  # list of predictor configurations
    primary_metric: str
    tolerance_grid: list  # [0.15, 0.25, 0.40] — frozen in advance
    generation: dict  # temperature, n_samples, max_new_tokens
    environment: dict
    exclusions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictionRecord:
    """One prediction on a sealed test observation.

    Stored BEFORE metrics are unblinded; y_true only lands here as a
    reference field for the subsequent comparison — it never enters the
    fit.
    """

    task_id: str
    sample_id: str
    band: str
    family: str
    fold: int
    y_true: int
    p_base: float
    p_confidence: float
    p_internal: float
    p_combined: float
    p_combined_minus_spectral: float

    def to_dict(self) -> dict:
        return asdict(self)
