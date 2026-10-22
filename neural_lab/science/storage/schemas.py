"""storage.schemas — машиночитаемые схемы: задача, паспорт, предсказание.

Эксперимент обязан иметь паспорт (experiment passport): формат
фиксирует ВСЁ, что нужно для воспроизведения, и заполняется из реальных
артефактов, а не из фиктивных строк.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Task:
    """Задача с проверяемым ответом (истина устанавливается независимо —
    перекомпиляцией формулы библиотекой, а не строкой из корпуса)."""

    task_id: str
    family: str  # семейство формул (mp_bounds, bbp_lambda_max, ...)
    band: str  # indist | near | far  (градиент сложности)
    prompt: str
    ground_truth: str  # эталонное продолжение (пересчитано библиотекой)
    params: dict = field(default_factory=dict)
    in_dist: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentPassport:
    """Паспорт запуска: всё, что нужно для проверки воспроизводимости."""

    experiment_id: str
    hypothesis_id: str
    protocol_version: str
    seed: int
    model_revision: str  # sha256 весов
    tokenizer_revision: str  # sha256 токенизатора
    model_config: dict
    dataset_hash: str  # sha256 канонического JSON набора задач
    corpus_train_sha256: str
    split_scheme: str  # grouped_kfold_v1
    split_params: dict
    feature_groups: dict  # имя группы -> список имён признаков
    predictors: list  # список конфигураций предсказателей
    primary_metric: str
    tolerance_grid: list  # [0.15, 0.25, 0.40] — зафиксировано заранее
    generation: dict  # temperature, n_samples, max_new_tokens
    environment: dict
    exclusions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictionRecord:
    """Одно предсказание на закрытом тестовом наблюдении.

    Сохраняется ДО раскрытия метрик; y_true попадает сюда только как
    справочное поле для последующего сравнения, в фит он не входит.
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
