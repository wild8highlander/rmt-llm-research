"""
predictors.baselines — обязательные простые baseline'ы.

Спектральный признак полезен только тогда, когда улучшает прогноз
относительно простых альтернатив:

  ConstantRateBaseline — всегда предсказывает базовую частоту ошибки
                         на обучающем фолде (нижняя граница осмысленности);
  FamilyRateBaseline   — базовая частота по семейству задач (семейство
                         известно на тесте; сильный «наивный» baseline).
"""

from __future__ import annotations

import numpy as np


class ConstantRateBaseline:
    """p = базовая частота положительного класса в train.

    Вырожденный train (один класс) не запрещён: предсказывается константа,
    а вырождение честно фиксируется метриками (ROC-AUC на одноклассовом
    тесте не определён -> None в отчёте).
    """

    def __init__(self):
        self.rate_: float | None = None

    def fit(self, X, y) -> ConstantRateBaseline:  # noqa: ARG002
        y = np.asarray(y, dtype=np.float64)
        self.rate_ = float(y.mean())
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.rate_ is None:
            raise RuntimeError("fit before predict")
        return np.full(len(X), self.rate_, dtype=np.float64)


class FamilyRateBaseline:
    """p = сглаженная частота положительного класса по семейству задачи.

    Требует передачи семейств при fit/predict (отдельным аргументом —
    семьи считаются «метаданными задачи», доступными на тесте).
    Сглаживание Лапласа защищает от нулевых частот на малых семействах.
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = float(smoothing)
        self.rate_: float | None = None
        self.family_rate: dict[str, float] = {}

    def fit(self, X, y, families=None) -> FamilyRateBaseline:  # noqa: ARG002
        y = np.asarray(y, dtype=np.float64)
        self.rate_ = float(y.mean())
        self.family_rate = {}
        if families is not None:
            for fam in np.unique(np.asarray(families)):
                m = np.asarray(families) == fam
                self.family_rate[str(fam)] = float(
                    (y[m].sum() + self.smoothing) / (m.sum() + 2 * self.smoothing)
                )
        return self

    def predict_proba(self, X, families=None) -> np.ndarray:
        if self.rate_ is None:
            raise RuntimeError("fit before predict")
        if families is None:
            return np.full(len(X), self.rate_, dtype=np.float64)
        return np.array(
            [self.family_rate.get(str(f), self.rate_) for f in families],
            dtype=np.float64,
        )
