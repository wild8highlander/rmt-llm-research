"""
predictors.baselines — mandatory simple baselines.

A spectral feature is only useful when it improves the prediction over
simple alternatives:

  ConstantRateBaseline — always predicts the base error rate of the
                         training fold (the lower bound of meaningfulness);
  FamilyRateBaseline   — the base rate per task family (the family is
                         known at test time; a strong "naive" baseline).
"""

from __future__ import annotations

import numpy as np


class ConstantRateBaseline:
    """p = the positive-class base rate in train.

    A degenerate train (one class) is not forbidden: a constant is
    predicted, and the degeneracy is honestly recorded by the metrics
    (ROC-AUC on a single-class test is undefined -> None in the report).
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
    """p = the smoothed positive-class rate for the task's family.

    Requires passing the families at fit/predict (as a separate argument —
    families count as "task metadata" available at test time). Laplace
    smoothing protects against zero rates on small families.
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
