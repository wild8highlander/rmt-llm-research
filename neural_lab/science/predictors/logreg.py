"""
predictors.logreg — logistic regression on pure NumPy (ADR-001).

IRLS/Newton with L2 regularization; features are standardized with the
statistics of the TRAINING fold (the standardization parameters are part
of the model, so no test statistics leak into preprocessing).
"""

from __future__ import annotations

import numpy as np


class LogisticRegression:
    """L2-regularized logistic regression (Newton, full-batch)."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.beta_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None

    # -- internal -------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu_) / self.sd_
        return np.hstack([np.ones((Z.shape[0], 1)), Z])

    # -- API -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X must be (n, d), y must be (n,)")
        if len(np.unique(y)) < 2:
            raise ValueError("need both classes in training fold")
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-12] = 1.0  # constant columns are not divided by 0
        A = self._design(X)
        d = A.shape[1]
        beta = np.zeros(d)
        reg = np.eye(d) * self.l2
        reg[0, 0] = 0.0  # the intercept is not regularized
        for _ in range(self.max_iter):
            p = self._sigmoid(A @ beta)
            W = np.clip(p * (1 - p), 1e-10, None)
            grad = A.T @ (y - p) - reg @ beta
            H = (A * W[:, None]).T @ A + reg
            step = np.linalg.solve(H + 1e-10 * np.eye(d), grad)
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.beta_ = beta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("fit before predict")
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(self._design(X) @ self.beta_)

    def fit_predict_oof(self, X, y, test_idx: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
        """Honest out-of-fold: fit on train_idx, predict test_idx."""
        self.fit(X[train_idx], y[train_idx])
        return self.predict_proba(X[test_idx])
