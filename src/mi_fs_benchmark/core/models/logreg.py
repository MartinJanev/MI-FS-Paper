from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


@dataclass
class LogisticRegressionModel(BaseModel):
    """
    Thin wrapper around scikit-learn logistic regression.

    Uses default L2 regularization (penalty argument omitted)
    for compatibility with scikit-learn >= 1.8.
    """

    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 200

    _clf: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LogisticRegressionModel:
        self._clf = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
        )
        # Handle both pandas and numpy inputs
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        self._clf.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        X_arr = X.values if hasattr(X, 'values') else X
        return self._clf.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        X_arr = X.values if hasattr(X, 'values') else X
        return self._clf.predict(X_arr)
