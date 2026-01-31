from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .base import BaseModel


@dataclass
class SVMModel(BaseModel):
    """
    SVM with probability calibration (via Platt scaling in SVC).
    """

    C: float = 1.0
    kernel: str = "linear"
    probability: bool = True

    _clf: SVC | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> SVMModel:
        self._clf = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=self.probability,
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
