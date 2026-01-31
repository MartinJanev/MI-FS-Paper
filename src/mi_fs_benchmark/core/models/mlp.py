from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from .base import BaseModel


@dataclass
class MLPModel(BaseModel):
    """
    Simple feed-forward neural network for tabular data.
    """

    hidden_layer_sizes: tuple[int, ...] = (128, 64)
    max_iter: int = 200
    random_state: int = 236040

    _clf: MLPClassifier | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MLPModel:
        self._clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(X.values, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict_proba(X.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict(X.values)
