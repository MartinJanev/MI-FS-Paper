from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract interface for downstream classifiers.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """
        Fit the model on training data.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities. Must return array of shape (n_samples, 2)
        for binary classification where column 1 is P(y=1).
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict hard labels (0/1).
        """
