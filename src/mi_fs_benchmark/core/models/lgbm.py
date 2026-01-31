from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseModel


@dataclass
class LGBMModel(BaseModel):
    """
    Gradient boosting on decision trees using LightGBM.
    """

    num_leaves: int = 64
    n_estimators: int = 400
    learning_rate: float = 0.05
    random_state: int = 236040

    _clf: lgb.LGBMClassifier | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> LGBMModel:
        self._clf = lgb.LGBMClassifier(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=-1,  # Suppress LightGBM output
            force_row_wise=True,  # Suppress threading warnings
        )
        # Suppress sklearn feature name warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            self._clf.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            return self._clf.predict_proba(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            return self._clf.predict(X)
