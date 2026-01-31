from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModel


@dataclass
class XGBModel(BaseModel):
    """
    Gradient boosting model using XGBoost.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 236040

    _clf: xgb.XGBClassifier | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> XGBModel:
        self._clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=1,  # Use single thread when parallel folds are running
            eval_metric="logloss",
            verbosity=0,  # Suppress XGBoost output
        )
        # Handle both pandas and numpy inputs
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        self._clf.fit(X_arr, y_arr, verbose=False)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        X_arr = X.values if hasattr(X, 'values') else X
        return self._clf.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        assert self._clf is not None
        X_arr = X.values if hasattr(X, 'values') else X
        return self._clf.predict(X_arr)
