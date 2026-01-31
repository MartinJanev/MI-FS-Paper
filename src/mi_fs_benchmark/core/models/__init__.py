from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from .base import BaseModel


@dataclass
class LogisticRegressionModel(BaseModel):
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 200
    _clf: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BaseModel:
        self._clf = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
        )
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(Y := y, "values") else y  # noqa: N806
        self._clf.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LogisticRegressionModel not fitted")
        X_arr = X.values if hasattr(X, "values") else X
        return self._clf.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LogisticRegressionModel not fitted")
        X_arr = X.values if hasattr(X, "values") else X
        return self._clf.predict(X_arr)


# H
@dataclass
class HistGBDTModel(BaseModel):
    learning_rate: float = 0.1
    max_depth: int | None = 5
    n_estimators: int = 150
    max_leaf_nodes: int | None = None
    random_state: int | None = 0
    _clf: HistGradientBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> BaseModel:
        self._clf = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            max_iter=self.n_estimators,
            random_state=self.random_state,
        )
        X_arr = X.values if hasattr(X, "values") else X
        y_arr = y.values if hasattr(Y := y, "values") else y  # noqa: N806
        self._clf.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("HistGBDTModel not fitted")
        X_arr = X.values if hasattr(X, "values") else X
        # HGB has predict_proba when binary; fall back to decision_function if absent
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X_arr)
        decision = self._clf.decision_function(X_arr)
        # Map decision_function to probabilities via sigmoid
        proba_pos = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - proba_pos, proba_pos]).T

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("HistGBDTModel not fitted")
        X_arr = X.values if hasattr(X, "values") else X
        return self._clf.predict(X_arr)


def create_model(name: str, **kwargs) -> BaseModel:
    mapping = {
        "logreg": LogisticRegressionModel,
        "hgbt": HistGBDTModel,
    }
    try:
        cls = mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {name!r}") from exc
    return cls(**kwargs)


__all__ = [
    "BaseModel",
    "create_model",
    "LogisticRegressionModel",
    "HistGBDTModel",
]
