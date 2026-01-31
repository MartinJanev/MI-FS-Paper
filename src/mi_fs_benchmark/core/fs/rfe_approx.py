# src/mi_fs_benchmark/fs/rfe_approx.py
from __future__ import annotations

import warnings
import numpy as np
from sklearn.svm import LinearSVC

from .base import FeatureSelector


class RFEApproxSelector(FeatureSelector):
    """
    Approximate Recursive Feature Elimination (RFE).

    For scalability, this implementation:
    - trains a linear SVM once,
    - uses the absolute value of coefficients as importance scores.

    It behaves like a single-step RFE with ranking by |w_j|.

    NEW: Works with numpy arrays directly.
    Suppresses convergence warnings for cleaner output.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self._ranking: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RFEApproxSelector:
        # Suppress sklearn convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            clf = LinearSVC(C=self.C, dual=False)
            clf.fit(X, y)

        coef = clf.coef_
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        scores = np.abs(coef)
        self._ranking = np.argsort(scores)[::-1]
        return self

    def rank_features(self) -> np.ndarray:
        if self._ranking is None:
            raise RuntimeError("RFEApproxSelector not fitted.")
        return self._ranking
