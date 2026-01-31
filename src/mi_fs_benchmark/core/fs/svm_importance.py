# src/mi_fs_benchmark/fs/svm_importance.py
from __future__ import annotations

import warnings
import numpy as np
from sklearn.svm import LinearSVC

from .base import FeatureSelector


class SVMImportanceSelector(FeatureSelector):
    """
    Feature selector based on linear SVM coefficients.

    Uses LinearSVC (hinge loss) and ranks features by |w_j|.

    NEW: Works with numpy arrays directly.
    Suppresses convergence warnings for cleaner output.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self._weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SVMImportanceSelector:
        # Suppress sklearn convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            clf = LinearSVC(C=self.C, dual=False)
            clf.fit(X, y)

        w = clf.coef_
        if w.ndim == 2:
            w = np.mean(np.abs(w), axis=0)
        self._weights = np.abs(w)
        return self

    def rank_features(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("SVMImportanceSelector not fitted.")
        return np.argsort(self._weights)[::-1]

    @property
    def weights_(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("SVMImportanceSelector not fitted.")
        return self._weights
