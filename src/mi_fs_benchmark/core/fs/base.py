# src/mi_fs_benchmark/fs/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FeatureSelector(ABC):
    """
    Strategy interface for feature-selection methods.

    KEY ARCHITECTURAL RULES:
    - Selectors receive ONLY preprocessed numpy arrays (no DataFrames)
    - Selectors must be pure functions of training data
    - Selectors never scale, discretize, or read files
    - Selectors never access test data
    - If a selector needs MI, it receives it as an argument

    Implementations should:
    - be fitted on training data only (no leakage),
    - store internal scores / rankings,
    - provide a stable ranking of feature indices (0..d-1).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        """
        Fit selector on training data.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_train, d).
        y : np.ndarray
            Training labels, shape (n_train,).

        Returns
        -------
        self
        """

    @abstractmethod
    def rank_features(self) -> np.ndarray:
        """
        Return a numpy array of feature indices, sorted from most to
        least important.

        Returns
        -------
        ranking : np.ndarray
            Feature indices in descending order of importance, shape (d,).
        """

    def select_k(self, k: int) -> np.ndarray:
        """
        Return indices of the top-k features.

        Parameters
        ----------
        k : int
            Number of features to select.

        Returns
        -------
        selected : np.ndarray
            Indices of top-k features, shape (k,).
        """
        ranking = self.rank_features()
        return ranking[:k]
