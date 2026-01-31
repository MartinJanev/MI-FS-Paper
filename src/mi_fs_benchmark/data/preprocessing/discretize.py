# src/mi_fs_benchmark/preprocessing/discretize.py
"""
Feature discretization for MI estimation.

Discretization improves MI estimation stability by:
- Reducing the impact of outliers
- Enabling consistent binning across train/test
- Making MI estimators more reliable on continuous features
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def discretize_features(
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_bins: int = 5,
        strategy: str = "quantile",
        variance_threshold: float = 1e-8,
        subsample: int | None = 200000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous features for MI estimation.

    Fits discretizer on training data and applies to both train and test.
    Handles low-variance features by using fewer bins or assigning constant values.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, d).
    X_test : np.ndarray
        Test features, shape (n_test, d).
    n_bins : int, default=10
        Number of bins per feature.
    strategy : str, default="quantile"
        Discretization strategy: "uniform", "quantile", or "kmeans".
    variance_threshold : float, default=1e-8
        Features with variance below this threshold are treated as constant.
    subsample : int or None, default=200000
        Maximum number of samples used to compute quantiles for computational efficiency.
        Use None to disable subsampling.

    Returns
    -------
    X_train_disc : np.ndarray
        Discretized training features, shape (n_train, d).
    X_test_disc : np.ndarray
        Discretized test features, shape (n_test, d).
    """
    n_features = X_train.shape[1]
    X_train_disc = np.zeros_like(X_train)
    X_test_disc = np.zeros_like(X_test)

    # Identify low-variance features
    feature_vars = np.var(X_train, axis=0)
    low_var_mask = feature_vars < variance_threshold

    # For low-variance features, assign constant bin (0)
    if np.any(low_var_mask):
        X_train_disc[:, low_var_mask] = 0
        X_test_disc[:, low_var_mask] = 0

    # For normal features, use KBinsDiscretizer
    normal_mask = ~low_var_mask
    if np.any(normal_mask):
        # Suppress warnings about small bin widths and future warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Bins whose width are too small.*are removed',
                category=UserWarning
            )
            warnings.filterwarnings(
                'ignore',
                message='Feature .* is constant and will be replaced with 0',
                category=UserWarning
            )
            warnings.filterwarnings(
                'ignore',
                category=FutureWarning
            )

            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy=strategy,
                subsample=subsample,
                quantile_method="averaged_inverted_cdf",
            )

            X_train_disc[:, normal_mask] = discretizer.fit_transform(X_train[:, normal_mask])
            X_test_disc[:, normal_mask] = discretizer.transform(X_test[:, normal_mask])

    return X_train_disc, X_test_disc
