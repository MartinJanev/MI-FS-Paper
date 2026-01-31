from __future__ import annotations

import warnings
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def compute_mi_lookup(
    X_train_disc: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 3,
    random_state: int | None = None,
    n_jobs: int = -1,
    device: str | None = None,
    use_gpu_approximation: bool = False,
) -> np.ndarray:
    """
    Compute I(X_j; Y) for all features.

    Parameters
    ----------
    X_train_disc : np.ndarray
        Discretized training features, shape (n_train, d).
    y_train : np.ndarray
        Training labels, shape (n_train,).
    n_neighbors : int, default=3
        Number of neighbors for MI estimator.
    random_state : int or None
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs to use (-1 means use all available).
    device : str or None
        Device for GPU computation ("cuda", "cuda:0", "cpu", or None for auto).
        Only used if use_gpu_approximation=True.
    use_gpu_approximation : bool, default=False
        If True, use GPU-accelerated correlation-based MI approximation.
        Much faster but less accurate. If False, use sklearn's exact MI.

    Returns
    -------
    mi_scores : np.ndarray
        MI scores for each feature, shape (d,).
    """
    # GPU-accelerated approximation
    if use_gpu_approximation:
        from ...core.fs.mi_gpu import compute_mi_gpu_correlation
        return compute_mi_gpu_correlation(X_train_disc, y_train, device=device)

    # Standard sklearn implementation (CPU)
    # Reduce parallelism for large datasets to avoid memory issues
    n_samples, n_features = X_train_disc.shape
    if n_samples > 100000 or n_features > 500:
        # Use fewer parallel jobs to reduce memory pressure
        n_jobs = min(4, n_jobs) if n_jobs == -1 else min(4, n_jobs)
        warnings.warn(
            f"Large dataset detected ({n_samples} samples, {n_features} features). "
            f"Reducing parallel jobs to {n_jobs} to avoid memory errors."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        mi_scores = mutual_info_classif(
            X_train_disc,
            y_train,
            discrete_features=True,  # Features are already discretized
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    return mi_scores
# src/mi_fs_benchmark/preprocessing/__init__.py
"""Preprocessing utilities for fold artifact generation."""

from .discretize import discretize_features
from .mi_precompute import compute_mi_lookup

__all__ = ["discretize_features", "compute_mi_lookup"]
