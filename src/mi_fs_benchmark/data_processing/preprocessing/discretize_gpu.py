# src/mi_fs_benchmark/preprocessing/discretize_gpu.py
"""
GPU-accelerated feature discretization for MI estimation.

Uses PyTorch for ~10-50x speedup on large datasets.
GPU-ONLY MODE: No CPU fallbacks - requires CUDA-enabled GPU.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Require PyTorch with CUDA
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-only mode enabled but CUDA is not available. "
            "Please ensure you have a CUDA-enabled GPU and PyTorch with CUDA support installed."
        )
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "GPU-only mode enabled but PyTorch is not installed. "
        "Install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    ) from e


def discretize_features_gpu(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_bins: int = 5,
    device: str | None = None,
    variance_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated feature discretization using quantile-based binning.

    This is much faster than sklearn's KBinsDiscretizer for large datasets.
    Uses PyTorch for GPU acceleration.

    GPU-ONLY MODE: Always uses CUDA device.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, d).
    X_test : np.ndarray
        Test features, shape (n_test, d).
    n_bins : int, default=5
        Number of bins per feature.
    device : str or None
        Device to use ("cuda", "cuda:0", etc., or None for "cuda:0").
    variance_threshold : float, default=1e-8
        Features with variance below this threshold are treated as constant.

    Returns
    -------
    X_train_disc : np.ndarray
        Discretized training features, shape (n_train, d), dtype int.
    X_test_disc : np.ndarray
        Discretized test features, shape (n_test, d), dtype int.
    """
    # Determine device
    if device is None:
        device = "cuda:0"
    elif not device.startswith("cuda"):
        raise ValueError(
            f"GPU-only mode enabled. Device must be 'cuda' or 'cuda:N', got '{device}'"
        )

    dev = torch.device(device)

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train).float().to(dev)
    X_test_t = torch.from_numpy(X_test).float().to(dev)

    n_train, n_features = X_train_t.shape
    n_test = X_test_t.shape[0]

    # Identify low-variance features
    feature_vars = torch.var(X_train_t, dim=0)
    low_var_mask = feature_vars < variance_threshold

    # Initialize output tensors
    X_train_disc = torch.zeros_like(X_train_t, dtype=torch.long)
    X_test_disc = torch.zeros_like(X_test_t, dtype=torch.long)

    # For low-variance features, assign constant bin (0)
    X_train_disc[:, low_var_mask] = 0
    X_test_disc[:, low_var_mask] = 0

    # For normal features, compute quantiles and discretize
    normal_mask = ~low_var_mask
    normal_indices = torch.where(normal_mask)[0]

    if len(normal_indices) > 0:
        for feat_idx in normal_indices:
            # Get training values for this feature
            train_vals = X_train_t[:, feat_idx]

            # Compute quantile boundaries
            # Create n_bins+1 boundaries (including min and max)
            quantiles = torch.linspace(0, 1, n_bins + 1, device=dev)
            boundaries = torch.quantile(train_vals, quantiles)

            # Handle duplicate boundaries (e.g., if feature has many identical values)
            unique_boundaries = torch.unique(boundaries)
            if len(unique_boundaries) < 2:
                # Feature is essentially constant, assign bin 0
                X_train_disc[:, feat_idx] = 0
                X_test_disc[:, feat_idx] = 0
                continue

            # Discretize train values
            # Use searchsorted to assign bins
            train_bins = torch.searchsorted(unique_boundaries[:-1], train_vals, right=True)
            train_bins = torch.clamp(train_bins, 0, len(unique_boundaries) - 2)
            X_train_disc[:, feat_idx] = train_bins

            # Discretize test values using same boundaries
            test_vals = X_test_t[:, feat_idx]
            test_bins = torch.searchsorted(unique_boundaries[:-1], test_vals, right=True)
            test_bins = torch.clamp(test_bins, 0, len(unique_boundaries) - 2)
            X_test_disc[:, feat_idx] = test_bins

    # Convert back to numpy
    return X_train_disc.cpu().numpy(), X_test_disc.cpu().numpy()


def standardize_features_gpu(
    X_train: np.ndarray,
    X_test: np.ndarray,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated feature standardization (z-score normalization).

    Computes mean and std on training data, applies to both train and test.
    Much faster than sklearn's StandardScaler for large datasets.

    GPU-ONLY MODE: Always uses CUDA device.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n_train, d).
    X_test : np.ndarray
        Test features, shape (n_test, d).
    device : str or None
        Device to use ("cuda", "cuda:0", etc., or None for "cuda:0").

    Returns
    -------
    X_train_scaled : np.ndarray
        Standardized training features.
    X_test_scaled : np.ndarray
        Standardized test features.
    """
    # Determine device
    if device is None:
        device = "cuda:0"
    elif not device.startswith("cuda"):
        raise ValueError(
            f"GPU-only mode enabled. Device must be 'cuda' or 'cuda:N', got '{device}'"
        )

    dev = torch.device(device)

    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train).float().to(dev)
    X_test_t = torch.from_numpy(X_test).float().to(dev)

    # Compute mean and std on training data
    mean = X_train_t.mean(dim=0)
    std = X_train_t.std(dim=0)

    # Avoid division by zero
    std = torch.where(std > 1e-10, std, torch.ones_like(std))

    # Standardize
    X_train_scaled = (X_train_t - mean) / std
    X_test_scaled = (X_test_t - mean) / std

    # Convert back to numpy
    return X_train_scaled.cpu().numpy(), X_test_scaled.cpu().numpy()

