"""
GPU-accelerated mutual information estimation.

This module provides GPU acceleration for MI calculations using PyTorch.
GPU-ONLY MODE: No CPU fallbacks - requires CUDA-enabled GPU.

GPU Utilization:
- Uses all available CUDA cores automatically via PyTorch
- Operations are parallelized across GPU threads
- Batch processing for memory efficiency
"""

from __future__ import annotations

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

    # Optimize PyTorch for maximum GPU utilization
    # Enable TF32 for faster matrix operations on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set benchmarking mode for optimal kernel selection
    torch.backends.cudnn.benchmark = True
except ImportError as e:
    raise ImportError(
        "GPU-only mode enabled but PyTorch is not installed. "
        "Install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    ) from e


def get_device(device: str | None = None) -> str:
    """
    Get the device to use for computation.

    GPU-ONLY MODE: Always returns a CUDA device.

    Parameters
    ----------
    device : str or None
        Requested device ("cuda", "cuda:0", "cuda:1", etc., or None for "cuda:0").
        If None, defaults to "cuda:0".

    Returns
    -------
    str
        The device string to use (always a cuda device).
    """
    if device is None:
        return "cuda:0"

    if not device.startswith("cuda"):
        raise ValueError(
            f"GPU-only mode enabled. Device must be 'cuda' or 'cuda:N', got '{device}'"
        )

    return device



def get_gpu_info() -> dict:
    """
    Get GPU information including core count and utilization.

    Returns
    -------
    dict
        GPU information including:
        - device_name: GPU model name
        - cuda_cores: Approximate CUDA core count (if known)
        - memory_total: Total GPU memory in GB
        - memory_available: Available GPU memory in GB
        - compute_capability: CUDA compute capability
    """
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    info = {
        "available": True,
        "device_name": props.name,
        "memory_total": props.total_memory / 1024**3,  # Convert to GB
        "memory_available": (props.total_memory - torch.cuda.memory_allocated()) / 1024**3,
        "compute_capability": f"{props.major}.{props.minor}",
    }

    # Estimate CUDA cores based on architecture
    # Note: This is an approximation based on common GPU architectures
    sm_count = props.multi_processor_count
    cores_per_sm = {
        (7, 5): 64,  # Turing (RTX 20xx)
        (8, 6): 128, # Ampere (RTX 30xx)
        (8, 9): 128, # Ada Lovelace (RTX 40xx)
    }
    cores = cores_per_sm.get((props.major, props.minor), 64) * sm_count
    info["cuda_cores"] = cores


    return info


def compute_mi_gpu_correlation(
    X: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fast GPU-accelerated MI approximation using correlation.

    This is an approximation: MI ≈ -0.5 * log(1 - ρ²) for Gaussian variables.
    For non-Gaussian data, this provides a fast proxy that preserves ranking.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Target vector, shape (n_samples,).
    device : str or None
        Device to use ("cuda", "cuda:0", etc., or None for "cuda:0").
    verbose : bool, default=False
        If True, print GPU usage information.

    Returns
    -------
    mi_scores : np.ndarray
        MI approximation scores for each feature, shape (n_features,).
    """
    dev = get_device(device)

    if verbose:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🎮 Computing MI on GPU: {dev} ({X.shape[1]} features, {X.shape[0]} samples)")

    # Convert to torch tensors and move to device
    X_tensor = torch.from_numpy(X).float().to(dev)
    y_tensor = torch.from_numpy(y).float().to(dev)

    if verbose:
        mem_used = torch.cuda.memory_allocated() / 1024**2
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"   GPU Memory: {mem_used:.1f} MB used")

    # Convert to torch tensors and move to device
    X_tensor = torch.from_numpy(X).float().to(dev)
    y_tensor = torch.from_numpy(y).float().to(dev)

    # Standardize
    X_std = (X_tensor - X_tensor.mean(dim=0)) / (X_tensor.std(dim=0) + 1e-10)
    y_std = (y_tensor - y_tensor.mean()) / (y_tensor.std() + 1e-10)

    # Compute correlations
    n = X_tensor.shape[0]
    correlations = (X_std.T @ y_std) / (n - 1)

    # MI approximation: -0.5 * log(1 - ρ²)
    # Clamp to avoid log(0)
    rho_squared = torch.clamp(correlations ** 2, 0, 0.9999)
    mi_approx = -0.5 * torch.log(1 - rho_squared)

    # Replace NaN/inf with 0
    mi_approx = torch.nan_to_num(mi_approx, nan=0.0, posinf=0.0, neginf=0.0)

    # Move back to CPU and convert to numpy
    return mi_approx.cpu().numpy()


def compute_pairwise_mi_gpu(
    X: np.ndarray,
    device: str | None = None,
    batch_size: int = 500,  # Increased from 100 for better GPU utilization
    verbose: bool = False,
) -> np.ndarray:
    """
    Fast GPU-accelerated pairwise MI matrix using correlation approximation.

    Computes I(X_i; X_j) for all feature pairs. Uses batching to handle
    large feature sets without exceeding GPU memory.

    GPU Utilization: This function fully utilizes all CUDA cores by:
    - Processing multiple features in parallel within each batch
    - Using optimized matrix operations (GEMM) that saturate GPU
    - Automatic parallelization across CUDA cores by PyTorch

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    device : str or None
        Device to use ("cuda", "cuda:0", etc., or None for "cuda:0").
    batch_size : int, default=500
        Number of features to process at once. Larger = more GPU utilization
        but more memory. Adjust based on your GPU memory (6GB can handle 500-1000).
    verbose : bool, default=False
        If True, print GPU usage information.

    Returns
    -------
    mi_matrix : np.ndarray
        Pairwise MI approximation matrix, shape (n_features, n_features).
    """
    dev = get_device(device)
    n_samples, n_features = X.shape

    if verbose:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🎮 Computing pairwise MI on GPU: {dev} ({n_features}×{n_features} matrix)")
        logger.info(f"   Batch size: {batch_size} features/batch")

    # Convert to torch and standardize
    X_tensor = torch.from_numpy(X).float().to(dev)
    X_std = (X_tensor - X_tensor.mean(dim=0)) / (X_tensor.std(dim=0) + 1e-10)

    # Compute correlation matrix in batches
    # Using GPU, this saturates all CUDA cores automatically
    mi_matrix = torch.zeros((n_features, n_features), device=dev)

    for i in range(0, n_features, batch_size):
        end_i = min(i + batch_size, n_features)
        batch = X_std[:, i:end_i]

        # Correlations between batch and all features
        # This @ operation uses cuBLAS which utilizes all GPU cores
        correlations = (batch.T @ X_std) / (n_samples - 1)

        # MI approximation (element-wise ops, also parallelized)
        rho_squared = torch.clamp(correlations ** 2, 0, 0.9999)
        mi_batch = -0.5 * torch.log(1 - rho_squared)
        mi_batch = torch.nan_to_num(mi_batch, nan=0.0, posinf=0.0, neginf=0.0)

        mi_matrix[i:end_i, :] = mi_batch

    # Ensure symmetry
    mi_matrix = (mi_matrix + mi_matrix.T) / 2

    return mi_matrix.cpu().numpy()


def compute_mi_gpu_kde(
    X: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
    n_bins: int = 50,
) -> np.ndarray:
    """
    GPU-accelerated MI estimation using histogram binning.

    More accurate than correlation approximation but slower.
    Good compromise between accuracy and speed.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Target vector, shape (n_samples,).
    device : str or None
        Device to use ("cuda", "cpu", or None for auto).
    n_bins : int
        Number of bins for histogram estimation.

    Returns
    -------
    mi_scores : np.ndarray
        MI scores for each feature, shape (n_features,).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Install with: pip install torch")

    dev = get_device(device)
    n_samples, n_features = X.shape

    # Convert to torch tensors
    X_tensor = torch.from_numpy(X).float().to(dev)
    y_tensor = torch.from_numpy(y).float().to(dev)

    # Discretize y into bins
    y_min, y_max = y_tensor.min(), y_tensor.max()
    y_bins = torch.floor((y_tensor - y_min) / (y_max - y_min + 1e-10) * (n_bins - 1)).long()
    y_bins = torch.clamp(y_bins, 0, n_bins - 1)

    mi_scores = torch.zeros(n_features, device=dev)

    for j in range(n_features):
        x_j = X_tensor[:, j]

        # Discretize feature
        x_min, x_max = x_j.min(), x_j.max()
        if x_max - x_min < 1e-10:
            continue
        x_bins = torch.floor((x_j - x_min) / (x_max - x_min + 1e-10) * (n_bins - 1)).long()
        x_bins = torch.clamp(x_bins, 0, n_bins - 1)

        # Compute joint histogram
        joint_hist = torch.zeros((n_bins, n_bins), device=dev)
        for i in range(n_samples):
            joint_hist[x_bins[i], y_bins[i]] += 1

        joint_prob = joint_hist / n_samples

        # Marginal distributions
        x_prob = joint_prob.sum(dim=1)
        y_prob = joint_prob.sum(dim=0)

        # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0.0
        for xi in range(n_bins):
            for yi in range(n_bins):
                pxy = joint_prob[xi, yi]
                if pxy > 1e-10:
                    px = x_prob[xi]
                    py = y_prob[yi]
                    if px > 1e-10 and py > 1e-10:
                        mi += pxy * torch.log(pxy / (px * py))

        mi_scores[j] = mi

    return mi_scores.cpu().numpy()

