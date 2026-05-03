# src/mi_fs_benchmark/cv/artifacts.py
"""
Fold artifact definitions and utilities.

A FoldArtifact is the canonical, immutable representation of a single
cross-validation fold. It contains all preprocessed data needed for
feature selection and model training.

Key principles:
- Selectors only see X_train_disc, y_train, and optional mi_lookup
- Models only see selected columns from train/test
- No downstream component touches raw DataFrames or storage format
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FoldArtifact:
    """
    Immutable fold artifact containing all preprocessed data for one CV fold.

    Attributes
    ----------
    fold_id : int
        Fold index (0-based).
    X_train_disc : np.ndarray
        Training features, discretized for MI estimation. Shape (n_train, d).
    X_test_disc : np.ndarray
        Test features, discretized using training bins. Shape (n_test, d).
    y_train : np.ndarray
        Training labels. Shape (n_train,).
    y_test : np.ndarray
        Test labels. Shape (n_test,).
    mi_lookup : Optional[np.ndarray]
        Precomputed MI scores I(X_j; Y) for each feature. Shape (d,).
        Used by MI-based selectors to avoid recomputing.
    feature_names : list[str]
        Feature names corresponding to columns of X.
    """

    fold_id: int
    X_train_disc: np.ndarray
    X_test_disc: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    mi_lookup: Optional[np.ndarray]
    feature_names: list[str]

    def __post_init__(self):
        """Validate artifact consistency."""
        n_train = self.X_train_disc.shape[0]
        n_test = self.X_test_disc.shape[0]
        d_train = self.X_train_disc.shape[1]
        d_test = self.X_test_disc.shape[1]

        if self.y_train.shape[0] != n_train:
            raise ValueError(f"y_train length {self.y_train.shape[0]} != n_train {n_train}")
        if self.y_test.shape[0] != n_test:
            raise ValueError(f"y_test length {self.y_test.shape[0]} != n_test {n_test}")
        if d_train != d_test:
            raise ValueError(f"Train features {d_train} != test features {d_test}")
        if len(self.feature_names) != d_train:
            raise ValueError(f"Feature names length {len(self.feature_names)} != d {d_train}")
        if self.mi_lookup is not None and self.mi_lookup.shape[0] != d_train:
            raise ValueError(f"MI lookup length {self.mi_lookup.shape[0]} != d {d_train}")

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X_train_disc.shape[1]

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return self.X_train_disc.shape[0]

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return self.X_test_disc.shape[0]

    def save(self, path: Path) -> None:
        """
        Save fold artifact to disk as .npz file.

        Parameters
        ----------
        path : Path
            Output path (e.g., data/processed/dataset/fold_0.npz).
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "fold_id": np.array(self.fold_id),
            "X_train_disc": self.X_train_disc,
            "X_test_disc": self.X_test_disc,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "feature_names": np.array(self.feature_names, dtype=object),
        }

        if self.mi_lookup is not None:
            save_dict["mi_lookup"] = self.mi_lookup

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load(cls, path: Path) -> FoldArtifact:
        """
        Load fold artifact from disk.

        Parameters
        ----------
        path : Path
            Path to .npz file.

        Returns
        -------
        FoldArtifact
        """
        data = np.load(path, allow_pickle=True)

        return cls(
            fold_id=int(data["fold_id"]),
            X_train_disc=data["X_train_disc"],
            X_test_disc=data["X_test_disc"],
            y_train=data["y_train"],
            y_test=data["y_test"],
            mi_lookup=data.get("mi_lookup", None),
            feature_names=list(data["feature_names"]),
        )


def load_all_folds(dataset_dir: Path) -> list[FoldArtifact]:
    """
    Load all fold artifacts from a dataset directory.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing fold_0.npz, fold_1.npz, etc.

    Returns
    -------
    list[FoldArtifact]
        List of fold artifacts, sorted by fold_id.
    """
    fold_paths = sorted(dataset_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {dataset_dir}")

    folds = [FoldArtifact.load(p) for p in fold_paths]
    folds.sort(key=lambda f: f.fold_id)

    return folds


def generate_fold_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    seed: int = 42,
    n_bins: int = 10,
    n_neighbors: int = 3,
) -> list[FoldArtifact]:
    """
    Generate fold artifacts on-the-fly from data arrays.

    This is useful for augmented/synthetic datasets where we don't have
    pre-saved fold artifacts.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Target labels, shape (n_samples,).
    feature_names : list[str]
        Feature names.
    n_splits : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random seed for fold splitting.
    n_bins : int, default=10
        Number of bins for discretization.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.

    Returns
    -------
    list[FoldArtifact]
        Generated fold artifacts.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from mi_fs_benchmark.data_processing.preprocessing import compute_mi_lookup, discretize_features

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    artifacts = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Discretize for MI estimation
        X_train_disc, X_test_disc = discretize_features(
            X_train_scaled,
            X_test_scaled,
            n_bins=n_bins,
            strategy="quantile",
        )

        # Precompute MI lookup table
        mi_lookup = compute_mi_lookup(
            X_train_disc,
            y_train_fold,
            n_neighbors=n_neighbors,
            random_state=seed,
            n_jobs=-1,
        )

        # Create fold artifact
        artifact = FoldArtifact(
            fold_id=fold_id,
            X_train_disc=X_train_disc,
            X_test_disc=X_test_disc,
            y_train=y_train_fold,
            y_test=y_test_fold,
            mi_lookup=mi_lookup,
            feature_names=feature_names,
        )

        artifacts.append(artifact)

    return artifacts


def load_metadata_only(dataset_dir: Path) -> dict:
    """
    Load only metadata from first fold artifact (fast, no heavy arrays).

    Parameters
    ----------
    dataset_dir : Path
        Directory containing fold_*.npz files.

    Returns
    -------
    dict
        Metadata including feature_names, n_features, n_samples.
    """
    fold_paths = sorted(dataset_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {dataset_dir}")

    # Load only specific arrays from first fold (without loading all data)
    with np.load(fold_paths[0], allow_pickle=True) as data:
        feature_names = list(data["feature_names"])
        n_features = len(feature_names)

        # Get sample counts from all folds
        n_samples = 0
        for fold_path in fold_paths:
            with np.load(fold_path) as fold_data:
                n_samples += len(fold_data["y_train"]) + len(fold_data["y_test"])

        return {
            "feature_names": feature_names,
            "n_features": n_features,
            "n_samples": n_samples,
            "n_folds": len(fold_paths),
        }


def get_fold_indices(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Pre-compute fold indices for reuse across multiple experiments.

    This allows you to generate the same folds for different feature sets
    without recomputing the splitting logic.

    Parameters
    ----------
    y : np.ndarray
        Target labels for stratified splitting.
    n_splits : int
        Number of folds.
    seed : int
        Random seed.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples.
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def generate_fold_artifacts_fast(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    seed: int = 42,
    n_bins: int = 10,
    n_neighbors: int = 3,
    use_gpu: bool = False,
    device: str | None = None,
) -> list[FoldArtifact]:
    """
    Generate fold artifacts using pre-computed fold indices (OPTIMIZED).

    This is much faster than generate_fold_artifacts() when you need to
    generate folds for multiple feature sets with the same sample split.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Target labels, shape (n_samples,).
    feature_names : list[str]
        Feature names.
    fold_indices : list[tuple[np.ndarray, np.ndarray]]
        Pre-computed fold indices from get_fold_indices().
    seed : int, default=42
        Random seed for MI estimation.
    n_bins : int, default=10
        Number of bins for discretization.
    n_neighbors : int, default=3
        Number of neighbors for MI estimation.
    use_gpu : bool, default=False
        Whether to use GPU acceleration for preprocessing and MI computation.
    device : str or None, default=None
        Device for GPU computation ("cuda", "cuda:0", "cpu", or None for auto).

    Returns
    -------
    list[FoldArtifact]
        Generated fold artifacts.
    """
    from sklearn.preprocessing import StandardScaler
    from mi_fs_benchmark.data_processing.preprocessing import compute_mi_lookup, discretize_features

    artifacts = []
    for fold_id, (train_idx, test_idx) in enumerate(fold_indices):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]

        # Standardize and discretize features (GPU-accelerated if enabled)
        if use_gpu:
            from mi_fs_benchmark.data_processing.preprocessing.discretize_gpu import (
                standardize_features_gpu,
                discretize_features_gpu
            )
            # GPU-accelerated preprocessing
            X_train_scaled, X_test_scaled = standardize_features_gpu(
                X_train_raw, X_test_raw, device=device
            )
            X_train_disc, X_test_disc = discretize_features_gpu(
                X_train_scaled, X_test_scaled, n_bins=n_bins, device=device
            )
        else:
            # CPU preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            X_train_disc, X_test_disc = discretize_features(
                X_train_scaled, X_test_scaled, n_bins=n_bins, strategy="quantile"
            )

        # Precompute MI lookup table (GPU-accelerated if enabled)
        mi_lookup = compute_mi_lookup(
            X_train_disc,
            y_train_fold,
            n_neighbors=n_neighbors,
            random_state=seed,
            n_jobs=-1,
            device=device,
            use_gpu_approximation=use_gpu,
        )

        # Create fold artifact
        artifact = FoldArtifact(
            fold_id=fold_id,
            X_train_disc=X_train_disc,
            X_test_disc=X_test_disc,
            y_train=y_train_fold,
            y_test=y_test_fold,
            mi_lookup=mi_lookup,
            feature_names=feature_names,
        )

        artifacts.append(artifact)

    return artifacts




