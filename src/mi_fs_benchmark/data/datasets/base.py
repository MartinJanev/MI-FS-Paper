from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class DatasetSplit:
    X: pd.DataFrame
    y: pd.Series


class BaseDataset(ABC):
    """
    Base dataset interface that loads from fold artifacts (.npz files).

    This class provides a common implementation for loading the full dataset
    by reconstructing it from fold artifacts, which are the canonical format
    after preprocessing.
    """

    def __init__(self, root: Path, target_column: str):
        self.root = root
        self.target_column = target_column

    def load_full(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load the entire dataset by reconstructing from fold artifacts.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y) for the complete dataset.
        """
        from mi_fs_benchmark.experiment.cv.artifacts import load_all_folds

        # Load all fold artifacts
        folds = load_all_folds(self.root)

        if not folds:
            raise FileNotFoundError(
                f"No fold artifacts found in {self.root}. "
                f"Please run the preparation script for this dataset first."
            )

        # Get feature names from first fold
        feature_names = folds[0].feature_names

        # Reconstruct full dataset from all folds
        # Each fold has train + test samples; we combine all of them
        all_X = []
        all_y = []

        for fold in folds:
            # Combine train and test from this fold
            all_X.append(fold.X_train_disc)
            all_X.append(fold.X_test_disc)
            all_y.append(fold.y_train)
            all_y.append(fold.y_test)

        # Concatenate all arrays
        X_full = np.vstack(all_X)
        y_full = np.concatenate(all_y)

        # Convert to DataFrame
        X_df = pd.DataFrame(X_full, columns=feature_names)
        y_series = pd.Series(y_full, name=self.target_column)

        return X_df, y_series

    def get_splits(self, n_splits: int, seed: int):
        """
        Return cross-validation splits from fold artifacts.

        Note: This loads pre-generated folds from artifacts, ignoring n_splits and seed.
        """
        from mi_fs_benchmark.experiment.cv.artifacts import load_all_folds

        folds = load_all_folds(self.root)

        splits = []
        for fold in folds:
            # Get feature names from fold
            feature_names = fold.feature_names

            train_split = DatasetSplit(
                X=pd.DataFrame(fold.X_train_disc, columns=feature_names),
                y=pd.Series(fold.y_train, name=self.target_column)
            )
            test_split = DatasetSplit(
                X=pd.DataFrame(fold.X_test_disc, columns=feature_names),
                y=pd.Series(fold.y_test, name=self.target_column)
            )
            splits.append((train_split, test_split))

        return splits
