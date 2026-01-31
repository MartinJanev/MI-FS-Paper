from __future__ import annotations

from mi_fs_benchmark.data.datasets.base import BaseDataset


def get_dataset_splits(
    dataset: BaseDataset,
    fold_scheme: str,
    n_splits: int,
    seed: int,
):
    """
    Adapter for different split schemes.

    For now, defers to dataset.get_splits; you can later add
    special handling for multilabel or time-aware splits based
    on `fold_scheme`.
    """
    return dataset.get_splits(n_splits=n_splits, seed=seed)
