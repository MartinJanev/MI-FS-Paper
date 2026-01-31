# src/mi_fs_benchmark/datasets/__init__.py
from __future__ import annotations

from pathlib import Path

from .base import BaseDataset
from .home_credit import HomeCreditDataset
from .nfl_bdb_2026 import NFLBDB2026Dataset
from .santander import SantanderDataset
from .arcene import ArceneDataset
from .ieee_cis_fraud import IEEECISFraudDataset


def create_dataset(name: str, root: str | Path | None, target_column: str) -> BaseDataset:
    """
    Factory for dataset objects.

    Parameters
    ----------
    name:
        Dataset identifier: 'santander', 'home_credit', 'nfl_bdb_2026'.
    root:
        Root directory containing processed Parquet files. If None, defaults to "data/processed/{name}".
    target_column:
        Name of the target column.

    Returns
    -------
    BaseDataset
    """
    if root is None:
        root = Path("data/processed") / name
    else:
        root = Path(root)
    mapping: dict[str, type[BaseDataset]] = {
        "santander": SantanderDataset,
        "home_credit": HomeCreditDataset,
        "nfl_bdb_2026": NFLBDB2026Dataset,
        "arcene": ArceneDataset,
        "ieee_cis_fraud": IEEECISFraudDataset,
    }
    try:
        cls = mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name!r}") from exc
    return cls(root=root, target_column=target_column)
