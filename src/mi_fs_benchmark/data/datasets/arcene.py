# src/mi_fs_benchmark/datasets/arcene.py
from __future__ import annotations

from .base import BaseDataset


class ArceneDataset(BaseDataset):
    """
    Dataset loader for the Arcene dataset.

    Loads from fold artifacts (.npz files) in the processed directory.
    The base class handles all loading logic.
    """
    pass
