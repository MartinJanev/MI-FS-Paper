from __future__ import annotations

from .base import BaseDataset


class SantanderDataset(BaseDataset):
    """
    Santander Customer Transaction Prediction (Kaggle).

    Loads from fold artifacts (.npz files) in the processed directory.
    The base class handles all loading logic.

    Note: ID columns (like 'ID_code') should be removed during preprocessing,
    not at load time.
    """
    pass

