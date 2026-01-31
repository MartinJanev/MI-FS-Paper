# src/mi_fs_benchmark/datasets/ieee_cis_fraud.py
from __future__ import annotations

from .base import BaseDataset


class IEEECISFraudDataset(BaseDataset):
    """
    Dataset loader for the IEEE CIS Fraud Detection dataset.

    Loads from fold artifacts (.npz files) in the processed directory.
    The base class handles all loading logic.
    """
    pass
