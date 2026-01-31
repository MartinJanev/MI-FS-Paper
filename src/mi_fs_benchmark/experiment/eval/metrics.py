from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_all(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard metrics for binary classification.

    Returns
    -------
    dict
        accuracy, roc_auc, pr_auc, f1, log_loss
    """
    metrics: dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    metrics["f1"] = float(f1_score(y_true, y_pred))
    metrics["log_loss"] = float(log_loss(y_true, y_proba))

    return metrics
