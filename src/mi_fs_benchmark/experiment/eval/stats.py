# src/mi_fs_benchmark/eval/stats.py
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def flatten_cv_results(results: Iterable) -> pd.DataFrame:
    """
    Convert a list of CVResult objects into a flat DataFrame
    where each row corresponds to a single fold score.

    Now includes run_id for multi-run experiment tracking.
    """
    rows = []
    for res in results:
        for fs in res.fold_scores:
            row = {
                "selector": res.selector_name,
                "model": res.model_name,
                "k": res.k,
                "run_id": getattr(res, 'run_id', 0),  # Backward compatible
            }
            row.update(fs)  # add metrics (accuracy, roc_auc, f1, etc.)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """
    Produce mean ± 95% CI for each selector × model × k.

    Parameters
    ----------
    df : DataFrame from flatten_cv_results
    metric : str
        Column name ('accuracy', 'roc_auc', 'f1', etc.)

    Returns
    -------
    DataFrame with columns:
        selector, model, k, mean, ci95
    """
    grouped = df.groupby(["selector", "model", "k"])[metric]

    summary = grouped.agg(["mean", "count", "std"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])

    return summary[["selector", "model", "k", "mean", "ci95"]]
