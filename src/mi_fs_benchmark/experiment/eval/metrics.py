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


def compute_statistical_significance(
    df_folds: "pd.DataFrame",
    metric_col: str = "roc_auc",
    mi_selectors: list[str] = ["mi", "mrmr"],
    p_value_threshold: float = 0.05,
    output_path: str | None = "results/statistical_significance.csv"
) -> "pd.DataFrame":
    """
    Compute paired t-test and Wilcoxon signed-rank test comparing EACH method
    against the best baseline for each (dataset, model, k) combination using fold-level outputs.

    Parameters
    ----------
    df_folds: pd.DataFrame
        DataFrame containing fold-level results with columns: 'dataset', 'model', 'k', 'selector', and the metric_col.
    metric_col: str
        The metric to evaluate (e.g., 'roc_auc' or 'accuracy').
    mi_selectors: list[str]
        List of selector names considered as 'MI methods'.
    p_value_threshold: float
        Threshold for significance.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the p-values and significance flags.
    """
    import pandas as pd
    from scipy.stats import ttest_rel, wilcoxon

    results = []
    
    # group by dataset, model, k
    groups = df_folds.groupby(["dataset", "model", "k"])
    for (ds, mod, k), group in groups:
        mi_data = group[group["selector"].isin(mi_selectors)]
        base_data = group[~group["selector"].isin(mi_selectors)]
        
        if base_data.empty:
            continue
            
        best_mi_sel = None
        if not mi_data.empty:
            mi_means = mi_data.groupby("selector")[metric_col].mean()
            best_mi_sel = mi_means.idxmax()

        # Find best baseline method (by mean metric)
        base_means = base_data.groupby("selector")[metric_col].mean()
        best_base_sel = base_means.idxmax()
        
        sort_cols = [c for c in ["run_id", "fold_id", "fold"] if c in group.columns]
        if not sort_cols:
            sort_cols = None

        if sort_cols:
            base_scores = base_data[base_data["selector"] == best_base_sel].sort_values(sort_cols)[metric_col].values
        else:
            base_scores = base_data[base_data["selector"] == best_base_sel][metric_col].values
        
        # Loop through ALL selectors
        for sel in group["selector"].unique():
            if sort_cols:
                sel_scores = group[group["selector"] == sel].sort_values(sort_cols)[metric_col].values
            else:
                sel_scores = group[group["selector"] == sel][metric_col].values

            # We need paired samples, so they must be equal length
            if len(sel_scores) != len(base_scores) or len(sel_scores) < 2:
                continue

            try:
                t_stat, t_p = ttest_rel(sel_scores, base_scores)
            except Exception:
                t_p = float('nan')

            try:
                w_res = wilcoxon(sel_scores, base_scores)
                w_p = float(w_res.pvalue)
            except Exception:
                w_p = float('nan')

            results.append({
                "dataset": ds,
                "model": mod,
                "k": k,
                "selector": sel,
                "best_mi": best_mi_sel,
                "best_baseline": best_base_sel,
                "t_test_p": t_p,
                "wilcoxon_p": w_p,
                "significant_ttest": t_p < p_value_threshold if not pd.isna(t_p) else False,
                "significant_wilcoxon": w_p < p_value_threshold if not pd.isna(w_p) else False
            })

    res_df = pd.DataFrame(results)
    if output_path is not None and not res_df.empty:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index=False)
    return res_df
