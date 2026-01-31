from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_efficiency_frontier(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    time_col: str = "fit_time",
    out_path: str | None = None,
) -> None:
    """
    Plot metric vs runtime (efficiency frontier) for different selectors.

    Expects df columns: selector, metric, time_col.
    """
    plt.figure()
    for selector, group in df.groupby("selector"):
        plt.scatter(group[time_col], group[metric], label=selector)
    plt.xlabel("Runtime (s)")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)

