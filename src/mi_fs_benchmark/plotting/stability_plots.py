from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_stability_vs_k(
    df: pd.DataFrame,
    out_path: str,
    summaries_dir: str | None = None,
) -> None:
    """Plot stability (Jaccard index) vs k for different selectors.

    Args:
        df: DataFrame with columns [selector, model, k, stability_jaccard]
        out_path: Path to save the plot
        summaries_dir: Optional path to summaries folder for experiment descriptions
    """
    plt.figure()
    for selector, group in df.groupby("selector"):
        ks = group["k"]
        stability = group["stability_jaccard"]
        plt.plot(ks, stability, marker="o", label=selector)
    plt.xlabel("k (selected features)")
    plt.ylabel("Mean pairwise Jaccard")
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
