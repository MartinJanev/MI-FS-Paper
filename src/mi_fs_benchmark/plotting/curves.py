from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_vs_k(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    summaries_dir: str | None = None,
) -> None:
    """
    Plot a metric (e.g. ROC-AUC) vs k for each selector.

    Expects df columns: selector, model, k, mean.
    """
    plt.figure()
    for selector, group in df.groupby("selector"):
        ks = group["k"]
        means = group["mean"]
        plt.plot(ks, means, marker="o", label=selector)
    plt.xlabel("k (selected features)")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)


def plot_all_metrics_combined(
    df: pd.DataFrame,
    metrics: list[str],
    out_path: str,
    dataset_name: str = "",
) -> None:
    """
    Plot all metrics on a single plot with different colors for each metric.

    Parameters
    ----------
    df:
        DataFrame with columns: selector, model, k, and metric columns
    metrics:
        List of metric names to plot
    out_path:
        Path to save the figure
    dataset_name:
        Optional dataset name for the title
    """
    if len(metrics) == 0:
        return

    # Define color palette for metrics
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(metrics))]

    fig, ax = plt.subplots(figsize=(10, 6))

    # For each selector, plot all metrics
    selectors = df["selector"].unique()

    # Define line styles for different selectors
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    for selector_idx, selector in enumerate(selectors):
        selector_df = df[df["selector"] == selector]
        line_style = line_styles[selector_idx % len(line_styles)]
        marker = markers[selector_idx % len(markers)]

        for metric_idx, metric in enumerate(metrics):
            grouped = selector_df.groupby("k")[metric].mean().reset_index()

            # Create label combining selector and metric
            label = f"{selector} - {metric.replace('_', ' ').upper()}"

            ax.plot(
                grouped["k"],
                grouped[metric],
                marker=marker,
                linestyle=line_style,
                color=colors[metric_idx],
                label=label,
                linewidth=2,
                markersize=6,
                alpha=0.8
            )

    ax.set_xlabel("k (selected features)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    if dataset_name:
        ax.set_title(f"{dataset_name} - All Metrics Combined", fontsize=14, fontweight='bold')

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


