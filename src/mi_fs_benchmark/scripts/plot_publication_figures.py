#!/usr/bin/env python3
# scripts/plot_publication_figures.py
"""
Generate publication-ready figures from aggregated results.

This script reads ONLY aggregated summary files (never raw results)
and produces clean, publication-quality figures with error bars.

Usage:
    python plot_publication_figures.py --dataset santander_short

Input:
    - results/aggregated/{dataset}_summary.csv
    - results/aggregated/{dataset}_stability_summary.csv (optional)

Output:
    - plots/publication/{dataset}_performance.png
    - plots/publication/{dataset}_stability.png (if applicable)
    - plots/publication/{dataset}_model_comparison.png
"""

from __future__ import annotations

import sys
from pathlib import Path
# For Google Colab compatibility
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mi_fs_benchmark.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate publication figures from aggregated results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., santander_short)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory (default: results/aggregated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: plots/publication)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy", "roc_auc", "f1"],
        help="Metrics to plot",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 4],
        help="Figure size (width height)",
    )
    return parser.parse_args()


def load_aggregated_results(input_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load aggregated summary files.

    Parameters
    ----------
    input_dir : Path
        Directory containing aggregated CSV files
    dataset : str
        Dataset name

    Returns
    -------
    summary_df : DataFrame
        Performance metrics summary
    stability_df : DataFrame or None
        Stability metrics summary (if exists)
    """
    summary_path = input_dir / f"{dataset}_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            f"Run aggregation first: python aggregate_results.py --dataset {dataset}"
        )

    summary_df = pd.read_csv(summary_path)
    logger.info(f"Loaded performance summary: {len(summary_df)} rows")

    # Try to load stability summary
    stability_path = input_dir / f"{dataset}_stability_summary.csv"
    stability_df = None

    if stability_path.exists():
        stability_df = pd.read_csv(stability_path)
        logger.info(f"Loaded stability summary: {len(stability_df)} rows")
    else:
        logger.info("No stability summary found (optional)")

    return summary_df, stability_df


def plot_performance_comparison(
    df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    figsize: tuple[float, float] = (12, 4),
):
    """
    Plot performance metrics across selectors with error bars.

    Creates a multi-panel figure with one subplot per metric.
    Each subplot shows selector performance vs. k with 95% CI error bars.

    Parameters
    ----------
    df : DataFrame
        Aggregated summary with {metric}_mean and {metric}_ci_* columns
    metrics : list of str
        Metrics to plot (e.g., ['accuracy', 'roc_auc', 'f1'])
    output_path : Path
        Where to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    # Filter to only metrics that exist in the data
    available_metrics = [m for m in metrics if f"{m}_mean" in df.columns]

    if not available_metrics:
        logger.warning(f"None of the requested metrics {metrics} found in data")
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Get unique selectors and assign colors
    selectors = sorted(df["selector"].unique())
    colors = sns.color_palette("husl", len(selectors))
    selector_colors = dict(zip(selectors, colors))

    # For each metric, plot selector curves
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        for selector in selectors:
            selector_data = df[df["selector"] == selector].sort_values("k")

            k_values = selector_data["k"].values
            means = selector_data[f"{metric}_mean"].values

            # Use CI if available, otherwise use SEM
            if f"{metric}_ci_lower" in selector_data.columns:
                ci_lower = selector_data[f"{metric}_ci_lower"].values
                ci_upper = selector_data[f"{metric}_ci_upper"].values
                errors_lower = means - ci_lower
                errors_upper = ci_upper - means
                errors = np.array([errors_lower, errors_upper])
            elif f"{metric}_sem" in selector_data.columns:
                sem = selector_data[f"{metric}_sem"].values
                errors = sem
            else:
                errors = None

            # Plot with error bars
            ax.errorbar(
                k_values,
                means,
                yerr=errors,
                label=selector,
                color=selector_colors[selector],
                marker="o",
                capsize=3,
                capthick=1.5,
            )

        # Formatting
        ax.set_xlabel("Number of Features (k)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} vs. k")
        ax.legend(frameon=True, loc="best")
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved performance comparison: {output_path.name}")
    plt.close(fig)


def plot_model_comparison(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
):
    """
    Plot model comparison for a single metric.

    Shows how different models perform with the same selector and k.

    Parameters
    ----------
    df : DataFrame
        Aggregated summary
    metric : str
        Metric to plot (e.g., 'roc_auc')
    output_path : Path
        Where to save the figure
    figsize : tuple
        Figure size
    """
    if f"{metric}_mean" not in df.columns:
        logger.warning(f"Metric {metric} not found in data")
        return

    models = sorted(df["model"].unique())
    selectors = sorted(df["selector"].unique())

    n_selectors = len(selectors)
    fig, axes = plt.subplots(1, n_selectors, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    colors = sns.color_palette("Set2", len(models))
    model_colors = dict(zip(models, colors))

    for idx, selector in enumerate(selectors):
        ax = axes[idx]
        selector_data = df[df["selector"] == selector]

        for model in models:
            model_data = selector_data[selector_data["model"] == model].sort_values("k")

            if len(model_data) == 0:
                continue

            k_values = model_data["k"].values
            means = model_data[f"{metric}_mean"].values

            # Error bars
            if f"{metric}_ci_lower" in model_data.columns:
                ci_lower = model_data[f"{metric}_ci_lower"].values
                ci_upper = model_data[f"{metric}_ci_upper"].values
                errors_lower = means - ci_lower
                errors_upper = ci_upper - means
                errors = np.array([errors_lower, errors_upper])
            else:
                errors = None

            ax.errorbar(
                k_values,
                means,
                yerr=errors,
                label=model,
                color=model_colors[model],
                marker="s",
                capsize=3,
            )

        ax.set_xlabel("Number of Features (k)")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{selector}")
        ax.legend(frameon=True)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle(f"Model Comparison: {metric.replace('_', ' ').title()}", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved model comparison: {output_path.name}")
    plt.close(fig)


def plot_stability(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float] = (8, 6),
):
    """
    Plot stability metrics across selectors.

    Parameters
    ----------
    df : DataFrame
        Stability summary
    output_path : Path
        Where to save the figure
    figsize : tuple
        Figure size
    """
    if df is None or df.empty:
        logger.info("No stability data to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    selectors = sorted(df["selector"].unique())
    colors = sns.color_palette("husl", len(selectors))
    selector_colors = dict(zip(selectors, colors))

    for selector in selectors:
        selector_data = df[df["selector"] == selector].sort_values("k")

        k_values = selector_data["k"].values
        means = selector_data["stability_mean"].values

        # Error bars
        if "stability_ci_lower" in selector_data.columns:
            ci_lower = selector_data["stability_ci_lower"].values
            ci_upper = selector_data["stability_ci_upper"].values
            errors_lower = means - ci_lower
            errors_upper = ci_upper - means
            errors = np.array([errors_lower, errors_upper])
        else:
            errors = None

        ax.errorbar(
            k_values,
            means,
            yerr=errors,
            label=selector,
            color=selector_colors[selector],
            marker="o",
            capsize=3,
        )

    ax.set_xlabel("Number of Features (k)")
    ax.set_ylabel("Stability (Jaccard Index)")
    ax.set_title("Feature Selection Stability vs. k")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved stability plot: {output_path.name}")
    plt.close(fig)


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))

    setup_logging()

    logger.info("=" * 80)
    logger.info("Publication Figure Generation")
    logger.info("=" * 80)

    # Determine directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = repo_root / "src" / "results" / "aggregated"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "plots" / "publication"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info("")

    # Load aggregated results
    try:
        summary_df, stability_df = load_aggregated_results(input_dir, args.dataset)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Generate figures
    logger.info("Generating figures...")
    logger.info("")

    # 1. Performance comparison across selectors
    perf_path = output_dir / f"{args.dataset}_performance.png"
    plot_performance_comparison(
        summary_df,
        args.metrics,
        perf_path,
        figsize=tuple(args.figsize),
    )

    # 2. Model comparison (use first metric)
    if args.metrics:
        model_path = output_dir / f"{args.dataset}_model_comparison.png"
        plot_model_comparison(summary_df, args.metrics[0], model_path)

    # 3. Stability plot (if available)
    if stability_df is not None and not stability_df.empty:
        stability_path = output_dir / f"{args.dataset}_stability.png"
        plot_stability(stability_df, stability_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Figure generation complete!")
    logger.info("=" * 80)
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("")


if __name__ == "__main__":
    main()

