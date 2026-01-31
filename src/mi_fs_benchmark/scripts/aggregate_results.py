#!/usr/bin/env python3
# scripts/aggregate_results.py
"""
Aggregate experimental results across multiple runs.

This script takes raw per-run CSV files and produces aggregated statistics
(mean, std, SEM, 95% CI) suitable for publication figures and tables.

Usage:
    python aggregate_results.py --dataset santander_short

Input:
    - results/raw/{dataset}_seed*.csv (multiple files from different runs)

Output:
    - results/aggregated/{dataset}_summary.csv
    - results/aggregated/{dataset}_stability_summary.csv (if stability metrics exist)
"""

from __future__ import annotations

import sys
from pathlib import Path
# For Google Colab compatibility
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import argparse
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from mi_fs_benchmark.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate multi-run experimental results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., santander_short, arcene_quick)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory containing raw results (default: results/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: results/aggregated)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum number of runs required for aggregation",
    )
    return parser.parse_args()


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Compute confidence interval for a sample.

    Parameters
    ----------
    values : array-like
        Sample values
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns
    -------
    ci_lower, ci_upper : tuple of float
        Lower and upper bounds of confidence interval
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    sem = stats.sem(values)

    # Guard against degenerate or non-finite SEM to avoid SciPy warnings
    if not np.isfinite(sem) or sem == 0:
        return np.nan, np.nan

    # Use t-distribution for small samples
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    return ci[0], ci[1]


def aggregate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate performance metrics across runs.

    Groups by: selector × model × k
    Computes: mean, std, sem, 95% CI for each metric

    Parameters
    ----------
    df : DataFrame
        Combined raw results from all runs

    Returns
    -------
    DataFrame
        Aggregated statistics with columns:
        - selector, model, k
        - {metric}_mean, {metric}_std, {metric}_sem, {metric}_ci_lower, {metric}_ci_upper
        - n_runs (number of runs aggregated)
        - n_folds_per_run (number of folds per run)
    """
    # Metrics to aggregate (exclude metadata columns)
    metadata_cols = {"selector", "model", "k", "run_id", "fold_id", "n_features"}
    metric_candidates = [col for col in df.columns if col not in metadata_cols]
    metric_cols = [col for col in metric_candidates if pd.api.types.is_numeric_dtype(df[col])]
    dropped_cols = sorted(set(metric_candidates) - set(metric_cols))
    if dropped_cols:
        logger.info(f"Skipping non-numeric columns in aggregation: {dropped_cols}")

    logger.info(f"Aggregating {len(metric_cols)} metrics: {metric_cols}")

    # Group by selector, model, k, and run_id first to get per-run averages
    # This ensures we're averaging over folds within each run, then over runs
    run_averages = df.groupby(["selector", "model", "k", "run_id"])[metric_cols].mean().reset_index()

    logger.info(f"Computed per-run averages: {len(run_averages)} rows")

    # Now aggregate across runs
    grouped = run_averages.groupby(["selector", "model", "k"])

    results = []

    for (selector, model, k), group in grouped:
        n_runs = len(group)

        row = {
            "selector": selector,
            "model": model,
            "k": k,
            "n_runs": n_runs,
        }

        # Compute statistics for each metric
        for metric in metric_cols:
            values = group[metric].values

            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values, ddof=1) if n_runs > 1 else 0.0
            row[f"{metric}_sem"] = stats.sem(values) if n_runs > 1 else 0.0

            # Compute 95% CI
            if n_runs >= 2:
                ci_lower, ci_upper = compute_confidence_interval(values)
                row[f"{metric}_ci_lower"] = ci_lower
                row[f"{metric}_ci_upper"] = ci_upper
            else:
                row[f"{metric}_ci_lower"] = np.nan
                row[f"{metric}_ci_upper"] = np.nan

        results.append(row)

    summary_df = pd.DataFrame(results)

    # Sort by selector, model, k for readability
    summary_df = summary_df.sort_values(["selector", "model", "k"]).reset_index(drop=True)

    return summary_df


def aggregate_stability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate stability metrics separately.

    Stability is computed per (selector, model, k, run), not per fold.
    So we aggregate directly across runs.

    Parameters
    ----------
    df : DataFrame
        Combined raw results

    Returns
    -------
    DataFrame
        Aggregated stability statistics
    """
    if "stability_jaccard" not in df.columns:
        logger.warning("No stability_jaccard column found. Skipping stability aggregation.")
        return pd.DataFrame()

    # Stability is constant within (selector, model, k, run_id)
    # Take one value per run
    stability_per_run = (
        df.groupby(["selector", "model", "k", "run_id"])["stability_jaccard"]
        .first()
        .reset_index()
    )

    grouped = stability_per_run.groupby(["selector", "model", "k"])

    results = []

    for (selector, model, k), group in grouped:
        n_runs = len(group)
        values = group["stability_jaccard"].values

        row = {
            "selector": selector,
            "model": model,
            "k": k,
            "n_runs": n_runs,
            "stability_mean": np.mean(values),
            "stability_std": np.std(values, ddof=1) if n_runs > 1 else 0.0,
            "stability_sem": stats.sem(values) if n_runs > 1 else 0.0,
        }

        if n_runs >= 2:
            ci_lower, ci_upper = compute_confidence_interval(values)
            row["stability_ci_lower"] = ci_lower
            row["stability_ci_upper"] = ci_upper
        else:
            row["stability_ci_lower"] = np.nan
            row["stability_ci_upper"] = np.nan

        results.append(row)

    stability_df = pd.DataFrame(results)
    stability_df = stability_df.sort_values(["selector", "model", "k"]).reset_index(drop=True)

    return stability_df


def load_raw_results(input_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Load all raw result files for a dataset.

    Parameters
    ----------
    input_dir : Path
        Directory containing raw CSV files
    dataset : str
        Dataset name

    Returns
    -------
    DataFrame
        Combined results from all runs
    """
    patterns = [
        f"{dataset}_seed*.csv",                 # legacy
        f"{dataset}__run*__seed*.csv",          # new pattern
    ]
    files = []
    for p in patterns:
        files.extend(list(input_dir.glob(p)))
    files = sorted(set(files))

    if not files:
        logger.error(f"No raw results found for dataset '{dataset}' in {input_dir}")
        raise FileNotFoundError(f"No raw results found for dataset '{dataset}'")

    logger.info(f"Found {len(files)} raw result files:")
    for f in files:
        logger.info(f"  - {f.name}")

    # Load and concatenate all files
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if "dataset" not in df.columns or df["dataset"].isna().all():
            df["dataset"] = dataset
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    logger.info(f"Loaded {len(combined)} total rows from {len(files)} files")

    return combined


def main():
    """Main execution function."""
    args = parse_args()

    # Setup paths
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "src"))

    setup_logging()

    logger.info("=" * 80)
    logger.info("Results Aggregation")
    logger.info("=" * 80)

    # Determine directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = repo_root / "src" / "results" / "raw"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "src" / "results" / "aggregated"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load raw results
    try:
        df = load_raw_results(input_dir, args.dataset)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Check minimum runs requirement
    n_runs = df["run_id"].nunique()
    logger.info(f"Number of unique runs: {n_runs}")

    if n_runs < args.min_runs:
        logger.error(f"Insufficient runs: found {n_runs}, need at least {args.min_runs}")
        logger.error("Run more experiments before aggregating.")
        sys.exit(1)

    # Display data summary
    logger.info("")
    logger.info("Data Summary:")
    logger.info(f"  Selectors: {sorted(df['selector'].unique())}")
    logger.info(f"  Models: {sorted(df['model'].unique())}")
    logger.info(f"  K values: {sorted(df['k'].unique())}")
    logger.info(f"  Runs: {sorted(df['run_id'].unique())}")
    logger.info(f"  Folds per run: {df.groupby('run_id')['fold_id'].nunique().values}")
    logger.info("")

    # Aggregate performance metrics
    logger.info("Aggregating performance metrics...")
    summary_df = aggregate_performance_metrics(df)

    summary_path = output_dir / f"{args.dataset}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved performance summary: {summary_path.relative_to(repo_root)}")
    logger.info(f"  Rows: {len(summary_df)}")

    # Aggregate stability metrics (if present)
    logger.info("")
    logger.info("Aggregating stability metrics...")
    stability_df = aggregate_stability_metrics(df)

    if not stability_df.empty:
        stability_path = output_dir / f"{args.dataset}_stability_summary.csv"
        stability_df.to_csv(stability_path, index=False)
        logger.info(f"✓ Saved stability summary: {stability_path.relative_to(repo_root)}")
        logger.info(f"  Rows: {len(stability_df)}")
    else:
        logger.info("  (No stability metrics to aggregate)")

    # Print sample of results
    logger.info("")
    logger.info("Sample of aggregated results:")
    logger.info("")

    # Show a few representative rows
    sample_metrics = ["accuracy_mean", "roc_auc_mean", "f1_mean"]
    display_cols = ["selector", "model", "k", "n_runs"] + [
        col for col in sample_metrics if col in summary_df.columns
    ]

    print(summary_df[display_cols].head(10).to_string(index=False))
    logger.info("")

    logger.info("=" * 80)
    logger.info("Aggregation complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  - Generate plots: python plot_publication_figures.py --dataset {args.dataset}")
    logger.info(f"  - View summary: {summary_path}")
    logger.info("")


if __name__ == "__main__":
    main()

