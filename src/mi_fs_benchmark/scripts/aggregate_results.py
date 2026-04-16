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
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from mi_fs_benchmark.experiment.eval.metrics import compute_statistical_significance

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
        default=None,
        help="Single dataset name (legacy mode, e.g., santander_short)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets to aggregate (e.g., santander,home_credit,ieee_cis_fraud)",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Auto-discover datasets from raw files and aggregate all of them",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip per-dataset aggregation and only build combined CSV outputs from existing aggregated files",
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


def _discover_datasets_from_raw(input_dir: Path) -> list[str]:
    """Infer dataset names from raw file patterns."""
    discovered: set[str] = set()

    for f in input_dir.glob("*__run*__seed*.csv"):
        name = f.name
        marker = "__run"
        if marker in name:
            discovered.add(name.split(marker, 1)[0])

    for f in input_dir.glob("*_seed*.csv"):
        name = f.name
        marker = "_seed"
        if marker in name:
            discovered.add(name.split(marker, 1)[0])

    return sorted(discovered)


def _resolve_dataset_list(args: argparse.Namespace, input_dir: Path) -> list[str]:
    datasets: list[str] = []

    if args.datasets:
        datasets.extend([d.strip() for d in args.datasets.split(",") if d.strip()])

    if args.dataset:
        datasets.append(args.dataset.strip())

    if args.all_datasets:
        datasets.extend(_discover_datasets_from_raw(input_dir))

    # preserve order while removing duplicates
    uniq: list[str] = []
    seen: set[str] = set()
    for d in datasets:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


def _infer_dataset(name: str) -> str:
    if name.endswith("_statistical_significance.csv"):
        return name[: -len("_statistical_significance.csv")]
    if name.endswith("_stability_summary.csv"):
        return name[: -len("_stability_summary.csv")]
    if name.endswith("_summary.csv"):
        return name[: -len("_summary.csv")]
    return Path(name).stem


def _ensure_dataset_column(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    else:
        out["dataset"] = out["dataset"].fillna(dataset).astype(str)
    return out


def _require_columns(df: pd.DataFrame, required: Sequence[str], file_path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")


def _collect_per_dataset_files(output_dir: Path, suffix: str, datasets: Sequence[str] | None = None) -> list[Path]:
    if datasets:
        files: list[Path] = []
        for ds in datasets:
            p = output_dir / f"{ds}{suffix}"
            if p.exists():
                files.append(p)
            else:
                logger.warning(f"Expected file not found for dataset '{ds}': {p.name}")
        return files

    return sorted(
        p
        for p in output_dir.glob(f"*{suffix}")
        if not p.name.startswith("combined_")
    )


def _combine_and_write(
    files: Sequence[Path],
    out_path: Path,
    required_cols: Sequence[str],
) -> pd.DataFrame:
    if not files:
        logger.warning(f"No files found for {out_path.name}; skipping")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        ds = _infer_dataset(f.name)
        df = _ensure_dataset_column(df, ds)
        _require_columns(df, required_cols, f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out_path, index=False)
    logger.info(f"✓ Wrote {out_path.name} ({len(combined)} rows)")
    return combined


def _build_combined_table_source(
    output_dir: Path,
    combined_perf: pd.DataFrame,
    combined_stab: pd.DataFrame,
    combined_sig: pd.DataFrame,
) -> pd.DataFrame:
    if combined_perf.empty:
        logger.warning("combined_summary.csv is empty; skipping combined_table_source.csv")
        return pd.DataFrame()

    key_perf = ["dataset", "selector", "model", "k"]
    _require_columns(combined_perf, key_perf, output_dir / "combined_summary.csv")

    table_df = combined_perf.copy()

    if not combined_stab.empty:
        key_stab = ["dataset", "selector", "model", "k"]
        _require_columns(combined_stab, key_stab, output_dir / "combined_stability_summary.csv")

        stab_payload = combined_stab.copy()
        if "n_runs" in stab_payload.columns:
            stab_payload = stab_payload.rename(columns={"n_runs": "n_runs_stability"})

        table_df = table_df.merge(stab_payload, on=key_stab, how="left", suffixes=("", "_stab"))

    if not combined_sig.empty:
        key_sig = ["dataset", "model", "k", "selector"]
        _require_columns(combined_sig, key_sig, output_dir / "combined_statistical_significance.csv")
        table_df = table_df.merge(combined_sig, on=key_sig, how="left", suffixes=("", "_sig"))

        # Primary significance: paired t-test. Secondary: Wilcoxon.
        if "best_mi" in table_df.columns:
            table_df["is_best_mi_row"] = table_df["selector"].astype(str) == table_df["best_mi"].astype(str)
        else:
            table_df["is_best_mi_row"] = False

        if "significant_ttest" in table_df.columns:
            sig_t = table_df["significant_ttest"].astype("boolean").fillna(False).astype(bool)
            table_df["is_significant_primary"] = sig_t
        else:
            table_df["is_significant_primary"] = False

        if "significant_wilcoxon" in table_df.columns:
            sig_w = table_df["significant_wilcoxon"].astype("boolean").fillna(False).astype(bool)
            table_df["is_significant_secondary"] = sig_w
        else:
            table_df["is_significant_secondary"] = False

    out_path = output_dir / "combined_table_source.csv"
    table_df.to_csv(out_path, index=False)
    logger.info(f"✓ Wrote {out_path.name} ({len(table_df)} rows)")
    return table_df


def combine_aggregated_outputs(output_dir: Path, datasets: Sequence[str] | None = None) -> None:
    """Combine per-dataset aggregated outputs into consistent combined files."""
    perf_files = _collect_per_dataset_files(output_dir, "_summary.csv", datasets=datasets)
    # strict filter to avoid accidental inclusion of stability/significance files
    perf_files = [
        p for p in perf_files
        if not p.name.endswith("_stability_summary.csv")
        and not p.name.endswith("_statistical_significance.csv")
    ]

    stab_files = _collect_per_dataset_files(output_dir, "_stability_summary.csv", datasets=datasets)
    sig_files = _collect_per_dataset_files(output_dir, "_statistical_significance.csv", datasets=datasets)

    combined_perf = _combine_and_write(
        perf_files,
        output_dir / "combined_summary.csv",
        required_cols=["dataset", "selector", "model", "k"],
    )
    combined_stab = _combine_and_write(
        stab_files,
        output_dir / "combined_stability_summary.csv",
        required_cols=["dataset", "selector", "model", "k"],
    )
    combined_sig = _combine_and_write(
        sig_files,
        output_dir / "combined_statistical_significance.csv",
        required_cols=["dataset", "model", "k", "selector"],
    )

    _build_combined_table_source(output_dir, combined_perf, combined_stab, combined_sig)


def _aggregate_single_dataset(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    min_runs: int,
    repo_root: Path,
) -> Path:
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Load raw results
    df = load_raw_results(input_dir, dataset)

    # Check minimum runs requirement
    if "run_id" not in df.columns:
        raise ValueError("Missing required column 'run_id' in raw results")
    n_runs = df["run_id"].nunique()
    logger.info(f"Number of unique runs: {n_runs}")

    if n_runs < min_runs:
        raise ValueError(f"Insufficient runs: found {n_runs}, need at least {min_runs}")

    # Display data summary
    logger.info("")
    logger.info("Data Summary:")
    logger.info(f"  Selectors: {sorted(df['selector'].unique())}")
    logger.info(f"  Models: {sorted(df['model'].unique())}")
    logger.info(f"  K values: {sorted(df['k'].unique())}")
    logger.info(f"  Runs: {sorted(df['run_id'].unique())}")
    if "fold_id" in df.columns:
        logger.info(f"  Folds per run: {df.groupby('run_id')['fold_id'].nunique().values}")
    logger.info("")

    # Aggregate performance metrics
    logger.info("Aggregating performance metrics...")
    summary_df = aggregate_performance_metrics(df)

    summary_path = output_dir / f"{dataset}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Saved performance summary: {summary_path.relative_to(repo_root)}")
    logger.info(f"  Rows: {len(summary_df)}")

    # Aggregate stability metrics (if present)
    logger.info("")
    logger.info("Aggregating stability metrics...")
    stability_df = aggregate_stability_metrics(df)

    if not stability_df.empty:
        stability_path = output_dir / f"{dataset}_stability_summary.csv"
        stability_df.to_csv(stability_path, index=False)
        logger.info(f"✓ Saved stability summary: {stability_path.relative_to(repo_root)}")
        logger.info(f"  Rows: {len(stability_df)}")
    else:
        logger.info("  (No stability metrics to aggregate)")

    # Compute statistical significance testing
    logger.info("")
    logger.info("Computing statistical significance...")
    sig_path = output_dir / f"{dataset}_statistical_significance.csv"
    try:
        sig_df = compute_statistical_significance(
            df_folds=df,
            metric_col="roc_auc",  # Default to ROC AUC for statistical tests
            mi_selectors=["mi", "mrmr"],
            p_value_threshold=0.05,
            output_path=str(sig_path)
        )
        logger.info(f"✓ Saved statistical significance summary: {sig_path.relative_to(repo_root)}")
        logger.info(f"  Rows: {len(sig_df)}")
    except Exception as e:
        logger.error(f"Failed to compute statistical significance: {e}")

    # Print sample of results
    logger.info("")
    logger.info("Sample of aggregated results:")
    logger.info("")

    sample_metrics = ["accuracy_mean", "roc_auc_mean", "f1_mean"]
    display_cols = ["selector", "model", "k", "n_runs"] + [
        col for col in sample_metrics if col in summary_df.columns
    ]

    print(summary_df[display_cols].head(10).to_string(index=False))
    logger.info("")

    return summary_path


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

    try:
        datasets = _resolve_dataset_list(args, input_dir)

        if not args.combine_only:
            if not datasets:
                raise ValueError("No datasets specified. Use --dataset, --datasets, or --all-datasets.")

            logger.info(f"Datasets to aggregate: {datasets}")
            logger.info("")

            for i, ds in enumerate(datasets, start=1):
                logger.info("-" * 80)
                logger.info(f"[{i}/{len(datasets)}] Aggregating dataset: {ds}")
                logger.info("-" * 80)
                _aggregate_single_dataset(
                    dataset=ds,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    min_runs=args.min_runs,
                    repo_root=repo_root,
                )
                logger.info("")

        logger.info("=" * 80)
        logger.info("Combining aggregated outputs")
        logger.info("=" * 80)
        combine_aggregated_outputs(output_dir, datasets=datasets or None)

    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Aggregation complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  - Generate plots: python plot_publication_figures.py --dataset <dataset>")
    logger.info(f"  - Combined outputs in: {output_dir}")
    logger.info("")


if __name__ == "__main__":
    main()

