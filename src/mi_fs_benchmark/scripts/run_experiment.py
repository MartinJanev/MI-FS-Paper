#!/usr/bin/env python3
# scripts/run_multi_seed_experiment.py
"""
Run feature selection experiments across multiple random seeds.

This script is the PRIMARY way to run experiments for publication.
It executes the same experiment configuration multiple times with different
random seeds to enable statistical aggregation and confidence intervals.

Usage:
    1. Edit the configuration variables below (EASY MODE)
    2. Run: python run_multi_seed_experiment.py

    OR use command-line arguments (ADVANCED):
    python run_multi_seed_experiment.py --config santander.yaml --n-runs 10

Output:
    - results/raw/{dataset}_seed{N}.csv for each run
    - Aggregation done separately via aggregate_results.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add repository src/ once for script execution contexts (local + Colab)."""
    src_path = str(Path(__file__).resolve().parents[3] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

# =============================================================================
# CONFIGURATION - Edit these variables for easy usage
# =============================================================================

# Use YAML for selectors/models/k; do not override here
CONFIG: str = "santander.yaml"
N_RUNS: int = 5                      # Number of independent runs (10-20 for publication)
USE_GPU: bool = True                  # Enable GPU acceleration (10-100x faster)
GPU_DEVICE: str = "cuda:0"            # GPU device ("cuda:0" for first GPU)
OUTPUT_DIR: str | None = None         # Output directory (None = auto: results/raw)
START_SEED: int | None = None         # Starting seed (None = use config seed)

# =============================================================================
# Advanced: You can leave the rest as-is
# =============================================================================

from mi_fs_benchmark.logging_utils import get_logger, setup_logging
from mi_fs_benchmark.experiment.pipeline import PipelineRunner, validate_experiment_setup
from mi_fs_benchmark.experiment.pipeline.runner import PipelineConfig
from mi_fs_benchmark.config import load_experiment_config

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments (optional - uses CONFIG variables as defaults)."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed feature selection experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG,
        help="Config file name (e.g., santander_short.yaml) or full path",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Number of independent runs with different seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory (default: results/raw)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=USE_GPU,
        help="Enable GPU acceleration for MI/MRMR",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (overrides --use-gpu and USE_GPU)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=GPU_DEVICE,
        help="GPU device to use",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=START_SEED,
        help="Starting seed (default: use seed from config)",
    )

    args = parser.parse_args()

    # Handle --no-gpu override
    if args.no_gpu:
        args.use_gpu = False

    return args


def resolve_config_path(repo_root: Path, config: str) -> Path:
    """Resolve config path allowing configs/ or src/configs/ relative to repo root."""
    p = Path(config)
    if p.is_absolute():
        return p
    if p.parent == Path(""):
        # try repo_root/configs then repo_root/src/configs
        candidates = [repo_root / "configs" / p, repo_root / "src" / "configs" / p]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]
    return (repo_root / p).resolve()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup repository paths
    repo_root = Path(__file__).resolve().parents[3]

    setup_logging()

    logger.info("=" * 80)
    logger.info("Multi-Seed Experiment Runner")
    logger.info("=" * 80)

    # Resolve config path
    cfg_path = resolve_config_path(repo_root, args.config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)

    # Validate configuration
    logger.info(f"Validating configuration: {cfg_path.name}")
    validation = validate_experiment_setup(cfg_path, check_test_data=False)

    for warning in validation.warnings:
        logger.warning(warning)

    if not validation.is_valid:
        logger.error(f"Validation failed for {cfg_path}")
        for error in validation.errors:
            logger.error(f"  ❌ {error}")
        sys.exit(1)

    logger.info("✓ Validation passed")

    # Load configuration
    cfg = load_experiment_config(str(cfg_path))
    dataset_name = cfg.dataset.name

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = repo_root / "src" / "results" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine seed range
    if args.start_seed is not None:
        base_seed = args.start_seed
    else:
        base_seed = cfg.seed

    logger.info("")
    logger.info("Experiment Configuration:")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Config: {cfg_path.name}")
    logger.info(f"  Number of runs: {args.n_runs}")
    logger.info(f"  Base seed: {base_seed}")
    logger.info(f"  Seed range: [{base_seed}, {base_seed + args.n_runs - 1}]")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("")

    # Display GPU information
    if args.use_gpu:
        try:
            from mi_fs_benchmark.core.fs.mi_gpu import get_gpu_info
            import torch
            gpu_info = get_gpu_info()
            logger.info("🎮 GPU ACCELERATION: ENABLED")
            logger.info(f"   Device: {args.device}")
            logger.info(f"   GPU Name: {gpu_info['device_name']}")
            logger.info(f"   CUDA Cores: {gpu_info['cuda_cores']}")
            logger.info(f"   Memory: {gpu_info['memory_total']:.1f} GB total, {gpu_info['memory_available']:.1f} GB available")
            logger.info(f"   Compute Capability: {gpu_info['compute_capability']}")
            logger.info(f"   PyTorch CUDA Version: {torch.version.cuda}")
            logger.info("")
        except Exception as e:
            logger.error(f"❌ GPU enabled but cannot access GPU: {e}")
            logger.error("   Make sure CUDA is available and PyTorch is installed with CUDA support")
            sys.exit(1)
    else:
        logger.warning("⚠️  GPU ACCELERATION: DISABLED")
        logger.warning("   This will be VERY slow. Consider enabling GPU with USE_GPU = True")
        logger.info("")

    # Run experiments for each seed
    total_start = time.time()
    successful_runs = 0
    failed_runs = []

    # Calculate experiment details for summary
    n_selectors = len(cfg.selectors)
    n_models = len(cfg.models)
    n_k_values = len(cfg.k_values)
    n_folds = cfg.cv_outer_folds
    total_evals_per_run = n_selectors * n_models * n_k_values * n_folds

    for run_idx in range(args.n_runs):
        run_seed = base_seed + run_idx
        run_num = run_idx + 1

        # Compact run header
        print()  # Blank line for separation
        logger.info("━" * 80)
        logger.info(f"🔄 RUN {run_num}/{args.n_runs} | Seed: {run_seed} | Progress: {run_num}/{args.n_runs} ({run_num/args.n_runs*100:.0f}%)")
        logger.info(f"   {n_selectors} selectors × {n_models} models × {n_k_values} k-values × {n_folds} folds = {total_evals_per_run} evaluations")
        logger.info("━" * 80)

        # Update config with current seed
        cfg_run = replace(cfg, seed=run_seed)

        # Define output path for this run
        run_id = run_idx + 1
        output_path = output_dir / f"{dataset_name}__run{run_id:03d}__seed{run_seed}.csv"

        try:
            # Create pipeline configuration
            pipeline_cfg = PipelineConfig(
                experiment_config=cfg_run,
                output_path=output_path,
                show_progress=True,
                use_gpu=args.use_gpu,
                device=args.device,
                run_id=run_id,
            )

            # Run experiment
            run_start = time.time()
            pipeline = PipelineRunner(pipeline_cfg)
            df = pipeline.run_k_sweep()
            run_elapsed = time.time() - run_start

            # Compact success summary
            avg_time = run_elapsed / total_evals_per_run
            logger.info(f"✅ Run {run_num} COMPLETE | Time: {run_elapsed / 60:.2f} min | Avg: {avg_time:.2f}s/eval | Rows: {len(df)}")
            logger.info(f"   📁 {output_path.relative_to(repo_root)}")
            successful_runs += 1

            # Show estimated time remaining
            if run_num < args.n_runs:
                remaining_runs = args.n_runs - run_num
                estimated_remaining = (run_elapsed * remaining_runs) / 60
                logger.info(f"   ⏱️  Estimated time remaining: {estimated_remaining:.1f} minutes ({estimated_remaining/60:.1f} hours)")

        except Exception as e:
            logger.error(f"❌ Run {run_num} FAILED | Error: {e}")
            failed_runs.append((run_num, run_seed, str(e)))


    # Summary
    total_elapsed = time.time() - total_start
    print()  # Blank line
    logger.info("━" * 80)
    logger.info("📊 EXPERIMENT SUMMARY")
    logger.info("━" * 80)
    logger.info(f"⏱️  Total Time: {total_elapsed / 60:.2f} minutes ({total_elapsed / 3600:.2f} hours)")
    logger.info(f"✅ Successful: {successful_runs}/{args.n_runs} runs")

    if successful_runs > 0:
        avg_time_per_run = total_elapsed / successful_runs
        logger.info(f"📈 Average: {avg_time_per_run / 60:.2f} min/run")

    if failed_runs:
        logger.warning(f"❌ Failed: {len(failed_runs)} runs")
        for run_num, seed, error in failed_runs:
            logger.warning(f"   • Run {run_num} (seed {seed}): {error}")

    logger.info("")
    logger.info("🎯 Next Steps:")
    logger.info(f"   1. Aggregate: python aggregate_results.py --dataset {dataset_name}")
    logger.info(f"   2. Plot:      python plot_publication_figures.py --dataset {dataset_name}")
    logger.info("━" * 80)

    if failed_runs:
        sys.exit(1)


if __name__ == "__main__":
    main()

