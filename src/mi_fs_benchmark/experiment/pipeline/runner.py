# src/mi_fs_benchmark/pipeline/runner.py
"""High-level pipeline runner for feature selection experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mi_fs_benchmark.config import ExperimentConfig
from mi_fs_benchmark.experiment.cv.runner import CVRunner
from mi_fs_benchmark.experiment.eval.stats import flatten_cv_results
from mi_fs_benchmark.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""

    experiment_config: ExperimentConfig
    output_path: Path
    show_progress: bool = True
    use_gpu: bool = False
    device: str = "cpu"
    run_id: int = 0  # Seed or run identifier for multi-run experiments


class PipelineRunner:
    """
    High-level orchestrator for feature selection experiments.

    Handles:
    - K-sweep experiments
    - Redundancy scaling experiments
    - Result aggregation and persistence
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run_k_sweep(self) -> pd.DataFrame:
        """
        Run a k-sweep experiment across all configured selectors, models, and k values.

        Returns
        -------
        pd.DataFrame
            Flattened results with one row per (selector, model, k, fold).
        """
        cfg = self.config.experiment_config

        logger.info("🚀 Starting experiment...")

        # Force sequential when GPU is requested to avoid multi-process GPU contention
        n_jobs = 1 if self.config.use_gpu else -1

        import time
        start_time = time.time()

        runner = CVRunner(
            cfg,
            show_progress=self.config.show_progress,
            n_jobs=n_jobs,
            use_gpu=self.config.use_gpu,
            device=self.config.device,
            run_id=self.config.run_id,
        )
        results = runner.run_k_sweep()

        elapsed = time.time() - start_time

        df = flatten_cv_results(results)

        # Save results
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.config.output_path, index=False)

        # Compact completion log
        total_evals = len(cfg.selectors) * len(cfg.models) * len(cfg.k_values) * cfg.cv_outer_folds
        avg_time = elapsed / total_evals
        logger.info(f"✓ Completed {total_evals} evaluations in {elapsed:.1f}s (avg: {avg_time:.2f}s/eval)")

        return df




def create_pipeline_runner(
    experiment_config: ExperimentConfig,
    output_path: Path | str,
    show_progress: bool = True,
) -> PipelineRunner:
    """
    Factory function to create a configured PipelineRunner.

    Parameters
    ----------
    experiment_config
        Experiment configuration (from YAML).
    output_path
        Where to save results parquet.
    show_progress
        Whether to show progress bars.

    Returns
    -------
    PipelineRunner
    """
    pipeline_cfg = PipelineConfig(
        experiment_config=experiment_config,
        output_path=Path(output_path),
        show_progress=show_progress,
    )
    return PipelineRunner(pipeline_cfg)
