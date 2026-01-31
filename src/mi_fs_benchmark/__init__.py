"""
MI Feature-Selection Benchmark.

This package provides a modular, config-driven pipeline to benchmark
mutual-information-based and baseline feature selection methods on
multiple Kaggle tabular datasets.
"""

# Note: Submodules should be imported directly:
#   from mi_fs_benchmark.core.fs import create_selector
#   from mi_fs_benchmark.data.datasets import create_dataset
#   from mi_fs_benchmark.experiment.cv.artifacts import FoldArtifact
#   from mi_fs_benchmark.scripts import run_k_sweep
#   etc.

__all__ = [
    "config",
    "logging_utils",
    "core",
    "data",
    "experiment",
    "scripts",
    "plotting",
]
