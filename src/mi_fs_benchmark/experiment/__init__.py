"""Experiment components for MI-FS Benchmark.

This module contains:
- cv: Cross-validation and fold artifacts
- eval: Evaluation metrics
- pipeline: Pipeline orchestration

Usage:
    from mi_fs_benchmark.experiment.cv.artifacts import FoldArtifact
    from mi_fs_benchmark.experiment.cv.runner import CVRunner
    from mi_fs_benchmark.experiment.eval.metrics import compute_all
    from mi_fs_benchmark.experiment.pipeline.runner import PipelineRunner
"""

__all__ = ['cv', 'eval', 'pipeline']

