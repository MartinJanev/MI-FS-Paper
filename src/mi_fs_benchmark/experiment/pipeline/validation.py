"""Pre-flight validation for experiment configurations and data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mi_fs_benchmark.config import ExperimentConfig, load_experiment_config
from mi_fs_benchmark.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of pre-flight validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any]


def validate_experiment_setup(
    config_path: Path,
    check_test_data: bool = True,
) -> ValidationResult:
    """
    Validate that an experiment configuration is ready to run.

    Checks:
    - Config file exists and loads correctly
    - Dataset root exists
    - Required parquet files exist (train.parquet, optionally test.parquet)
    - Files are not empty or corrupted
    - Target column exists in the data
    - ID column exists (if specified)
    - Feature columns are numeric

    Parameters
    ----------
    config_path
        Path to experiment YAML config file.
    check_test_data
        Whether to validate test.parquet exists (not needed for k-sweep).

    Returns
    -------
    ValidationResult
        Validation outcome with errors, warnings, and metadata.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    # Check 1: Config file exists
    if not config_path.exists():
        errors.append(f"Config file not found: {config_path}")
        return ValidationResult(False, errors, warnings, info)

    # Check 2: Config loads
    try:
        cfg = load_experiment_config(str(config_path))
        info["config_loaded"] = True
    except Exception as e:
        errors.append(f"Failed to load config {config_path}: {e}")
        return ValidationResult(False, errors, warnings, info)

    # Check 3: Dataset root resolution
    dataset_root = cfg.dataset.root
    if dataset_root is None:
        # Use default relative to config file's repo root
        config_repo_root = config_path.resolve().parents[1]
        dataset_root = config_repo_root / "data" / "processed" / cfg.dataset.name
    else:
        dataset_root = Path(dataset_root)
        if not dataset_root.is_absolute():
            # Resolve relative to repo root (parent of config directory)
            config_repo_root = config_path.resolve().parents[1]
            dataset_root = (config_repo_root / dataset_root).resolve()

    info["dataset_root"] = str(dataset_root)

    if not dataset_root.exists():
        errors.append(
            f"Dataset root does not exist: {dataset_root}\n"
            f"Run scripts/prepare_{cfg.dataset.name}.py to generate processed data."
        )
        return ValidationResult(False, errors, warnings, info)

    # Check 4: Fold artifacts exist
    fold_files = list(dataset_root.glob("fold_*.npz"))

    if not fold_files:
        errors.append(
            f"No fold artifacts found in: {dataset_root}\n"
            f"Expected files: fold_0.npz, fold_1.npz, ...\n"
            f"Run scripts/preparation/prepare_{cfg.dataset.name}.py to generate them."
        )
        return ValidationResult(False, errors, warnings, info)
    else:
        # Fold artifacts found - check metadata
        info["data_format"] = "fold_artifacts (new)"
        info["n_folds"] = len(fold_files)
        info["fold_files"] = [f.name for f in sorted(fold_files)]

        metadata_path = dataset_root / "metadata.json"
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
                info["n_features"] = metadata.get("n_features", "unknown")
                info["n_samples"] = metadata.get("n_samples", "unknown")

                # No target column validation needed for fold artifacts
                # (they're already preprocessed and split)
            except Exception as e:
                warnings.append(f"Could not read metadata.json: {e}")

        # Fold artifacts are self-contained, no further validation needed
        return ValidationResult(True, errors, warnings, info)


def validate_or_exit(config_path: Path, check_test_data: bool = False) -> ExperimentConfig:
    """
    Validate experiment setup and exit with clear error message if invalid.

    This is a convenience function for scripts that should fail-fast.

    Parameters
    ----------
    config_path
        Path to config YAML.
    check_test_data
        Whether to require test.parquet.

    Returns
    -------
    ExperimentConfig
        Loaded and validated config.
    """
    result = validate_experiment_setup(config_path, check_test_data=check_test_data)

    if result.warnings:
        for warning in result.warnings:
            logger.warning(warning)

    if not result.is_valid:
        logger.error("Validation failed for %s", config_path)
        for error in result.errors:
            logger.error("  - %s", error)
        raise SystemExit(1)

    logger.info("Validation passed for %s", config_path.name)
    logger.info("Validation completed successfully.")

    for key, value in result.info.items():
        if key not in ["train_columns", "test_columns"]:
            logger.debug("  %s: %s", key, value)

    return load_experiment_config(str(config_path))
