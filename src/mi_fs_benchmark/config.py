from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

SelectorName = Literal[
    "mi",
    "mrmr",
    "anova",
    "variance",
    "l1_logreg",
    "tree_importance",
    "boruta",
    "shap"
]

ModelName = Literal["logreg", "hgbt"]


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for a dataset and its splitting policy.
    """

    name: str
    root: Path | None
    fold_scheme: str
    target_column: str
    id_column: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for a single experiment run.

    Attributes
    ----------
    dataset:
        DatasetConfig describing source and splitting policy.
    selectors:
        List of feature-selection strategies to benchmark.
    models:
        List of downstream classifiers.
    k_values:
        Values of k (number of features) to evaluate.
    cv_outer_folds:
        Number of outer CV folds.
    cv_repeats:
        Number of CV repetitions (currently handled externally if used).
    seed:
        Global random seed.
    tracking_uri:
        Optional MLflow/W&B tracking URI.
    """

    dataset: DatasetConfig
    selectors: list[SelectorName]
    models: list[ModelName]
    k_values: list[int]
    cv_outer_folds: int
    cv_repeats: int
    seed: int
    tracking_uri: str | None = None
    # Optional caps to keep “messy” datasets lightweight
    max_rows_for_selector_fit: Optional[int] = None
    max_features_after_encoding: Optional[int] = None
    # Selector fit progress heartbeat (simple periodic text logs)
    selector_fit_progress_enabled: bool = True
    selector_fit_progress_interval_sec: int = 50


def _find_repo_root(start: Path) -> Path:
    """Find repo root by walking up to pyproject.toml or fallback to start."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cur


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML configuration file.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    ds_raw = raw["dataset"]
    root_path = ds_raw.get("root")

    resolved_root: Path | None
    if root_path is None:
        resolved_root = None
    else:
        p = Path(root_path)
        if p.is_absolute():
            resolved_root = p
        else:
            # Resolve relative to the config file location first; fallback to repo root
            resolved_root = (path.parent / p).resolve()
            if not resolved_root.exists():
                repo_root = _find_repo_root(path)
                resolved_root = (repo_root / p).resolve()

    dataset = DatasetConfig(
        name=ds_raw["name"],
        root=resolved_root,
        fold_scheme=ds_raw["fold_scheme"],
        target_column=ds_raw["target_column"],
        id_column=ds_raw.get("id_column"),
    )

    return ExperimentConfig(
        dataset=dataset,
        selectors=list(raw["selectors"]),
        models=list(raw["models"]),
        k_values=list(raw["k_values"]),
        cv_outer_folds=int(raw["cv_outer_folds"]),
        cv_repeats=int(raw["cv_repeats"]),
        seed=int(raw["seed"]),
        tracking_uri=raw.get("tracking_uri"),
        max_rows_for_selector_fit=raw.get("max_rows_for_selector_fit"),
        max_features_after_encoding=raw.get("max_features_after_encoding"),
        selector_fit_progress_enabled=bool(raw.get("selector_fit_progress_enabled", True)),
        selector_fit_progress_interval_sec=int(raw.get("selector_fit_progress_interval_sec", 10)),
    )
