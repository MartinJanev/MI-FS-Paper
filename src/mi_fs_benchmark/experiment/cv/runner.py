from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import warnings
import time

import numpy as np
import time

from mi_fs_benchmark.config import ExperimentConfig
from mi_fs_benchmark.experiment.cv.artifacts import load_all_folds, FoldArtifact
from mi_fs_benchmark.experiment.eval import metrics as metrics_mod
from mi_fs_benchmark.experiment.eval import stability as stability_mod
from mi_fs_benchmark.core.fs import create_selector
from mi_fs_benchmark.logging_utils import get_logger
from mi_fs_benchmark.core.models import create_model

logger = get_logger(__name__)


@dataclass
class CVResult:
    selector_name: str
    model_name: str
    k: int
    fold_scores: List[Dict[str, Any]]
    run_id: int = 0  # Seed or run identifier for multi-run experiments


def _tqdm(iterable, **kwargs):
    """Local tqdm wrapper (keeps tqdm optional and avoids import cost when off)."""
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _run_fold_worker(args):
    """
    Worker function for parallel fold processing.

    This function is picklable and can be used with ProcessPoolExecutor.
    """
    selector_name, model_name, k, fold_data, seed, use_gpu, device, max_rows_fit = args

    # Reconstruct fold artifact from data
    X_train = fold_data["X_train"]
    X_test = fold_data["X_test"]
    y_train = fold_data["y_train"]
    y_test = fold_data["y_test"]
    mi_lookup = fold_data["mi_lookup"]
    fold_id = fold_data["fold_id"]
    n_features = fold_data["n_features"]

    # Create selector with GPU parameters
    selector_kwargs = {"random_state": seed, "seed": seed}

    if mi_lookup is not None:
        selector_lower = selector_name.lower()
        if selector_lower in ["mi", "mutual_information"]:
            selector_kwargs["mi_scores"] = mi_lookup
        elif selector_lower == "mrmr":
            selector_kwargs["mi_scores"] = mi_lookup

    # Add GPU parameters for MI and MRMR selectors
    selector_lower = selector_name.lower()
    if selector_lower in ["mi", "mutual_information", "mrmr"]:
        selector_kwargs["use_gpu_approximation"] = use_gpu
        selector_kwargs["device"] = device


    selector = create_selector(selector_name, **selector_kwargs)

    selector_start = time.time()
    if max_rows_fit is not None:
        X_fit, y_fit = _subsample_rows_for_selector(X_train, y_train, max_rows_fit, seed, fold_id)
    else:
        X_fit, y_fit = X_train, y_train
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)
        selector.fit(X_fit, y_fit)
    eff_k = min(k, X_train.shape[1])
    if eff_k < k:
        logger.debug(f"Clamping k from {k} to {eff_k} due to limited features in worker fold {fold_id}")
    feature_idx = selector.select_k(eff_k)
    fit_time_selector = time.time() - selector_start

    # Apply feature selection
    X_train_sel = X_train[:, feature_idx]
    X_test_sel = X_test[:, feature_idx]

    # Train model
    model = create_model(model_name)
    model_start = time.time()
    model.fit(X_train_sel, y_train)
    fit_time_model = time.time() - model_start

    # Predict
    proba = model.predict_proba(X_test_sel)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Compute metrics
    scores = metrics_mod.compute_all(
        y_true=y_test,
        y_proba=proba,
        y_pred=pred,
    )
    scores["fold_id"] = fold_id
    scores["n_features"] = n_features
    scores["k"] = k
    scores["selector"] = selector_name
    scores["model"] = model_name
    scores["dataset"] = fold_data.get("dataset") if "dataset" in fold_data else None
    scores["fit_time_selector"] = fit_time_selector
    scores["fit_time_model"] = fit_time_model
    scores["seed"] = seed

    return scores, feature_idx.tolist(), fold_id


def _cap_features_by_variance(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_lookup: Optional[np.ndarray],
    feature_names: list[str],
    cap: int,
):
    if cap is None or cap <= 0 or X_train.shape[1] <= cap:
        return X_train, X_test, mi_lookup, feature_names
    v = np.var(X_train, axis=0)
    idx = np.argsort(v)[::-1][:cap]
    X_train_c = X_train[:, idx]
    X_test_c = X_test[:, idx]
    mi_c = mi_lookup[idx] if mi_lookup is not None else None
    names_c = [feature_names[i] for i in idx] if feature_names else []
    return X_train_c, X_test_c, mi_c, names_c


def _subsample_rows_for_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_rows: Optional[int],
    seed: int,
    fold_id: int,
):
    if max_rows is None or X_train.shape[0] <= max_rows:
        return X_train, y_train
    rng = np.random.RandomState(seed + 1000 * fold_id)
    idx = rng.choice(X_train.shape[0], size=max_rows, replace=False)
    return X_train[idx], y_train[idx]


def _drop_constant_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_lookup: Optional[np.ndarray],
    feature_names: list[str],
):
    """Remove zero-variance columns to avoid downstream selector warnings."""
    if X_train.size == 0:
        return X_train, X_test, mi_lookup, feature_names
    var = np.var(X_train, axis=0)
    mask = var > 0.0
    if mask.all():
        return X_train, X_test, mi_lookup, feature_names
    X_train_c = X_train[:, mask]
    X_test_c = X_test[:, mask]
    mi_c = mi_lookup[mask] if mi_lookup is not None else None
    names_c = [feature_names[i] for i, keep in enumerate(mask) if keep] if feature_names else []
    return X_train_c, X_test_c, mi_c, names_c


class CVRunner:
    """
    Cross-validation experiment runner for k-sweep feature-selection
    benchmarks.

    NEW ARCHITECTURE:
    - Loads preprocessed fold artifacts (no raw data access)
    - No preprocessing in runner (scaling/discretization already done)
    - Selectors receive only numpy arrays
    - Uses precomputed MI when available

    Responsibilities:
    - load fold artifacts from disk,
    - for each selector/model/k:
        - fit selector on artifact.X_train_disc,
        - select top-k features,
        - fit model on selected features,
        - compute metrics on test,
        - aggregate stability statistics.
    """

    def __init__(self, cfg: ExperimentConfig, *, show_progress: bool = True, n_jobs: int = 1, use_gpu: bool = False, device: str = "cpu", run_id: int = 0) -> None:
        self.cfg = cfg
        self.show_progress = bool(show_progress)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.use_gpu = use_gpu
        self.device = device
        self.run_id = run_id  # Track which run this is (for multi-seed experiments)

        # Silence sklearn.parallel delayed warning globally
        warnings.filterwarnings(
            "ignore",
            message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
            category=UserWarning,
        )

    def run_k_sweep(self, folds: Optional[List[FoldArtifact]] = None) -> List[CVResult]:
        """
        Run k-sweep experiment using preprocessed fold artifacts.

        Now supports parallel processing across folds for faster execution.

        Parameters
        ----------
        folds : Optional[List[FoldArtifact]]
            Pre-generated fold artifacts. If None, loads from artifact_dir.

        Returns
        -------
        List[CVResult]
            Results for all selector/model/k combinations.
        """
        # Load or use provided fold artifacts
        if folds is None:
            artifact_dir = Path(self.cfg.dataset.root) / "fold_artifacts"
            if not artifact_dir.exists():
                artifact_dir = Path(self.cfg.dataset.root)

            folds = load_all_folds(artifact_dir)

            if not folds:
                raise FileNotFoundError(f"No fold artifacts found in {artifact_dir}")

        # Check dataset size and decide on processing mode
        n_samples = len(folds[0].X_train_disc)
        n_features = folds[0].X_train_disc.shape[1]

        # Force sequential for large datasets to avoid memory issues
        force_sequential = n_samples > 100000 or n_features > 1000

        if force_sequential and self.n_jobs > 1:
            logger.warning(
                f"⚠️  Large dataset: {n_samples} samples × {n_features} features → Sequential mode"
            )
            # Override n_jobs to force sequential
            self.n_jobs = 1

        if len(folds) != self.cfg.cv_outer_folds:
            logger.warning(
                f"Config specifies {self.cfg.cv_outer_folds} folds, "
                f"but {len(folds)} artifacts found. Using {len(folds)}."
            )

        total = (
            len(self.cfg.selectors)
            * len(self.cfg.models)
            * len(self.cfg.k_values)
            * len(folds)
        )

        # One global progress bar across all folds/combos
        pbar = _tqdm(
            range(total),
            disable=not self.show_progress,
            desc="k-sweep",
            unit="fold",
            leave=True,
        )
        pbar_iter = iter(pbar)

        results: list[CVResult] = []

        # Convert folds to serializable format for multiprocessing
        fold_data_list = []
        processed_folds: list[FoldArtifact] = []
        for fold in folds:
            Xtr, Xte, mi_lu, fnames = fold.X_train_disc, fold.X_test_disc, fold.mi_lookup, getattr(fold, "feature_names", [])
            Xtr, Xte, mi_lu, fnames = _drop_constant_features(Xtr, Xte, mi_lu, fnames)
            if self.cfg.max_features_after_encoding is not None:
                Xtr, Xte, mi_lu, fnames = _cap_features_by_variance(
                    Xtr, Xte, mi_lu, fnames, int(self.cfg.max_features_after_encoding)
                )
            fold_data_list.append({
                "X_train": Xtr,
                "X_test": Xte,
                "y_train": fold.y_train,
                "y_test": fold.y_test,
                "mi_lookup": mi_lu,
                "fold_id": fold.fold_id,
                "n_features": Xtr.shape[1],
                "dataset": self.cfg.dataset.name,
                "feature_names": fnames,
            })

            class ProcessedFold:
                def __init__(self, fd):
                    self.X_train_disc = fd["X_train"]
                    self.X_test_disc = fd["X_test"]
                    self.y_train = fd["y_train"]
                    self.y_test = fd["y_test"]
                    self.mi_lookup = fd["mi_lookup"]
                    self.fold_id = fd["fold_id"]
                    self.n_features = fd["n_features"]
                    self.feature_names = fd.get("feature_names", [])

            processed_folds.append(ProcessedFold(fold_data_list[-1]))

        # Use processed folds for ranking and sequential execution to keep indices aligned
        folds = processed_folds

        # Validate feature cap vs k grid early
        if self.cfg.max_features_after_encoding is not None:
            cap = int(self.cfg.max_features_after_encoding)
            for k in self.cfg.k_values:
                if k > cap:
                    raise ValueError(
                        f"k={k} exceeds max_features_after_encoding={cap}. Reduce k or increase cap."
                    )

        # Estimate memory requirements for multiprocessing
        # If data is too large, fall back to sequential processing
        use_parallel = self.n_jobs > 1
        if use_parallel and fold_data_list:
            # Estimate memory per fold (in MB)
            sample_fold = fold_data_list[0]
            memory_per_fold_mb = (
                sample_fold["X_train"].nbytes +
                sample_fold["X_test"].nbytes +
                sample_fold["y_train"].nbytes +
                sample_fold["y_test"].nbytes
            ) / (1024 ** 2)

            # Conservative threshold: 500 MB per fold for safe multiprocessing
            # This accounts for pickling overhead and process memory duplication
            if memory_per_fold_mb > 500:
                logger.warning(
                    f"Dataset is large ({memory_per_fold_mb:.1f} MB per fold). "
                    f"Falling back to sequential processing to avoid memory issues."
                )
                use_parallel = False

        for selector_name in self.cfg.selectors:
            # Precompute ranking once per selector per fold to avoid refitting per k/model
            selector_rankings = self._precompute_selector_rankings(selector_name, folds)

            # With caching, stick to sequential path for simplicity and to avoid pickling large rankings
            use_parallel_selector = False

            for model_name in self.cfg.models:
                for k in self.cfg.k_values:
                    if use_parallel_selector:
                        fold_scores, selected_sets = self._run_folds_parallel(
                            selector_name, model_name, k, fold_data_list, pbar, pbar_iter
                        )
                    else:
                        fold_scores, selected_sets = self._run_folds_sequential(
                            selector_name, model_name, k, folds, pbar, pbar_iter, selector_rankings
                        )

                    # Stability across folds
                    jacc = stability_mod.mean_pairwise_jaccard(selected_sets)
                    for fs in fold_scores:
                        fs["stability_jaccard"] = jacc

                    results.append(
                        CVResult(
                            selector_name=selector_name,
                            model_name=model_name,
                            k=k,
                            fold_scores=fold_scores,
                            run_id=self.run_id,
                        )
                    )

        try:
            pbar.close()
        except Exception:
            pass

        return results

    def _precompute_selector_rankings(self, selector_name: str, folds: list[FoldArtifact]):
        """Fit selector once per fold and store full ranking and fit time."""
        rankings = {}
        for fold in folds:
            X_train = fold.X_train_disc
            y_train = fold.y_train
            selector_kwargs = {"random_state": self.cfg.seed, "seed": self.cfg.seed}

            if fold.mi_lookup is not None:
                selector_lower = selector_name.lower()
                if selector_lower in ["mi", "mutual_information", "mrmr"]:
                    selector_kwargs["mi_scores"] = fold.mi_lookup

            selector_lower = selector_name.lower()
            if selector_lower in ["mi", "mutual_information", "mrmr"]:
                selector_kwargs["use_gpu_approximation"] = self.use_gpu
                selector_kwargs["device"] = self.device

            selector = create_selector(selector_name, **selector_kwargs)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
                warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)
                sel_start = time.time()
                if self.cfg.max_rows_for_selector_fit is not None:
                    X_fit, y_fit = _subsample_rows_for_selector(
                        X_train, y_train, int(self.cfg.max_rows_for_selector_fit), self.cfg.seed, fold.fold_id
                    )
                else:
                    X_fit, y_fit = X_train, y_train
                selector.fit(X_fit, y_fit)
                # Request full ranking by selecting all features
                feature_idx = selector.select_k(X_train.shape[1])
                fit_time_selector = time.time() - sel_start

            rankings[fold.fold_id] = {
                "feature_idx": feature_idx,
                "fit_time_selector": fit_time_selector,
            }

        return rankings

    def _run_folds_sequential(self, selector_name, model_name, k, folds, pbar, pbar_iter, selector_rankings=None):
        """Sequential fold processing (original behavior)."""
        fold_scores: list[dict[str, Any]] = []
        selected_sets: list[list[int]] = []

        for fold in folds:
            if self.show_progress:
                try:
                    pbar.set_postfix(
                        {
                            "selector": selector_name,
                            "model": model_name,
                            "k": k,
                            "fold": fold.fold_id,
                        }
                    )
                except Exception:
                    pass

            score_dict, selected_idx = self._run_single_fold(
                selector_name, model_name, k, fold, selector_rankings
            )
            fold_scores.append(score_dict)
            selected_sets.append(selected_idx)

            try:
                next(pbar_iter)
            except StopIteration:
                pass

        return fold_scores, selected_sets

    def _run_folds_parallel(self, selector_name, model_name, k, fold_data_list, pbar, pbar_iter):
        """Parallel fold processing using ProcessPoolExecutor."""
        fold_scores: list[dict[str, Any]] = [None] * len(fold_data_list)
        selected_sets: list[list[int]] = [None] * len(fold_data_list)

        # Prepare work items with GPU parameters
        work_items = [
            (selector_name, model_name, k, fold_data, self.cfg.seed, self.use_gpu, self.device,
             self.cfg.max_rows_for_selector_fit)
            for fold_data in fold_data_list
        ]

        try:
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_idx = {
                    executor.submit(_run_fold_worker, work): idx
                    for idx, work in enumerate(work_items)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        score_dict, selected_idx, fold_id = future.result()
                        fold_scores[idx] = score_dict
                        selected_sets[idx] = selected_idx

                        if self.show_progress:
                            try:
                                pbar.set_postfix(
                                    {
                                        "selector": selector_name,
                                        "model": model_name,
                                        "k": k,
                                        "fold": fold_id,
                                    }
                                )
                            except Exception:
                                pass

                        try:
                            next(pbar_iter)
                        except StopIteration:
                            pass

                    except MemoryError as e:
                        logger.error(f"Memory error in parallel processing: {e}")
                        logger.warning("Falling back to sequential processing...")
                        # Close executor and fall back to sequential
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        logger.error(f"Fold {idx} failed: {e}")
                        raise

        except MemoryError:
            # Fall back to sequential processing
            logger.warning("Retrying with sequential processing due to memory constraints...")
            # Reconstruct folds from fold_data_list
            from mi_fs_benchmark.experiment.cv.artifacts import FoldArtifact
            folds = []
            for fd in fold_data_list:
                # Create a minimal fold object for sequential processing
                class MinimalFold:
                    def __init__(self, data):
                        self.X_train_disc = data["X_train"]
                        self.X_test_disc = data["X_test"]
                        self.y_train = data["y_train"]
                        self.y_test = data["y_test"]
                        self.mi_lookup = data["mi_lookup"]
                        self.fold_id = data["fold_id"]
                        self.n_features = data["n_features"]

                folds.append(MinimalFold(fd))

            return self._run_folds_sequential(
                selector_name, model_name, k, folds, pbar, pbar_iter
            )

        return fold_scores, selected_sets

    def _run_single_fold(
        self,
        selector_name: str,
        model_name: str,
        k: int,
        fold,
        selector_rankings=None,
    ) -> Tuple[Dict[str, Any], list[int]]:
        """
        Run a single fold evaluation.

        NEW: Uses fold artifact directly, no preprocessing.

        Parameters
        ----------
        selector_name : str
            Name of selector to use.
        model_name : str
            Name of model to use.
        k : int
            Number of features to select.
        fold : FoldArtifact
            Preprocessed fold artifact.

        Returns
        -------
        scores : dict
            Metric scores for this fold.
        selected_idx : list[int]
            Indices of selected features.
        """
        X_train, X_test, mi_lu, _ = _drop_constant_features(
            fold.X_train_disc, fold.X_test_disc, fold.mi_lookup, getattr(fold, "feature_names", [])
        )
        y_train = fold.y_train
        y_test = fold.y_test

        # Create selector (pass MI lookup if available for MI-based methods)
        selector_kwargs = {"random_state": self.cfg.seed, "seed": self.cfg.seed}

        # If selector is MI-based and we have precomputed MI, use it
        if mi_lu is not None:
            selector_lower = selector_name.lower()
            if selector_lower in ["mi", "mutual_information"]:
                selector_kwargs["mi_scores"] = mi_lu
            elif selector_lower == "mrmr":
                # MRMR also benefits from precomputed relevance scores
                selector_kwargs["mi_scores"] = mi_lu

        # Add GPU parameters for MI and MRMR selectors
        selector_lower = selector_name.lower()
        if selector_lower in ["mi", "mutual_information", "mrmr"]:
            selector_kwargs["use_gpu_approximation"] = self.use_gpu
            selector_kwargs["device"] = self.device

        selector = create_selector(selector_name, **selector_kwargs)

        use_cached = selector_rankings is not None and fold.fold_id in selector_rankings

        eff_k = min(k, X_train.shape[1])
        if eff_k < k:
            logger.debug(f"Clamping k from {k} to {eff_k} for fold {fold.fold_id} due to limited features")

        if use_cached:
            cached = selector_rankings[fold.fold_id]
            feature_idx_full = cached["feature_idx"]
            feature_idx = feature_idx_full[:eff_k]
            fit_time_selector = cached.get("fit_time_selector", 0.0)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
                warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)
                sel_start = time.time()
                selector.fit(X_train, y_train)
                feature_idx = selector.select_k(eff_k)
                fit_time_selector = time.time() - sel_start

        # Apply feature selection
        X_train_sel = X_train[:, feature_idx]
        X_test_sel = X_test[:, feature_idx]

        # Train model
        model = create_model(model_name)
        model_start = time.time()
        model.fit(X_train_sel, y_train)
        fit_time_model = time.time() - model_start

        # Predict
        proba = model.predict_proba(X_test_sel)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # Compute metrics
        scores = metrics_mod.compute_all(
            y_true=y_test,
            y_proba=proba,
            y_pred=pred,
        )
        scores["fold_id"] = fold.fold_id
        scores["n_features"] = X_train.shape[1]
        scores["k"] = k
        scores["selector"] = selector_name
        scores["model"] = model_name
        scores["fit_time_selector"] = fit_time_selector
        scores["fit_time_model"] = fit_time_model
        scores["seed"] = self.cfg.seed
        scores["dataset"] = self.cfg.dataset.name

        return scores, feature_idx.tolist()