# src/mi_fs_benchmark/fs/__init__.py
from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier

from .base import FeatureSelector


@dataclass
class MutualInformationSelector(FeatureSelector):
    n_neighbors: int = 3
    random_state: int | None = None
    mi_scores: np.ndarray | None = None
    device: str | None = None
    use_gpu_approximation: bool = False
    _scores: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        if self.mi_scores is not None:
            if self.mi_scores.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Precomputed MI scores length {self.mi_scores.shape[0]} "
                    f"does not match n_features {X.shape[1]}"
                )
            self._scores = self.mi_scores
        else:
            if self.use_gpu_approximation:
                from .mi_gpu import compute_mi_gpu_correlation

                self._scores = compute_mi_gpu_correlation(X, y, device=self.device)
            else:
                self._scores = mutual_info_classif(
                    X,
                    y,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )
        return self

    def rank_features(self) -> np.ndarray:
        if self._scores is None:
            raise RuntimeError("MutualInformationSelector not fitted.")
        return np.argsort(self._scores)[::-1]


@dataclass
class MRMRSelector(FeatureSelector):
    lambda_redundancy: float = 0.5
    max_features: int | None = None
    n_neighbors: int = 3
    random_state: int | None = None
    mi_scores: np.ndarray | None = None
    device: str | None = None
    use_gpu_approximation: bool = False
    _ranking: np.ndarray | None = None
    _relevance: np.ndarray | None = None
    _redundancy: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        n_samples, n_features = X.shape
        if self.mi_scores is not None:
            if self.mi_scores.shape[0] != n_features:
                raise ValueError(
                    f"Precomputed MI scores length {self.mi_scores.shape[0]} "
                    f"does not match n_features {n_features}"
                )
            relevance = self.mi_scores
        else:
            if self.use_gpu_approximation:
                from .mi_gpu import compute_mi_gpu_correlation

                relevance = compute_mi_gpu_correlation(X, y, device=self.device)
            else:
                relevance = mutual_info_classif(
                    X,
                    y,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )
        if self.use_gpu_approximation:
            from .mi_gpu import compute_pairwise_mi_gpu

            redundancy = compute_pairwise_mi_gpu(X, device=self.device)
        else:
            redundancy = self._compute_redundancy_cpu(X, n_features)
        max_feats = self.max_features or n_features
        selected: list[int] = []
        remaining: set[int] = set(range(n_features))
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)
        while len(selected) < max_feats and remaining:
            best_feat: int | None = None
            best_score = -np.inf
            sel_arr = np.array(selected, dtype=int)
            for j in remaining:
                mean_red = float(np.mean(redundancy[j, sel_arr])) if sel_arr.size > 0 else 0.0
                score = relevance[j] - self.lambda_redundancy * mean_red
                if score > best_score:
                    best_score = score
                    best_feat = j
            if best_feat is not None:
                selected.append(best_feat)
                remaining.remove(best_feat)
        self._ranking = np.array(selected, dtype=int)
        self._relevance = relevance
        self._redundancy = redundancy
        return self

    def _compute_redundancy_cpu(self, X: np.ndarray, n_features: int) -> np.ndarray:
        redundancy = np.zeros((n_features, n_features), dtype=float)
        for j in range(n_features):
            xj = X[:, j].reshape(-1, 1)
            for k in range(j + 1, n_features):
                xk = X[:, k]
                mi_jk = mutual_info_regression(
                    xj,
                    xk,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )[0]
                redundancy[j, k] = mi_jk
                redundancy[k, j] = mi_jk
        return redundancy

    def rank_features(self) -> np.ndarray:
        if self._ranking is None:
            raise RuntimeError("MRMRSelector not fitted.")
        return self._ranking


@dataclass
class AnovaSelector(FeatureSelector):
    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        from sklearn.feature_selection import f_classif

        f_scores, _ = f_classif(X, y)
        self._scores = f_scores
        return self

    def rank_features(self) -> np.ndarray:
        if not hasattr(self, "_scores") or self._scores is None:
            raise RuntimeError("AnovaSelector not fitted.")
        return np.argsort(self._scores)[::-1]


@dataclass
class VarianceSelector(FeatureSelector):
    threshold: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FeatureSelector:
        variances = np.var(X, axis=0)
        self._scores = variances
        return self

    def rank_features(self) -> np.ndarray:
        if not hasattr(self, "_scores") or self._scores is None:
            raise RuntimeError("VarianceSelector not fitted.")
        return np.argsort(self._scores)[::-1]


@dataclass
class L1LogRegSelector(FeatureSelector):
    C: float = 1.0
    penalty: str = "l1"
    solver: str = "liblinear"
    max_iter: int = 1000
    random_state: int | None = None
    _coefs: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        clf.fit(X, y)
        self._coefs = np.abs(clf.coef_).mean(axis=0)
        return self

    def rank_features(self) -> np.ndarray:
        if self._coefs is None:
            raise RuntimeError("L1LogRegSelector not fitted.")
        return np.argsort(self._coefs)[::-1]


@dataclass
class TreeImportanceSelector(FeatureSelector):
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int | None = None
    n_jobs: int = -1
    _importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FeatureSelector:
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf.fit(X, y)
        self._importances = rf.feature_importances_
        return self

    def rank_features(self) -> np.ndarray:
        if self._importances is None:
            raise RuntimeError("TreeImportanceSelector not fitted.")
        return np.argsort(self._importances)[::-1]


@dataclass
class BorutaSelector(FeatureSelector):
    n_estimators: int | str = "auto"
    random_state: int | None = None
    use_gpu: bool = False  # Allows configuration from runner
    device: str = "cpu"  # Allows configuration from runner
    _ranks: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        from boruta import BorutaPy
        from xgboost import XGBClassifier

        # Only use GPU if requested and available
        xgb_device = "cuda" if self.use_gpu else "cpu"

        rf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            tree_method="hist",
            device=xgb_device,
            random_state=self.random_state,
            n_jobs=-1 if not self.use_gpu else 1  # Avoid CPU thread contention on GPU
        )

        boruta_sel = BorutaPy(rf, n_estimators=self.n_estimators, random_state=self.random_state)
        boruta_sel.fit(X, y)
        self._ranks = boruta_sel.ranking_
        return self

    def rank_features(self) -> np.ndarray:
        if self._ranks is None:
            raise RuntimeError("BorutaSelector not fitted.")
        # Lower rank (1) is better for Boruta
        return np.argsort(self._ranks)


@dataclass
class ShapSelector(FeatureSelector):
    random_state: int | None = None
    use_gpu: bool = False
    device: str = "cpu"
    _importances: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        import shap
        import lightgbm as lgb

        lgb_params = {
            "random_state": self.random_state,
            "n_jobs": -1 if not self.use_gpu else 1,
            "verbose": -1
        }

        if self.use_gpu:
            lgb_params["device"] = "gpu"

        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(X, y)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)

        self._importances = importance
        return self

    def rank_features(self) -> np.ndarray:
        if self._importances is None:
            raise RuntimeError("ShapSelector not fitted.")
        # Higher SHAP values are better
        return np.argsort(self._importances)[::-1]

def create_selector(name: str, **kwargs) -> FeatureSelector:
    """
    Factory for feature selection strategies.

    Parameters
    ----------
    name:
        Selector identifier string.
    kwargs:
        Extra keyword arguments passed to selector constructor.

    Returns
    -------
    FeatureSelector
    """
    mapping: dict[str, type[FeatureSelector]] = {
        "mi": MutualInformationSelector,
        "mrmr": MRMRSelector,
        "anova": AnovaSelector,
        "variance": VarianceSelector,
        "l1_logreg": L1LogRegSelector,
        "tree_importance": TreeImportanceSelector,
        "boruta": BorutaSelector,
        "shap": ShapSelector,
    }
    try:
        cls = mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unknown selector: {name!r}") from exc

    # Filter kwargs to only include parameters that the constructor accepts
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return cls(**valid_kwargs)


__all__ = [
    "FeatureSelector",
    "create_selector",
    "MutualInformationSelector",
    "MRMRSelector",
    "AnovaSelector",
    "VarianceSelector",
    "L1LogRegSelector",
    "TreeImportanceSelector",
    "BorutaSelector",
    "ShapSelector",
]
