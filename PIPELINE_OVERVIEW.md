# MI-FS Benchmark Pipeline Overview

This document provides a comprehensive, end-to-end overview of the experimental framework for the mutual information (MI) and baseline feature selection benchmarking codebase. The system focuses on robust cross-validation (CV), scalable feature selection strategies (including GPU-accelerated variants), consistent evaluation metrics across varying subsets of selected features ($K$), and publication-ready reporting.

---

## 1. Dataset Preparation (`src/mi_fs_benchmark/data/datasets/`)
Before running any feature selection experiments, individual datasets must be loaded and standardized through an extensible interface.
- **`BaseDataset` Inheritance**: Datasets are implemented as subclasses of `BaseDataset` (e.g., `HomeCreditDataset`, `SantanderDataset`, `IEEECISFraudDataset`). This base class orchestrates unified data loading, memory optimization, and standardization.
- **Specific Processing Scripts**: Each dataset might have historical or external prep scripts (e.g., transforming categorical arrays to numerical hashes, computing target encoded features, casting columns to memory-efficient dtypes like `int8` or `float32`).
- **Standardized Formats**: The framework ultimately parses datasets locally (e.g., via Parquet or CSV readers) and isolates the feature space (`X`) from the target space (`y`) before feeding them into the CV splitter.

---

## 2. Global Pre-computation (`src/mi_fs_benchmark/data/preprocessing/`)
Feature selection metrics—especially those involving Mutual Information (MI) or kernel-based approaches—are computationally bottlenecked by large feature spaces or numerous rows.
- **GPU-Accelerated MI (`mi_gpu.py`)**: To speed up iterative multi-seed experiments on high-dimensional data, pairwise feature correlations and MI scores against the target can be calculated via CUDA devices using customized tensor operations.
- **Discretization Pipelines**: Continuous features often undergo standard binning or discretizations (e.g., equal-width or quantile-based via `discretize.py`) inside the preprocessing steps to satisfy MI estimator requirements and improve signal detection stability.
- **Caching**: The framework is structured to enable the reuse of these precomputed relevance and redundancy matrices across loops over different $K$ limits, averting redundant processing.

---

## 3. Configuration Management (`src/configs/`)
Everything is orchestrated through YAML configurations (`home_credit.yaml`, `santander.yaml`, `ieee_fraud.yaml`, `arcene.yaml`). These tightly define:
- **Datasets**: Which dataset subclass to instantiate and what data paths.
- **Feature Estimators (Selectors)**: A list of feature selection strategies to benchmark (e.g., `mi`, `mrmr`, `anova`, `variance`, `l1_logreg`, `tree_importance`).
- **Downstream Models**: The classifiers used to evaluate the utility of the selected subsets (e.g., `hgbt` for HistGradientBoosting, `logreg` for linear LogisticRegression).
- **Sweep Parameters**: The specific $K$-values (number of features to select) to iterate over (e.g., [10, 20, 50, 100, 200]).
- **CV Topology**: Outer CV folds and the number of repeating seeds necessary to establish statistical confidence.

---

## 4. Experiment Execution (`src/mi_fs_benchmark/scripts/run_multi_seed_experiment.py`)
This primary entry point dispatches multi-fold, multi-seed cross-validation experiments based on the configuration profiles.

### Cross-Validation & State Tracking
- **`cv/runner.py`**: Manages iterating over outer CV folds and distinct deterministic random seeds. It relies on standard stratifications to ensure imbalanced datasets are split accurately. Work is distributed, and outcomes are returned as a list of `FoldArtifact` results.
- **Robust Exception Handling**: Folds and combinations are processed safely. If a specific pipeline segment fails, the runner can log and skip it rather than aborting the multi-day experiment.

### Pipeline Architecture (`pipeline/runner.py`)
Given a single train-test split (a `MinimalFold` or similar construct), the pipeline dynamically strings together:
1. **Fit Selector (`core/fs/`)**: The chosen selector (`MutualInformationSelector`, `MRMRSelector`, etc.) consumes the training X and y to yield a ranked list of importance or pre-selected subsets.
2. **Transform Feature Space**: Modifies the training and hold-out matrices to retain only the top $K$ features.
3. **Train Base Model (`core/models/__init__.py`)**: Trains the downstream classifier.
    - `HistGBDTModel`: A light non-linear gradient-boosted tree class capable of implicitly handling feature interactions.
    - `LogisticRegressionModel`: A simple linear classifier capable of measuring direct linear utility without the benefit of trees.
4. **Validation Inference (`pipeline/validation.py`)**: Generates test predictions (proba and discrete predictions) and routes them into the next stage for metric scoring.

---

## 5. Evaluation & Metrics Gathering (`src/mi_fs_benchmark/experiment/eval/`)
Once predictions are compiled for the test set of each processed fold, comprehensive metrics are derived.
- **Validation Metrics (`metrics.py`)**: Computes classification success measures: ROC-AUC, PR-AUC, Accuracy, and F1. Calculations apply over specific combinations of {Dataset, Model, Selector, $K$, Fold, Seed}.
- **Selection Stability (`stability.py`)**: Evaluates how the feature selector varies relative to changes in training data across folds. Generates robustness indicators such as **Jaccard similarity** and **Kuncheva Index** for the identified subsets to determine if a selector is merely highlighting noise.

---

## 6. Result Aggregation (`src/mi_fs_benchmark/scripts/aggregate_results.py`)
Because parallel runs typically dump isolated JSON or CSV logs representing individual chunks (e.g., `results/home_credit__run001__seed...`), aggregation scripts combine these into unified dataframes.
- Validates the presence of expected results across the matrix of $(Dataset \times Selector \times Model \times K \times Seed)$.
- Groups metrics by the experimental axes.
- Computes aggregate statistics (e.g., `mean`, `standard deviation`) and confidence intervals (e.g., `ci_lower`, `ci_upper`) for plotting routines. Returns outputs like `combined_summary.csv` or `stability_summary.csv`.

---

## 7. Plotting & Reporting (`src/mi_fs_benchmark/scripts/make_paper_plots.py` & `make_paper_plots_mi_based.py`)
The pipeline culminates with scripts specialized for academic presentation. Utilizing `matplotlib` and parsing `combined_summary.csv`, these scripts output clear visual stories.
- **Format**: All generated plots are strictly rendered and saved in `.png` high-resolution formats to save directory space.
- **Performance vs $K$**: Grids showcasing how models (e.g., HGBT, LogReg) scale performance (ROC-AUC) as $K$ increases, across grouped datasets and colored by selector.
- **Stability Gradients**: Similar plots substituting classification performance for Jaccard/Kuncheva indices to contrast selector stability at scaling $K$.
- **Rank Correlation Matrices**: Visual maps identifying how different feature selection approaches theoretically align with one another (Kendall or Spearman rank).
- **LaTeX Table Creation**: Automatic `.tex` table code generation isolating the best performing pairs at the optimal threshold points ($K^*$), alongside standard deviations and ranking annotations.
