# Reviewer Response & Revision Plan

Based on the reviews provided and the content of the project files, here is a detailed plan to address the reviewers' concerns and the impact on the current codebase.

## Summary of Reviewer Feedback
* **Reviewer #1 (Weak Accept)**:
    * **Strengths**: Methodological rigor and the fold-aware, leakage-free protocol.
    * **Weaknesses**: Low algorithmic novelty.
    * **Actionable Items**: Add statistical significance tests (e.g., p-values), include confidence intervals in performance tables, and add modern baselines such as **Boruta** or **SHAP-guided filtering**.
* **Reviewer #2 (Accept)**:
    * **Strengths**: Thorough analysis across multiple datasets, models, and metrics (performance, stability, cost).
    * **Weaknesses**: Missing discussion on deep learning-based feature selection.
    * **Actionable Items**: Compare or discuss the findings in the context of modern deep learning approaches for feature selection.

---

## Action Plan & Codebase Impact

To address the reviewer feedback effectively, **new experiments are required**, but only for the newly proposed methods. Existing MI or classical baseline experiments do not need to be rerun.

### 1. Requires New Experiments (High Impact)
Reviewer #1's request for **Boruta** and **SHAP** baselines will take the most computational time.
* **Code Changes**: Implement two new classes (`BorutaSelector` and `ShapSelector`) that inherit from `FeatureSelector` inside `src/mi_fs_benchmark/core/fs/__init__.py`.
* **Configuration**: Update the YAML config files in `src/configs/` to include `"boruta"` and `"shap"` in the feature estimators list.
* **Execution**: Run `src/mi_fs_benchmark/scripts/run_multi_seed_experiment.py` targeting just these new selectors.
* **Aggregation**: Run `src/mi_fs_benchmark/scripts/aggregate_results.py` to merge the new Boruta/SHAP logs with existing results, followed by regenerating the plots with `make_paper_plots.py`.

### 2. Requires Scripting, but NO New Experiments (Medium Impact)
To fulfill Reviewer #1's request for **statistical significance (p-values)**, models do not need to be retrained.
* **Data Availability**: The pipeline already records validation metrics per fold/seed. 
* **Implementation**: Write a new script (or append to `src/mi_fs_benchmark/experiment/eval/metrics.py`) that loads the fold-level results from CSV logs and runs a Wilcoxon signed-rank test or paired t-test comparing the top MI metric against the top baseline metric.
* **Tables**: The `combined_summary.csv` already contains `ci_lower` and `ci_upper`. The LaTeX generation logic in `make_paper_plots.py` needs to be updated to print the $\pm$ CI bounds and append significance asterisks.

### 3. Text-Only Changes (Low Impact)
Reviewer #2's request regarding **Deep Learning feature selection** only requires adding a paragraph to the LaTeX manuscript.
* No changes to the codebase or experiments are needed.
* **Action**: Conceptually contrast the efficient tabular approaches against heavier neural architecture methods (like LassoNet or Concrete Autoencoders) in the paper's Discussion or Related Work section. Explain that while DL methods model complex non-linearities, they often require significantly more computational resources and tuning compared to the MI and classical methods explored in this study, which are optimized for tabular data efficiency.
