# MI-FS Benchmark

Mutual-information feature-selection benchmark for tabular datasets with multi-run CV, aggregation, and paper-ready plotting.

## What this repository does

- Runs feature-selection experiments across selectors, models, and `k` values.
- Supports multi-seed execution for statistical confidence.
- Aggregates metrics into mean/std/SEM/CI summaries.
- Produces publication plots and LaTeX tables from aggregated CSVs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Optional (if running GPU MI paths):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

GPU check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

## End-to-end workflow

### 1) Prepare processed folds (one-time per dataset)

Run the preparation script for your dataset:

```powershell
python src/mi_fs_benchmark/scripts/preparation/prepare_santander.py
python src/mi_fs_benchmark/scripts/preparation/prepare_home_credit.py
python src/mi_fs_benchmark/scripts/preparation/prepare_ieee_fraud.py
python src/mi_fs_benchmark/scripts/preparation/prepare_arcene.py
```

Expected outputs per dataset in `src/data/processed/<dataset>/`:

- `fold_0.npz`, `fold_1.npz`, ...
- `metadata.json`

### 2) Run experiments (multi-seed)

Main runner: `src/mi_fs_benchmark/scripts/run_experiment.py`

```powershell
python src/mi_fs_benchmark/scripts/run_experiment.py --config santander.yaml --n-runs 10 --use-gpu --device cuda:0
```

Defaults are defined at the top of `run_experiment.py` (`CONFIG`, `N_RUNS`, `USE_GPU`, `GPU_DEVICE`).

Raw outputs are written to `src/results/raw/` with names like:

- `santander__run001__seed236040.csv`

### 3) Aggregate results

Single dataset:

```powershell
python src/mi_fs_benchmark/scripts/aggregate_results.py --dataset santander
```

Multiple datasets:

```powershell
python src/mi_fs_benchmark/scripts/aggregate_results.py --datasets santander,home_credit,ieee_cis_fraud
```

Auto-discover all datasets from raw outputs:

```powershell
python src/mi_fs_benchmark/scripts/aggregate_results.py --all-datasets
```

Main outputs in `src/results/aggregated/`:

- `<dataset>_summary.csv`
- `<dataset>_stability_summary.csv`
- `<dataset>_statistical_significance.csv`
- `combined_summary.csv`
- `combined_stability_summary.csv`
- `combined_statistical_significance.csv`
- `combined_table_source.csv`

### 4) Generate paper figures

Primary plotting script:

```powershell
python src/mi_fs_benchmark/scripts/make_paper_plots.py --root . --outdir paper_figs
```

Alternative plotting/table script:

```powershell
python src/mi_fs_benchmark/scripts/make_paper_plots_2.py --summary_csv src/results/aggregated/combined_summary.csv --stability_csv src/results/aggregated/combined_stability_summary.csv --table_source_csv src/results/aggregated/combined_table_source.csv --outdir paper_figs
```

Typical outputs are written to `paper_figs/`.

## Config files

Available configs in `src/configs/`:

- `santander.yaml`
- `home_credit.yaml`
- `ieee_fraud.yaml`
- `arcene.yaml`

Common config fields:

```yaml
dataset:
  name: santander
  root: src/data/processed/santander

selectors: [variance, anova, l1_logreg, tree_importance, mi, mrmr, boruta, shap]
models: [logreg, hgbt]
k_values: [5, 10, 20, 35, 50, 75, 100, 150]
cv_outer_folds: 5
```

## Repository map

- `src/mi_fs_benchmark/core/fs/`: selectors (MI, mRMR, baselines)
- `src/mi_fs_benchmark/core/models/`: downstream models
- `src/mi_fs_benchmark/experiment/`: CV, pipeline, evaluation
- `src/mi_fs_benchmark/scripts/`: run, aggregate, plot, preparation
- `src/configs/`: dataset experiment configs
- `src/data/`: raw and processed data
- `src/results/`: raw and aggregated experiment outputs

## Documentation

- `ESSENTIALS.md`: compact operational guide
