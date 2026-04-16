# MI-FS Essentials

A compact guide to run the benchmark and understand the code flow.

## 1) What this project does

Benchmark feature selectors (MI and baselines) across datasets, models, and feature counts (`k`) using multi-seed cross-validation, then aggregate and plot publication-ready results.

## 2) Minimal run sequence

```powershell
# 1. Run experiments
python src/mi_fs_benchmark/scripts/run_multi_seed_experiment.py

# 2. Aggregate results
python src/mi_fs_benchmark/scripts/aggregate_results.py --dataset santander_short

# 3. Generate plots/tables
python src/mi_fs_benchmark/scripts/make_paper_plots.py
```

## 3) Required inputs

- Processed data in `src/data/processed/{dataset}/`
- Experiment config in `src/configs/*.yaml`
- Python environment with required dependencies

## 4) Core configuration knobs

In YAML configs:
- `selectors`: feature selection methods
- `models`: downstream estimators
- `k_values`: selected-feature sweep
- CV parameters: folds/seeds/splits

In run script/CLI:
- dataset config (`--config`)
- number of runs/seeds (`--n-runs`)
- device options (if enabled in your setup)

## 5) Key outputs

- Raw run results: `src/results/raw/`
- Aggregated metrics: `src/results/aggregated/`
- Figures and LaTeX tables: `paper_figs/` and/or `final_plots/`

## 6) Pipeline map

1. `data/datasets/`: load + standardize datasets
2. `data/preprocessing/`: preprocess and MI-related precompute/caching
3. `core/fs/`: fit selector and rank/select top-`k` features
4. `core/models/`: train downstream model
5. `experiment/eval/`: compute metrics + stability
6. `scripts/aggregate_results.py`: summarize across runs
7. `scripts/make_paper_plots.py`: render figures/tables

## 7) Runtime behavior to remember

- Experiments may switch parallel/sequential CV based on memory/size settings.
- Stability metrics (e.g., Jaccard/Kuncheva) complement predictive metrics.
- Aggregation reports mean/std and confidence intervals for comparisons.

