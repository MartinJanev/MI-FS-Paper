# MI-FS Essentials

Short version: prepare the data, run the experiment, aggregate the CSVs, then make plots/tables.

## Fast start

```powershell
# 1) Install once
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 2) Prepare a dataset once
python src/mi_fs_benchmark/scripts/preparation/prepare_home_credit.py

# 3) Run the benchmark
python src/mi_fs_benchmark/scripts/run_experiment.py --config src/configs/home_credit.yaml --n-runs 10 --no-gpu

# 4) Aggregate results
python src/mi_fs_benchmark/scripts/aggregate_results.py --dataset home_credit

# 5) Generate paper figures/tables
python src/mi_fs_benchmark/scripts/make_paper_plots.py --root . --outdir paper_figs
```

## Inputs you need

- Raw source files in `src/data/raw/<dataset>/` before preprocessing
- Santander, Home Credit, and IEEE-CIS Fraud come from Kaggle competition downloads
- Arcene uses the ARCENE benchmark files in `src/data/raw/arcene/`
- Processed folds in `src/data/processed/<dataset>/` are created by the preparation scripts
- A config from `src/configs/`
- A working Python environment

## Main outputs

- Raw CSVs: `src/results/raw/`
- Aggregated summaries: `src/results/aggregated/`
- Figures and tables: `paper_figs/` and `final_plots/`

## What the visuals mean

- Shaded plot bands show confidence intervals around the mean, usually 95% CI.
- A `*` in the reviewer table means the selector is flagged by the primary paired t-test against the best baseline in the generated significance file.
- Confusion matrices are displayed as `TP` top-left, `FN` top-right, `FP` bottom-left, `TN` bottom-right.

## If you are changing the code

The flow is:

1. `src/mi_fs_benchmark/scripts/preparation/` prepares folds.
2. `src/mi_fs_benchmark/core/fs/` ranks/selects features.
3. `src/mi_fs_benchmark/core/models/` trains the downstream model.
4. `src/mi_fs_benchmark/experiment/` computes metrics, stability, and significance.
5. `src/mi_fs_benchmark/scripts/aggregate_results.py` merges runs.
6. `src/mi_fs_benchmark/scripts/make_paper_plots.py` and `make_paper_plots_2.py` render figures and tables.

## Reminder

If a score looks strange, check the raw fold-level CSVs first. That is also where you start if you need evidence for a very low F1 value.
