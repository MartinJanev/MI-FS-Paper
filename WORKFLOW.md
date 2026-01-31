# Workflow Questions & Answers

## Q1: Do I need the raw data or can I work with processed data directly?

### Answer: **You can work with processed data directly**

Once you have the processed data in `src/data/processed/{dataset_name}/`, you do NOT need the raw data anymore for running experiments.

### Why?

The workflow is:
1. **One-time**: Raw data → Preprocessing → Processed data (saved to disk)
2. **Repeated**: Processed data → Experiments → Results

### What you need:

```
src/data/processed/{dataset_name}/
├── X.npy              # Feature matrix
├── y.npy              # Target labels
└── metadata.json      # Dataset info
```

### When you DO need raw data:
- First time processing a new dataset
- Re-processing with different parameters
- Verifying data integrity

### When you DON'T need raw data:
- Running experiments with different selectors/models
- Testing different k values
- Multi-seed experiments
- Cross-validation runs

**Recommendation**: Keep raw data for reproducibility, but experiments only touch processed data.

---

## Q2: After preprocessing, what are the next steps?

### Complete Workflow

#### Step 1: Prepare dataset (one-time)
```bash
# Create processed data and fold artifacts
python src/mi_fs_benchmark/scripts/preparation/prepare_santander.py
# or other datasets: prepare_home_credit.py, prepare_ieee_fraud.py
```

**Output**: Creates processed data in `src/data/processed/{dataset}/` and fold artifacts there.

#### Step 2: Run Experiments (repeatable)
```bash
cd src/mi_fs_benchmark/scripts
python run_multi_seed_experiment.py
# CLI alt: python run_multi_seed_experiment.py --config santander_short.yaml --n-runs 10 --device cuda:0
```

**Configuration in script**:
```python
CONFIG = "santander_short.yaml"
N_RUNS = 10
USE_GPU = True
GPU_DEVICE = "cuda:0"
```

**Output**: 
- Raw results: `src/results/raw/{dataset}__run###__seed###.csv`
- One CSV per run/seed

#### Step 3: Aggregate Results (after runs)
```bash
python aggregate_results.py --dataset santander_short
```

**Output**: 
- Aggregated CSV: `src/results/aggregated/{dataset}_summary.csv`
- Stability CSV: `src/results/aggregated/{dataset}_stability_summary.csv` (if stability metrics exist)

#### Step 4: Generate Plots
```bash
python plot_publication_figures.py --dataset santander_short
```

**Output**: Plots in `plots/publication/` (and combined variants if requested)

### Summary Pipeline

```
Raw Data
   ↓
[scripts/preparation/prepare_*.py]
   ↓
Processed Data + Fold Artifacts (X.npy, y.npy, fold_*.npz)
   ↓
[run_multi_seed_experiment.py] ← multi-run seeds
   ↓
Raw Results (one CSV per run)
   ↓
[aggregate_results.py]
   ↓
Aggregated Results (mean ± std ± CI)
   ↓
[plot_publication_figures.py]
   ↓
Publication Plots
```

### GPU-Only Mode

- ✅ GPU required for MI/MRMR acceleration; PyTorch with CUDA must be installed
- ⚠️ `--no-gpu` exists only for debugging and will be very slow
- Parallel vs sequential fold processing is automatic based on memory; not a CPU/GPU switch

---

## Q3: When does it switch to sequential mode instead of parallel?

### Answer: Based on Memory Estimation

The system switches from **parallel** to **sequential** fold processing when:

### Condition 1: Memory per Fold > 500 MB
```python
memory_per_fold_mb = (
    X_train.nbytes + X_test.nbytes + 
    y_train.nbytes + y_test.nbytes
) / (1024 ** 2)

if memory_per_fold_mb > 500:
    # Switch to sequential
    use_parallel = False
```

### Condition 2: Dataset Size Threshold
```python
n_samples = X.shape[0]
n_features = X.shape[1]

force_sequential = (
    n_samples > 50_000 or 
    n_features > 1_000
)

if force_sequential:
    n_jobs = 1  # Sequential mode
```

### What's the Difference?

**Parallel Mode** (`n_jobs > 1`):
- Processes multiple CV folds simultaneously using multiprocessing
- Faster but uses more memory
- Each fold runs in separate CPU process
- Good for small/medium datasets

**Sequential Mode** (`n_jobs = 1`):
- Processes one CV fold at a time
- Slower but memory-safe
- Single process
- Required for large datasets

### Examples

| Dataset | Samples | Features | Memory/Fold | Mode |
|---------|---------|----------|-------------|------|
| Arcene Quick | 100 | 10,000 | ~40 MB | Parallel ✅ |
| Santander Short | 10,000 | 200 | ~80 MB | Parallel ✅ |
| Santander Full | 200,000 | 200 | ~1,600 MB | Sequential ⚠️ |
| Home Credit | 300,000 | 500 | ~3,000 MB | Sequential ⚠️ |

### GPU vs Parallel Processing

**Important**: These are SEPARATE concepts:

- **GPU Mode**: Where computation happens (GPU vs CPU)
  - Controlled by: `use_gpu=True`, `use_gpu_approximation=True`
  - Affects: Preprocessing, MI calculation
  - Your change: Now GPU-only, no CPU fallback

- **Parallel Mode**: Whether folds run simultaneously
  - Controlled by: `n_jobs` parameter
  - Affects: CV fold processing (multiprocessing)
  - Still has memory-based fallback (this is NOT a GPU/CPU fallback!)

### You Can Have:
- ✅ **GPU + Parallel**: GPU for MI/preprocessing, parallel CV folds
- ✅ **GPU + Sequential**: GPU for MI/preprocessing, sequential CV folds
- ❌ **CPU + anything**: Not possible anymore with your GPU-only config

---

## Quick Reference

### File Locations

```
MI-FS/
├── src/data/
│   ├── raw/{dataset}/           # Raw data (needed once)
│   └── processed/{dataset}/     # Processed data + fold artifacts
│
├── src/results/
│   ├── raw/                     # Per-run results
│   └── aggregated/              # Combined results
│
└── plots/
    ├── publication/             # Paper-ready plots
    └── combined/                # Multi-metric plots
```

### Key Scripts

| Script | Purpose | Run Frequency |
|--------|---------|---------------|
| `scripts/preparation/prepare_*.py` | Create processed data + folds | Once per dataset |
| `run_multi_seed_experiment.py` | Run experiment | Repeat (multi-seed) |
| `aggregate_results.py` | Combine seeds | Once after runs |
| `plot_publication_figures.py` | Generate plots | Once after aggregation |

### Configuration Hierarchy

```
1. Config YAML (santander_short.yaml)
   ├── Dataset settings
   ├── CV folds
   ├── k values
   ├── Selectors
   └── Models

2. Script Parameters (run_multi_seed_experiment.py)
   ├── CONFIG (which YAML to use)
   ├── N_RUNS (how many seeds)
   ├── USE_GPU (enable GPU preprocessing)
   └── GPU_DEVICE (which GPU)

3. Runtime Decisions
   ├── Parallel vs Sequential (automatic based on memory)
   └── GPU availability (now fails if not available)
```

---

## Available Datasets

- `santander_short.yaml` (small/fast)
- `santander.yaml` (medium)
- `home_credit.yaml` (large)
- `ieee_fraud.yaml` (optional confirmatory)
- Excluded: `arcene*.yaml`, `nfl_bdb.yaml` (see `src/configs/_excluded/README.md`)

---

**Date**: January 27, 2026  
**Status**: ✅ Complete workflow documentation with GPU-only mode
