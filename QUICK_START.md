# Quick Start Guide - Running Experiments

**GPU-ONLY MODE**: Requires a CUDA-enabled GPU and PyTorch with CUDA support (no CPU fallback in practice).

## ✅ Two-Step Usage (Recommended)

### Step 1: Edit Configuration
Open `src/mi_fs_benchmark/scripts/run_multi_seed_experiment.py`:

```python
CONFIG: str = "santander_short.yaml"  # Which dataset/config
N_RUNS: int = 10                      # How many independent runs
GPU_DEVICE: str = "cuda:0"            # Which GPU (cuda:0, cuda:1, etc.)
```

### Step 2: Run
```bash
cd src/mi_fs_benchmark/scripts
python run_multi_seed_experiment.py
```

**That's it!** GPU acceleration is on by default (use `--no-gpu` only for debugging; it will be very slow).

---

## Examples

### Quick Smoke Test (3 runs, small dataset)
```python
CONFIG = "santander_short.yaml"
N_RUNS = 3
GPU_DEVICE = "cuda:0"
```

### Publication Run (10+ runs, main dataset)
```python
CONFIG = "santander.yaml"
N_RUNS = 10
GPU_DEVICE = "cuda:0"
```

### Multi-GPU System (use second GPU)
```python
CONFIG = "home_credit.yaml"
N_RUNS = 10
GPU_DEVICE = "cuda:1"  # Use second GPU
```

---

## Available Datasets

| Config File | Size | GPU Time/Run |
|-------------|------|--------------|
| `santander_short.yaml` | Small | ~1 min |
| `santander.yaml` | Medium | ~3 min |
| `home_credit.yaml` | Large | ~10 min |
| `ieee_fraud.yaml` (optional) | Large | ~15 min |

_Excluded: arcene, nfl_bdb (see src/configs/_excluded/README.md for rationale)._ 

---

## Complete Workflow

```bash
# 1. Run experiments (edit CONFIG first)
python run_multi_seed_experiment.py

# 2. Aggregate results
python aggregate_results.py --dataset santander_short

# 3. Generate plots
python plot_publication_figures.py --dataset santander_short
```

See **[WORKFLOW.md](WORKFLOW.md)** for detailed explanation.

---

**Last Updated**: January 27, 2026  
**Mode**: GPU-ONLY (no CPU fallback)
