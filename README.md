# MI-FS Benchmark

**Mutual Information Feature Selection Benchmark for Research Papers**

A rigorous, reproducible framework for benchmarking MI-based feature selection methods on tabular datasets with **multi-run experiments**, **GPU acceleration**, and **statistical aggregation**.

**Mode**: GPU-ONLY (requires CUDA-enabled GPU; `--no-gpu` is only for slow debugging)

---

## ⚡ Quick Start

### 1. Install
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows (or source .venv/bin/activate for Linux/Mac)
pip install -e .

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Experiments
```bash
cd src/mi_fs_benchmark/scripts

# Edit CONFIG variable in run_multi_seed_experiment.py, then:
python run_multi_seed_experiment.py
# or CLI: python run_multi_seed_experiment.py --config santander_short.yaml --n-runs 10
```

### 3. Aggregate & Plot
```bash
python aggregate_results.py --dataset santander_short
python plot_publication_figures.py --dataset santander_short
```

**Time:** ~30 minutes for 10 runs with GPU (santander_short)

See **[QUICK_START.md](QUICK_START.md)** for details.

---

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| **[QUICK_START.md](QUICK_START.md)** | Fastest way to run experiments |
| **[WORKFLOW.md](WORKFLOW.md)** | Complete workflow explanation (Q&A) |

### 🎮 GPU Quick Check

```python
import torch
print("✅ GPU Available" if torch.cuda.is_available() else "❌ No GPU")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Expected Log Snippet at Experiment Start:**
```
🎮 GPU ACCELERATION: ENABLED
   Device: cuda:0
   GPU Name: <your GPU>
   Memory: <total> GB total, <available> GB available
```

---

## 🎯 Research-Grade Workflow

This benchmark enforces **research-grade experimental practices**:

1. **Multi-run experiments**: Run with multiple random seeds (default: 10)
2. **GPU-accelerated**: 10-100x speedup for MI/MRMR calculations
3. **Statistical aggregation**: Compute mean, std, SEM, 95% CI across runs
4. **Publication-ready plots**: Error bars, clean layout, 300 DPI

---

## 🎯 What This Provides

### Feature Selection Methods
- **MI (Mutual Information)** - Information-theoretic relevance (GPU-accelerated)
- **mRMR** - Minimum redundancy maximum relevance (GPU-accelerated)
- **ANOVA F-test** - Statistical baseline
- **Variance Threshold** - Unsupervised sanity check
- **L1-Logistic Regression** - Embedded sparse selector

### Classifiers
- **Logistic Regression** - Linear baseline

### Datasets (Configs)
- **santander_short.yaml** - Small/fast
- **santander.yaml** - Medium
- **home_credit.yaml** - Large
- **ieee_fraud.yaml** - Optional confirmatory
- Excluded (archival only): `arcene*.yaml`, `nfl_bdb.yaml` (see `src/configs/_excluded/README.md`)

### Evaluation
- 3-fold cross-validation (full datasets) or reduced folds per config
- Metrics: Accuracy, ROC-AUC, F1-Score, PR-AUC, Log Loss
- Feature selection stability (Jaccard index)
- Statistical reporting: mean ± std ± 95% CI

---

## 🔧 Advanced Configuration

All settings in `src/configs/*.yaml`:

```yaml
selectors: [variance, anova, l1_logreg, mi, mrmr]
models: [logreg]
k_values: [20, 50, 100]
```

---

## 📊 Output Format

**Aggregated Results** (`results/aggregated/{dataset}_summary.csv`):
```csv
selector,model,k,n_runs,accuracy_mean,accuracy_std,accuracy_ci_lower,accuracy_ci_upper,...
mi,logreg,20,10,0.8995,0.0012,0.8987,0.9003,...
```

**Publication Figures** (`plots/publication/{dataset}_*.png`):
- Performance comparison with error bars
- Model comparison across selectors
- Stability analysis

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🎯 Key Features

✅ **GPU-Only Mode** - Full GPU acceleration, no CPU fallback  
✅ **Simple** - 2-step workflow, clear documentation  
✅ **Fast** - GPU-accelerated MI/MRMR calculations  
✅ **Reproducible** - Fixed seeds, standard methods  
✅ **Flexible** - Easy to add selectors/datasets  
✅ **Research-Grade** - Multi-run experiments with statistical aggregation  
✅ **Tested** - Comprehensive unit tests included

---

## 📝 Citation

If you use this benchmark in your research:

```bibtex
@software{mi_fs_benchmark_2026,
  title = {MI-FS Benchmark: GPU-Accelerated Feature Selection Framework},
  author = {Martin Janev},
  year = {2026},
  url = {https://github.com/yourusername/MI-FS}
}
```

---

## Paper-ready plots and table

To regenerate the publication figures directly from the aggregated CSVs, use the helper script.

Inputs (place alongside your CSVs or point `--root`):
- `santander_summary.csv`, `santander_stability_summary.csv`
- `home_credit_summary.csv`, `home_credit_stability_summary.csv`
- `ieee_cis_fraud_summary.csv`, `ieee_cis_fraud_stability_summary.csv`
- `arcene_summary.csv`, `arcene_stability_summary.csv`

Command (default prefers the logistic regression model if present):

```bash
python -m mi_fs_benchmark.scripts.make_paper_plots --outdir paper_figs --model_preference logreg
```

Outputs (created under `paper_figs/`):
- `fig_perf_<model>.pdf/png`: ROC-AUC vs k with 95% CI (faceted by dataset)
- `fig_stability_<model>.pdf/png`: stability vs k with 95% CI
- `table_best_at_kstar.tex`: compact table; for each selector, k* maximizes mean ROC-AUC and stability is reported at the same k*

If your column names differ, adjust the candidate lists in `_guess_col()` inside `scripts/make_paper_plots.py`.

---

**Last Updated**: January 27, 2026  
**Version**: 2.0 (GPU-Only Mode)  
**License**: MIT

