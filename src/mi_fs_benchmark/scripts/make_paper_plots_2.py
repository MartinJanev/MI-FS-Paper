# scripts/make_paper_plots_mi_based.py
# Drop-in plotting script for aggregated CSVs.
# Fixes:
# 1) per-model predictive performance vs k plots (multi-panel by dataset)
# 2) title/legend overlap (reserved top band + anchored fig.legend)
# 3) k* LaTeX tables regenerated
# 4) includes mRMR as MI-based selector

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Paper ordering / inclusion
# --------------------------
DATASET_ORDER = [
    "santander",
    "home_credit",
    "ieee_cis_fraud",
]

MODEL_ORDER = ["logreg", "hgbt"]

# MI-based + Standard selectors
SELECTOR_ORDER = [
    "variance",
    "anova",
    "l1_logreg",
    "tree_importance",
    "mi",
    "mrmr",
    "boruta",
    "shap",
]

SELECTOR_LABELS = {
    "variance": "Variance",
    "anova": "ANOVA",
    "l1_logreg": "L1-LogReg",
    "tree_importance": "RF-Imp",
    "mi": "MI",
    "mrmr": "mRMR",
    "boruta": "Boruta",
    "shap": "SHAP",
}

# Family-level groupings and colors for MI-vs-standard plots.
FAMILY_DEFS: Dict[str, List[str]] = {
    "mi": ["mi", "mrmr"],
    "standard": ["variance", "anova", "l1_logreg", "tree_importance", "boruta", "shap"],
}

FAMILY_COLORS: Dict[str, str] = {
    "mi": "#1f77b4",
    "standard": "#ff7f0e",
}


@dataclass
class Cols:
    dataset: str
    model: str
    selector: str
    k: str
    metric_mean: str
    metric_std: Optional[str] = None
    metric_ci_lo: Optional[str] = None
    metric_ci_hi: Optional[str] = None


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _infer_columns(df: pd.DataFrame) -> Cols:
    cols = {c: _norm(c) for c in df.columns}

    def find_one(cands: Iterable[str]) -> str:
        for c in df.columns:
            if cols[c] in cands:
                return c
        raise KeyError(f"Missing required column; expected one of {sorted(set(cands))}. Got: {list(df.columns)}")

    dataset = find_one(["dataset", "data", "ds"])
    model = find_one(["model", "clf", "estimator"])
    selector = find_one(["selector", "fs", "feature_selector", "method"])
    k = find_one(["k", "k_features", "n_features", "num_features", "selected_features"])

    # Metric mean
    mean = None
    for c in df.columns:
        lc = cols[c]
        if lc in ("mean", "roc_auc_mean", "score_mean"):
            mean = c
            break
    if mean is None:
        for c in df.columns:
            lc = cols[c]
            if "mean" in lc and ("auc" in lc or "roc" in lc or "score" in lc):
                mean = c
                break
    if mean is None:
        raise KeyError("Could not infer metric mean column (e.g., mean / roc_auc_mean).")

    # Optional: std, CI bounds
    std = None
    for c in df.columns:
        lc = cols[c]
        if lc in ("std", "roc_auc_std", "score_std"):
            std = c
            break

    ci_lo = None
    ci_hi = None
    for c in df.columns:
        lc = cols[c]
        if lc in ("ci_low", "ci_lo", "lower", "lower_ci", "mean_ci_low", "roc_auc_ci_low"):
            ci_lo = c
        if lc in ("ci_high", "ci_hi", "upper", "upper_ci", "mean_ci_high", "roc_auc_ci_high"):
            ci_hi = c

    return Cols(dataset=dataset, model=model, selector=selector, k=k, metric_mean=mean, metric_std=std, metric_ci_lo=ci_lo, metric_ci_hi=ci_hi)

def _drop_nan_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    # Remove rows where the plotted metric is NaN (these break lines into invisible fragments)
    return df.dropna(subset=[metric_col]).copy()


def _infer_stability_columns(df: pd.DataFrame) -> Cols:
    cols = {c: _norm(c) for c in df.columns}

    def find_one(cands: Iterable[str]) -> str:
        for c in df.columns:
            if cols[c] in cands:
                return c
        raise KeyError(f"Missing required column; expected one of {sorted(set(cands))}. Got: {list(df.columns)}")

    dataset = find_one(["dataset", "data", "ds"])
    model = find_one(["model", "clf", "estimator"])
    selector = find_one(["selector", "fs", "feature_selector", "method"])
    k = find_one(["k", "k_features", "n_features", "num_features", "selected_features"])

    mean = None
    for c in df.columns:
        lc = cols[c]
        if lc in ("stability_mean", "mean", "jaccard_mean"):
            mean = c
            break
    if mean is None:
        for c in df.columns:
            lc = cols[c]
            if "mean" in lc and ("stability" in lc or "jaccard" in lc):
                mean = c
                break
    if mean is None:
        raise KeyError("Could not infer stability mean column (e.g., stability_mean).")

    std = None
    for c in df.columns:
        lc = cols[c]
        if lc in ("stability_std", "std", "jaccard_std"):
            std = c
            break

    return Cols(dataset=dataset, model=model, selector=selector, k=k, metric_mean=mean, metric_std=std)


def _canonicalize(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    out = df.copy()
    out[cols.dataset] = out[cols.dataset].astype(str).map(_norm)
    out[cols.model] = out[cols.model].astype(str).map(_norm)
    out[cols.selector] = out[cols.selector].astype(str).map(_norm)
    out[cols.selector] = out[cols.selector].replace({"rf": "tree_importance"})
    out[cols.k] = pd.to_numeric(out[cols.k], errors="coerce")
    out = out.dropna(subset=[cols.k])
    out[cols.k] = out[cols.k].astype(int)
    return out


def _filter_and_sort(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    df = df[df[cols.dataset].isin(DATASET_ORDER)]
    df = df[df[cols.model].isin(MODEL_ORDER)]
    df = df[df[cols.selector].isin(SELECTOR_ORDER)]

    ds_rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    m_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    s_rank = {s: i for i, s in enumerate(SELECTOR_ORDER)}

    df = (
        df.assign(
            _ds=df[cols.dataset].map(ds_rank),
            _m=df[cols.model].map(m_rank),
            _s=df[cols.selector].map(s_rank),
        )
        .sort_values(["_m", "_ds", "_s", cols.k])
        .drop(columns=["_ds", "_m", "_s"])
    )
    return df


def _compute_ci_band(df: pd.DataFrame, cols: Cols, z: float = 1.96):
    y = pd.to_numeric(df[cols.metric_mean], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(y)):
        return y, None, None

    # Prefer explicit CI columns only if they contain at least some non-NaN values
    if cols.metric_ci_lo and cols.metric_ci_hi:
        lo = pd.to_numeric(df[cols.metric_ci_lo], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[cols.metric_ci_hi], errors="coerce").to_numpy(dtype=float)
        if not (np.all(np.isnan(lo)) or np.all(np.isnan(hi))):
            return y, lo, hi

    # Otherwise fall back to std if available and non-NaN
    if cols.metric_std:
        sd = pd.to_numeric(df[cols.metric_std], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isnan(sd)):
            lo = y - z * sd
            hi = y + z * sd
            return y, lo, hi

    return y, None, None


def _resolve_metric_cols(df: pd.DataFrame, metric_prefix: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    mean = f"{metric_prefix}_mean"
    if mean not in df.columns:
        raise KeyError(f"Missing required metric column: {mean}")
    std = f"{metric_prefix}_std" if f"{metric_prefix}_std" in df.columns else None
    ci_lo = f"{metric_prefix}_ci_lower" if f"{metric_prefix}_ci_lower" in df.columns else None
    ci_hi = f"{metric_prefix}_ci_upper" if f"{metric_prefix}_ci_upper" in df.columns else None
    return mean, std, ci_lo, ci_hi


def _compute_ci_band_named(df: pd.DataFrame, mean_col: str, std_col: Optional[str], ci_lo_col: Optional[str], ci_hi_col: Optional[str], z: float = 1.96):
    y = pd.to_numeric(df[mean_col], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(y)):
        return y, None, None

    if ci_lo_col and ci_hi_col:
        lo = pd.to_numeric(df[ci_lo_col], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(df[ci_hi_col], errors="coerce").to_numpy(dtype=float)
        if not (np.all(np.isnan(lo)) or np.all(np.isnan(hi))):
            return y, lo, hi

    if std_col:
        sd = pd.to_numeric(df[std_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isnan(sd)):
            return y, y - z * sd, y + z * sd

    return y, None, None


def _dataset_key(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def _load_dataset_feature_caps(csv_path: Optional[Path]) -> Dict[str, int]:
    if csv_path is None or not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    if "Dataset" not in df.columns or "Features" not in df.columns:
        return {}

    caps: Dict[str, int] = {}
    for _, row in df.iterrows():
        ds_key = _dataset_key(row["Dataset"])
        try:
            caps[ds_key] = int(row["Features"])
        except (TypeError, ValueError):
            continue
    return caps


def _combined_ylim(
    vals: List[float],
    default_ylim: Tuple[float, float] = (0.0, 1.0),
    min_span: float = 0.0,
) -> Tuple[float, float]:
    if vals and np.isfinite(vals).any():
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        span = (y_max - y_min) if y_max > y_min else 0.0
        if min_span > 0 and span < min_span:
            center = 0.5 * (y_min + y_max)
            y_min = center - 0.5 * min_span
            y_max = center + 0.5 * min_span
            span = min_span
        if span <= 0:
            span = 1.0
        # Slightly larger top pad so high-performing curves do not touch the boundary.
        low_pad = 0.02 * span
        high_pad = 0.06 * span
        return (y_min - low_pad, y_max + high_pad)
    return default_ylim



def _reserve_top_band(fig: plt.Figure, top: float = 0.90) -> None:
    # Leaves space for suptitle + legend so they never collide.
    fig.subplots_adjust(top=top)


def _add_suptitle_and_legend(fig: plt.Figure, handles: List, labels: List[str]) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=min(len(labels), 6),
        frameon=False,
    )


def _add_bottom_legend(fig: plt.Figure, handles: List, labels: List[str], y: float = 0.02) -> None:
    # Shared legend just under the axes area (tight, minimal gap)
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=min(len(labels), 5),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )


def _set_local_ylim(ax: plt.Axes, y_min: float, y_max: float, pad: float = 0.02, clip01: bool = False) -> None:
    # Dynamically scale per-axis limits to the min/max of plotted values with small padding.
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if y_max - y_min < 1e-6:
        y_min -= 0.01
        y_max += 0.01
    y_min -= pad
    y_max += pad
    if clip01:
        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)
    ax.set_ylim(y_min, y_max)

def _pad_y_axes(axes, tick_pad: int = 12, label_pad: int = 16) -> None:
    axes_iter = axes.flat if hasattr(axes, "flat") else axes
    for ax in axes_iter:
        ax.tick_params(axis="y", pad=tick_pad)
        ax.yaxis.labelpad = label_pad


def _set_dynamic_ylim(ax, y_min: float, y_max: float, pad_ratio: float = 0.05) -> None:
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    span = y_max - y_min
    pad = span * pad_ratio if span > 0 else max(abs(y_max), 1.0) * pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_performance_vs_k(perf_df: pd.DataFrame, cols: Cols, model: str, outdir: Path, metric_name: str = "ROC-AUC") -> Path:
    sub = perf_df[perf_df[cols.model] == model]

    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(6.5, 4.2 * len(DATASET_ORDER)), sharey=False)
    fig.subplots_adjust(top=0.9, bottom=0.16, hspace=0.35)

    handles: List = []
    labels: List[str] = []

    for j, ds in enumerate(DATASET_ORDER):
        ax = axes[j]

        ds_df = sub[sub[cols.dataset] == ds]

        y_min, y_max = np.inf, -np.inf
        for sel in SELECTOR_ORDER:
            s_df = ds_df[ds_df[cols.selector] == sel]
            if s_df.empty:
                continue

            s_df = _drop_nan_metric(s_df, cols.metric_mean)
            if s_df.empty:
                continue

            x = s_df[cols.k].to_numpy()
            y, lo, hi = _compute_ci_band(s_df, cols)

            line, = ax.plot(x, y, linewidth=2)
            y_min = min(y_min, np.nanmin(lo) if lo is not None else np.nanmin(y))
            y_max = max(y_max, np.nanmax(hi) if hi is not None else np.nanmax(y))
            if lo is not None and hi is not None:
                ax.fill_between(x, lo, hi, alpha=0.15)

            if j == 0:
                handles.append(line)
                labels.append(SELECTOR_LABELS.get(sel, sel))

        _set_local_ylim(ax, y_min, y_max, pad=0.02, clip01=False)

        ax.set_title(ds.replace("_", " ").title(), pad=8)
        ax.set_xlabel("k (selected features)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"{metric_name} (mean ± 95% CI)")

    _add_bottom_legend(fig, handles, labels, y=0.01)

    outpath = outdir / f"fig_perf_{model}_tabular.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_stability_vs_k(stab_df: pd.DataFrame, cols: Cols, model: str, outdir: Path) -> Path:
    sub = stab_df[stab_df[cols.model] == model]

    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(6.5, 4.2 * len(DATASET_ORDER)), sharey=False)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)

    handles: List = []
    labels: List[str] = []

    for j, ds in enumerate(DATASET_ORDER):
        ax = axes[j]
        ds_df = sub[sub[cols.dataset] == ds]

        y_min, y_max = np.inf, -np.inf
        for sel in SELECTOR_ORDER:
            s_df = ds_df[ds_df[cols.selector] == sel]
            if s_df.empty:
                continue

            x = s_df[cols.k].to_numpy()
            y = s_df[cols.metric_mean].to_numpy(dtype=float)

            line, = ax.plot(x, y, linewidth=2)
            if cols.metric_std:
                sd = s_df[cols.metric_std].to_numpy(dtype=float)
                ax.fill_between(x, y - 1.96 * sd, y + 1.96 * sd, alpha=0.15)
            if cols.metric_std:
                sd = s_df[cols.metric_std].to_numpy(dtype=float)
                y_min = min(y_min, np.nanmin(y - 1.96 * sd))
                y_max = max(y_max, np.nanmax(y + 1.96 * sd))
            else:
                y_min = min(y_min, np.nanmin(y))
                y_max = max(y_max, np.nanmax(y))

            if j == 0:
                handles.append(line)
                labels.append(SELECTOR_LABELS.get(sel, sel))

        _set_local_ylim(ax, y_min, y_max, pad=0.02, clip01=False)

        ax.set_title(ds.replace("_", " ").title(), pad=8)
        ax.set_xlabel("k (selected features)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Stability (mean ± 95% CI)")

    _add_bottom_legend(fig, handles, labels, y=0.01)

    outpath = outdir / f"fig_stability_{model}_tabular.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_kstar_table(perf_df: pd.DataFrame, perf_cols: Cols, stab_df: pd.DataFrame, stab_cols: Cols, model: str, outpath: Path, metric_name: str = "ROC-AUC") -> None:
    rows = []

    perf_m = perf_df[perf_df[perf_cols.model] == model]
    stab_m = stab_df[stab_df[stab_cols.model] == model]
    
    # Load statistical significance results if available
    sig_df = None
    sig_path = Path("../../../src/results/aggregated/combined_statistical_significance.csv")
    if sig_path.exists():
        sig_df = pd.read_csv(sig_path)
    else:
        sig_path = Path("results/statistical_significance.csv")
        if sig_path.exists():
            sig_df = pd.read_csv(sig_path)

    for ds in DATASET_ORDER:
        for sel in SELECTOR_ORDER:
            s_perf = perf_m[(perf_m[perf_cols.dataset] == ds) & (perf_m[perf_cols.selector] == sel)]
            if s_perf.empty:
                continue

            idx = int(s_perf[perf_cols.metric_mean].astype(float).idxmax())
            best = perf_df.loc[idx]
            kstar = int(best[perf_cols.k])
            auc = float(best[perf_cols.metric_mean])
            
            # Extract confidence intervals if available
            ci_lower = None
            ci_upper = None
            if perf_cols.metric_ci_lo and perf_cols.metric_ci_lo in best.index and pd.notna(best[perf_cols.metric_ci_lo]):
                ci_lower = float(best[perf_cols.metric_ci_lo])
            if perf_cols.metric_ci_hi and perf_cols.metric_ci_hi in best.index and pd.notna(best[perf_cols.metric_ci_hi]):
                ci_upper = float(best[perf_cols.metric_ci_hi])

            s_stab = stab_m[
                (stab_m[stab_cols.dataset] == ds)
                & (stab_m[stab_cols.selector] == sel)
                & (stab_m[stab_cols.k] == kstar)
            ]
            stab = float(s_stab[stab_cols.metric_mean].iloc[0]) if not s_stab.empty else np.nan
            
            # Check significance
            is_significant = False
            pval_str = ""
            if sig_df is not None:
                sig_match = sig_df[(sig_df["dataset"] == ds) & (sig_df["model"] == model) & (sig_df["k"] == kstar) & (sig_df["selector"] == sel)]
                if not sig_match.empty:
                    if "t_test_p" in sig_match.columns and pd.notna(sig_match["t_test_p"].iloc[0]):
                        pval_str = f" ($p={float(sig_match['t_test_p'].iloc[0]):.3f}$)"
                    if sig_match["significant_wilcoxon"].iloc[0]:
                        is_significant = True

            rows.append((ds, sel, kstar, auc, stab, ci_lower, ci_upper, is_significant, pval_str))

    def tex_escape(s: str) -> str:
        return s.replace("_", "\\_")

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append(f"Dataset & Selector & $k^\\star$ & {metric_name} & Stability \\\\")
    lines.append("\\midrule")

    for ds in DATASET_ORDER:
        ds_rows = [r for r in rows if r[0] == ds]
        if not ds_rows:
            continue

        for (ds_, sel, kstar, auc, stab, ci_lower, ci_upper, is_significant, pval_str) in ds_rows:
            ds_name = tex_escape(ds_.replace("_", " ").title())
            sel_name = tex_escape(SELECTOR_LABELS.get(sel, sel))
            
            auc_s = f"{auc:.3f}"
            if ci_lower is not None and ci_upper is not None:
                # Assuming symmetric CI for display, or show standard error equivalent
                err = (ci_upper - ci_lower) / 2
                auc_s = f"${auc:.3f} \\pm {err:.3f}$"
            
            if is_significant:
                auc_s += "$^*$"
                
            if pval_str:
                sel_name += pval_str
                
            stab_s = "--" if np.isnan(stab) else f"{stab:.3f}"
            lines.append(f"{ds_name} & {sel_name} & {kstar:d} & {auc_s} & {stab_s} \\\\")
        lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Best performance at $k^\\star$ for model {tex_escape(model)}; stability reported at the same $k^\\star$. * indicates statistically significant improvement over best baseline ($p < 0.05$).}}")
    lines.append(f"\\label{{tab:kstar_{tex_escape(model)}}}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_performance_combined(
    perf_df: pd.DataFrame,
    cols: Cols,
    outdir: Path,
    metric_name: str = "ROC-AUC",
    dataset_feature_caps: Optional[Dict[str, int]] = None,
) -> Path:
    """Grid with rows=datasets and cols=models; one legend at top."""
    if perf_df.empty:
        return outdir / "fig_perf_tabular_combined.png"

    # rows=datasets, cols=models
    nrows, ncols = len(DATASET_ORDER), len(MODEL_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=False)
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.35, wspace=0.25)
    axes_arr = np.array(axes).reshape(nrows, ncols)
    handles: List = []
    labels: List[str] = []

    for i, ds in enumerate(DATASET_ORDER):
        ds_df = perf_df[perf_df[cols.dataset] == ds]
        # per-dataset ylim across models
        vals: List[float] = []
        for model in MODEL_ORDER:
            s = ds_df[ds_df[cols.model] == model]
            if s.empty:
                continue
            y, lo, hi = _compute_ci_band(s, cols)
            if lo is not None and hi is not None:
                vals += [np.nanmin(lo), np.nanmax(hi)]
            else:
                vals += [np.nanmin(y), np.nanmax(y)]
        ylim = _combined_ylim(vals, default_ylim=(0.0, 1.0))

        for j, model in enumerate(MODEL_ORDER):
            ax = axes_arr[i, j]
            s_df = ds_df[ds_df[cols.model] == model]
            if s_df.empty:
                ax.set_title(f"{ds.replace('_', ' ').title()} — {model} (no data)", pad=8)
                ax.axis("off")
                continue

            for sel in SELECTOR_ORDER:
                ss = s_df[s_df[cols.selector] == sel]
                if ss.empty:
                    continue
                ss = _drop_nan_metric(ss, cols.metric_mean)
                if ss.empty:
                    continue
                x = ss[cols.k].to_numpy()
                y, lo, hi = _compute_ci_band(ss, cols)
                line, = ax.plot(x, y, linewidth=2)
                if lo is not None and hi is not None:
                    ax.fill_between(x, lo, hi, alpha=0.15)
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(SELECTOR_LABELS.get(sel, sel))

            ax.set_title(f"{ds.replace('_', ' ').title()} — {model}", pad=8)
            if i == nrows - 1:
                ax.set_xlabel("k (selected features)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*ylim)
            cap = (dataset_feature_caps or {}).get(ds)
            if cap is not None and cap < 150:
                ax.set_xlim(right=float(cap))
            if j == 0:
                ax.set_ylabel(f"{metric_name} (mean ± 95% CI)")

    _add_bottom_legend(fig, handles, labels, y=0.02)

    out_png = outdir / "fig_perf_tabular_combined.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_metric_combined_all_selectors(
    perf_df: pd.DataFrame,
    cols: Cols,
    outdir: Path,
    *,
    metric_prefix: str,
    metric_name: str,
    out_name: str,
    dataset_feature_caps: Optional[Dict[str, int]] = None,
    per_subplot_ylim: bool = False,
    shared_ylim_all: bool = False,
    min_ylim_span: float = 0.0,
) -> Path:
    if perf_df.empty:
        return outdir / out_name

    mean_col, std_col, ci_lo_col, ci_hi_col = _resolve_metric_cols(perf_df, metric_prefix)

    nrows, ncols = len(DATASET_ORDER), len(MODEL_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=False)
    fig.subplots_adjust(top=0.92, bottom=0.16, hspace=0.35, wspace=0.25)
    axes_arr = np.array(axes).reshape(nrows, ncols)
    handles: List = []
    labels: List[str] = []

    global_ylim: Optional[Tuple[float, float]] = None
    if shared_ylim_all:
        vals_all: List[float] = []
        for _, s in perf_df.groupby([cols.dataset, cols.model]):
            if s.empty:
                continue
            y, lo, hi = _compute_ci_band_named(s, mean_col, std_col, ci_lo_col, ci_hi_col)
            if lo is not None and hi is not None:
                vals_all += [np.nanmin(lo), np.nanmax(hi)]
            else:
                vals_all += [np.nanmin(y), np.nanmax(y)]
        global_ylim = _combined_ylim(vals_all, default_ylim=(0.0, 1.0), min_span=min_ylim_span)

    for i, ds in enumerate(DATASET_ORDER):
        ds_df = perf_df[perf_df[cols.dataset] == ds]
        vals: List[float] = []
        for model in MODEL_ORDER:
            s = ds_df[ds_df[cols.model] == model]
            if s.empty:
                continue
            y, lo, hi = _compute_ci_band_named(s, mean_col, std_col, ci_lo_col, ci_hi_col)
            if lo is not None and hi is not None:
                vals += [np.nanmin(lo), np.nanmax(hi)]
            else:
                vals += [np.nanmin(y), np.nanmax(y)]

        ylim = _combined_ylim(vals, default_ylim=(0.0, 1.0), min_span=min_ylim_span)

        for j, model in enumerate(MODEL_ORDER):
            ax = axes_arr[i, j]
            s_df = ds_df[ds_df[cols.model] == model]
            if s_df.empty:
                ax.set_title(f"{ds.replace('_', ' ').title()} - {model} (no data)", pad=8)
                ax.axis("off")
                continue

            for sel in SELECTOR_ORDER:
                ss = s_df[s_df[cols.selector] == sel]
                if ss.empty:
                    continue
                ss = _drop_nan_metric(ss, mean_col)
                if ss.empty:
                    continue

                x = ss[cols.k].to_numpy()
                y, lo, hi = _compute_ci_band_named(ss, mean_col, std_col, ci_lo_col, ci_hi_col)
                line, = ax.plot(x, y, linewidth=2)
                if lo is not None and hi is not None:
                    ax.fill_between(x, lo, hi, alpha=0.15)
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(SELECTOR_LABELS.get(sel, sel))

            if per_subplot_ylim:
                vals_local: List[float] = []
                for sel in SELECTOR_ORDER:
                    ss = s_df[s_df[cols.selector] == sel]
                    if ss.empty:
                        continue
                    ss = _drop_nan_metric(ss, mean_col)
                    if ss.empty:
                        continue
                    y, lo, hi = _compute_ci_band_named(ss, mean_col, std_col, ci_lo_col, ci_hi_col)
                    if lo is not None and hi is not None:
                        vals_local += [np.nanmin(lo), np.nanmax(hi)]
                    else:
                        vals_local += [np.nanmin(y), np.nanmax(y)]
                ylim_model = _combined_ylim(vals_local, default_ylim=(0.0, 1.0), min_span=min_ylim_span)
            elif global_ylim is not None:
                ylim_model = global_ylim
            else:
                ylim_model = ylim

            ax.set_title(f"{ds.replace('_', ' ').title()} - {model}", pad=8)
            if i == nrows - 1:
                ax.set_xlabel("k (selected features)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*ylim_model)
            cap = (dataset_feature_caps or {}).get(ds)
            if cap is not None and cap < 150:
                ax.set_xlim(right=float(cap))
            if j == 0:
                ax.set_ylabel(f"{metric_name} (mean ± 95% CI)")

    _add_bottom_legend(fig, handles, labels, y=0.03)

    out_png = outdir / out_name
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _extract_kstar_rows_combined(table_source_df: pd.DataFrame) -> pd.DataFrame:
    if table_source_df.empty:
        return table_source_df

    req = ["dataset", "selector", "model", "k", "roc_auc_mean"]
    missing = [c for c in req if c not in table_source_df.columns]
    if missing:
        raise KeyError(f"combined_table_source is missing columns: {missing}")

    sub = table_source_df.copy()
    sub["_avg_auc"] = sub.groupby(["dataset", "selector", "k"])["roc_auc_mean"].transform("mean")
    idx = sub.groupby(["dataset", "selector"])["_avg_auc"].idxmax()
    kstars = sub.loc[idx, ["dataset", "selector", "k"]].drop_duplicates().rename(columns={"k": "k_star"})

    out = table_source_df.merge(
        kstars,
        on=["dataset", "selector"],
        how="inner",
    )
    out = out[out["k"] == out["k_star"]].copy()
    return out


def plot_family_metric_combined(
    perf_df: pd.DataFrame,
    cols: Cols,
    outdir: Path,
    *,
    metric_prefix: str,
    metric_name: str,
    out_name: str,
    dataset_feature_caps: Optional[Dict[str, int]] = None,
) -> Path:
    if perf_df.empty:
        return outdir / out_name

    mean_col, _, ci_lo_col, ci_hi_col = _resolve_metric_cols(perf_df, metric_prefix)

    rows = []
    for fam, sels in FAMILY_DEFS.items():
        sub = perf_df[perf_df[cols.selector].isin(sels)]
        if sub.empty:
            continue

        group_cols = [cols.dataset, cols.model, cols.k]
        agg = sub.groupby(group_cols, as_index=False).agg(family_mean=(mean_col, "mean"))
        if ci_lo_col and ci_hi_col and ci_lo_col in sub.columns and ci_hi_col in sub.columns:
            ci_df = sub.groupby(group_cols, as_index=False).agg(
                family_ci_lo=(ci_lo_col, "min"),
                family_ci_hi=(ci_hi_col, "max"),
            )
            agg = agg.merge(ci_df, on=group_cols, how="left")
        else:
            agg["family_ci_lo"] = np.nan
            agg["family_ci_hi"] = np.nan

        agg["family"] = fam
        rows.append(agg)

    if not rows:
        return outdir / out_name

    fam_df = pd.concat(rows, ignore_index=True)

    nrows, ncols = len(DATASET_ORDER), len(MODEL_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=False)
    fig.subplots_adjust(top=0.92, bottom=0.16, hspace=0.35, wspace=0.25)
    axes_arr = np.array(axes).reshape(nrows, ncols)
    handles: List = []
    labels: List[str] = []

    for i, ds in enumerate(DATASET_ORDER):
        ds_df = fam_df[fam_df[cols.dataset] == ds]
        vals: List[float] = []
        for model in MODEL_ORDER:
            s = ds_df[ds_df[cols.model] == model]
            if s.empty:
                continue
            vals += [float(np.nanmin(s["family_mean"])), float(np.nanmax(s["family_mean"]))]
        ylim = _combined_ylim(vals, default_ylim=(0.0, 1.0), min_span=0.003)

        for j, model in enumerate(MODEL_ORDER):
            ax = axes_arr[i, j]
            s_df = ds_df[ds_df[cols.model] == model]
            if s_df.empty:
                ax.set_title(f"{ds.replace('_', ' ').title()} - {model} (no data)", pad=8)
                ax.axis("off")
                continue

            for fam in ["mi", "standard"]:
                ss = s_df[s_df["family"] == fam].sort_values(cols.k)
                if ss.empty:
                    continue
                line, = ax.plot(
                    ss[cols.k].to_numpy(),
                    ss["family_mean"].to_numpy(),
                    linewidth=2.2,
                    color=FAMILY_COLORS.get(fam),
                    label=fam.upper() if fam == "mi" else "Standard",
                )
                if i == 0 and j == 0:
                    handles.append(line)
                    labels.append(fam.upper() if fam == "mi" else "Standard")

            ax.set_title(f"{ds.replace('_', ' ').title()} - {model}", pad=8)
            if i == nrows - 1:
                ax.set_xlabel("k (selected features)")
            ax.set_ylabel(f"{metric_name} (family mean)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*ylim)
            cap = (dataset_feature_caps or {}).get(ds)
            if cap is not None and cap < 150:
                ax.set_xlim(right=float(cap))

    _add_bottom_legend(fig, handles, labels, y=0.03)
    out_png = outdir / out_name
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_efficiency_frontier_kstar(kstar_rows: pd.DataFrame, outdir: Path) -> Path:
    if kstar_rows.empty:
        return outdir / "fig_efficiency_frontier_kstar.png"

    if "fit_time_selector_mean" not in kstar_rows.columns or "roc_auc_mean" not in kstar_rows.columns:
        raise KeyError("Need 'fit_time_selector_mean' and 'roc_auc_mean' columns in combined_table_source")

    fig, axes = plt.subplots(len(DATASET_ORDER), len(MODEL_ORDER), figsize=(9.2, 9.6), sharex=False, sharey=False)
    axes_arr = np.array(axes).reshape(len(DATASET_ORDER), len(MODEL_ORDER))

    for i, ds in enumerate(DATASET_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            ax = axes_arr[i, j]
            sub = kstar_rows[(kstar_rows["dataset"] == ds) & (kstar_rows["model"] == model)]
            if sub.empty:
                ax.axis("off")
                continue

            for sel in SELECTOR_ORDER:
                ss = sub[sub["selector"] == sel]
                if ss.empty:
                    continue
                ax.scatter(
                    ss["fit_time_selector_mean"],
                    ss["roc_auc_mean"],
                    label=SELECTOR_LABELS.get(sel, sel),
                    s=52,
                    alpha=0.9,
                )

            ax.set_title(f"{ds.replace('_', ' ').title()} - {model}")
            ax.set_xlabel("Selector runtime (s)")
            ax.set_ylabel("ROC-AUC at k*")
            ax.grid(True, alpha=0.25)

    handles, labels = axes_arr[0, 0].get_legend_handles_labels()
    _add_bottom_legend(fig, handles, labels, y=0.03)
    fig.subplots_adjust(bottom=0.19, hspace=0.38, wspace=0.28)

    out_png = outdir / "fig_efficiency_frontier_kstar.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def make_kstar_summary_reviewer_table(kstar_rows: pd.DataFrame, outpath: Path) -> None:
    if kstar_rows.empty:
        return

    def tex_escape(s: str) -> str:
        return str(s).replace("_", "\\_")

    def fmt_with_ci(mean_v: float, lo_v: float, hi_v: float, *, decimals: int = 5) -> str:
        if np.isfinite(lo_v) and np.isfinite(hi_v):
            err = (hi_v - lo_v) / 2.0
            return f"${mean_v:.{decimals}f} \\pm {err:.{decimals}f}$"
        return f"{mean_v:.{decimals}f}"

    def extract_metric_row(row: pd.DataFrame, metric: str, decimals: int = 5) -> str:
        if row.empty:
            return "--"
        mean_col = f"{metric}_mean"
        if mean_col not in row.columns or pd.isna(row[mean_col].iloc[0]):
            return "--"
        m = float(row[mean_col].iloc[0])
        lo_col = f"{metric}_ci_lower"
        hi_col = f"{metric}_ci_upper"
        lo = float(row[lo_col].iloc[0]) if lo_col in row.columns and pd.notna(row[lo_col].iloc[0]) else np.nan
        hi = float(row[hi_col].iloc[0]) if hi_col in row.columns and pd.notna(row[hi_col].iloc[0]) else np.nan
        return fmt_with_ci(m, lo, hi, decimals=decimals)

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Peak performance and stability at the optimal subset size $k^\\star$ with confidence intervals for AUC, F1, and log-loss.}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\begin{tabular}{p{1.25cm}lrrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Selector & $k^\\star$ & AUC$_{\\text{L}}$ & AUC$_{\\text{H}}$ & F1$_{\\text{L}}$ & F1$_{\\text{H}}$ & LogLoss$_{\\text{L}}$ & LogLoss$_{\\text{H}}$ & Stability & Sig. (L, H) \\\\")
    lines.append("\\midrule")

    for ds in DATASET_ORDER:
        ds_rows = kstar_rows[kstar_rows["dataset"] == ds]
        selectors_present = [s for s in SELECTOR_ORDER if not ds_rows[ds_rows["selector"] == s].empty]
        if not selectors_present:
            continue

        lines.append(f"\\multirow[t]{{{len(selectors_present)}}}{{1.4cm}}{{{tex_escape(ds.replace('_', ' ').title())}}}")
        first = True
        for sel in selectors_present:
            g = ds_rows[ds_rows["selector"] == sel].copy()
            k_star = int(g["k_star"].iloc[0])

            row_l = g[g["model"] == "logreg"].head(1)
            row_h = g[g["model"] == "hgbt"].head(1)

            stab_vals = pd.to_numeric(g.get("stability_mean", np.nan), errors="coerce")
            stab = float(np.nanmean(stab_vals.to_numpy(dtype=float))) if np.isfinite(stab_vals.to_numpy(dtype=float)).any() else np.nan

            def get_pval(r):
                if "t_test_p" in r.columns and not r.empty and pd.notna(r["t_test_p"].iloc[0]):
                    return f"{float(r['t_test_p'].iloc[0]):.3f}"
                return "--"

            sig_l = get_pval(row_l)
            sig_h = get_pval(row_h)
            
            sig_vals = pd.Series(False, index=g.index)
            if "is_significant_primary" in g.columns:
                sig_vals = g["is_significant_primary"].astype("boolean").fillna(False).astype(bool)
            
            star = ""
            if bool(sig_vals.any()):
                star = "$^*$"
                
            if sig_l == "--" and sig_h == "--":
                sig = f"--{star}"
            else:
                sig = f"{sig_l}, {sig_h}{star}"

            prefix = "& " if first else "& "
            first = False
            auc_l_s = extract_metric_row(row_l, "roc_auc", decimals=5)
            auc_h_s = extract_metric_row(row_h, "roc_auc", decimals=5)
            f1_l_s = extract_metric_row(row_l, "f1", decimals=6)
            f1_h_s = extract_metric_row(row_h, "f1", decimals=6)
            ll_l_s = extract_metric_row(row_l, "log_loss", decimals=5)
            ll_h_s = extract_metric_row(row_h, "log_loss", decimals=5)
            stab_s = f"{stab:.5f}" if np.isfinite(stab) else "--"
            lines.append(
                f"{prefix}{tex_escape(SELECTOR_LABELS.get(sel, sel))} & {k_star:d} & {auc_l_s} & {auc_h_s} & {f1_l_s} & {f1_h_s} & {ll_l_s} & {ll_h_s} & {stab_s} & {sig} \\\\")

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:kstar_summary_reviewer}")
    lines.append("\\end{table}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, default="combined_summary.csv")
    ap.add_argument("--stability_csv", type=str, default="combined_stability_summary.csv")
    ap.add_argument("--table_source_csv", type=str, default="combined_table_source.csv")
    ap.add_argument("--dataset_desc_csv", type=str, default=None)
    ap.add_argument(
        "--ci_metrics",
        type=str,
        default="accuracy,f1,log_loss,stability_jaccard",
        help="Comma-separated metric prefixes to render as combined CI plots when present",
    )
    ap.add_argument("--outdir", type=str, default="paper_figs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(args.summary_csv)
    stab = pd.read_csv(args.stability_csv)
    table_source_path = Path(args.table_source_csv)
    table_source_df = pd.read_csv(table_source_path) if table_source_path.exists() else pd.DataFrame()
    if args.dataset_desc_csv:
        dataset_desc_path = Path(args.dataset_desc_csv)
    else:
        dataset_desc_path = Path(__file__).resolve().parents[1] / "data" / "dataset_descriptions.csv"
    dataset_feature_caps = _load_dataset_feature_caps(dataset_desc_path)

    perf_cols = _infer_columns(perf)
    stab_cols = _infer_stability_columns(stab)

    perf = _canonicalize(perf, perf_cols)
    stab = _canonicalize(stab, stab_cols)

    perf = _filter_and_sort(perf, perf_cols)
    stab = _filter_and_sort(stab, stab_cols)

    metric_title_map: Dict[str, str] = {
        "accuracy": "Accuracy",
        "roc_auc": "ROC-AUC",
        "f1": "F1-score",
        "log_loss": "Log-loss",
        "stability_jaccard": "Stability (Jaccard)",
    }

    requested_ci_metrics = [m.strip() for m in args.ci_metrics.split(",") if m.strip()]

    for m in MODEL_ORDER:
        plot_performance_vs_k(perf, perf_cols, m, outdir)
        plot_stability_vs_k(stab, stab_cols, m, outdir)
        make_kstar_table(
            perf_df=perf,
            perf_cols=perf_cols,
            stab_df=stab,
            stab_cols=stab_cols,
            model=m,
            outpath=outdir / f"table_best_at_kstar_{m}_tabular.tex",
        )

    # Combined performance grid across models x datasets
    plot_performance_combined(perf, perf_cols, outdir, dataset_feature_caps=dataset_feature_caps)

    # Combined accuracy grid across models x datasets (all selectors)
    plot_metric_combined_all_selectors(
        perf,
        perf_cols,
        outdir,
        metric_prefix="accuracy",
        metric_name="Accuracy",
        out_name="fig_accuracy_tabular_combined.png",
        dataset_feature_caps=dataset_feature_caps,
        per_subplot_ylim=True,
        shared_ylim_all=False,
        min_ylim_span=0.0001,
    )

    # Family-level MI vs standard plot (combined across selectors)
    if "accuracy_mean" in perf.columns:
        plot_family_metric_combined(
            perf,
            perf_cols,
            outdir,
            metric_prefix="accuracy",
            metric_name="Accuracy",
            out_name="fig_family_accuracy_combined.png",
            dataset_feature_caps=dataset_feature_caps,
        )

    # Additional combined CI plots for available metrics (f1, log-loss, stability_jaccard, ...)
    for metric_prefix in requested_ci_metrics:
        if metric_prefix == "accuracy":
            # Already produced above with stable filename used in manuscript workflow.
            continue
        if f"{metric_prefix}_mean" not in perf.columns:
            continue
        safe_prefix = metric_prefix.replace("-", "_")
        plot_metric_combined_all_selectors(
            perf,
            perf_cols,
            outdir,
            metric_prefix=metric_prefix,
            metric_name=metric_title_map.get(metric_prefix, metric_prefix.replace("_", " ").title()),
            out_name=f"fig_{safe_prefix}_tabular_combined.png",
            dataset_feature_caps=dataset_feature_caps,
            per_subplot_ylim=(metric_prefix == "f1"),
        )

    # Reviewer-focused additions based on per-selector k* rows
    if not table_source_df.empty:
        kstar_rows = _extract_kstar_rows_combined(table_source_df)
        plot_efficiency_frontier_kstar(kstar_rows, outdir)
        make_kstar_summary_reviewer_table(kstar_rows, outdir / "table_kstar_summary_reviewer.tex")


if __name__ == "__main__":
    main()
