#!/usr/bin/env python3
"""
Paper-grade plotting for MI vs standard feature selection results.

This script is designed for the "MI vs Standard Feature Selection" paper workflow:
- It consumes the *aggregated* outputs (either the per-dataset pairs OR the "combined_*" files).
- It produces compact, publication-ready multi-panel figures with consistent axes and legends.

Default paper settings (can be overridden via CLI):
- Datasets: Santander, Home Credit, IEEE-CIS Fraud
- Models: logreg (linear baseline), hgbt (tree ensemble baseline)
- Selectors: variance, anova, l1_logreg, tree_importance, mi
- Metrics: ROC-AUC and stability (mean Jaccard)

Outputs:
- fig_roc_auc_<model>.pdf/.png
- fig_stability_<model>.pdf/.png
- aa_fixed.pdf (single PDF that concatenates all figures above)

Usage (most common):
  python make_paper_plots.py --summary_csv combined_summary.csv --stability_csv combined_stability_summary.csv --outdir paper_figs

If per-dataset files exist (e.g., santander_summary.csv, santander_stability_summary.csv), you can omit --summary_csv/--stability_csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

# ----------------------------- Canonical names -----------------------------

DEFAULT_DATASETS = ["santander", "home_credit", "ieee_cis_fraud"]
DEFAULT_MODELS = ["logreg", "hgbt"]
DEFAULT_SELECTORS = ["mi", "anova", "l1_logreg", "tree_importance", "variance"]
FAMILY_DEFS = {
    "mi": ["mi"],
    "standard": ["anova", "l1_logreg", "tree_importance", "variance"],
}

DATASET_TITLES: Dict[str, str] = {
    "santander": "Santander",
    "home_credit": "Home Credit",
    "ieee_cis_fraud": "IEEE-CIS Fraud",
    "arcene": "ARCENE",
}

SELECTOR_LABELS: Dict[str, str] = {
    "mi": "MI (univariate)",
    "anova": "ANOVA F",
    "l1_logreg": "L1-LogReg",
    "tree_importance": "RF importance",
    "variance": "Variance",
}

MODEL_LABELS: Dict[str, str] = {
    "logreg": "logreg",
    "hgbt": "hgbt",
    "lgbm": "lgbm",
}

# Color mapping for family-level plots
FAMILY_COLORS: Dict[str, str] = {
    "mi": "#1f77b4",       # blue
    "standard": "#ff7f0e", # orange
}
# Distinct colors when plotting both models together
FAMILY_MODEL_COLORS: Dict[Tuple[str, str], str] = {
    ("mi", "logreg"): "#1f77b4",
    ("mi", "hgbt"): "#145a86",
    ("standard", "logreg"): "#ff7f0e",
    ("standard", "hgbt"): "#c85a00",
}


# Correlation methods allowed for ranking heatmaps
RANK_METHODS = {"spearman", "kendall"}


def _save_figure(fig: plt.Figure, out_pdf: Optional[Path], out_png: Optional[Path]) -> None:
    """Save figure to requested output formats, creating parent directories as needed."""
    target = out_png or out_pdf
    if target is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")


# ----------------------------- Loading helpers -----------------------------

def _find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_combined(summary_csv: Path, stability_csv: Optional[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = pd.read_csv(summary_csv)
    if stability_csv is None:
        # some setups keep stability in the same file; we still try to load the explicit stability file first
        raise FileNotFoundError("stability_csv not provided and could not be inferred.")
    stability_df = pd.read_csv(stability_csv)
    return summary_df, stability_df


def _load_per_dataset(input_dir: Path, datasets: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    perf_frames = []
    stab_frames = []
    for ds in datasets:
        p = input_dir / f"{ds}_summary.csv"
        s = input_dir / f"{ds}_stability_summary.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "dataset" not in {c.lower() for c in df.columns}:
                df["dataset"] = ds
            perf_frames.append(df)
        if s.exists():
            df = pd.read_csv(s)
            if "dataset" not in {c.lower() for c in df.columns}:
                df["dataset"] = ds
            stab_frames.append(df)
    if not perf_frames or not stab_frames:
        raise FileNotFoundError(f"Could not find per-dataset aggregated files in {input_dir}")
    return pd.concat(perf_frames, ignore_index=True), pd.concat(stab_frames, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ----------------------------- Plotting helpers -----------------------------

def _panel_1x3(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    selectors: Sequence[str],
    model: str,
    y_mean: str,
    y_lo: str,
    y_hi: str,
    title: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    ylim_default: Tuple[float, float],
) -> None:
    ds_list = list(datasets)
    n = len(ds_list)
    if n <= 3:
        nrows, ncols = 1, n
        figsize = (11.0, 3.2)
    else:
        nrows, ncols = 2, 2
        figsize = (10.5, 6.4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes_arr = np.array(axes).reshape(-1)

    sub_all = df[(df["dataset"].isin(ds_list)) & (df["model"] == model) & (df["selector"].isin(selectors))].copy()

    vals: List[float] = []
    for ds in ds_list:
        s = sub_all[sub_all["dataset"] == ds]
        if s.empty:
            continue
        lo = s[y_lo].to_numpy(dtype=float) if y_lo in s else np.full(len(s), np.nan)
        hi = s[y_hi].to_numpy(dtype=float) if y_hi in s else np.full(len(s), np.nan)
        mean = s[y_mean].to_numpy(dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            vals += [np.nanmin(lo), np.nanmax(hi)]
        else:
            vals += [np.nanmin(mean), np.nanmax(mean)]

    if vals and np.isfinite(vals).any():
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
        ylim = (y_min - pad, y_max + pad)
    else:
        ylim = ylim_default

    for ax, ds in zip(axes_arr, ds_list):
        sub = sub_all[sub_all["dataset"] == ds]
        if sub.empty:
            ax.axis("off")
            continue

        # Only plot selectors present for this dataset/model
        sel_present = [sel for sel in selectors if not sub[sub["selector"] == sel].empty]
        for sel in sel_present:
            ss = sub[sub["selector"] == sel].sort_values("k")
            x = ss["k"].to_numpy(dtype=float)
            y = ss[y_mean].to_numpy(dtype=float)
            ax.plot(x, y, label=SELECTOR_LABELS.get(sel, sel), linewidth=1.8)

            lo = ss[y_lo].to_numpy(dtype=float) if y_lo in ss else np.full(len(ss), np.nan)
            hi = ss[y_hi].to_numpy(dtype=float) if y_hi in ss else np.full(len(ss), np.nan)
            if np.isfinite(lo).any() and np.isfinite(hi).any() and np.nanmax(hi - lo) > 0:
                ax.fill_between(x, lo, hi, alpha=0.15)

        ax.set_xlabel("Selected features k")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(*ylim)

    # turn off any unused axes
    for ax in axes_arr[len(ds_list):]:
        ax.axis("off")

    axes_arr[0].set_ylabel(y_label)

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.12))

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _panel_models_datasets(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    selectors: Sequence[str],
    models: Sequence[str],
    y_mean: str,
    y_lo: str,
    y_hi: str,
    title: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    ylim_default: Tuple[float, float],
) -> None:
    nrows, ncols = len(models), len(datasets)
    if nrows == 0 or ncols == 0:
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    handles_ref: List = []
    labels_ref: List[str] = []

    sub_all = df[(df["dataset"].isin(datasets)) & (df["model"].isin(models)) & (df["selector"].isin(selectors))].copy()

    for i, model in enumerate(models):
        model_df = sub_all[sub_all["model"] == model]
        vals: List[float] = []
        for ds in datasets:
            s = model_df[model_df["dataset"] == ds]
            if s.empty:
                continue
            lo = s[y_lo].to_numpy(dtype=float) if y_lo in s else np.full(len(s), np.nan)
            hi = s[y_hi].to_numpy(dtype=float) if y_hi in s else np.full(len(s), np.nan)
            mean = s[y_mean].to_numpy(dtype=float)
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                vals += [np.nanmin(lo), np.nanmax(hi)]
            else:
                vals += [np.nanmin(mean), np.nanmax(mean)]
        if vals and np.isfinite(vals).any():
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
            ylim = (y_min - pad, y_max + pad)
        else:
            ylim = ylim_default

        for j, ds in enumerate(datasets):
            ax = axes_arr[i, j]
            ds_df = model_df[model_df["dataset"] == ds]
            if ds_df.empty:
                ax.axis("off")
                continue
            sel_present = [sel for sel in selectors if not ds_df[ds_df["selector"] == sel].empty]
            for sel in sel_present:
                ss = ds_df[ds_df["selector"] == sel].sort_values("k")
                x = ss["k"].to_numpy(dtype=float)
                y = ss[y_mean].to_numpy(dtype=float)
                line, = ax.plot(x, y, label=SELECTOR_LABELS.get(sel, sel), linewidth=1.8)
                lo = ss[y_lo].to_numpy(dtype=float) if y_lo in ss else np.full(len(ss), np.nan)
                hi = ss[y_hi].to_numpy(dtype=float) if y_hi in ss else np.full(len(ss), np.nan)
                if np.isfinite(lo).any() and np.isfinite(hi).any() and np.nanmax(hi - lo) > 0:
                    ax.fill_between(x, lo, hi, alpha=0.15)
                if i == 0 and j == 0:
                    handles_ref.append(line)
                    labels_ref.append(SELECTOR_LABELS.get(sel, sel))

            if i == nrows - 1:
                ax.set_xlabel("Selected features k")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(y_label)

    if handles_ref:
        fig.legend(handles_ref, labels_ref, loc="upper center", ncol=min(6, len(labels_ref)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _family_panel(
    df: pd.DataFrame,
    *,
    families: Dict[str, Sequence[str]],
    model: str,
    y_mean: str,
    y_lo: str,
    y_hi: str,
    title: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    ylim_default: Tuple[float, float],
) -> None:
    """Aggregate across datasets: one panel, one line per family (e.g., MI vs Standard)."""
    sub = df[(df["model"] == model)].copy()
    if sub.empty:
        return

    agg_rows = []
    for fam_name, sel_list in families.items():
        fam_df = sub[sub["selector"].isin(sel_list)]
        if fam_df.empty:
            continue
        for k in sorted(fam_df["k"].dropna().unique()):
            kdf = fam_df[fam_df["k"] == k]
            if kdf.empty:
                continue
            mean_val = kdf[y_mean].mean()
            lo_val = kdf[y_lo].min() if y_lo in kdf else np.nan
            hi_val = kdf[y_hi].max() if y_hi in kdf else np.nan
            agg_rows.append({"family": fam_name, "k": k, y_mean: mean_val, y_lo: lo_val, y_hi: hi_val})
    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

    vals: List[float] = []
    for _, row in agg.iterrows():
        vals.extend([row[y_mean]])
        if np.isfinite(row[y_lo]):
            vals.append(row[y_lo])
        if np.isfinite(row[y_hi]):
            vals.append(row[y_hi])
    if vals and np.isfinite(vals).any():
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
        ylim = (y_min - pad, y_max + pad)
    else:
        ylim = ylim_default

    for fam_name, sel_list in families.items():
        ss = agg[agg["family"] == fam_name].sort_values("k")
        if ss.empty:
            continue
        x = ss["k"].to_numpy(dtype=float)
        y = ss[y_mean].to_numpy(dtype=float)
        label = fam_name.upper() if fam_name.lower() == "mi" else fam_name.capitalize()
        ax.plot(x, y, label=label, linewidth=1.8, color=FAMILY_COLORS.get(fam_name))
        lo = ss[y_lo].to_numpy(dtype=float)
        hi = ss[y_hi].to_numpy(dtype=float)
        if np.isfinite(lo).any() and np.isfinite(hi).any() and np.nanmax(hi - lo) > 0:
            ax.fill_between(x, lo, hi, alpha=0.15, color=FAMILY_COLORS.get(fam_name))

    ax.set_xlabel("Selected features k")
    ax.set_ylabel(y_label)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _family_panel_combined(
    df: pd.DataFrame,
    *,
    families: Dict[str, Sequence[str]],
    models: Sequence[str],
    y_mean: str,
    y_lo: str,
    y_hi: str,
    title: str,
    y_label: str,
    out_pdf: Path,
    out_png: Path,
    ylim_default: Tuple[float, float],
) -> None:
    if not models:
        return

    # stack models vertically instead of side-by-side
    fig, axes = plt.subplots(len(models), 1, figsize=(6.5, 4.0 * len(models)), constrained_layout=True)
    axes_arr = np.array(axes).reshape(-1)

    for ax, model in zip(axes_arr, models):
        sub = df[(df["model"] == model)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        agg_rows = []
        for fam_name, sel_list in families.items():
            fam_df = sub[sub["selector"].isin(sel_list)]
            if fam_df.empty:
                continue
            for k in sorted(fam_df["k"].dropna().unique()):
                kdf = fam_df[fam_df["k"] == k]
                if kdf.empty:
                    continue
                mean_val = kdf[y_mean].mean()
                lo_val = kdf[y_lo].min() if y_lo in kdf else np.nan
                hi_val = kdf[y_hi].max() if y_hi in kdf else np.nan
                agg_rows.append({"family": fam_name, "k": k, y_mean: mean_val, y_lo: lo_val, y_hi: hi_val})
        agg = pd.DataFrame(agg_rows)
        if agg.empty:
            ax.axis("off")
            continue

        vals: List[float] = []
        for _, row in agg.iterrows():
            vals.append(row[y_mean])
            if np.isfinite(row[y_lo]):
                vals.append(row[y_lo])
            if np.isfinite(row[y_hi]):
                vals.append(row[y_hi])
        if vals and np.isfinite(vals).any():
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
            ylim = (y_min - pad, y_max + pad)
        else:
            ylim = ylim_default

        for fam_name, sel_list in families.items():
            ss = agg[agg["family"] == fam_name].sort_values("k")
            if ss.empty:
                continue
            x = ss["k"].to_numpy(dtype=float)
            y = ss[y_mean].to_numpy(dtype=float)
            label = fam_name.upper() if fam_name.lower() == "mi" else fam_name.capitalize()
            color = FAMILY_COLORS.get(fam_name)
            ax.plot(x, y, label=label, linewidth=1.8, color=color)
            lo = ss[y_lo].to_numpy(dtype=float)
            hi = ss[y_hi].to_numpy(dtype=float)
            if np.isfinite(lo).any() and np.isfinite(hi).any() and np.nanmax(hi - lo) > 0:
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)

        ax.set_xlabel("Selected features k")
        ax.set_ylabel(y_label)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _family_panel_combined_dual_metric(
    df: pd.DataFrame,
    *,
    families: Dict[str, Sequence[str]],
    models: Sequence[str],
    metrics: Sequence[Tuple[str, str, str]],
    y_labels: Sequence[str],
    out_pdf: Path,
    out_png: Path,
    ylim_defaults: Sequence[Tuple[float, float]],
) -> None:
    """Single figure that stacks both combined family plots (e.g., accuracy & ROC-AUC)."""
    if not models or not metrics:
        return

    nrows, ncols = len(metrics), len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.2 * nrows), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    handles_ref: List = []
    labels_ref: List[str] = []

    for r, (y_mean, y_lo, y_hi) in enumerate(metrics):
        for c, model in enumerate(models):
            ax = axes_arr[r, c]
            sub = df[df["model"] == model].copy()
            if sub.empty:
                ax.axis("off")
                continue

            agg_rows = []
            for fam_name, sel_list in families.items():
                fam_df = sub[sub["selector"].isin(sel_list)]
                if fam_df.empty:
                    continue
                for k in sorted(fam_df["k"].dropna().unique()):
                    kdf = fam_df[fam_df["k"] == k]
                    if kdf.empty:
                        continue
                    mean_val = kdf[y_mean].mean()
                    lo_val = kdf[y_lo].min() if y_lo in kdf else np.nan
                    hi_val = kdf[y_hi].max() if y_hi in kdf else np.nan
                    agg_rows.append({"family": fam_name, "k": k, y_mean: mean_val, y_lo: lo_val, y_hi: hi_val})
            agg = pd.DataFrame(agg_rows)
            if agg.empty:
                ax.axis("off")
                continue

            vals: List[float] = []
            for _, row in agg.iterrows():
                vals.append(row[y_mean])
                if np.isfinite(row[y_lo]):
                    vals.append(row[y_lo])
                if np.isfinite(row[y_hi]):
                    vals.append(row[y_hi])
            if vals and np.isfinite(vals).any():
                y_min = float(np.nanmin(vals))
                y_max = float(np.nanmax(vals))
                pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
                ylim = (y_min - pad, y_max + pad)
            else:
                ylim = ylim_defaults[r]

            for fam_name, sel_list in families.items():
                ss = agg[agg["family"] == fam_name].sort_values("k")
                if ss.empty:
                    continue
                x = ss["k"].to_numpy(dtype=float)
                y = ss[y_mean].to_numpy(dtype=float)
                label = fam_name.upper() if fam_name.lower() == "mi" else fam_name.capitalize()
                color = FAMILY_COLORS.get(fam_name)
                line, = ax.plot(x, y, label=label, linewidth=1.8, color=color)
                lo = ss[y_lo].to_numpy(dtype=float)
                hi = ss[y_hi].to_numpy(dtype=float)
                if np.isfinite(lo).any() and np.isfinite(hi).any() and np.nanmax(hi - lo) > 0:
                    ax.fill_between(x, lo, hi, alpha=0.15, color=color)
                if r == 0 and c == 0:
                    handles_ref.append(line)
                    labels_ref.append(label)

            if r == nrows - 1:
                ax.set_xlabel("Selected features k")
            if c == 0:
                ax.set_ylabel(y_labels[r])
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25)
            ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)

    if handles_ref:
        fig.legend(handles_ref, labels_ref, loc="upper center", ncol=min(4, len(handles_ref)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _compute_rank_corr(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    selectors: Sequence[str],
    model: str,
    score_col: str,
    method: str,
) -> Optional[pd.DataFrame]:
    if method not in RANK_METHODS:
        raise ValueError(f"method must be one of {RANK_METHODS}")
    sub = df[(df["model"] == model) & (df["dataset"].isin(datasets)) & (df["selector"].isin(selectors))].copy()
    if sub.empty:
        return None
    agg = sub.groupby(["dataset", "selector"], as_index=False)[score_col].mean()
    pivot = agg.pivot_table(index="dataset", columns="selector", values=score_col)
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    corr = ranks.corr(method=method)
    return corr.reindex(index=selectors, columns=selectors)


def _draw_corr_heatmap(ax: plt.Axes, corr: pd.DataFrame, selectors: Sequence[str], title: str, vmin: float, vmax: float) -> AxesImage:
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap="coolwarm")
    ax.set_xticks(range(len(selectors)))
    ax.set_xticklabels([SELECTOR_LABELS.get(s, s) for s in selectors], rotation=45, ha="right")
    ax.set_yticks(range(len(selectors)))
    ax.set_yticklabels([SELECTOR_LABELS.get(s, s) for s in selectors])
    # annotate values
    for i, s_row in enumerate(selectors):
        for j, s_col in enumerate(selectors):
            val = corr.loc[s_row, s_col]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    return im


def _rank_correlation_heatmap_combined(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    selectors: Sequence[str],
    models: Sequence[str],
    score_col: str = "roc_auc_mean",
    method: str = "spearman",
    out_pdf: Path,
    out_png: Path,
) -> None:
    corrs = {}
    for m in models:
        c = _compute_rank_corr(df, datasets=datasets, selectors=selectors, model=m, score_col=score_col, method=method)
        if c is not None:
            corrs[m] = c
    if not corrs:
        return

    vmin, vmax = -1.0, 1.0
    n = len(corrs)
    # stack heatmaps vertically
    fig, axes = plt.subplots(n, 1, figsize=(5.0, 4.5 * n), constrained_layout=True)
    axes_arr = np.array(axes).reshape(-1)

    im = None
    for ax, (m, corr) in zip(axes_arr, corrs.items()):
        im = _draw_corr_heatmap(ax, corr, selectors, title=f"{method.title()} — {MODEL_LABELS.get(m, m)}", vmin=vmin, vmax=vmax)

    if im is not None:
        fig.colorbar(im, ax=axes_arr, fraction=0.046, pad=0.04)

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def _rank_correlation_heatmap(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str],
    selectors: Sequence[str],
    model: str,
    score_col: str = "roc_auc_mean",
    method: str = "spearman",
    out_pdf: Path = None,
    out_png: Path = None,
    figsize=(5.0, 4.5),
) -> None:
    """Single-model rank correlation heatmap with value annotations."""
    corr = _compute_rank_corr(df, datasets=datasets, selectors=selectors, model=model, score_col=score_col, method=method)
    if corr is None:
        return

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = _draw_corr_heatmap(
        ax,
        corr,
        selectors,
        title=f"{method.title()} rank corr — {MODEL_LABELS.get(model, model)}",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)




# --------------------------------- Main -----------------------------------

def _family_panel_all_models_single(summary_df, families, models, y_mean, y_lo, y_hi, y_label, out_pdf, out_png, ylim_default):
    """One plot overlaying (family, model) lines across k for all models."""
    sub = summary_df[summary_df["model"].isin(models)].copy()
    if sub.empty:
        return

    agg_rows = []
    for fam_name, sel_list in families.items():
        fam_df = sub[sub["selector"].isin(sel_list)]
        if fam_df.empty:
            continue
        for model in models:
            mdf = fam_df[fam_df["model"] == model]
            if mdf.empty:
                continue
            for k in sorted(mdf["k"].dropna().unique()):
                kdf = mdf[mdf["k"] == k]
                agg_rows.append({
                    "family": fam_name,
                    "model": model,
                    "k": k,
                    y_mean: kdf[y_mean].mean(),
                    y_lo: kdf[y_lo].min() if y_lo in kdf else np.nan,
                    y_hi: kdf[y_hi].max() if y_hi in kdf else np.nan,
                })

    agg = pd.DataFrame(agg_rows)
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)

    vals: List[float] = []
    for _, row in agg.iterrows():
        vals.append(row[y_mean])
        if np.isfinite(row[y_lo]):
            vals.append(row[y_lo])
        if np.isfinite(row[y_hi]):
            vals.append(row[y_hi])
    if vals and np.isfinite(vals).any():
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        span = (y_max - y_min) if y_max > y_min else 1.0
        pad = span * 0.009  # Increased padding to spread lines more
        ylim = (max(0.88, y_min - pad), min(1.0, y_max + pad))
    else:
        ylim = ylim_default

    handles: List = []
    labels: List[str] = []
    for fam_name, sel_list in families.items():
        for model in models:
            ss = agg[(agg["family"] == fam_name) & (agg["model"] == model)].sort_values("k")
            if ss.empty:
                continue
            x = ss["k"].to_numpy(dtype=float)
            y = ss[y_mean].to_numpy(dtype=float)
            label = f"{fam_name.upper() if fam_name.lower()== 'mi' else fam_name.capitalize()} — {MODEL_LABELS.get(model, model)}"
            color = FAMILY_MODEL_COLORS.get((fam_name, model), FAMILY_COLORS.get(fam_name))
            line, = ax.plot(x, y, label=label, linewidth=1.8, color=color)
            # No shaded CI band
            handles.append(line)
            labels.append(label)

    ax.set_xlabel("Selected features k")
    ax.set_ylabel(y_label)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            ncol=2,
            columnspacing=1.2,
            handletextpad=0.6,
        )

    _save_figure(fig, out_pdf, out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="paper_figs")
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--summary_csv", type=str, default=None, help="Path to combined_summary.csv (optional)")
    ap.add_argument("--stability_csv", type=str, default=None, help="Path to combined_stability_summary.csv (optional)")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory containing per-dataset *_summary.csv files (optional)")
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--selectors", type=str, default=",".join(DEFAULT_SELECTORS))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]

    # Load data (prefer combined files if provided; else auto-detect in root)
    if args.summary_csv:
        summary_csv = (root / args.summary_csv).resolve() if not Path(args.summary_csv).is_absolute() else Path(args.summary_csv)
        stability_csv = None
        if args.stability_csv:
            stability_csv = (root / args.stability_csv).resolve() if not Path(args.stability_csv).is_absolute() else Path(args.stability_csv)
        else:
            stability_csv = _find_first_existing([root / "combined_stability_summary.csv"])
        summary_df, stability_df = _load_combined(summary_csv, stability_csv)
    else:
        input_dir = Path(args.input_dir).resolve() if args.input_dir else root
        # try common locations
        guess = _find_first_existing([
            input_dir / "src" / "results" / "aggregated",
            input_dir / "results" / "aggregated",
            input_dir,
        ])
        if guess is None:
            raise FileNotFoundError("Could not locate an aggregated results directory.")
        summary_df, stability_df = _load_per_dataset(guess, datasets)

    # Ensure required columns exist
    req_cols = {"dataset", "selector", "model", "k"}
    for c in req_cols:
        if c not in summary_df.columns and c.lower() not in {x.lower() for x in summary_df.columns}:
            raise ValueError(f"Missing required column '{c}' in summary CSV.")

    # Coerce numeric columns used by plots
    summary_df = _coerce_numeric(summary_df, [
        "k",
        "roc_auc_mean", "roc_auc_ci_lower", "roc_auc_ci_upper",
        "stability_jaccard_mean", "stability_jaccard_ci_lower", "stability_jaccard_ci_upper",
    ])

    # Generate figures
    generated: List[Path] = []

    # Combined rank-correlation heatmaps (one per method across all models)
    for method in RANK_METHODS:
        rank_pdf = outdir / f"fig_rank_corr_{method}_combined.pdf"
        rank_png = outdir / f"fig_rank_corr_{method}_combined.png"
        _rank_correlation_heatmap_combined(
            summary_df,
            datasets=datasets,
            selectors=selectors,
            models=models,
            score_col="roc_auc_mean",
            method=method,
            out_pdf=rank_pdf,
            out_png=rank_png,
        )
        generated.append(rank_pdf)

    # Combined performance (models x datasets grid)
    perf_comb_pdf = outdir / "fig_perf_tabular_combined.pdf"
    perf_comb_png = outdir / "fig_perf_tabular_combined.png"
    _panel_models_datasets(
        summary_df,
        datasets=datasets,
        selectors=selectors,
        models=models,
        y_mean="roc_auc_mean",
        y_lo="roc_auc_ci_lower",
        y_hi="roc_auc_ci_upper",
        title="Predictive performance vs feature budget (all models)",
        y_label="ROC-AUC (mean ± 95% CI)",
        out_pdf=perf_comb_pdf,
        out_png=perf_comb_png,
        ylim_default=(0.5, 1.0),
    )
    generated.append(perf_comb_pdf)

    # Combined family accuracy and ROC-AUC across models
    fam_acc_comb_pdf = outdir / "fig_family_acc_combined.pdf"
    fam_acc_comb_png = outdir / "fig_family_acc_combined.png"
    _family_panel_combined(
        summary_df,
        families=FAMILY_DEFS,
        models=models,
        y_mean="accuracy_mean",
        y_lo="accuracy_ci_lower" if "accuracy_ci_lower" in summary_df.columns else "accuracy_mean",
        y_hi="accuracy_ci_upper" if "accuracy_ci_upper" in summary_df.columns else "accuracy_mean",
        title="Family-level Accuracy vs k (MI vs Standard)",
        y_label="Accuracy (mean ± 95% CI)",
        out_pdf=fam_acc_comb_pdf,
        out_png=fam_acc_comb_png,
        ylim_default=(0.5, 1.0),
    )
    generated.append(fam_acc_comb_pdf)

    fam_auc_comb_pdf = outdir / "fig_family_roc_auc_combined.pdf"
    fam_auc_comb_png = outdir / "fig_family_roc_auc_combined.png"
    _family_panel_combined(
        summary_df,
        families=FAMILY_DEFS,
        models=models,
        y_mean="roc_auc_mean",
        y_lo="roc_auc_ci_lower",
        y_hi="roc_auc_ci_upper",
        title="Family-level ROC-AUC vs k (MI vs Standard)",
        y_label="ROC-AUC (mean ± 95% CI)",
        out_pdf=fam_auc_comb_pdf,
        out_png=fam_auc_comb_png,
        ylim_default=(0.5, 1.0),
    )
    generated.append(fam_auc_comb_pdf)

    # Single-panel family accuracy overlaying both models
    fam_acc_one_pdf = outdir / "fig_family_accuracy_all_models.pdf"
    fam_acc_one_png = outdir / "fig_family_accuracy_all_models.png"
    _family_panel_all_models_single(
        summary_df,
        families=FAMILY_DEFS,
        models=models,
        y_mean="accuracy_mean",
        y_lo="accuracy_ci_lower" if "accuracy_ci_lower" in summary_df.columns else "accuracy_mean",
        y_hi="accuracy_ci_upper" if "accuracy_ci_upper" in summary_df.columns else "accuracy_mean",
        y_label="Accuracy (mean ± 95% CI)",
        out_pdf=fam_acc_one_pdf,
        out_png=fam_acc_one_png,
        ylim_default=(0.92, 0.95),
    )
    generated.append(fam_acc_one_pdf)

    # Dual-metric combined family figure (accuracy + ROC-AUC)
    acc_lo = "accuracy_ci_lower" if "accuracy_ci_lower" in summary_df.columns else "accuracy_mean"
    acc_hi = "accuracy_ci_upper" if "accuracy_ci_upper" in summary_df.columns else "accuracy_mean"
    fam_dual_pdf = outdir / "fig_family_acc_roc_auc_combined_dual.pdf"
    fam_dual_png = outdir / "fig_family_acc_roc_auc_combined_dual.png"
    _family_panel_combined_dual_metric(
        summary_df,
        families=FAMILY_DEFS,
        models=models,
        metrics=[
            ("accuracy_mean", acc_lo, acc_hi),
            ("roc_auc_mean", "roc_auc_ci_lower", "roc_auc_ci_upper"),
        ],
        y_labels=["Accuracy (mean ± 95% CI)", "ROC-AUC (mean ± 95% CI)"],
        out_pdf=fam_dual_pdf,
        out_png=fam_dual_png,
        ylim_defaults=[(0.5, 1.0), (0.5, 1.0)],
    )
    generated.append(fam_dual_pdf)

    for model in models:
        # ROC-AUC
        roc_pdf = outdir / f"fig_roc_auc_{model}.pdf"
        roc_png = outdir / f"fig_roc_auc_{model}.png"
        _panel_1x3(
            summary_df,
            datasets=datasets,
            selectors=selectors,
            model=model,
            y_mean="roc_auc_mean",
            y_lo="roc_auc_ci_lower",
            y_hi="roc_auc_ci_upper",
            title="Predictive performance vs feature budget",
            y_label="ROC-AUC (mean ± 95% CI)",
            out_pdf=roc_pdf,
            out_png=roc_png,
            ylim_default=(0.5, 1.0),
        )
        generated.append(roc_pdf)

        # Stability (use columns from the combined summary if present)
        stab_pdf = outdir / f"fig_stability_{model}.pdf"
        stab_png = outdir / f"fig_stability_{model}.png"
        _panel_1x3(
            summary_df,
            datasets=datasets,
            selectors=selectors,
            model=model,
            y_mean="stability_jaccard_mean",
            y_lo="stability_jaccard_ci_lower",
            y_hi="stability_jaccard_ci_upper",
            title="Selection stability vs feature budget",
            y_label="Stability (Jaccard, mean ± 95% CI)",
            out_pdf=stab_pdf,
            out_png=stab_png,
            ylim_default=(0.0, 1.0),
        )
        generated.append(stab_pdf)

        # Family-level aggregates across datasets: MI vs Standard selectors
        fam_acc_pdf = outdir / f"fig_family_accuracy_{model}.pdf"
        fam_acc_png = outdir / f"fig_family_accuracy_{model}.png"
        _family_panel(
            summary_df,
            families=FAMILY_DEFS,
            model=model,
            y_mean="accuracy_mean",
            y_lo="accuracy_ci_lower" if "accuracy_ci_lower" in summary_df.columns else "accuracy_mean",
            y_hi="accuracy_ci_upper" if "accuracy_ci_upper" in summary_df.columns else "accuracy_mean",
            title="Family-level Accuracy vs k (MI vs Standard)",
            y_label="Accuracy (mean ± 95% CI)",
            out_pdf=fam_acc_pdf,
            out_png=fam_acc_png,
            ylim_default=(0.5, 1.0),
        )

        fam_auc_pdf = outdir / f"fig_family_roc_auc_{model}.pdf"
        fam_auc_png = outdir / f"fig_family_roc_auc_{model}.png"
        _family_panel(
            summary_df,
            families=FAMILY_DEFS,
            model=model,
            y_mean="roc_auc_mean",
            y_lo="roc_auc_ci_lower",
            y_hi="roc_auc_ci_upper",
            title="Family-level ROC-AUC vs k (MI vs Standard)",
            y_label="ROC-AUC (mean ± 95% CI)",
            out_pdf=fam_auc_pdf,
            out_png=fam_auc_png,
            ylim_default=(0.5, 1.0),
        )

        # Selector rank correlation heatmap
        for method in RANK_METHODS:
            rank_pdf = outdir / f"fig_rank_corr_{method}_{model}.pdf"
            rank_png = outdir / f"fig_rank_corr_{method}_{model}.png"
            _rank_correlation_heatmap(
                summary_df,
                datasets=datasets,
                selectors=selectors,
                model=model,
                score_col="roc_auc_mean",
                method=method,
                out_pdf=rank_pdf,
                out_png=rank_png,
                figsize=(5.0, 4.5),
            )
            generated.append(rank_pdf)

    # # Single PDF for convenience (like your aa.pdf)
    # if generated:
    #     _concat_pdfs(generated, outdir / "aa_fixed.pdf")

    print("Done.")
    print(f"Outputs in: {outdir}")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Selectors: {selectors}")


if __name__ == "__main__":
    main()
