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
from typing import Iterable, List, Optional, Tuple

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
    "rf",
    "mi",
    "mrmr",
]

SELECTOR_LABELS = {
    "variance": "Variance",
    "anova": "ANOVA",
    "l1_logreg": "L1-LogReg",
    "rf": "RF-Imp",
    "mi": "MI",
    "mrmr": "mRMR",
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
    _reserve_top_band(fig, top=0.82)

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

        ax.set_xlabel("k (selected features)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"{metric_name} (mean ± 95% CI)")

    _add_suptitle_and_legend(fig, handles, labels)

    outpath = outdir / f"fig_perf_{model}_tabular.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outdir / f"fig_perf_{model}_tabular.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_stability_vs_k(stab_df: pd.DataFrame, cols: Cols, model: str, outdir: Path) -> Path:
    sub = stab_df[stab_df[cols.model] == model]

    fig, axes = plt.subplots(len(DATASET_ORDER), 1, figsize=(6.5, 4.2 * len(DATASET_ORDER)), sharey=False)
    _reserve_top_band(fig, top=0.82)

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

        ax.set_xlabel("k (selected features)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Stability (mean ± 95% CI)")

    _add_suptitle_and_legend(fig, handles, labels)

    outpath = outdir / f"fig_stability_{model}_tabular.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outdir / f"fig_stability_{model}_tabular.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def make_kstar_table(perf_df: pd.DataFrame, perf_cols: Cols, stab_df: pd.DataFrame, stab_cols: Cols, model: str, outpath: Path, metric_name: str = "ROC-AUC") -> None:
    rows = []

    perf_m = perf_df[perf_df[perf_cols.model] == model]
    stab_m = stab_df[stab_df[stab_cols.model] == model]

    for ds in DATASET_ORDER:
        for sel in SELECTOR_ORDER:
            s_perf = perf_m[(perf_m[perf_cols.dataset] == ds) & (perf_m[perf_cols.selector] == sel)]
            if s_perf.empty:
                continue

            idx = int(s_perf[perf_cols.metric_mean].astype(float).idxmax())
            best = perf_df.loc[idx]
            kstar = int(best[perf_cols.k])
            auc = float(best[perf_cols.metric_mean])

            s_stab = stab_m[
                (stab_m[stab_cols.dataset] == ds)
                & (stab_m[stab_cols.selector] == sel)
                & (stab_m[stab_cols.k] == kstar)
            ]
            stab = float(s_stab[stab_cols.metric_mean].iloc[0]) if not s_stab.empty else np.nan

            rows.append((ds, sel, kstar, auc, stab))

    def tex_escape(s: str) -> str:
        return s.replace("_", "\\_")

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append(f"Dataset & Selector & $k^\star$ & {metric_name} & Stability \\")
    lines.append("\\midrule")

    for ds in DATASET_ORDER:
        ds_rows = [r for r in rows if r[0] == ds]
        if not ds_rows:
            continue

        for (ds_, sel, kstar, auc, stab) in ds_rows:
            ds_name = tex_escape(ds_.replace("_", " ").title())
            sel_name = tex_escape(SELECTOR_LABELS.get(sel, sel))
            auc_s = f"{auc:.4f}"
            stab_s = "--" if np.isnan(stab) else f"{stab:.4f}"
            lines.append(f"{ds_name} & {sel_name} & {kstar:d} & {auc_s} & {stab_s} \\")
        lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Best performance at $k^\\star$ for model {tex_escape(model)}; stability reported at the same $k^\\star$.}}")
    lines.append(f"\\label{{tab:kstar_{tex_escape(model)}}}")
    lines.append("\\end{table}")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def plot_performance_combined(perf_df: pd.DataFrame, cols: Cols, outdir: Path, metric_name: str = "ROC-AUC") -> Path:
    """Grid with rows=datasets and cols=models; one legend at top."""
    if perf_df.empty:
        return outdir / "fig_perf_tabular_combined.pdf"

    # rows=datasets, cols=models
    nrows, ncols = len(DATASET_ORDER), len(MODEL_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), sharey=False)
    # Add extra vertical space: keep margin above plots generous, and fine-tune legend/title spacing
    fig.subplots_adjust(top=0.78, hspace=0.50)
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
        if vals and np.isfinite(vals).any():
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
            ylim = (y_min - pad, y_max + pad)
        else:
            ylim = (0.0, 1.0)

        for j, model in enumerate(MODEL_ORDER):
            ax = axes_arr[i, j]
            s_df = ds_df[ds_df[cols.model] == model]
            if s_df.empty:
                ax.set_title(f"{ds.replace('_', ' ').title()} — {model} (no data)")
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

            ax.set_title(f"{ds.replace('_', ' ').title()} — {model}")
            if i == nrows - 1:
                ax.set_xlabel("k (selected features)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(f"{metric_name} (mean ± 95% CI)")

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False, bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("Predictive performance vs feature budget (all models)", y=0.95)

    out_pdf = outdir / "fig_perf_tabular_combined.pdf"
    out_png = outdir / "fig_perf_tabular_combined.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, default="combined_summary.csv")
    ap.add_argument("--stability_csv", type=str, default="combined_stability_summary.csv")
    ap.add_argument("--outdir", type=str, default="paper_figs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_csv(args.summary_csv)
    stab = pd.read_csv(args.stability_csv)

    perf_cols = _infer_columns(perf)
    stab_cols = _infer_stability_columns(stab)

    perf = _canonicalize(perf, perf_cols)
    stab = _canonicalize(stab, stab_cols)

    perf = _filter_and_sort(perf, perf_cols)
    stab = _filter_and_sort(stab, stab_cols)

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
    plot_performance_combined(perf, perf_cols, outdir)


if __name__ == "__main__":
    main()
