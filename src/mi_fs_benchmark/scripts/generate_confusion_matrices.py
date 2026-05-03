#!/usr/bin/env python3
"""Generate confusion-matrix evidence from raw fold-level benchmark outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldStats:
    n_test: int
    n_pos: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_dataset_raw(raw_dir: Path, dataset: str) -> pd.DataFrame:
    patterns = [
        f"{dataset}_seed*.csv",
        f"{dataset}__run*__seed*.csv",
    ]
    files = sorted({p for pattern in patterns for p in raw_dir.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No raw files found for dataset '{dataset}' in {raw_dir}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if "dataset" not in df.columns:
        df["dataset"] = dataset
    return df


def _fold_artifact_dir(processed_dir: Path) -> Path:
    with_artifacts = processed_dir / "fold_artifacts"
    return with_artifacts if with_artifacts.exists() else processed_dir


def _load_fold_stats(processed_dir: Path) -> Dict[int, FoldStats]:
    fold_dir = _fold_artifact_dir(processed_dir)
    fold_paths = sorted(fold_dir.glob("fold_*.npz"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold_*.npz files found in {fold_dir}")

    out: Dict[int, FoldStats] = {}
    for fp in fold_paths:
        data = np.load(fp, allow_pickle=True)
        fold_id = int(data["fold_id"])
        y_test = np.asarray(data["y_test"]).astype(int)
        out[fold_id] = FoldStats(n_test=int(y_test.size), n_pos=int(y_test.sum()))
    return out


def _infer_confusion_from_metrics(accuracy: float, f1: float, n_test: int, n_pos: int) -> Tuple[int, int, int, int]:
    n_neg = n_test - n_pos
    if n_test <= 0:
        return 0, 0, 0, 0

    if not np.isfinite(accuracy):
        raise ValueError("Cannot infer confusion matrix without finite accuracy")

    f1 = 0.0 if not np.isfinite(f1) else float(f1)
    errors = max(0.0, n_test * (1.0 - float(accuracy)))

    if f1 <= 0.0 or errors <= 0.0:
        tp = 0
    elif f1 >= 1.0:
        tp = n_pos
    else:
        tp_float = (f1 * errors) / (2.0 * (1.0 - f1))
        tp = int(round(tp_float))

    tp = max(0, min(tp, n_pos))
    tn = int(round((accuracy * n_test) - tp))
    tn = max(0, min(tn, n_neg))

    fp = n_neg - tn
    fn = n_pos - tp

    # Keep total exact after integer rounding/clipping.
    delta = n_test - (tn + fp + fn + tp)
    if delta != 0:
        tn = max(0, min(n_neg, tn + delta))
        fp = n_neg - tn

    return tn, fp, fn, tp


def _ensure_confusion_columns(df: pd.DataFrame, fold_stats: Dict[int, FoldStats]) -> pd.DataFrame:
    out = df.copy()
    has_counts = all(c in out.columns for c in ["tn", "fp", "fn", "tp"])

    tns, fps, fns, tps, sources = [], [], [], [], []

    for _, row in out.iterrows():
        fold_id = int(row["fold_id"])
        if fold_id not in fold_stats:
            raise KeyError(f"Missing fold stats for fold_id={fold_id}")

        if has_counts and all(pd.notna(row[c]) for c in ["tn", "fp", "fn", "tp"]):
            tn, fp, fn, tp = (int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"]))
            source = "raw_counts"
        else:
            fs = fold_stats[fold_id]
            tn, fp, fn, tp = _infer_confusion_from_metrics(
                accuracy=float(row["accuracy"]),
                f1=float(row["f1"]),
                n_test=fs.n_test,
                n_pos=fs.n_pos,
            )
            source = "inferred_from_accuracy_f1"

        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        sources.append(source)

    out["tn"] = tns
    out["fp"] = fps
    out["fn"] = fns
    out["tp"] = tps
    out["confusion_source"] = sources

    denom = (2 * out["tp"]) + out["fp"] + out["fn"]
    out["f1_from_confusion"] = np.where(denom > 0, (2 * out["tp"]) / denom, 0.0)
    out["f1_abs_diff"] = (out["f1"] - out["f1_from_confusion"]).abs()
    return out


def _aggregate_by_setting(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["selector", "model", "k"], as_index=False)
        .agg(
            runs=("run_id", "nunique"),
            folds=("fold_id", "count"),
            accuracy_mean=("accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            f1_mean=("f1", "mean"),
            f1_from_confusion_mean=("f1_from_confusion", "mean"),
            f1_abs_diff_max=("f1_abs_diff", "max"),
            tn=("tn", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tp=("tp", "sum"),
        )
    )

    denom = (2 * grouped["tp"]) + grouped["fp"] + grouped["fn"]
    grouped["f1_from_aggregated_confusion"] = np.where(denom > 0, (2 * grouped["tp"]) / denom, 0.0)
    grouped["precision_from_aggregated_confusion"] = np.where(
        (grouped["tp"] + grouped["fp"]) > 0,
        grouped["tp"] / (grouped["tp"] + grouped["fp"]),
        0.0,
    )
    grouped["recall_from_aggregated_confusion"] = np.where(
        (grouped["tp"] + grouped["fn"]) > 0,
        grouped["tp"] / (grouped["tp"] + grouped["fn"]),
        0.0,
    )

    return grouped.sort_values(["model", "f1_mean", "roc_auc_mean"], ascending=[True, True, False]).reset_index(drop=True)


def _plot_confusion_heatmap(tn: int, fp: int, fn: int, tp: int, title: str, out_path: Path) -> None:
    # Display order requested for readability: [[TP, FN], [FP, TN]].
    mat = np.array([[tp, fn], [fp, tn]], dtype=float)
    total = mat.sum() if mat.sum() > 0 else 1.0
    vmax = float(mat.max()) if mat.size else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    # Use a lighter map range to keep annotation text highly legible.
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=vmax * 1.6)

    for i in range(2):
        for j in range(2):
            count = int(mat[i, j])
            pct = 100.0 * mat[i, j] / total
            ax.text(j, i, f"{count}\n({pct:.2f}%)", ha="center", va="center", color="black", fontsize=10)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 1", "Pred 0"])
    ax.set_yticklabels(["True 1", "True 0"])
    ax.set_title(title, pad=22)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description="Generate confusion matrix evidence from benchmark raw outputs")
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--raw-dir", type=Path, default=repo / "src" / "results" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Defaults to src/data/processed/<dataset>")
    parser.add_argument("--output-dir", type=Path, default=repo / "src" / "results" / "aggregated")
    parser.add_argument("--fig-dir", type=Path, default=repo / "paper_figs")
    parser.add_argument("--selector", type=str, default=None, help="Optional: plot this selector only")
    parser.add_argument("--model", type=str, default=None, help="Optional: plot this model only")
    parser.add_argument("--k", type=int, default=None, help="Optional: plot this k only")
    return parser.parse_args()


def _pretty_label(token: str) -> str:
    key = str(token).strip().lower()
    lookup = {
        "home_credit": "Home Credit",
        "ieee_cis_fraud": "IEEE-CIS Fraud",
        "logreg": "LogReg",
        "hgbt": "HGBT",
        "l1_logreg": "L1-LogReg",
        "tree_importance": "Tree Importance",
        "mrmr": "mRMR",
        "mi": "MI",
        "roc_auc": "ROC-AUC",
    }
    if key in lookup:
        return lookup[key]
    return key.replace("_", " ").title()


def _build_title(dataset: str, model: str, focus: str, selector: str, k: int) -> str:
    return f"{_pretty_label(dataset)} - {_pretty_label(model)}\n{focus}: {_pretty_label(selector)} (k={k})"


def _pick_kstar_row(sub: pd.DataFrame) -> pd.Series:
    sort_cols = ["roc_auc_mean", "f1_mean"]
    asc = [False, False]
    if "stability_jaccard_mean" in sub.columns:
        sort_cols.append("stability_jaccard_mean")
        asc.append(False)
    # Deterministic tie-breakers.
    sort_cols.extend(["selector", "k"])
    asc.extend([True, True])
    return sub.sort_values(sort_cols, ascending=asc).iloc[0]


def main() -> None:
    args = parse_args()
    dataset = args.dataset.strip()

    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = _repo_root() / "src" / "data" / "processed" / dataset

    raw_df = _load_dataset_raw(args.raw_dir, dataset)
    fold_stats = _load_fold_stats(processed_dir)
    fold_df = _ensure_confusion_columns(raw_df, fold_stats)

    settings_df = _aggregate_by_setting(fold_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_out = args.output_dir / f"{dataset}_confusion_fold_level.csv"
    setting_out = args.output_dir / f"{dataset}_confusion_by_setting.csv"
    fold_df.to_csv(fold_out, index=False)
    settings_df.to_csv(setting_out, index=False)

    # Create one k* snapshot per model, aligned with peak performance table logic.
    for model in sorted(settings_df["model"].unique()):
        sub = settings_df[settings_df["model"] == model]
        if sub.empty:
            continue

        kstar_row = _pick_kstar_row(sub)

        kstar_title = _build_title(
            dataset=dataset,
            model=model,
            focus="k* Peak ROC-AUC",
            selector=str(kstar_row["selector"]),
            k=int(kstar_row["k"]),
        )
        kstar_path = args.fig_dir / f"fig_confusion_{dataset}_{model}_kstar_peak.png"
        _plot_confusion_heatmap(
            int(kstar_row["tn"]),
            int(kstar_row["fp"]),
            int(kstar_row["fn"]),
            int(kstar_row["tp"]),
            kstar_title,
            kstar_path,
        )

    # Optional custom plot for an explicitly requested setting.
    if args.selector and args.model and args.k is not None:
        hit = settings_df[
            (settings_df["selector"] == args.selector)
            & (settings_df["model"] == args.model)
            & (settings_df["k"] == args.k)
        ]
        if not hit.empty:
            row = hit.iloc[0]
            title = _build_title(
                dataset=dataset,
                model=args.model,
                focus="Selected Setting",
                selector=args.selector,
                k=args.k,
            )
            out = args.fig_dir / f"fig_confusion_{dataset}_{args.model}_{args.selector}_k{args.k}.png"
            _plot_confusion_heatmap(int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"]), title, out)

    print(f"Saved: {fold_out}")
    print(f"Saved: {setting_out}")


if __name__ == "__main__":
    main()

