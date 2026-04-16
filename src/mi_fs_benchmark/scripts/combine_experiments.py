#!/usr/bin/env python3
"""Combine matching raw result CSVs from two selector-subset folders.

This script matches CSV files by exact filename (intersection-only) between two
input directories, concatenates rows for each matched file, and writes merged
CSVs to an output directory using the same filenames.

Default behavior matches the current project layout:
- dir-a: src/results/raw
- dir-b: src/results/old_Paper/raw
- output: src/results/temp

Usage:
    python combine_raw_selector_subsets.py
    python combine_raw_selector_subsets.py --dir-a src/results/raw --dir-b src/results/old_Paper/raw --output-dir src/results/temp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine matching raw CSVs from two folders (intersection-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir-a",
        type=Path,
        default=None,
        help="First raw-results directory.",
    )
    parser.add_argument(
        "--dir-b",
        type=Path,
        default=None,
        help="Second raw-results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save combined CSVs.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _csv_map(folder: Path) -> dict[str, Path]:
    return {p.name: p for p in folder.glob("*.csv") if p.is_file()}


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_keys = [
        "selector",
        "model",
        "k",
        "run_id",
        "fold_id",
    ]
    present = [k for k in sort_keys if k in df.columns]
    if present:
        return df.sort_values(present).reset_index(drop=True)
    return df.reset_index(drop=True)


def combine(dir_a: Path, dir_b: Path, output_dir: Path) -> None:
    if not dir_a.exists() or not dir_a.is_dir():
        raise FileNotFoundError(f"dir-a does not exist or is not a directory: {dir_a}")
    if not dir_b.exists() or not dir_b.is_dir():
        raise FileNotFoundError(f"dir-b does not exist or is not a directory: {dir_b}")

    map_a = _csv_map(dir_a)
    map_b = _csv_map(dir_b)

    matched = sorted(set(map_a).intersection(map_b))
    only_a = sorted(set(map_a) - set(map_b))
    only_b = sorted(set(map_b) - set(map_a))

    if not matched:
        raise SystemExit("No matching CSV filenames found between dir-a and dir-b.")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows_written = 0
    files_written = 0
    total_duplicates_dropped = 0

    for name in matched:
        df_a = pd.read_csv(map_a[name])
        df_b = pd.read_csv(map_b[name])

        if list(df_a.columns) != list(df_b.columns):
            raise ValueError(
                "Column mismatch for file "
                f"'{name}'.\n"
                f"dir-a columns: {list(df_a.columns)}\n"
                f"dir-b columns: {list(df_b.columns)}"
            )

        # Keep dir-a rows when exact duplicates are present in both sources.
        merged = pd.concat([df_a, df_b], ignore_index=True)
        before_dedup = len(merged)
        merged = merged.drop_duplicates(keep="first")
        duplicates_dropped = before_dedup - len(merged)
        merged = _sort_rows(merged)

        out_path = output_dir / name
        merged.to_csv(out_path, index=False)

        files_written += 1
        total_rows_written += len(merged)
        total_duplicates_dropped += duplicates_dropped

    print("[OK] Raw subset merge complete")
    print(f"  dir-a: {dir_a}")
    print(f"  dir-b: {dir_b}")
    print(f"  output-dir: {output_dir}")
    print(f"  matched files: {len(matched)}")
    print(f"  files written: {files_written}")
    print(f"  total rows written: {total_rows_written}")
    print(f"  duplicate rows dropped from dir-b: {total_duplicates_dropped}")

    if only_a:
        print(f"  skipped (only in dir-a): {len(only_a)}")
    if only_b:
        print(f"  skipped (only in dir-b): {len(only_b)}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    dir_a = args.dir_a or (repo_root / "src" / "results" / "raw")
    dir_b = args.dir_b or (repo_root / "src" / "results" / "old_Paper" / "raw")
    output_dir = args.output_dir or (repo_root / "src" / "results" / "temp")

    combine(dir_a.resolve(), dir_b.resolve(), output_dir.resolve())


if __name__ == "__main__":
    main()

