# scripts/prepare_santander.py
from __future__ import annotations

import sys
from pathlib import Path
# For Google Colab compatibility
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mi_fs_benchmark.experiment.cv.artifacts import FoldArtifact
from mi_fs_benchmark.logging_utils import get_logger, setup_logging
from mi_fs_benchmark.data.preprocessing import compute_mi_lookup, discretize_features

logger = get_logger(__name__)


def main():
    setup_logging()

    # Use absolute paths relative to project root
    repo_root = Path(__file__).resolve().parents[4]  # Go up to MI-FS root
    raw_dir = repo_root / "src" / "data" / "raw" / "santander"
    out_dir = repo_root / "src" / "data" / "processed" / "santander"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"The raw data directory does not exist: {raw_dir}")

    print("\n" + "=" * 70)
    print("📊 SANTANDER DATASET - FOLD ARTIFACT GENERATION")
    print("=" * 70)
    print(f"📂 Raw data: {raw_dir}")
    print(f"💾 Output: {out_dir}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Load raw data
    # -------------------------------------------------------------------------
    train_csv = raw_dir / "train.csv"
    print("📥 Step 1/4: Loading raw data...")
    print(f"   Reading: {train_csv.name}")

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing: {train_csv}")

    df = pd.read_csv(train_csv)

    # Santander has 'target' column
    target_column = "target"
    if target_column not in df.columns:
        # Try alternative target column names
        target_column = "TARGET"

    y = df[target_column].values

    # Drop ID column if exists
    id_cols = [col for col in df.columns if col.lower() in ["id", "id_code"]]
    X = df.drop(columns=[target_column] + id_cols)

    print(f"   ✓ Loaded {len(df):,} samples")

    # -------------------------------------------------------------------------
    # Step 2: Preprocessing
    # -------------------------------------------------------------------------
    print("\n🔧 Step 2/4: Preprocessing features...")
    print("   - Selecting numeric features")
    X = X.select_dtypes(include=[np.number])
    print(f"     ✓ {X.shape[1]} numeric features found")

    print("   - Imputing missing values with median")
    X = X.fillna(X.median())
    print(f"     ✓ Missing values handled")

    feature_names = list(X.columns)
    X_np = X.values

    print(f"\n📐 Final dataset shape: {X_np.shape[0]:,} samples × {X_np.shape[1]} features")
    print(f"🎯 Target distribution: Class 0={np.sum(y==0):,}, Class 1={np.sum(y==1):,}")

    # -------------------------------------------------------------------------
    # Step 3: Define folds
    # -------------------------------------------------------------------------
    n_splits = 5
    seed = 42

    print(f"\n✂️  Step 3/4: Creating {n_splits} stratified folds (seed={seed})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # -------------------------------------------------------------------------
    # Step 4: Generate fold artifacts
    # -------------------------------------------------------------------------
    print(f"\n⚙️  Step 4/4: Generating fold artifacts...")
    print("=" * 70)

    metadata = {
        "dataset": "santander",
        "n_splits": n_splits,
        "seed": seed,
        "n_samples": int(X_np.shape[0]),
        "n_features": int(X_np.shape[1]),
        "feature_names": feature_names,
        "target_column": target_column,
    }

    fold_iterator = enumerate(skf.split(X_np, y))
    for fold_id, (train_idx, test_idx) in tqdm(
        fold_iterator,
        total=n_splits,
        desc="Processing folds",
        unit="fold",
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        X_train_raw = X_np[train_idx]
        X_test_raw = X_np[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Discretize for MI estimation
        X_train_disc, X_test_disc = discretize_features(
            X_train_scaled,
            X_test_scaled,
            n_bins=10,
            strategy="quantile",
            subsample=200000,  # Speed up quantile computation
        )

        # Precompute MI lookup table
        mi_lookup = compute_mi_lookup(
            X_train_disc,
            y_train,
            n_neighbors=3,
            random_state=seed,
            n_jobs=-1,
        )

        # Create fold artifact
        artifact = FoldArtifact(
            fold_id=fold_id,
            X_train_disc=X_train_disc,
            X_test_disc=X_test_disc,
            y_train=y_train,
            y_test=y_test,
            mi_lookup=mi_lookup,
            feature_names=feature_names,
        )

        fold_path = out_dir / f"fold_{fold_id}.npz"
        artifact.save(fold_path)

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ SANTANDER PREPROCESSING COMPLETED")
    print("=" * 70)
    print(f"📁 Generated {n_splits} fold artifacts in: {out_dir}")
    print(f"📄 Metadata saved to: {metadata_path.name}")
    print("\n💡 Next step: Run experiments with:")
    print("   python run_multi_seed_experiment.py --config santander.yaml --n-runs 10")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
