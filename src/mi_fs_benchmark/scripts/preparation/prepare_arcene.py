# src/mi_fs_benchmark/scripts/prepare_arcene.py
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
    raw_dir = repo_root / "src" / "data" / "raw" / "arcene" / "ARCENE"
    out_dir = repo_root / "src" / "data" / "processed" / "arcene"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"The raw data directory does not exist: {raw_dir}")

    print("\n" + "=" * 70)
    print("📊 ARCENE DATASET - FOLD ARTIFACT GENERATION")
    print("=" * 70)
    print(f"📂 Raw data: {raw_dir}")
    print(f"💾 Output: {out_dir}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Step 1: Load raw data
    # -------------------------------------------------------------------------
    train_data_file = raw_dir / "arcene_train.data"
    train_labels_file = raw_dir / "arcene_train.labels"
    test_data_file = raw_dir / "arcene_valid.data"  # Arcene uses 'valid' for test set

    print("📥 Step 1/4: Loading raw data...")
    print(f"   Reading: {train_data_file.name}, {train_labels_file.name}, and {test_data_file.name}")

    if not train_data_file.exists() or not train_labels_file.exists():
        raise FileNotFoundError(f"Missing Arcene training files in: {raw_dir}")
    if not test_data_file.exists():
        raise FileNotFoundError(f"Missing Arcene test file in: {raw_dir}")

    # Load training features (space-delimited)
    X_train = pd.read_csv(train_data_file, sep=' ', header=None)
    X_train = X_train.dropna(axis=1, how='all')

    # Load training labels
    y_train = pd.read_csv(train_labels_file, header=None, names=['target'])
    y_train = y_train['target'].values
    # Convert labels from {-1, 1} to {0, 1}
    y_train = (y_train + 1) // 2

    # Load test features (space-delimited)
    X_test = pd.read_csv(test_data_file, sep=' ', header=None)
    X_test = X_test.dropna(axis=1, how='all')

    print(f"   ✓ Loaded {len(X_train):,} training samples")
    print(f"   ✓ Loaded {len(X_test):,} test samples")

    # -------------------------------------------------------------------------
    # Step 2: Preprocessing
    # -------------------------------------------------------------------------
    print("\n🔧 Step 2/4: Preprocessing features...")
    print("   - All features are numeric (10,000 features)")
    print(f"     ✓ {X_train.shape[1]} numeric features found")

    print("   - Checking for missing values")
    if X_train.isnull().any().any():
        print("   - Imputing training missing values with median")
        X_train = X_train.fillna(X_train.median())
    if X_test.isnull().any().any():
        print("   - Imputing test missing values with training median")
        X_test = X_test.fillna(X_train.median())
    print(f"     ✓ Missing values handled")

    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Convert to numpy arrays for fold processing
    X_train_np = X_train.values
    X_test_np = X_test.values

    print(f"\n📐 Training shape: {X_train_np.shape[0]:,} samples × {X_train_np.shape[1]} features")
    print(f"📐 Test shape: {X_test_np.shape[0]:,} samples × {X_test_np.shape[1]} features")


    print(f"🎯 Training target distribution: Class 0={np.sum(y_train==0):,}, Class 1={np.sum(y_train==1):,}")

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
        "dataset": "arcene",
        "n_splits": n_splits,
        "seed": seed,
        "n_samples": int(X_train_np.shape[0]),
        "n_features": int(X_train_np.shape[1]),
        "feature_names": feature_names,
        "target_column": "target",
    }

    fold_iterator = enumerate(skf.split(X_train_np, y_train))
    for fold_id, (train_idx, test_idx) in tqdm(
        fold_iterator,
        total=n_splits,
        desc="Processing folds",
        unit="fold",
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        X_train_raw = X_train_np[train_idx]
        X_test_raw = X_train_np[test_idx]
        y_train_fold = y_train[train_idx]
        y_test_fold = y_train[test_idx]

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
        )

        # Precompute MI lookup table
        mi_lookup = compute_mi_lookup(
            X_train_disc,
            y_train_fold,
            n_neighbors=3,
            random_state=seed,
            n_jobs=-1,
        )

        # Create fold artifact
        artifact = FoldArtifact(
            fold_id=fold_id,
            X_train_disc=X_train_disc,
            X_test_disc=X_test_disc,
            y_train=y_train_fold,
            y_test=y_test_fold,
            mi_lookup=mi_lookup,
            feature_names=feature_names,
        )

        fold_path = out_dir / f"fold_{fold_id}.npz"
        artifact.save(fold_path)

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ ARCENE PREPROCESSING COMPLETED")
    print("=" * 70)
    print(f"📁 Generated {n_splits} fold artifacts in: {out_dir}")
    print(f"📄 Metadata saved to: {metadata_path.name}")
    print("\n💡 Next step: Run experiments with:")
    print("   python run_multi_seed_experiment.py --config arcene.yaml --n-runs 10")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
