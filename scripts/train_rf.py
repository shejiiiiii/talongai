# scripts/train_rf.py
"""
Train the Random Forest on the labeled spectral dataset.

After training, set USE_RF = True in config.py to switch inference to RF.

Run from the project root:
    python scripts/train_rf.py

NOTE: The preprocessing order here is SavGol → ratio features → SNV (applied
to the full 39-feature matrix). The inference path in src/analyzer.py applies
SNV first (to 18 channels), then adds ratio features. These are inconsistent
and should be unified when retraining from scratch.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from config import (
    OUTPUT_FILE,
    MODEL_DIR, RF_MODEL_NAME, RF_CLASSES,
    RANDOM_STATE, TEST_SIZE, Z_THRESHOLD,
)
from src.analyzer import add_ratio_features


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X: np.ndarray) -> np.ndarray:
    """SavGol smooth → ratio features → SNV normalise (RF-specific pipeline)."""
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = add_ratio_features(X)   # 18 → 39 features
    X = (X - np.mean(X, axis=1, keepdims=True)) / (
        np.std(X, axis=1, keepdims=True) + 1e-8
    )
    return X


# ─────────────────────────────────────────────────────────────────────────────
#  Outlier removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_outliers(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    z_threshold: float = Z_THRESHOLD,
) -> tuple:
    """
    Drop scans that are outliers within their eggplant group.

    Runs after ratio features are added (39-feature space) so the norm-based
    distance reflects the full feature set the RF will actually see.
    This differs from src/dataset.py's outlier removal which runs on raw 18
    channels — intentional, not a bug.
    """
    keep = np.ones(len(X), dtype=bool)
    for eid in np.unique(ids):
        mask  = ids == eid
        norms = np.linalg.norm(X[mask], axis=1)
        mu, sig = norms.mean(), norms.std()
        if sig == 0:
            continue
        outlier = (
            (norms < mu - z_threshold * sig) |
            (norms > mu + z_threshold * sig)
        )
        keep[np.where(mask)[0][outlier]] = False
    print(f"  Outlier removal: dropped {(~keep).sum()}, kept {keep.sum()}")
    return X[keep], y[keep], ids[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  Grouped train/test split
# ─────────────────────────────────────────────────────────────────────────────

def grouped_split(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
) -> tuple:
    """
    Split by eggplant ID with stratification across healthy/infested classes.
    No augmentation — RF doesn't benefit from the Gaussian noise augmentation
    used for the CNN.
    """
    unique_ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(RANDOM_STATE)

    h_ids   = [i for i in unique_ids if y[ids == i][0] == 0]
    inf_ids = [i for i in unique_ids if y[ids == i][0] == 1]
    rng.shuffle(h_ids)
    rng.shuffle(inf_ids)

    n_h_test  = max(1, int(len(h_ids)   * TEST_SIZE))
    n_inf_test = max(1, int(len(inf_ids) * TEST_SIZE))

    test_ids  = set(h_ids[:n_h_test]   + inf_ids[:n_inf_test])
    train_ids = set(h_ids[n_h_test:]   + inf_ids[n_inf_test:])

    train_mask = np.array([i in train_ids for i in ids])
    test_mask  = np.array([i in test_ids  for i in ids])

    print(f"\n  Train eggplants ({len(train_ids)}): {sorted(train_ids)}")
    print(f"  Test  eggplants ({len(test_ids)}):  {sorted(test_ids)}")

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== RANDOM FOREST TRAINING ===\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(OUTPUT_FILE)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    zero_mask = ~(df.iloc[:, 3:].values == 0.0).any(axis=1)
    df = df[zero_mask].reset_index(drop=True)

    X   = df.iloc[:, 3:].values.astype(float)
    ids = df.iloc[:, 1].str.replace(r'_S\d$', '', regex=True).values

    enc = LabelEncoder()
    y   = enc.fit_transform(df.iloc[:, 0].values)
    np.save(RF_CLASSES, enc.classes_)
    print(f"  Classes  : {enc.classes_}  (0={enc.classes_[0]}, 1={enc.classes_[1]})")
    print(f"  Total scans : {len(X)}")
    print(f"  Healthy: {(y == 0).sum()},  Infested: {(y == 1).sum()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\nPreprocessing...")
    X = preprocess(X)
    print(f"  Features after ratio addition: {X.shape[1]}")

    X, y, ids = remove_outliers(X, y, ids)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = grouped_split(X, y, ids)
    print(f"\n  Train scans : {len(X_train)},  Test scans: {len(X_test)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators     = 500,
        max_depth        = 12,
        min_samples_leaf = 5,
        max_features     = 'sqrt',
        class_weight     = 'balanced',
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 50}")
    print(f"  Test accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=enc.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    print(f"    [[TN={cm[0, 0]}  FP={cm[0, 1]}]]")
    print(f"    [[FN={cm[1, 0]}  TP={cm[1, 1]}]]")

    print("\n  Top channels by RF importance:")
    for rank, idx in enumerate(np.argsort(rf.feature_importances_)[::-1][:10]):
        print(f"    Rank {rank + 1:>2}: feature {idx + 1:<3}  {rf.feature_importances_[idx]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RF_MODEL_NAME, 'wb') as f:
        pickle.dump({'model': rf, 'classes': enc.classes_}, f)
    print(f"\nModel saved: {RF_MODEL_NAME}")


if __name__ == '__main__':
    main()
