# src/dataset.py
"""
Dataset loading, preprocessing, augmentation, and train/test splitting.

Replaces both src/dataset.py and data_collection/dataset.py.

Key improvements over the old src/dataset.py:
  - Fixes spectral column index (was 2, should be 3 — old version included Timestamp)
  - Grouped split by eggplant ID prevents individual eggplants from appearing
    in both train and test sets (data leakage fix)
  - Stratified ID split keeps healthy/infested balance consistent across sets
  - Per-eggplant outlier removal before splitting
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from config import (
    DATA_FILE, INPUT_SHAPE,
    RANDOM_STATE, Z_THRESHOLD, CNN_CLASSES,
)

# CSV column layout: Label(0), Eggplant_ID(1), Timestamp(2), Ch_1..Ch_18(3-20)
_SPECTRAL_START = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Outlier removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_outlier_scans(
    X: np.ndarray,
    y_raw: np.ndarray,
    ids: np.ndarray,
    z_threshold: float = Z_THRESHOLD,
) -> tuple:
    """
    Drop scans that are statistical outliers within their own eggplant group.
    Operates on raw (pre-encoded) labels so it can be called before LabelEncoder.
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
    return X[keep], y_raw[keep], ids[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    copies: int        = 2,
    noise_std: float   = 0.02,
    scale_range: float = 0.05,
) -> tuple:
    """
    Create augmented copies of each training sample.

    copies      : extra versions of each sample to generate
    noise_std   : std-dev of additive Gaussian noise (simulates sensor noise)
    scale_range : ± range for multiplicative scaling (simulates bulb variation)
    """
    X_aug, y_aug = [X], [y]
    for _ in range(copies):
        noise = np.random.normal(0, noise_std, X.shape)
        scale = np.random.uniform(
            1 - scale_range, 1 + scale_range,
            (X.shape[0], 1, 1),
        )
        X_aug.append(X * scale + noise)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────────────────────────────────────
#  Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_data() -> tuple:
    """
    Load DATA_FILE, clean, smooth, normalise, remove outliers, encode labels.

    Returns
    -------
    X       : (n_samples, 18, 1)  — reshaped for CNN input
    y       : (n_samples,)        — integer-encoded labels (0=Healthy, 1=Infested)
    ids     : (n_samples,)        — base eggplant ID per scan (e.g. 'H01', 'I03')
    classes : ndarray of class name strings
    """
    print(f"Loading dataset: {DATA_FILE}")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Run scripts/collect.py first."
        )

    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    # Drop rows where any spectral channel is exactly zero
    spectral_raw = df.iloc[:, _SPECTRAL_START:].values
    df = df[~(spectral_raw == 0.0).any(axis=1)].reset_index(drop=True)

    # Strip _S0 / _S1 sensor suffix to get the per-eggplant base ID
    ids   = df.iloc[:, 1].str.replace(r'_S\d$', '', regex=True).values
    X     = df.iloc[:, _SPECTRAL_START:].values.astype(float)
    y_raw = df.iloc[:, 0].values

    print("  Applying Savitzky-Golay smoothing...")
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)

    print("  Applying SNV normalisation...")
    X = (X - np.mean(X, axis=1, keepdims=True)) / (
        np.std(X, axis=1, keepdims=True) + 1e-8
    )

    X, y_raw, ids = remove_outlier_scans(X, y_raw, ids)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    np.save(CNN_CLASSES, encoder.classes_)
    print(f"  Classes : {encoder.classes_}")
    print(f"  Healthy : {(y == 0).sum()}, Infested: {(y == 1).sum()}")

    # Reshape for CNN: (n, 18) → (n, 18, 1)
    X = X.reshape(X.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1])

    return X, y, ids, encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
#  Grouped train/test split
# ─────────────────────────────────────────────────────────────────────────────

def get_train_test_split() -> tuple:
    """
    Split by eggplant ID so no individual eggplant appears in both sets.

    Healthy and infested ID lists are shuffled and split independently so
    the class ratio stays consistent between train and test.

    Returns (X_train, X_test, y_train, y_test, class_weights_dict).
    Training set is augmented; test set is kept clean.
    """
    X, y, ids, _ = load_and_preprocess_data()

    rng = np.random.default_rng(RANDOM_STATE)

    # Separate IDs by class using the label of the first scan for each ID
    unique_ids = np.array(sorted(set(ids)))
    h_ids   = [i for i in unique_ids if y[ids == i][0] == 0]
    inf_ids = [i for i in unique_ids if y[ids == i][0] == 1]
    rng.shuffle(h_ids)
    rng.shuffle(inf_ids)

    n_h_test  = max(1, int(len(h_ids)   * 0.2))
    n_inf_test = max(1, int(len(inf_ids) * 0.2))

    test_ids  = set(h_ids[:n_h_test]   + inf_ids[:n_inf_test])
    train_ids = set(h_ids[n_h_test:]   + inf_ids[n_inf_test:])

    train_mask = np.array([i in train_ids for i in ids])
    test_mask  = np.array([i in test_ids  for i in ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n  Train eggplants ({len(train_ids)}): {sorted(train_ids)}")
    print(f"  Test  eggplants ({len(test_ids)}):  {sorted(test_ids)}")
    print(f"  Train scans : {len(X_train)},  Test scans: {len(X_test)}")

    print("  Augmenting training set...")
    X_train, y_train = augment_data(X_train, y_train, copies=2)
    print(f"  Training scans after augmentation: {len(X_train)}")

    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    return X_train, X_test, y_train, y_test, dict(enumerate(weights))
