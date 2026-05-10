# data_collection/dataset.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_collection.config_collect import OUTPUT_FILE

INPUT_SHAPE  = (18, 1)
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
#  Outlier removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_outlier_scans(
    X: np.ndarray, y: np.ndarray, ids: np.ndarray,
    z_threshold: float = 2.5,
) -> tuple:
    """Drop scans that are statistical outliers within their own eggplant."""
    keep = np.ones(len(X), dtype=bool)
    for eid in np.unique(ids):
        mask  = ids == eid
        norms = np.linalg.norm(X[mask], axis=1)
        mu, sig = norms.mean(), norms.std()
        outlier = (norms < mu - z_threshold * sig) | (norms > mu + z_threshold * sig)
        keep[np.where(mask)[0][outlier]] = False
    print(f"  Outlier removal: dropped {(~keep).sum()}, kept {keep.sum()}")
    return X[keep], y[keep], ids[keep]


# ─────────────────────────────────────────────────────────────────────────────
#  Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_data():
    """
    Returns (X, y, ids, classes) where X is (n, 18, 1).
    ids is the base eggplant ID (strip _S0 / _S1 suffix).
    """
    print(f"Loading dataset: {OUTPUT_FILE}")
    try:
        df = pd.read_csv(OUTPUT_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"{OUTPUT_FILE} not found. Run collect.py first.")

    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    # Drop zero readings — spectral data starts at column 3
    zero_mask = ~(df.iloc[:, 3:].values == 0.0).any(axis=1)
    df = df[zero_mask].reset_index(drop=True)

    # Base ID strips the _S0 / _S1 sensor suffix
    df['Base_ID'] = df['Eggplant_ID'].str.replace(r'_S\d$', '', regex=True)

    X    = df.iloc[:, 3:-1].values.astype(float)  # spectral cols (drop Base_ID)
    y_raw = df.iloc[:, 0].values
    ids  = df['Base_ID'].values

    print("  Applying Savitzky-Golay smoothing...")
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)

    print("  Applying SNV normalisation...")
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

    X, y_raw_clean, ids = remove_outlier_scans(X, y_raw, ids)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw_clean)
    np.save('data_collection/classes_v2.npy', encoder.classes_)
    print(f"  Classes: {encoder.classes_}")

    X = X.reshape(X.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1])
    return X, y, ids, encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_data(
    X: np.ndarray, y: np.ndarray,
    copies: int = 2, noise_std: float = 0.02, scale_range: float = 0.05,
) -> tuple:
    X_aug, y_aug = [X], [y]
    for _ in range(copies):
        noise = np.random.normal(0, noise_std, X.shape)
        scale = np.random.uniform(1 - scale_range, 1 + scale_range,
                                  (X.shape[0], 1, 1))
        X_aug.append(X * scale + noise)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-eggplant grouped train/test split
# ─────────────────────────────────────────────────────────────────────────────

def get_train_test_split() -> tuple:
    """
    Splits by eggplant ID so no individual eggplant leaks between sets.
    Returns (X_train, X_test, y_train, y_test, class_weights_dict).
    """
    X, y, ids, classes = load_and_preprocess_data()

    unique_ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_ids)

    split_idx = int(len(unique_ids) * 0.8)
    train_ids = set(unique_ids[:split_idx])
    test_ids  = set(unique_ids[split_idx:])

    train_mask = np.array([i in train_ids for i in ids])
    test_mask  = np.array([i in test_ids  for i in ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\n  Train eggplants : {sorted(train_ids)}")
    print(f"  Test  eggplants : {sorted(test_ids)}")
    print(f"  Train scans: {len(X_train)},  Test scans: {len(X_test)}")

    print("  Augmenting training set...")
    X_train, y_train = augment_data(X_train, y_train, copies=2)
    print(f"  Training scans after augmentation: {len(X_train)}")

    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    return X_train, X_test, y_train, y_test, dict(enumerate(weights))
