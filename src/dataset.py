# src/dataset.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from config import DATA_FILE, INPUT_SHAPE, TEST_SIZE, RANDOM_STATE


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_data(
    X: np.ndarray,
    y: np.ndarray,
    copies: int   = 2,
    noise_std: float    = 0.02,
    scale_range: float  = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create augmented copies of each training sample.

    copies      : how many extra versions of each sample to make
    noise_std   : std-dev of additive Gaussian noise (simulates sensor noise)
    scale_range : ± range for multiplicative scaling (simulates bulb variation)
    """
    X_aug, y_aug = [X], [y]
    for _ in range(copies):
        noise = np.random.normal(0, noise_std, X.shape)
        scale = np.random.uniform(1 - scale_range, 1 + scale_range,
                                  (X.shape[0], 1, 1))
        X_aug.append(X * scale + noise)
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────────────────────────────────────
#  Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load DATA_FILE, clean, smooth, normalise, encode labels, and reshape for CNN.

    Returns
    -------
    X               : (n_samples, 18, 1)
    y               : (n_samples,)  — integer-encoded labels
    class_weights   : dict {0: w0, 1: w1}
    """
    print(f"Loading dataset: {DATA_FILE}")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"{DATA_FILE} not found. Run data_collection/collect.py first.")

    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    # Drop rows where any spectral channel is exactly zero
    spectral = df.iloc[:, 2:].values
    df = df[~(spectral == 0.0).any(axis=1)]

    X = df.iloc[:, 2:].values.astype(float)
    y_raw = df.iloc[:, 0].values

    # Savitzky-Golay smoothing
    print("  Applying Savitzky-Golay smoothing...")
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)

    # Standard Normal Variate (SNV) normalisation
    print("  Applying SNV normalisation...")
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    np.save('classes.npy', encoder.classes_)
    print(f"  Classes: {encoder.classes_}")

    # Class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y
    )

    # Reshape for CNN: (n, 18) → (n, 18, 1)
    X = X.reshape(X.shape[0], INPUT_SHAPE[0], INPUT_SHAPE[1])

    return X, y, dict(enumerate(weights))


def get_train_test_split() -> tuple:
    """
    Returns (X_train, X_test, y_train, y_test, class_weights_dict).
    Training set is augmented; test set is kept clean.
    """
    X, y, weights = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("  Augmenting training set...")
    X_train, y_train = augment_data(X_train, y_train, copies=2)
    print(f"  Training samples after augmentation: {len(X_train)}")

    return X_train, X_test, y_train, y_test, weights
