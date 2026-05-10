# scripts/baseline.py
"""
Quick baseline comparison: SVM and Random Forest on the spectral dataset.

Uses a simple random train/test split (not grouped by eggplant ID), so
results may be slightly optimistic due to same-eggplant leakage. For a
rigorous evaluation matched to the CNN, use scripts/evaluate_cnn.py instead.

Run from the project root:
    python scripts/baseline.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config import DATA_FILE, TEST_SIZE, RANDOM_STATE

# CSV column layout: Label(0), Eggplant_ID(1), Timestamp(2), Ch_1..Ch_18(3-20)
_SPECTRAL_START = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_flat_data() -> tuple:
    """
    Load and preprocess DATA_FILE, returning flat (n, 18) feature arrays.
    Same SavGol + SNV pipeline as the CNN, without reshaping or augmentation.
    """
    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    spectral = df.iloc[:, _SPECTRAL_START:].values
    df = df[~(spectral == 0.0).any(axis=1)].reset_index(drop=True)

    X     = df.iloc[:, _SPECTRAL_START:].values.astype(float)
    y_raw = df.iloc[:, 0].values

    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (
        np.std(X, axis=1, keepdims=True) + 1e-8
    )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return X, y, encoder.classes_


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    X, y, classes = load_flat_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    models = {
        'SVM (RBF)': SVC(
            kernel       = 'rbf',
            C            = 10,
            gamma        = 'scale',
            class_weight = 'balanced',
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators = 200,
            class_weight = 'balanced',
            random_state = RANDOM_STATE,
        ),
    }

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print('=' * 50)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"  Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=classes))

    # RF channel importances (18 channels only — no ratio features here)
    rf = models['Random Forest']
    print("=== RF FEATURE IMPORTANCES ===")
    for rank, idx in enumerate(np.argsort(rf.feature_importances_)[::-1]):
        print(f"  Rank {rank + 1:>2}: Ch{idx + 1:<3}  {rf.feature_importances_[idx]:.4f}")


if __name__ == '__main__':
    main()
