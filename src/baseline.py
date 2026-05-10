# src/baseline.py
"""
Quick baseline comparison: SVM and Random Forest vs the CNN.
Run from the project root:  python -m src.baseline
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import DATA_FILE, TEST_SIZE, RANDOM_STATE
import pandas as pd


def load_flat_data():
    """Same pipeline as the CNN dataset, but returns flat (n, 18) arrays."""
    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    spectral = df.iloc[:, 2:].values
    df = df[~(spectral == 0.0).any(axis=1)]

    X = df.iloc[:, 2:].values.astype(float)
    y_raw = df.iloc[:, 0].values

    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return X, y, encoder.classes_


if __name__ == '__main__':
    X, y, classes = load_flat_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        'SVM (RBF)':     SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print('='*50)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"  Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=classes))

    # Feature importances from RF
    rf = models['Random Forest']
    importances = rf.feature_importances_
    print("\n=== RF FEATURE IMPORTANCES ===")
    for rank, idx in enumerate(np.argsort(importances)[::-1]):
        print(f"  Rank {rank+1:>2}: Ch{idx+1:<3}  {importances[idx]:.4f}")
