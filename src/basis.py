# baseline.py

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from config import DATA_FILE, TEST_SIZE, RANDOM_STATE


def load_flat_data():
    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    spectral_data = df.iloc[:, 2:].values
    zero_mask = ~(spectral_data == 0.0).any(axis=1)
    df = df[zero_mask]

    X = df.iloc[:, 2:].values
    y = df.iloc[:, 0].values

    # Same preprocessing as your CNN pipeline
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y, encoder.classes_


if __name__ == "__main__":
    X, y, classes = load_flat_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "SVM (RBF kernel)":   SVC(kernel='rbf', C=10, gamma='scale'),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print('='*50)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=classes))

    # Add after the Random Forest block
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n=== RANDOM FOREST FEATURE IMPORTANCES ===")
    for rank, idx in enumerate(indices):
        print(f"  Rank {rank+1:>2}: Ch{idx+1:<3}  importance: {importances[idx]:.4f}")
