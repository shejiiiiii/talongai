# scripts/evaluate_cnn.py
"""
K-fold leave-one-eggplant-out evaluation of the trained CNN model.

Evaluates on DATA_FILE (the training dataset) using stratified folds built
at the eggplant level, so no individual eggplant leaks across folds.

Run from the project root:
    python scripts/evaluate_cnn.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from config import (
    DATA_FILE, MODEL_NAME,
    EVAL_FOLDS, RANDOM_STATE,
)
from src.dataset import remove_outlier_scans

# CSV column layout: Label(0), Eggplant_ID(1), Timestamp(2), Ch_1..Ch_18(3-20)
_SPECTRAL_START = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Load & preprocess
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    """
    Load DATA_FILE, preprocess, and remove outliers.

    Returns
    -------
    X      : (n, 18)       — SNV-normalised spectral features
    ids    : (n,)          — base eggplant ID per scan (e.g. 'H01')
    labels : (n,)          — string class labels ('Healthy' / 'Infested')
    """
    df = pd.read_csv(DATA_FILE)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    zero_mask = ~(df.iloc[:, _SPECTRAL_START:].values == 0.0).any(axis=1)
    df = df[zero_mask].reset_index(drop=True)

    ids    = df.iloc[:, 1].str.replace(r'_S\d$', '', regex=True).values
    X      = df.iloc[:, _SPECTRAL_START:].values.astype(float)
    labels = df.iloc[:, 0].values

    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (
        np.std(X, axis=1, keepdims=True) + 1e-8
    )

    X, labels, ids = remove_outlier_scans(X, labels, ids)
    return X, ids, labels


# ─────────────────────────────────────────────────────────────────────────────
#  Fold builder
# ─────────────────────────────────────────────────────────────────────────────

def build_folds(ids: np.ndarray, labels: np.ndarray) -> list:
    """
    Build EVAL_FOLDS stratified folds by eggplant ID.

    Healthy and infested ID lists are shuffled then sliced independently so
    each fold contains a proportional mix of both classes.

    Returns a list of (test_id_set, h_test_ids, inf_test_ids) tuples.
    """
    unique_ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(RANDOM_STATE)

    h_ids   = [i for i in unique_ids if labels[ids == i][0] == 'Healthy']
    inf_ids = [i for i in unique_ids if labels[ids == i][0] == 'Infested']
    rng.shuffle(h_ids)
    rng.shuffle(inf_ids)

    h_folds   = [h_ids[i::EVAL_FOLDS]   for i in range(EVAL_FOLDS)]
    inf_folds = [inf_ids[i::EVAL_FOLDS] for i in range(EVAL_FOLDS)]

    return [
        (set(h_folds[i] + inf_folds[i]), h_folds[i], inf_folds[i])
        for i in range(EVAL_FOLDS)
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"=== CNN {EVAL_FOLDS}-FOLD LEAVE-ONE-EGGPLANT-OUT EVALUATION ===\n")
    print(f"Model : {MODEL_NAME}")
    print(f"Data  : {DATA_FILE}\n")

    X, ids, labels = load_data()

    enc   = LabelEncoder()
    y     = enc.fit_transform(labels)
    X_cnn = X.reshape(X.shape[0], 18, 1)

    model  = tf.keras.models.load_model(MODEL_NAME)
    folds  = build_folds(ids, labels)

    all_true, all_pred = [], []
    fold_accs = []

    for fold_idx, (test_ids, h_test, inf_test) in enumerate(folds):
        te_mask    = np.array([i in test_ids for i in ids])
        X_te, y_te = X_cnn[te_mask], y[te_mask]

        probs = model.predict(X_te, verbose=0).flatten()
        preds = (probs > 0.5).astype(int)
        acc   = (preds == y_te).mean()

        fold_accs.append(acc)
        all_true.extend(y_te)
        all_pred.extend(preds)

        print(f"Fold {fold_idx + 1} | Accuracy: {acc * 100:.2f}%")
        print(f"  H  : {sorted(h_test)}")
        print(f"  I  : {sorted(inf_test)}\n")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    print('=' * 60)
    print(
        f"Mean : {np.mean(fold_accs) * 100:.2f}% "
        f"± {np.std(fold_accs) * 100:.2f}%\n"
    )
    print(classification_report(
        all_true, all_pred,
        target_names=enc.classes_,
        digits=3,
    ))

    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix:")
    print(f"  [[TN={cm[0, 0]}  FP={cm[0, 1]}]]")
    print(f"  [[FN={cm[1, 0]}  TP={cm[1, 1]}]]")


if __name__ == '__main__':
    main()
