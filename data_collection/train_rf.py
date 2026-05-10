# data_collection/train_rf.py
"""
Train the Random Forest on the collected v5 dataset.
Use this when you want to switch the inference backend to RF.
After training, update config.py → RF_MODEL_NAME and
follow the switch instructions in src/analyzer.py.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight

from data_collection.config_collect import OUTPUT_FILE

MODEL_OUTPUT = "data_collection/eggplant_pest_rf_v4.pkl"
RANDOM_STATE = 42


def add_ratio_features(X: np.ndarray) -> np.ndarray:
    """
    Pairwise ratios of key channels.
    Cancels distance / intensity effects — only spectral shape remains.
    Key channels based on RF feature importance: Ch4, Ch5, Ch6, Ch10, Ch11, Ch12, Ch13.
    """
    key_channels = [3, 4, 5, 9, 10, 11, 12]   # 0-indexed
    ratios = []
    for i in key_channels:
        for j in key_channels:
            if i < j:
                ratios.append((X[:, i] / (X[:, j] + 1e-8)).reshape(-1, 1))
    return np.hstack([X, np.hstack(ratios)])   # 18 + 21 = 39 features


def preprocess(X: np.ndarray) -> np.ndarray:
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
    X = add_ratio_features(X)
    X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)
    return X


def remove_outliers(X, y, ids, z_threshold=2.5):
    keep = np.ones(len(X), dtype=bool)
    for eid in np.unique(ids):
        mask = ids == eid
        norms = np.linalg.norm(X[mask], axis=1)
        mu, sig = norms.mean(), norms.std()
        outlier = (norms < mu - z_threshold*sig) | (norms > mu + z_threshold*sig)
        keep[np.where(mask)[0][outlier]] = False
    print(f"  Outlier removal: dropped {(~keep).sum()}, kept {keep.sum()}")
    return X[keep], y[keep], ids[keep]


def grouped_split(X, y, ids, test_size=0.2):
    """Splits by eggplant ID — no individual leaks between train and test."""
    unique_ids = np.array(sorted(set(ids)))
    rng = np.random.default_rng(RANDOM_STATE)

    h_ids   = [i for i in unique_ids if y[ids == i][0] == 0]
    inf_ids = [i for i in unique_ids if y[ids == i][0] == 1]
    rng.shuffle(h_ids);   rng.shuffle(inf_ids)

    n_h   = max(1, int(len(h_ids)   * test_size))
    n_inf = max(1, int(len(inf_ids) * test_size))

    test_ids  = set(h_ids[:n_h]   + inf_ids[:n_inf])
    train_ids = set(h_ids[n_h:]   + inf_ids[n_inf:])

    train_mask = np.array([i in train_ids for i in ids])
    test_mask  = np.array([i in test_ids  for i in ids])

    print(f"\n  Train eggplants ({len(train_ids)}): {sorted(train_ids)}")
    print(f"  Test  eggplants ({len(test_ids)}):  {sorted(test_ids)}")

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


if __name__ == '__main__':
    print("=== RANDOM FOREST TRAINING ===\n")

    df = pd.read_csv(OUTPUT_FILE)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

    zero_mask = ~(df.iloc[:, 3:].values == 0.0).any(axis=1)
    df = df[zero_mask].reset_index(drop=True)

    X   = df.iloc[:, 3:].values.astype(float)
    ids = df['Eggplant_ID'].str.replace(r'_S\d$', '', regex=True).values

    enc = LabelEncoder()
    y   = enc.fit_transform(df.iloc[:, 0].values)
    np.save('data_collection/rf_classes.npy', enc.classes_)
    print(f"  Classes: {enc.classes_}  (0={enc.classes_[0]}, 1={enc.classes_[1]})")
    print(f"  Total scans: {len(X)}")
    print(f"  Healthy: {(y==0).sum()},  Infested: {(y==1).sum()}")

    print("\nPreprocessing...")
    X = preprocess(X)
    print(f"  Features after ratio addition: {X.shape[1]}")

    X, y, ids = remove_outliers(X, y, ids)

    X_train, X_test, y_train, y_test = grouped_split(X, y, ids)
    print(f"\n  Train scans: {len(X_train)},  Test scans: {len(X_test)}")

    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    print(f"  Class weights: {dict(enumerate(weights))}")

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators  = 500,
        max_depth     = 12,
        min_samples_leaf = 5,
        max_features  = 'sqrt',
        class_weight  = 'balanced',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  Test accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=enc.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    print(f"    [[TN={cm[0,0]}  FP={cm[0,1]}]]")
    print(f"    [[FN={cm[1,0]}  TP={cm[1,1]}]]")

    print("\n  Top channels by RF importance:")
    for rank, idx in enumerate(np.argsort(rf.feature_importances_)[::-1]):
        print(f"    Rank {rank+1:>2}: Ch{idx+1:<3}  {rf.feature_importances_[idx]:.4f}")

    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump({'model': rf, 'classes': enc.classes_}, f)
    print(f"\nModel saved: {MODEL_OUTPUT}")
