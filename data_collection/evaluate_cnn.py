# data_collection/evaluate_cnn.py
"""5-fold leave-one-eggplant-out evaluation of the CNN model."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from data_collection.config_collect import OUTPUT_FILE

MODEL_FILE = "data_collection/eggplant_pest_model_v2.keras"
K          = 5

# ── Load & preprocess ─────────────────────────────────────────────────────────
df = pd.read_csv(OUTPUT_FILE)
df.iloc[:, 0] = df.iloc[:, 0].str.strip().str.title()

zero_mask = ~(df.iloc[:, 3:].values == 0.0).any(axis=1)
df = df[zero_mask].reset_index(drop=True)

df['Base_ID'] = df['Eggplant_ID'].str.replace(r'_S\d$', '', regex=True)

X      = df.iloc[:, 3:-1].values.astype(float)
ids    = df['Base_ID'].values
labels = df.iloc[:, 0].values

X = savgol_filter(X, window_length=3, polyorder=2, axis=1)
X = (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

# Outlier removal
keep = np.ones(len(X), dtype=bool)
for eid in np.unique(ids):
    mask  = ids == eid
    norms = np.linalg.norm(X[mask], axis=1)
    mu, sig = norms.mean(), norms.std()
    outlier = (norms < mu - 2.5*sig) | (norms > mu + 2.5*sig)
    keep[np.where(mask)[0][outlier]] = False

X      = X[keep]
ids    = ids[keep]
labels = labels[keep]

enc = LabelEncoder()
y   = enc.fit_transform(labels)
X_cnn = X.reshape(X.shape[0], 18, 1)

model = tf.keras.models.load_model(MODEL_FILE)

# ── Build stratified folds by eggplant ID ─────────────────────────────────────
unique_ids = np.array(sorted(set(ids)))
h_ids   = [i for i in unique_ids if labels[ids == i][0] == 'Healthy']
inf_ids = [i for i in unique_ids if labels[ids == i][0] == 'Infested']

rng = np.random.default_rng(42)
rng.shuffle(h_ids);  rng.shuffle(inf_ids)

h_folds   = [h_ids[i::K]   for i in range(K)]
inf_folds = [inf_ids[i::K] for i in range(K)]

# ── Cross-validation ──────────────────────────────────────────────────────────
all_true, all_pred = [], []
fold_accs = []

print('=== CNN 5-FOLD EVALUATION ===\n')
for fold in range(K):
    test_ids  = set(h_folds[fold] + inf_folds[fold])
    te_mask   = np.array([i in test_ids for i in ids])
    X_te, y_te = X_cnn[te_mask], y[te_mask]

    probs = model.predict(X_te, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)
    acc   = (preds == y_te).mean()
    fold_accs.append(acc)
    all_true.extend(y_te)
    all_pred.extend(preds)

    h_test   = sorted(e for e in test_ids if labels[ids == e][0] == 'Healthy')
    inf_test = sorted(e for e in test_ids if labels[ids == e][0] == 'Infested')
    print(f"Fold {fold+1} | Accuracy: {acc*100:.2f}%")
    print(f"  H  : {h_test}")
    print(f"  I  : {inf_test}\n")

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print('='*60)
print(f"Mean: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%\n")
print(classification_report(all_true, all_pred,
      target_names=enc.classes_, digits=3))

cm = confusion_matrix(all_true, all_pred)
print("Confusion matrix:")
print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]]")
print(f"  [[FN={cm[1,0]}  TP={cm[1,1]}]]")
