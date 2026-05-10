# src/analyzer.py

import pickle

import numpy as np
from scipy.signal import savgol_filter

from config import (
    USE_RF,
    MODEL_NAME, RF_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
)

if USE_RF:
    pass   # pickle already imported above
else:
    from tensorflow.keras.models import load_model as keras_load_model


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_ratio_features(X: np.ndarray) -> np.ndarray:
    """
    Append pairwise channel ratios to the feature matrix.

    Uses the most discriminative channels (Ch4–6, Ch10–13 in 1-indexed terms)
    identified by RF feature importance. Cancels distance/intensity variation
    and exposes spectral shape differences.

    Expands 18 channels → 39 features (18 original + 21 ratios).

    Module-level so both PestAnalyzer and scripts/train_rf.py can import it
    without reaching into a private class method.
    """
    key_channels = [3, 4, 5, 9, 10, 11, 12]   # 0-indexed → Ch4–6, Ch10–13
    ratios = []
    for i in key_channels:
        for j in key_channels:
            if i < j:
                ratios.append((X[:, i] / (X[:, j] + 1e-8)).reshape(-1, 1))
    return np.hstack([X, np.hstack(ratios)])


# ─────────────────────────────────────────────────────────────────────────────
#  Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class PestAnalyzer:

    def __init__(self):
        self._model = self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        if USE_RF:
            try:
                with open(RF_MODEL_NAME, 'rb') as f:
                    payload = pickle.load(f)
                print(f"RF model loaded: {RF_MODEL_NAME}")
                return payload['model']
            except Exception as exc:
                print(f"Error loading RF model: {exc}")
                raise
        else:
            try:
                model = keras_load_model(MODEL_NAME)
                print(f"CNN model loaded: {MODEL_NAME}")
                return model
            except Exception as exc:
                print(f"Error loading CNN model: {exc}")
                raise

    # ──────────────────────────────────────────────────────────────────────────
    #  Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(readings_list: list) -> np.ndarray:
        """Savitzky-Golay smooth → SNV normalise (applied to raw 18 channels)."""
        data = np.array(readings_list, dtype=float)
        data = savgol_filter(data, window_length=3, polyorder=2, axis=1)
        mean = np.mean(data, axis=1, keepdims=True)
        std  = np.std(data,  axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    # ──────────────────────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict_raw(self, readings_list: list) -> float:
        """
        Returns a raw score in [0, 1] where > 0.5 means 'Infested'.

        NOTE — RF preprocessing order mismatch:
        Inference applies SNV before adding ratio features (SavGol → SNV → ratios).
        Training in scripts/train_rf.py uses the opposite order (SavGol → ratios
        → SNV on the 39-feature matrix). Unify both when retraining from scratch.
        """
        data = self._preprocess(readings_list)

        if USE_RF:
            data  = add_ratio_features(data)
            probs = self._model.predict_proba(data)
            return float(probs[:, 1].max())
        else:
            cnn_input = data.reshape(len(readings_list), 18, 1)
            preds     = self._model.predict(cnn_input, verbose=0)
            return float(max(p[0] for p in preds))

    def predict(self, readings_list: list) -> tuple[str, float, bool]:
        """
        Returns (label, confidence_pct, was_flipped).

        Segments whose confidence falls below CONFIDENCE_THRESHOLD or reaches
        exactly 100% are considered ambiguous and flipped to the opposite label.
        """
        raw = self.predict_raw(readings_list)

        if raw > 0.5:
            label, confidence = "Infested", raw * 100.0
        else:
            label, confidence = "Healthy",  (1.0 - raw) * 100.0

        if confidence < CONFIDENCE_THRESHOLD or confidence == 100.0:
            label = "Healthy" if label == "Infested" else "Infested"
            return label, confidence, True

        return label, confidence, False

    # ──────────────────────────────────────────────────────────────────────────
    #  Voting
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def majority_vote(votes: list) -> tuple[str, int, int]:
        """
        Any 'Infested' vote among the segments flags the whole eggplant.
        Returns (final_label, infested_count, healthy_count).
        """
        infested = votes.count("Infested")
        healthy  = votes.count("Healthy")
        winner   = "Infested" if infested >= 1 else "Healthy"
        return winner, infested, healthy
