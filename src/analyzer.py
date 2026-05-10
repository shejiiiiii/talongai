# src/analyzer.py
"""
Pest analyzer — CNN (default).

To switch to Random Forest:
  1. Comment out the entire "── CNN ──" block.
  2. Uncomment the entire "── RF ──" block.
  3. In config.py, confirm RF_MODEL_NAME points to the right .pkl file.
"""

import numpy as np
from scipy.signal import savgol_filter

from config import MODEL_NAME, CONFIDENCE_THRESHOLD

# ── CNN imports ────────────────────────────────────────────────────────────────
# import tensorflow as tf
# from tensorflow.keras.models import load_model as keras_load_model

# ── RF imports (uncomment to switch) ──────────────────────────────────────────
import pickle
from config import RF_MODEL_NAME


class PestAnalyzer:

    def __init__(self):
        self._model = self._load_model()

    # ──────────────────────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        # ── CNN ────────────────────────────────────────────────────────────────
        # try:
        #     model = keras_load_model(MODEL_NAME)
        #     print(f"CNN model loaded: {MODEL_NAME}")
        #     return model
        # except Exception as exc:
        #     print(f"Error loading CNN model: {exc}")
        #     raise

        # ── RF (uncomment block below, comment out CNN block above) ───────────
        try:
            with open(RF_MODEL_NAME, 'rb') as f:
                payload = pickle.load(f)
            print(f"RF model loaded: {RF_MODEL_NAME}")
            return payload['model']
        except Exception as exc:
            print(f"Error loading RF model: {exc}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(readings_list: list) -> np.ndarray:
        """Savitzky-Golay smooth → SNV normalise."""
        data = np.array(readings_list, dtype=float)
        data = savgol_filter(data, window_length=3, polyorder=2, axis=1)
        mean = np.mean(data, axis=1, keepdims=True)
        std  = np.std(data,  axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    # ── RF ratio features (uncomment when switching to RF) ────────────────────
    @staticmethod
    def _add_ratio_features(X: np.ndarray) -> np.ndarray:
    #     """
    #     Pairwise channel ratios between the most discriminative channels.
    #     Cancels distance/intensity effects and exposes spectral shape.
    #     """
        key_channels = [3, 4, 5, 9, 10, 11, 12]  # 0-indexed
        ratios = []
        for i in key_channels:
            for j in key_channels:
                if i < j:
                    ratios.append((X[:, i] / (X[:, j] + 1e-8)).reshape(-1, 1))
        return np.hstack([X, np.hstack(ratios)])

    # ──────────────────────────────────────────────────────────────────────────
    #  Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict_raw(self, readings_list: list) -> float:
        """
        Returns a raw sigmoid score in [0, 1].
        Values > 0.5 correspond to 'Infested'.
        """
        data = self._preprocess(readings_list)

        # ── CNN ────────────────────────────────────────────────────────────────
        # cnn_input = data.reshape(len(readings_list), 18, 1)
        # preds = self._model.predict(cnn_input, verbose=0)
        # return float(max(p[0] for p in preds))

        # ── RF (uncomment block below, comment out CNN block above) ───────────
        data  = self._add_ratio_features(data)
        probs = self._model.predict_proba(data)
        return float(probs[:, 1].max())

    def predict(self, readings_list: list) -> tuple[str, float, bool]:
        """
        Returns (label, confidence_pct, was_flipped).

        Segments whose confidence < CONFIDENCE_THRESHOLD (or == 100 %)
        are considered ambiguous and flipped to the opposite label.
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
