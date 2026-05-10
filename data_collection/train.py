# data_collection/train.py
"""Train the CNN on the collected v5 dataset."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from data_collection.dataset import get_train_test_split
from src.model import build_model

MODEL_OUTPUT = "data_collection/eggplant_pest_model_v5.keras"
BATCH_SIZE   = 32
EPOCHS       = 200

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, weights = get_train_test_split()

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_OUTPUT, save_best_only=True,
                        monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=25,
                      restore_best_weights=True),
    ]

    print("\nStarting CNN training...")
    model.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_test, y_test),
        class_weight    = weights,
        callbacks       = callbacks,
        verbose         = 1,
    )

    print("\n─── Final evaluation ───────────────────────────")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Model saved  : {MODEL_OUTPUT}")
