# src/train.py

from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from src.dataset import get_train_test_split
from src.model   import build_model
from config import MODEL_NAME, BATCH_SIZE, EPOCHS

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, weights = get_train_test_split()

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_NAME, save_best_only=True,
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
    print(f"Model saved to: {MODEL_NAME}")
