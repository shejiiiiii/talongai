# src/model.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization, Reshape, Multiply, LeakyReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from config import INPUT_SHAPE, LEARNING_RATE


def se_block(x, filters: int, reduction: int = 4):
    """Squeeze-and-Excitation channel attention block."""
    se = GlobalAveragePooling1D()(x)
    se = Dense(filters // reduction, activation='relu',
               kernel_regularizer=l2(0.001))(se)
    se = Dense(filters, activation='sigmoid',
               kernel_regularizer=l2(0.001))(se)
    se = Reshape((1, filters))(se)
    return Multiply()([x, se])


def build_model() -> tf.keras.Model:
    """
    SE-CNN for 18-channel spectral input.

    Input  : (batch, 18, 1)
    Output : (batch, 1)  — sigmoid probability of 'Infested'
    """
    inputs = Input(shape=INPUT_SHAPE)

    # ── Block 1 ───────────────────────────────────────────────────────────────
    x = Conv1D(32, kernel_size=3, padding='same',
               kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = se_block(x, filters=32)
    x = MaxPooling1D(pool_size=2)(x)            # 18 → 9

    # ── Block 2 ───────────────────────────────────────────────────────────────
    x = Conv1D(64, kernel_size=3, padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = se_block(x, filters=64)
    # no pooling — keep 9 time steps

    # ── Block 3 ───────────────────────────────────────────────────────────────
    x = Conv1D(64, kernel_size=3, padding='same',
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = se_block(x, filters=64)

    x = GlobalAveragePooling1D()(x)

    # ── Classification head ───────────────────────────────────────────────────
    x = Dense(32, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.6)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    # Label smoothing prevents the model from pushing sigmoid to 0 or 1;
    # targets become 0.05 / 0.95 instead of 0 / 1.
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=['accuracy'],
    )
    return model
