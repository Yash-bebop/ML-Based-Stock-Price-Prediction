"""Neural model builders used by the notebook."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def make_lstm_sequences(values: np.ndarray, targets: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert feature arrays into rolling LSTM sequences."""

    X_seq, y_seq = [], []
    for idx in range(look_back, len(values)):
        X_seq.append(values[idx - look_back : idx])
        y_seq.append(targets[idx])
    return np.asarray(X_seq), np.asarray(y_seq)


def build_lstm(input_shape, learning_rate: float = 1e-3) -> Sequential:
    """Create the LSTM regressor from the notebook's sequence experiment."""

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.25),
            LSTM(32),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_neural_stacker(input_dim: int, learning_rate: float = 1e-3) -> Sequential:
    """Create the MLP stacker that learns over base-model signals."""

    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dropout(0.10),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def default_callbacks(patience: int = 8):
    """Callbacks used by both LSTM and stacker training."""

    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=max(2, patience // 2), factor=0.5),
    ]
