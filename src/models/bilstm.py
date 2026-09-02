from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.dataset_config import normalize_label_name


class BiLSTMClassifier:
    def __init__(self, num_classes: int, sequence_length: int, feature_dim: int):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(self.sequence_length, self.feature_dim))
        x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def compile(self) -> None:
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, validation_split: float = 0.1, epochs: int = 20, batch_size: int = 32):
        self.compile()
        permutation = np.random.default_rng(42).permutation(len(X_train))
        return self.model.fit(
            X_train[permutation],
            y_train[permutation],
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model = keras.models.load_model(path)
        return self.model


def _normalize_label_value(value) -> str:
    if pd.isna(value):
        return 'nan'
    return normalize_label_name(value)


def build_label_mapping(df: pd.DataFrame, label_col: str = 'label') -> dict:
    labels = sorted({_normalize_label_value(label) for label in df[label_col].tolist() if _normalize_label_value(label)})
    return {label: idx for idx, label in enumerate(labels)}


def prepare_sequence_dataset(df: pd.DataFrame, sequence_length: int = 30, label_col: str = 'label') -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError('No feature columns available for sequence preparation.')

    label_map = build_label_mapping(df, label_col=label_col)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        values = row[feature_cols].to_numpy(dtype=np.float32)
        if values.shape[0] == 0:
            continue

        if len(values) < sequence_length:
            padded = np.pad(values, (0, max(sequence_length - len(values), 0)), mode='constant')
            seq = np.tile(padded, (sequence_length, 1))[:sequence_length]
        else:
            seq = np.tile(values, (sequence_length, 1))[:sequence_length]

        sequences.append(seq.reshape(sequence_length, -1))
        labels.append(label_map[_normalize_label_value(row[label_col])])

    if not sequences:
        raise ValueError('No valid sequences could be generated from the dataset.')

    return np.stack(sequences, axis=0).astype(np.float32), np.asarray(labels, dtype=np.int32)


def train_bilstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    sequence_length: int,
    feature_dim: int,
    model_path: str = 'models/sign_bilstm.keras',
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> tuple[keras.Model, tf.keras.callbacks.History]:
    model = BiLSTMClassifier(num_classes=num_classes, sequence_length=sequence_length, feature_dim=feature_dim)
    history = model.train(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
    )
    model.save(model_path)
    return model.model, history


def train_bilstm_from_csv(
    input_csv: str,
    model_path: str = 'models/sign_bilstm.keras',
    sequence_length: int = 30,
    label_col: str = 'label',
    epochs: int = 20,
    batch_size: int = 32,
) -> tuple[keras.Model, tf.keras.callbacks.History, list[str]]:
    df = pd.read_csv(input_csv)
    X_train, y_train = prepare_sequence_dataset(df, sequence_length=sequence_length, label_col=label_col)
    class_names = sorted({normalize_label_name(name) for name in df[label_col].tolist() if normalize_label_name(name)})
    feature_dim = X_train.shape[-1]
    num_classes = len(class_names)
    model, history = train_bilstm_model(
        X_train,
        y_train,
        num_classes=num_classes,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
    )
    return model, history, class_names
