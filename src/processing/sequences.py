from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset_config import build_class_mapping, normalize_label_name


def create_fixed_length_sequences(df: pd.DataFrame, sequence_length: int = 30, label_col: str = 'label') -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError('No feature columns available for sequence preparation.')

    class_mapping = build_class_mapping(df[label_col].tolist())
    sequences = []
    labels = []

    for _, row in df.iterrows():
        values = row[feature_cols].to_numpy(dtype=float)
        seq = np.tile(values, (sequence_length, 1))
        sequences.append(seq)
        labels.append(class_mapping[normalize_label_name(row[label_col])])

    if not sequences:
        raise ValueError('No valid sequences could be generated from the dataset.')

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int32)
