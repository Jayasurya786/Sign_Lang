from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from src.processing.features import _to_landmark_matrix


def _rotate_landmarks(landmarks: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])

    center = landmarks[0]
    translated = landmarks - center
    rotated = np.dot(translated, rotation.T)
    return rotated + center


def _scale_landmarks(landmarks: np.ndarray, scale: float) -> np.ndarray:
    center = landmarks[0]
    shifted = landmarks - center
    scaled = shifted * scale
    return scaled + center


def _add_noise(landmarks: np.ndarray, std: float = 0.02) -> np.ndarray:
    noise = np.random.normal(0.0, std, size=landmarks.shape)
    return landmarks + noise


def _flip_horizontal(landmarks: np.ndarray) -> np.ndarray:
    flipped = landmarks.copy()
    flipped[:, 0] = -flipped[:, 0]
    return flipped


def augment_landmarks(
    df: pd.DataFrame,
    augment_factor: int = 2,
    rotation_range: tuple[float, float] = (-12.0, 12.0),
    scale_range: tuple[float, float] = (0.92, 1.08),
    noise_std: float = 0.015,
    flip_probability: float = 0.4,
    include_original: bool = True,
) -> pd.DataFrame:
    feature_columns = [col for col in df.columns if col != 'label']
    if not feature_columns:
        raise ValueError('No feature columns available for augmentation.')

    labels = df['label'].to_list()
    landmarks_all = _to_landmark_matrix(df)  # Shape: (N, 21, 3)

    records: list[dict] = []
    num_features = landmarks_all.shape[1] * landmarks_all.shape[2]
    feature_names = [f'f{i}' for i in range(num_features)]

    for idx, (label, landmarks) in enumerate(zip(labels, landmarks_all)):
        if include_original:
            flat_orig = landmarks.reshape(-1)
            row_dict = {'label': label}
            for fn, val in zip(feature_names, flat_orig):
                row_dict[fn] = float(val)
            records.append(row_dict)

        for _ in range(augment_factor):
            sample = landmarks.copy()
            if flip_probability > 0 and random.random() < flip_probability:
                sample = _flip_horizontal(sample)

            angle = random.uniform(rotation_range[0], rotation_range[1])
            scale = random.uniform(scale_range[0], scale_range[1])
            sample = _rotate_landmarks(sample, angle)
            sample = _scale_landmarks(sample, scale)
            sample = _add_noise(sample, std=noise_std)

            flat = sample.reshape(-1)
            row_dict = {'label': label}
            for fn, val in zip(feature_names, flat):
                row_dict[fn] = float(val)
            records.append(row_dict)

    return pd.DataFrame(records)

