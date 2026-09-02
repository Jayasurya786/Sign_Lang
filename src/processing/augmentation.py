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


def augment_landmarks(df: pd.DataFrame, augment_factor: int = 3, rotation_range: tuple[float, float] = (-15.0, 15.0), scale_range: tuple[float, float] = (0.9, 1.1), noise_std: float = 0.02) -> pd.DataFrame:
    feature_columns = [col for col in df.columns if col != 'label']
    if not feature_columns:
        raise ValueError('No feature columns available for augmentation.')

    augmented_rows = []
    for _, row in df.iterrows():
        landmarks = _to_landmark_matrix(pd.DataFrame([row]))[0]
        for _ in range(augment_factor):
            sample = landmarks.copy()
            angle = random.uniform(rotation_range[0], rotation_range[1])
            scale = random.uniform(scale_range[0], scale_range[1])
            sample = _rotate_landmarks(sample, angle)
            sample = _scale_landmarks(sample, scale)
            sample = _add_noise(sample, std=noise_std)

            feature_values = sample.reshape(-1)
            record = {'label': row['label']}
            for idx, value in enumerate(feature_values):
                record[f'f{idx}'] = float(value)
            augmented_rows.append(record)

    augmented_df = pd.DataFrame(augmented_rows)
    return augmented_df
