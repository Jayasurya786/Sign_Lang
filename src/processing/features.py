from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


HAND_JOINTS = {
    'wrist': 0,
    'thumb_cmc': 1,
    'thumb_mcp': 2,
    'thumb_ip': 3,
    'thumb_tip': 4,
    'index_mcp': 5,
    'index_pip': 6,
    'index_dip': 7,
    'index_tip': 8,
    'middle_mcp': 9,
    'middle_pip': 10,
    'middle_dip': 11,
    'middle_tip': 12,
    'ring_mcp': 13,
    'ring_pip': 14,
    'ring_dip': 15,
    'ring_tip': 16,
    'pinky_mcp': 17,
    'pinky_pip': 18,
    'pinky_dip': 19,
    'pinky_tip': 20,
}


def _to_landmark_matrix(df: pd.DataFrame) -> np.ndarray:
    feature_cols = [col for col in df.columns if col != 'label']
    if not feature_cols:
        raise ValueError('No feature columns found for landmark extraction.')

    if all(col.startswith('f') for col in feature_cols[: min(3, len(feature_cols))]):
        values = df[feature_cols].to_numpy(dtype=float)
        return values.reshape(len(df), -1, 3)

    if all(re.match(r'^(x|y|z)_\d+$', col) for col in feature_cols):
        index_ids = sorted({int(re.search(r'(\d+)$', col).group(1)) for col in feature_cols})
        landmark_data = []
        for idx in index_ids:
            x = df[f'x_{idx}'].to_numpy(dtype=float)
            y = df[f'y_{idx}'].to_numpy(dtype=float)
            z = df[f'z_{idx}'].to_numpy(dtype=float)
            landmark_data.append(np.column_stack([x, y, z]))
        return np.stack(landmark_data, axis=1)

    raise ValueError('Unsupported landmark column format. Expected f0.. or x_i/y_i/z_i naming.')


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _compute_sample_features(sample: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}

    finger_pairs = [
        ('wrist', 'thumb_tip'),
        ('wrist', 'index_tip'),
        ('wrist', 'middle_tip'),
        ('wrist', 'ring_tip'),
        ('wrist', 'pinky_tip'),
        ('thumb_tip', 'index_tip'),
        ('thumb_tip', 'middle_tip'),
        ('thumb_tip', 'ring_tip'),
        ('thumb_tip', 'pinky_tip'),
        ('index_tip', 'middle_tip'),
        ('middle_tip', 'ring_tip'),
        ('ring_tip', 'pinky_tip'),
        ('index_tip', 'pinky_tip'),
        ('index_mcp', 'index_tip'),
        ('middle_mcp', 'middle_tip'),
        ('ring_mcp', 'ring_tip'),
        ('pinky_mcp', 'pinky_tip'),
        ('thumb_tip', 'index_mcp'),
        ('thumb_tip', 'index_pip'),
        ('thumb_tip', 'middle_mcp'),
        ('thumb_tip', 'middle_pip'),
        ('thumb_tip', 'ring_mcp'),
        ('thumb_tip', 'ring_pip'),
        ('thumb_tip', 'pinky_mcp'),
    ]

    for name_a, name_b in finger_pairs:
        idx_a = HAND_JOINTS[name_a]
        idx_b = HAND_JOINTS[name_b]
        features[f'dist_{name_a}_{name_b}'] = _euclidean_distance(sample[idx_a], sample[idx_b])

    # Extension / curl ratios: ratio of wrist-to-fingertip vs wrist-to-MCP
    wrist_pt = sample[HAND_JOINTS['wrist']]
    for finger_name, tip_name, mcp_name in [
        ('thumb', 'thumb_tip', 'thumb_mcp'),
        ('index', 'index_tip', 'index_mcp'),
        ('middle', 'middle_tip', 'middle_mcp'),
        ('ring', 'ring_tip', 'ring_mcp'),
        ('pinky', 'pinky_tip', 'pinky_mcp'),
    ]:
        tip_pt = sample[HAND_JOINTS[tip_name]]
        mcp_pt = sample[HAND_JOINTS[mcp_name]]
        d_tip = _euclidean_distance(wrist_pt, tip_pt)
        d_mcp = _euclidean_distance(wrist_pt, mcp_pt)
        features[f'ratio_ext_{finger_name}'] = float(d_tip / (d_mcp + 1e-6))

    angle_triplets = [
        ('wrist', 'index_mcp', 'index_tip'),
        ('wrist', 'middle_mcp', 'middle_tip'),
        ('wrist', 'ring_mcp', 'ring_tip'),
        ('wrist', 'pinky_mcp', 'pinky_tip'),
        ('thumb_mcp', 'thumb_tip', 'index_tip'),
        ('index_mcp', 'index_pip', 'index_tip'),
        ('middle_mcp', 'middle_pip', 'middle_tip'),
        ('ring_mcp', 'ring_pip', 'ring_tip'),
        ('pinky_mcp', 'pinky_pip', 'pinky_tip'),
        ('thumb_mcp', 'thumb_ip', 'thumb_tip'),
        ('index_pip', 'index_dip', 'index_tip'),
        ('middle_pip', 'middle_dip', 'middle_tip'),
        ('ring_pip', 'ring_dip', 'ring_tip'),
        ('pinky_pip', 'pinky_dip', 'pinky_tip'),
    ]

    for name_a, name_b, name_c in angle_triplets:
        idx_a = HAND_JOINTS[name_a]
        idx_b = HAND_JOINTS[name_b]
        idx_c = HAND_JOINTS[name_c]
        features[f'angle_{name_a}_{name_b}_{name_c}'] = _angle_between(sample[idx_a], sample[idx_b], sample[idx_c])

    # 3D Palm normal orientation vector
    v1 = sample[HAND_JOINTS['index_mcp']] - wrist_pt
    v2 = sample[HAND_JOINTS['pinky_mcp']] - wrist_pt
    normal = np.cross(v1, v2)
    norm_len = float(np.linalg.norm(normal))
    if norm_len > 1e-6:
        normal /= norm_len
    features['palm_normal_x'] = float(normal[0])
    features['palm_normal_y'] = float(normal[1])
    features['palm_normal_z'] = float(normal[2])

    return features


def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    landmark_matrix = _to_landmark_matrix(df)
    rows = []

    for sample in landmark_matrix:
        rows.append(_compute_sample_features(sample))

    feature_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def normalize_feature_columns(df: pd.DataFrame, feature_cols: Iterable[str] | None = None) -> pd.DataFrame:
    normalized = df.copy()
    if feature_cols is None:
        feature_cols = [col for col in normalized.columns if col not in {'label'}]

    # A single live frame has no meaningful min/max range. Normalizing it
    # against itself turns every feature into zero and makes predictions
    # identical, so preserve its calculated landmark features.
    if len(normalized) < 2:
        return normalized

    for col in feature_cols:
        values = normalized[col].astype(float)
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            normalized[col] = 0.0
        else:
            normalized[col] = (values - min_val) / (max_val - min_val)
    return normalized


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_geometric_features(df)
    return enriched
