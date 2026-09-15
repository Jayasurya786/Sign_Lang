#!/usr/bin/env python3
"""Train the upgraded Sign Language BiLSTM model on the combined dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from app import (
    MODEL_CLASSES_PATH,
    MODEL_METRICS_PATH,
    MODEL_PATH,
    _build_model_quality_report,
    _save_model_classes,
    _save_model_quality_report,
    load_image_dataset_as_dataframe,
    resolve_training_dataset,
)
from src.models.bilstm import train_bilstm_model
from src.processing.augmentation import augment_landmarks
from src.processing.cleaning import clean_landmark_dataset
from src.processing.features import build_feature_dataset
from src.processing.sequences import create_fixed_length_sequences


def main():
    dataset_path = resolve_training_dataset()
    print(f'Loading dataset from: {dataset_path}')
    train_df = load_image_dataset_as_dataframe(dataset_path)
    print(f'Raw dataset rows: {len(train_df)}')

    normalized_df = train_df.copy()
    normalized_df['label'] = normalized_df['label'].astype(str).str.strip().str.upper()

    print('Cleaning landmark dataset...')
    cleaned = clean_landmark_dataset(normalized_df)
    print(f'Cleaned rows: {len(cleaned)}')

    print('Applying landmark augmentation (mirroring, rotation, jitter)...')
    augmented = augment_landmarks(cleaned, augment_factor=1, include_original=True)
    print(f'Augmented rows: {len(augmented)}')

    print('Extracting enriched geometric features (distances, extension ratios, angles, palm normal)...')
    features = build_feature_dataset(augmented)
    print(f'Enriched feature shape: {features.shape}')

    model_classes = sorted(features['label'].dropna().unique().tolist())
    num_classes = len(model_classes)
    print(f'Classes ({num_classes}): {model_classes}')

    print('Generating sequence tensors...')
    X, y = create_fixed_length_sequences(features, sequence_length=30)
    print(f'Sequence tensor X shape: {X.shape}, y shape: {y.shape}')

    feature_dim = X.shape[-1]
    epochs = 25
    batch_size = 64

    print(f'Starting BiLSTM training for {epochs} epochs (batch_size={batch_size})...')
    model, history = train_bilstm_model(
        X,
        y,
        num_classes=num_classes,
        sequence_length=30,
        feature_dim=feature_dim,
        model_path=str(MODEL_PATH),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
    )
    _save_model_classes(model_classes)
    print(f'Model saved to {MODEL_PATH} and classes saved to {MODEL_CLASSES_PATH}')

    print('Evaluating model quality and computing confusion metrics...')
    quality_report = _build_model_quality_report(model, X, y, labels=list(range(num_classes)))
    _save_model_quality_report(quality_report)
    print(f'Quality report saved to {MODEL_METRICS_PATH}')

    overall = quality_report.get('overall', {})
    print('\n================ MODEL EVALUATION RESULTS ================')
    print(f"Accuracy:  {overall.get('accuracy', 0.0) * 100:.2f}%")
    print(f"Precision: {overall.get('precision', 0.0) * 100:.2f}%")
    print(f"Recall:    {overall.get('recall', 0.0) * 100:.2f}%")
    print(f"F1 Score:  {overall.get('f1_score', 0.0) * 100:.2f}%")
    print('==========================================================\n')


if __name__ == '__main__':
    main()

