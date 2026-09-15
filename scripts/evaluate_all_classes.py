#!/usr/bin/env python3
"""Evaluate live prediction accuracy across all 26 ASL alphabet classes using real images."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output and unbuffered flush
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from app import (
    MODEL_PATH,
    _load_model_classes,
    build_live_sequence,
    resolve_prediction_label,
)
from src.data.dataset_config import ASL_FULL_CLASS_LIST
from src.data.landmarks import LandmarkExtractor
from src.processing.features import build_feature_dataset


def evaluate_all_classes(samples_per_class: int = 15):
    root = Path('data/online_asl')
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    class_names = _load_model_classes(MODEL_PATH)
    extractor = LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.30)

    overall_correct = 0
    overall_total = 0
    per_class_results = {}

    print(f"\nEvaluating accuracy across all 26 classes ({samples_per_class} images per class)...", flush=True)
    print("-" * 65, flush=True)

    for letter in sorted(ASL_FULL_CLASS_LIST):
        class_dir = root / letter
        if not class_dir.exists():
            print(f"Class {letter}: directory missing!", flush=True)
            continue

        images = sorted(class_dir.glob('*.jpg'))
        if not images:
            print(f"Class {letter}: no images found!", flush=True)
            continue

        # Hold-out samples from the end of the folder
        eval_samples = images[-samples_per_class:] if len(images) >= samples_per_class else images
        class_correct = 0
        class_total = 0
        confidences = []

        for img_path in eval_samples:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            vector = extractor.extract_landmarks_from_frame(frame)
            if vector is None:
                # Landmark could not be detected in image
                continue

            # Feature extraction
            feat_df = pd.DataFrame(np.asarray(vector, dtype=float).reshape(1, -1))
            feat_df.columns = [f'f{i}' for i in range(feat_df.shape[1])]
            enriched = build_feature_dataset(feat_df)
            arr = enriched.drop(columns=['label'], errors='ignore').to_numpy(dtype=np.float32)

            seq = build_live_sequence(arr[0], sequence_length=30)
            preds = model.predict(seq, verbose=0)[0]

            pred_res = resolve_prediction_label(preds, class_names=class_names, threshold=0.03)
            pred_label = pred_res['label']
            conf = pred_res['confidence']

            is_correct = (pred_label == letter)
            if is_correct:
                class_correct += 1
            class_total += 1
            confidences.append(conf)

        acc = (class_correct / class_total * 100) if class_total > 0 else 0.0
        avg_conf = (sum(confidences) / len(confidences) * 100) if confidences else 0.0
        per_class_results[letter] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': acc,
            'avg_confidence': avg_conf,
        }

        overall_correct += class_correct
        overall_total += class_total

        status = 'EXCELLENT' if acc >= 90.0 else ('GOOD' if acc >= 75.0 else 'FAIR')
        print(f"[{status:9s}] Class {letter}: {class_correct:2d}/{class_total:2d} ({acc:5.1f}%) | Avg Confidence: {avg_conf:5.1f}%", flush=True)

    total_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0

    print("-" * 65, flush=True)
    print("TOTAL ACCURACY ACROSS ALL 26 CLASSES ON REAL IMAGES:")
    print(f"  Total Valid Images:  {overall_total}")
    print(f"  Correct Signs:       {overall_correct}")
    print(f"  Overall Accuracy:    {total_accuracy:.2f}%")
    print("-" * 65 + "\n", flush=True)

    return per_class_results, total_accuracy


if __name__ == '__main__':
    evaluate_all_classes(samples_per_class=15)

