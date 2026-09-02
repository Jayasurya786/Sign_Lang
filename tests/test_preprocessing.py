import numpy as np
import pandas as pd

from app import _build_model_quality_report
from src.models.evaluation import (build_confusion_matrix_report,
                                  compute_per_class_metrics,
                                  rank_class_difficulty,
                                  summarize_most_confused_pairs)
from src.processing.cleaning import normalize_landmarks, remove_outliers_iqr


def test_remove_outliers_iqr_removes_extreme_values():
    df = pd.DataFrame({
        'x_0': [1, 2, 2, 2, 100],
        'y_0': [1, 2, 2, 2, 3],
        'label': ['hello'] * 5,
    })

    cleaned = remove_outliers_iqr(df, feature_cols=['x_0', 'y_0'])

    assert len(cleaned) <= 4
    assert 100 not in cleaned['x_0'].values


def test_normalize_landmarks_scales_values_between_zero_and_one():
    df = pd.DataFrame({
        'x_0': [10, 20, 30],
        'y_0': [5, 10, 15],
        'label': ['hello', 'hello', 'hello'],
    })

    normalized = normalize_landmarks(df, feature_cols=['x_0', 'y_0'])

    assert normalized['x_0'].min() >= 0
    assert normalized['x_0'].max() <= 1
    assert normalized['y_0'].min() >= 0
    assert normalized['y_0'].max() <= 1


def test_compute_per_class_metrics_returns_real_quality_scores():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_per_class_metrics(y_true, y_pred, labels=[0, 1])

    assert len(metrics) == 2
    assert metrics[0]['label'] == 0
    assert metrics[0]['support'] == 2
    assert metrics[0]['accuracy'] >= 0
    assert metrics[1]['label'] == 1
    assert metrics[1]['recall'] >= 0


def test_build_confusion_matrix_and_difficulty_ranking_works():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    matrix_report = build_confusion_matrix_report(y_true, y_pred, labels=[0, 1])
    ranking = rank_class_difficulty(matrix_report['per_class_metrics'])
    pairs = summarize_most_confused_pairs(matrix_report['matrix'], matrix_report['labels'])

    assert matrix_report['labels'] == [0, 1]
    assert matrix_report['matrix'][0][0] == 1
    assert matrix_report['matrix'][1][0] == 0
    assert len(ranking) == 2
    assert ranking[0]['difficulty'] >= ranking[1]['difficulty']
    assert pairs[0]['actual'] == 0
    assert pairs[0]['predicted'] == 1
    assert pairs[0]['count'] == 1


def test_build_model_quality_report_handles_single_sample_classes():
    class DummyModel:
        def predict(self, X, verbose=0):
            return np.eye(2)[np.zeros(len(X), dtype=int)]

    X = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    report = _build_model_quality_report(DummyModel(), X, y, labels=[0, 1])

    assert 'overall' in report
    assert 'per_class_metrics' in report
    assert report['overall']['accuracy'] >= 0.0
