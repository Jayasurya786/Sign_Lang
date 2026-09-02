from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_per_class_metrics(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))

    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = []

    for idx, label in enumerate(labels):
        support = int(cm[idx].sum())
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)

        accuracy = tp / support if support else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        confusion_rate = (support - tp) / support if support else 0.0
        confusion_score = 1.0 - confusion_rate
        quality_score = (accuracy + precision + recall) / 3.0

        metrics.append({
            'label': label,
            'support': support,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'quality_score': float(quality_score),
            'confusion_score': float(confusion_score),
            'confusion_rate': float(confusion_rate),
        })

    return metrics


def build_confusion_matrix_report(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))

    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = compute_per_class_metrics(y_true, y_pred, labels=labels)

    return {
        'labels': labels,
        'matrix': matrix.tolist(),
        'per_class_metrics': metrics,
    }


def rank_class_difficulty(per_class_metrics):
    ranked = []
    for entry in per_class_metrics:
        difficulty = 1.0 - float(entry.get('quality_score', 0.0))
        ranked.append({
            'label': entry.get('label'),
            'support': entry.get('support', 0),
            'quality_score': float(entry.get('quality_score', 0.0)),
            'difficulty': float(difficulty),
            'recall': float(entry.get('recall', 0.0)),
        })

    ranked.sort(key=lambda item: (-item['difficulty'], -item['support']))
    return ranked


def summarize_most_confused_pairs(matrix, labels):
    matrix = np.asarray(matrix)
    labels = list(labels)
    pairs = []

    for actual_idx, actual_label in enumerate(labels):
        for predicted_idx, predicted_label in enumerate(labels):
            if actual_idx == predicted_idx:
                continue
            count = int(matrix[actual_idx, predicted_idx])
            if count <= 0:
                continue
            pairs.append({
                'actual': actual_label,
                'predicted': predicted_label,
                'count': count,
                'severity': float(count / max(matrix[actual_idx, :].sum(), 1)),
            })

    pairs.sort(key=lambda item: (-item['count'], -item['severity']))
    return pairs


def evaluate_model(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm,
        'labels': labels,
    }
