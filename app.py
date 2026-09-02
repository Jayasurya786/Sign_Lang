from __future__ import annotations

import base64
import io
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from src.data.dataset_config import ASL_FULL_CLASS_LIST, validate_dataset_classes
from src.data.landmarks import LandmarkExtractor
from src.models.bilstm import train_bilstm_model
from src.models.evaluation import (build_confusion_matrix_report,
                                  compute_per_class_metrics,
                                  rank_class_difficulty,
                                  summarize_most_confused_pairs)
from src.processing.cleaning import clean_landmark_dataset
from src.processing.features import build_feature_dataset
from src.processing.sequences import create_fixed_length_sequences

app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_PATH = Path('src/models/sign_bilstm.keras')
MODEL_METRICS_PATH = Path('src/models/model_quality.json')
MODEL_CLASSES_PATH = Path('src/models/model_classes.json')
ONLINE_IMAGE_DATASET = Path('data/online_asl')
LANDMARK_CACHE_VERSION = 'v2-relative-scale'
DATASET_CANDIDATES = [
    Path(os.environ.get('SIGN_LANG_DATASET_PATH', '')).expanduser() if os.environ.get('SIGN_LANG_DATASET_PATH') else None,
    ONLINE_IMAGE_DATASET,
]
DEFAULT_UNKNOWN_THRESHOLD = 0.03
LIVE_FEATURE_WINDOW = deque(maxlen=30)
TRAINING_STATE = {'status': 'idle', 'message': 'Ready', 'dataset_source': ''}


def resolve_training_dataset() -> Path:
    seen = set()
    for candidate in DATASET_CANDIDATES:
        if candidate is None:
            continue
        path = candidate.resolve() if candidate.is_absolute() else (Path.cwd() / candidate).resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists():
            return path
    return ONLINE_IMAGE_DATASET


def load_image_dataset_as_dataframe(dataset_dir: str | Path) -> pd.DataFrame:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_path}')
    if not dataset_path.is_dir():
        return pd.read_csv(dataset_path)

    output_path = Path('data/processed/online_asl_landmarks.csv')
    metadata_path = output_path.with_suffix('.meta.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_is_current = False
    if metadata_path.exists():
        try:
            cache_is_current = json.loads(metadata_path.read_text(encoding='utf-8')).get('version') == LANDMARK_CACHE_VERSION
        except (OSError, json.JSONDecodeError):
            pass
    if cache_is_current and output_path.exists() and output_path.stat().st_size > 0:
        return pd.read_csv(output_path)

    extractor = LandmarkExtractor()
    dataframe = extractor.extract_dataset_from_directory(str(dataset_path), str(output_path))
    metadata_path.write_text(json.dumps({'version': LANDMARK_CACHE_VERSION}, indent=2), encoding='utf-8')
    return dataframe


def _save_model_quality_report(report):
    MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_METRICS_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')


def _load_model_classes(model_path: str | Path = MODEL_PATH) -> list[str]:
    classes_path = MODEL_CLASSES_PATH if Path(model_path).resolve() == MODEL_PATH.resolve() else Path(model_path).with_suffix('.classes.json')
    if classes_path.exists():
        try:
            classes = json.loads(classes_path.read_text(encoding='utf-8'))
            if isinstance(classes, list) and classes:
                return [str(value).upper() for value in classes]
        except (OSError, json.JSONDecodeError):
            pass
    return ASL_FULL_CLASS_LIST


def _save_model_classes(classes: list[str], model_path: str | Path = MODEL_PATH):
    classes_path = MODEL_CLASSES_PATH if Path(model_path).resolve() == MODEL_PATH.resolve() else Path(model_path).with_suffix('.classes.json')
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text(json.dumps(classes, indent=2), encoding='utf-8')


def resolve_prediction_label(prediction_scores: np.ndarray, class_names: list[str] | None = None, threshold: float | None = None) -> dict:
    scores = np.asarray(prediction_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError('Prediction scores are empty.')

    threshold_value = DEFAULT_UNKNOWN_THRESHOLD if threshold is None else float(threshold)
    threshold_value = max(0.02, min(0.35, threshold_value))

    class_list = list(class_names) if class_names else ASL_FULL_CLASS_LIST[:scores.size]
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])
    resolved_name = class_list[best_index] if best_index < len(class_list) else str(best_index)
    label = 'unknown' if confidence < threshold_value else resolved_name

    return {
        'label': label,
        'confidence': confidence,
        'is_unknown': label == 'unknown',
        'threshold': threshold_value,
        'class_index': best_index,
    }


def build_prediction_preview(predictions: list[str] | tuple[str, ...] | None) -> dict:
    clean_labels = []
    for value in predictions or []:
        label = str(value or '').strip().upper()
        if not label or label in {'UNKNOWN', 'NONE', 'NAN'}:
            continue
        clean_labels.append(label)

    if not clean_labels:
        return {
            'letters': [],
            'word': '',
            'sentence': '',
            'preview': '',
        }

    letters = clean_labels
    word = ''.join(letters)
    sentence = word
    return {
        'letters': letters,
        'word': word,
        'sentence': sentence,
        'preview': sentence,
    }


def _build_model_quality_report(model, X, y, labels=None):
    if X.size == 0 or y.size == 0:
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'overall': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            },
            'per_class_metrics': [],
        }

    if len(X) < 2:
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'overall': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            },
            'per_class_metrics': [],
        }

    if labels is None:
        labels = sorted(np.unique(y).tolist())

    unique_labels = np.unique(y)
    unique_counts = np.unique(y, return_counts=True)[1]
    can_stratify = len(unique_labels) > 1 and np.all(unique_counts >= 2)

    indices = np.arange(len(X))
    test_size = 0.2
    if can_stratify:
        min_test_size = len(unique_labels) / max(1, len(X))
        test_size = max(0.2, min(0.5, min_test_size))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    X_val = X[val_idx]
    y_val = y[val_idx]
    if X_val.size == 0 or len(np.unique(y_val)) == 0:
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'overall': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
            },
            'per_class_metrics': [],
        }

    y_pred_proba = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    confusion_report = build_confusion_matrix_report(y_val, y_pred, labels=labels)
    per_class_metrics = confusion_report['per_class_metrics']
    difficulty_ranking = rank_class_difficulty(per_class_metrics)
    most_confused_pairs = summarize_most_confused_pairs(confusion_report['matrix'], confusion_report['labels'])

    from sklearn.metrics import f1_score, precision_score, recall_score
    overall = {
        'accuracy': float(np.mean(y_pred == y_val)) if len(y_val) else 0.0,
        'precision': float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_val, y_pred, average='weighted', zero_division=0)),
    }

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'overall': overall,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'labels': confusion_report['labels'],
            'matrix': confusion_report['matrix'],
        },
        'difficulty_ranking': difficulty_ranking,
        'most_confused_pairs': most_confused_pairs,
    }


@app.route('/')
def index():
    return render_template('home.html', active_page='home')


@app.route('/dataset')
def dataset_page():
    return render_template('dataset.html', active_page='dataset')


@app.route('/model')
def model_page():
    return render_template('model.html', active_page='model')


@app.route('/insights')
def insights_page():
    return render_template('insights.html', active_page='insights')


@app.route('/api/status')
def api_status():
    dataset_path = resolve_training_dataset()
    return jsonify({
        'status': 'running',
        'active_dataset_exists': dataset_path.exists(),
        'active_dataset': str(dataset_path),
        'model_exists': MODEL_PATH.exists(),
        'project': 'Sign Language Recognition',
    })


@app.route('/api/model-metrics')
def api_model_metrics():
    if MODEL_METRICS_PATH.exists():
        try:
            report = json.loads(MODEL_METRICS_PATH.read_text(encoding='utf-8'))
            return jsonify(report)
        except Exception:
            pass

    if not MODEL_PATH.exists():
        return jsonify({
            'status': 'not-trained',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'overall': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'per_class_metrics': [],
        })

    try:
        dataset_path = resolve_training_dataset()
        train_df = load_image_dataset_as_dataframe(dataset_path)

        cleaned = clean_landmark_dataset(train_df)
        features = build_feature_dataset(cleaned)
        X, y = create_fixed_length_sequences(features, sequence_length=30)
        model = tf.keras.models.load_model(str(MODEL_PATH))
        report = _build_model_quality_report(model, X, y, labels=list(range(int(y.max()) + 1)))
        _save_model_quality_report(report)
        return jsonify(report)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return jsonify({
            'status': 'error',
            'message': str(exc),
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'overall': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'per_class_metrics': [],
        }), 500


@app.route('/api/train-status')
def api_train_status():
    return jsonify(TRAINING_STATE)


@app.route('/api/dataset-stats')
def api_dataset_stats():
    dataset_path = resolve_training_dataset()
    if not dataset_path.exists():
        return jsonify({
            'total_rows': 0,
            'classes_available': 0,
            'cleaned_sample_count': 0,
            'last_training_status': 'online dataset missing',
            'source': str(dataset_path),
            'class_distribution': [],
            'split_overview': {'train': 0, 'validation': 0, 'test': 0},
            'low_quality_labels': [],
            'expected_signs': ASL_FULL_CLASS_LIST,
            'missing_signs': ASL_FULL_CLASS_LIST,
        })

    try:
        train_df = load_image_dataset_as_dataframe(dataset_path)
        if train_df.empty or 'label' not in train_df.columns:
            return jsonify({
                'status': 'error',
                'message': 'No hand landmarks could be extracted from the online images.',
                'dataset_source': str(dataset_path),
            }), 400
        normalized_df = train_df.copy()
        missing_signs = []

        total_rows = int(len(train_df))
        classes_available = int(normalized_df['label'].nunique()) if 'label' in normalized_df.columns else 0

        cleaned = clean_landmark_dataset(normalized_df)
        features = build_feature_dataset(cleaned)
        cleaned_sample_count = int(len(features))

        class_distribution = []
        if 'label' in normalized_df.columns:
            for label_name, count in normalized_df['label'].value_counts().items():
                class_distribution.append({'label': str(label_name), 'count': int(count)})

        split_overview = {'train': 0, 'validation': 0, 'test': 0}
        if class_distribution:
            total = sum(item['count'] for item in class_distribution)
            split_overview['train'] = max(1, int(total * 0.7))
            split_overview['validation'] = max(1, int(total * 0.15))
            split_overview['test'] = max(1, total - split_overview['train'] - split_overview['validation'])

        low_quality_labels = [
            item['label'] for item in class_distribution if item['count'] < 20
        ]

        last_training_status = 'ready' if MODEL_PATH.exists() else 'not trained'

        return jsonify({
            'total_rows': total_rows,
            'classes_available': classes_available,
            'cleaned_sample_count': cleaned_sample_count,
            'last_training_status': last_training_status,
            'source': str(dataset_path),
            'class_distribution': class_distribution,
            'split_overview': split_overview,
            'low_quality_labels': low_quality_labels,
            'expected_signs': ASL_FULL_CLASS_LIST,
            'missing_signs': missing_signs,
        })
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return jsonify({
            'total_rows': 0,
            'classes_available': 0,
            'cleaned_sample_count': 0,
            'last_training_status': f'error: {str(exc)}',
            'source': str(dataset_path),
            'class_distribution': [],
            'split_overview': {'train': 0, 'validation': 0, 'test': 0},
            'low_quality_labels': [],
            'expected_signs': ASL_FULL_CLASS_LIST,
            'missing_signs': ASL_FULL_CLASS_LIST,
        }), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    dataset_path = resolve_training_dataset()
    if not dataset_path.exists():
        return jsonify({'status': 'error', 'message': 'Training dataset not found.'}), 400

    try:
        TRAINING_STATE.update({
            'status': 'extracting',
            'message': 'Analyzing online dataset images for hand landmarks...',
            'dataset_source': str(dataset_path),
        })
        train_df = load_image_dataset_as_dataframe(dataset_path)
        if train_df.empty or 'label' not in train_df.columns:
            return jsonify({'status': 'error', 'message': 'No hand landmarks could be extracted from the online images.'}), 400
        normalized_df = train_df.copy()
        normalized_df['label'] = normalized_df['label'].map(lambda value: str(value).strip().upper())

        TRAINING_STATE.update({'status': 'preparing', 'message': 'Preparing training features...'})
        cleaned = clean_landmark_dataset(normalized_df)
        features = build_feature_dataset(cleaned)
        X, y = create_fixed_length_sequences(features, sequence_length=30)

        if X.size == 0 or y.size == 0:
            return jsonify({'status': 'error', 'message': 'Training data could not be generated.'}), 400

        model_classes = sorted(features['label'].dropna().astype(str).str.upper().unique().tolist())
        num_classes = len(model_classes)
        TRAINING_STATE.update({'status': 'training', 'message': f'Training model with {len(X)} sequences...'})
        model, history = train_bilstm_model(
            X,
            y,
            num_classes=num_classes,
            sequence_length=30,
            feature_dim=X.shape[-1],
            model_path=str(MODEL_PATH),
            epochs=int(os.environ.get('SIGN_LANG_TRAIN_EPOCHS', '20')),
            batch_size=int(os.environ.get('SIGN_LANG_TRAIN_BATCH_SIZE', '32')),
        )
        _save_model_classes(model_classes)
        TRAINING_STATE.update({'status': 'complete', 'message': 'Training completed successfully.'})

        quality_report = _build_model_quality_report(model, X, y, labels=list(range(num_classes)))
        _save_model_quality_report(quality_report)

        last_epoch = history.history
        last_acc = float(np.mean(last_epoch['accuracy'][-1:])) if 'accuracy' in last_epoch else 0.0
        return jsonify({
            'status': 'success',
            'message': 'BiLSTM model trained successfully.',
            'dataset_source': str(dataset_path),
            'num_classes': num_classes,
            'feature_dim': X.shape[-1],
            'sequence_length': 30,
            'last_accuracy': last_acc,
            'model_path': str(MODEL_PATH),
            'training_quality': quality_report,
            'expected_signs': model_classes,
        })
    except Exception as exc:  # pragma: no cover - runtime safeguard
        TRAINING_STATE.update({'status': 'error', 'message': str(exc)})
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get('image')
    model_path = payload.get('model_path') or str(MODEL_PATH)
    unknown_threshold = float(payload.get('unknown_threshold', DEFAULT_UNKNOWN_THRESHOLD))

    if not image_b64:
        return jsonify({'status': 'error', 'message': 'No image payload provided.'}), 400

    if not isinstance(image_b64, str):
        return jsonify({'status': 'error', 'message': 'Invalid image payload: expected a base64 string.'}), 400

    if not image_b64.startswith('data:image/') or ',' not in image_b64:
        return jsonify({'status': 'error', 'message': 'Invalid image payload: expected a data URL image.'}), 400

    try:
        header, encoded = image_b64.split(',', 1)
        if not header.startswith('data:image/'):
            return jsonify({'status': 'error', 'message': 'Invalid image payload: missing image data URL header.'}), 400
        if not encoded.strip():
            return jsonify({'status': 'error', 'message': 'Invalid image payload: empty image body.'}), 400

        img_data = base64.b64decode(encoded, validate=True)
        image = np.asarray(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({
                'status': 'success',
                'label': 'unknown',
                'word': '',
                'sentence': '',
                'preview': '',
                'confidence': 0.0,
                'is_unknown': True,
                'threshold': unknown_threshold,
                'class_index': -1,
            })

        extractor = LandmarkExtractor()
        vector = extractor.extract_landmarks_from_frame(frame)
        if vector is None:
            LIVE_FEATURE_WINDOW.clear()
            return jsonify({
                'status': 'success',
                'label': 'unknown',
                'word': '',
                'sentence': '',
                'preview': '',
                'confidence': 0.0,
                'is_unknown': True,
                'threshold': unknown_threshold,
                'class_index': -1,
            })

        feature_df = pd.DataFrame(np.asarray(vector, dtype=float).reshape(1, -1))
        feature_df.columns = [f'f{i}' for i in range(feature_df.shape[1])]
        feature_df = build_feature_dataset(feature_df)
        arr = feature_df.drop(columns=['label'], errors='ignore').to_numpy(dtype=np.float32)
        LIVE_FEATURE_WINDOW.append(arr[0])
        sequence_rows = list(LIVE_FEATURE_WINDOW)
        if len(sequence_rows) < 30:
            sequence_rows = [sequence_rows[0]] * (30 - len(sequence_rows)) + sequence_rows
        sequence = np.asarray(sequence_rows, dtype=np.float32).reshape(1, 30, -1)

        if not Path(model_path).exists():
            return jsonify({'status': 'error', 'message': 'Model file not found. Train the model first.'}), 404

        model = __import__('tensorflow').keras.models.load_model(model_path)
        preds = model.predict(sequence, verbose=0)
        class_names = _load_model_classes(model_path)
        if len(class_names) != preds.shape[1]:
            class_names = class_names[:preds.shape[1]] if len(class_names) > preds.shape[1] else [str(i) for i in range(preds.shape[1])]
        result = resolve_prediction_label(preds[0], class_names=class_names, threshold=unknown_threshold)
        preview = build_prediction_preview([result['label']])
        return jsonify({
            'status': 'success',
            'label': result['label'],
            'word': preview['word'],
            'sentence': preview['sentence'],
            'preview': preview['preview'],
            'confidence': result['confidence'],
            'is_unknown': result['is_unknown'],
            'threshold': result['threshold'],
            'class_index': result['class_index'],
        })
    except (ValueError, TypeError) as exc:  # pragma: no cover - runtime safeguard
        return jsonify({'status': 'error', 'message': f'Invalid image payload: {str(exc)}'}), 400
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/api/collect', methods=['POST'])
def api_collect():
    payload = request.get_json(silent=True) or {}
    label = (payload.get('label') or 'online').strip() or 'online'

    try:
        return jsonify({
            'status': 'success',
            'message': f'Online dataset source: {ONLINE_IMAGE_DATASET}. Use the Train model action after downloading it.',
            'source': str(ONLINE_IMAGE_DATASET),
            'label': label,
            'samples': 0,
            'labels': 26,
        })
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return jsonify({'status': 'error', 'message': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
