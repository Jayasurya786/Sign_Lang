from __future__ import annotations

import collections
from collections import deque

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data.landmarks import LandmarkExtractor
from src.processing.features import build_feature_dataset


def majority_vote(predictions: list[str]) -> str:
    if not predictions:
        return 'No prediction'
    votes = collections.Counter(predictions)
    return votes.most_common(1)[0][0]


def _landmark_vector_to_sequence(landmark_vector: np.ndarray, sequence_length: int = 30) -> np.ndarray:
    if landmark_vector.size == 0:
        raise ValueError('Empty landmark vector received from MediaPipe.')

    sample_df = pd.DataFrame(np.array(landmark_vector, dtype=float).reshape(1, -1))
    sample_df.columns = [f'f{i}' for i in range(sample_df.shape[1])]
    feature_df = build_feature_dataset(sample_df)

    feature_values = feature_df.drop(columns=['label'], errors='ignore').to_numpy(dtype=float)
    if feature_values.shape[1] == 0:
        raise ValueError('Feature extraction produced no columns for live prediction.')

    current = np.asarray(feature_values, dtype=np.float32).reshape(-1)
    padded = np.zeros((sequence_length, current.shape[0]), dtype=np.float32)
    padded[-1] = current
    return padded.reshape(1, sequence_length, -1)


def load_trained_model(model_path: str):
    model = keras.models.load_model(model_path)
    return model


def predict_live(model, class_names: list[str], camera_index: int = 0, max_frames: int = 15, sequence_length: int = 30):
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam for live recognition.')

    predictions: list[str] = []
    sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Sign Language Recognition', frame)

        landmark_vector = extractor.extract_landmarks_from_frame(frame)
        if landmark_vector is not None:
            sequence_feature = _landmark_vector_to_sequence(landmark_vector, sequence_length=sequence_length)
            sequence_buffer.append(sequence_feature[0])

            if len(sequence_buffer) == sequence_length:
                model_input = np.stack(list(sequence_buffer), axis=0).reshape(1, sequence_length, -1)
                prediction_scores = model.predict(model_input, verbose=0)
                predicted_idx = int(np.argmax(prediction_scores, axis=1)[0])
                predicted_label = class_names[predicted_idx]
                predictions.append(predicted_label)
                if len(predictions) > max_frames:
                    predictions = predictions[-max_frames:]
                print('Current vote:', majority_vote(predictions[-10:]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return majority_vote(predictions) if predictions else 'No prediction'
