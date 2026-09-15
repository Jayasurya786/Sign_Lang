import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.data.dataset_config import normalize_archive_label_name


class LandmarkExtractor:
    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 2, min_detection_confidence: float = 0.35):
        self.mp_hands = mp.solutions.hands
        self.max_num_hands = max_num_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_landmarks_from_frame(self, frame):
        norm_vec, _, _ = self.extract_landmarks_with_raw_points(frame)
        return norm_vec

    def extract_landmarks_with_raw_points(self, frame) -> tuple[np.ndarray | None, list[list[list[float]]], int]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks is None or len(results.multi_hand_landmarks) == 0:
            return None, [], 0

        raw_hands: list[list[list[float]]] = []
        for hand_landmarks in results.multi_hand_landmarks:
            raw_hands.append([[float(lm.x), float(lm.y), float(lm.z)] for lm in hand_landmarks.landmark])

        # Primary hand (index 0) for relative scale normalization
        primary = results.multi_hand_landmarks[0]
        points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in primary.landmark],
            dtype=np.float32,
        )
        points -= points[0]
        scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
        if scale > 1e-6:
            points /= scale

        coordinates = []
        for point in points:
            coordinates.extend(point.tolist())

        normalized_vector = np.array(coordinates, dtype=np.float32)
        return normalized_vector, raw_hands, len(raw_hands)

    def extract_dataset_from_directory(self, input_dir: str, output_path: str = "data/processed/landmarks.csv"):
        rows = []
        input_path = Path(input_dir)
        max_images_per_class = int(os.environ.get('SIGN_LANG_MAX_IMAGES_PER_CLASS', '0'))

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        for label_dir in sorted(input_path.iterdir()):
            if not label_dir.is_dir():
                continue

            label = normalize_archive_label_name(label_dir.name)
            if not label or label not in set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                continue

            image_files = sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg")) + sorted(label_dir.glob("*.jpeg"))
            if max_images_per_class > 0:
                image_files = image_files[:max_images_per_class]
            for image_path in image_files:
                frame = cv2.imread(str(image_path))
                if frame is not None and 'Pre-Processed Data' in input_path.name:
                    raw_path = input_path.parent / 'Gesture Image Data' / label_dir.name / image_path.name
                    raw_frame = cv2.imread(str(raw_path))
                    if raw_frame is not None:
                        frame = raw_frame
                if frame is None:
                    continue

                landmarks = self.extract_landmarks_from_frame(frame)
                if landmarks is None:
                    continue

                rows.append({"label": label, **{f"f{i}": float(value) for i, value in enumerate(landmarks)}})

        output_df = pd.DataFrame(rows)
        output_df.to_csv(output_path, index=False)
        return output_df


def extract_hand_landmarks(input_dir: str, output_path: str = "data/processed/landmarks.csv"):
    extractor = LandmarkExtractor()
    return extractor.extract_dataset_from_directory(input_dir=input_dir, output_path=output_path)
