import numpy as np
import pandas as pd
import pytest

from src.models.bilstm import BiLSTMClassifier, prepare_sequence_dataset
from src.processing.augmentation import augment_landmarks, _flip_horizontal
from src.processing.features import build_feature_dataset, _compute_sample_features, HAND_JOINTS


def test_compute_sample_features_has_all_discriminative_metrics():
    # Synthetic landmark sample of 21 joints with (x, y, z)
    sample = np.zeros((21, 3), dtype=np.float32)
    # Set thumb tip and index tip to different positions
    sample[HAND_JOINTS['thumb_tip']] = [0.2, 0.5, 0.1]
    sample[HAND_JOINTS['index_tip']] = [0.1, 0.8, 0.0]
    sample[HAND_JOINTS['middle_tip']] = [0.0, 0.85, 0.0]
    sample[HAND_JOINTS['ring_tip']] = [-0.1, 0.8, 0.0]
    sample[HAND_JOINTS['pinky_tip']] = [-0.2, 0.7, 0.0]
    sample[HAND_JOINTS['index_mcp']] = [0.1, 0.4, 0.0]
    sample[HAND_JOINTS['pinky_mcp']] = [-0.15, 0.35, 0.0]

    features = _compute_sample_features(sample)

    # Check that extension ratios are present
    assert 'ratio_ext_thumb' in features
    assert 'ratio_ext_index' in features
    assert 'ratio_ext_middle' in features
    assert 'ratio_ext_ring' in features
    assert 'ratio_ext_pinky' in features

    # Check thumb to knuckle distances
    assert 'dist_thumb_tip_index_mcp' in features
    assert 'dist_thumb_tip_middle_mcp' in features
    assert 'dist_thumb_tip_ring_mcp' in features

    # Check fingertip distances
    assert 'dist_index_tip_middle_tip' in features
    assert 'dist_middle_tip_ring_tip' in features

    # Check palm normal vector components
    assert 'palm_normal_x' in features
    assert 'palm_normal_y' in features
    assert 'palm_normal_z' in features


def test_build_feature_dataset_enriches_dataframe():
    # 63 dummy features (21 joints * 3)
    data = {'label': ['A', 'B']}
    for i in range(63):
        data[f'f{i}'] = [0.1 * i, 0.2 * i]
    df = pd.DataFrame(data)

    enriched = build_feature_dataset(df)
    assert len(enriched) == 2
    assert 'label' in enriched.columns
    # Raw features (63) + new geometric features should be >= 100 columns
    assert enriched.shape[1] > 100


def test_augment_landmarks_adds_flipped_and_jittered_samples():
    data = {'label': ['A']}
    for i in range(63):
        data[f'f{i}'] = [float(i) * 0.01]
    df = pd.DataFrame(data)

    augmented = augment_landmarks(df, augment_factor=2, include_original=True)
    assert len(augmented) == 3  # 1 original + 2 augmented
    assert all(col in augmented.columns for col in df.columns)


def test_horizontal_flip_inverts_x_coordinate():
    landmarks = np.array([
        [0.5, 0.2, 0.1],
        [-0.3, 0.4, 0.2],
    ], dtype=np.float32)

    flipped = _flip_horizontal(landmarks)
    assert np.isclose(flipped[0, 0], -0.5)
    assert np.isclose(flipped[0, 1], 0.2)
    assert np.isclose(flipped[1, 0], 0.3)


def test_bilstm_classifier_builds_with_enriched_dimensions():
    model = BiLSTMClassifier(num_classes=26, sequence_length=30, feature_dim=110)
    assert model.model is not None
    assert model.model.input_shape == (None, 30, 110)
    assert model.model.output_shape == (None, 26)


def test_tflite_model_executes_successfully():
    from pathlib import Path
    import tensorflow as tf
    tflite_path = Path('src/models/sign_bilstm.tflite')
    if not tflite_path.exists():
        pytest.skip('TFLite model not generated yet.')

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_input = np.zeros(input_details[0]['shape'], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    assert preds.shape[-1] == 26
    assert np.isclose(np.sum(preds), 1.0, atol=1e-3)


def test_landmark_extractor_supports_multi_hand_signature():
    from src.data.landmarks import LandmarkExtractor
    extractor = LandmarkExtractor(max_num_hands=2)
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    norm_vec, raw_hands, num_hands = extractor.extract_landmarks_with_raw_points(dummy_frame)

    assert norm_vec is None
    assert raw_hands == []
    assert num_hands == 0


