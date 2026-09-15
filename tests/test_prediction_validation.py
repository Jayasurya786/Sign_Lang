import numpy as np

from app import app, build_prediction_preview, resolve_prediction_label, resolve_training_dataset
from src.data.dataset_config import normalize_archive_label_name, normalize_label_name


def test_archive_folder_labels_keep_their_raw_names():
    assert normalize_archive_label_name('0') == '0'
    assert normalize_archive_label_name('A') == 'A'
    assert normalize_label_name(0) == '0'


def test_resolve_training_dataset_prefers_archive_image_folder():
    dataset_path = resolve_training_dataset()
    assert dataset_path.name == 'online_asl'


def test_build_prediction_preview_forms_words_and_sentence():
    preview = build_prediction_preview(['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D'])
    assert preview['word'] == 'HELLOWORLD'
    assert preview['sentence'] == 'HELLOWORLD'
    assert preview['letters'] == ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']


def test_api_predict_rejects_non_data_url_payload():
    client = app.test_client()
    response = client.post('/api/predict', json={'image': 'not-a-data-url'})
    assert response.status_code == 400
    payload = response.get_json()
    assert 'Invalid image payload' in payload['message']


def test_resolve_prediction_label_accepts_realistic_confidence():
    result = resolve_prediction_label(np.array([0.18, 0.82]))
    assert result['label'] == 'B'
    assert result['confidence'] == 0.82
    assert result['is_unknown'] is False
    assert result['threshold'] == 0.03


def test_resolve_prediction_label_accepts_low_configured_threshold():
    result = resolve_prediction_label(np.array([0.04, 0.96]), threshold=0.02)
    assert result['label'] == 'B'
    assert result['threshold'] == 0.02


def test_update_prediction_buffer_accumulates_sentence_without_repeating_letter():
    from app import update_prediction_buffer

    assert update_prediction_buffer('H', now_ms=0) == 'H'
    assert update_prediction_buffer('E', now_ms=500) == 'HE'
    assert update_prediction_buffer('L', now_ms=1000) == 'HEL'
    assert update_prediction_buffer('L', now_ms=1500) == 'HEL'
    assert update_prediction_buffer('O', now_ms=2000) == 'HELO'


def test_live_prediction_window_keeps_recent_frames():
    from app import LIVE_FEATURE_WINDOW

    LIVE_FEATURE_WINDOW.clear()
    LIVE_FEATURE_WINDOW.extend([[1.0, 2.0], [3.0, 4.0]])
    assert list(LIVE_FEATURE_WINDOW)[-1] == [3.0, 4.0]
    LIVE_FEATURE_WINDOW.clear()


def test_live_sequence_uses_current_pose_for_every_timestep():
    from app import build_live_sequence

    sequence = build_live_sequence(np.array([1.0, 2.0]), sequence_length=3)

    assert sequence.shape == (1, 3, 2)
    assert np.array_equal(sequence[0], np.array([[1.0, 2.0]] * 3, dtype=np.float32))


def test_unknown_prediction_does_not_raise_scope_error():
    from app import clear_prediction_buffer, smooth_and_commit_prediction

    clear_prediction_buffer()
    result = smooth_and_commit_prediction('unknown', confidence=0.01, now_ms=0)

    assert result['label'] == ''
    assert result['word'] == ''
