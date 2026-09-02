import numpy as np
import pandas as pd

from src.models.bilstm import prepare_sequence_dataset


def test_prepare_sequence_dataset_creates_bilstm_ready_tensors():
    df = pd.DataFrame({
        'label': [0, 1, 1],
        'f0': [1.0, 2.0, 3.0],
        'f1': [4.0, 5.0, 6.0],
        'f2': [7.0, 8.0, 9.0],
    })

    X, y = prepare_sequence_dataset(df, sequence_length=3)

    assert X.shape[0] == 3
    assert X.shape[1] == 3
    assert X.shape[2] == 3
    assert y.shape[0] == 3
    assert set(np.unique(y)).issubset({0, 1})


def test_prepare_sequence_dataset_handles_full_asl_string_labels():
    df = pd.DataFrame({
        'label': ['A', 'B', 'A'],
        'f0': [1.0, 2.0, 3.0],
        'f1': [4.0, 5.0, 6.0],
        'f2': [7.0, 8.0, 9.0],
    })

    X, y = prepare_sequence_dataset(df, sequence_length=3)

    assert X.shape[0] == 3
    assert set(np.unique(y)).issubset({0, 1})
    assert y.dtype.kind in {'i', 'u'}
