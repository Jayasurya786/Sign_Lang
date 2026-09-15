#!/usr/bin/env python3
"""Export trained Keras BiLSTM sign language model to TensorFlow Lite (.tflite)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf

from app import MODEL_PATH


def export_to_tflite(
    keras_model_path: Path = MODEL_PATH,
    output_tflite_path: Path = Path('src/models/sign_bilstm.tflite'),
) -> Path:
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model not found at {keras_model_path}")

    print(f"Loading Keras model from {keras_model_path}...")
    model = tf.keras.models.load_model(str(keras_model_path))

    print("Wrapping model in concrete execution signature...")
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, 30, model.input_shape[-1]], tf.float32)
    )

    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    tflite_model = converter.convert()

    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)

    keras_size = os.path.getsize(keras_model_path) / 1024
    tflite_size = os.path.getsize(output_tflite_path) / 1024
    print(f"Successfully exported TFLite model to {output_tflite_path}")
    print(f"  Keras model size:  {keras_size:.1f} KB")
    print(f"  TFLite model size: {tflite_size:.1f} KB ({tflite_size/keras_size*100:.1f}%)")

    # Verify numerical output alignment between Keras and TFLite
    print("\nVerifying numerical output alignment between Keras and TFLite...")
    interpreter = tf.lite.Interpreter(model_path=str(output_tflite_path))
    interpreter.allocate_tensors()

    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    dummy_input = np.random.randn(1, 30, model.input_shape[-1]).astype(np.float32)
    keras_preds = model.predict(dummy_input, verbose=0)

    interpreter.set_tensor(inp_det[0]['index'], dummy_input)
    interpreter.invoke()
    tflite_preds = interpreter.get_tensor(out_det[0]['index'])

    max_diff = float(np.max(np.abs(keras_preds - tflite_preds)))
    print(f"  Maximum absolute difference: {max_diff:.8f}")
    assert max_diff < 1e-3, f"TFLite output diverged: max diff = {max_diff}"
    print("[OK] Verification passed: TFLite model outputs match Keras model!\n")

    return output_tflite_path


if __name__ == '__main__':
    export_to_tflite()
