"""Predict vegetation class for NDVI image patches."""
from __future__ import annotations
import os
import numpy as np
from PIL import Image
from src.ndvi import CLASS_NAMES
from src.preprocessing import normalize_ndvi_for_cnn


def _load_patch(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".npy": patch = np.load(path).astype(np.float32)
    elif extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff"): patch = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 127.5 - 1.0
    else: raise ValueError(f"Unsupported file type: {extension}")
    if patch.ndim != 2: raise ValueError("Expected a 2D NDVI patch.")
    return patch


def _preprocess(patch, target=64):
    if patch.shape != (target, target):
        patch = np.asarray(Image.fromarray(patch, mode="F").resize((target, target), Image.Resampling.BILINEAR))
    return np.nan_to_num(normalize_ndvi_for_cnn(patch), nan=0.5).astype(np.float32)[None, ..., None]


def predict_one(model, patch_path):
    probabilities = model.predict(_preprocess(_load_patch(patch_path)), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {"path": patch_path, "class_index": index, "class_name": CLASS_NAMES[index], "confidence": float(probabilities[index]), "probabilities": probabilities.tolist()}


def predict_many(model_path, patch_paths):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    return [predict_one(model, path) for path in patch_paths]


def pretty_print(result):
    print(f"Prediction: {result['class_name']} ({result['confidence'] * 100:.1f}%)")
