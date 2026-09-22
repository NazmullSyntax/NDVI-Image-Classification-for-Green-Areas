"""Patch extraction and spatial dataset splitting."""
from __future__ import annotations

import os
import numpy as np
from src.ndvi import classify_ndvi, NDVI_THRESHOLDS
from src.preprocessing import normalize_ndvi_for_cnn

PATCH_SIZE = 64


def extract_patches(ndvi: np.ndarray, patch_size: int = PATCH_SIZE, stride: int = 32, min_valid_ratio: float = 0.7):
    height, width = ndvi.shape
    patches, coords = [], []
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            patch = ndvi[row:row + patch_size, col:col + patch_size]
            if np.isfinite(patch).mean() >= min_valid_ratio:
                patches.append(patch)
                coords.append((row, col))
    if not patches:
        return np.empty((0, patch_size, patch_size), np.float32), np.empty((0, 2), np.int32)
    return np.stack(patches).astype(np.float32), np.asarray(coords, dtype=np.int32)


def label_patches(patches: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    labels = []
    for patch in patches:
        classes = classify_ndvi(patch, thresholds)
        valid = classes[classes >= 0]
        labels.append(int(np.bincount(valid, minlength=4).argmax()) if valid.size else -1)
    return np.asarray(labels, dtype=np.int64)


def spatial_split(coords: np.ndarray, labels: np.ndarray | None = None, train_frac: float = 0.70,
                  val_frac: float = 0.15, test_frac: float = 0.15):
    if len(coords) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    train_end = int(len(order) * train_frac)
    val_end = train_end + int(len(order) * val_frac)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def prepare_dataset(ndvi: np.ndarray, patch_size: int = PATCH_SIZE, stride: int = 32, out_dir: str | None = None):
    patches, coords = extract_patches(ndvi, patch_size, stride)
    labels = label_patches(patches, NDVI_THRESHOLDS)
    keep = labels >= 0
    patches, coords, labels = patches[keep], coords[keep], labels[keep]
    if len(patches) < 3:
        raise ValueError("At least three valid patches are required to create train/validation/test splits.")
    train_idx, val_idx, test_idx = spatial_split(coords, labels)
    features = np.nan_to_num(normalize_ndvi_for_cnn(patches), nan=0.5).astype(np.float32)[..., np.newaxis]
    data = {"X_train": features[train_idx], "y_train": labels[train_idx], "X_val": features[val_idx],
            "y_val": labels[val_idx], "X_test": features[test_idx], "y_test": labels[test_idx], "coords": coords}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "ndvi_patches.npz"), **data)
    return data
