"""NDVI calculation, classification, and raster helpers."""
from __future__ import annotations

import numpy as np
import rasterio

NDVI_THRESHOLDS = {"very_low_max": 0.20, "low_max": 0.40, "moderate_max": 0.60}
CLASS_NAMES = ["Very Low Vegetation", "Low Vegetation", "Moderate Vegetation", "High Vegetation"]


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    if red.shape != nir.shape:
        raise ValueError(f"Red and NIR shapes differ: {red.shape} vs {nir.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi = np.asarray(ndvi, dtype=np.float32)
    ndvi[~np.isfinite(ndvi)] = np.nan
    return np.clip(ndvi, -1.0, 1.0)


def read_bands(red_path: str, nir_path: str):
    with rasterio.open(red_path) as red_src:
        red = red_src.read(1)
        profile = red_src.profile.copy()
        red_crs, red_transform = red_src.crs, red_src.transform
    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1)
        if nir.shape != red.shape:
            raise ValueError(f"Red and NIR shapes differ: {red.shape} vs {nir.shape}")
    profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    return red, nir, profile, red_crs, red_transform


def classify_ndvi(ndvi: np.ndarray, thresholds: dict | None = None) -> np.ndarray:
    thresholds = thresholds or NDVI_THRESHOLDS
    classes = np.full(np.shape(ndvi), -1, dtype=np.int8)
    valid = np.isfinite(ndvi)
    values = np.asarray(ndvi)[valid]
    labels = np.zeros(values.shape, dtype=np.int8)
    labels[values >= thresholds["very_low_max"]] = 1
    labels[values >= thresholds["low_max"]] = 2
    labels[values >= thresholds["moderate_max"]] = 3
    classes[valid] = labels
    return classes


def ndvi_statistics(ndvi: np.ndarray) -> dict:
    valid = np.asarray(ndvi)[np.isfinite(ndvi)]
    if valid.size == 0:
        return {key: float("nan") for key in ("min", "max", "mean", "median", "std", "count")}
    return {"min": float(valid.min()), "max": float(valid.max()), "mean": float(valid.mean()),
            "median": float(np.median(valid)), "std": float(valid.std()), "count": int(valid.size)}


def class_percentages(classes: np.ndarray) -> dict:
    valid = np.asarray(classes)[np.asarray(classes) >= 0]
    total = valid.size
    return {name: float((valid == index).sum() / total * 100) if total else 0.0
            for index, name in enumerate(CLASS_NAMES)}
