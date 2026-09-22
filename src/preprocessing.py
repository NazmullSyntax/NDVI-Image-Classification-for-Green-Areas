"""NDVI cleaning, normalization, and GeoTIFF I/O."""
from __future__ import annotations

import numpy as np
import rasterio


def clean_ndvi(ndvi: np.ndarray, min_valid: float = -1.0, max_valid: float = 1.0) -> np.ndarray:
    result = np.asarray(ndvi, dtype=np.float32).copy()
    result[~np.isfinite(result)] = np.nan
    result[(result < min_valid) | (result > max_valid)] = np.nan
    return result


def fill_small_gaps(ndvi: np.ndarray, max_gap_pixels: int = 2) -> np.ndarray:
    result = np.asarray(ndvi, dtype=np.float32).copy()
    height, width = result.shape
    for row, col in zip(*np.where(np.isnan(result))):
        window = result[max(0, row - max_gap_pixels):min(height, row + max_gap_pixels + 1),
                        max(0, col - max_gap_pixels):min(width, col + max_gap_pixels + 1)]
        values = window[np.isfinite(window)]
        if values.size:
            result[row, col] = values.mean()
    return result


def normalize_ndvi_for_cnn(ndvi: np.ndarray) -> np.ndarray:
    return ((np.asarray(ndvi, dtype=np.float32) + 1.0) / 2.0).astype(np.float32)


def load_ndvi_geotiff(path: str):
    with rasterio.open(path) as source:
        ndvi = source.read(1).astype(np.float32)
        profile = source.profile.copy()
        crs, transform = source.crs, source.transform
    nodata = profile.get("nodata")
    if nodata is not None and np.isfinite(nodata):
        ndvi[ndvi == nodata] = np.nan
    return ndvi, profile, crs, transform


def save_geotiff(array: np.ndarray, profile: dict, path: str, dtype: str = "float32") -> None:
    output_profile = profile.copy()
    output_profile.update(dtype=dtype, count=1, compress="lzw", nodata=np.nan)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(np.asarray(array, dtype=dtype), 1)
