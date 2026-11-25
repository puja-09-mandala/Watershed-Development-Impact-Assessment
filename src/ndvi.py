"""NDVI computation and change maps from imported or band-based rasters."""

from __future__ import annotations

import numpy as np

from config import CHANGE_THRESHOLDS
from src.io_utils import RasterBundle, write_raster
from config import PATHS


def ndvi_from_bands(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    denom = nir + red
    out = np.full_like(nir, np.nan, dtype=np.float32)
    valid = np.isfinite(nir) & np.isfinite(red) & (np.abs(denom) > 1e-6)
    out[valid] = (nir[valid] - red[valid]) / denom[valid]
    return np.clip(out, -1.0, 1.0)


def ndvi_change(ndvi_2004: np.ndarray, ndvi_2024: np.ndarray) -> np.ndarray:
    return ndvi_2024 - ndvi_2004


def classify_ndvi_change(change: np.ndarray, threshold: float = CHANGE_THRESHOLDS["ndvi"]) -> np.ndarray:
    """1 = increase, 0 = no significant change, -1 = decrease."""
    out = np.zeros_like(change, dtype=np.float32)
    out[change > threshold] = 1
    out[change < -threshold] = -1
    out[~np.isfinite(change)] = np.nan
    return out


def write_ndvi_products(ndvi_2004: RasterBundle, ndvi_2024: RasterBundle) -> dict:
    change = ndvi_change(ndvi_2004.array, ndvi_2024.array)
    classes = classify_ndvi_change(change)
    files = {
        "ndvi_change": write_raster(
            PATHS["processed"] / "Rajapeta_NDVI_Change_2004_2024.tif",
            change,
            ndvi_2004.profile,
        ),
        "ndvi_change_class": write_raster(
            PATHS["processed"] / "Rajapeta_NDVI_Change_Class.tif",
            classes,
            ndvi_2004.profile,
        ),
    }
    return files
