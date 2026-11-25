"""Land surface temperature conversion and change maps."""

from __future__ import annotations

import numpy as np

from config import CHANGE_THRESHOLDS, LANDSAT, PATHS
from src.io_utils import RasterBundle, write_raster


def scaled_st_to_celsius(scaled: np.ndarray, year: int) -> np.ndarray:
    spec = LANDSAT[str(year)]
    kelvin = scaled * spec["scale_st"] + spec["offset_st"]
    celsius = kelvin - 273.15
    celsius = np.where((celsius > -50) & (celsius < 80), celsius, np.nan)
    return celsius.astype(np.float32)


def ensure_celsius(array: np.ndarray) -> np.ndarray:
    """If values look like Kelvin or Collection-2 scaled integers, convert."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return array
    median = float(np.nanmedian(finite))
    if median > 200:  # Kelvin
        return (array - 273.15).astype(np.float32)
    if median > 80:  # likely still scaled DN
        return scaled_st_to_celsius(array, 2024)
    return array.astype(np.float32)


def lst_change(lst_2004: np.ndarray, lst_2024: np.ndarray) -> np.ndarray:
    return lst_2024 - lst_2004


def classify_lst_change(change: np.ndarray, threshold: float = CHANGE_THRESHOLDS["lst_c"]) -> np.ndarray:
    out = np.zeros_like(change, dtype=np.float32)
    out[change > threshold] = 1
    out[change < -threshold] = -1
    out[~np.isfinite(change)] = np.nan
    return out


def write_lst_products(lst_2004: RasterBundle, lst_2024: RasterBundle) -> dict:
    a = ensure_celsius(lst_2004.array)
    b = ensure_celsius(lst_2024.array)
    lst_2004.array = a
    lst_2024.array = b
    change = lst_change(a, b)
    classes = classify_lst_change(change)
    return {
        "lst_change": write_raster(
            PATHS["processed"] / "Rajapeta_LST_Change_2004_2024.tif",
            change,
            lst_2004.profile,
        ),
        "lst_change_class": write_raster(
            PATHS["processed"] / "Rajapeta_LST_Change_Class.tif",
            classes,
            lst_2004.profile,
        ),
    }
