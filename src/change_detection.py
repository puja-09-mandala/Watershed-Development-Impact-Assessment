"""Pixel-wise change detection for NDVI, LST, and LULC."""

from __future__ import annotations

import numpy as np

from src.ndvi import classify_ndvi_change
from src.lst import classify_lst_change
from src.lulc import lulc_change_map


def summarize_classes(classified: np.ndarray, labels: dict) -> dict:
    finite = classified[np.isfinite(classified)]
    total = finite.size
    out = {}
    if total == 0:
        return out
    for value, name in labels.items():
        count = int((finite == value).sum())
        out[name] = {"pixels": count, "percent": float(100.0 * count / total)}
    return out


def ndvi_change_summary(change: np.ndarray) -> dict:
    classes = classify_ndvi_change(change)
    return summarize_classes(
        classes,
        {-1: "Decrease", 0: "No significant change", 1: "Increase"},
    )


def lst_change_summary(change: np.ndarray) -> dict:
    classes = classify_lst_change(change)
    return summarize_classes(
        classes,
        {-1: "Decrease", 0: "No significant change", 1: "Increase"},
    )


def lulc_change_summary(a: np.ndarray, b: np.ndarray) -> dict:
    change = lulc_change_map(a, b)
    return summarize_classes(change, {0: "No class change", 1: "Class change"})
