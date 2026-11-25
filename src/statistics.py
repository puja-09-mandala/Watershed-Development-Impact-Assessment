"""Descriptive statistics for raster layers. Never invent values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS
def raster_stats(array: np.ndarray, name: str) -> dict:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"name": name, "n": 0, "note": "No valid pixels"}
    return {
        "name": name,
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
    }


def histogram_data(array: np.ndarray, n_bins: int = 20) -> dict:
    """Equal-width histogram of valid pixels for client-side chart rendering."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(finite, bins=n_bins)
    return {
        "bins": [round(float(e), 6) for e in edges],
        "counts": [int(c) for c in counts],
    }


def paired_histogram(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int = 20,
    label_a: str = "2004",
    label_b: str = "2024",
) -> dict:
    """Histograms of two arrays over ONE shared bin axis.

    Required for a truthful 2004-vs-2024 distribution chart: independent
    bin edges would place bars at incomparable positions.
    """
    fa, fb = np.asarray(a)[np.isfinite(a)], np.asarray(b)[np.isfinite(b)]
    if fa.size == 0 or fb.size == 0:
        return {"bins": [], "labels": [], "counts_a": [], "counts_b": [],
                "label_a": label_a, "label_b": label_b}
    lo = float(min(fa.min(), fb.min()))
    hi = float(max(fa.max(), fb.max()))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    counts_a, edges = np.histogram(fa, bins=n_bins, range=(lo, hi))
    counts_b, _ = np.histogram(fb, bins=n_bins, range=(lo, hi))
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    # Keep full precision for NDVI (small decimals); LST midpoints ~30-45.
    decimals = 4 if abs(hi) < 5 else 2
    return {
        "bins": [round(float(e), 6) for e in edges],
        "labels": [round(float(m), decimals) for m in mids],
        "counts_a": [int(c) for c in counts_a],
        "counts_b": [int(c) for c in counts_b],
        "label_a": label_a,
        "label_b": label_b,
    }


def derived_change_stats(rows: list[dict]) -> dict:
    """Compute NDVI % change and LST °C change from real raster statistics only.

    Returns an empty dict for any comparison whose inputs are unavailable —
    missing values are never invented.
    """
    by_name = {r["name"]: r for r in rows if r.get("n", 0)}
    out: dict = {}

    n04, n24 = by_name.get("ndvi_2004"), by_name.get("ndvi_2024")
    if n04 and n24:
        delta = n24["mean"] - n04["mean"]
        ndvi = {
            "mean_2004": n04["mean"],
            "mean_2024": n24["mean"],
            "mean_change": delta,
            "std_2004": n04["std"],
            "std_2024": n24["std"],
        }
        # Percent change is only meaningful when the baseline mean is non-zero.
        if abs(n04["mean"]) > 1e-6:
            ndvi["percent_change"] = 100.0 * delta / abs(n04["mean"])
        out["ndvi"] = ndvi

    l04, l24 = by_name.get("lst_2004"), by_name.get("lst_2024")
    if l04 and l24:
        out["lst"] = {
            "mean_2004": l04["mean"],
            "mean_2024": l24["mean"],
            "mean_change_c": l24["mean"] - l04["mean"],
            "min_2004": l04["min"],
            "max_2004": l04["max"],
            "min_2024": l24["min"],
            "max_2024": l24["max"],
            "std_2004": l04["std"],
            "std_2024": l24["std"],
        }
    return out


def stats_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def save_stats(rows: list[dict], extras: dict | None = None) -> Path:
    PATHS["stats"].mkdir(parents=True, exist_ok=True)
    df = stats_table(rows)
    csv_path = PATHS["stats"] / "raster_statistics.csv"
    json_path = PATHS["stats"] / "analysis_summary.json"
    df.to_csv(csv_path, index=False)
    payload = {"rasters": rows}
    if extras:
        payload.update(extras)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path
