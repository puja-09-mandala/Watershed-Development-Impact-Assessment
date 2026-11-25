"""Multi-epoch (5-year) NDVI/LST/LULC trend statistics.

Only epochs that actually exist as GeoTIFFs are reported - never fabricated.
Any file matching Rajapeta_{NDVI|LST|LULC}_YYYY.tif in data/raw,
data/processed or Rajapet/GeoTiff is picked up automatically, so exporting the
2009/2014/2019 rasters extends the trend charts without code changes.

GEE helper: gee/rajapeta_multiyear_export.js exports the missing epochs.
"""

from __future__ import annotations

import re

import numpy as np

from config import LULC_CLASSES, PATHS
from src.io_utils import find_raster, load_raster

EPOCH_RE = re.compile(r"^Rajapeta_(NDVI|LST|LULC)_(\d{4})\.tif$")
FALLBACK_EPOCHS = (2004, 2009, 2014, 2019, 2024)
def available_epochs(kind: str) -> list[int]:
    """Years for which Rajapeta_{kind}_YYYY.tif exists on disk."""
    years = set()
    for folder in (PATHS["raw"], PATHS["processed"]):
        try:
            for p in folder.glob(f"Rajapeta_{kind}_????.tif"):
                m = EPOCH_RE.match(p.name)
                if m:
                    years.add(int(m.group(2)))
        except OSError:
            continue
    # find_raster also searches the Rajapet/GeoTiff export folder.
    for year in FALLBACK_EPOCHS:
        if find_raster(f"Rajapeta_{kind}_{year}.tif"):
            years.add(year)
    return sorted(years)
def _epoch_stats(kind: str, unit: str) -> dict:
    years, means, medians = [], [], []
    for year in available_epochs(kind):
        path = find_raster(f"Rajapeta_{kind}_{year}.tif")
        if path is None:
            continue
        arr = load_raster(path).array
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        years.append(year)
        means.append(round(float(np.mean(finite)), 4))
        medians.append(round(float(np.median(finite)), 4))
    return {"years": years, "mean": means, "median": medians, "unit": unit}


def _lulc_trend() -> dict:
    years, class_areas = [], {}
    for year in available_epochs("LULC"):
        path = find_raster(f"Rajapeta_LULC_{year}.tif")
        if path is None:
            continue
        arr = load_raster(path).array
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        years.append(year)
        vals, cnts = np.unique(finite.astype(int), return_counts=True)
        counts = dict(zip(vals.tolist(), cnts.tolist()))
        class_areas[str(year)] = {
            str(cid): round(float(counts.get(cid, 0) * 30 * 30 / 10000.0), 2)
            for cid in LULC_CLASSES
        }
    return {"years": years, "classes": LULC_CLASSES, "class_areas_ha": class_areas}


def compute_trends(rasters: dict | None = None) -> dict:
    """Trend series for the charts page (NDVI/LST always; LULC when present)."""
    ndvi = _epoch_stats("NDVI", "")
    lst = _epoch_stats("LST", "C")
    lulc = _lulc_trend()

    expected = list(FALLBACK_EPOCHS)
    missing = sorted(
        set(expected) - set(ndvi["years"]) | (set(expected) - set(lst["years"]))
    )
    note_parts = []
    if missing:
        note_parts.append(
            f"NDVI epochs present: {ndvi['years']}; LST epochs present: {lst['years']}. "
            f"Missing epochs: {missing}. Export them with gee/rajapeta_multiyear_export.js "
            "(GEE Code Editor) as Rajapeta_NDVI_YYYY.tif / Rajapeta_LST_YYYY.tif into "
            "data/raw/ and re-run `python main.py`; the trend charts extend automatically."
        )
    missing_lulc = [y for y in expected if y not in lulc["years"]]
    if missing_lulc:
        note_parts.append(
            f"LULC epochs present: {lulc['years'] or 'none'}; missing: {missing_lulc}."
        )
    if not note_parts:
        note_parts.append("All five 5-year epochs are available.")

    return {
        "ndvi": ndvi,
        "lst": lst,
        "lulc": lulc,
        "note": " ".join(note_parts),
        "policy": (
            "Only epochs with actual GeoTIFFs are plotted. No intermediate-year "
            "values are interpolated or simulated."
        ),
    }