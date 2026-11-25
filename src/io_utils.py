"""Shared raster I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

from config import PATHS, EXPECTED_RASTERS, ROOT

# GeoTIFF export folder used by the GEE/Rajapet workflow.
ROOT_RAJAPET_GEOTIFF = ROOT / "Rajapet" / "GeoTiff"


class RasterBundle:
    def __init__(self, path: Path, array: np.ndarray, profile: dict, nodata):
        self.path = path
        self.array = array
        self.profile = profile
        self.nodata = nodata
        self.transform = profile["transform"]
        self.crs = profile.get("crs")
        self.bounds = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )

    def with_array(self, array: np.ndarray, path: Path | None = None) -> "RasterBundle":
        return RasterBundle(path or self.path, array, self.profile.copy(), self.nodata)


def find_raster(filename: str, search_dirs: Optional[list[Path]] = None) -> Optional[Path]:
    # User-provided GeoTIFFs may live in data/raw/ or the Rajapet/GeoTiff export
    # folder. Matching is by exact expected filename, so unrelated rasters
    # (e.g. Mahbubnagar) are never picked up.
    dirs = search_dirs or [
        PATHS["raw"],
        PATHS["processed"],
        PATHS["boundary"],
        ROOT_RAJAPET_GEOTIFF,
    ]
    for folder in dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate
        matches = list(folder.rglob(filename))
        if matches:
            return matches[0]
    return None


def load_raster(path: Path) -> RasterBundle:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)
        array = np.where(np.isfinite(array), array, np.nan)
        return RasterBundle(path, array, profile, nodata)


def write_raster(path: Path, array: np.ndarray, profile: dict, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    return path


def raster_metadata(bundle: RasterBundle) -> dict:
    res = (bundle.transform.a, -bundle.transform.e)
    return {
        "path": str(bundle.path),
        "crs": str(bundle.crs) if bundle.crs else None,
        "width": int(bundle.profile["width"]),
        "height": int(bundle.profile["height"]),
        "resolution_x": float(res[0]),
        "resolution_y": float(res[1]),
        "nodata": bundle.nodata,
        "valid_pixels": int(np.isfinite(bundle.array).sum()),
        "invalid_pixels": int((~np.isfinite(bundle.array)).sum()),
        "bounds": [float(v) for v in bundle.bounds],
    }


def align_to_reference(source: RasterBundle, reference: RasterBundle) -> RasterBundle:
    dst = np.full((reference.profile["height"], reference.profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=source.array,
        destination=dst,
        src_transform=source.transform,
        src_crs=source.crs,
        dst_transform=reference.transform,
        dst_crs=reference.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    profile = reference.profile.copy()
    return RasterBundle(source.path, dst, profile, np.nan)


def discover_inputs() -> dict[str, Optional[Path]]:
    found = {}
    for key, name in EXPECTED_RASTERS.items():
        found[key] = find_raster(name)
    return found
