"""Convert GeoTIFFs to PNG + Leaflet-ready metadata for the web GIS.

Color conventions match the Mahbubnagar LULC workflow:
  NDVI  — brown (bare) → yellow-green → dark green (vegetation)
  LST   — dark blue (cool) → cyan → yellow → orange → dark red (hot)
  LULC  — Water #1f77b4, Vegetation #2ca02c, Built-up #d62728,
           Barren #8c6d4b, Agricultural #e6c300
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from matplotlib import cm
from matplotlib import colors as mcolors
from PIL import Image

from config import PATHS
from src.io_utils import RasterBundle

# ---------------------------------------------------------------------------
# Color ramps
# ---------------------------------------------------------------------------

# NDVI: natural satellite green scale (like GEE NDVI palette)
# bare/low → pale yellow → light green → medium green → deep forest green
NDVI_RAMP = mcolors.LinearSegmentedColormap.from_list(
    "ndvi_natural",
    [
        "#d7c29e",  # pale tan (bare soil / very low NDVI)
        "#e8d87a",  # pale yellow (sparse)
        "#c8e06a",  # yellow-green (very sparse vegetation)
        "#8cc94a",  # light green
        "#4aab2e",  # medium green
        "#1e8c1e",  # green
        "#0a6614",  # dark green
        "#003d0a",  # deep forest green (dense vegetation)
    ],
)

# LST: natural thermal — cool blue → white → warm orange → hot red
# Matches standard land surface temperature visualization
LST_RAMP = mcolors.LinearSegmentedColormap.from_list(
    "lst_natural",
    [
        "#313695",  # deep blue (very cool)
        "#4575b4",  # blue
        "#74add1",  # light blue
        "#e0f3f8",  # near white (moderate)
        "#fee090",  # pale yellow
        "#fc8d59",  # orange
        "#d73027",  # red
        "#a50026",  # dark red (very hot)
    ],
)

PALETTES = {
    "ndvi": NDVI_RAMP,
    "ndvi_change": cm.RdYlGn,   # red=decrease, green=increase
    "lst": LST_RAMP,
    "lst_change": cm.RdBu_r,    # red=warming, blue=cooling
    "lulc": None,
    "lulc_change": None,
}

# Natural earth-tone LULC class colors (satellite-style)
LULC_CLASS_COLORS = {
    1: (70, 130, 180),   # Water — steel blue (natural water)
    2: (34, 139, 34),    # Vegetation — forest green (natural)
    3: (178, 34, 34),    # Built-up — firebrick red
    4: (194, 178, 128),  # Barren Land — sandy/tan (natural bare soil)
    5: (154, 205, 50),   # Agricultural/Cropland — yellow-green (crops)
}

LULC_CHANGE_COLORS = {
    0: (200, 204, 210, 180),  # No change — grey semi-transparent
    1: (20, 20, 20, 220),     # Changed — near-black
}

UNITS = {
    "ndvi": "", "ndvi_change": " ΔNDVI",
    "lst": " °C", "lst_change": " °C",
    "lulc": "", "lulc_change": "",
}


def lulc_class_list() -> list[dict]:
    from config import LULC_CLASSES
    out = []
    for cid, name in LULC_CLASSES.items():
        rgb = LULC_CLASS_COLORS.get(cid, (150, 150, 150))
        out.append({"id": cid, "name": name, "color": "#{:02x}{:02x}{:02x}".format(*rgb)})
    return out


def _cmap(kind: str):
    return PALETTES.get(kind, cm.viridis)


def _ramp_css(kind: str, n: int = 7) -> str:
    if kind in ("lulc", "lulc_change"):
        stops = ["#{:02x}{:02x}{:02x}".format(*rgb[:3]) for rgb in LULC_CLASS_COLORS.values()]
        return "linear-gradient(90deg, " + ", ".join(stops) + ")"
    cmap = _cmap(kind)
    hexes = []
    for i in range(n):
        r, g, b, _ = cmap(i / (n - 1))
        hexes.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return "linear-gradient(90deg, " + ", ".join(hexes) + ")"


def _stretch(array: np.ndarray, kind: str) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    if "change" in kind and kind != "lulc_change":
        mag = max(abs(lo), abs(hi), 1e-9)
        return -mag, mag
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo, hi


def load_village_geometries() -> list[dict] | None:
    geojson_path = PATHS["boundary"] / "rajapeta_boundary.geojson"
    if not geojson_path.exists():
        return None
    try:
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("type") == "FeatureCollection":
        geoms = [f["geometry"] for f in data.get("features", []) if f.get("geometry")]
    elif data.get("type") == "Feature":
        geoms = [data["geometry"]] if data.get("geometry") else []
    else:
        geoms = [data] if data.get("type") else []
    return geoms or None


def _village_mask(bundle: RasterBundle) -> np.ndarray | None:
    geoms = load_village_geometries()
    if not geoms:
        return None
    transform = bundle.profile.get("transform")
    if transform is None:
        return None
    try:
        from rasterio.features import geometry_mask
        from rasterio.crs import CRS
        from rasterio.warp import transform_geom

        if bundle.crs is not None and bundle.crs != CRS.from_epsg(4326):
            geoms = [transform_geom("EPSG:4326", bundle.crs, g) for g in geoms]
        inside = geometry_mask(
            geoms,
            out_shape=(bundle.profile["height"], bundle.profile["width"]),
            transform=transform,
            invert=True,
        )
        return inside
    except Exception:
        return None


def _colorize(array: np.ndarray, kind: str, vmin: float, vmax: float) -> np.ndarray:
    rgba = np.zeros(array.shape + (4,), dtype=np.uint8)
    valid = np.isfinite(array)
    if not valid.any():
        return rgba

    if kind == "lulc":
        for cls, (r, g, b) in LULC_CLASS_COLORS.items():
            mask = (array == cls)
            rgba[mask] = (r, g, b, 220)
        return rgba

    if kind == "lulc_change":
        for val, color in LULC_CHANGE_COLORS.items():
            mask = (array == val) & valid
            rgba[mask] = color
        return rgba

    scaled = np.clip((array - vmin) / (vmax - vmin + 1e-9), 0, 1)
    colored = (_cmap(kind)(scaled) * 255).astype(np.uint8)
    rgba[valid] = colored[valid]
    rgba[~valid, 3] = 0
    return rgba


def raster_to_png(bundle: RasterBundle, name: str, kind: str, vmin=None, vmax=None) -> dict:
    PATHS["web_data"].mkdir(parents=True, exist_ok=True)
    arr = bundle.array
    vmin, vmax = _stretch(arr, kind)
    rgba = _colorize(arr, kind, vmin, vmax)

    mask = _village_mask(bundle)
    if mask is not None:
        rgba[~mask, 3] = 0

    png_path = PATHS["web_data"] / f"{name}.png"
    Image.fromarray(rgba).save(png_path)

    west, south, east, north = bundle.bounds
    inside_px = int(np.count_nonzero(np.isfinite(arr) & (mask if mask is not None else True)))
    return {
        "name": name,
        "png": f"data/{name}.png",
        "bounds": [[south, west], [north, east]],
        "vmin": float(vmin),
        "vmax": float(vmax),
        "kind": kind,
        "unit": UNITS.get(kind, ""),
        "ramp_css": _ramp_css(kind),
        "masked_to_boundary": mask is not None,
        "valid_pixels": inside_px,
        "available": True,
        "width": int(bundle.profile["width"]),
        "height": int(bundle.profile["height"]),
    }


def write_web_manifest(layers: dict, boundary_geojson: dict | None, stats: dict) -> Path:
    PATHS["web_data"].mkdir(parents=True, exist_ok=True)
    payload = {
        "study_area": {
            "village": "Rajapeta (Rajapet)",
            "mandal": "Wanaparthy Mandal",
            "district": "Wanaparthy",
            "state": "Telangana, India",
            "period": "2004–2024",
        },
        "layers": layers,
        "boundary": boundary_geojson,
        "stats": stats,
        "lulc_classes": lulc_class_list(),
        "lulc_available": bool(
            layers.get("lulc_2004", {}).get("available")
            or layers.get("lulc_2024", {}).get("available")
        ),
    }
    path = PATHS["web_data"] / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
