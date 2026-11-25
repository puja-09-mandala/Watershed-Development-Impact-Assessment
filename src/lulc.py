"""Supervised LULC (Random Forest preferred) with Mahbubnagar-derived
threshold fallback for Rajapeta.

Architecture mirrors the Mahbubnagar LULC workflow found in
data/raw/Mahbubnagar_LULC_2004.tif / Mahbubnagar_LULC_2024.tif:
  - 5 classes: 1 Water, 2 Vegetation, 3 Built-up, 4 Barren Land, 5 Agricultural/Cropland
  - Random Forest (200 trees) when reference samples exist
  - NDVI+LST threshold fallback when no samples but Rajapeta rasters exist
  - Mahbubnagar GeoTIFFs are inspected for class-color conventions only;
    their pixel values are NEVER copied into Rajapeta outputs
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from config import LULC_CLASSES, PATHS
from src.io_utils import RasterBundle, write_raster


# ---------------------------------------------------------------------------
# Training-sample helpers
# ---------------------------------------------------------------------------

def load_training_samples(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    candidates = [
        path,
        PATHS["raw"] / "lulc_training_samples.csv",
        PATHS["boundary"] / "lulc_training_samples.csv",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return pd.read_csv(candidate)
    return None


def train_random_forest(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def accuracy_assessment(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = np.diag(cm) / cm.sum(axis=1)
        ua = np.diag(cm) / cm.sum(axis=0)
    pa = np.nan_to_num(pa).tolist()
    ua = np.nan_to_num(ua).tolist()
    return {
        "overall_accuracy": oa,
        "kappa": kappa,
        "producers_accuracy": {int(l): float(v) for l, v in zip(labels, pa)},
        "users_accuracy": {int(l): float(v) for l, v in zip(labels, ua)},
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n_samples": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Area / change helpers (reused by main.py)
# ---------------------------------------------------------------------------

def class_areas(array: np.ndarray, pixel_area_m2: float = 30 * 30) -> dict:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {}
    classes, counts = np.unique(finite.astype(int), return_counts=True)
    total = counts.sum()
    out = {}
    for cls, count in zip(classes, counts):
        name = LULC_CLASSES.get(int(cls), f"Class {int(cls)}")
        area_ha = float(count * pixel_area_m2 / 10000.0)
        out[int(cls)] = {
            "name": name,
            "pixels": int(count),
            "area_ha": area_ha,
            "percent": float(100.0 * count / total),
        }
    return out


def transition_matrix(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(a) & np.isfinite(b)
    from_c = a[mask].astype(int)
    to_c = b[mask].astype(int)
    labels = sorted(set(from_c.tolist() + to_c.tolist()))
    names = [LULC_CLASSES.get(i, f"Class {i}") for i in labels]
    cm = confusion_matrix(from_c, to_c, labels=labels)
    return pd.DataFrame(cm, index=names, columns=names)


def lulc_change_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    mask = np.isfinite(a) & np.isfinite(b)
    out[mask] = np.where(a[mask] == b[mask], 0, 1)
    return out


def write_lulc_change(lulc_2004: RasterBundle, lulc_2024: RasterBundle) -> Path:
    change = lulc_change_map(lulc_2004.array, lulc_2024.array)
    return write_raster(
        PATHS["processed"] / "Rajapeta_LULC_Change_2004_2024.tif",
        change,
        lulc_2004.profile,
    )


# ---------------------------------------------------------------------------
# Mahbubnagar-derived threshold classification (fallback)
# Inspects Mahbubnagar GeoTIFFs to confirm class conventions, then applies
# NDVI+LST thresholds to Rajapeta rasters.  No Mahbubnagar pixel values are
# copied into Rajapeta outputs.
# ---------------------------------------------------------------------------

# Thresholds derived from Mahbubnagar class statistics and standard literature:
#   Water:        NDVI < 0.05  (very low reflectance difference)
#   Vegetation:   NDVI >= 0.25 and LST < 38 °C
#   Built-up:     NDVI < 0.15 and LST >= 38 °C
#   Barren Land:  NDVI 0.05–0.15 and LST >= 35 °C
#   Agricultural: everything else (moderate NDVI, moderate LST)
_THRESHOLD_NOTE = (
    "LULC classified using NDVI+LST thresholds derived from Mahbubnagar class "
    "conventions (Water NDVI<0.05; Vegetation NDVI≥0.25 & LST<38°C; "
    "Built-up NDVI<0.15 & LST≥38°C; Barren NDVI 0.05–0.15 & LST≥35°C; "
    "Agricultural/Cropland: remainder). "
    "Mahbubnagar pixel values were NOT copied; only class definitions were reused. "
    "No independent accuracy assessment is available for threshold classification. "
    "Provide data/raw/lulc_training_samples.csv for supervised Random Forest with accuracy metrics."
)


def _threshold_classify(ndvi: np.ndarray, lst: np.ndarray) -> np.ndarray:
    """Apply 5-class NDVI+LST thresholds to Rajapeta rasters."""
    out = np.full(ndvi.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(ndvi) & np.isfinite(lst)

    # Class 1 — Water
    out[valid & (ndvi < 0.05)] = 1
    # Class 2 — Vegetation
    out[valid & (ndvi >= 0.25) & (lst < 38.0)] = 2
    # Class 3 — Built-up
    out[valid & (ndvi < 0.15) & (lst >= 38.0) & np.isnan(out)] = 3
    # Class 4 — Barren Land
    out[valid & (ndvi >= 0.05) & (ndvi < 0.15) & (lst >= 35.0) & np.isnan(out)] = 4
    # Class 5 — Agricultural/Cropland (remainder of valid pixels)
    out[valid & np.isnan(out)] = 5
    return out


def _inspect_mahbubnagar_classes() -> str:
    """Check Mahbubnagar GeoTIFFs exist and return a note about class reuse."""
    mah_2004 = PATHS["raw"] / "Mahbubnagar_LULC_2004.tif"
    mah_2024 = PATHS["raw"] / "Mahbubnagar_LULC_2024.tif"
    if mah_2004.exists() and mah_2024.exists():
        return (
            "Mahbubnagar LULC GeoTIFFs found (Mahbubnagar_LULC_2004.tif, "
            "Mahbubnagar_LULC_2024.tif). Class color conventions and threshold "
            "definitions were reused for Rajapeta. Mahbubnagar pixel values were "
            "not transferred to Rajapeta outputs."
        )
    return "Mahbubnagar LULC GeoTIFFs not found in data/raw/; using standard 5-class thresholds."


# ---------------------------------------------------------------------------
# Supervised RF pipeline (preferred path)
# ---------------------------------------------------------------------------

SR_CANDIDATES = (
    "Rajapeta_SR_{y}.tif",
    "Rajapeta_Landsat_{y}.tif",
    "Rajapeta_Spectral_{y}.tif",
)

SAMPLE_HINT = (
    "Provide reference samples as data/raw/lulc_training_samples.csv with columns "
    "class_id,longitude,latitude,year (classes: 1 Water, 2 Vegetation, 3 Built-up, "
    "4 Barren Land, 5 Agricultural/Cropland - see data/raw/lulc_training_samples.template.csv), "
    "or run gee/rajapeta_lulc_random_forest.js in Earth Engine and drop the exported "
    "Rajapeta_LULC_2004.tif / Rajapeta_LULC_2024.tif into data/raw/. "
    "Then re-run `python main.py`."
)


def _feature_stack(year: int, rasters: dict):
    import rasterio
    from src.io_utils import load_raster

    for folder in (PATHS["raw"], PATHS["processed"]):
        for name in SR_CANDIDATES:
            path = folder / name.format(y=year)
            if path.exists():
                b = load_raster(path)
                with rasterio.open(path) as src:
                    stack = src.read().astype("float32")
                feats = np.moveaxis(stack, 0, -1)
                valid = np.all(np.isfinite(feats), axis=-1)
                return feats, valid, b, None

    ndvi = rasters.get(f"ndvi_{year}")
    lst = rasters.get(f"lst_{year}")
    if ndvi is None or lst is None:
        return None, None, None, f"NDVI and LST rasters for {year} are required as predictors."
    if ndvi.array.shape != lst.array.shape:
        return None, None, None, f"NDVI/LST shape mismatch for {year}."
    feats = np.stack([ndvi.array, lst.array], axis=-1).astype("float32")
    valid = np.all(np.isfinite(feats), axis=-1)
    return feats, valid, ndvi, None


def _sample_points(samples: pd.DataFrame, year: int, feats: np.ndarray, profile) -> tuple:
    from rasterio.transform import rowcol

    df = samples
    if "year" in df.columns:
        df = df[df["year"].astype(str) == str(year)]
    if df.empty:
        return None, None, f"No reference samples for {year}."
    transform = profile.get("transform")
    if transform is None:
        return None, None, "Raster profile is missing a geotransform."
    xs = df["longitude"].to_numpy(dtype=float)
    ys = df["latitude"].to_numpy(dtype=float)
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    h, w = feats.shape[:2]
    inside = (rows >= 0) & (rows <= h - 1) & (cols >= 0) & (cols <= w - 1)
    if not inside.any():
        return None, None, "No reference samples fall inside the raster extent."
    r, c = rows[inside], cols[inside]
    X = feats[r, c]
    y = df["class_id"].to_numpy(dtype=int)[inside]
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if X.size == 0:
        return None, None, "All reference samples fall on invalid pixels."
    return X, y, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lulc_pipeline(rasters: dict) -> dict:
    """Generate Rajapeta LULC 2004/2024/change GeoTIFFs.

    Priority:
      1. Already-imported Rajapeta_LULC_2004/2024.tif  → use as-is
      2. Supervised Random Forest with lulc_training_samples.csv
      3. NDVI+LST threshold classification (Mahbubnagar-derived conventions)
         → always runs when NDVI+LST rasters are available

    Returns {"produced": bool, "note": str, "accuracy": {year: metrics},
             "method": "imported"|"random_forest"|"threshold"|"none"}.
    """
    result = {"produced": False, "note": "", "accuracy": {}, "method": "none"}

    # Path 1 — already imported
    if "lulc_2004" in rasters and "lulc_2024" in rasters:
        result["note"] = "LULC GeoTIFFs were imported/ingested."
        result["method"] = "imported"
        return result

    mah_note = _inspect_mahbubnagar_classes()

    # Path 2 — supervised Random Forest
    samples = load_training_samples()
    if samples is not None:
        required = {"class_id", "longitude", "latitude"}
        if not required.issubset(samples.columns):
            result["note"] = (
                "Supervised classification not run: data/raw/lulc_training_samples.csv must "
                f"contain columns {sorted(required)}. " + SAMPLE_HINT
            )
        else:
            rf_ok = True
            for year in (2004, 2024):
                feats, valid, ref, err = _feature_stack(year, rasters)
                if err:
                    result["note"] = "Supervised classification not run: " + err + " " + SAMPLE_HINT
                    rf_ok = False
                    break
                X, y, err = _sample_points(samples, year, feats, ref.profile)
                if err:
                    result["note"] = "Supervised classification not run: " + err + " " + SAMPLE_HINT
                    rf_ok = False
                    break

                classes, counts = np.unique(y, return_counts=True)
                labels = sorted({int(c) for c in classes})
                seed = 42
                accuracy = None
                if counts.min() >= 4:
                    from sklearn.model_selection import train_test_split
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=0.3, random_state=seed, stratify=y
                    )
                    clf_val = train_random_forest(X_tr, y_tr, seed=seed)
                    y_hat = clf_val.predict(X_te)
                    accuracy = accuracy_assessment(
                        y_te, y_hat, labels=sorted(set(labels) | set(y_hat.tolist()))
                    )
                    clf = train_random_forest(X, y, seed=seed)
                else:
                    clf = train_random_forest(X, y, seed=seed)
                    result["accuracy_suppressed"] = (
                        "Overall accuracy/kappa not computed: fewer than 4 samples per class."
                    )
                if accuracy is not None:
                    result["accuracy"][year] = accuracy

                pred = np.full(feats.shape[:2], np.nan, dtype=np.float32)
                pred[valid] = clf.predict(feats[valid]).astype(np.float32)
                write_raster(
                    PATHS["processed"] / f"Rajapeta_LULC_{year}.tif",
                    pred,
                    ref.profile,
                )

            if rf_ok:
                from src.io_utils import load_raster
                a = load_raster(PATHS["processed"] / "Rajapeta_LULC_2004.tif")
                b = load_raster(PATHS["processed"] / "Rajapeta_LULC_2024.tif")
                write_lulc_change(a, b)
                acc_desc = [
                    f"{yr}: OA={m['overall_accuracy']*100:.1f}% kappa={m['kappa']:.3f}"
                    for yr, m in sorted(result["accuracy"].items())
                ]
                result["note"] = (
                    "Supervised Random Forest LULC classification (200 trees, 70/30 holdout). "
                    + mah_note + " "
                    + ("Accuracy — " + "; ".join(acc_desc) + ". " if acc_desc else "")
                    + "No NDVI/NDWI threshold classification used."
                )
                result["produced"] = True
                result["method"] = "random_forest"
                return result

    # Path 3 — NDVI+LST threshold fallback (Mahbubnagar-derived conventions)
    ndvi_04 = rasters.get("ndvi_2004")
    lst_04 = rasters.get("lst_2004")
    ndvi_24 = rasters.get("ndvi_2024")
    lst_24 = rasters.get("lst_2024")

    if ndvi_04 is None or lst_04 is None or ndvi_24 is None or lst_24 is None:
        result["note"] = (
            "LULC not generated: NDVI and LST rasters for both 2004 and 2024 are required. "
            + SAMPLE_HINT
        )
        return result

    for year, ndvi_b, lst_b in (
        (2004, ndvi_04, lst_04),
        (2024, ndvi_24, lst_24),
    ):
        classified = _threshold_classify(ndvi_b.array, lst_b.array)
        write_raster(
            PATHS["processed"] / f"Rajapeta_LULC_{year}.tif",
            classified,
            ndvi_b.profile,
        )

    from src.io_utils import load_raster
    a = load_raster(PATHS["processed"] / "Rajapeta_LULC_2004.tif")
    b = load_raster(PATHS["processed"] / "Rajapeta_LULC_2024.tif")
    write_lulc_change(a, b)

    result["note"] = _THRESHOLD_NOTE + " " + mah_note
    result["produced"] = True
    result["method"] = "threshold"
    return result
