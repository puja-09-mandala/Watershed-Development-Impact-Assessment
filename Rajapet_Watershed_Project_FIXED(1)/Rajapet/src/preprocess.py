import numpy as np
import rasterio
from pathlib import Path

RAW_DIR  = Path("Rajapet/GeoTiff")
OUT_DIR  = Path("Rajapet/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========================================")
print("STEP 1: RAJAPET PREPROCESSING")
print("========================================")

def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data

def normalize(data):
    valid = data[np.isfinite(data)]
    mn, mx = valid.min(), valid.max()
    return (data - mn) / (mx - mn + 1e-8)

ndvi_2004 = read_tif(RAW_DIR / "Rajapeta_NDVI_2004.tif")
ndvi_2024 = read_tif(RAW_DIR / "Rajapeta_NDVI_2024.tif")
lst_2004  = read_tif(RAW_DIR / "Rajapeta_LST_2004.tif")
lst_2024  = read_tif(RAW_DIR / "Rajapeta_LST_2024.tif")

print(f"NDVI 2004 shape: {ndvi_2004.shape}, range: {np.nanmin(ndvi_2004):.3f} to {np.nanmax(ndvi_2004):.3f}")
print(f"NDVI 2024 shape: {ndvi_2024.shape}, range: {np.nanmin(ndvi_2024):.3f} to {np.nanmax(ndvi_2024):.3f}")
print(f"LST  2004 shape: {lst_2004.shape},  range: {np.nanmin(lst_2004):.2f} to {np.nanmax(lst_2004):.2f}")
print(f"LST  2024 shape: {lst_2024.shape},  range: {np.nanmin(lst_2024):.2f} to {np.nanmax(lst_2024):.2f}")

ndvi_2004_norm = normalize(np.nan_to_num(ndvi_2004, nan=0.0))
ndvi_2024_norm = normalize(np.nan_to_num(ndvi_2024, nan=0.0))
lst_2004_norm  = normalize(np.nan_to_num(lst_2004,  nan=0.0))
lst_2024_norm  = normalize(np.nan_to_num(lst_2024,  nan=0.0))

# Auto patch size based on image dimensions
h, w = ndvi_2004_norm.shape
print(f"Image size: {h} x {w}")
PATCH_SIZE = min(128, h, w)
STRIDE = PATCH_SIZE // 2
print(f"Using PATCH_SIZE={PATCH_SIZE}, STRIDE={STRIDE}")
patches_2004, patches_2024 = [], []

for r in range(0, h - PATCH_SIZE + 1, STRIDE):
    for c in range(0, w - PATCH_SIZE + 1, STRIDE):
        p04 = np.stack([
            ndvi_2004_norm[r:r+PATCH_SIZE, c:c+PATCH_SIZE],
            lst_2004_norm[r:r+PATCH_SIZE,  c:c+PATCH_SIZE]
        ], axis=-1)
        p24 = np.stack([
            ndvi_2024_norm[r:r+PATCH_SIZE, c:c+PATCH_SIZE],
            lst_2024_norm[r:r+PATCH_SIZE,  c:c+PATCH_SIZE]
        ], axis=-1)
        patches_2004.append(p04)
        patches_2024.append(p24)

patches_2004 = np.array(patches_2004, dtype=np.float32)
patches_2024 = np.array(patches_2024, dtype=np.float32)

# Shape: (N, 2, 128, 128, 2) — CNN-LSTM input format
X = np.stack([patches_2004, patches_2024], axis=1)

np.save(OUT_DIR / "ndvi_2004_norm.npy", ndvi_2004_norm)
np.save(OUT_DIR / "ndvi_2024_norm.npy", ndvi_2024_norm)
np.save(OUT_DIR / "lst_2004_norm.npy",  lst_2004_norm)
np.save(OUT_DIR / "lst_2024_norm.npy",  lst_2024_norm)
np.save(OUT_DIR / "ndvi_2004_raw.npy",  ndvi_2004)
np.save(OUT_DIR / "ndvi_2024_raw.npy",  ndvi_2024)
np.save(OUT_DIR / "lst_2004_raw.npy",   lst_2004)
np.save(OUT_DIR / "lst_2024_raw.npy",   lst_2024)
np.save(OUT_DIR / "X_rajapet.npy",      X)

print(f"\nPatches created: {X.shape}  (N, 2, 128, 128, 2)")
print("Saved to Rajapet/processed/")
print("\n========================================")
print("PREPROCESSING COMPLETE")
print("========================================")
