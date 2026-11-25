import rasterio
import numpy as np
from pathlib import Path

# -----------------------------------------
# Paths
# -----------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# Function to read GeoTIFF
# -----------------------------------------

def read_tif(filename):
    path = RAW_DIR / filename

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        profile = src.profile

        # Convert NoData values to NaN
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    return data, profile


# -----------------------------------------
# Normalize data between 0 and 1
# -----------------------------------------

def normalize(data):

    valid = data[np.isfinite(data)]

    min_value = np.min(valid)
    max_value = np.max(valid)

    normalized = (data - min_value) / (
        max_value - min_value + 1e-8
    )

    return normalized


# -----------------------------------------
# Process NDVI
# -----------------------------------------

print("Processing NDVI...")

ndvi2004, profile = read_tif(
    "Mahbubnagar_NDVI_2004.tif"
)

ndvi2024, _ = read_tif(
    "Mahbubnagar_NDVI_2024.tif"
)

ndvi2004_norm = normalize(ndvi2004)
ndvi2024_norm = normalize(ndvi2024)


# -----------------------------------------
# Process LST
# -----------------------------------------

print("Processing LST...")

lst2004, _ = read_tif(
    "Mahbubnagar_LST_2004.tif"
)

lst2024, _ = read_tif(
    "Mahbubnagar_LST_2024.tif"
)

lst2004_norm = normalize(lst2004)
lst2024_norm = normalize(lst2024)


# -----------------------------------------
# Process LULC
# -----------------------------------------

print("Processing LULC...")

lulc2004, _ = read_tif(
    "Mahbubnagar_LULC_2004.tif"
)

lulc2024, _ = read_tif(
    "Mahbubnagar_LULC_2024.tif"
)


# -----------------------------------------
# Save processed arrays
# -----------------------------------------

np.save(
    PROCESSED_DIR / "ndvi_2004.npy",
    ndvi2004_norm
)

np.save(
    PROCESSED_DIR / "ndvi_2024.npy",
    ndvi2024_norm
)

np.save(
    PROCESSED_DIR / "lst_2004.npy",
    lst2004_norm
)

np.save(
    PROCESSED_DIR / "lst_2024.npy",
    lst2024_norm
)

np.save(
    PROCESSED_DIR / "lulc_2004.npy",
    lulc2004
)

np.save(
    PROCESSED_DIR / "lulc_2024.npy",
    lulc2024
)


# -----------------------------------------
# Final information
# -----------------------------------------

print("\n================================")
print("PREPROCESSING COMPLETED")
print("================================")

print("NDVI 2004 shape:", ndvi2004_norm.shape)
print("NDVI 2024 shape:", ndvi2024_norm.shape)

print("LST 2004 shape:", lst2004_norm.shape)
print("LST 2024 shape:", lst2024_norm.shape)

print("LULC 2004 shape:", lulc2004.shape)
print("LULC 2024 shape:", lulc2024.shape)

print("\nProcessed files saved in:")
print(PROCESSED_DIR)