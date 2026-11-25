import rasterio
import numpy as np
import os


# ============================================================
# PROJECT DATA PATH
# ============================================================

DATA_DIR = "Rajapet/GeoTiff"


# ============================================================
# GEOTIFF FILES
# ============================================================

files = [
    "Rajapeta_NDVI_2004.tif",
    "Rajapeta_NDVI_2024.tif",
    "Rajapeta_NDVI_Change_2004_2024.tif",
    "Rajapeta_LST_2004.tif",
    "Rajapeta_LST_2024.tif",
    "Rajapeta_LST_Change_2004_2024.tif"
]


# ============================================================
# CHECK EACH FILE
# ============================================================

for filename in files:

    filepath = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    if not os.path.exists(filepath):
        print("❌ File NOT FOUND")
        print("Path:", filepath)
        continue

    with rasterio.open(filepath) as src:

        data = src.read(1).astype(float)

        # Convert NoData to NaN
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        print("✅ File found")

        print("Width       :", src.width)
        print("Height      :", src.height)
        print("Bands       :", src.count)

        print("CRS         :", src.crs)

        print("Resolution  :", src.res)

        print("Bounds      :", src.bounds)

        print("NoData      :", src.nodata)

        print("Min         :", np.nanmin(data))

        print("Max         :", np.nanmax(data))

        print("Mean        :", np.nanmean(data))

        print("Median      :", np.nanmedian(data))

        print("Valid Pixels:",
              np.sum(~np.isnan(data)))

        print("Total Pixels:",
              data.size)


print("\n")
print("=" * 70)
print("GEOTIFF CHECK COMPLETED")
print("=" * 70)