import numpy as np
import rasterio
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

GEOTIFF_DIR = PROJECT_DIR / "Rajapet" / "GeoTiff"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "patches"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

FILES = {
    "ndvi_2004": GEOTIFF_DIR / "Rajapeta_NDVI_2004.tif",
    "ndvi_2024": GEOTIFF_DIR / "Rajapeta_NDVI_2024.tif",
    "lst_2004": GEOTIFF_DIR / "Rajapeta_LST_2004.tif",
    "lst_2024": GEOTIFF_DIR / "Rajapeta_LST_2024.tif"
}


# ============================================================
# PATCH SETTINGS
# ============================================================

PATCH_SIZE = 128

# Overlapping patches
STRIDE = 32


# ============================================================
# LOAD GEOTIFF
# ============================================================

def load_geotiff(path):

    print("Loading:", path)

    with rasterio.open(path) as src:

        data = src.read(1).astype(np.float32)

        nodata = src.nodata

        print("  NoData value:", nodata)

        if nodata is not None:
            data[data == nodata] = np.nan

    return data


# ============================================================
# FILL NaN VALUES
# ============================================================

def fill_nan_with_median(data, name):

    nan_count = np.isnan(data).sum()

    total_pixels = data.size

    valid_pixels = total_pixels - nan_count

    print(f"\n{name}")
    print("  Total pixels :", total_pixels)
    print("  Valid pixels :", valid_pixels)
    print("  NaN pixels   :", nan_count)

    if valid_pixels == 0:
        raise ValueError(
            f"ERROR: {name} contains no valid pixels."
        )

    median_value = np.nanmedian(data)

    print(
        "  Median used for NaN filling:",
        median_value
    )

    data = np.where(
        np.isnan(data),
        median_value,
        data
    )

    return data


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING GEOTIFF DATA")
print("==============================================")

ndvi2004 = load_geotiff(FILES["ndvi_2004"])
ndvi2024 = load_geotiff(FILES["ndvi_2024"])

lst2004 = load_geotiff(FILES["lst_2004"])
lst2024 = load_geotiff(FILES["lst_2024"])


# ============================================================
# IMAGE INFORMATION
# ============================================================

print("\n==============================================")
print("IMAGE INFORMATION")
print("==============================================")

print("NDVI 2004 shape:", ndvi2004.shape)
print("NDVI 2024 shape:", ndvi2024.shape)

print("LST 2004 shape :", lst2004.shape)
print("LST 2024 shape :", lst2024.shape)


# ============================================================
# CHECK SHAPES
# ============================================================

if not (
    ndvi2004.shape
    == ndvi2024.shape
    == lst2004.shape
    == lst2024.shape
):

    raise ValueError(
        "ERROR: All GeoTIFF images must have the same dimensions."
    )


height, width = ndvi2004.shape

print("\nOriginal image size:", height, "x", width)
print("Patch size:", PATCH_SIZE)
print("Stride:", STRIDE)


# ============================================================
# HANDLE NaN VALUES
# ============================================================

print("\n==============================================")
print("HANDLING INVALID / NaN PIXELS")
print("==============================================")


ndvi2004 = fill_nan_with_median(
    ndvi2004,
    "NDVI 2004"
)

ndvi2024 = fill_nan_with_median(
    ndvi2024,
    "NDVI 2024"
)

lst2004 = fill_nan_with_median(
    lst2004,
    "LST 2004"
)

lst2024 = fill_nan_with_median(
    lst2024,
    "LST 2024"
)


# ============================================================
# PADDING
# ============================================================

padded_height = max(
    height,
    PATCH_SIZE
)

padded_width = max(
    width,
    PATCH_SIZE
)


# Make dimensions compatible with stride

if padded_height > PATCH_SIZE:

    remainder_h = (
        padded_height - PATCH_SIZE
    ) % STRIDE

    if remainder_h != 0:

        padded_height += (
            STRIDE - remainder_h
        )


if padded_width > PATCH_SIZE:

    remainder_w = (
        padded_width - PATCH_SIZE
    ) % STRIDE

    if remainder_w != 0:

        padded_width += (
            STRIDE - remainder_w
        )


print("\n==============================================")
print("PADDING INFORMATION")
print("==============================================")

print(
    "Padded image size:",
    padded_height,
    "x",
    padded_width
)


# ============================================================
# PADDING FUNCTION
# ============================================================

def pad_image(
    image,
    target_height,
    target_width
):

    pad_height = (
        target_height - image.shape[0]
    )

    pad_width = (
        target_width - image.shape[1]
    )

    return np.pad(
        image,
        (
            (0, pad_height),
            (0, pad_width)
        ),
        mode="edge"
    )


# ============================================================
# APPLY PADDING
# ============================================================

ndvi2004 = pad_image(
    ndvi2004,
    padded_height,
    padded_width
)

ndvi2024 = pad_image(
    ndvi2024,
    padded_height,
    padded_width
)

lst2004 = pad_image(
    lst2004,
    padded_height,
    padded_width
)

lst2024 = pad_image(
    lst2024,
    padded_height,
    padded_width
)


# ============================================================
# VERIFY NaN AFTER PADDING
# ============================================================

print("\n==============================================")
print("NaN CHECK AFTER PADDING")
print("==============================================")

print(
    "NDVI 2004 NaN:",
    np.isnan(ndvi2004).sum()
)

print(
    "NDVI 2024 NaN:",
    np.isnan(ndvi2024).sum()
)

print(
    "LST 2004 NaN:",
    np.isnan(lst2004).sum()
)

print(
    "LST 2024 NaN:",
    np.isnan(lst2024).sum()
)


# ============================================================
# STORAGE
# ============================================================

patches_2004 = []
patches_2024 = []


# ============================================================
# CREATE PATCHES
# ============================================================

print("\n==============================================")
print("CREATING 128 x 128 PATCHES")
print("==============================================")


patch_number = 0


for row in range(
    0,
    padded_height - PATCH_SIZE + 1,
    STRIDE
):

    for col in range(
        0,
        padded_width - PATCH_SIZE + 1,
        STRIDE
    ):

        # ----------------------------------------------------
        # 2004 NDVI
        # ----------------------------------------------------

        ndvi_patch_2004 = ndvi2004[
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]


        # ----------------------------------------------------
        # 2004 LST
        # ----------------------------------------------------

        lst_patch_2004 = lst2004[
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]


        # ----------------------------------------------------
        # 2024 NDVI
        # ----------------------------------------------------

        ndvi_patch_2024 = ndvi2024[
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]


        # ----------------------------------------------------
        # 2024 LST
        # ----------------------------------------------------

        lst_patch_2024 = lst2024[
            row:row + PATCH_SIZE,
            col:col + PATCH_SIZE
        ]


        # ----------------------------------------------------
        # STACK NDVI + LST
        # ----------------------------------------------------

        patch2004 = np.stack(
            [
                ndvi_patch_2004,
                lst_patch_2004
            ],
            axis=-1
        )


        patch2024 = np.stack(
            [
                ndvi_patch_2024,
                lst_patch_2024
            ],
            axis=-1
        )


        # ----------------------------------------------------
        # FINAL NaN CHECK
        # ----------------------------------------------------

        if (
            np.isnan(patch2004).any()
            or np.isnan(patch2024).any()
        ):

            continue


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        patches_2004.append(
            patch2004
        )

        patches_2024.append(
            patch2024
        )

        patch_number += 1


# ============================================================
# CONVERT TO NUMPY
# ============================================================

patches_2004 = np.array(
    patches_2004,
    dtype=np.float32
)

patches_2024 = np.array(
    patches_2024,
    dtype=np.float32
)


# ============================================================
# PATCH INFORMATION
# ============================================================

print("\n==============================================")
print("PATCH INFORMATION")
print("==============================================")

print(
    "2004 patches shape:",
    patches_2004.shape
)

print(
    "2024 patches shape:",
    patches_2024.shape
)

print(
    "Total valid patches:",
    patch_number
)


# ============================================================
# CHECK
# ============================================================

if patch_number == 0:

    raise ValueError(
        "ERROR: Still no patches were created."
    )


# ============================================================
# SAVE INDIVIDUAL PATCHES
# ============================================================

np.save(
    OUTPUT_DIR / "patches_2004.npy",
    patches_2004
)

np.save(
    OUTPUT_DIR / "patches_2024.npy",
    patches_2024
)


# ============================================================
# TEMPORAL STACKING
# ============================================================

# Shape:
#
# (number_of_patches,
#  2,
#  128,
#  128,
#  2)
#
# Time:
# 0 -> 2004
# 1 -> 2024
#
# Channels:
# 0 -> NDVI
# 1 -> LST


temporal_patches = np.stack(
    [
        patches_2004,
        patches_2024
    ],
    axis=1
)


# ============================================================
# SAVE TEMPORAL DATASET
# ============================================================

np.save(
    OUTPUT_DIR / "temporal_patches.npy",
    temporal_patches
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n==============================================")
print("PATCH CREATION COMPLETED")
print("==============================================")

print(
    "2004 patches:",
    patches_2004.shape
)

print(
    "2024 patches:",
    patches_2024.shape
)

print(
    "Temporal dataset:",
    temporal_patches.shape
)

print("\nFeatures:")
print("Channel 0 -> NDVI")
print("Channel 1 -> LST")

print("\nTime steps:")
print("Time step 0 -> 2004")
print("Time step 1 -> 2024")

print("\nSaved files:")

print(
    OUTPUT_DIR / "patches_2004.npy"
)

print(
    OUTPUT_DIR / "patches_2024.npy"
)

print(
    OUTPUT_DIR / "temporal_patches.npy"
)

print("\n==============================================")
print("DONE")
print("==============================================")