import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "patches"
    / "temporal_patches.npy"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "normalized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEMPORAL DATA
# ============================================================

print("\n==============================================")
print("LOADING TEMPORAL PATCHES")
print("==============================================")

data = np.load(INPUT_FILE)

print("Input shape:", data.shape)


# ============================================================
# EXPECTED SHAPE
# ============================================================

# Expected:
#
# (number_of_patches,
#  time_steps,
#  height,
#  width,
#  channels)
#
# Current:
#
# (3, 2, 128, 128, 2)
#
# Channel 0 = NDVI
# Channel 1 = LST

if data.ndim != 5:

    raise ValueError(
        "ERROR: Expected 5-dimensional data."
    )


num_patches = data.shape[0]
time_steps = data.shape[1]
height = data.shape[2]
width = data.shape[3]
channels = data.shape[4]


print("\n==============================================")
print("DATA INFORMATION")
print("==============================================")

print("Number of patches:", num_patches)
print("Time steps:", time_steps)
print("Height:", height)
print("Width:", width)
print("Channels:", channels)


# ============================================================
# CHECK CHANNELS
# ============================================================

if channels != 2:

    raise ValueError(
        "ERROR: Expected 2 channels: NDVI and LST."
    )


# ============================================================
# CREATE COPY
# ============================================================

normalized_data = data.copy().astype(
    np.float32
)


# ============================================================
# NDVI NORMALIZATION
# ============================================================

print("\n==============================================")
print("NORMALIZING NDVI")
print("==============================================")


ndvi = normalized_data[:, :, :, :, 0]


ndvi_min = np.min(ndvi)
ndvi_max = np.max(ndvi)


print("NDVI minimum:", ndvi_min)
print("NDVI maximum:", ndvi_max)


# Min-Max normalization
#
# New range = 0 to 1

if ndvi_max != ndvi_min:

    normalized_data[:, :, :, :, 0] = (
        (ndvi - ndvi_min)
        / (ndvi_max - ndvi_min)
    )

else:

    normalized_data[:, :, :, :, 0] = 0.0


# ============================================================
# LST NORMALIZATION
# ============================================================

print("\n==============================================")
print("NORMALIZING LST")
print("==============================================")


lst = normalized_data[:, :, :, :, 1]


lst_min = np.min(lst)
lst_max = np.max(lst)


print("LST minimum:", lst_min)
print("LST maximum:", lst_max)


# Min-Max normalization
#
# New range = 0 to 1

if lst_max != lst_min:

    normalized_data[:, :, :, :, 1] = (
        (lst - lst_min)
        / (lst_max - lst_min)
    )

else:

    normalized_data[:, :, :, :, 1] = 0.0


# ============================================================
# VERIFY NORMALIZED DATA
# ============================================================

print("\n==============================================")
print("NORMALIZATION CHECK")
print("==============================================")


normalized_ndvi = (
    normalized_data[:, :, :, :, 0]
)

normalized_lst = (
    normalized_data[:, :, :, :, 1]
)


print(
    "Normalized NDVI minimum:",
    np.min(normalized_ndvi)
)

print(
    "Normalized NDVI maximum:",
    np.max(normalized_ndvi)
)

print(
    "Normalized LST minimum:",
    np.min(normalized_lst)
)

print(
    "Normalized LST maximum:",
    np.max(normalized_lst)
)


# ============================================================
# SAVE NORMALIZED DATA
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "normalized_temporal_patches.npy"
)


np.save(
    OUTPUT_FILE,
    normalized_data
)


# ============================================================
# SAVE NORMALIZATION PARAMETERS
# ============================================================

normalization_parameters = np.array(
    [
        [ndvi_min, ndvi_max],
        [lst_min, lst_max]
    ],
    dtype=np.float32
)


PARAMETER_FILE = (
    OUTPUT_DIR
    / "normalization_parameters.npy"
)


np.save(
    PARAMETER_FILE,
    normalization_parameters
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("NORMALIZATION COMPLETED")
print("==============================================")


print(
    "Normalized data shape:",
    normalized_data.shape
)


print("\nChannel information:")
print("Channel 0 -> NDVI -> normalized to 0-1")
print("Channel 1 -> LST  -> normalized to 0-1")


print("\nSaved files:")

print(
    OUTPUT_FILE
)

print(
    PARAMETER_FILE
)


print("\n==============================================")
print("DONE")
print("==============================================")