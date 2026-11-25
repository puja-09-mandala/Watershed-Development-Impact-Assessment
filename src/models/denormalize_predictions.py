import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

PREDICTION_DIR = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
)

NORMALIZATION_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "normalized"
)

SPLIT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "split"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "denormalized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

PREDICTED_2024_FILE = (
    PREDICTION_DIR
    / "predicted_2024.npy"
)

NORMALIZATION_FILE = (
    NORMALIZATION_DIR
    / "normalization_parameters.npy"
)

TEST_DATA_FILE = (
    SPLIT_DIR
    / "test_data.npy"
)


# ============================================================
# LOAD FILES
# ============================================================

print("\n==============================================")
print("LOADING PREDICTION DATA")
print("==============================================")

predicted_2024 = np.load(
    PREDICTED_2024_FILE
)

normalization_parameters = np.load(
    NORMALIZATION_FILE
)

test_data = np.load(
    TEST_DATA_FILE
)


print(
    "Predicted 2024 shape:",
    predicted_2024.shape
)

print(
    "Normalization parameters shape:",
    normalization_parameters.shape
)

print(
    "Test data shape:",
    test_data.shape
)


# ============================================================
# READ NORMALIZATION PARAMETERS
# ============================================================

# normalization_parameters:
#
# Row 0 -> NDVI
# Row 1 -> LST
#
# Column 0 -> minimum
# Column 1 -> maximum

ndvi_min = normalization_parameters[0, 0]
ndvi_max = normalization_parameters[0, 1]

lst_min = normalization_parameters[1, 0]
lst_max = normalization_parameters[1, 1]


print("\n==============================================")
print("NORMALIZATION PARAMETERS")
print("==============================================")

print("NDVI minimum:", ndvi_min)
print("NDVI maximum:", ndvi_max)

print("LST minimum :", lst_min)
print("LST maximum :", lst_max)


# ============================================================
# DENORMALIZATION FUNCTION
# ============================================================

def denormalize(
    normalized_data,
    minimum,
    maximum
):

    return (
        normalized_data
        * (maximum - minimum)
        + minimum
    )


# ============================================================
# EXTRACT PREDICTED NDVI AND LST
# ============================================================

predicted_ndvi_normalized = (
    predicted_2024[:, :, :, 0]
)

predicted_lst_normalized = (
    predicted_2024[:, :, :, 1]
)


# ============================================================
# DENORMALIZE PREDICTED 2024
# ============================================================

print("\n==============================================")
print("DENORMALIZING PREDICTIONS")
print("==============================================")


predicted_ndvi_2024 = denormalize(
    predicted_ndvi_normalized,
    ndvi_min,
    ndvi_max
)


predicted_lst_2024 = denormalize(
    predicted_lst_normalized,
    lst_min,
    lst_max
)


# ============================================================
# EXTRACT ACTUAL 2004 DATA
# ============================================================

actual_2004_ndvi_normalized = (
    test_data[:, 0, :, :, 0]
)

actual_2004_lst_normalized = (
    test_data[:, 0, :, :, 1]
)


# ============================================================
# DENORMALIZE ACTUAL 2004
# ============================================================

actual_ndvi_2004 = denormalize(
    actual_2004_ndvi_normalized,
    ndvi_min,
    ndvi_max
)


actual_lst_2004 = denormalize(
    actual_2004_lst_normalized,
    lst_min,
    lst_max
)


# ============================================================
# EXTRACT ACTUAL 2024 DATA
# ============================================================

actual_2024_ndvi_normalized = (
    test_data[:, 1, :, :, 0]
)

actual_2024_lst_normalized = (
    test_data[:, 1, :, :, 1]
)


# ============================================================
# DENORMALIZE ACTUAL 2024
# ============================================================

actual_ndvi_2024 = denormalize(
    actual_2024_ndvi_normalized,
    ndvi_min,
    ndvi_max
)


actual_lst_2024 = denormalize(
    actual_2024_lst_normalized,
    lst_min,
    lst_max
)


# ============================================================
# PREDICTED CHANGE
# ============================================================

predicted_ndvi_change = (
    predicted_ndvi_2024
    - actual_ndvi_2004
)


predicted_lst_change = (
    predicted_lst_2024
    - actual_lst_2004
)


# ============================================================
# ACTUAL CHANGE
# ============================================================

actual_ndvi_change = (
    actual_ndvi_2024
    - actual_ndvi_2004
)


actual_lst_change = (
    actual_lst_2024
    - actual_lst_2004
)


# ============================================================
# SAVE PREDICTED 2024
# ============================================================

np.save(
    OUTPUT_DIR
    / "predicted_ndvi_2024_actual_scale.npy",
    predicted_ndvi_2024
)


np.save(
    OUTPUT_DIR
    / "predicted_lst_2024_celsius.npy",
    predicted_lst_2024
)


# ============================================================
# SAVE ACTUAL 2004
# ============================================================

np.save(
    OUTPUT_DIR
    / "actual_ndvi_2004.npy",
    actual_ndvi_2004
)


np.save(
    OUTPUT_DIR
    / "actual_lst_2004_celsius.npy",
    actual_lst_2004
)


# ============================================================
# SAVE ACTUAL 2024
# ============================================================

np.save(
    OUTPUT_DIR
    / "actual_ndvi_2024.npy",
    actual_ndvi_2024
)


np.save(
    OUTPUT_DIR
    / "actual_lst_2024_celsius.npy",
    actual_lst_2024
)


# ============================================================
# SAVE PREDICTED CHANGE
# ============================================================

np.save(
    OUTPUT_DIR
    / "predicted_ndvi_change.npy",
    predicted_ndvi_change
)


np.save(
    OUTPUT_DIR
    / "predicted_lst_change_celsius.npy",
    predicted_lst_change
)


# ============================================================
# SAVE ACTUAL CHANGE
# ============================================================

np.save(
    OUTPUT_DIR
    / "actual_ndvi_change.npy",
    actual_ndvi_change
)


np.save(
    OUTPUT_DIR
    / "actual_lst_change_celsius.npy",
    actual_lst_change
)


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n==============================================")
print("DENORMALIZED RESULTS")
print("==============================================")


print("\nPredicted NDVI 2024:")
print(
    "Minimum:",
    np.min(predicted_ndvi_2024)
)
print(
    "Maximum:",
    np.max(predicted_ndvi_2024)
)
print(
    "Mean:",
    np.mean(predicted_ndvi_2024)
)


print("\nPredicted LST 2024 (°C):")
print(
    "Minimum:",
    np.min(predicted_lst_2024)
)
print(
    "Maximum:",
    np.max(predicted_lst_2024)
)
print(
    "Mean:",
    np.mean(predicted_lst_2024)
)


print("\nPredicted NDVI Change:")
print(
    "Minimum:",
    np.min(predicted_ndvi_change)
)
print(
    "Maximum:",
    np.max(predicted_ndvi_change)
)
print(
    "Mean:",
    np.mean(predicted_ndvi_change)
)


print("\nPredicted LST Change (°C):")
print(
    "Minimum:",
    np.min(predicted_lst_change)
)
print(
    "Maximum:",
    np.max(predicted_lst_change)
)
print(
    "Mean:",
    np.mean(predicted_lst_change)
)


# ============================================================
# COMPARE ACTUAL VS PREDICTED
# ============================================================

ndvi_mae = np.mean(
    np.abs(
        predicted_ndvi_2024
        - actual_ndvi_2024
    )
)


lst_mae = np.mean(
    np.abs(
        predicted_lst_2024
        - actual_lst_2024
    )
)


print("\n==============================================")
print("ACTUAL VS PREDICTED ERROR")
print("==============================================")


print(
    "NDVI MAE:",
    ndvi_mae
)

print(
    "LST MAE (°C):",
    lst_mae
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("DENORMALIZATION COMPLETED")
print("==============================================")


print("\nSaved inside:")

print(
    OUTPUT_DIR
)

print("\nMain outputs:")

print(
    "predicted_ndvi_2024_actual_scale.npy"
)

print(
    "predicted_lst_2024_celsius.npy"
)

print(
    "predicted_ndvi_change.npy"
)

print(
    "predicted_lst_change_celsius.npy"
)

print("\n==============================================")
print("DONE")
print("==============================================")