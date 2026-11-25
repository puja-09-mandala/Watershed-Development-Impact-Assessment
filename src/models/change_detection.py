import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

DENORMALIZED_DIR = (
    PROJECT_DIR
    / "outputs"
    / "denormalized"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "change_detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

PREDICTED_NDVI_2024_FILE = (
    DENORMALIZED_DIR
    / "predicted_ndvi_2024_actual_scale.npy"
)

PREDICTED_LST_2024_FILE = (
    DENORMALIZED_DIR
    / "predicted_lst_2024_celsius.npy"
)

ACTUAL_NDVI_2004_FILE = (
    DENORMALIZED_DIR
    / "actual_ndvi_2004.npy"
)

ACTUAL_LST_2004_FILE = (
    DENORMALIZED_DIR
    / "actual_lst_2004_celsius.npy"
)

ACTUAL_NDVI_2024_FILE = (
    DENORMALIZED_DIR
    / "actual_ndvi_2024.npy"
)

ACTUAL_LST_2024_FILE = (
    DENORMALIZED_DIR
    / "actual_lst_2024_celsius.npy"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING DATA FOR CHANGE DETECTION")
print("==============================================")


predicted_ndvi_2024 = np.load(
    PREDICTED_NDVI_2024_FILE
)

predicted_lst_2024 = np.load(
    PREDICTED_LST_2024_FILE
)

actual_ndvi_2004 = np.load(
    ACTUAL_NDVI_2004_FILE
)

actual_lst_2004 = np.load(
    ACTUAL_LST_2004_FILE
)

actual_ndvi_2024 = np.load(
    ACTUAL_NDVI_2024_FILE
)

actual_lst_2024 = np.load(
    ACTUAL_LST_2024_FILE
)


# ============================================================
# SHAPES
# ============================================================

print("\n==============================================")
print("DATA SHAPES")
print("==============================================")


print(
    "Predicted NDVI 2024:",
    predicted_ndvi_2024.shape
)

print(
    "Predicted LST 2024:",
    predicted_lst_2024.shape
)

print(
    "Actual NDVI 2004:",
    actual_ndvi_2004.shape
)

print(
    "Actual LST 2004:",
    actual_lst_2004.shape
)

print(
    "Actual NDVI 2024:",
    actual_ndvi_2024.shape
)

print(
    "Actual LST 2024:",
    actual_lst_2024.shape
)


# ============================================================
# CHANGE DETECTION
# ============================================================

print("\n==============================================")
print("CALCULATING CHANGE")
print("==============================================")


# ------------------------------------------------------------
# ACTUAL CHANGE
# ------------------------------------------------------------

actual_ndvi_change = (
    actual_ndvi_2024
    - actual_ndvi_2004
)

actual_lst_change = (
    actual_lst_2024
    - actual_lst_2004
)


# ------------------------------------------------------------
# PREDICTED CHANGE
# ------------------------------------------------------------

predicted_ndvi_change = (
    predicted_ndvi_2024
    - actual_ndvi_2004
)

predicted_lst_change = (
    predicted_lst_2024
    - actual_lst_2004
)


# ============================================================
# CHANGE ERROR
# ============================================================

ndvi_change_error = (
    predicted_ndvi_change
    - actual_ndvi_change
)

lst_change_error = (
    predicted_lst_change
    - actual_lst_change
)


# ============================================================
# SAVE CHANGE ARRAYS
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

np.save(
    OUTPUT_DIR
    / "ndvi_change_error.npy",
    ndvi_change_error
)

np.save(
    OUTPUT_DIR
    / "lst_change_error_celsius.npy",
    lst_change_error
)


# ============================================================
# FUNCTION FOR STATISTICS
# ============================================================

def print_statistics(
    name,
    data,
    unit=""
):

    print(f"\n{name}")

    print(
        "  Minimum:",
        np.min(data),
        unit
    )

    print(
        "  Maximum:",
        np.max(data),
        unit
    )

    print(
        "  Mean:",
        np.mean(data),
        unit
    )

    print(
        "  Median:",
        np.median(data),
        unit
    )


# ============================================================
# ACTUAL CHANGE STATISTICS
# ============================================================

print("\n==============================================")
print("ACTUAL CHANGE STATISTICS")
print("==============================================")


print_statistics(
    "NDVI Change (2004 → 2024)",
    actual_ndvi_change
)

print_statistics(
    "LST Change (2004 → 2024)",
    actual_lst_change,
    "°C"
)


# ============================================================
# PREDICTED CHANGE STATISTICS
# ============================================================

print("\n==============================================")
print("PREDICTED CHANGE STATISTICS")
print("==============================================")


print_statistics(
    "Predicted NDVI Change",
    predicted_ndvi_change
)

print_statistics(
    "Predicted LST Change",
    predicted_lst_change,
    "°C"
)


# ============================================================
# CHANGE DETECTION ERROR
# ============================================================

print("\n==============================================")
print("CHANGE DETECTION ERROR")
print("==============================================")


ndvi_change_mae = np.mean(
    np.abs(ndvi_change_error)
)

lst_change_mae = np.mean(
    np.abs(lst_change_error)
)


ndvi_change_rmse = np.sqrt(
    np.mean(
        ndvi_change_error ** 2
    )
)

lst_change_rmse = np.sqrt(
    np.mean(
        lst_change_error ** 2
    )
)


print(
    "NDVI Change MAE:",
    ndvi_change_mae
)

print(
    "NDVI Change RMSE:",
    ndvi_change_rmse
)

print(
    "LST Change MAE:",
    lst_change_mae,
    "°C"
)

print(
    "LST Change RMSE:",
    lst_change_rmse,
    "°C"
)


# ============================================================
# CHANGE DIRECTION
# ============================================================

print("\n==============================================")
print("CHANGE DIRECTION")
print("==============================================")


# NDVI

ndvi_increase = (
    actual_ndvi_change > 0
)

ndvi_decrease = (
    actual_ndvi_change < 0
)

ndvi_no_change = (
    actual_ndvi_change == 0
)


# LST

lst_increase = (
    actual_lst_change > 0
)

lst_decrease = (
    actual_lst_change < 0
)

lst_no_change = (
    actual_lst_change == 0
)


print("\nActual NDVI:")

print(
    "  Increased pixels:",
    np.sum(ndvi_increase)
)

print(
    "  Decreased pixels:",
    np.sum(ndvi_decrease)
)

print(
    "  No-change pixels:",
    np.sum(ndvi_no_change)
)


print("\nActual LST:")

print(
    "  Increased pixels:",
    np.sum(lst_increase)
)

print(
    "  Decreased pixels:",
    np.sum(lst_decrease)
)

print(
    "  No-change pixels:",
    np.sum(lst_no_change)
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("CHANGE DETECTION COMPLETED")
print("==============================================")


print("\nOutputs saved in:")

print(
    OUTPUT_DIR
)

print("\nFiles:")

print(
    "actual_ndvi_change.npy"
)

print(
    "actual_lst_change_celsius.npy"
)

print(
    "predicted_ndvi_change.npy"
)

print(
    "predicted_lst_change_celsius.npy"
)

print(
    "ndvi_change_error.npy"
)

print(
    "lst_change_error_celsius.npy"
)


print("\n==============================================")
print("DONE")
print("==============================================")