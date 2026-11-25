import numpy as np
from pathlib import Path

import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

MODEL_FILE = (
    PROJECT_DIR
    / "outputs"
    / "models"
    / "cnn_lstm_model.keras"
)

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "split"
    / "test_data.npy"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==============================================")
print("LOADING CNN-LSTM MODEL")
print("==============================================")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n==============================================")
print("LOADING TEST DATA")
print("==============================================")

test_data = np.load(
    TEST_FILE
)

print(
    "Test data shape:",
    test_data.shape
)


# ============================================================
# PREPARE INPUT
# ============================================================

# Input format:
#
# (samples,
#  time_steps,
#  128,
#  128,
#  2)
#
# Channel 0 = NDVI
# Channel 1 = LST
#
# Time step 0 = 2004
# Time step 1 = 2024

X_test = test_data


# ============================================================
# ACTUAL 2024 DATA
# ============================================================

actual_2024 = test_data[:, 1, :, :, :]


print("\n==============================================")
print("DATA INFORMATION")
print("==============================================")

print(
    "Input shape:",
    X_test.shape
)

print(
    "Actual 2024 shape:",
    actual_2024.shape
)


# ============================================================
# GENERATE PREDICTION
# ============================================================

print("\n==============================================")
print("GENERATING PREDICTION")
print("==============================================")


predicted_2024 = model.predict(
    X_test,
    verbose=1
)


print(
    "Predicted 2024 shape:",
    predicted_2024.shape
)


# ============================================================
# SPLIT PREDICTION INTO NDVI AND LST
# ============================================================

predicted_ndvi_2024 = (
    predicted_2024[:, :, :, 0]
)

predicted_lst_2024 = (
    predicted_2024[:, :, :, 1]
)


actual_ndvi_2024 = (
    actual_2024[:, :, :, 0]
)

actual_lst_2024 = (
    actual_2024[:, :, :, 1]
)


# ============================================================
# 2004 DATA
# ============================================================

ndvi_2004 = (
    test_data[:, 0, :, :, 0]
)

lst_2004 = (
    test_data[:, 0, :, :, 1]
)


# ============================================================
# CHANGE DETECTION
# ============================================================

print("\n==============================================")
print("CALCULATING CHANGE")
print("==============================================")


# Change in normalized scale

predicted_ndvi_change = (
    predicted_ndvi_2024
    - ndvi_2004
)

predicted_lst_change = (
    predicted_lst_2024
    - lst_2004
)


actual_ndvi_change = (
    actual_ndvi_2024
    - ndvi_2004
)

actual_lst_change = (
    actual_lst_2024
    - lst_2004
)


# ============================================================
# SAVE FULL PREDICTION
# ============================================================

predicted_file = (
    OUTPUT_DIR
    / "predicted_2024.npy"
)

np.save(
    predicted_file,
    predicted_2024
)


# ============================================================
# SAVE PREDICTED NDVI
# ============================================================

predicted_ndvi_file = (
    OUTPUT_DIR
    / "predicted_ndvi_2024.npy"
)

np.save(
    predicted_ndvi_file,
    predicted_ndvi_2024
)


# ============================================================
# SAVE PREDICTED LST
# ============================================================

predicted_lst_file = (
    OUTPUT_DIR
    / "predicted_lst_2024.npy"
)

np.save(
    predicted_lst_file,
    predicted_lst_2024
)


# ============================================================
# SAVE CHANGE MAPS
# ============================================================

predicted_ndvi_change_file = (
    OUTPUT_DIR
    / "predicted_ndvi_change.npy"
)

predicted_lst_change_file = (
    OUTPUT_DIR
    / "predicted_lst_change.npy"
)


np.save(
    predicted_ndvi_change_file,
    predicted_ndvi_change
)

np.save(
    predicted_lst_change_file,
    predicted_lst_change
)


# ============================================================
# SAVE ACTUAL CHANGE
# ============================================================

actual_ndvi_change_file = (
    OUTPUT_DIR
    / "actual_ndvi_change.npy"
)

actual_lst_change_file = (
    OUTPUT_DIR
    / "actual_lst_change.npy"
)


np.save(
    actual_ndvi_change_file,
    actual_ndvi_change
)

np.save(
    actual_lst_change_file,
    actual_lst_change
)


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print("\n==============================================")
print("PREDICTION STATISTICS")
print("==============================================")


print(
    "Predicted NDVI min:",
    np.min(predicted_ndvi_2024)
)

print(
    "Predicted NDVI max:",
    np.max(predicted_ndvi_2024)
)

print(
    "Predicted NDVI mean:",
    np.mean(predicted_ndvi_2024)
)


print(
    "\nPredicted LST min:",
    np.min(predicted_lst_2024)
)

print(
    "Predicted LST max:",
    np.max(predicted_lst_2024)
)

print(
    "Predicted LST mean:",
    np.mean(predicted_lst_2024)
)


# ============================================================
# CHANGE STATISTICS
# ============================================================

print("\n==============================================")
print("CHANGE STATISTICS")
print("==============================================")


print(
    "Predicted NDVI change mean:",
    np.mean(predicted_ndvi_change)
)

print(
    "Predicted LST change mean:",
    np.mean(predicted_lst_change)
)


# ============================================================
# COMPARE PREDICTED VS ACTUAL 2024
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
print("PREDICTION ERROR")
print("==============================================")


print(
    "NDVI prediction MAE:",
    ndvi_mae
)

print(
    "LST prediction MAE:",
    lst_mae
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("PREDICTION COMPLETED")
print("==============================================")


print("\nSaved files:")

print(
    predicted_file
)

print(
    predicted_ndvi_file
)

print(
    predicted_lst_file
)

print(
    predicted_ndvi_change_file
)

print(
    predicted_lst_change_file
)

print(
    actual_ndvi_change_file
)

print(
    actual_lst_change_file
)


print("\n==============================================")
print("DONE")
print("==============================================")