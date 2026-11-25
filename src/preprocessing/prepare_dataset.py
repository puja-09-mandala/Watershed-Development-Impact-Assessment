import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# PATHS
# ==========================================

PATCH_DIR = Path("data/processed/patches")
OUTPUT_DIR = Path("data/processed/dataset")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD PATCHES
# ==========================================

print("Loading patches...")

x2004 = np.load(
    PATCH_DIR / "patches_2004.npy"
)

x2024 = np.load(
    PATCH_DIR / "patches_2024.npy"
)

y2004 = np.load(
    PATCH_DIR / "labels_2004.npy"
)

y2024 = np.load(
    PATCH_DIR / "labels_2024.npy"
)


print("2004 input shape:", x2004.shape)
print("2024 input shape:", x2024.shape)


# ==========================================
# CREATE TEMPORAL SEQUENCE
# ==========================================
#
# Sequence:
# Time 1 → 2004
# Time 2 → 2024
#
# Final shape:
# (samples, time, height, width, channels)
#
# channels:
# 0 → NDVI
# 1 → LST
# ==========================================

X = np.stack(
    [x2004, x2024],
    axis=1
)


print("\nTemporal sequence shape:", X.shape)


# ==========================================
# CREATE CHANGE TARGET
# ==========================================
#
# For the first version, use the 2024 LULC
# as the prediction target.
#
# Later we can create a dedicated change
# classification target.
# ==========================================

y = y2024


print("Target shape:", y.shape)


# ==========================================
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

indices = np.arange(len(X))


# 70% training, 30% temporary
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    shuffle=True
)


# 15% validation, 15% testing
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    shuffle=True
)


# ==========================================
# CREATE DATASETS
# ==========================================

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]


# ==========================================
# PRINT SPLIT INFORMATION
# ==========================================

print("\n================================")
print("DATASET SPLIT")
print("================================")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


print("\nX_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\ny_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


# ==========================================
# SAVE DATASETS
# ==========================================

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


print("\n================================")
print("DATASET PREPARATION COMPLETE")
print("================================")

print("Saved in:")
print(OUTPUT_DIR)