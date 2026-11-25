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
    / "normalized"
    / "normalized_temporal_patches.npy"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "split"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.67

RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING NORMALIZED DATA")
print("==============================================")

data = np.load(INPUT_FILE)

print("Input shape:", data.shape)


# ============================================================
# DATA INFORMATION
# ============================================================

num_samples = data.shape[0]

print("\nNumber of samples:", num_samples)


# ============================================================
# SHUFFLE SAMPLE INDICES
# ============================================================

np.random.seed(RANDOM_SEED)

indices = np.arange(num_samples)

np.random.shuffle(indices)


print("\nShuffled indices:", indices)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_size = int(
    num_samples * TRAIN_RATIO
)

# Make sure at least one sample
# remains in both sets

if train_size < 1:
    train_size = 1

if train_size >= num_samples:
    train_size = num_samples - 1


train_indices = indices[:train_size]

test_indices = indices[train_size:]


# ============================================================
# CREATE DATASETS
# ============================================================

train_data = data[train_indices]

test_data = data[test_indices]


# ============================================================
# INFORMATION
# ============================================================

print("\n==============================================")
print("TRAIN / TEST SPLIT")
print("==============================================")

print(
    "Train samples:",
    train_data.shape[0]
)

print(
    "Test samples:",
    test_data.shape[0]
)

print(
    "Train shape:",
    train_data.shape
)

print(
    "Test shape:",
    test_data.shape
)


# ============================================================
# SAVE
# ============================================================

train_file = (
    OUTPUT_DIR
    / "train_data.npy"
)

test_file = (
    OUTPUT_DIR
    / "test_data.npy"
)

indices_file = (
    OUTPUT_DIR
    / "split_indices.npy"
)


np.save(
    train_file,
    train_data
)

np.save(
    test_file,
    test_data
)

np.save(
    indices_file,
    np.array(
        [
            train_indices,
            test_indices
        ],
        dtype=object
    ),
    allow_pickle=True
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("TRAIN / TEST SPLIT COMPLETED")
print("==============================================")

print("Saved:")
print(train_file)
print(test_file)
print(indices_file)

print("\n==============================================")
print("DONE")
print("==============================================")