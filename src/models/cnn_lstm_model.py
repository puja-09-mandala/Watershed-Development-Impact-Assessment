import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(".")

TRAIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "split"
    / "train_data.npy"
)

TEST_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "split"
    / "test_data.npy"
)

MODEL_DIR = (
    PROJECT_DIR
    / "outputs"
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

TIME_STEPS = 2
HEIGHT = 128
WIDTH = 128
CHANNELS = 2

EPOCHS = 20
BATCH_SIZE = 1


# ============================================================
# LOAD TRAIN / TEST DATA
# ============================================================

print("\n==============================================")
print("LOADING TRAIN / TEST DATA")
print("==============================================")

train_data = np.load(TRAIN_FILE)
test_data = np.load(TEST_FILE)

print("Train data shape:", train_data.shape)
print("Test data shape :", test_data.shape)


# ============================================================
# VERIFY DATA
# ============================================================

expected_train_shape = (
    train_data.shape[0],
    TIME_STEPS,
    HEIGHT,
    WIDTH,
    CHANNELS
)

if train_data.shape != expected_train_shape:

    raise ValueError(
        f"Unexpected train shape: {train_data.shape}"
    )


# ============================================================
# CNN-LSTM MODEL
# ============================================================

print("\n==============================================")
print("BUILDING CNN-LSTM MODEL")
print("==============================================")


model = models.Sequential(
    [

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        layers.Input(
            shape=(
                TIME_STEPS,
                HEIGHT,
                WIDTH,
                CHANNELS
            )
        ),


        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------

        layers.TimeDistributed(
            layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            )
        ),


        layers.TimeDistributed(
            layers.MaxPooling2D(
                pool_size=(2, 2)
            )
        ),


        layers.TimeDistributed(
            layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same"
            )
        ),


        layers.TimeDistributed(
            layers.MaxPooling2D(
                pool_size=(2, 2)
            )
        ),


        # ----------------------------------------------------
        # CONVERT SPATIAL FEATURES
        # INTO FEATURE VECTORS
        # ----------------------------------------------------

        layers.TimeDistributed(
            layers.GlobalAveragePooling2D()
        ),


        # ----------------------------------------------------
        # LSTM TEMPORAL LEARNING
        # ----------------------------------------------------

        layers.LSTM(
            64,
            return_sequences=False
        ),


        # ----------------------------------------------------
        # DENSE LAYERS
        # ----------------------------------------------------

        layers.Dense(
            64,
            activation="relu"
        ),


        layers.Dropout(
            0.2
        ),


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        layers.Dense(
            HEIGHT * WIDTH * CHANNELS,
            activation="sigmoid"
        ),


        layers.Reshape(
            (
                HEIGHT,
                WIDTH,
                CHANNELS
            )
        )

    ]
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n==============================================")
print("MODEL SUMMARY")
print("==============================================")

model.summary()


# ============================================================
# TRAINING DATA
# ============================================================

# Input:
# 2004 + 2024 temporal sequence
#
# Target:
# 2024 spatial representation

X_train = train_data

Y_train = train_data[:, -1, :, :, :]


# ============================================================
# TEST DATA
# ============================================================

X_test = test_data

Y_test = test_data[:, -1, :, :, :]


print("\n==============================================")
print("TRAINING DATA")
print("==============================================")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==============================================")
print("TRAINING CNN-LSTM")
print("==============================================")


history = model.fit(
    X_train,
    Y_train,
    validation_data=(
        X_test,
        Y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n==============================================")
print("MODEL EVALUATION")
print("==============================================")


loss, mae = model.evaluate(
    X_test,
    Y_test,
    verbose=0
)


print("Test Loss:", loss)
print("Test MAE :", mae)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = (
    MODEL_DIR
    / "cnn_lstm_model.keras"
)


model.save(
    MODEL_FILE
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_file = (
    MODEL_DIR
    / "training_history.npy"
)


np.save(
    history_file,
    history.history,
    allow_pickle=True
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("CNN-LSTM TRAINING COMPLETED")
print("==============================================")

print("Test Loss:", loss)
print("Test MAE :", mae)

print("\nModel saved at:")
print(MODEL_FILE)

print("\nTraining history saved at:")
print(history_file)

print("\n==============================================")
print("DONE")
print("==============================================") 