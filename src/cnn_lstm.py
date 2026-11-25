"""CNN-LSTM prototype for spatial feature extraction and reconstruction.

With only 2004 and 2024 observations this is a representation-learning /
reconstruction / change-analysis support model, not a validated forecast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MODEL, PATCH, PATHS


def _tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    return tf, layers, models, optimizers


def build_cnn_lstm(
    time_steps: int = PATCH["time_steps"],
    height: int = PATCH["size"],
    width: int = PATCH["size"],
    channels: int = 2,
    n_classes: int | None = None,
):
    tf, layers, models, optimizers = _tf()
    inp = layers.Input(shape=(time_steps, height, width, channels), name="spatiotemporal_input")

    x = layers.TimeDistributed(
        layers.Conv2D(MODEL["conv1_filters"], MODEL["kernel_size"], padding="same"),
        name="td_conv1",
    )(inp)
    x = layers.TimeDistributed(layers.Activation("relu"), name="td_relu1")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool1")(x)

    x = layers.TimeDistributed(
        layers.Conv2D(MODEL["conv2_filters"], MODEL["kernel_size"], padding="same"),
        name="td_conv2",
    )(x)
    x = layers.TimeDistributed(layers.Activation("relu"), name="td_relu2")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool2")(x)

    feat_h, feat_w, feat_c = height // 4, width // 4, MODEL["conv2_filters"]
    x = layers.TimeDistributed(layers.Reshape((feat_h * feat_w * feat_c,)), name="td_flatten")(x)
    x = layers.LSTM(MODEL["lstm_units"], name="temporal_lstm")(x)
    x = layers.Dropout(MODEL["dropout"])(x)
    x = layers.Dense(MODEL["dense_units"], activation="relu", name="dense_latent")(x)
    x = layers.Dense(feat_h * feat_w * feat_c, activation="relu", name="spatial_restore")(x)
    x = layers.Reshape((feat_h, feat_w, feat_c))(x)
    x = layers.UpSampling2D(2, name="up1")(x)
    x = layers.Conv2D(MODEL["conv1_filters"], 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2, name="up2")(x)
    recon = layers.Conv2D(channels, 3, padding="same", activation="sigmoid", name="reconstruction")(x)

    model = models.Model(inp, recon, name="cnn_lstm_reconstruction")

    if n_classes:
        cls = layers.Conv2D(n_classes, 1, activation="softmax", name="lulc_head")(x)
        model = models.Model(inp, [recon, cls], name="cnn_lstm_recon_lulc")

    model.compile(
        optimizer=optimizers.Adam(MODEL["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def reconstruction_targets(patches: np.ndarray) -> np.ndarray:
    """Predict / reconstruct the 2024 spatial fields from the two-date sequence."""
    return patches[:, 1, :, :, :]


def train_and_evaluate(patches: np.ndarray, use_lulc_head: bool = False) -> dict:
    PATHS["models"].mkdir(parents=True, exist_ok=True)
    n = len(patches)
    if n < 4:
        return {
            "trained": False,
            "reason": f"Insufficient patches ({n}). Need at least 4 valid 128x128 patches.",
        }

    from sklearn.model_selection import train_test_split

    y = reconstruction_targets(patches)
    x_train, x_test, y_train, y_test = train_test_split(
        patches,
        y,
        test_size=MODEL["test_split"],
        random_state=MODEL["random_seed"],
    )

    model = build_cnn_lstm()
    history = model.fit(
        x_train,
        y_train,
        validation_split=MODEL["validation_split"],
        epochs=MODEL["epochs"],
        batch_size=min(MODEL["batch_size"], len(x_train)),
        verbose=1,
    )
    model_path = PATHS["models"] / "cnn_lstm_prototype.keras"
    model.save(model_path)

    pred = model.predict(x_test, verbose=0)
    metrics = regression_metrics(y_test, pred)
    metrics.update(
        {
            "trained": True,
            "n_patches": int(n),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "model_path": str(model_path),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
            "role": (
                "Spatio-temporal CNN-LSTM prototype for representation learning, "
                "reconstruction, and change-analysis support. Not a validated "
                "long-term forecasting model."
            ),
        }
    )
    np.savez_compressed(
        PATHS["models"] / "test_predictions.npz",
        y_true=y_test,
        y_pred=pred,
        x_test=x_test,
    )
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    per_channel = {}
    names = ["ndvi", "lst"]
    for i, name in enumerate(names):
        if y_true.shape[-1] <= i:
            break
        e = y_pred[..., i] - y_true[..., i]
        per_channel[name] = {
            "mae": float(np.mean(np.abs(e))),
            "mse": float(np.mean(e ** 2)),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
        }
    return {"mae": mae, "mse": mse, "rmse": rmse, "per_channel": per_channel}


def mosaic_predictions(
    pred_patches: np.ndarray,
    origins: list[tuple[int, int]],
    height: int,
    width: int,
    patch_size: int = PATCH["size"],
) -> np.ndarray:
    """Average overlapping reconstructed 2024 patches into a full map (C last)."""
    channels = pred_patches.shape[-1]
    acc = np.zeros((height, width, channels), dtype=np.float64)
    wgt = np.zeros((height, width, 1), dtype=np.float64)
    for patch, (y, x) in zip(pred_patches, origins):
        # Patch origins may fall near the raster edge (the stack is padded
        # when the scene is smaller than one patch), so clip the write region
        # to the original raster bounds and slice the patch accordingly.
        y2 = min(y + patch_size, height)
        x2 = min(x + patch_size, width)
        py, px = y2 - y, x2 - x
        if py <= 0 or px <= 0:
            continue
        acc[y:y2, x:x2] += patch[:py, :px]
        wgt[y:y2, x:x2] += 1
    wgt[wgt == 0] = np.nan
    return (acc / wgt).astype(np.float32)
