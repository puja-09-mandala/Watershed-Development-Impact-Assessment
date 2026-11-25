import numpy as np
import tensorflow as tf
from pathlib import Path
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL_PATH = "outputs/models/best_cnn_lstm.keras"
DATA_DIR   = Path("Rajapet/processed")
OUT_DIR    = Path("Rajapet/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]
LULC_RGB    = [[200,200,200],[0,100,255],[0,170,60],[255,50,50],[210,160,50]]

print("\n========================================")
print("STEP 2: CNN-LSTM PREDICTION — RAJAPET")
print("========================================")

ndvi_04 = np.load(DATA_DIR / "ndvi_2004_norm.npy")
ndvi_24 = np.load(DATA_DIR / "ndvi_2024_norm.npy")
lst_04  = np.load(DATA_DIR / "lst_2004_norm.npy")
lst_24  = np.load(DATA_DIR / "lst_2024_norm.npy")
h, w    = ndvi_04.shape
print(f"Image size: {h} x {w}")

# Resize to 128x128 for model input
def r128(arr):
    return resize(arr, (128, 128), anti_aliasing=True).astype(np.float32)

p04 = np.stack([r128(ndvi_04), r128(lst_04)], axis=-1)
p24 = np.stack([r128(ndvi_24), r128(lst_24)], axis=-1)
X   = np.array([[p04, p24]], dtype=np.float32)
print(f"Model input shape: {X.shape}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

print("Running inference...")
y_pred_probs = model.predict(X, batch_size=1, verbose=1)
y_pred_128   = np.argmax(y_pred_probs[0], axis=-1)

# Resize prediction back to original image size
lulc_2024_map = resize(y_pred_128.astype(np.float32), (h, w),
                       order=0, anti_aliasing=False).astype(np.uint8)

np.save(OUT_DIR / "lulc_2024_predicted.npy", lulc_2024_map)

def lulc_pct(arr):
    return {LULC_NAMES[i]: round(np.sum(arr == i) / arr.size * 100, 2) for i in range(5)}

dist = lulc_pct(lulc_2024_map)
print("\nPredicted LULC 2024 Distribution:")
for k, v in dist.items():
    print(f"  {k}: {v}%")

def to_rgb(lulc):
    rgb = np.zeros((*lulc.shape, 3), dtype=np.uint8)
    for i, c in enumerate(LULC_RGB):
        rgb[lulc == i] = c
    return rgb

ndvi_raw = np.load(DATA_DIR / "ndvi_2024_raw.npy")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Rajapet — CNN-LSTM LULC Prediction", fontsize=14, fontweight='bold')
axes[0].imshow(ndvi_raw, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
axes[0].set_title("NDVI 2024")
axes[0].axis('off')
axes[1].imshow(to_rgb(lulc_2024_map))
axes[1].set_title("Predicted LULC 2024")
axes[1].axis('off')
patches = [mpatches.Patch(color=LULC_COLORS[i], label=f"{LULC_NAMES[i]} {dist[LULC_NAMES[i]]}%") for i in range(5)]
axes[1].legend(handles=patches, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "lulc_prediction.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSaved: {OUT_DIR}/lulc_prediction.png")
print("\n========================================")
print("PREDICTION COMPLETE")
print("========================================")
