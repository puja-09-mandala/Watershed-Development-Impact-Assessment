import os
import io
import base64
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from skimage.transform import resize

RAW_DIR  = Path("Rajapet/GeoTiff")
DATA_DIR = Path("Rajapet/processed")
OUT_DIR  = Path("Rajapet/outputs")
MAP_DIR  = Path("Rajapet/outputs/maps")
for d in [OUT_DIR, MAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]
LULC_RGB    = [[200,200,200],[0,100,255],[0,170,60],[255,50,50],[210,160,50]]

print("\n========================================")
print("READING DATA")
print("========================================")

def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data

def normalize(data):
    d = np.nan_to_num(data, nan=0.0)
    mn, mx = d.min(), d.max()
    return (d - mn) / (mx - mn + 1e-8)

ndvi_2004_raw = read_tif(RAW_DIR / "Rajapeta_NDVI_2004.tif")
ndvi_2024_raw = read_tif(RAW_DIR / "Rajapeta_NDVI_2024.tif")
lst_2004_raw  = read_tif(RAW_DIR / "Rajapeta_LST_2004.tif")
lst_2024_raw  = read_tif(RAW_DIR / "Rajapeta_LST_2024.tif")
lulc_2024     = np.load(OUT_DIR / "lulc_2024_predicted.npy").astype(np.uint8)

# Predict LULC 2004 using same model
print("Predicting LULC 2004...")
model = tf.keras.models.load_model("outputs/models/best_cnn_lstm.keras")

def predict_lulc(ndvi_raw, lst_raw):
    h, w = ndvi_raw.shape
    def r128(arr):
        return resize(normalize(arr), (128,128), anti_aliasing=True).astype(np.float32)
    p = np.stack([r128(ndvi_raw), r128(lst_raw)], axis=-1)
    # For single year prediction, stack same year twice as temporal input
    X = np.array([[p, p]], dtype=np.float32)
    pred = model.predict(X, batch_size=1, verbose=0)
    pred_128 = np.argmax(pred[0], axis=-1)
    return resize(pred_128.astype(np.float32), (h, w), order=0, anti_aliasing=False).astype(np.uint8)

lulc_2004 = predict_lulc(ndvi_2004_raw, lst_2004_raw)
np.save(OUT_DIR / "lulc_2004_predicted.npy", lulc_2004)

def lulc_pct(arr):
    return {LULC_NAMES[i]: round(np.sum(arr==i)/arr.size*100, 2) for i in range(5)}

dist_04 = lulc_pct(lulc_2004)
dist_24 = lulc_pct(lulc_2024)

ndvi_mean_04 = round(float(np.nanmean(ndvi_2004_raw)), 4)
ndvi_mean_24 = round(float(np.nanmean(ndvi_2024_raw)), 4)
ndvi_delta   = round(ndvi_mean_24 - ndvi_mean_04, 4)
lst_mean_04  = round(float(np.nanmean(lst_2004_raw)), 2)
lst_mean_24  = round(float(np.nanmean(lst_2024_raw)), 2)
lst_delta    = round(lst_mean_24 - lst_mean_04, 2)

print(f"NDVI: {ndvi_mean_04} -> {ndvi_mean_24} ({ndvi_delta:+.4f})")
print(f"LST : {lst_mean_04}C -> {lst_mean_24}C ({lst_delta:+.2f}C)")
print(f"LULC 2004: {dist_04}")
print(f"LULC 2024: {dist_24}")

# ============================================================
# IMAGE GENERATION
# ============================================================

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def to_rgb_lulc(lulc):
    rgb = np.zeros((*lulc.shape, 3), dtype=np.uint8)
    for i, c in enumerate(LULC_RGB):
        rgb[lulc == i] = c
    return rgb

vmin_lst = min(np.nanmin(lst_2004_raw), np.nanmin(lst_2024_raw))
vmax_lst = max(np.nanmax(lst_2004_raw), np.nanmax(lst_2024_raw))

def ndvi_img(data, title):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(data, cmap='RdYlGn', vmin=-0.3, vmax=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04)
    return to_b64(fig)

def lst_img(data, title):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(data, cmap='turbo', vmin=vmin_lst, vmax=vmax_lst)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04, label="°C")
    return to_b64(fig)

def lulc_img(lulc, title, dist):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(to_rgb_lulc(lulc))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    patches = [mpatches.Patch(color=LULC_COLORS[i], label=f"{LULC_NAMES[i]} {dist[LULC_NAMES[i]]}%") for i in range(5)]
    ax.legend(handles=patches, loc='lower right', fontsize=7, framealpha=0.9)
    return to_b64(fig)

def change_img(change, title, cmap, unit=""):
    fig, ax = plt.subplots(figsize=(5,4))
    lim = max(np.nanpercentile(np.abs(change), 98), 0.01)
    im = ax.imshow(change, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04, label=unit)
    return to_b64(fig)

print("Generating images...")
n04  = ndvi_img(ndvi_2004_raw, f"NDVI 2004\nMean: {ndvi_mean_04}")
n24  = ndvi_img(ndvi_2024_raw, f"NDVI 2024\nMean: {ndvi_mean_24}")
nchg = change_img(ndvi_2024_raw - ndvi_2004_raw, f"NDVI Change\n({ndvi_delta:+.4f})", "RdYlGn")
l04  = lst_img(lst_2004_raw,  f"LST 2004\nMean: {lst_mean_04}°C")
l24  = lst_img(lst_2024_raw,  f"LST 2024\nMean: {lst_mean_24}°C")
lchg = change_img(lst_2024_raw - lst_2004_raw, f"LST Change\n({lst_delta:+.2f}°C)", "RdBu_r", "°C")
u04  = lulc_img(lulc_2004, "LULC 2004\n(Experimental Prediction)", dist_04)
u24  = lulc_img(lulc_2024, "LULC 2024\n(Experimental Prediction)", dist_24)

# LULC change map
lulc_change_arr = (lulc_2004 != lulc_2024).astype(np.uint8)
changed_pct = round(float(np.sum(lulc_change_arr) / lulc_change_arr.size * 100), 2)
fig, ax = plt.subplots(figsize=(5,4))
ax.imshow(lulc_change_arr, cmap='RdYlGn_r', vmin=0, vmax=1)
ax.set_title(f"LULC Change\n{changed_pct}% area changed", fontsize=11, fontweight='bold')
ax.axis('off')
uchg = to_b64(fig)

print("Done.")

# ============================================================
# HTML
# ============================================================

def img(b64):
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);">'

nc = "#1a9850" if ndvi_delta > 0 else "#d73027"
lc = "#d73027" if lst_delta  > 0 else "#1a9850"

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Rajapet — Change Analysis Maps</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: Arial, sans-serif; background:#f0f4f8; color:#222; }}
  .header {{ background:linear-gradient(135deg,#1a3a5c,#2c7bb6); color:white; padding:24px 40px; text-align:center; }}
  .header h1 {{ font-size:22px; margin-bottom:6px; }}
  .header p  {{ font-size:13px; opacity:0.85; }}
  .stats-bar {{ display:flex; justify-content:center; gap:30px; background:#1a3a5c; padding:14px; flex-wrap:wrap; }}
  .stat {{ text-align:center; color:white; }}
  .stat .val {{ font-size:20px; font-weight:bold; color:#7ecef4; }}
  .stat .lbl {{ font-size:11px; opacity:0.8; }}
  .section {{ max-width:1200px; margin:28px auto; padding:0 24px; }}
  .sec-title {{ font-size:17px; font-weight:bold; color:#1a3a5c; margin-bottom:14px;
                border-left:5px solid #2c7bb6; padding-left:12px; background:#e8f4fd;
                padding:10px 12px; border-radius:0 6px 6px 0; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; margin-bottom:10px; }}
  .card {{ background:white; border-radius:10px; padding:10px; box-shadow:0 2px 10px rgba(0,0,0,0.08); }}
  .card-title {{ font-size:12px; font-weight:bold; color:#555; margin-bottom:6px; text-align:center; }}
  .label-2004 {{ color:#2c7bb6; }}
  .label-2024 {{ color:#d73027; }}
  .label-chg  {{ color:#555; }}
  footer {{ text-align:center; padding:20px; font-size:12px; color:#888; }}
</style>
</head>
<body>

<div class="header">
  <h1>Rajapet — Environmental Change Analysis Maps</h1>
  <p>CNN-LSTM Deep Learning + Remote Sensing &nbsp;|&nbsp; 2004 vs 2024 &nbsp;|&nbsp; Telangana, India</p>
</div>

<div class="stats-bar">
  <div class="stat"><div class="val">{ndvi_mean_04}</div><div class="lbl">NDVI 2004</div></div>
  <div class="stat"><div class="val">{ndvi_mean_24}</div><div class="lbl">NDVI 2024</div></div>
  <div class="stat"><div class="val" style="color:{nc};">{ndvi_delta:+.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">{lst_mean_04}°C</div><div class="lbl">LST 2004</div></div>
  <div class="stat"><div class="val">{lst_mean_24}°C</div><div class="lbl">LST 2024</div></div>
  <div class="stat"><div class="val" style="color:{lc};">{lst_delta:+.2f}°C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{changed_pct}%</div><div class="lbl">LULC Changed</div></div>
  <div class="stat"><div class="val">Not validated</div><div class="lbl">Model Accuracy</div></div>
</div>

<div class="section">
  <div class="sec-title">🌿 NDVI — Normalized Difference Vegetation Index</div>
  <div class="grid3">
    <div class="card"><div class="card-title label-2004">NDVI 2004</div>{img(n04)}</div>
    <div class="card"><div class="card-title label-2024">NDVI 2024</div>{img(n24)}</div>
    <div class="card"><div class="card-title label-chg">NDVI Change (2004 → 2024)</div>{img(nchg)}</div>
  </div>
</div>

<div class="section">
  <div class="sec-title">🌡 LST — Land Surface Temperature</div>
  <div class="grid3">
    <div class="card"><div class="card-title label-2004">LST 2004</div>{img(l04)}</div>
    <div class="card"><div class="card-title label-2024">LST 2024</div>{img(l24)}</div>
    <div class="card"><div class="card-title label-chg">LST Change (2004 → 2024)</div>{img(lchg)}</div>
  </div>
</div>

<div class="section">
  <div class="sec-title">🏔 LULC — Land Use Land Cover (Experimental Prediction)</div>
  <div class="grid3">
    <div class="card"><div class="card-title label-2004">LULC 2004 (Predicted)</div>{img(u04)}</div>
    <div class="card"><div class="card-title label-2024">LULC 2024 (Predicted)</div>{img(u24)}</div>
    <div class="card"><div class="card-title label-chg">LULC Change Map</div>{img(uchg)}</div>
  </div>
</div>

<footer>CNN-LSTM Model: Accuracy Not validated | F1 Not validated | IoU Not validated &nbsp;|&nbsp; Data: Google Earth Engine (Landsat)</footer>
</body>
</html>"""

out = MAP_DIR / "Rajapet_Comparison.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

import subprocess
subprocess.Popen(["cmd", "/c", "start", "", str(out.resolve())])
print(f"\nSaved: {out.resolve()}")
print("========================================")
print("COMPARISON MAPS COMPLETE")
print("========================================")
