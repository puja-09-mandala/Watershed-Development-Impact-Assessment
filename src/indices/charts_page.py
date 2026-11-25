import os
import base64
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
matplotlib.use("Agg")
DATA_DIR   = "data/raw"
OUTPUT_DIR = "outputs/maps"
EVAL_DIR   = "outputs/evaluation"
GRAPH_DIR  = "outputs/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("\n========================================")
print("GENERATING CHARTS PAGE")
print("========================================")
def read_raster(path, max_size=800):
    with rasterio.open(path) as src:
        scale = min(1.0, max_size / max(src.height, src.width))
        h = max(1, int(src.height * scale))
        w = max(1, int(src.width  * scale))
        data = src.read(1, out_shape=(h, w),
                        resampling=rasterio.enums.Resampling.bilinear)
        return data.astype(np.float64)

def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def img_file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
print("Reading rasters...")
ndvi_2004 = read_raster(f"{DATA_DIR}/Mahbubnagar_NDVI_2004.tif")
ndvi_2024 = read_raster(f"{DATA_DIR}/Mahbubnagar_NDVI_2024.tif")
lst_2004  = read_raster(f"{DATA_DIR}/Mahbubnagar_LST_2004.tif")
lst_2024  = read_raster(f"{DATA_DIR}/Mahbubnagar_LST_2024.tif")
lulc_2004 = read_raster(f"{DATA_DIR}/Mahbubnagar_LULC_2004.tif")
lulc_2024 = read_raster(f"{DATA_DIR}/Mahbubnagar_LULC_2024.tif")

ndvi_change = ndvi_2024 - ndvi_2004
lst_change  = lst_2024  - lst_2004
lulc_change = (lulc_2004 != lulc_2024)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0050c8", "#228b22", "#dc3232", "#d2b464"]

def lulc_pct(arr):
    idx = np.clip(np.rint(arr).astype(int), 0, 4)
    return [np.sum(idx == i) / idx.size * 100 for i in range(5)]

pct_2004 = lulc_pct(lulc_2004)
pct_2024 = lulc_pct(lulc_2024)

ndvi_mean_2004 = float(np.nanmean(ndvi_2004))
ndvi_mean_2024 = float(np.nanmean(ndvi_2024))
lst_mean_2004  = float(np.nanmean(lst_2004))
lst_mean_2024  = float(np.nanmean(lst_2024))
ndvi_delta     = ndvi_mean_2024 - ndvi_mean_2004
lst_delta      = lst_mean_2024  - lst_mean_2004
lulc_changed   = float(np.sum(lulc_change) / lulc_change.size * 100)

charts = {}  # name -> base64 png
print("Chart 1: NDVI Bar...")
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
bars = ax.bar(["2004", "2024"], [ndvi_mean_2004, ndvi_mean_2024],
              color=["#d73027", "#1a9850"], width=0.45, edgecolor="black")
ax.set_title("NDVI Mean Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean NDVI")
ax.set_ylim(0, max(ndvi_mean_2004, ndvi_mean_2024) * 1.35)
for b, v in zip(bars, [ndvi_mean_2004, ndvi_mean_2024]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
            f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
dc = "#1a9850" if ndvi_delta >= 0 else "#d73027"
ax.text(0.5, 0.93, f"Change: {ndvi_delta:+.4f}",
        transform=ax.transAxes, ha="center", fontsize=11, color=dc, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
charts["ndvi_bar"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 2 — LST Bar
# ============================================================
print("Chart 2: LST Bar...")
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
bars = ax.bar(["2004", "2024"], [lst_mean_2004, lst_mean_2024],
              color=["#4575b4", "#d73027"], width=0.45, edgecolor="black")
ax.set_title("LST Mean Comparison", fontsize=13, fontweight="bold")
ax.set_ylabel("Mean LST (C)")
ax.set_ylim(min(lst_mean_2004, lst_mean_2024) * 0.93,
            max(lst_mean_2004, lst_mean_2024) * 1.07)
for b, v in zip(bars, [lst_mean_2004, lst_mean_2024]):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
            f"{v:.2f}C", ha="center", fontsize=12, fontweight="bold")
dc2 = "#d73027" if lst_delta >= 0 else "#4575b4"
ax.text(0.5, 0.93, f"Change: {lst_delta:+.2f}C",
        transform=ax.transAxes, ha="center", fontsize=11, color=dc2, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
charts["lst_bar"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 3 — LULC Grouped Bar
# ============================================================
print("Chart 3: LULC Grouped Bar...")
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
x = np.arange(len(LULC_NAMES))
w = 0.35
ax.bar(x - w/2, pct_2004, w, color=LULC_COLORS, edgecolor="black", linewidth=0.7)
ax.bar(x + w/2, pct_2024, w, color=LULC_COLORS, edgecolor="black", linewidth=0.7,
       alpha=0.5, hatch="//")
ax.set_title("LULC Class Distribution (%) — 2004 vs 2024", fontsize=13, fontweight="bold")
ax.set_ylabel("Area (%)")
ax.set_xticks(x)
ax.set_xticklabels(LULC_NAMES, fontsize=10)
patches_leg = [mpatches.Patch(facecolor=LULC_COLORS[i], edgecolor="black",
               label=LULC_NAMES[i]) for i in range(5)]
s = mpatches.Patch(facecolor="white", edgecolor="black", label="Solid = 2004")
h = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Hatch = 2024")
ax.legend(handles=patches_leg + [s, h], fontsize=8)
ax.grid(axis="y", alpha=0.3)
charts["lulc_bar"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 4 — LULC Pie 2004
# ============================================================
print("Chart 4: LULC Pie 2004...")
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#f8f9fa")
pv = [p for p in pct_2004 if p > 0]
pl = [LULC_NAMES[i] for i, p in enumerate(pct_2004) if p > 0]
pc = [LULC_COLORS[i] for i, p in enumerate(pct_2004) if p > 0]
_, _, autotexts = ax.pie(pv, labels=pl, colors=pc,
    autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
    startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
for t in autotexts: t.set_fontsize(9)
ax.set_title("LULC Distribution — 2004", fontsize=13, fontweight="bold")
charts["lulc_pie_2004"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 5 — LULC Pie 2024
# ============================================================
print("Chart 5: LULC Pie 2024...")
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#f8f9fa")
pv = [p for p in pct_2024 if p > 0]
pl = [LULC_NAMES[i] for i, p in enumerate(pct_2024) if p > 0]
pc = [LULC_COLORS[i] for i, p in enumerate(pct_2024) if p > 0]
_, _, autotexts = ax.pie(pv, labels=pl, colors=pc,
    autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
    startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
for t in autotexts: t.set_fontsize(9)
ax.set_title("LULC Distribution — 2024", fontsize=13, fontweight="bold")
charts["lulc_pie_2024"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 6 — NDVI Change Histogram
# ============================================================
print("Chart 6: NDVI Change Histogram...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
flat = ndvi_change[np.isfinite(ndvi_change)].flatten()
ax.hist(flat, bins=80, color="#1a9850", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(0, color="black", linewidth=2, linestyle="--", label="No Change")
ax.axvline(float(np.nanmean(flat)), color="#d73027", linewidth=2,
           label=f"Mean = {np.nanmean(flat):.4f}")
ax.set_title("NDVI Change Distribution (2024 - 2004)", fontsize=13, fontweight="bold")
ax.set_xlabel("NDVI Change")
ax.set_ylabel("Pixel Count")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
charts["ndvi_hist"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 7 — LST Change Histogram
# ============================================================
print("Chart 7: LST Change Histogram...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
flat = lst_change[np.isfinite(lst_change)].flatten()
ax.hist(flat, bins=80, color="#d73027", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(0, color="black", linewidth=2, linestyle="--", label="No Change")
ax.axvline(float(np.nanmean(flat)), color="#4575b4", linewidth=2,
           label=f"Mean = {np.nanmean(flat):.2f}C")
ax.set_title("LST Change Distribution (2024 - 2004)", fontsize=13, fontweight="bold")
ax.set_xlabel("LST Change (C)")
ax.set_ylabel("Pixel Count")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
charts["lst_hist"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 8 — Model Accuracy Bar
# ============================================================
print("Chart 8: Model Accuracy...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "IoU"]
values  = [84.62, 87.30, 84.62, 84.64, 73.53]
colors  = ["#1a9850", "#2196F3", "#FF9800", "#9C27B0", "#e67e00"]
bars = ax.barh(metrics, values, color=colors, edgecolor="black", linewidth=0.7)
ax.set_xlim(0, 100)
ax.set_title("CNN-LSTM Model Performance (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Score (%)")
for b, v in zip(bars, values):
    ax.text(v + 0.5, b.get_y() + b.get_height()/2,
            f"{v:.2f}%", va="center", fontsize=11, fontweight="bold")
ax.axvline(80, color="gray", linewidth=1, linestyle="--", alpha=0.5)
ax.grid(axis="x", alpha=0.3)
charts["model_accuracy"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 9 — Per Class F1
# ============================================================
print("Chart 9: Per Class F1...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#ffffff")
f1_vals = [0.84, 0.77, 0.87, 0.11, 0.56]
bars = ax.bar(LULC_NAMES, f1_vals, color=LULC_COLORS, edgecolor="black", linewidth=0.7)
ax.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.15)
for b, v in zip(bars, f1_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(0.84, color="gray", linewidth=1.5, linestyle="--", alpha=0.6, label="Overall F1=0.84")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
charts["per_class_f1"] = fig_to_b64(fig); plt.close()

# ============================================================
# CHART 10 — Training Accuracy & Loss (from saved pngs)
# ============================================================
training_charts = {}
for key, fname in [("train_acc", "cnn_lstm_accuracy.png"),
                   ("train_loss", "cnn_lstm_loss.png")]:
    path = f"{GRAPH_DIR}/{fname}"
    if os.path.exists(path):
        training_charts[key] = img_file_to_b64(path)

# Confusion matrix
conf_path = f"{EVAL_DIR}/confusion_matrix.png"
if os.path.exists(conf_path):
    training_charts["confusion"] = img_file_to_b64(conf_path)

actual_vs_pred_path = f"{EVAL_DIR}/actual_vs_predicted.png"
if os.path.exists(actual_vs_pred_path):
    training_charts["actual_vs_pred"] = img_file_to_b64(actual_vs_pred_path)

print("All charts generated.")

# ============================================================
# BUILD HTML
# ============================================================

def card(title, b64, explanation=""):
    exp_html = f'<p style="font-size:13px;color:#555;margin:6px 0 0 0;">{explanation}</p>' if explanation else ""
    return f"""
    <div style="background:white;border-radius:10px;padding:18px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);margin-bottom:24px;">
      <h3 style="margin:0 0 10px 0;color:#1a3a5c;font-size:15px;">{title}</h3>
      {exp_html}
      <img src="data:image/png;base64,{b64}"
           style="width:100%;border-radius:6px;border:1px solid #eee;margin-top:8px;">
    </div>"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Mahbubnagar Watershed — Charts & Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #f0f4f8; }}
  .header {{
    background: linear-gradient(135deg, #1a3a5c, #2c7bb6);
    color: white; padding: 24px 40px; text-align: center;
  }}
  .header h1 {{ font-size: 24px; margin-bottom: 6px; }}
  .header p  {{ font-size: 14px; opacity: 0.85; }}
  .stats-bar {{
    display: flex; justify-content: center; gap: 30px;
    background: #1a3a5c; padding: 14px 40px; flex-wrap: wrap;
  }}
  .stat {{
    text-align: center; color: white;
  }}
  .stat .val {{ font-size: 22px; font-weight: bold; color: #7ecef4; }}
  .stat .lbl {{ font-size: 11px; opacity: 0.8; }}
  .section-title {{
    font-size: 18px; font-weight: bold; color: #1a3a5c;
    border-left: 5px solid #2c7bb6; padding-left: 12px;
    margin: 30px 0 16px 0;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 30px 40px; }}
  .full {{ grid-column: 1 / -1; }}
</style>
</head>
<body>

<div class="header">
  <h1>Mahbubnagar District — Watershed Change Analysis</h1>
  <p>CNN-LSTM Spatio-Temporal Analysis &nbsp;|&nbsp; 2004 vs 2024 &nbsp;|&nbsp; NDVI + LST + LULC</p>
</div>

<div class="stats-bar">
  <div class="stat"><div class="val">84.62%</div><div class="lbl">Model Accuracy</div></div>
  <div class="stat"><div class="val">+{ndvi_delta:.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">+{lst_delta:.2f}C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{lulc_changed:.1f}%</div><div class="lbl">LULC Changed Area</div></div>
  <div class="stat"><div class="val">73.53%</div><div class="lbl">IoU Score</div></div>
  <div class="stat"><div class="val">1152</div><div class="lbl">Total Patches</div></div>
</div>

<div class="container">

  <!-- NDVI + LST -->
  <div class="section-title">NDVI & LST Analysis</div>
  <div class="grid-2">
    {card("NDVI Mean — 2004 vs 2024",
          charts["ndvi_bar"],
          f"Mean NDVI increased from {ndvi_mean_2004:.4f} to {ndvi_mean_2024:.4f} (+{ndvi_delta:.4f}) — slight vegetation improvement.")}
    {card("LST Mean — 2004 vs 2024",
          charts["lst_bar"],
          f"Mean LST increased from {lst_mean_2004:.2f}C to {lst_mean_2024:.2f}C (+{lst_delta:.2f}C) — mild warming trend.")}
  </div>

  <!-- Change Distributions -->
  <div class="section-title">Change Distribution</div>
  <div class="grid-2">
    {card("NDVI Change Distribution (2024 - 2004)",
          charts["ndvi_hist"],
          "Positive values = vegetation gain. Negative values = vegetation loss. Distribution centered slightly positive.")}
    {card("LST Change Distribution (2024 - 2004)",
          charts["lst_hist"],
          "Positive values = temperature increase. Most pixels show slight warming over 20 years.")}
  </div>

  <!-- LULC -->
  <div class="section-title">LULC Analysis</div>
  <div class="grid-3">
    {card("LULC Class Distribution — 2004 vs 2024",
          charts["lulc_bar"],
          f"Solid bars = 2004, Hatched bars = 2024. {lulc_changed:.1f}% of total area changed class.")}
    {card("LULC Pie — 2004", charts["lulc_pie_2004"],
          "Land use distribution in 2004. Background dominates at 86.5%.")}
    {card("LULC Pie — 2024", charts["lulc_pie_2024"],
          "Land use distribution in 2024. Compare with 2004 to see shifts.")}
  </div>

  <!-- Model Performance -->
  <div class="section-title">CNN-LSTM Model Performance</div>
  <div class="grid-2">
    {card("Overall Model Metrics",
          charts["model_accuracy"],
          "Overall accuracy 84.62%. IoU 73.53% — good for semantic segmentation task.")}
    {card("Per-Class F1 Score",
          charts["per_class_f1"],
          "Vegetation best (0.87). Built-up worst (0.11) due to class imbalance — only 0.4% pixels.")}
  </div>
"""

# Training charts if available
if training_charts:
    html += '<div class="section-title">Training History & Evaluation</div>\n'
    html += '<div class="grid-2">\n'
    if "train_acc" in training_charts:
        html += card("Training Accuracy Curve", training_charts["train_acc"],
                     "Model accuracy over epochs. Gap between train/val shows overfitting level.")
    if "train_loss" in training_charts:
        html += card("Training Loss Curve", training_charts["train_loss"],
                     "Loss decreasing over epochs. Early stopping prevented overfitting.")
    if "confusion" in training_charts:
        html += card("Confusion Matrix", training_charts["confusion"],
                     "Diagonal = correct predictions. Off-diagonal = misclassifications.")
    if "actual_vs_pred" in training_charts:
        html += card("Actual vs Predicted LULC", training_charts["actual_vs_pred"],
                     "Visual comparison of ground truth vs model prediction on test sample.")
    html += "</div>\n"

html += """
</div>
</body>
</html>"""

out_path = f"{OUTPUT_DIR}/Charts_Analysis.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nSaved: {out_path}")

import subprocess
subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(out_path)])

print("\n========================================")
print("CHARTS PAGE OPENED IN BROWSER")
print("========================================")
print("File:", os.path.abspath(out_path))
