import os
import io
import base64
import time
import requests
import numpy as np
import rasterio
import folium
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from folium import LayerControl
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

RAW_DIR  = Path("Rajapet/GeoTiff")
DATA_DIR = Path("Rajapet/processed")
OUT_DIR  = Path("Rajapet/outputs")
LLM_DIR  = Path("Rajapet/outputs/llm")
MAP_DIR  = Path("Rajapet/outputs/maps")
for d in [OUT_DIR, LLM_DIR, MAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]
LULC_RGB    = [[200,200,200],[0,100,255],[0,170,60],[255,50,50],[210,160,50]]

# ============================================================
# STEP 1: READ RAW TIF FILES
# ============================================================
print("\n========================================")
print("STEP 1: READING RAW TIF FILES")
print("========================================")

def read_tif_with_bounds(path, max_size=1200):
    with rasterio.open(path) as src:
        scale = min(1.0, max_size / max(src.height, src.width))
        h = max(1, int(src.height * scale))
        w = max(1, int(src.width  * scale))
        data = src.read(1, out_shape=(h, w),
                        resampling=rasterio.enums.Resampling.bilinear)
        if src.nodata is not None:
            data = data.astype(float)
            data[data == src.nodata] = np.nan
        bounds = src.bounds
        center = [(bounds.bottom + bounds.top) / 2,
                  (bounds.left  + bounds.right) / 2]
    return data.astype(float), bounds, center

ndvi_2004, ndvi_bounds, center = read_tif_with_bounds(RAW_DIR / "Rajapeta_NDVI_2004.tif")
ndvi_2024, _, _                = read_tif_with_bounds(RAW_DIR / "Rajapeta_NDVI_2024.tif")
lst_2004,  lst_bounds,  _      = read_tif_with_bounds(RAW_DIR / "Rajapeta_LST_2004.tif")
lst_2024,  _, _                = read_tif_with_bounds(RAW_DIR / "Rajapeta_LST_2024.tif")
lulc_2024 = np.load(OUT_DIR / "lulc_2024_predicted.npy").astype(float)

print(f"Center: {center}")
print(f"NDVI shape: {ndvi_2004.shape}, LST shape: {lst_2004.shape}, LULC shape: {lulc_2024.shape}")

# ============================================================
# STEP 2: COMPUTE STATISTICS
# ============================================================
print("\n========================================")
print("STEP 2: COMPUTING STATISTICS")
print("========================================")

ndvi_mean_04  = round(float(np.nanmean(ndvi_2004)), 4)
ndvi_mean_24  = round(float(np.nanmean(ndvi_2024)), 4)
ndvi_delta    = round(ndvi_mean_24 - ndvi_mean_04, 4)
ndvi_max_gain = round(float(np.nanmax(ndvi_2024 - ndvi_2004)), 4)
ndvi_max_loss = round(float(np.nanmin(ndvi_2024 - ndvi_2004)), 4)

lst_mean_04   = round(float(np.nanmean(lst_2004)), 2)
lst_mean_24   = round(float(np.nanmean(lst_2024)), 2)
lst_delta     = round(lst_mean_24 - lst_mean_04, 2)
lst_max       = round(float(np.nanmax(lst_2024)), 2)
lst_min       = round(float(np.nanmin(lst_2024)), 2)

def lulc_pct(arr):
    a = np.clip(np.rint(arr).astype(int), 0, 4)
    return {LULC_NAMES[i]: round(np.sum(a == i) / a.size * 100, 2) for i in range(5)}

dist_24 = lulc_pct(lulc_2024)

print(f"NDVI: {ndvi_mean_04} -> {ndvi_mean_24} ({ndvi_delta:+.4f})")
print(f"LST : {lst_mean_04}C -> {lst_mean_24}C ({lst_delta:+.2f}C)")
print(f"LULC: {dist_24}")

# ============================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================

def to_url(data, layer_type):
    data = np.nan_to_num(data.astype(float), nan=0.0)
    if layer_type == "ndvi":
        norm = np.clip((data + 1) / 2.0, 0, 1)
        cmap = matplotlib.colormaps.get_cmap("RdYlGn")
        rgba = (cmap(norm) * 255).astype(np.uint8)
    elif layer_type == "lst":
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)
        norm = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
        cmap = matplotlib.colormaps.get_cmap("turbo")
        rgba = (cmap(norm) * 255).astype(np.uint8)
    elif layer_type == "lulc":
        colors = np.array([[200,200,200,255],[0,100,255,255],[0,170,60,255],[255,50,50,255],[210,160,50,255]])
        idx = np.clip(np.rint(data).astype(int), 0, 4)
        rgba = colors[idx]
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def change_url(change, cmap_name):
    change = np.nan_to_num(change, nan=0.0)
    limit  = max(np.nanpercentile(np.abs(change), 98), 0.001)
    norm   = np.clip((change + limit) / (2 * limit), 0, 1)
    cmap   = matplotlib.colormaps.get_cmap(cmap_name)
    rgba   = (cmap(norm) * 255).astype(np.uint8)
    rgba[:,:,3] = 200
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def bb(bounds):
    return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]

def base_map(title):
    m = folium.Map(location=center, zoom_start=12, control_scale=True)
    folium.plugins.Fullscreen().add_to(m)
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
    background:white;padding:10px 20px;border:2px solid #333;border-radius:6px;
    font-size:16px;font-weight:bold;text-align:center;">{title}</div>"""))
    return m

def add_legend(m, items):
    html = '<div style="position:fixed;bottom:30px;left:20px;z-index:9999;background:white;padding:12px;border:2px solid #333;border-radius:6px;font-size:13px;"><b>Legend</b><br><br>'
    for color, label in items:
        html += f'<span style="color:{color};font-size:20px;">■</span> {label}<br>'
    html += '</div>'
    m.get_root().html.add_child(folium.Element(html))

# ============================================================
# STEP 4: NDVI MAP
# ============================================================
print("\n========================================")
print("STEP 3: CREATING MAPS")
print("========================================")

try:
    import folium.plugins
except:
    pass

ndvi_change = ndvi_2024 - ndvi_2004

m = base_map("Rajapet — NDVI Map (2004 vs 2024)")
folium.raster_layers.ImageOverlay(image=to_url(ndvi_2004, "ndvi"), bounds=bb(ndvi_bounds), name="NDVI 2004", opacity=0.75, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(ndvi_2024, "ndvi"), bounds=bb(ndvi_bounds), name="NDVI 2024", opacity=0.75, show=True).add_to(m)
folium.raster_layers.ImageOverlay(image=change_url(ndvi_change, "RdYlGn"), bounds=bb(ndvi_bounds), name="NDVI Change", opacity=0.80, show=False).add_to(m)
add_legend(m, [("#ff0000","Low NDVI"),("#ffff00","Medium NDVI"),("#00aa00","High NDVI"),("#ff00ff","Changed Area")])
LayerControl(collapsed=False).add_to(m)
m.save(str(MAP_DIR / "NDVI_Map.html"))
print("NDVI_Map.html saved")

# ============================================================
# STEP 5: LST MAP
# ============================================================
lst_change = lst_2024 - lst_2004

m = base_map("Rajapet — LST Map (2004 vs 2024)")
folium.raster_layers.ImageOverlay(image=to_url(lst_2004, "lst"), bounds=bb(lst_bounds), name="LST 2004", opacity=0.75, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(lst_2024, "lst"), bounds=bb(lst_bounds), name="LST 2024", opacity=0.75, show=True).add_to(m)
folium.raster_layers.ImageOverlay(image=change_url(lst_change, "RdBu_r"), bounds=bb(lst_bounds), name="LST Change", opacity=0.80, show=False).add_to(m)
add_legend(m, [("#0000ff","Cool"),("#ffff00","Warm"),("#ff0000","Hot")])
LayerControl(collapsed=False).add_to(m)
m.save(str(MAP_DIR / "LST_Map.html"))
print("LST_Map.html saved")

# ============================================================
# STEP 6: LULC MAP
# ============================================================
lulc_bounds_obj = ndvi_bounds  # use same bounds as NDVI

m = base_map("Rajapet — LULC Map (CNN-LSTM Predicted 2024)")
folium.raster_layers.ImageOverlay(image=to_url(lulc_2024, "lulc"), bounds=bb(lulc_bounds_obj), name="LULC 2024 (Predicted)", opacity=0.85, show=True).add_to(m)
add_legend(m, list(zip(LULC_COLORS, [f"{n} {dist_24[n]}%" for n in LULC_NAMES])))
LayerControl(collapsed=False).add_to(m)
m.save(str(MAP_DIR / "LULC_Map.html"))
print("LULC_Map.html saved")

# ============================================================
# STEP 7: COMBINED MAP
# ============================================================
m = base_map("Rajapet — Combined Change Analysis (2004 vs 2024)")
folium.raster_layers.ImageOverlay(image=to_url(ndvi_2024, "ndvi"),  bounds=bb(ndvi_bounds),      name="NDVI 2024",          opacity=0.75, show=True).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(ndvi_2004, "ndvi"),  bounds=bb(ndvi_bounds),      name="NDVI 2004",          opacity=0.75, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=change_url(ndvi_change,"RdYlGn"), bounds=bb(ndvi_bounds),name="NDVI Change",         opacity=0.80, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(lst_2024, "lst"),    bounds=bb(lst_bounds),       name="LST 2024",           opacity=0.75, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(lst_2004, "lst"),    bounds=bb(lst_bounds),       name="LST 2004",           opacity=0.75, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=change_url(lst_change,"RdBu_r"), bounds=bb(lst_bounds),  name="LST Change",         opacity=0.80, show=False).add_to(m)
folium.raster_layers.ImageOverlay(image=to_url(lulc_2024, "lulc"), bounds=bb(lulc_bounds_obj),   name="LULC 2024 Predicted",opacity=0.85, show=False).add_to(m)
add_legend(m, list(zip(LULC_COLORS, LULC_NAMES)) + [("#ff0000","Hot LST"),("#00aa00","High NDVI")])
LayerControl(collapsed=False).add_to(m)
m.save(str(MAP_DIR / "Combined_Map.html"))
print("Combined_Map.html saved")

# ============================================================
# STEP 8: CHANGE DETECTION IMAGES
# ============================================================
print("\n========================================")
print("STEP 4: SAVING CHANGE IMAGES")
print("========================================")

def save_map_img(data, title, cmap, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

save_map_img(ndvi_change, "NDVI Change (2004→2024)", "RdYlGn", OUT_DIR / "ndvi_change.png")
save_map_img(lst_change,  "LST Change (2004→2024)",  "RdBu_r", OUT_DIR / "lst_change.png")

def to_rgb(lulc):
    rgb = np.zeros((*lulc.shape, 3), dtype=np.uint8)
    for i, c in enumerate(LULC_RGB):
        rgb[lulc.astype(int) == i] = c
    return rgb

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(to_rgb(lulc_2024))
ax.set_title("Predicted LULC 2024 — Rajapet", fontsize=13, fontweight='bold')
ax.axis('off')
patches = [mpatches.Patch(color=LULC_COLORS[i], label=f"{LULC_NAMES[i]} {dist_24[LULC_NAMES[i]]}%") for i in range(5)]
ax.legend(handles=patches, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "lulc_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("Change images saved.")

# ============================================================
# STEP 9: MISTRAL LLM REPORT
# ============================================================
print("\n========================================")
print("STEP 5: CALLING MISTRAL LLM")
print("========================================")

if not api_key or api_key == "YOUR_API_KEY_HERE":
    raise ValueError("MISTRAL_API_KEY not set in .env file")

prompt = f"""You are an expert environmental scientist. Analyze this satellite remote sensing data for Rajapet, Andhra Pradesh, India (2004-2024) and write a detailed environmental change analysis report.

DATA:
- NDVI 2004: {ndvi_mean_04}, NDVI 2024: {ndvi_mean_24}, Change: {ndvi_delta:+.4f}, Max Gain: {ndvi_max_gain}, Max Loss: {ndvi_max_loss}
- LST 2004: {lst_mean_04}C, LST 2024: {lst_mean_24}C, Change: {lst_delta:+.2f}C, Range: {lst_min}C to {lst_max}C
- LULC 2024 (CNN-LSTM predicted): Background {dist_24['Background']}%, Water {dist_24['Water']}%, Vegetation {dist_24['Vegetation']}%, Built-up {dist_24['Built-up']}%, Barren {dist_24['Barren']}%
- CNN-LSTM Model: Accuracy 84.62%, F1 84.64%, IoU 73.53%

Write report with sections:
1. EXECUTIVE SUMMARY
2. NDVI ANALYSIS
3. LST ANALYSIS
4. LULC ANALYSIS
5. ENVIRONMENTAL CONCERNS
6. RECOMMENDATIONS
7. CONCLUSION"""

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {"model": "open-mistral-7b", "messages": [{"role": "user", "content": prompt}]}

llm_output = None
for attempt in range(5):
    try:
        resp = requests.post("https://api.mistral.ai/v1/chat/completions",
                             headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            llm_output = resp.json()["choices"][0]["message"]["content"]
            break
        elif resp.status_code == 429:
            wait = (attempt + 1) * 20
            print(f"Rate limited. Waiting {wait}s... ({attempt+1}/5)")
            time.sleep(wait)
        else:
            print(f"API Error {resp.status_code}: {resp.text}")
            break
    except Exception as e:
        print(f"Error: {e}")
        break

if not llm_output:
    raise RuntimeError("Mistral API failed.")

print(llm_output)

# ============================================================
# STEP 10: SAVE HTML REPORT
# ============================================================
print("\n========================================")
print("STEP 6: SAVING HTML REPORT")
print("========================================")

def img_b64(path):
    if os.path.exists(str(path)):
        with open(str(path), "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;margin-top:8px;">'
    return '<p style="color:#aaa;font-size:12px;">Image not found</p>'

def lulc_bar(label, pct, color):
    w = min(pct * 5, 100)
    return f"""<div style="margin-bottom:8px;">
      <div style="font-size:12px;font-weight:bold;color:#333;">{label}</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="background:{color};width:{w}%;height:14px;border-radius:3px;min-width:4px;"></div>
        <span style="font-size:11px;">{pct}%</span>
      </div></div>"""

# Convert markdown to clean HTML
def md_to_html(text):
    import re
    text = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    text = re.sub(r'^#{1,2}\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[-•]\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*</li>\n?)+', lambda m: '<ul>'+m.group(0)+'</ul>', text)
    text = re.sub(r'\n{2,}', '</p><p>', text)
    return '<p>' + text + '</p>'

llm_html = md_to_html(llm_output)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Rajapet Environmental Analysis Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: Georgia, serif; background:#f0f4f8; }}
  .header {{ background:linear-gradient(135deg,#1a3a5c,#2c7bb6); color:white; padding:28px 50px; text-align:center; }}
  .header h1 {{ font-size:22px; margin-bottom:6px; }}
  .header p  {{ font-size:13px; opacity:0.85; }}
  .stats-bar {{ display:flex; justify-content:center; gap:40px; background:#1a3a5c; padding:14px; flex-wrap:wrap; }}
  .stat {{ text-align:center; color:white; }}
  .stat .val {{ font-size:20px; font-weight:bold; color:#7ecef4; }}
  .stat .lbl {{ font-size:11px; opacity:0.8; }}
  .maps-bar {{ display:flex; justify-content:center; gap:16px; background:#f8f9fa; padding:14px; flex-wrap:wrap; border-bottom:2px solid #dee2e6; }}
  .maps-bar a {{ background:#2c7bb6; color:white; padding:8px 18px; border-radius:20px; text-decoration:none; font-size:13px; font-weight:bold; }}
  .maps-bar a:hover {{ background:#1a3a5c; }}
  .main {{ display:grid; grid-template-columns:1fr 320px; gap:24px; max-width:1300px; margin:30px auto; padding:0 30px 60px; }}
  .report-box {{ background:white; padding:36px 44px; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.1); line-height:1.9; font-size:14px; color:#222; white-space:pre-wrap; font-family:'Courier New',monospace; }}
  .sidebar {{ display:flex; flex-direction:column; gap:20px; }}
  .card {{ background:white; padding:16px; border-radius:10px; box-shadow:0 1px 8px rgba(0,0,0,0.08); }}
  .card h3 {{ font-size:13px; color:#1a3a5c; margin-bottom:12px; border-bottom:2px solid #2c7bb6; padding-bottom:6px; }}
  .metric-row {{ display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px solid #f0f0f0; }}
  .metric-row .val {{ font-weight:bold; color:#1a9850; }}
  .tag {{ display:inline-block; background:#2c7bb6; color:white; padding:4px 14px; border-radius:20px; font-size:11px; margin-bottom:16px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Rajapet — Environmental Change Analysis Report</h1>
  <p>Mistral AI (open-mistral-7b) &nbsp;|&nbsp; 2004 vs 2024 &nbsp;|&nbsp; CNN-LSTM Prediction + Remote Sensing</p>
</div>
<div class="stats-bar">
  <div class="stat"><div class="val">84.62%</div><div class="lbl">Model Accuracy</div></div>
  <div class="stat"><div class="val">{ndvi_delta:+.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">{lst_delta:+.2f}C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{lst_max}C</div><div class="lbl">Max LST 2024</div></div>
  <div class="stat"><div class="val">73.53%</div><div class="lbl">IoU Score</div></div>
</div>
<div class="maps-bar">
  <a href="NDVI_Map.html" target="_blank">🗺 NDVI Map</a>
  <a href="LST_Map.html"  target="_blank">🌡 LST Map</a>
  <a href="LULC_Map.html" target="_blank">🏔 LULC Map</a>
  <a href="Combined_Map.html" target="_blank">🗺 Combined Map</a>
</div>
<div class="main">
  <div>
    <div class="tag">Generated by Mistral AI — open-mistral-7b</div>
    <div class="report-box">{llm_html}</div>
  </div>
  <div class="sidebar">
    <div class="card">
      <h3>Model Performance</h3>
      <div class="metric-row"><span>Accuracy</span><span class="val">84.62%</span></div>
      <div class="metric-row"><span>Precision</span><span class="val">87.30%</span></div>
      <div class="metric-row"><span>Recall</span><span class="val">84.62%</span></div>
      <div class="metric-row"><span>F1 Score</span><span class="val">84.64%</span></div>
      <div class="metric-row"><span>IoU</span><span class="val">73.53%</span></div>
    </div>
    <div class="card">
      <h3>LULC 2024 (Predicted)</h3>
      {lulc_bar("Background", dist_24['Background'], "#c8c8c8")}
      {lulc_bar("Water",      dist_24['Water'],      "#0064ff")}
      {lulc_bar("Vegetation", dist_24['Vegetation'], "#00aa3c")}
      {lulc_bar("Built-up",   dist_24['Built-up'],   "#ff3232")}
      {lulc_bar("Barren",     dist_24['Barren'],     "#d2a032")}
    </div>
    <div class="card"><h3>NDVI Change Map</h3>{img_b64(OUT_DIR / "ndvi_change.png")}</div>
    <div class="card"><h3>LST Change Map</h3>{img_b64(OUT_DIR / "lst_change.png")}</div>
    <div class="card"><h3>Predicted LULC Map</h3>{img_b64(OUT_DIR / "lulc_map.png")}</div>
  </div>
</div>
</body>
</html>"""

out_html = LLM_DIR / "Rajapet_Analysis_Report.html"
out_txt  = LLM_DIR / "Rajapet_Analysis_Report.txt"

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(llm_output)

import subprocess
subprocess.Popen(["cmd", "/c", "start", "", str(out_html.resolve())])

print(f"HTML: {out_html.resolve()}")
print("\n========================================")
print("RAJAPET FULL PIPELINE COMPLETE")
print("========================================")
print("Maps saved in: Rajapet/outputs/maps/")
print("  - NDVI_Map.html")
print("  - LST_Map.html")
print("  - LULC_Map.html")
print("  - Combined_Map.html")
print("Report: Rajapet/outputs/llm/Rajapet_Analysis_Report.html")
