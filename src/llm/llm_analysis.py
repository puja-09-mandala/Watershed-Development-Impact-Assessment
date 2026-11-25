import os
import base64
import time
import numpy as np
import rasterio
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key or api_key == "YOUR_API_KEY_HERE":
    raise ValueError("MISTRAL_API_KEY not set in .env file")

DATA_DIR   = "data/raw"
OUTPUT_DIR = "outputs/llm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n========================================")
print("STEP 1: COMPUTING STATISTICS")
print("========================================")

def read_raster(path, max_size=400):
    with rasterio.open(path) as src:
        scale = min(1.0, max_size / max(src.height, src.width))
        h = max(1, int(src.height * scale))
        w = max(1, int(src.width  * scale))
        data = src.read(1, out_shape=(h, w),
                        resampling=rasterio.enums.Resampling.bilinear)
        return data.astype(np.float64)

ndvi_2004 = read_raster(f"{DATA_DIR}/Mahbubnagar_NDVI_2004.tif")
ndvi_2024 = read_raster(f"{DATA_DIR}/Mahbubnagar_NDVI_2024.tif")
lst_2004  = read_raster(f"{DATA_DIR}/Mahbubnagar_LST_2004.tif")
lst_2024  = read_raster(f"{DATA_DIR}/Mahbubnagar_LST_2024.tif")
lulc_2004 = read_raster(f"{DATA_DIR}/Mahbubnagar_LULC_2004.tif")
lulc_2024 = read_raster(f"{DATA_DIR}/Mahbubnagar_LULC_2024.tif")

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]

def lulc_pct(arr):
    idx = np.clip(np.rint(arr).astype(int), 0, 4)
    return {LULC_NAMES[i]: round(np.sum(idx == i) / idx.size * 100, 2) for i in range(5)}

ndvi_change = ndvi_2024 - ndvi_2004
lst_change  = lst_2024  - lst_2004
lulc_change = (lulc_2004 != lulc_2024)

ndvi_mean_04  = round(float(np.nanmean(ndvi_2004)), 4)
ndvi_mean_24  = round(float(np.nanmean(ndvi_2024)), 4)
ndvi_delta    = round(float(np.nanmean(ndvi_change)), 4)
ndvi_max_gain = round(float(np.nanmax(ndvi_change)), 4)
ndvi_max_loss = round(float(np.nanmin(ndvi_change)), 4)

lst_mean_04   = round(float(np.nanmean(lst_2004)), 2)
lst_mean_24   = round(float(np.nanmean(lst_2024)), 2)
lst_delta     = round(float(np.nanmean(lst_change)), 2)
lst_max       = round(float(np.nanmax(lst_2024)), 2)
lst_min       = round(float(np.nanmin(lst_2024)), 2)

dist_04       = lulc_pct(lulc_2004)
dist_24       = lulc_pct(lulc_2024)
changed_pct   = round(float(np.sum(lulc_change) / lulc_change.size * 100), 2)

print(f"NDVI: {ndvi_mean_04} -> {ndvi_mean_24} (change: {ndvi_delta:+.4f})")
print(f"LST : {lst_mean_04}C -> {lst_mean_24}C (change: {lst_delta:+.2f}C)")
print(f"LULC changed: {changed_pct}%")


# ============================================================
# STEP 2 — MISTRAL API CALL
# ============================================================

print("\n========================================")
print("STEP 2: CALLING MISTRAL API")
print("========================================")

import requests

prompt = f"""You are an expert environmental scientist. Analyze this satellite remote sensing data for Mahbubnagar District, Telangana, India (2004-2024) and write a detailed environmental change analysis report.

DATA:
- NDVI 2004: {ndvi_mean_04}, NDVI 2024: {ndvi_mean_24}, Change: {ndvi_delta:+.4f}, Max Gain: {ndvi_max_gain}, Max Loss: {ndvi_max_loss}
- LST 2004: {lst_mean_04}C, LST 2024: {lst_mean_24}C, Change: {lst_delta:+.2f}C, Range: {lst_min}C to {lst_max}C
- LULC 2004: Background {dist_04['Background']}%, Water {dist_04['Water']}%, Vegetation {dist_04['Vegetation']}%, Built-up {dist_04['Built-up']}%, Barren {dist_04['Barren']}%
- LULC 2024: Background {dist_24['Background']}%, Water {dist_24['Water']}%, Vegetation {dist_24['Vegetation']}%, Built-up {dist_24['Built-up']}%, Barren {dist_24['Barren']}%
- Total LULC Changed: {changed_pct}%
- CNN-LSTM Model: Accuracy 84.62%, F1 84.64%, IoU 73.53%

Write report with these sections:
1. EXECUTIVE SUMMARY
2. NDVI ANALYSIS
3. LST ANALYSIS
4. LULC ANALYSIS
5. ENVIRONMENTAL CONCERNS
6. RECOMMENDATIONS
7. CONCLUSION"""

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "open-mistral-7b",
    "messages": [{"role": "user", "content": prompt}]
}

llm_output = None
for attempt in range(5):
    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        if resp.status_code == 200:
            llm_output = resp.json()["choices"][0]["message"]["content"]
            break
        elif resp.status_code == 429:
            wait = (attempt + 1) * 20
            print(f"Rate limited. Waiting {wait}s... (attempt {attempt+1}/5)")
            time.sleep(wait)
        else:
            print(f"API Error {resp.status_code}: {resp.text}")
            break
    except Exception as e:
        print(f"Request error: {e}")
        break

if not llm_output:
    raise RuntimeError("Mistral API call failed after all retries.")

print(llm_output)


# ============================================================
# STEP 3 — SAVE AS HTML + TXT
# ============================================================

print("\n========================================")
print("STEP 3: SAVING REPORT")
print("========================================")

def img_tag(path, alt):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:6px;margin-top:8px;">'
    return f'<p style="color:#aaa;font-size:12px;">{alt} not found</p>'

ndvi_img_tag = img_tag("outputs/change_detection/ndvi_change.png", "NDVI change map")
lst_img_tag  = img_tag("outputs/change_detection/lst_change.png",  "LST change map")
lulc_img_tag = img_tag("outputs/change_detection/lulc_change.png", "LULC change map")

def lulc_bar(label, pct04, pct24, color):
    w04 = min(pct04 * 5, 100)
    w24 = min(pct24 * 5, 100)
    return f"""
    <div style="margin-bottom:10px;">
      <div style="font-size:12px;font-weight:bold;color:#333;margin-bottom:3px;">{label}</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:11px;width:35px;color:#666;">2004</span>
        <div style="background:{color};width:{w04}%;height:14px;border-radius:3px;min-width:4px;"></div>
        <span style="font-size:11px;color:#333;">{pct04}%</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:3px;">
        <span style="font-size:11px;width:35px;color:#666;">2024</span>
        <div style="background:{color};width:{w24}%;height:14px;border-radius:3px;min-width:4px;opacity:0.6;"></div>
        <span style="font-size:11px;color:#333;">{pct24}%</span>
      </div>
    </div>"""

llm_html = llm_output.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Mahbubnagar LLM Analysis Report</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: Georgia, serif; background:#f0f4f8; }}
  .header {{
    background: linear-gradient(135deg,#1a3a5c,#2c7bb6);
    color:white; padding:28px 50px; text-align:center;
  }}
  .header h1 {{ font-size:22px; margin-bottom:6px; }}
  .header p  {{ font-size:13px; opacity:0.85; }}
  .stats-bar {{
    display:flex; justify-content:center; gap:40px;
    background:#1a3a5c; padding:14px; flex-wrap:wrap;
  }}
  .stat {{ text-align:center; color:white; }}
  .stat .val {{ font-size:20px; font-weight:bold; color:#7ecef4; }}
  .stat .lbl {{ font-size:11px; opacity:0.8; }}
  .main {{ display:grid; grid-template-columns:1fr 340px; gap:24px;
           max-width:1300px; margin:30px auto; padding:0 30px 60px; }}
  .report-box {{
    background:white; padding:36px 44px; border-radius:10px;
    box-shadow:0 2px 12px rgba(0,0,0,0.1);
    line-height:1.9; font-size:14px; color:#222; white-space:pre-wrap;
    font-family: 'Courier New', monospace;
  }}
  .sidebar {{ display:flex; flex-direction:column; gap:20px; }}
  .card {{
    background:white; padding:16px; border-radius:10px;
    box-shadow:0 1px 8px rgba(0,0,0,0.08);
  }}
  .card h3 {{ font-size:13px; color:#1a3a5c; margin-bottom:12px;
              border-bottom:2px solid #2c7bb6; padding-bottom:6px; }}
  .metric-row {{ display:flex; justify-content:space-between;
                 font-size:12px; padding:4px 0; border-bottom:1px solid #f0f0f0; }}
  .metric-row .val {{ font-weight:bold; color:#1a9850; }}
  .tag {{
    display:inline-block; background:#2c7bb6; color:white;
    padding:4px 14px; border-radius:20px; font-size:11px; margin-bottom:16px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Mahbubnagar District — Environmental Change Analysis Report</h1>
  <p>Mistral AI (open-mistral-7b) &nbsp;|&nbsp; 2004 vs 2024 &nbsp;|&nbsp; CNN-LSTM + Remote Sensing</p>
</div>

<div class="stats-bar">
  <div class="stat"><div class="val">84.62%</div><div class="lbl">Model Accuracy</div></div>
  <div class="stat"><div class="val">{ndvi_delta:+.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">{lst_delta:+.2f}C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{changed_pct}%</div><div class="lbl">LULC Changed</div></div>
  <div class="stat"><div class="val">73.53%</div><div class="lbl">IoU Score</div></div>
</div>

<div class="main">

  <div>
    <div class="tag">Generated by Mistral AI — open-mistral-7b</div>
    <div class="report-box">{llm_html}</div>
  </div>

  <div class="sidebar">

    <div class="card">
      <h3>Model Performance</h3>
      <div class="metric-row"><span>Accuracy</span> <span class="val">84.62%</span></div>
      <div class="metric-row"><span>Precision</span><span class="val">87.30%</span></div>
      <div class="metric-row"><span>Recall</span>   <span class="val">84.62%</span></div>
      <div class="metric-row"><span>F1 Score</span> <span class="val">84.64%</span></div>
      <div class="metric-row"><span>IoU</span>      <span class="val">73.53%</span></div>
    </div>

    <div class="card">
      <h3>Per-Class F1 Score</h3>
      <div class="metric-row"><span>Background</span> <span class="val">0.84</span></div>
      <div class="metric-row"><span>Water</span>      <span class="val">0.77</span></div>
      <div class="metric-row"><span>Vegetation</span> <span class="val">0.87</span></div>
      <div class="metric-row"><span>Built-up</span>   <span style="font-weight:bold;color:#dc3232;">0.11</span></div>
      <div class="metric-row"><span>Barren</span>     <span class="val">0.56</span></div>
    </div>

    <div class="card">
      <h3>LULC Distribution</h3>
      {lulc_bar("Background",  dist_04['Background'],  dist_24['Background'],  "#c8c8c8")}
      {lulc_bar("Water",       dist_04['Water'],       dist_24['Water'],       "#0064ff")}
      {lulc_bar("Vegetation",  dist_04['Vegetation'],  dist_24['Vegetation'],  "#00aa3c")}
      {lulc_bar("Built-up",    dist_04['Built-up'],    dist_24['Built-up'],    "#ff3232")}
      {lulc_bar("Barren",      dist_04['Barren'],      dist_24['Barren'],      "#d2a032")}
    </div>

    <div class="card">
      <h3>NDVI Change Map</h3>
      {ndvi_img_tag}
    </div>

    <div class="card">
      <h3>LST Change Map</h3>
      {lst_img_tag}
    </div>

    <div class="card">
      <h3>LULC Change Map</h3>
      {lulc_img_tag}
    </div>

  </div>
</div>
</body>
</html>"""

out_html = f"{OUTPUT_DIR}/LLM_Analysis_Report.html"
out_txt  = f"{OUTPUT_DIR}/LLM_Analysis_Report.txt"

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(llm_output)

import subprocess
subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(out_html)])

print("HTML:", os.path.abspath(out_html))
print("TXT :", os.path.abspath(out_txt))
print("\n========================================")
print("MISTRAL LLM REPORT GENERATED")
print("========================================")
