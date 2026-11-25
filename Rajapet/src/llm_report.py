import os
import base64
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key or api_key == "YOUR_API_KEY_HERE":
    raise ValueError("MISTRAL_API_KEY not set in .env file")

DATA_DIR = Path("Rajapet/processed")
OUT_DIR  = Path("Rajapet/outputs")
LLM_DIR  = Path("Rajapet/outputs/llm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_DIR.mkdir(parents=True, exist_ok=True)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]
LULC_RGB    = [[200,200,200],[0,100,255],[0,170,60],[255,50,50],[210,160,50]]

print("\n========================================")
print("STEP 3: CHANGE DETECTION — RAJAPET")
print("========================================")

ndvi_2004 = np.load(DATA_DIR / "ndvi_2004_raw.npy")
ndvi_2024 = np.load(DATA_DIR / "ndvi_2024_raw.npy")
lst_2004  = np.load(DATA_DIR / "lst_2004_raw.npy")
lst_2024  = np.load(DATA_DIR / "lst_2024_raw.npy")
lulc_2024 = np.load(OUT_DIR  / "lulc_2024_predicted.npy")

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
    return {LULC_NAMES[i]: round(np.sum(arr == i) / arr.size * 100, 2) for i in range(5)}

dist_24 = lulc_pct(lulc_2024)

print(f"NDVI: {ndvi_mean_04} -> {ndvi_mean_24} ({ndvi_delta:+.4f})")
print(f"LST : {lst_mean_04}C -> {lst_mean_24}C ({lst_delta:+.2f}C)")
print(f"LULC 2024: {dist_24}")

# Change maps
def save_change_map(data, title, cmap, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

save_change_map(ndvi_2024 - ndvi_2004, "NDVI Change (2004→2024)", "RdYlGn", OUT_DIR / "ndvi_change.png")
save_change_map(lst_2024  - lst_2004,  "LST Change (2004→2024)",  "RdBu_r", OUT_DIR / "lst_change.png")

def to_rgb(lulc):
    rgb = np.zeros((*lulc.shape, 3), dtype=np.uint8)
    for i, c in enumerate(LULC_RGB):
        rgb[lulc == i] = c
    return rgb

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(to_rgb(lulc_2024))
ax.set_title("Predicted LULC 2024 — Rajapet", fontsize=13, fontweight='bold')
ax.axis('off')
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=LULC_COLORS[i], label=f"{LULC_NAMES[i]} {dist_24[LULC_NAMES[i]]}%") for i in range(5)]
ax.legend(handles=patches, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "lulc_map.png", dpi=150, bbox_inches='tight')
plt.close()

print("Change maps saved.")

# ============================================================
# STEP 4 — MISTRAL LLM REPORT
# ============================================================

print("\n========================================")
print("STEP 4: CALLING MISTRAL LLM")
print("========================================")

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
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers, json=payload, timeout=120
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
        print(f"Error: {e}")
        break

if not llm_output:
    raise RuntimeError("Mistral API failed after all retries.")

print(llm_output)

# ============================================================
# STEP 5 — SAVE HTML REPORT
# ============================================================

def img_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
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

llm_html = llm_output.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

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
  <p>Mistral AI (open-mistral-7b) &nbsp;|&nbsp; 2004 vs 2024 &nbsp;|&nbsp; CNN-LSTM Prediction</p>
</div>
<div class="stats-bar">
  <div class="stat"><div class="val">84.62%</div><div class="lbl">Model Accuracy</div></div>
  <div class="stat"><div class="val">{ndvi_delta:+.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">{lst_delta:+.2f}C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{lst_max}C</div><div class="lbl">Max LST 2024</div></div>
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
    <div class="card"><h3>NDVI Change Map</h3>{img_b64(str(OUT_DIR / "ndvi_change.png"))}</div>
    <div class="card"><h3>LST Change Map</h3>{img_b64(str(OUT_DIR / "lst_change.png"))}</div>
    <div class="card"><h3>Predicted LULC Map</h3>{img_b64(str(OUT_DIR / "lulc_map.png"))}</div>
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

print(f"\nHTML: {out_html.resolve()}")
print(f"TXT : {out_txt.resolve()}")
print("\n========================================")
print("RAJAPET REPORT COMPLETE")
print("========================================")
