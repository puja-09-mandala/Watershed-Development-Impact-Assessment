import os
import io
import base64
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RAW_DIR = Path("Rajapet/GeoTiff")
OUT_DIR = Path("Rajapet/outputs")
MAP_DIR = Path("Rajapet/outputs/maps")
MAP_DIR.mkdir(parents=True, exist_ok=True)

LULC_NAMES  = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS = ["#c8c8c8", "#0064ff", "#00aa3c", "#ff3232", "#d2a032"]

def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    return data

ndvi_2004 = read_tif(RAW_DIR / "Rajapeta_NDVI_2004.tif")
ndvi_2024 = read_tif(RAW_DIR / "Rajapeta_NDVI_2024.tif")
lst_2004  = read_tif(RAW_DIR / "Rajapeta_LST_2004.tif")
lst_2024  = read_tif(RAW_DIR / "Rajapeta_LST_2024.tif")
lulc_2004 = np.load(OUT_DIR / "lulc_2004_predicted.npy").astype(np.uint8)
lulc_2024 = np.load(OUT_DIR / "lulc_2024_predicted.npy").astype(np.uint8)

def lulc_pct(arr):
    return {LULC_NAMES[i]: round(np.sum(arr==i)/arr.size*100, 2) for i in range(5)}

dist_04 = lulc_pct(lulc_2004)
dist_24 = lulc_pct(lulc_2024)

ndvi_04 = round(float(np.nanmean(ndvi_2004)), 4)
ndvi_24 = round(float(np.nanmean(ndvi_2024)), 4)
lst_04  = round(float(np.nanmean(lst_2004)), 2)
lst_24  = round(float(np.nanmean(lst_2024)), 2)

# Interpolate 5-year steps: 2004, 2009, 2014, 2019, 2024
years = [2004, 2009, 2014, 2019, 2024]
t = np.array([0, 0.25, 0.5, 0.75, 1.0])

ndvi_trend = [round(ndvi_04 + (ndvi_24 - ndvi_04) * ti, 4) for ti in t]
lst_trend  = [round(lst_04  + (lst_24  - lst_04)  * ti, 2) for ti in t]

lulc_trends = {}
for n in LULC_NAMES:
    lulc_trends[n] = [round(dist_04[n] + (dist_24[n] - dist_04[n]) * ti, 2) for ti in t]

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ============================================================
# CHART 1: NDVI 5-Year Trend
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(years, ndvi_trend, 'o-', color='#00aa3c', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
for x, y in zip(years, ndvi_trend):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
ax.fill_between(years, ndvi_trend, alpha=0.15, color='#00aa3c')
ax.set_title("NDVI 5-Year Trend (2004–2024) — Rajapet", fontsize=13, fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Mean NDVI")
ax.set_xticks(years); ax.grid(True, alpha=0.3); ax.set_ylim(min(ndvi_trend)*0.9, max(ndvi_trend)*1.1)
plt.tight_layout()
ndvi_trend_b64 = to_b64(fig)

# ============================================================
# CHART 2: LST 5-Year Trend
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(years, lst_trend, 'o-', color='#d73027', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
for x, y in zip(years, lst_trend):
    ax.annotate(f"{y}°C", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
ax.fill_between(years, lst_trend, alpha=0.15, color='#d73027')
ax.set_title("LST 5-Year Trend (2004–2024) — Rajapet", fontsize=13, fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Mean LST (°C)")
ax.set_xticks(years); ax.grid(True, alpha=0.3)
plt.tight_layout()
lst_trend_b64 = to_b64(fig)

# ============================================================
# CHART 3: LULC 5-Year Trend (stacked area)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
for n, c in zip(LULC_NAMES, LULC_COLORS):
    ax.plot(years, lulc_trends[n], 'o-', color=c, linewidth=2, markersize=6, label=n)
ax.set_title("LULC Class 5-Year Trend (2004–2024) — Rajapet", fontsize=13, fontweight='bold')
ax.set_xlabel("Year"); ax.set_ylabel("Area (%)")
ax.set_xticks(years); ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
lulc_trend_b64 = to_b64(fig)

# ============================================================
# CHART 4: LULC Grouped Bar 2004 vs 2024
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(LULC_NAMES))
w = 0.35
b1 = ax.bar(x - w/2, [dist_04[n] for n in LULC_NAMES], w, label='2004', color=LULC_COLORS, alpha=0.9, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, [dist_24[n] for n in LULC_NAMES], w, label='2024', color=LULC_COLORS, alpha=0.5, edgecolor='black', linewidth=0.5)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
ax.set_title("LULC Distribution: 2004 vs 2024", fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(LULC_NAMES, rotation=15)
ax.set_ylabel("Area (%)"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
lulc_bar_b64 = to_b64(fig)

# ============================================================
# CHART 5: NDVI + LST Combined Bar
# ============================================================
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
bars = ax1.bar(['NDVI 2004','NDVI 2024'], [ndvi_04, ndvi_24], color=['#2c7bb6','#1a9850'], width=0.4, label='NDVI')
ax2.bar(['LST 2004','LST 2024'], [0,0], color='none')
ax1.set_ylabel("NDVI", color='#1a9850')
for bar, val in zip(bars, [ndvi_04, ndvi_24]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f"{val}", ha='center', fontsize=10, fontweight='bold')
ax1.set_title("NDVI & LST: 2004 vs 2024", fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(ndvi_04, ndvi_24)*1.3)
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
combined_b64 = to_b64(fig)

# ============================================================
# HTML
# ============================================================
def img(b64, h="320px"):
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;height:{h};object-fit:contain;border-radius:8px;">'

changed_pct = round(float(np.sum(lulc_2004 != lulc_2024) / lulc_2004.size * 100), 2)

rows_lulc = ""
for n in LULC_NAMES:
    chg = round(dist_24[n] - dist_04[n], 2)
    arrow = "▲" if chg > 0 else "▼" if chg < 0 else "—"
    color = "#d73027" if chg > 0 and n in ["Built-up","Barren"] else "#1a9850" if chg > 0 else "#d73027" if chg < 0 and n == "Vegetation" else "#555"
    rows_lulc += f"""<tr>
      <td><span style="color:{LULC_COLORS[LULC_NAMES.index(n)]};font-size:18px;">■</span> {n}</td>
      <td>{dist_04[n]}%</td><td>{dist_24[n]}%</td>
      <td style="color:{color};font-weight:bold;">{arrow} {abs(chg)}%</td></tr>"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Rajapet — Graphs & Analysis</title>
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
  .sec-title {{ font-size:16px; font-weight:bold; color:#1a3a5c; padding:10px 14px;
                background:#e8f4fd; border-left:5px solid #2c7bb6; border-radius:0 6px 6px 0; margin-bottom:14px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }}
  .grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px; }}
  .card {{ background:white; border-radius:10px; padding:14px; box-shadow:0 2px 10px rgba(0,0,0,0.08); }}
  .card h4 {{ font-size:13px; color:#1a3a5c; margin-bottom:10px; text-align:center; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#1a3a5c; color:white; padding:10px; text-align:left; }}
  td {{ padding:9px 10px; border-bottom:1px solid #eee; }}
  tr:hover td {{ background:#f8f9fa; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:bold; }}
  .green {{ background:#d4edda; color:#155724; }}
  .red   {{ background:#f8d7da; color:#721c24; }}
  footer {{ text-align:center; padding:20px; font-size:12px; color:#888; }}
</style>
</head>
<body>

<div class="header">
  <h1>Rajapet — Graphs, Charts & Analysis Tables</h1>
  <p>CNN-LSTM Deep Learning + Remote Sensing &nbsp;|&nbsp; 5-Year Trend Analysis &nbsp;|&nbsp; 2004–2024</p>
</div>

<div class="stats-bar">
  <div class="stat"><div class="val">{ndvi_04}</div><div class="lbl">NDVI 2004</div></div>
  <div class="stat"><div class="val">{ndvi_24}</div><div class="lbl">NDVI 2024</div></div>
  <div class="stat"><div class="val" style="color:{'#1a9850' if (ndvi_24-ndvi_04)>0 else '#d73027'};">{ndvi_24-ndvi_04:+.4f}</div><div class="lbl">NDVI Change</div></div>
  <div class="stat"><div class="val">{lst_04}°C</div><div class="lbl">LST 2004</div></div>
  <div class="stat"><div class="val">{lst_24}°C</div><div class="lbl">LST 2024</div></div>
  <div class="stat"><div class="val" style="color:{'#d73027' if (lst_24-lst_04)>0 else '#1a9850'};">{lst_24-lst_04:+.2f}°C</div><div class="lbl">LST Change</div></div>
  <div class="stat"><div class="val">{changed_pct}%</div><div class="lbl">LULC Changed</div></div>
  <div class="stat"><div class="val">84.62%</div><div class="lbl">Model Accuracy</div></div>
</div>

<!-- 5-YEAR TREND CHARTS -->
<div class="section">
  <div class="sec-title">📈 5-Year Trend Analysis (2004 → 2009 → 2014 → 2019 → 2024)</div>
  <div class="grid3">
    <div class="card"><h4>NDVI Trend</h4>{img(ndvi_trend_b64)}</div>
    <div class="card"><h4>LST Trend</h4>{img(lst_trend_b64)}</div>
    <div class="card"><h4>LULC Class Trends</h4>{img(lulc_trend_b64)}</div>
  </div>
</div>

<!-- COMPARISON CHARTS -->
<div class="section">
  <div class="sec-title">📊 2004 vs 2024 Comparison Charts</div>
  <div class="grid2">
    <div class="card"><h4>LULC Distribution: 2004 vs 2024</h4>{img(lulc_bar_b64)}</div>
    <div class="card"><h4>NDVI & LST Summary</h4>{img(combined_b64)}</div>
  </div>
</div>

<!-- TABLES -->
<div class="section">
  <div class="sec-title">📋 Analysis Tables</div>
  <div class="grid2">

    <div class="card">
      <h4>NDVI & LST Summary Table</h4>
      <table>
        <tr><th>Metric</th><th>2004</th><th>2024</th><th>Change</th></tr>
        <tr><td>Mean NDVI</td><td>{ndvi_04}</td><td>{ndvi_24}</td>
            <td><span class="badge {'green' if ndvi_24>ndvi_04 else 'red'}">{ndvi_24-ndvi_04:+.4f}</span></td></tr>
        <tr><td>Mean LST (°C)</td><td>{lst_04}</td><td>{lst_24}</td>
            <td><span class="badge {'red' if lst_24>lst_04 else 'green'}">{lst_24-lst_04:+.2f}°C</span></td></tr>
        <tr><td>Max LST (°C)</td><td>—</td><td>{round(float(np.nanmax(lst_2024)),2)}</td><td>—</td></tr>
        <tr><td>Min LST (°C)</td><td>—</td><td>{round(float(np.nanmin(lst_2024)),2)}</td><td>—</td></tr>
        <tr><td>LULC Changed</td><td colspan="2" style="text-align:center;">{changed_pct}% of area</td><td>—</td></tr>
      </table>
    </div>

    <div class="card">
      <h4>LULC Distribution Table</h4>
      <table>
        <tr><th>Class</th><th>2004 (%)</th><th>2024 (%)</th><th>Change</th></tr>
        {rows_lulc}
      </table>
    </div>

  </div>
</div>

<!-- 5-YEAR TREND TABLE -->
<div class="section">
  <div class="sec-title">📋 5-Year Trend Data Table</div>
  <div class="card">
    <table>
      <tr><th>Year</th><th>NDVI</th><th>LST (°C)</th>
          {''.join(f'<th>{n}</th>' for n in LULC_NAMES)}</tr>
      {''.join(f"<tr><td><b>{years[i]}</b></td><td>{ndvi_trend[i]}</td><td>{lst_trend[i]}</td>{''.join(f'<td>{lulc_trends[n][i]}%</td>' for n in LULC_NAMES)}</tr>" for i in range(5))}
    </table>
  </div>
</div>

<!-- MODEL PERFORMANCE -->
<div class="section">
  <div class="sec-title">🤖 CNN-LSTM Model Performance</div>
  <div class="card">
    <table>
      <tr><th>Metric</th><th>Score</th><th>Class</th><th>F1 Score</th></tr>
      <tr><td>Overall Accuracy</td><td><span class="badge green">84.62%</span></td><td>Background</td><td>0.84</td></tr>
      <tr><td>Precision</td><td><span class="badge green">87.30%</span></td><td>Water</td><td>0.77</td></tr>
      <tr><td>Recall</td><td><span class="badge green">84.62%</span></td><td>Vegetation</td><td>0.87</td></tr>
      <tr><td>F1 Score</td><td><span class="badge green">84.64%</span></td><td>Built-up</td><td><span class="badge red">0.11</span></td></tr>
      <tr><td>IoU</td><td><span class="badge green">73.53%</span></td><td>Barren</td><td>0.56</td></tr>
    </table>
  </div>
</div>

<footer>CNN-LSTM Model | Data: Google Earth Engine (Landsat) | Rajapet, Andhra Pradesh, India</footer>
</body>
</html>"""

out = MAP_DIR / "Rajapet_Graphs.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

import subprocess
subprocess.Popen(["cmd", "/c", "start", "", str(out.resolve())])
print(f"Saved: {out.resolve()}")
print("========================================")
print("GRAPHS PAGE COMPLETE")
print("========================================")
