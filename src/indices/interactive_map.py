import os
import io
import base64
import numpy as np
import rasterio
import rasterio.warp
import folium
import matplotlib
import matplotlib.colors as mcolors
from PIL import Image
from folium.plugins import Fullscreen, MiniMap
from pathlib import Path

DATA_DIR   = Path("data/raw")
OUTPUT_DIR = Path("outputs/maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LULC_NAMES      = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
LULC_COLORS_HEX = ["#b4b4b4",    "#0064ff", "#00aa3c",   "#ff3232",  "#d2a032"]
LULC_RGB        = [[180,180,180],[0,100,255],[0,170,60],  [255,50,50],[210,160,50]]

print("Reading data at full resolution...")

def read_raster_full(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        b = src.bounds
    return data, [[b.bottom,b.left],[b.top,b.right]], [(b.bottom+b.top)/2,(b.left+b.right)/2]

ndvi_2004, ndvi_bounds, center = read_raster_full(DATA_DIR/"Mahbubnagar_NDVI_2004.tif")
ndvi_2024, _,           _      = read_raster_full(DATA_DIR/"Mahbubnagar_NDVI_2024.tif")
lst_2004,  lst_bounds,  _      = read_raster_full(DATA_DIR/"Mahbubnagar_LST_2004.tif")
lst_2024,  _,           _      = read_raster_full(DATA_DIR/"Mahbubnagar_LST_2024.tif")
lulc_2004, lulc_bounds, _      = read_raster_full(DATA_DIR/"Mahbubnagar_LULC_2004.tif")
lulc_2024, _,           _      = read_raster_full(DATA_DIR/"Mahbubnagar_LULC_2024.tif")

print(f"NDVI shape: {ndvi_2004.shape}, LST shape: {lst_2004.shape}, LULC shape: {lulc_2004.shape}")

ndvi_2004 = np.nan_to_num(ndvi_2004, nan=0.0)
ndvi_2024 = np.nan_to_num(ndvi_2024, nan=0.0)
lst_2004  = np.nan_to_num(lst_2004,  nan=0.0)
lst_2024  = np.nan_to_num(lst_2024,  nan=0.0)

ndvi_m04 = round(float(np.nanmean(ndvi_2004[ndvi_2004!=0])),4)
ndvi_m24 = round(float(np.nanmean(ndvi_2024[ndvi_2024!=0])),4)
lst_m04  = round(float(np.nanmean(lst_2004[lst_2004>0])),2)
lst_m24  = round(float(np.nanmean(lst_2024[lst_2024>0])),2)

def lulc_pct(arr):
    idx = np.clip(np.rint(arr).astype(int),0,4)
    return {LULC_NAMES[i]: round(np.sum(idx==i)/idx.size*100,2) for i in range(5)}

dist_04     = lulc_pct(lulc_2004)
dist_24     = lulc_pct(lulc_2024)
changed_pct = round(float(np.sum(np.rint(lulc_2004).astype(int)!=np.rint(lulc_2024).astype(int))/lulc_2004.size*100),2)

# ============================================================
# HIGH QUALITY IMAGE RENDERING
# ============================================================

def to_url(data, cmap_name, vmin=None, vmax=None):
    d = data.astype(float).copy()
    mask = (d != 0) & np.isfinite(d)
    if vmin is None: vmin = float(np.percentile(d[mask], 2))  if mask.any() else 0
    if vmax is None: vmax = float(np.percentile(d[mask], 98)) if mask.any() else 1
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-10), 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[~mask, 3] = 0       # transparent where no data
    rgba[mask,   3] = 220    # semi-transparent overlay
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def change_url(change, cmap_name):
    d = change.astype(float).copy()
    lim = max(float(np.nanpercentile(np.abs(d), 98)), 0.01)
    norm = np.clip((d + lim) / (2 * lim), 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[:,:,3] = 210
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def lulc_url(lulc):
    idx  = np.clip(np.rint(lulc).astype(int), 0, 4)
    rgba = np.zeros((*idx.shape, 4), dtype=np.uint8)
    for i, c in enumerate(LULC_RGB):
        mask = idx == i
        rgba[mask, :3] = c
        rgba[mask,  3] = 225
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ============================================================
# GEE-STYLE MAP TEMPLATE
# ============================================================

def build_gee_map(title, subtitle, layers, legend_items, colorbar=None):
    m = folium.Map(
        location=center,
        zoom_start=9,
        tiles=None,
        control_scale=True,
        prefer_canvas=True
    )

    # OpenStreetMap
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name="🗺 OpenStreetMap",
        max_zoom=19,
        max_native_zoom=19
    ).add_to(m)

    # CartoDB Dark (GEE dark style)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="🌑 Dark (GEE Style)",
        max_zoom=19
    ).add_to(m)

    # CartoDB Light
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://carto.com/">CARTO</a>',
        name="⬜ Light",
        max_zoom=19
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # Add layers
    for name, url, bounds, show in layers:
        folium.raster_layers.ImageOverlay(
            image=url, bounds=bounds,
            name=name, opacity=0.85,
            show=show, cross_origin=False,
            zindex=10
        ).add_to(m)

    # GEE-style top bar
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;top:0;left:0;right:0;z-index:9999;
    background:linear-gradient(135deg,#1a3a5c,#2c7bb6);
    padding:10px 20px;display:flex;align-items:center;justify-content:space-between;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);">
      <div>
        <div style="color:white;font-size:15px;font-weight:bold;">{title}</div>
        <div style="color:rgba(255,255,255,0.75);font-size:11px;">{subtitle}</div>
      </div>
      <div style="color:rgba(255,255,255,0.8);font-size:11px;text-align:right;">
        CNN-LSTM Model &nbsp;|&nbsp; Accuracy: 84.62% &nbsp;|&nbsp; IoU: 73.53%
      </div>
    </div>
    <div style="height:52px;"></div>
    """))

    # Opacity sliders panel (GEE style)
    slider_html = ""
    for i, (name, url, bounds, show) in enumerate(layers):
        val = "0.85" if show else "0.0"
        pct = "85%" if show else "0%"
        clean = name.replace("'","").replace('"','')
        slider_html += f"""
        <div style="margin:8px 0;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:11px;color:#ddd;">{clean}</span>
            <span id="pct_{i}" style="font-size:11px;color:#7ecef4;font-weight:bold;">{pct}</span>
          </div>
          <input type="range" min="0" max="1" step="0.05" value="{val}"
            style="width:100%;accent-color:#2c7bb6;"
            oninput="updateLayer({i}, this.value)">
        </div>"""

    m.get_root().html.add_child(folium.Element(f"""
    <div id="layerPanel" style="position:fixed;top:62px;right:10px;z-index:9998;
    background:rgba(26,58,92,0.92);padding:14px;border-radius:10px;
    box-shadow:0 2px 14px rgba(0,0,0,0.4);width:220px;backdrop-filter:blur(4px);">
      <div style="color:white;font-size:13px;font-weight:bold;margin-bottom:10px;
      border-bottom:1px solid rgba(255,255,255,0.2);padding-bottom:8px;">
        🎚 Layer Opacity
      </div>
      {slider_html}
    </div>
    <script>
    function updateLayer(idx, val) {{
      document.getElementById('pct_'+idx).innerText = Math.round(val*100)+'%';
      var imgs = document.querySelectorAll('.leaflet-image-layer');
      if(imgs[idx]) imgs[idx].style.opacity = val;
    }}
    </script>
    """))

    # Legend panel (GEE style bottom left)
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
        f'<div style="width:20px;height:20px;background:{c};border-radius:3px;'
        f'border:1px solid rgba(255,255,255,0.2);flex-shrink:0;"></div>'
        f'<span style="font-size:11px;color:#ddd;">{l}</span></div>'
        for c, l in legend_items
    )
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:30px;left:10px;z-index:9998;
    background:rgba(26,58,92,0.92);padding:14px 16px;border-radius:10px;
    box-shadow:0 2px 14px rgba(0,0,0,0.4);min-width:200px;backdrop-filter:blur(4px);">
      <div style="color:white;font-size:13px;font-weight:bold;margin-bottom:10px;
      border-bottom:1px solid rgba(255,255,255,0.2);padding-bottom:8px;">Legend</div>
      {rows}
    </div>
    """))

    # Colorbar (GEE style bottom center)
    if colorbar:
        cb_title, cb_colors, cb_labels = colorbar
        bars = "".join(f'<div style="flex:1;background:{c};height:16px;"></div>' for c in cb_colors)
        lbls = "".join(f'<span style="flex:1;text-align:center;font-size:10px;color:#aaa;">{l}</span>' for l in cb_labels)
        m.get_root().html.add_child(folium.Element(f"""
        <div style="position:fixed;bottom:30px;left:50%;transform:translateX(-50%);z-index:9998;
        background:rgba(26,58,92,0.92);padding:10px 16px;border-radius:10px;
        box-shadow:0 2px 14px rgba(0,0,0,0.4);width:300px;backdrop-filter:blur(4px);">
          <div style="font-size:11px;color:#7ecef4;font-weight:bold;margin-bottom:6px;text-align:center;">{cb_title}</div>
          <div style="display:flex;border-radius:4px;overflow:hidden;">{bars}</div>
          <div style="display:flex;margin-top:4px;">{lbls}</div>
        </div>
        """))

    folium.LayerControl(collapsed=True, position="topleft").add_to(m)
    return m

# ============================================================
# MAP 1: NDVI
# ============================================================
print("\nRendering NDVI map...")
vmin_n = float(np.percentile(ndvi_2004[ndvi_2004!=0], 2))
vmax_n = float(np.percentile(ndvi_2024[ndvi_2024!=0], 98))

m = build_gee_map(
    title="Mahbubnagar District — NDVI Analysis",
    subtitle="Normalized Difference Vegetation Index  |  2004 vs 2024  |  Landsat",
    layers=[
        (f"🌿 NDVI 2004  (Mean: {ndvi_m04})", to_url(ndvi_2004,"RdYlGn",vmin_n,vmax_n), ndvi_bounds, False),
        (f"🌿 NDVI 2024  (Mean: {ndvi_m24})", to_url(ndvi_2024,"RdYlGn",vmin_n,vmax_n), ndvi_bounds, True),
        (f"📈 NDVI Change  ({ndvi_m24-ndvi_m04:+.4f})", change_url(ndvi_2024-ndvi_2004,"RdYlGn"), ndvi_bounds, False),
    ],
    legend_items=[
        ("#d73027","Very Low  (< 0.1)  — Barren/Urban"),
        ("#fc8d59","Low  (0.1–0.2)  — Sparse Veg"),
        ("#fee08b","Moderate  (0.2–0.4)  — Shrubs"),
        ("#91cf60","Good  (0.4–0.6)  — Agriculture"),
        ("#1a9850","Dense  (> 0.6)  — Forest"),
    ],
    colorbar=(
        f"NDVI  |  Min: {round(vmin_n,2)}  →  Max: {round(vmax_n,2)}  |  2004 Mean: {ndvi_m04}  →  2024 Mean: {ndvi_m24}",
        ["#d73027","#fc8d59","#fee08b","#d9ef8b","#91cf60","#1a9850"],
        [f"{round(vmin_n,1)}","0.1","0.2","0.4","0.6",f"{round(vmax_n,1)}"]
    )
)
m.save(str(OUTPUT_DIR/"NDVI_Map.html"))
print("  ✓ NDVI_Map.html")

# ============================================================
# MAP 2: LST
# ============================================================
print("Rendering LST map...")
vmin_l = float(np.percentile(lst_2004[lst_2004>0], 2))
vmax_l = float(np.percentile(lst_2024[lst_2024>0], 98))

m = build_gee_map(
    title="Mahbubnagar District — LST Analysis",
    subtitle="Land Surface Temperature  |  2004 vs 2024  |  Landsat",
    layers=[
        (f"🌡 LST 2004  (Mean: {lst_m04}°C)", to_url(lst_2004,"turbo",vmin_l,vmax_l), lst_bounds, False),
        (f"🌡 LST 2024  (Mean: {lst_m24}°C)", to_url(lst_2024,"turbo",vmin_l,vmax_l), lst_bounds, True),
        (f"📈 LST Change  ({lst_m24-lst_m04:+.2f}°C)", change_url(lst_2024-lst_2004,"RdBu_r"), lst_bounds, False),
    ],
    legend_items=[
        ("#30123b",f"Cool  (< 25°C)  — Water/Forest"),
        ("#1ac7c2",f"Moderate  (25–35°C)"),
        ("#f1ca3a",f"Warm  (35–50°C)  — Agriculture"),
        ("#e4460a",f"Hot  (50–60°C)  — Urban/Barren"),
        ("#7a0403",f"Very Hot  (> 60°C)  — Bare Rock"),
    ],
    colorbar=(
        f"LST (°C)  |  Min: {round(vmin_l,1)}°C  →  Max: {round(vmax_l,1)}°C  |  2004: {lst_m04}°C  →  2024: {lst_m24}°C",
        ["#30123b","#1ac7c2","#f1ca3a","#e4460a","#7a0403"],
        [f"{round(vmin_l)}°","30°","40°","55°",f"{round(vmax_l)}°"]
    )
)
m.save(str(OUTPUT_DIR/"LST_Map.html"))
print("  ✓ LST_Map.html")

# ============================================================
# MAP 3: LULC
# ============================================================
print("Rendering LULC map...")

chg      = (np.rint(lulc_2004).astype(int) != np.rint(lulc_2024).astype(int))
chg_rgba = np.zeros((*chg.shape,4), dtype=np.uint8)
chg_rgba[chg] = [255, 80, 0, 220]
buf = io.BytesIO()
Image.fromarray(chg_rgba,"RGBA").save(buf,format="PNG")
chg_url_str = "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

m = build_gee_map(
    title="Mahbubnagar District — LULC Analysis",
    subtitle="Land Use Land Cover  |  2004 vs 2024  |  CNN-LSTM Predicted",
    layers=[
        ("🏔 LULC 2004", lulc_url(lulc_2004), lulc_bounds, False),
        ("🏔 LULC 2024", lulc_url(lulc_2024), lulc_bounds, True),
        (f"🔄 LULC Change  ({changed_pct}%)", chg_url_str, lulc_bounds, False),
    ],
    legend_items=[
        (LULC_COLORS_HEX[0], f"Background  —  {dist_04['Background']}% → {dist_24['Background']}%"),
        (LULC_COLORS_HEX[1], f"Water  —  {dist_04['Water']}% → {dist_24['Water']}%"),
        (LULC_COLORS_HEX[2], f"Vegetation  —  {dist_04['Vegetation']}% → {dist_24['Vegetation']}%"),
        (LULC_COLORS_HEX[3], f"Built-up  —  {dist_04['Built-up']}% → {dist_24['Built-up']}%"),
        (LULC_COLORS_HEX[4], f"Barren  —  {dist_04['Barren']}% → {dist_24['Barren']}%"),
        ("#ff5000",           f"Changed Area  —  {changed_pct}%"),
    ],
    colorbar=None
)
m.save(str(OUTPUT_DIR/"LULC_Map.html"))
print("  ✓ LULC_Map.html")

# Open all 3
import subprocess
for f in ["NDVI_Map.html","LST_Map.html","LULC_Map.html"]:
    subprocess.Popen(["cmd","/c","start","",str((OUTPUT_DIR/f).resolve())])

print("\n========================================")
print("ALL 3 GEE-STYLE MAPS CREATED")
print("========================================")
