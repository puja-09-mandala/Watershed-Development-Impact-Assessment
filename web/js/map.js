/* map.js — GEE Code Editor style map controller */

async function loadManifest() {
  try {
    const r = await fetch("data/manifest.json");
    if (!r.ok) throw new Error();
    return await r.json();
  } catch {
    return { layers: {}, boundary: null, stats: {}, lulc_classes: [], lulc_available: false };
  }
}

const RAMP = {
  ndvi:        "linear-gradient(to right,#d4b483,#f5e642,#a8d96c,#4caf50,#1b7a1b,#0a4f0a)",
  ndvi_change: "linear-gradient(to right,#d73027,#fc8d59,#fee090,#ffffff,#91cf60,#1a9850)",
  lst:         "linear-gradient(to right,#2ecc71,#a8d96c,#f5e642,#ffa500,#e74c3c)",
  lst_change:  "linear-gradient(to right,#2166ac,#74add1,#e0f3f8,#ffffff,#fee090,#f46d43,#d73027)",
  lulc:        null,
  lulc_change: null,
};

const RAMP_NOTE = {
  ndvi:        "Dark green = dense vegetation · Yellow = sparse · Tan = bare/soil",
  ndvi_change: "Green = NDVI increase · White = no change · Red = decrease",
  lst:         "Green = cool · Yellow = moderate · Red = hot (°C)",
  lst_change:  "Blue = cooling · White = no change · Red = warming (°C)",
  lulc:        "Water · Vegetation · Built-up · Barren · Cropland",
  lulc_change: "White = no class change · Red = class changed",
};

function fmt(v, d, u) {
  return (typeof v === "number" && isFinite(v)) ? v.toFixed(d) + (u || "") : "n/a";
}
function decimals(k) { return (k || "").startsWith("ndvi") ? 3 : 2; }

function renderLegend(manifest, activeKeys) {
  const box = document.getElementById("legend-content");
  if (!box) return;
  const key = activeKeys[activeKeys.length - 1];
  if (!key) { box.innerHTML = "<p class='leg-note'>Select a layer.</p>"; return; }
  const meta = manifest.layers[key];
  if (!meta || !meta.available) { box.innerHTML = "<p class='leg-note'>Layer not available.</p>"; return; }

  if (meta.kind === "lulc") {
    const cls = manifest.lulc_classes || [];
    box.innerHTML = cls.map(c =>
      `<div class="leg-swatch"><i style="background:${c.color}"></i>${c.name}</div>`
    ).join("") + `<p class="leg-note">${RAMP_NOTE.lulc}</p>`;
    return;
  }
  if (meta.kind === "lulc_change") {
    box.innerHTML =
      `<div class="leg-swatch"><i style="background:#e0e0e0"></i>No change</div>` +
      `<div class="leg-swatch"><i style="background:#c0392b"></i>Class changed</div>` +
      `<p class="leg-note">${RAMP_NOTE.lulc_change}</p>`;
    return;
  }

  const d = decimals(meta.kind), u = meta.unit || "";
  const ramp = RAMP[meta.kind] || meta.ramp_css || "";
  box.innerHTML =
    `<div class="leg-bar" style="background:${ramp}"></div>` +
    `<div class="leg-labels">` +
    `<span>${fmt(meta.vmin, d, u)}</span>` +
    `<span>${fmt((meta.vmin + meta.vmax) / 2, d, u)}</span>` +
    `<span>${fmt(meta.vmax, d, u)}</span>` +
    `</div>` +
    `<p class="leg-note">${RAMP_NOTE[meta.kind] || ""}</p>`;
}

function renderStats(manifest, key) {
  const box = document.getElementById("stats-content");
  if (!box) return;
  const stats = manifest.stats || {};

  if (!key) { box.innerHTML = "<div class='stat-row'><span>Select a layer.</span></div>"; return; }
  if (key.startsWith("lulc")) {
    box.innerHTML = `<div class='stat-row' style='display:block;font-size:11px;color:#666'>${stats.lulc_note || "LULC data not available."}</div>`;
    return;
  }

  const row = (stats.rasters || []).find(r => r.name === key && r.n > 0);
  if (!row) { box.innerHTML = "<div class='stat-row'><span>No statistics.</span></div>"; return; }

  const d = decimals(key), u = key.startsWith("lst") ? " °C" : "";
  let html =
    `<div class="stat-row"><strong>${key.replace(/_/g, " ").toUpperCase()}</strong></div>` +
    `<div class="stat-row">Mean <span>${fmt(row.mean, d, u)}</span></div>` +
    `<div class="stat-row">Median <span>${fmt(row.median, d, u)}</span></div>` +
    `<div class="stat-row">Min <span>${fmt(row.min, d, u)}</span></div>` +
    `<div class="stat-row">Max <span>${fmt(row.max, d, u)}</span></div>` +
    `<div class="stat-row">Std <span>${fmt(row.std, d, u)}</span></div>` +
    `<div class="stat-row">Pixels <span>${Number(row.n).toLocaleString()}</span></div>`;

  if (key === "ndvi_change" || key === "lst_change") {
    const kind = key.startsWith("ndvi") ? "ndvi" : "lst";
    const cs = (stats.change_summaries || {})[kind] || {};
    const inc = cs["Increase"] || { percent: 0 };
    const dec = cs["Decrease"] || { percent: 0 };
    const stb = cs["No significant change"] || { percent: 0 };
    html +=
      `<div class="stat-row"><strong>Change Classes</strong></div>` +
      `<div class="stat-row">${kind === "lst" ? "Cooling" : "Increase"} <span>${fmt(inc.percent, 2, "%")}</span></div>` +
      `<div class="stat-row">No change <span>${fmt(stb.percent, 2, "%")}</span></div>` +
      `<div class="stat-row">${kind === "lst" ? "Warming" : "Decrease"} <span>${fmt(dec.percent, 2, "%")}</span></div>`;
  } else {
    const der = (stats.derived || {})[key.startsWith("ndvi") ? "ndvi" : "lst"];
    if (der) {
      const isN = key.startsWith("ndvi");
      const chg = isN ? der.mean_change : der.mean_change_c;
      const pct = isN && typeof der.percent_change === "number"
        ? ` (${fmt(der.percent_change, 2, "%")})` : "";
      html += `<div class="stat-row">2004→2024 Δ <span>${fmt(chg, isN ? 4 : 3, isN ? "" : " °C")}${pct}</span></div>`;
    }
  }
  box.innerHTML = html;
}

function initMapPage(options) {
  loadManifest().then(manifest => {
    const topo = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "© Esri" }
    );
    const imagery = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "© Esri" }
    );

    const map = L.map("map", { zoomControl: true, attributionControl: true });
    topo.addTo(map);
    L.control.scale({ metric: true, imperial: false }).addTo(map);

    let boundary = null;
    if (manifest.boundary) {
      boundary = L.geoJSON(manifest.boundary, {
        style: { color: "#ff0000", weight: 2, fillOpacity: 0 }
      });
      boundary.addTo(map);
      const b = boundary.getBounds();
      if (b.isValid()) map.fitBounds(b, { padding: [20, 20] });
    }

    // Build raster overlays
    const layerObjs = {};
    options.layerKeys.forEach(key => {
      const meta = manifest.layers[key];
      if (meta && meta.available && meta.png) {
        layerObjs[key] = L.imageOverlay(meta.png, meta.bounds, { opacity: 0.85, crossOrigin: true });
      }
    });

    if (!boundary) {
      const first = options.layerKeys.find(k => layerObjs[k]);
      if (first) map.fitBounds(manifest.layers[first].bounds, { padding: [20, 20] });
      else map.setView([16.353, 78.014], 13);
    }

    // Active keys tracking
    const activeKeys = new Set();

    function syncLayers() {
      options.layerKeys.forEach(key => {
        if (!layerObjs[key]) return;
        if (activeKeys.has(key)) {
          if (!map.hasLayer(layerObjs[key])) layerObjs[key].addTo(map);
        } else {
          if (map.hasLayer(layerObjs[key])) map.removeLayer(layerObjs[key]);
        }
      });
      if (boundary) boundary.bringToFront();
      const arr = [...activeKeys];
      renderLegend(manifest, arr);
      renderStats(manifest, arr[arr.length - 1] || null);
    }

    // Build checkbox layer panel
    const panel = document.getElementById("layer-panel");
    if (panel) {
      options.layerKeys.forEach((key, i) => {
        const row = document.createElement("div");
        row.className = "gee-layer-row";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.id = "cb_" + key;
        if (!layerObjs[key]) cb.disabled = true;

        // Default: check first available layer
        if (i === 0 && layerObjs[key]) {
          cb.checked = true;
          activeKeys.add(key);
          row.classList.add("checked");
        }

        const lbl = document.createElement("label");
        lbl.htmlFor = "cb_" + key;
        lbl.textContent = options.labels[key] || key;
        if (!layerObjs[key]) lbl.style.color = "#aaa";

        cb.addEventListener("change", () => {
          if (cb.checked) {
            activeKeys.add(key);
            row.classList.add("checked");
          } else {
            activeKeys.delete(key);
            row.classList.remove("checked");
          }
          syncLayers();
        });

        row.appendChild(cb);
        row.appendChild(lbl);
        panel.appendChild(row);
      });
    }

    // Basemap toggle
    let activeBm = topo;
    document.querySelectorAll(".gee-bm-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const next = btn.dataset.bm === "imagery" ? imagery : topo;
        if (activeBm !== next) { map.removeLayer(activeBm); next.addTo(map); activeBm = next; }
        document.querySelectorAll(".gee-bm-btn").forEach(b => b.classList.toggle("active", b === btn));
      });
    });

    // Notice
    const notice = document.getElementById("notice");
    if (notice) {
      const missing = options.layerKeys.filter(k => !layerObjs[k]);
      if (options.requireAll && missing.length) {
        notice.style.display = "block";
        notice.textContent = options.missingMessage || (manifest.stats || {}).lulc_note || "GeoTIFF data not available yet.";
      } else {
        notice.style.display = "none";
      }
    }

    // Initial render
    setTimeout(() => syncLayers(), 80);
  });
}
