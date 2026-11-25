async function loadStats() {
  try { const r = await fetch("data/stats.json"); return r.ok ? await r.json() : null; }
  catch { return null; }
}

Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
Chart.defaults.font.size   = 12;
Chart.defaults.color       = "#333";

const GRID = "rgba(0,0,0,0.06)";

function fmt(v, d = 3, u = "") {
  return typeof v === "number" && isFinite(v) ? v.toFixed(d) + u : "—";
}
function rrow(stats, name) {
  return (stats.rasters || []).find(r => r.name === name && r.n > 0) || null;
}
function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}
function addSection(root, title) {
  const s = el("div", "ch-section");
  s.appendChild(el("div", "ch-sec-title", title));
  const g = el("div", "ch-grid");
  s.appendChild(g);
  root.appendChild(s);
  return g;
}
function card(grid, title, note, wide) {
  const c = el("div", wide ? "ch-card wide" : "ch-card");
  if (title) c.appendChild(el("div", "ch-card-title", title));
  if (note)  c.appendChild(el("div", "ch-card-note", note));
  grid.appendChild(c);
  return c;
}
function emptyCard(grid, title, msg) {
  const c = card(grid, title, null, false);
  c.appendChild(el("div", "ch-empty", msg));
}
function mkCanvas(parent) {
  const box = el("div", "ch-box");
  const cv  = document.createElement("canvas");
  box.appendChild(cv);
  parent.appendChild(box);
  return cv;
}

/* tooltip */
function tooltip(dec, unit) {
  return {
    backgroundColor: "#fff", borderColor: "#ddd", borderWidth: 1,
    titleColor: "#111", bodyColor: "#444", padding: 10,
    callbacks: {
      label(ctx) {
        const v = ctx.parsed.y;
        return isFinite(v) ? `${ctx.dataset.label || ""}: ${fmt(v, dec, unit)}` : "";
      }
    }
  };
}

/* base opts */
function opts(xL, yL, cfg = {}) {
  return {
    responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
    plugins: {
      legend: { position: "bottom", labels: { usePointStyle: true, pointStyle: "circle", boxWidth: 9, padding: 14, color: "#333" } },
      tooltip: tooltip(cfg.dec ?? 3, cfg.unit ?? ""),
    },
    scales: {
      x: {
        title: { display: !!xL, text: xL, color: "#555", font: { size: 11 } },
        ticks: { color: "#555", maxRotation: cfg.rot ?? 0, autoSkip: cfg.skip !== false },
        grid: { color: GRID }
      },
      y: {
        title: { display: !!yL, text: yL, color: "#555", font: { size: 11 } },
        ticks: { color: "#555", callback: cfg.yFmt },
        grid: { color: GRID },
        beginAtZero: !!cfg.zero,
        ...(cfg.yMin !== undefined ? { min: cfg.yMin } : {}),
        ...(cfg.yMax !== undefined ? { max: cfg.yMax } : {}),
      }
    }
  };
}

/* ── Trend line chart ── */
function trendChart(grid, title, note, labels, datasets, xL, yL, cfg) {
  const c = card(grid, title, note, true);
  new Chart(mkCanvas(c), { type: "line", data: { labels, datasets }, options: opts(xL, yL, cfg) });
}

/* ── Bar chart ── */
function barChart(grid, title, note, labels, datasets, yL, cfg, wide) {
  const c = card(grid, title, note, !!wide);
  new Chart(mkCanvas(c), { type: "bar", data: { labels, datasets }, options: opts(null, yL, cfg) });
}

/* ── Change detection bar (green/grey/red) ── */
function changeBar(grid, title, note, inc, stb, dec, incLbl, stbLbl, decLbl, yL) {
  // summary badges
  const c = card(grid, title, note, false);
  const badges = el("div", "chg-bars");
  badges.innerHTML = `
    <div class="chg-bar-item chg-inc"><div class="val">${fmt(inc,1,"%")}</div><div class="lbl">${incLbl}</div></div>
    <div class="chg-bar-item chg-stb"><div class="val">${fmt(stb,1,"%")}</div><div class="lbl">${stbLbl}</div></div>
    <div class="chg-bar-item chg-dec"><div class="val">${fmt(dec,1,"%")}</div><div class="lbl">${decLbl}</div></div>`;
  c.appendChild(badges);
  new Chart(mkCanvas(c), {
    type: "bar",
    data: {
      labels: [incLbl, stbLbl, decLbl],
      datasets: [{ label: "% pixels", data: [inc, stb, dec],
        backgroundColor: ["#2e7d32","#9e9e9e","#c62828"], borderWidth: 0, borderRadius: 6 }]
    },
    options: opts(null, yL, { dec: 1, unit: "%", zero: true, yFmt: v => v + "%" })
  });
}

/* ── Paired histogram ── */
function pairedBar(grid, title, note, ph, xL, dec) {
  const c = card(grid, title, note, false);
  new Chart(mkCanvas(c), {
    type: "bar",
    data: {
      labels: ph.labels.map(v => Number(v).toFixed(dec)),
      datasets: [
        { label: ph.label_a, data: ph.counts_a, backgroundColor: "rgba(139,90,43,0.75)", borderWidth: 0 },
        { label: ph.label_b, data: ph.counts_b, backgroundColor: "rgba(46,125,50,0.75)",  borderWidth: 0 }
      ]
    },
    options: opts(xL, "Pixel count", { dec: 0, zero: true, rot: 45, yFmt: v => Number(v).toLocaleString() })
  });
}

/* ── Scatter ── */
function scatterChart(grid, title, note, sc, xL, yL, dec) {
  const pts = sc.actual.map((a, i) => ({ x: a, y: sc.predicted[i] }));
  const all = sc.actual.concat(sc.predicted);
  const lo = Math.min(...all), hi = Math.max(...all);
  const c = card(grid, title, note, false);
  new Chart(mkCanvas(c), {
    type: "scatter",
    data: {
      datasets: [
        { type: "scatter", label: `Pixels (n=${sc.n.toLocaleString()})`, data: pts, backgroundColor: "rgba(26,115,232,0.35)", pointRadius: 2 },
        { type: "line", label: "1:1 line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: "#e53935", borderDash: [5,4], borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: opts(xL, yL, { dec })
  });
}

/* ── Stats table ── */
function statsTable(grid, rows) {
  if (!rows || !rows.length) return;
  const c = card(grid, "Raster Statistics Summary", null, true);
  const t = el("table", "ch-tbl");
  t.innerHTML = `<thead><tr>
    <th>Layer</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Valid Pixels</th>
  </tr></thead>`;
  const tb = el("tbody");
  rows.forEach(r => {
    if (!r.n) return;
    const isLst = r.name.includes("lst"), u = isLst ? " °C" : "", d = isLst ? 2 : 4;
    const tr = el("tr");
    tr.innerHTML = `
      <td class="tbl-name">${r.name.replace(/_/g," ")}</td>
      <td>${fmt(r.mean,d,u)}</td><td>${fmt(r.median,d,u)}</td>
      <td>${fmt(r.std,d)}</td><td>${fmt(r.min,d,u)}</td>
      <td>${fmt(r.max,d,u)}</td><td>${r.n.toLocaleString()}</td>`;
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  c.appendChild(t);
}

/* ── LULC table ── */
function lulcTable(grid, la, areaChange) {
  if (!la || !Object.keys(la["2004"] || {}).length) return;
  const c = card(grid, "LULC Class Areas & Change", null, true);
  const t = el("table", "ch-tbl");
  t.innerHTML = `<thead><tr>
    <th>Class</th><th>2004 Area (ha)</th><th>2004 %</th><th>2024 Area (ha)</th><th>2024 %</th><th>Δ ha</th><th>Δ %</th>
  </tr></thead>`;
  const tb = el("tbody");
  const ids = Array.from(new Set([...Object.keys(la["2004"]), ...Object.keys(la["2024"] || {})]));
  ids.forEach(id => {
    const a04 = la["2004"][id] || {}, a24 = (la["2024"] || {})[id] || {};
    const ac = (areaChange || {})[id] || {};
    const delta = ac.delta_ha, pct = ac.delta_percent;
    const sign = delta > 0 ? "+" : "";
    const cls  = delta > 0 ? "pos" : delta < 0 ? "neg" : "";
    const tr = el("tr");
    tr.innerHTML = `
      <td class="tbl-name">${a04.name || a24.name || "Class "+id}</td>
      <td>${fmt(a04.area_ha,2," ha")}</td><td>${fmt(a04.percent,1,"%")}</td>
      <td>${fmt(a24.area_ha,2," ha")}</td><td>${fmt(a24.percent,1,"%")}</td>
      <td class="${cls}">${typeof delta==="number" ? sign+fmt(delta,2," ha") : "—"}</td>
      <td class="${cls}">${typeof pct==="number"   ? sign+fmt(pct,1,"%")    : "—"}</td>`;
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  c.appendChild(t);
}

/* ══ MAIN ══ */
async function renderCharts() {
  const root   = document.getElementById("charts");
  const notice = document.getElementById("notice");
  const stats  = await loadStats();

  if (!stats || !(stats.rasters || []).length) {
    notice.style.display = "block";
    notice.textContent = "No statistics found. Run `python main.py` then reload.";
    return;
  }
  notice.style.display = "none";

  const derived = stats.derived || {};
  const change  = stats.change_summaries || {};
  const tr      = stats.trends || {};
  const ph      = stats.paired_histograms || {};
  const sc      = stats.cnn_scatter || {};
  const la      = stats.lulc_areas || {};
  const ltr     = tr.lulc || {};
  const cnn     = stats.cnn_lstm || {};

  /* 1 — Stats table */
  const g0 = addSection(root, "Raster Statistics");
  statsTable(g0, stats.rasters);

  /* 2 — Trends */
  const g1 = addSection(root, "Temporal Trends (2004 → 2024)");

  if (tr.ndvi?.years?.length >= 2) {
    const yrs = tr.ndvi.years.map(String);
    trendChart(g1,
      "NDVI Trend — Rajapeta Village",
      `Mean NDVI: ${fmt(tr.ndvi.mean[0],4)} (2004) → ${fmt(tr.ndvi.mean[1],4)} (2024)  |  Change: +${fmt(tr.ndvi.mean[1]-tr.ndvi.mean[0],4)}`,
      yrs,
      [
        { label: "Mean NDVI", data: tr.ndvi.mean,
          borderColor: "#2e7d32", backgroundColor: "rgba(46,125,50,0.10)",
          fill: true, tension: 0, pointRadius: 9, pointBackgroundColor: "#2e7d32",
          pointBorderColor: "#fff", pointBorderWidth: 3, borderWidth: 3 },
        { label: "Median NDVI", data: tr.ndvi.median,
          borderColor: "#f9a825", backgroundColor: "transparent",
          borderDash: [7,4], tension: 0, pointRadius: 7,
          pointBackgroundColor: "#f9a825", pointBorderColor: "#fff", pointBorderWidth: 2, borderWidth: 2 }
      ],
      "Year", "NDVI", { dec: 4 }
    );
  } else {
    emptyCard(g1, "NDVI Trend", "Only 2004 & 2024 available. Export 2009/2014/2019 from GEE to extend.");
  }

  if (tr.lst?.years?.length >= 2) {
    const yrs = tr.lst.years.map(String);
    trendChart(g1,
      "LST Trend — Rajapeta Village (°C)",
      `Mean LST: ${fmt(tr.lst.mean[0],2," °C")} (2004) → ${fmt(tr.lst.mean[1],2," °C")} (2024)  |  Change: ${fmt(tr.lst.mean[1]-tr.lst.mean[0],2," °C")}`,
      yrs,
      [
        { label: "Mean LST (°C)", data: tr.lst.mean,
          borderColor: "#c62828", backgroundColor: "rgba(198,40,40,0.10)",
          fill: true, tension: 0, pointRadius: 9, pointBackgroundColor: "#c62828",
          pointBorderColor: "#fff", pointBorderWidth: 3, borderWidth: 3 },
        { label: "Median LST (°C)", data: tr.lst.median,
          borderColor: "#1565c0", backgroundColor: "transparent",
          borderDash: [7,4], tension: 0, pointRadius: 7,
          pointBackgroundColor: "#1565c0", pointBorderColor: "#fff", pointBorderWidth: 2, borderWidth: 2 }
      ],
      "Year", "LST (°C)", { dec: 2, unit: " °C",
        yMin: Math.floor(Math.min(...tr.lst.mean,...tr.lst.median))-1,
        yMax: Math.ceil(Math.max(...tr.lst.mean,...tr.lst.median))+1 }
    );
  }

  if (ltr.years?.length >= 2) {
    const colors = ["#1565c0","#2e7d32","#c62828","#795548","#f9a825"];
    trendChart(g1,
      "LULC Class-Area Trend (ha)",
      "Area per class from classified GeoTIFFs.",
      ltr.years.map(String),
      Object.entries(ltr.classes || {}).map(([cid, name], i) => ({
        label: name,
        data: ltr.years.map(y => (ltr.class_areas_ha?.[String(y)] || {})[cid] ?? null),
        borderColor: colors[i%5], backgroundColor: "transparent",
        tension: 0, pointRadius: 7, pointBackgroundColor: colors[i%5],
        pointBorderColor: "#fff", pointBorderWidth: 2, borderWidth: 2
      })),
      "Year", "Area (ha)", { dec: 2, unit: " ha", zero: true }
    );
  }

  /* 3 — 2004 vs 2024 comparison */
  const g2 = addSection(root, "2004 vs 2024 Comparison");
  const n04 = rrow(stats,"ndvi_2004"), n24 = rrow(stats,"ndvi_2024");
  const l04 = rrow(stats,"lst_2004"),  l24 = rrow(stats,"lst_2024");

  if (n04 && n24) {
    const pct = derived.ndvi?.percent_change;
    barChart(g2, "Mean NDVI: 2004 vs 2024",
      typeof pct==="number" ? `Relative change: +${fmt(pct,1,"%")}` : "",
      ["2004","2024"],
      [{ label: "Mean NDVI", data: [n04.mean, n24.mean],
        backgroundColor: ["rgba(139,90,43,0.85)","rgba(46,125,50,0.85)"],
        borderWidth: 0, borderRadius: 6 }],
      "Mean NDVI", { dec: 4, yFmt: v => Number(v).toFixed(3) });
  }
  if (l04 && l24) {
    const d = derived.lst || {};
    barChart(g2, "Mean LST: 2004 vs 2024 (°C)",
      typeof d.mean_change_c==="number" ? `Mean change: ${fmt(d.mean_change_c,2," °C")}` : "",
      ["2004","2024"],
      [{ label: "Mean LST (°C)", data: [l04.mean, l24.mean],
        backgroundColor: ["rgba(198,40,40,0.85)","rgba(21,101,192,0.85)"],
        borderWidth: 0, borderRadius: 6 }],
      "LST (°C)", { dec: 2, unit: " °C",
        yMin: Math.floor(Math.min(l04.mean,l24.mean))-2,
        yMax: Math.ceil(Math.max(l04.mean,l24.mean))+2 });
  }

  /* 4 — Change detection (green/grey/red style) */
  const g3 = addSection(root, "Change Detection (2004 → 2024)");
  const nc = change.ndvi, lc = change.lst;

  if (nc) {
    changeBar(g3,
      "NDVI Change Classes",
      "Threshold: |ΔNDVI| > 0.05  |  Green = increase, Red = decrease",
      nc["Increase"]?.percent, nc["No significant change"]?.percent, nc["Decrease"]?.percent,
      "▲ Increase", "→ No change", "▼ Decrease",
      "% of valid pixels"
    );
  }
  if (lc) {
    changeBar(g3,
      "LST Change Classes",
      "Threshold: |ΔLST| > 1°C  |  Green = warming, Red = cooling",
      lc["Increase"]?.percent, lc["No significant change"]?.percent, lc["Decrease"]?.percent,
      "▲ Warming", "→ No change", "▼ Cooling",
      "% of valid pixels"
    );
  }

  /* change raster stats */
  const ncr = rrow(stats,"ndvi_change");
  if (ncr) {
    barChart(g3, "NDVI Change Raster (ΔNDVI)", "Mean / Median / Min / Max of pixel-wise change.",
      ["Mean","Median","Min","Max"],
      [{ label: "ΔNDVI", data: [ncr.mean,ncr.median,ncr.min,ncr.max],
        backgroundColor: ["#2e7d32","#66bb6a","#c62828","#1b5e20"], borderWidth: 0, borderRadius: 6 }],
      "ΔNDVI", { dec: 4 });
  }
  const lcr = rrow(stats,"lst_change");
  if (lcr) {
    barChart(g3, "LST Change Raster (Δ°C)", "Mean / Median / Min / Max of pixel-wise change.",
      ["Mean","Median","Min","Max"],
      [{ label: "ΔLST (°C)", data: [lcr.mean,lcr.median,lcr.min,lcr.max],
        backgroundColor: ["#c62828","#ef9a9a","#1565c0","#b71c1c"], borderWidth: 0, borderRadius: 6 }],
      "ΔLST (°C)", { dec: 2, unit: " °C" });
  }

  /* 5 — Distributions */
  const g4 = addSection(root, "Pixel Value Distributions");
  if (ph.ndvi?.counts_a?.length) pairedBar(g4, "NDVI Distribution: 2004 vs 2024", "Shared bin axis — directly comparable.", ph.ndvi, "NDVI", 3);
  if (ph.lst?.counts_a?.length)  pairedBar(g4, "LST Distribution: 2004 vs 2024 (°C)", "Shared bin axis — directly comparable.", ph.lst, "LST (°C)", 1);

  /* 6 — LULC */
  const g5 = addSection(root, "LULC Analysis");
  if (stats.lulc_available && la["2004"] && Object.keys(la["2004"]).length) {
    const ids    = Array.from(new Set([...Object.keys(la["2004"]), ...Object.keys(la["2024"]||{})]));
    const names  = ids.map(id => la["2004"][id]?.name || la["2024"]?.[id]?.name || `Class ${id}`);
    const colors = ["#1565c0","#2e7d32","#c62828","#795548","#f9a825"];
    barChart(g5, "LULC Area: 2004 vs 2024 (ha)", null, names,
      [
        { label: "2004", data: ids.map(id => la["2004"][id]?.area_ha ?? 0), backgroundColor: colors.map(c=>c+"cc"), borderWidth: 0, borderRadius: 4 },
        { label: "2024", data: ids.map(id => (la["2024"]||{})[id]?.area_ha ?? 0), backgroundColor: colors.map(c=>c+"55"), borderWidth: 2, borderColor: colors, borderRadius: 4 }
      ],
      "Area (ha)", { dec: 2, unit: " ha", zero: true }, false);
    lulcTable(g5, la, stats.lulc_area_change);
  } else {
    emptyCard(g5, "LULC Area Comparison", stats.lulc_note || "LULC data not available.");
  }

  /* 7 — CNN-LSTM */
  const g6 = addSection(root, "CNN-LSTM Reconstruction");
  if (sc.ndvi?.actual?.length) {
    scatterChart(g6, "CNN-LSTM: Actual vs Predicted NDVI",
      `MAE=${fmt(cnn.mae,4)} · RMSE=${fmt(cnn.rmse,4)} · n_test=${cnn.n_test}`,
      sc.ndvi, "Actual NDVI", "Predicted NDVI", 4);
  } else {
    emptyCard(g6, "CNN-LSTM: Actual vs Predicted NDVI", stats.cnn_note || "Run `python main.py` to train.");
  }
  if (sc.lst?.actual?.length) {
    scatterChart(g6, "CNN-LSTM: Actual vs Predicted LST (°C)",
      `MAE=${fmt(cnn.mae,4)} · RMSE=${fmt(cnn.rmse,4)} · n_test=${cnn.n_test}`,
      sc.lst, "Actual LST (°C)", "Predicted LST (°C)", 2);
  } else {
    emptyCard(g6, "CNN-LSTM: Actual vs Predicted LST", stats.cnn_note || "Run `python main.py` to train.");
  }
}

document.addEventListener("DOMContentLoaded", renderCharts);
