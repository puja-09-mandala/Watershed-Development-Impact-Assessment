async function fetchJSON(f) {
  try { const r = await fetch(f); return r.ok ? await r.json() : null; } catch { return null; }
}
async function fetchText(f) {
  try { const r = await fetch(f); return r.ok ? await r.text() : null; } catch { return null; }
}
function fmt(v, d = 4, u = "") {
  return typeof v === "number" && isFinite(v) ? v.toFixed(d) + u : "n/a";
}
function rrow(stats, name) {
  return (stats.rasters || []).find(r => r.name === name && r.n > 0) || null;
}
function md2html(src) {
  if (!src) return "";
  return src
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/^### (.*)$/gm, "<h3>$1</h3>")
    .replace(/^## (.*)$/gm, "<h2>$1</h2>")
    .replace(/^# (.*)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^- (.*)$/gm, "<li>$1</li>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/^/, "<p>").concat("</p>");
}

function statsTable(stats, names) {
  const rows = names.map(n => rrow(stats, n)).filter(Boolean).map(r => {
    const isLst = r.name.startsWith("lst"), d = isLst ? 2 : 4, u = isLst ? " °C" : "";
    return `<tr>
      <td>${r.name.replace(/_/g," ")}</td>
      <td>${fmt(r.mean,d,u)}</td><td>${fmt(r.median,d,u)}</td>
      <td>${fmt(r.min,d,u)}</td><td>${fmt(r.max,d,u)}</td>
      <td>${fmt(r.std,d)}</td><td>${Number(r.n).toLocaleString()}</td>
    </tr>`;
  }).join("");
  if (!rows) return "<p><em>Statistics not available.</em></p>";
  return `<table class="s-tbl"><thead><tr>
    <th>Layer</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std</th><th>Valid px</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function changeDetectionCard(stats) {
  const nc = (stats.change_summaries || {}).ndvi;
  const lc = (stats.change_summaries || {}).lst;
  const lu = (stats.change_summaries || {}).lulc;

  function chgTable(c, headCls, labels) {
    if (!c) return "<p><em>Not available.</em></p>";
    const entries = Object.entries(c);
    const rows = entries.map(([k, v]) => {
      const key = k.toLowerCase();
      const rowCls = key.includes("increase") || key.includes("class change") ? "c-row-inc"
                   : key.includes("decrease") ? "c-row-dec" : "c-row-stb";
      return `<tr class="${rowCls}">
        <td>${labels[k] || k}</td>
        <td style="text-align:right">${Number(v.pixels).toLocaleString()}</td>
        <td style="text-align:right">${fmt(v.percent,2,"%")}</td>
      </tr>`;
    }).join("");
    return `<table class="c-tbl"><thead class="${headCls}"><tr>
      <th>Class</th><th style="text-align:right">Pixels</th><th style="text-align:right">%</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
  }

  const ndviLabels = { "Increase": "▲ Increase (ΔNDVI > +0.05)", "No significant change": "→ No significant change", "Decrease": "▼ Decrease (ΔNDVI < −0.05)" };
  const lstLabels  = { "Increase": "▲ Warming (ΔLST > +1°C)", "No significant change": "→ No significant change", "Decrease": "▼ Cooling (ΔLST < −1°C)" };
  const lulcLabels = { "Class change": "↔ Class changed", "No class change": "→ No class change" };

  // summary bars
  function bars(c, incKey, decKey, stbKey) {
    if (!c) return "";
    const inc = c[incKey] || {}, stb = c[stbKey] || {}, dec = c[decKey] || {};
    return `<div class="chg-bar-wrap">
      <div class="chg-bar-item chg-inc"><div class="val">${fmt(inc.percent,1,"%")}</div><div class="lbl">${incKey}</div></div>
      <div class="chg-bar-item chg-stb"><div class="val">${fmt(stb.percent,1,"%")}</div><div class="lbl">No change</div></div>
      <div class="chg-bar-item chg-dec"><div class="val">${fmt(dec.percent,1,"%")}</div><div class="lbl">${decKey}</div></div>
    </div>`;
  }

  return `
    <p style="font-size:12px;color:#666;margin-bottom:14px">NDVI change = NDVI 2024 − NDVI 2004 &nbsp;|&nbsp; LST change = LST 2024 − LST 2004 &nbsp;|&nbsp; Thresholds: NDVI ±0.05, LST ±1°C</p>

    <p style="font-size:12px;font-weight:700;color:#2e7d32;margin:0 0 6px">NDVI Change</p>
    ${bars(nc, "Increase", "Decrease", "No significant change")}
    ${chgTable(nc, "c-tbl-head-ndvi", ndviLabels)}

    <p style="font-size:12px;font-weight:700;color:#c62828;margin:16px 0 6px">LST Change</p>
    ${bars(lc, "Increase", "Decrease", "No significant change")}
    ${chgTable(lc, "c-tbl-head-lst", lstLabels)}

    <p style="font-size:12px;font-weight:700;color:#1565c0;margin:16px 0 6px">LULC Change</p>
    ${chgTable(lu, "c-tbl-head-lulc", lulcLabels)}`;
}

function lulcTable(stats) {
  const la = stats.lulc_areas || {};
  const ac = stats.lulc_area_change || {};
  const hasLulc = stats.lulc_available && Object.keys(la["2004"] || {}).length;
  if (!hasLulc) return `<p style="color:#888;font-size:13px">${stats.lulc_note || "LULC data not available yet."}</p>`;
  const ids = Array.from(new Set([...Object.keys(la["2004"]||{}), ...Object.keys(la["2024"]||{})]));
  const rows = ids.map(id => {
    const a = la["2004"][id] || {}, b = (la["2024"]||{})[id] || {}, c = ac[id] || {};
    const delta = c.delta_ha, pct = c.delta_percent;
    const sign = delta > 0 ? "+" : "";
    const cls = delta > 0 ? "td-pos" : delta < 0 ? "td-neg" : "";
    return `<tr>
      <td>${a.name || b.name || "Class "+id}</td>
      <td>${fmt(a.area_ha,2," ha")}</td><td>${fmt(a.percent,1,"%")}</td>
      <td>${fmt(b.area_ha,2," ha")}</td><td>${fmt(b.percent,1,"%")}</td>
      <td class="${cls}">${typeof delta==="number" ? sign+fmt(delta,2," ha") : "—"}</td>
      <td class="${cls}">${typeof pct==="number" ? sign+fmt(pct,1,"%") : "—"}</td>
    </tr>`;
  }).join("");
  return `<table class="lulc-tbl"><thead><tr>
    <th>Class</th><th>2004 ha</th><th>2004 %</th><th>2024 ha</th><th>2024 %</th><th>Δ ha</th><th>Δ %</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function keyObs(stats) {
  const obs = [];
  const n04 = rrow(stats,"ndvi_2004"), n24 = rrow(stats,"ndvi_2024");
  const l04 = rrow(stats,"lst_2004"),  l24 = rrow(stats,"lst_2024");
  const d = stats.derived || {};
  if (n04 && n24) {
    const dir = n24.mean > n04.mean ? "higher" : n24.mean < n04.mean ? "lower" : "unchanged";
    const pct = typeof d.ndvi?.percent_change === "number" ? ` (${fmt(d.ndvi.percent_change,1,"%")} relative change)` : "";
    obs.push(`Mean NDVI 2024 <strong>${fmt(n24.mean,4)}</strong> is <strong>${dir}</strong> than 2004 <strong>${fmt(n04.mean,4)}</strong> — Δ = <strong>${fmt(d.ndvi?.mean_change,4)}</strong>${pct}.`);
  }
  if (l04 && l24) {
    const dir = l24.mean > l04.mean ? "warmer" : l24.mean < l04.mean ? "cooler" : "unchanged";
    obs.push(`Mean LST 2024 <strong>${fmt(l24.mean,2," °C")}</strong> is <strong>${dir}</strong> than 2004 <strong>${fmt(l04.mean,2," °C")}</strong> — Δ = <strong>${fmt(d.lst?.mean_change_c,2," °C")}</strong>.`);
  }
  const nc = (stats.change_summaries||{}).ndvi;
  if (nc) obs.push(`NDVI: <strong>${fmt(nc["Increase"]?.percent,1,"%")}</strong> pixels increased, <strong>${fmt(nc["Decrease"]?.percent,1,"%")}</strong> decreased, <strong>${fmt(nc["No significant change"]?.percent,1,"%")}</strong> no significant change (threshold ±0.05).`);
  const lc = (stats.change_summaries||{}).lst;
  if (lc) obs.push(`LST: <strong>${fmt(lc["Decrease"]?.percent,1,"%")}</strong> pixels cooled, <strong>${fmt(lc["Increase"]?.percent,1,"%")}</strong> warmed, <strong>${fmt(lc["No significant change"]?.percent,1,"%")}</strong> no significant change (threshold ±1°C).`);
  const lu = (stats.change_summaries||{}).lulc;
  if (lu) obs.push(`LULC: <strong>${fmt(lu["Class change"]?.percent,1,"%")}</strong> pixels changed class, <strong>${fmt(lu["No class change"]?.percent,1,"%")}</strong> unchanged.`);
  const cnn = stats.cnn_lstm || {};
  obs.push(cnn.trained
    ? `CNN-LSTM reconstruction: MAE <strong>${fmt(cnn.mae,4)}</strong>, RMSE <strong>${fmt(cnn.rmse,4)}</strong> (n_test=${cnn.n_test}). Prototype result — not a validated forecast.`
    : "CNN-LSTM was not trained in this run.");
  return `<ul class="obs-list">${obs.map(o=>`<li>${o}</li>`).join("")}</ul>`;
}

function envInterp(stats) {
  const d = stats.derived || {};
  const parts = [];
  if (d.ndvi) parts.push(`The NDVI difference observed between 2004 and 2024 is a change <em>observed during the study period</em>. Within the watershed-development context of Rajapeta, plausible associated factors include rainfall variability, irrigation availability, cropping patterns, land-management practices, and seasonal differences in image acquisition. This analysis does not attribute the change causally to watershed development works.`);
  if (d.lst) parts.push(`The surface-temperature difference (${fmt(d.lst.mean_change_c,2," °C")} mean LST) observed during the study period may be associated with vegetation-cover change, soil-moisture conditions, land-use change, regional climate variability, and sensor/season differences between Landsat 5 (2004) and Landsat 8 (2024).`);
  if (!parts.length) parts.push("Insufficient computed statistics for an environmental interpretation.");
  parts.push("These statements are interpretations of observed results, not evidence of causation.");
  return parts.map(p=>`<p>${p}</p>`).join("");
}

function card(title, body) {
  return `<div class="r-card"><div class="r-card-title">${title}</div>${body}</div>`;
}

function renderReport(stats, manifest, interpretation) {
  const sa = stats.study_area || {};
  const html =
    card("Study Area",
      `<div class="meta-row">
        <span><strong>Village:</strong> ${sa.village || "Rajapeta (Rajapet)"}</span>
        <span><strong>Mandal:</strong> ${sa.mandal || "Wanaparthy Mandal"}</span>
        <span><strong>District:</strong> ${sa.district || "Wanaparthy"}</span>
        <span><strong>State:</strong> ${sa.state || "Telangana, India"}</span>
        <span><strong>Period:</strong> ${sa.period || "2004–2024"}</span>
        <span><strong>Sensors:</strong> Landsat 5 (2004) &amp; Landsat 8 (2024), 30 m</span>
      </div>`) +

    card("1. NDVI Analysis", statsTable(stats, ["ndvi_2004","ndvi_2024","ndvi_change"])) +
    card("2. LST Analysis",  statsTable(stats, ["lst_2004","lst_2024","lst_change"])) +
    card("3. LULC Analysis", lulcTable(stats)) +
    card("4. Change Detection", changeDetectionCard(stats)) +
    card("5. Key Observations", keyObs(stats)) +

    card("6. AI-Generated Interpretation (Mistral)",
      `<div class="ai-box">${
        md2html(interpretation) ||
        "<p><em>Click <strong>Regenerate AI Interpretation</strong> to generate live Mistral analysis. Without a key, the report uses computed statistics only.</em></p>"
      }</div>`) +

    card("7. Environmental Interpretation",
      `<div style="font-size:13px;color:#444;line-height:1.75">${envInterp(stats)}</div>`) +

    card("8. Limitations",
      `<ul class="lim-list">
        <li>Only two dates (2004, 2024) — no validated multi-year forecasting is claimed.</li>
        <li>Cloud-free median composites may mix phenological and seasonal conditions.</li>
        <li>Causal attribution of any change to watershed development is <strong>not</strong> claimed.</li>
        <li>LULC accuracy is reported only when independent reference samples exist.</li>
        <li>Landsat 5 and Landsat 8 sensors differ; cross-sensor consistency is approximate.</li>
        <li>CNN-LSTM patches share spatial context; metrics are indicative, not independent-validation scores.</li>
      </ul>`);

  document.getElementById("report-body").innerHTML = html;
}

async function loadReport() {
  const [stats, manifest, interpretation] = await Promise.all([
    fetchJSON("data/stats.json"),
    fetchJSON("data/manifest.json"),
    fetchText("data/llm_interpretation.md"),
  ]);
  const notice = document.getElementById("notice");
  if (!stats || !(stats.rasters || []).length) {
    notice.style.display = "block";
    notice.textContent = "No computed statistics found. Run `python main.py` then reload.";
    document.getElementById("report-body").innerHTML = "<p style='color:#888'>No analysis results yet.</p>";
    return;
  }
  notice.style.display = "none";
  renderReport(stats, manifest, interpretation);
}

async function initToolbar() {
  const statusEl = document.getElementById("llm-status");
  const btn = document.getElementById("regenerate");
  if (!statusEl || !btn) return;
  try {
    const res = await fetch("/api/llm/status");
    const j = await res.json();
    statusEl.innerHTML = j.key_configured
      ? "Mistral API: <strong style='color:#2e7d32'>configured ✓</strong>"
      : "Mistral API: <strong style='color:#c62828'>not configured</strong> — set MISTRAL_API_KEY in .env and restart serve.py";
  } catch { statusEl.textContent = "Server status unavailable."; }

  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Generating…";
    statusEl.textContent = "Requesting AI interpretation…";
    try {
      const res = await fetch("/api/llm/interpret", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      });
      const j = await res.json();
      if (!res.ok || j.error) throw new Error(j.error || "request failed");
      const [stats, manifest] = await Promise.all([fetchJSON("data/stats.json"), fetchJSON("data/manifest.json")]);
      if (stats) renderReport(stats, manifest, j.text);
      statusEl.innerHTML = j.source === "mistral"
        ? "Regenerated with <strong style='color:#2e7d32'>live Mistral ✓</strong>"
        : "Regenerated with <strong style='color:#e65100'>statistics fallback</strong> — set MISTRAL_API_KEY in .env and restart serve.py";
    } catch (e) {
      statusEl.textContent = "Generation failed: " + e.message;
    } finally {
      btn.disabled = false; btn.textContent = "⟳ Regenerate AI Interpretation";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => { loadReport(); initToolbar(); });
