/* ═══════════════════════════════════════════════════════
   RAJAPETA WATERSHED — script.js
   Full-screen map | NDVI · LST · LULC on-map analysis
   ═══════════════════════════════════════════════════════ */

// ── 1. Boundary coords ───────────────────────────────────────
const COORDS = [
  [16.368712,78.021089],[16.366574,78.022436],[16.366611,78.024050],
  [16.365875,78.025284],[16.363715,78.025862],[16.362217,78.025553],
  [16.361880,78.026070],[16.362013,78.026756],[16.361708,78.026891],
  [16.361157,78.026350],[16.360873,78.026565],[16.360639,78.025628],
  [16.357653,78.025478],[16.357024,78.025846],[16.355900,78.025795],
  [16.347262,78.032489],[16.340451,78.036701],[16.337309,78.032642],
  [16.338165,78.016312],[16.338535,77.997197],[16.339891,77.995594],
  [16.342004,77.994258],[16.346133,77.990636],[16.348536,77.990199],
  [16.353571,77.990584],[16.355095,77.991089],[16.355450,77.990749],
  [16.355239,77.990666],[16.356204,77.990699],[16.356398,77.992443],
  [16.357479,77.992633],[16.358917,77.992318],[16.359462,77.992898],
  [16.360394,77.993185],[16.362935,77.992312],[16.363016,77.992945],
  [16.363488,77.993092],[16.365463,77.992728],[16.365383,77.993035],
  [16.366442,77.992404],[16.366124,77.995181],[16.365466,77.997355],
  [16.365427,77.999559],[16.364910,78.002218],[16.364836,78.011998],
  [16.368712,78.021089]
];

// ── 2. Analysis data (replace with your real GEE/Python values) ──
const DATA = {
  ndvi: {
    ndvi2004:   { min:-0.12, max:0.68, mean:0.31, delta:null,  label:'NDVI 2004',
                  insight:'Moderate vegetation cover in 2004. Agricultural land dominant.' },
    ndvi2024:   { min:-0.08, max:0.74, mean:0.38, delta:null,  label:'NDVI 2024',
                  insight:'Vegetation improved by 2024. Increased green cover observed.' },
    ndviChange: { min:-0.18, max:0.22, mean:0.07, delta:'+0.07', label:'NDVI Change 2004–2024',
                  insight:'Net positive change. Some areas show vegetation loss (red zones).' }
  },
  lst: {
    lst2004:   { min:26.4, max:42.1, mean:33.8, delta:null,   label:'LST 2004',
                 insight:'Mean surface temp 33.8°C in 2004. Moderate heat stress.' },
    lst2024:   { min:28.2, max:45.6, mean:36.5, delta:null,   label:'LST 2024',
                 insight:'Mean temp rose to 36.5°C by 2024. Urban expansion visible.' },
    lstChange: { min:-1.2, max:5.8,  mean:2.7,  delta:'+2.7°C', label:'LST Change 2004–2024',
                 insight:'Temperature increased by ~2.7°C on average over 20 years.' }
  },
  lulc: {
    lulc2004: {
      label:'LULC 2004',
      w:4.2, a:52.1, f:18.6, b:8.3,  n:16.8
    },
    lulc2024: {
      label:'LULC 2024',
      w:3.8, a:44.7, f:15.2, b:14.9, n:21.4
    }
  }
};

// ── 3. Map init ───────────────────────────────────────────────
const map = L.map('map', {
  center: [16.353, 78.013],
  zoom: 13,
  zoomControl: false,
  zoomSnap: 0.25
});

// Base layers — Street map as DEFAULT (not satellite)
const baseStreet = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Tiles &copy; Esri', maxZoom: 19 }
).addTo(map);   // ← default

const baseTopo = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Tiles &copy; Esri', maxZoom: 19 }
);

const baseSat = L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  { attribution: 'Tiles &copy; Esri', maxZoom: 19 }
);

// ── 4. Rajapeta boundary ─────────────────────────────────────
const boundary = L.polygon(COORDS, {
  color: '#ff5a36', weight: 2.5, dashArray: '7,5',
  lineJoin: 'round', fillColor: '#3b82f6', fillOpacity: 0.12
}).addTo(map);

boundary.bindPopup(`
  <div class="pu-head">📍 Rajapeta Village</div>
  <div class="pu-body">
    <div class="pu-row"><span class="pu-key">Mandal</span><span>Wanaparthy</span></div>
    <div class="pu-row"><span class="pu-key">State</span><span>Telangana, India</span></div>
    <div class="pu-row"><span class="pu-key">Study</span><span>Watershed Development</span></div>
    <div><span class="pu-badge">🌊 Remote Sensing Study</span></div>
  </div>`, { maxWidth: 220 });

boundary.on('mouseover', function(){ this.setStyle({ fillOpacity:0.30, weight:3.5 }); });
boundary.on('mouseout',  function(){ this.setStyle({ fillOpacity:0.12, weight:2.5 }); });

// ── 5. Overlay polygon factory ───────────────────────────────
// Each "layer" is a styled polygon fill simulating a raster overlay.
// Replace with real GeoTIFF layers when available.
function makeOverlay(fillColor, opacity) {
  return L.polygon(COORDS, {
    stroke: false, fillColor, fillOpacity: opacity
  });
}

const OVERLAYS = {
  ndvi2004:   makeOverlay('#52b788', 0),
  ndvi2024:   makeOverlay('#1a9850', 0),
  ndviChange: makeOverlay('#fee08b', 0),
  lst2004:    makeOverlay('#f46d43', 0),
  lst2024:    makeOverlay('#d73027', 0),
  lstChange:  makeOverlay('#74add1', 0),
  lulc2004:   makeOverlay('#78c679', 0),
  lulc2024:   makeOverlay('#238b45', 0),
  lulcChange: makeOverlay('#e31a1c', 0)
};

let activeOverlay = null;

function showOverlay(key, opacity) {
  if (activeOverlay) map.removeLayer(activeOverlay);
  const lyr = OVERLAYS[key];
  lyr.setStyle({ fillOpacity: opacity });
  lyr.addTo(map);
  // keep boundary on top
  boundary.bringToFront();
  activeOverlay = lyr;
}

// ── 6. Leaflet controls ──────────────────────────────────────
L.control.zoom({ position: 'topleft' }).addTo(map);
L.control.scale({ position: 'bottomright', imperial: false }).addTo(map);

L.control.layers(
  { 'Street Map': baseStreet, 'Topographic': baseTopo, 'Satellite': baseSat },
  { 'Rajapeta Boundary': boundary },
  { position: 'topright', collapsed: true }
).addTo(map);

map.fitBounds(boundary.getBounds(), { padding: [60, 60] });

// ── 7. Chart.js setup ────────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Segoe UI', Arial, sans-serif";
Chart.defaults.font.size = 10;

const COPTS = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { labels: { boxWidth: 9, padding: 6 } } },
  scales: {
    x: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { font: { size: 9 } } },
    y: { grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 9 } } }
  }
};

const ndviChart = new Chart(document.getElementById('ndvi-chart'), {
  type: 'bar',
  data: {
    labels: ['2004', '2024'],
    datasets: [{
      label: 'Mean NDVI',
      data: [0.31, 0.38],
      backgroundColor: ['rgba(26,152,80,0.65)', 'rgba(26,152,80,1)'],
      borderRadius: 4, borderWidth: 0
    }]
  },
  options: { ...COPTS, plugins: { ...COPTS.plugins,
    title: { display: true, text: 'Mean NDVI — 2004 vs 2024', font: { size: 10 }, color: '#1e3a5c' }
  }}
});

const lstChart = new Chart(document.getElementById('lst-chart'), {
  type: 'bar',
  data: {
    labels: ['2004', '2024'],
    datasets: [{
      label: 'Mean LST (°C)',
      data: [33.8, 36.5],
      backgroundColor: ['rgba(244,109,67,0.7)', 'rgba(215,48,39,1)'],
      borderRadius: 4, borderWidth: 0
    }]
  },
  options: { ...COPTS, plugins: { ...COPTS.plugins,
    title: { display: true, text: 'Mean LST °C — 2004 vs 2024', font: { size: 10 }, color: '#1e3a5c' }
  }}
});

const lulcChart = new Chart(document.getElementById('lulc-chart'), {
  type: 'bar',
  data: {
    labels: ['Water', 'Agri', 'Forest', 'Built', 'Barren'],
    datasets: [
      { label: '2004', data: [4.2,52.1,18.6,8.3,16.8],
        backgroundColor: ['#1a6faf','#78c679','#238b45','#e31a1c','#d4b483'],
        borderRadius: 3, borderWidth: 0 },
      { label: '2024', data: [3.8,44.7,15.2,14.9,21.4],
        backgroundColor: ['#1a6fafaa','#78c679aa','#238b45aa','#e31a1caa','#d4b483aa'],
        borderRadius: 3, borderWidth: 0 }
    ]
  },
  options: { ...COPTS, plugins: { ...COPTS.plugins,
    title: { display: true, text: 'LULC Area % — 2004 vs 2024', font: { size: 10 }, color: '#1e3a5c' }
  }}
});

// ── 8. Legend updater ────────────────────────────────────────
function setLegend(title, rows) {
  document.getElementById('leg-title').textContent = title;
  document.getElementById('leg-body').innerHTML = rows.map(r => {
    if (r.ramp) return `
      <div class="leg-row">
        <span class="leg-ramp" style="background:${r.ramp}"></span>
        <span class="leg-txt">${r.label}</span>
      </div>`;
    return `
      <div class="leg-row">
        <span class="leg-swatch" style="background:${r.color}"></span>
        <span class="leg-txt">${r.label}</span>
      </div>`;
  }).join('');
}

// ── 9. NDVI logic ────────────────────────────────────────────
let curNdvi = 'ndvi2004';

function applyNdvi(key) {
  curNdvi = key;
  const d = DATA.ndvi[key];
  const op = +document.getElementById('ndvi-op').value / 100;
  showOverlay(key, op);

  document.getElementById('ndvi-min').textContent   = d.min.toFixed(2);
  document.getElementById('ndvi-max').textContent   = d.max.toFixed(2);
  document.getElementById('ndvi-mean').textContent  = d.mean.toFixed(2);
  document.getElementById('ndvi-delta').textContent =
    d.delta ?? ('+' + (DATA.ndvi.ndvi2024.mean - DATA.ndvi.ndvi2004.mean).toFixed(2));

  if (key === 'ndviChange') {
    setLegend('NDVI Change 2004–2024', [
      { color: '#d73027', label: 'Vegetation Loss' },
      { color: '#ffffbf', label: 'No Change' },
      { color: '#1a9850', label: 'Vegetation Gain' }
    ]);
  } else {
    setLegend(d.label, [
      { ramp: 'linear-gradient(90deg,#8c510a,#d8b365,#f6e8c3,#c7eae5,#5ab4ac,#01665e)',
        label: 'Low → High NDVI' }
    ]);
  }
}

document.querySelectorAll('#ndvi-seg .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#ndvi-seg .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyNdvi(btn.dataset.key);
  });
});

document.getElementById('ndvi-op').addEventListener('input', function () {
  document.getElementById('ndvi-opval').textContent = this.value + '%';
  if (activeOverlay === OVERLAYS[curNdvi])
    activeOverlay.setStyle({ fillOpacity: +this.value / 100 });
});

// ── 10. LST logic ────────────────────────────────────────────
let curLst = 'lst2004';

function applyLst(key) {
  curLst = key;
  const d = DATA.lst[key];
  const op = +document.getElementById('lst-op').value / 100;
  showOverlay(key, op);

  document.getElementById('lst-min').textContent   = d.min.toFixed(1);
  document.getElementById('lst-max').textContent   = d.max.toFixed(1);
  document.getElementById('lst-mean').textContent  = d.mean.toFixed(1);
  document.getElementById('lst-delta').textContent =
    d.delta ?? ('+' + (DATA.lst.lst2024.mean - DATA.lst.lst2004.mean).toFixed(1));

  if (key === 'lstChange') {
    setLegend('LST Change 2004–2024', [
      { color: '#313695', label: 'Cooling' },
      { color: '#ffffbf', label: 'No Change' },
      { color: '#a50026', label: 'Warming' }
    ]);
  } else {
    setLegend(d.label, [
      { ramp: 'linear-gradient(90deg,#313695,#74add1,#fee090,#f46d43,#a50026)',
        label: 'Cool → Hot (°C)' }
    ]);
  }
}

document.querySelectorAll('#lst-seg .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#lst-seg .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyLst(btn.dataset.key);
  });
});

document.getElementById('lst-op').addEventListener('input', function () {
  document.getElementById('lst-opval').textContent = this.value + '%';
  if (activeOverlay === OVERLAYS[curLst])
    activeOverlay.setStyle({ fillOpacity: +this.value / 100 });
});

// ── 11. LULC logic ───────────────────────────────────────────
let curLulc = 'lulc2004';

function applyLulc(key) {
  curLulc = key;
  const op = +document.getElementById('lulc-op').value / 100;
  showOverlay(key, op);

  const d04 = DATA.lulc.lulc2004;
  const d24 = DATA.lulc.lulc2024;

  // Always show both years + delta in table
  const fmt = v => v.toFixed(1) + '%';
  const delta = (a, b) => { const d = (b-a).toFixed(1); return (d>0?'+':'')+d+'%'; };

  document.getElementById('lc04-w').textContent = fmt(d04.w);
  document.getElementById('lc24-w').textContent = fmt(d24.w);
  document.getElementById('lcd-w').textContent  = delta(d04.w, d24.w);

  document.getElementById('lc04-a').textContent = fmt(d04.a);
  document.getElementById('lc24-a').textContent = fmt(d24.a);
  document.getElementById('lcd-a').textContent  = delta(d04.a, d24.a);

  document.getElementById('lc04-f').textContent = fmt(d04.f);
  document.getElementById('lc24-f').textContent = fmt(d24.f);
  document.getElementById('lcd-f').textContent  = delta(d04.f, d24.f);

  document.getElementById('lc04-b').textContent = fmt(d04.b);
  document.getElementById('lc24-b').textContent = fmt(d24.b);
  document.getElementById('lcd-b').textContent  = delta(d04.b, d24.b);

  document.getElementById('lc04-n').textContent = fmt(d04.n);
  document.getElementById('lc24-n').textContent = fmt(d24.n);
  document.getElementById('lcd-n').textContent  = delta(d04.n, d24.n);

  setLegend('LULC — ' + DATA.lulc[key].label, [
    { color: '#1a6faf', label: 'Water Body' },
    { color: '#78c679', label: 'Agriculture' },
    { color: '#238b45', label: 'Forest / Scrub' },
    { color: '#e31a1c', label: 'Built-up' },
    { color: '#d4b483', label: 'Barren Land' }
  ]);
}

document.querySelectorAll('#lulc-seg .seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#lulc-seg .seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyLulc(btn.dataset.key);
  });
});

document.getElementById('lulc-op').addEventListener('input', function () {
  document.getElementById('lulc-opval').textContent = this.value + '%';
  if (activeOverlay === OVERLAYS[curLulc])
    activeOverlay.setStyle({ fillOpacity: +this.value / 100 });
});

// ── 12. Tab switching (titlebar tabs) ────────────────────────
const PANELS = { ndvi: 'fp-ndvi', lst: 'fp-lst', lulc: 'fp-lulc' };

document.querySelectorAll('.tb-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;

    // Update tab button styles
    document.querySelectorAll('.tb-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Show correct float panel
    document.querySelectorAll('.float-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(PANELS[tab]).classList.add('active');

    // Remove any active overlay when switching tabs
    if (activeOverlay) { map.removeLayer(activeOverlay); activeOverlay = null; }

    // Auto-apply default year for the tab
    if (tab === 'ndvi') applyNdvi(curNdvi);
    if (tab === 'lst')  applyLst(curLst);
    if (tab === 'lulc') applyLulc(curLulc);
  });
});

// ── 13. Init on load ─────────────────────────────────────────
applyNdvi('ndvi2004');
applyLulc('lulc2004');   // pre-fill LULC table silently
applyNdvi('ndvi2004');   // restore NDVI as default active tab
