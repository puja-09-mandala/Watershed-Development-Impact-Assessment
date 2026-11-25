/**
 * Google Earth Engine — Rajapeta Village Watershed Study (2004–2024)
 * Wanaparthy Mandal, Wanaparthy District, Telangana, India
 *
 * Landsat 5 TM  (2004): LANDSAT/LT05/C02/T1_L2
 * Landsat 8 OLI (2024): LANDSAT/LC08/C02/T1_L2
 *
 * SETUP: Replace 'users/YOUR_USERNAME/Rajapeta_Village_Boundary' with your asset path.
 * Run in GEE Code Editor → Tasks tab → download GeoTIFFs → place in data/raw/
 *
 * MISTRAL KEY: Set in .env file at line 1 → MISTRAL_API_KEY=your_key_here
 * (Do NOT paste the key in this script)
 */

// ─── 1. STUDY AREA ────────────────────────────────────────────────────────────
var STUDY_AREA = ee.FeatureCollection('users/YOUR_USERNAME/Rajapeta_Village_Boundary');
var roi = STUDY_AREA.geometry();
Map.centerObject(roi, 13);

// ─── 2. CLOUD MASKING ─────────────────────────────────────────────────────────
function maskLandsatSR(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 1).eq(0)  // dilated cloud
    .and(qa.bitwiseAnd(1 << 2).eq(0))      // cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))      // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0));     // cloud shadow
  return image.updateMask(mask);
}

function scaleOptical(image) {
  return image.multiply(0.0000275).add(-0.2);
}

function scaleThermalC(image) {
  // Collection 2 Level-2: DN → Kelvin → Celsius
  return image.multiply(0.00341802).add(149.0).subtract(273.15);
}

// ─── 3. MEDIAN COMPOSITES ─────────────────────────────────────────────────────
function medianComposite(collectionId, start, end, opticalBands, thermalBand) {
  var col = ee.ImageCollection(collectionId)
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskLandsatSR);
  var optical = scaleOptical(col.select(opticalBands).median()).clip(roi);
  var thermal = scaleThermalC(col.select(thermalBand).median()).clip(roi);
  return {optical: optical, thermal: thermal};
}

var ls5 = medianComposite(
  'LANDSAT/LT05/C02/T1_L2', '2004-01-01', '2004-12-31',
  ['SR_B3', 'SR_B4'], 'ST_B6'
);
var ls8 = medianComposite(
  'LANDSAT/LC08/C02/T1_L2', '2024-01-01', '2024-12-31',
  ['SR_B4', 'SR_B5'], 'ST_B10'
);

// ─── 4. NDVI ──────────────────────────────────────────────────────────────────
var ndvi2004   = ls5.optical.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI');
var ndvi2024   = ls8.optical.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
var ndviChange = ndvi2024.subtract(ndvi2004).rename('NDVI_Change');

// ─── 5. LST ───────────────────────────────────────────────────────────────────
var lst2004   = ls5.thermal.rename('LST');
var lst2024   = ls8.thermal.rename('LST');
var lstChange = lst2024.subtract(lst2004).rename('LST_Change');

// ─── 6. NDWI ──────────────────────────────────────────────────────────────────
// Landsat 5: Green=SR_B2, NIR=SR_B4 | Landsat 8: Green=SR_B3, NIR=SR_B5
var ndwi2004 = ls5.optical.normalizedDifference(['SR_B3', 'SR_B4']).rename('NDWI');
var ndwi2024 = ls8.optical.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');

// ─── 7. LULC THRESHOLD CLASSIFICATION ────────────────────────────────────────
// Classes: 1=Water, 2=Vegetation, 3=Built-up, 4=Barren, 5=Cropland
// Same thresholds used in src/lulc.py (Mahbubnagar-derived conventions)
function classifyLULC(ndvi, lst) {
  var water      = ndvi.lt(0.05);
  var vegetation = ndvi.gte(0.25).and(lst.lt(38));
  var builtup    = ndvi.lt(0.15).and(lst.gte(38));
  var barren     = ndvi.gte(0.05).and(ndvi.lt(0.15)).and(lst.gte(35));

  return ee.Image(0)
    .where(water,      1)
    .where(vegetation, 2)
    .where(builtup,    3)
    .where(barren,     4)
    .where(ee.Image(0).eq(0).and(water.not()).and(vegetation.not())
                             .and(builtup.not()).and(barren.not()), 5)
    .clip(roi)
    .rename('LULC');
}

var lulc2004   = classifyLULC(ndvi2004, lst2004);
var lulc2024   = classifyLULC(ndvi2024, lst2024);
var lulcChange = lulc2024.neq(lulc2004).rename('LULC_Change');

// ─── 8. STATISTICS (printed to Console) ──────────────────────────────────────
print('=== NDVI Statistics ===');
print('NDVI 2004 mean:', ndvi2004.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30, maxPixels: 1e9}));
print('NDVI 2024 mean:', ndvi2024.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30, maxPixels: 1e9}));
print('=== LST Statistics ===');
print('LST 2004 mean (°C):', lst2004.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30, maxPixels: 1e9}));
print('LST 2024 mean (°C):', lst2024.reduceRegion({reducer: ee.Reducer.mean(), geometry: roi, scale: 30, maxPixels: 1e9}));

// ─── 9. MAP LAYERS ────────────────────────────────────────────────────────────

// NDVI layers
Map.addLayer(ndvi2004, {
  min: -1, max: 1,
  palette: ['white', 'yellow', 'lightgreen', 'green', 'darkgreen']
}, 'NDVI 2004', true);

Map.addLayer(ndvi2024, {
  min: -1, max: 1,
  palette: ['white', 'yellow', 'lightgreen', 'green', 'darkgreen']
}, 'NDVI 2024', false);

Map.addLayer(ndviChange, {
  min: -0.5, max: 0.5,
  palette: ['red', 'orange', 'white', 'lightgreen', 'darkgreen']
}, 'NDVI Change (2024−2004)', false);

// LST layers
Map.addLayer(lst2004, {
  min: 20, max: 50,
  palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
}, 'LST 2004 (°C)', false);

Map.addLayer(lst2024, {
  min: 20, max: 50,
  palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
}, 'LST 2024 (°C)', false);

Map.addLayer(lstChange, {
  min: -10, max: 10,
  palette: ['blue', 'cyan', 'white', 'yellow', 'orange', 'red']
}, 'LST Change (2024−2004) °C', false);

// NDWI layers
Map.addLayer(ndwi2004, {
  min: -1, max: 1,
  palette: ['brown', 'yellow', 'white', 'lightblue', 'blue']
}, 'NDWI 2004', false);

Map.addLayer(ndwi2024, {
  min: -1, max: 1,
  palette: ['brown', 'yellow', 'white', 'lightblue', 'blue']
}, 'NDWI 2024', false);

// LULC layers — discrete palette: Water, Vegetation, Built-up, Barren, Cropland
var lulcVis = {min: 1, max: 5, palette: ['4682B4', '228B22', 'B22222', 'C2B280', '9ACD32']};
Map.addLayer(lulc2004,   lulcVis, 'LULC 2004 (threshold)', false);
Map.addLayer(lulc2024,   lulcVis, 'LULC 2024 (threshold)', false);
Map.addLayer(lulcChange, {min: 0, max: 1, palette: ['white', 'red']}, 'LULC Change', false);

// Boundary on top (always visible, red outline)
Map.addLayer(
  ee.Image().paint(STUDY_AREA, 0, 2),
  {palette: ['FF0000']},
  'Rajapeta Boundary'
);

// ─── 10. EXPORT GeoTIFFs ──────────────────────────────────────────────────────
function exportRaster(image, name) {
  Export.image.toDrive({
    image: image.toFloat(),
    description: name,
    folder: 'Rajapeta_Watershed',
    fileNamePrefix: name,
    region: roi,
    scale: 30,
    maxPixels: 1e13,
    crs: 'EPSG:4326',
    fileFormat: 'GeoTIFF'
  });
}

exportRaster(ndvi2004,   'Rajapeta_NDVI_2004');
exportRaster(ndvi2024,   'Rajapeta_NDVI_2024');
exportRaster(ndviChange, 'Rajapeta_NDVI_Change_2004_2024');
exportRaster(lst2004,    'Rajapeta_LST_2004');
exportRaster(lst2024,    'Rajapeta_LST_2024');
exportRaster(lstChange,  'Rajapeta_LST_Change_2004_2024');
exportRaster(ndwi2004,   'Rajapeta_NDWI_2004');
exportRaster(ndwi2024,   'Rajapeta_NDWI_2024');
exportRaster(lulc2004,   'Rajapeta_LULC_2004');
exportRaster(lulc2024,   'Rajapeta_LULC_2024');
exportRaster(lulcChange, 'Rajapeta_LULC_Change_2004_2024');
