/**
 * Export Rajapeta NDVI / LST composites for the 5-year study epochs:
 * 2004, 2009, 2014, 2019, 2024.
 *
 * The local pipeline already has 2004 and 2024. Run this script in the GEE
 * Code Editor to generate the MISSING epochs (2009, 2014, 2019), then download
 * the exported GeoTIFFs into data/raw/ using these exact names:
 *
 *   Rajapeta_NDVI_2009.tif  Rajapeta_NDVI_2014.tif  Rajapeta_NDVI_2019.tif
 *   Rajapeta_LST_2009.tif   Rajapeta_LST_2014.tif   Rajapeta_LST_2019.tif
 *
 * After placing them, run `python main.py` - the /charts trend graphs pick up
 * every epoch that exists and never plot missing years.
 *
 * NOTE: replace STUDY_AREA with your Rajapeta village boundary asset.
 */
var STUDY_AREA = ee.FeatureCollection('users/YOUR_USERNAME/Rajapeta_Village_Boundary');
var roi = STUDY_AREA.geometry();
// Landsat 5 for pre-2013 epochs, Landsat 8 for 2013 onwards.
var CONFIG = {
  2004: {col: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', red: 'SR_B3', thermal: 'ST_B6'},
  2009: {col: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', red: 'SR_B3', thermal: 'ST_B6'},
  2014: {col: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', thermal: 'ST_B10'},
  2019: {col: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', thermal: 'ST_B10'},
  2024: {col: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', thermal: 'ST_B10'}
};

function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
}

Object.keys(CONFIG).forEach(function (yearKey) {
  var year = parseInt(yearKey, 10);
  var cfg = CONFIG[yearKey];
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = ee.Date.fromYMD(year, 12, 31);

  // NDVI: annual median composite from scaled surface reflectance.
  var sr = ee.ImageCollection(cfg.col)
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskClouds)
    .median()
    .multiply(0.0000275)
    .add(-0.2)
    .clip(roi);

  var ndvi = sr.normalizedDifference([cfg.nir, cfg.red]).rename('NDVI');
  Export.image.toDrive({
    image: ndvi.toFloat(),
    description: 'Rajapeta_NDVI_' + year,
    folder: 'Rajapeta_Watershed',
    region: roi,
    scale: 30,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });

  // LST: Landsat Collection 2 surface temperature converted to Celsius.
  var st = ee.ImageCollection(cfg.col)
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskClouds)
    .median()
    .select(cfg.thermal)
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST_C')
    .clip(roi);

  Export.image.toDrive({
    image: st.toFloat(),
    description: 'Rajapeta_LST_' + year,
    folder: 'Rajapeta_Watershed',
    region: roi,
    scale: 30,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});
