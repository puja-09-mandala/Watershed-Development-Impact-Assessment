/**
 * Optional supervised Random Forest LULC in Earth Engine.
 * Requires labeled FeatureCollections for 2004 and 2024 (not NDVI thresholds).
 * Export class maps as Rajapeta_LULC_2004 / 2024 after you have reference samples.
 */

var STUDY_AREA = ee.FeatureCollection('users/YOUR_USERNAME/Rajapeta_Village_Boundary');
var roi = STUDY_AREA.geometry();

function maskLandsatSR(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
}

function composite(collectionId, start, end, bands) {
  return ee.ImageCollection(collectionId)
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskLandsatSR)
    .select(bands)
    .median()
    .multiply(0.0000275)
    .add(-0.2)
    .clip(roi);
}

var img2004 = composite('LANDSAT/LT05/C02/T1_L2', '2004-01-01', '2004-12-31',
  ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']);
var img2024 = composite('LANDSAT/LC08/C02/T1_L2', '2024-01-01', '2024-12-31',
  ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']);

// Replace these with your labeled points (property: landcover)
var training2004 = ee.FeatureCollection('users/YOUR_USERNAME/Rajapeta_LULC_samples_2004');
var training2024 = ee.FeatureCollection('users/YOUR_USERNAME/Rajapeta_LULC_samples_2024');

function classify(image, samples, bandList) {
  var trained = ee.Classifier.smileRandomForest(200).train({
    features: image.sampleRegions({collection: samples, properties: ['landcover'], scale: 30}),
    classProperty: 'landcover',
    inputProperties: bandList
  });
  return image.classify(trained);
}

var lulc2004 = classify(img2004, training2004, img2004.bandNames());
var lulc2024 = classify(img2024, training2024, img2024.bandNames());
var lulcChange = lulc2024.neq(lulc2004);

Export.image.toDrive({image: lulc2004.toByte(), description: 'Rajapeta_LULC_2004', folder: 'Rajapeta_Watershed', region: roi, scale: 30, maxPixels: 1e13});
Export.image.toDrive({image: lulc2024.toByte(), description: 'Rajapeta_LULC_2024', folder: 'Rajapeta_Watershed', region: roi, scale: 30, maxPixels: 1e13});
Export.image.toDrive({image: lulcChange.toByte(), description: 'Rajapeta_LULC_Change_2004_2024', folder: 'Rajapeta_Watershed', region: roi, scale: 30, maxPixels: 1e13});
