
var roi = ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq('ADM2_NAME', 'Mahbubnagar'));

Map.centerObject(roi, 9);

function maskLandsat(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
              .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
}

var l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(roi)
          .filterDate('2004-01-01', '2004-12-31')
          .map(maskLandsat)
          .median();

var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(roi)
          .filterDate('2024-01-01', '2024-12-31')
          .map(maskLandsat)
          .median();


function classify(image, nir, red, swir, green) {
  
  var ndvi = image.normalizedDifference([nir, red]);
  var ndbi = image.normalizedDifference([swir, nir]);
  var ndwi = image.normalizedDifference([green, nir]);

  var water = ndwi.gt(0.25);
  var vegetation = ndvi.gt(0.4);
  var builtup = ndbi.gt(0.2);
  
  var lulc = ee.Image(0)
    .where(water, 1)
    .where(vegetation, 2)
    .where(builtup, 3)
    .where(ndvi.lt(0.2).and(ndwi.lt(0)), 4);
    
  return lulc;
}


// Apply classification
var lulc_2004 = classify(l5, 'SR_B4', 'SR_B3', 'SR_B5', 'SR_B2').clip(roi);
var lulc_2024 = classify(l8, 'SR_B5', 'SR_B4', 'SR_B6', 'SR_B3').clip(roi);
var lulcVis = {
  min: 1,
  max: 4,
  palette: [
    '#1f78b4', // Water → clear blue
    '#33a02c', // Vegetation → strong green
    '#e31a1c', // Built-up → bright red
    '#ff7f00'  // Barren → orange
  ]
};

var layer2004 = ui.Map.Layer(lulc_2004, lulcVis, 'LULC 2004', true);
var layer2024 = ui.Map.Layer(lulc_2024, lulcVis, 'LULC 2024', false);

Map.layers().reset([layer2004, layer2024]);
