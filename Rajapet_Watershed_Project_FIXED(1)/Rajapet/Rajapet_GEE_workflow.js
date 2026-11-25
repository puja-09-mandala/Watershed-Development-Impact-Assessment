var studyArea = ee.Geometry.Rectangle([77.990116, 16.3373007, 78.0367386, 16.3688316]);

function maskL5(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(mask).multiply(0.0000275).add(-0.2);
}

function maskL8(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return img.updateMask(mask).multiply(0.0000275).add(-0.2);
}

function addNDVI(img, nir, red) {
  return img.addBands(img.normalizedDifference([nir, red]).rename('NDVI'));
}

var l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(studyArea).filterDate('2004-01-01','2004-12-31')
  .map(maskL5).map(function(i){ return addNDVI(i,'SR_B4','SR_B3'); });

var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(studyArea).filterDate('2024-01-01','2024-12-31')
  .map(maskL8).map(function(i){ return addNDVI(i,'SR_B5','SR_B4'); });

var ndvi2004 = l5.select('NDVI').median().clip(studyArea);
var ndvi2024 = l8.select('NDVI').median().clip(studyArea);
var ndviChange = ndvi2024.subtract(ndvi2004).rename('NDVI_Change');

Map.centerObject(studyArea, 13);
Map.addLayer(ndvi2004, {min:-0.2,max:0.8,palette:['8c510a','f6e8c3','91cf60','1a9850']}, 'NDVI 2004');
Map.addLayer(ndvi2024, {min:-0.2,max:0.8,palette:['8c510a','f6e8c3','91cf60','1a9850']}, 'NDVI 2024');
Map.addLayer(ndviChange, {min:-0.3,max:0.3,palette:['d73027','ffffbf','1a9850']}, 'NDVI Change');

Export.image.toDrive({image: ndvi2004, description:'Rajapeta_NDVI_2004', region:studyArea, scale:30, maxPixels:1e13});
Export.image.toDrive({image: ndvi2024, description:'Rajapeta_NDVI_2024', region:studyArea, scale:30, maxPixels:1e13});
Export.image.toDrive({image: ndviChange, description:'Rajapeta_NDVI_Change_2004_2024', region:studyArea, scale:30, maxPixels:1e13});
