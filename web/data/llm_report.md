# Spatio-Temporal Assessment of Watershed Development Impacts on Land Use, Vegetation Dynamics, and Surface Temperature Using CNN-LSTM and Mistral-Based Multimodal LLM

**Study area:** Rajapeta (Rajapet), Wanaparthy Mandal, Wanaparthy District, Telangana, India
**Period:** 2004–2024

## 1. Abstract
This dissertation-style system analyses Landsat-derived NDVI and land-surface temperature (LST) for Rajapeta village, Wanaparthy, Telangana, between 2004 and 2024. Optional supervised Random Forest land-use/land-cover (LULC) maps are ingested when provided. A CNN-LSTM prototype learns spatial–temporal representations from 128 × 128 patches. A Mistral-based multimodal LLM interprets maps and pipeline statistics and writes this report. The work does not claim that watershed development caused the observed changes; it documents observed changes during the study period within the watershed-development context.

## 2. Introduction
Watershed-development programmes alter land management, vegetation, and energy-balance patterns, but satellite time series also respond to rainfall, cropping, irrigation, and urban growth. This project implements a reproducible remote-sensing, GIS, deep-learning, and multimodal-LLM workflow focused on Rajapeta, Wanaparthy—not Mahbubnagar.

## 3. Study Area
Rajapeta (Rajapet) Village, Wanaparthy Mandal, Wanaparthy District, Telangana, India. Analysis is clipped to the supplied village boundary. Spatial resolution is 30 m (Landsat Collection 2 Level-2).

## 4. Dataset
- 2004: Landsat 5, `LANDSAT/LT05/C02/T1_L2` (NIR = SR_B4, RED = SR_B3, thermal = ST_B6).
- 2024: Landsat 8, `LANDSAT/LC08/C02/T1_L2` (NIR = SR_B5, RED = SR_B4, thermal = ST_B10).
GeoTIFF inputs are imported by the user. LULC rasters are optional.

## 5. Methodology
Satellite composites are cloud-masked, median-composited, scaled, clipped, and exported. NDVI, LST, and optional LULC products feed patch extraction, CNN spatial encoding, LSTM temporal encoding, reconstruction, change detection, statistics, Web GIS, and LLM interpretation.

## 6. Preprocessing
GeoTIFFs are loaded; CRS, resolution, dimensions, and NoData are verified; invalid pixels are flagged; rasters are aligned; 128 × 128 patches are stacked over 2004 and 2024, normalized, and split for training and testing.

## 7. NDVI Analysis
NDVI = (NIR − RED) / (NIR + RED). Change = NDVI 2024 − NDVI 2004.

## 8. LST Analysis
Collection-2 surface temperature is converted to Celsius. Change = LST 2024 − LST 2004.

## 9. LULC Analysis
LULC rasters were ingested. Class areas and the transition matrix were computed from pixel labels. Accuracy metrics are omitted unless independent reference samples exist. No training-sample CSV was found, so kappa and overall accuracy were not computed.
Final LULC is not produced from NDVI/NDWI thresholds. Supervised Random Forest is used when reference samples exist. Accuracy (overall, producer, user, kappa) is reported only from those samples.

## 10. CNN-LSTM Method
Spatio-temporal CNN-LSTM prototype for representation learning, reconstruction, and change-analysis support. Not a validated long-term forecasting model. Held-out MAE=0.1710, RMSE=0.2412, MSE=0.0582 (n_test=2). Limitation: patches are overlapping sliding-window 128x128 crops of a single small scene, so training and test patches share spatial context; these metrics are indicative reconstruction errors, not independent-validation scores.

## 11. Change Detection
Increase, decrease, and no-significant-change classes are derived with documented thresholds for NDVI and LST. LULC change is pixel-wise class comparison.

## 12. Mistral Multimodal LLM
The LLM receives maps and numerical summaries. It does not classify pixels. It writes interpretation, discussion, limitations, and future work without inventing numbers.

## 13. Results
- ndvi_2004: n=14648, mean=0.1398, median=0.1381, min=0.0214, max=0.2614, std=0.0226
- ndvi_2024: n=14648, mean=0.1806, median=0.1751, min=-0.0054, max=0.3403, std=0.0338
- ndvi_change: n=14648, mean=0.0408, median=0.0378, min=-0.2612, max=0.2257, std=0.0378
- lst_2004: n=14648, mean=38.9590, median=39.1562, min=31.1170, max=45.2556, std=2.2204
- lst_2024: n=14648, mean=37.1893, median=37.2352, min=29.8369, max=43.9038, std=2.0407
- lst_change: n=14648, mean=-1.7697, median=-1.5825, min=-10.3703, max=5.1492, std=2.4263
- lulc_2004: n=14648, mean=3.7783, median=3.0000, min=1.0000, max=5.0000, std=0.8969
- lulc_2024: n=14648, mean=4.6625, median=5.0000, min=1.0000, max=5.0000, std=0.8152
- lulc_change: n=14648, mean=0.6388, median=1.0000, min=0.0000, max=1.0000, std=0.4804

## 14. Discussion
_MISTRAL_API_KEY is not set. Interpretation is generated from pipeline statistics only; no numbers are invented._
## NDVI interpretation
For Rajapeta (Rajapet), Wanaparthy Mandal, Wanaparthy District, Telangana, India (2004–2024), computed NDVI summaries are: NDVI 2004 mean=0.1398, median=0.1381; NDVI 2024 mean=0.1806, median=0.1751; NDVI change mean=0.0408.
These values describe greenness from Landsat surface reflectance and should be read as observed vegetation-index change, not as a causal watershed-treatment effect.

## LST interpretation
Computed land-surface temperature summaries (Celsius) are: LST 2004 mean=38.9590 °C; LST 2024 mean=37.1893 °C; LST change mean=-1.7697 °C.
Spatial hotspots in the LST change map, where available, indicate warming or cooling relative to 2004, subject to seasonal composite differences.

## LULC interpretation
LULC rasters were ingested. Class areas and the transition matrix were computed from pixel labels. Accuracy metrics are omitted unless independent reference samples exist. No training-sample CSV was found, so kappa and overall accuracy were not computed.
Supervised Random Forest classification is used only when labeled reference samples and LULC rasters are provided. No NDVI-threshold land-cover map is treated as a final LULC product.

## Change detection interpretation
NDVI change is NDVI 2024 − NDVI 2004. LST change is LST 2024 − LST 2004. Increase / decrease / no significant change classes use documented thresholds (NDVI 0.05; LST 1 °C). LULC change is a pixel-wise class comparison when both years exist.

## CNN-LSTM interpretation
CNN-LSTM reconstruction MAE=0.1710, RMSE=0.2412, MSE=0.0582 on 2 held-out patches. Spatio-temporal CNN-LSTM prototype for representation learning, reconstruction, and change-analysis support. Not a validated long-term forecasting model.

## Integrated interpretation
Joint reading of NDVI and LST is restricted to pixels that are valid in both products. Inverse NDVI–LST patterns (greener and cooler, or sparser and warmer) are noted only when both layers are present. Observed changes during the study period within the watershed-development context may also reflect rainfall, climate variability, cropping, irrigation, land management, urbanization, and seasonality.

## Scientific discussion
The workflow follows a remote-sensing plus CNN-LSTM plus multimodal LLM architecture. The LLM does not classify land cover; it interprets maps and statistics produced by the quantitative pipeline.

## Limitations
Two annual composites (2004 and 2024) cannot support validated multi-year forecasting. Cloud-free median composites can still mix phenological stages. LULC accuracy is reported only when independent reference samples exist.

## Conclusion
The system documents spatio-temporal NDVI and LST patterns for Rajapeta, Wanaparthy, and supports reconstruction-based CNN-LSTM analysis when sufficient 128×128 patches are generated.

## Future work
Add additional years, seasonal composites, rainfall covariates, field-verified LULC samples, and independent spatial validation before any operational forecast claim.

## 15. Limitations
Two dates cannot support validated long-term forecasting. Seasonal compositing, mixed pixels, and missing LULC labels remain important constraints. Causal attribution to watershed works is not claimed.

## 16. Conclusion
The Rajapeta, Wanaparthy workflow provides satellite-derived NDVI/LST analysis, optional LULC ingestion, CNN-LSTM reconstruction support, interactive GIS, and LLM-assisted scientific writing.

## 17. Future Scope
Multi-year stacks, seasonal windows, rainfall covariates, field LULC samples, and independent validation.

## 18. References
References are limited to datasets and methods actually used:
- USGS / NASA Landsat Collection 2 Level-2 (Landsat 5 TM; Landsat 8 OLI/TIRS).
- Rouse, J.W., et al. (1973). Monitoring vegetation systems in the Great Plains with ERTS.
- Google Earth Engine Data Catalog: LANDSAT/LT05/C02/T1_L2; LANDSAT/LC08/C02/T1_L2.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
- LeCun, Y., et al. Convolutional networks for images.
- Breiman, L. (2001). Random forests. Machine Learning.
- Mistral AI. Pixtral multimodal model documentation.
Do not add unread papers.
