"""Publication-style research paper (template filled with pipeline evidence)."""

from __future__ import annotations

from pathlib import Path

from config import PATHS, STUDY_AREA


def write_paper(stats_text: str, llm_text: str, extras: dict) -> Path:
    sa = STUDY_AREA
    PATHS["paper"].mkdir(parents=True, exist_ok=True)
    lulc_note = extras.get("lulc_note", "LULC GeoTIFF data not available yet.")
    cnn = extras.get("cnn_note", "The CNN-LSTM prototype was not trained in this run.")
    text = f"""# Spatio-Temporal Assessment of Watershed Development Impacts on Land Use, Vegetation Dynamics, and Surface Temperature Using CNN-LSTM and Mistral-Based Multimodal LLM

## ABSTRACT
This paper presents a reproducible remote-sensing and deep-learning workflow for Rajapeta (Rajapet) Village, Wanaparthy Mandal, Wanaparthy District, Telangana, India, spanning 2004–2024. Landsat 5 and Landsat 8 Collection 2 Level-2 data provide NDVI and land-surface temperature. Optional supervised Random Forest LULC maps are ingested when reference-based products exist. A CNN-LSTM network extracts spatial features with TimeDistributed convolutions and temporal structure with LSTM units, then reconstructs 128 × 128 spatial fields. A Mistral multimodal LLM interprets maps and computed statistics for scientific reporting. With only two observation years, the network is described as a prototype for representation learning, reconstruction, and change-analysis support, not as a validated long-term forecast. Observed changes are discussed within the watershed-development context without causal attribution to programme interventions.

## KEYWORDS
Rajapeta; Wanaparthy; NDVI; land surface temperature; LULC; CNN-LSTM; change detection; Mistral multimodal LLM; watershed development

## 1. INTRODUCTION
Monitoring vegetation greenness, surface temperature, and land cover is central to evaluating landscapes where watershed works, agriculture, and settlement coexist. This study implements an end-to-end pipeline from GeoTIFF ingest through CNN-LSTM reconstruction and multimodal interpretation. The study area is Rajapeta, Wanaparthy; Mahbubnagar is not used as the current administrative study area.

## 2. STUDY AREA
{sa['village']}, {sa['mandal']}, {sa['district']} District, {sa['state']}. The analysis period is {sa['period']}. Processing is constrained to the official village boundary supplied with the spatial data. Landsat spatial resolution is 30 m.

## 3. DATASETS
Landsat 5 TM Collection 2 Level-2 (`LANDSAT/LT05/C02/T1_L2`) for 2004 and Landsat 8 OLI/TIRS Collection 2 Level-2 (`LANDSAT/LC08/C02/T1_L2`) for 2024. Optical bands are scaled with the Collection-2 surface-reflectance coefficients. Thermal surface temperature (ST_B6 / ST_B10) is converted to Celsius. LULC GeoTIFFs, if present, must originate from supervised classification rather than NDVI thresholds.

## 4. METHODOLOGY

### 4.1 Data Preprocessing
Load GeoTIFF; verify CRS, resolution, dimensions, and NoData; detect invalid pixels; align rasters; clip to the common village extent; generate 128 × 128 patches; stack 2004 and 2024; normalize; split train/test.

### 4.2 NDVI Extraction
Landsat 5: NIR = SR_B4, RED = SR_B3. Landsat 8: NIR = SR_B5, RED = SR_B4. NDVI = (NIR − RED)/(NIR + RED). Change = NDVI_2024 − NDVI_2004.

### 4.3 LST Extraction
Collection-2 thermal surface temperature is converted to Celsius. Change = LST_2024 − LST_2004.

### 4.4 LULC Classification
{lulc_note}
Preferred classifier: Random Forest. Classes depend on reference data (for example water, vegetation, built-up, barren/agriculture). Accuracy assessment is performed only when labeled samples exist.

### 4.5 CNN Spatial Feature Extraction
TimeDistributed Conv2D → activation → MaxPooling2D, repeated, to capture vegetation structure, temperature pattern, and landscape heterogeneity.

### 4.6 LSTM Temporal Learning
CNN features feed an LSTM that represents the 2004→2024 transition.

### 4.7 Prediction/Reconstruction
Dense restoration and upsampling reconstruct 128 × 128 NDVI/LST fields for 2024. An optional pixel-wise softmax head is reserved for labeled LULC.

### 4.8 Change Detection
Continuous differencing for NDVI and LST; class comparison for LULC; mapped as increase, decrease, or no significant change where thresholds apply.

### 4.9 Mistral Multimodal LLM Interpretation
Maps plus statistics are passed to a Mistral multimodal model for interpretation and report drafting. The LLM does not invent numbers.

## 5. RESULTS

### 5.1 NDVI Results
See computed statistics below and Figures 6–8.

### 5.2 LST Results
See computed statistics below and Figures 9–11.

### 5.3 LULC Results
{lulc_note}

### 5.4 Change Detection
See change maps and class percentages in the JSON statistics file.

### 5.5 CNN-LSTM Results
{cnn}

### 5.6 LLM Interpretation
{llm_text}

### Computed raster statistics
{stats_text}

## 6. DISCUSSION
Observed NDVI and LST patterns should be interpreted jointly with land management, rainfall, irrigation, cropping calendars, and possible urbanization. Agreement or disagreement between reconstructed and observed 2024 fields indicates how well the prototype captures spatial structure, not forecast skill over unseen years.

## 7. LIMITATIONS
Two dates; possible phenological mismatch in annual medians; mixed pixels at 30 m; LULC labels may be absent; LLM text is only as valid as the supplied numbers; no causal identification of watershed works.

## 8. CONCLUSION
The Rajapeta, Wanaparthy system delivers satellite analysis, optional LULC, CNN-LSTM reconstruction, interactive GIS, and multimodal reporting for 2004–2024.

## 9. FUTURE WORK
Additional years and seasons, climate covariates, field-verified LULC, uncertainty maps, and independent hold-out villages.

## 10. REFERENCES
1. USGS. Landsat Collection 2 Level-2 Science Products.
2. Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1973). Monitoring vegetation systems in the Great Plains with ERTS. NASA SP-351.
3. Google Earth Engine Catalog. LANDSAT/LT05/C02/T1_L2; LANDSAT/LC08/C02/T1_L2.
4. Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.
5. LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). Gradient-based learning applied to document recognition.
6. Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.
7. Mistral AI. Pixtral / multimodal API documentation.
"""
    path = PATHS["paper"] / "Rajapeta_Watershed_Research_Paper.md"
    path.write_text(text, encoding="utf-8")
    return path
