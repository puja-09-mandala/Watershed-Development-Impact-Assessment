    # Rajapeta Watershed CNN-LSTM Academic System

Spatio-temporal assessment of vegetation, land-surface temperature, and optional land use/land cover for **Rajapeta (Rajapet) Village, Wanaparthy Mandal, Wanaparthy District, Telangana, India (2004–2024)**.

This is **not** a Mahbubnagar study. Clip all rasters to the Rajapeta village boundary.

CNN-LSTM, with only 2004 and 2024 observations, is a **prototype for representation learning, reconstruction, and change-analysis support**. It is not a validated long-term forecasting model.

Do not claim that watershed development caused the observed changes. Report **observed changes during the study period within the watershed-development context**. Rainfall, climate variability, cropping, irrigation, land management, urbanization, and seasonality may also contribute.

## Web GIS (5 pages)

Run `python serve.py` and the app opens directly into the NDVI page (no home/dashboard page):

1. `/ndvi` (`web/ndvi.html`) — NDVI 2004, NDVI 2024, NDVI change, village boundary, numeric legend  
2. `/lst` (`web/lst.html`) — LST 2004, LST 2024, LST change, village boundary, numeric legend  
3. `/lulc` (`web/lulc.html`) — LULC layers, or "LULC GeoTIFF data not available yet."  
4. `/llm-report` (`web/llm.html`) — input map selection, NDVI/LST/LULC/change/statistics sections, Mistral interpretation, key observations, environmental interpretation, limitations, downloadable report  
5. `/charts` (`web/charts.html`) — Chart.js analytics computed from `web/data/stats.json` (mean NDVI/LST comparisons, change statistics, NDVI/LST distributions, LULC area chart only when classified data exists)

Each map page has an interactive Leaflet map with zoom/pan, layer control (2004 / 2024 / change), legends with actual value ranges, the study-area boundary, opacity control, and computed statistics.

## Quick start

```bash
cd Watershed_CNN_LSTM
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put the GeoTIFFs in `data/raw/` **or** `Rajapet/GeoTiff/` (both are searched; matching is by exact expected filename). Replace `data/boundary/rajapeta_boundary.geojson` with the official village polygon.

```bash
python main.py   # process GeoTIFFs, train CNN-LSTM, write stats/manifest/reports
python serve.py  # http://127.0.0.1:8080/ndvi
```

Routes: `http://127.0.0.1:8080/ndvi` · `/lst` · `/lulc` · `/llm-report` · `/charts`

## Earth Engine

Edit `gee/rajapeta_landsat_export.js`: set the Rajapeta village asset, run in the Code Editor, download GeoTIFFs into `data/raw/`.

- 2004: `LANDSAT/LT05/C02/T1_L2` (NIR `SR_B4`, RED `SR_B3`, thermal `ST_B6`)
- 2024: `LANDSAT/LC08/C02/T1_L2` (NIR `SR_B5`, RED `SR_B4`, thermal `ST_B10`)

NDVI = (NIR − RED) / (NIR + RED). LST is Collection-2 surface temperature converted to Celsius.

## LULC

Do not use NDVI/NDWI thresholds as the final LULC method. Import supervised Random Forest maps when ready. Accuracy (overall, producer, user, kappa) is computed only if labeled samples exist. No fake LULC maps or accuracies are generated.

## Mistral LLM

Set `MISTRAL_API_KEY` in `.env`. The LLM interprets maps and pipeline statistics. It does not classify pixels and must not invent numbers. Without a key, the report is filled only from computed statistics.

Outputs:

- `outputs/reports/Rajapeta_Watershed_LLM_Report.md`
- `outputs/reports/Rajapeta_Watershed_LLM_Report.docx`
- `outputs/reports/Rajapeta_Watershed_LLM_Report.pdf`
- `outputs/paper/Rajapeta_Watershed_Research_Paper.md`

## Architecture

Satellite data → preprocess / cloud mask / median composite / scale / clip → GeoTIFF → NDVI + LST + LULC → 128×128 patches → temporal stack → normalize → train/test → CNN → LSTM → reconstruction → change detection → statistics → Mistral LLM → report + 5-page Web GIS.
