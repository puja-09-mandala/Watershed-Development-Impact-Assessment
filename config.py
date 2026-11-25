"""Project configuration for the Rajapeta watershed CNN-LSTM system."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

STUDY_AREA = {
    "village": "Rajapeta (Rajapet)",
    "mandal": "Wanaparthy Mandal",
    "district": "Wanaparthy",
    "state": "Telangana, India",
    "country": "India",
    "period": "2004–2024",
    "year_start": 2004,
    "year_end": 2024,
    "resolution_m": 30,
    "crs_expected": "EPSG:4326",
}
PATHS = {
    "raw": ROOT / "data" / "raw",
    "boundary": ROOT / "data" / "boundary",
    "processed": ROOT / "data" / "processed",
    "patches": ROOT / "data" / "patches",
    "models": ROOT / "data" / "models",
    "outputs": ROOT / "outputs",
    "maps": ROOT / "outputs" / "maps",
    "charts": ROOT / "outputs" / "charts",
    "stats": ROOT / "outputs" / "stats",
    "reports": ROOT / "outputs" / "reports",
    "paper": ROOT / "outputs" / "paper",
    "web": ROOT / "web",
    "web_data": ROOT / "web" / "data",
    "figures": ROOT / "outputs" / "figures",
}

EXPECTED_RASTERS = {
    "ndvi_2004": "Rajapeta_NDVI_2004.tif",
    "ndvi_2024": "Rajapeta_NDVI_2024.tif",
    "ndvi_change": "Rajapeta_NDVI_Change_2004_2024.tif",
    "lst_2004": "Rajapeta_LST_2004.tif",
    "lst_2024": "Rajapeta_LST_2024.tif",
    "lst_change": "Rajapeta_LST_Change_2004_2024.tif",
    "lulc_2004": "Rajapeta_LULC_2004.tif",
    "lulc_2024": "Rajapeta_LULC_2024.tif",
    "lulc_change": "Rajapeta_LULC_Change_2004_2024.tif",
}

LANDSAT = {
    "2004": {
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "nir": "SR_B4",
        "red": "SR_B3",
        "thermal": "ST_B6",
        "scale_sr": 0.0000275,
        "offset_sr": -0.2,
        "scale_st": 0.00341802,
        "offset_st": 149.0,
    },
    "2024": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "nir": "SR_B5",
        "red": "SR_B4",
        "thermal": "ST_B10",
        "scale_sr": 0.0000275,
        "offset_sr": -0.2,
        "scale_st": 0.00341802,
        "offset_st": 149.0,
    },
}

PATCH = {
    "size": 128,
    "stride": 64,
    "channels": ["ndvi", "lst"],
    "time_steps": 2,
}

MODEL = {
    "name": "cnn_lstm_prototype",
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel_size": 3,
    "lstm_units": 128,
    "dense_units": 256,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 8,
    "epochs": 30,
    "validation_split": 0.2,
    "test_split": 0.2,
    "random_seed": 42,
}

CHANGE_THRESHOLDS = {
    "ndvi": 0.05,
    "lst_c": 1.0,
}

LULC_CLASSES = {
    1: "Water",
    2: "Vegetation",
    3: "Built-up",
    4: "Barren Land",
    5: "Agricultural/Cropland",
}

MISTRAL = {
    "api_key_env": "MISTRAL_API_KEY",
    "model": os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"),
    "max_tokens": 4000,
}

REPORT_NAMES = {
    "md": "Rajapeta_Watershed_LLM_Report.md",
    "docx": "Rajapeta_Watershed_LLM_Report.docx",
    "pdf": "Rajapeta_Watershed_LLM_Report.pdf",
}


def ensure_directories() -> None:
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    for extra in ("logs",):
        (ROOT / extra).mkdir(parents=True, exist_ok=True)
