"""Mistral multimodal interpretation. Numbers come only from the analysis payload."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from config import MISTRAL, PATHS, STUDY_AREA

load_dotenv()

SYSTEM_PROMPT = """You are a scientific writing assistant for a remote-sensing dissertation.
Study area: Rajapeta (Rajapet) Village, Wanaparthy Mandal, Wanaparthy District, Telangana, India, 2004–2024.
You interpret maps and statistics only. You are NOT a land-cover classifier.

Rules:
- Use ONLY numerical values provided in the JSON payload. Never invent statistics, accuracies, or citations.
- Do not claim that watershed development caused the observed changes.
- Use phrasing such as: "Observed changes during the study period within the watershed-development context."
- Other contributing factors may include rainfall, climate variability, land-management practices, cropping patterns, land-use change, irrigation, urbanization, and seasonal differences, unless quantified evidence is supplied.
- If LULC rasters or CNN-LSTM metrics are missing, say they are not available. Do not fabricate them.
- With only 2004 and 2024 observations, describe CNN-LSTM as a spatio-temporal prototype for representation learning, reconstruction, and change-analysis support, not a validated long-term forecasting model.
- Produce these sections: NDVI interpretation; LST interpretation; LULC interpretation; change detection; CNN-LSTM interpretation; integrated interpretation; scientific discussion; limitations; conclusion; future work.
"""


def _encode_image(path: Path) -> str | None:
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().replace(".", "")
    mime = "image/png" if suffix == "png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def collect_map_images() -> list[Path]:
    names = [
        "Figure_06_NDVI_2004.png",
        "Figure_07_NDVI_2024.png",
        "Figure_08_NDVI_Change.png",
        "Figure_09_LST_2004.png",
        "Figure_10_LST_2024.png",
        "Figure_11_LST_Change.png",
        "Figure_12_LULC_2004.png",
        "Figure_13_LULC_2024.png",
        "Figure_14_LULC_Change.png",
        "Figure_15_Actual_vs_Predicted_NDVI.png",
        "Figure_16_Actual_vs_Predicted_LST.png",
    ]
    return [PATHS["figures"] / n for n in names if (PATHS["figures"] / n).exists()]


def mistral_available() -> bool:
    """True when MISTRAL_API_KEY is configured (never exposes the key itself)."""
    load_dotenv()
    return bool(os.getenv(MISTRAL["api_key_env"], "").strip())


def interpret(payload: dict) -> str:
    api_key = os.getenv(MISTRAL["api_key_env"], "").strip()
    if not api_key:
        return fallback_interpretation(payload)

    content = [
        {
            "type": "text",
            "text": (
                "Interpret the Rajapeta watershed analysis. "
                "JSON evidence follows. Maps are attached when available.\n\n"
                + json.dumps(payload, indent=2)[:12000]
            ),
        }
    ]
    for img in collect_map_images()[:9]:
        encoded = _encode_image(img)
        if encoded:
            content.append({"type": "image_url", "image_url": encoded})

    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": MISTRAL["max_tokens"],
                "temperature": 0.2,
            },
            timeout=180,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in text
            )
        return text.strip()
    except Exception as exc:
        return fallback_interpretation(payload, error=str(exc))


def fallback_interpretation(payload: dict, error: str | None = None) -> str:
    rasters = {r["name"]: r for r in payload.get("rasters", []) if r.get("n", 0)}
    model = payload.get("cnn_lstm", {})
    lulc_note = payload.get("lulc_note", "LULC GeoTIFF data not available yet.")
    sa = STUDY_AREA

    def s(name, field):
        row = rasters.get(name)
        if not row:
            return "not available"
        return f"{row[field]:.4f}"

    header = []
    if error:
        header.append(
            f"_Mistral API was not used ({error}). The following interpretation uses only computed pipeline statistics._\n"
        )
    elif not os.getenv(MISTRAL["api_key_env"], "").strip():
        header.append(
            "_MISTRAL_API_KEY is not set. Interpretation is generated from pipeline statistics only; no numbers are invented._\n"
        )

    ndvi = (
        f"NDVI 2004 mean={s('ndvi_2004','mean')}, median={s('ndvi_2004','median')}; "
        f"NDVI 2024 mean={s('ndvi_2024','mean')}, median={s('ndvi_2024','median')}; "
        f"NDVI change mean={s('ndvi_change','mean')}."
    )
    lst = (
        f"LST 2004 mean={s('lst_2004','mean')} °C; LST 2024 mean={s('lst_2024','mean')} °C; "
        f"LST change mean={s('lst_change','mean')} °C."
    )
    cnn = "CNN-LSTM metrics are not available."
    if model.get("trained"):
        cnn = (
            f"CNN-LSTM reconstruction MAE={model.get('mae'):.4f}, RMSE={model.get('rmse'):.4f}, "
            f"MSE={model.get('mse'):.4f} on {model.get('n_test')} held-out patches. "
            f"{model.get('role')}"
        )

    body = f"""
## NDVI interpretation
For {sa['village']}, {sa['mandal']}, {sa['district']} District, {sa['state']} ({sa['period']}), computed NDVI summaries are: {ndvi}
These values describe greenness from Landsat surface reflectance and should be read as observed vegetation-index change, not as a causal watershed-treatment effect.

## LST interpretation
Computed land-surface temperature summaries (Celsius) are: {lst}
Spatial hotspots in the LST change map, where available, indicate warming or cooling relative to 2004, subject to seasonal composite differences.

## LULC interpretation
{lulc_note}
Supervised Random Forest classification is used only when labeled reference samples and LULC rasters are provided. No NDVI-threshold land-cover map is treated as a final LULC product.

## Change detection interpretation
NDVI change is NDVI 2024 − NDVI 2004. LST change is LST 2024 − LST 2004. Increase / decrease / no significant change classes use documented thresholds (NDVI 0.05; LST 1 °C). LULC change is a pixel-wise class comparison when both years exist.

## CNN-LSTM interpretation
{cnn}

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
""".strip()
    return "\n".join(header) + body
