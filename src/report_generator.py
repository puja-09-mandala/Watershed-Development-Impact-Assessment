"""Automated scientific report in Markdown, DOCX, and PDF."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

from config import PATHS, REPORT_NAMES, STUDY_AREA


def build_markdown(stats_text: str, llm_text: str, extras: dict) -> str:
    sa = STUDY_AREA
    lulc_note = extras.get("lulc_note", "LULC GeoTIFF data not available yet.")
    cnn = extras.get("cnn_note", "CNN-LSTM was not trained because insufficient valid patches were available.")
    return f"""# Spatio-Temporal Assessment of Watershed Development Impacts on Land Use, Vegetation Dynamics, and Surface Temperature Using CNN-LSTM and Mistral-Based Multimodal LLM

**Study area:** {sa['village']}, {sa['mandal']}, {sa['district']} District, {sa['state']}
**Period:** {sa['period']}

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
{lulc_note}
Final LULC is not produced from NDVI/NDWI thresholds. Supervised Random Forest is used when reference samples exist. Accuracy (overall, producer, user, kappa) is reported only from those samples.

## 10. CNN-LSTM Method
{cnn}

## 11. Change Detection
Increase, decrease, and no-significant-change classes are derived with documented thresholds for NDVI and LST. LULC change is pixel-wise class comparison.

## 12. Mistral Multimodal LLM
The LLM receives maps and numerical summaries. It does not classify pixels. It writes interpretation, discussion, limitations, and future work without inventing numbers.

## 13. Results
{stats_text}

## 14. Discussion
{llm_text}

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
"""


def write_markdown(text: str) -> Path:
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    path = PATHS["reports"] / REPORT_NAMES["md"]
    path.write_text(text, encoding="utf-8")
    (PATHS["web_data"] / "llm_report.md").write_text(text, encoding="utf-8")
    return path


def write_docx(md_text: str, figure_paths: list[Path]) -> Path:
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    path = PATHS["reports"] / REPORT_NAMES["docx"]
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    for block in md_text.split("\n"):
        if block.startswith("# "):
            doc.add_heading(block[2:], level=0)
        elif block.startswith("## "):
            doc.add_heading(block[3:], level=1)
        elif block.startswith("**") and block.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(block.strip("*"))
            run.bold = True
        elif block.strip():
            doc.add_paragraph(block)
    for fig in figure_paths[:16]:
        if fig.exists():
            doc.add_paragraph(fig.stem.replace("_", " "))
            try:
                doc.add_picture(str(fig), width=Inches(5.8))
            except Exception:
                pass
    doc.save(path)
    return path


def write_pdf(md_text: str, figure_paths: list[Path]) -> Path:
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    path = PATHS["reports"] / REPORT_NAMES["pdf"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontName="Times-Roman", fontSize=10, leading=13))
    styles.add(ParagraphStyle(name="H", parent=styles["Heading1"], fontName="Times-Bold", fontSize=13, leading=16))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=0.8 * inch, rightMargin=0.8 * inch)
    story = []
    for block in md_text.split("\n"):
        if block.startswith("# "):
            story.append(Paragraph(block[2:], styles["Title"]))
        elif block.startswith("## "):
            story.append(Paragraph(block[3:], styles["H"]))
        elif block.strip():
            story.append(Paragraph(block.replace("&", "&amp;"), styles["Body"]))
        story.append(Spacer(1, 6))
    story.append(PageBreak())
    for fig in figure_paths:
        if fig.exists():
            story.append(Paragraph(fig.stem.replace("_", " "), styles["H"]))
            try:
                story.append(Image(str(fig), width=6.2 * inch, height=4.6 * inch, kind="proportional"))
            except Exception:
                pass
            story.append(Spacer(1, 10))
    doc.build(story)
    return path


def generate_reports(md_text: str) -> dict:
    figs = sorted(PATHS["figures"].glob("Figure_*.png"))
    md = write_markdown(md_text)
    docx = write_docx(md_text, figs)
    pdf = write_pdf(md_text, figs)
    return {"md": str(md), "docx": str(docx), "pdf": str(pdf)}
