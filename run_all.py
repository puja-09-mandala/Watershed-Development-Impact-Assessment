"""Run the complete Rajapeta pipeline end-to-end.

Delegates to main.run(), which performs:
GeoTIFF discovery -> validation/alignment -> NDVI/LST/LULC products ->
128x128 patch extraction -> normalization -> CNN-LSTM training ->
predictions/error rasters -> change detection -> statistics -> web export ->
Mistral/fallback interpretation -> MD/DOCX/PDF reports.
"""

from main import run

if __name__ == "__main__":
    summary = run()
    print("\nOpen the web GIS with:  python serve.py")
    print("Routes: /ndvi | /lst | /lulc | /llm-report | /charts")