import rasterio
from pathlib import Path

data_path = Path("data/raw")

files = list(data_path.glob("*.tif"))

print("Number of GeoTIFF files:", len(files))

for file in files:
    with rasterio.open(file) as src:
        print("\n-----------------------------")
        print("File:", file.name)
        print("Width:", src.width)
        print("Height:", src.height)
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Data type:", src.dtypes[0])