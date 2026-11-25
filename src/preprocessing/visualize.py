import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# PATH
# ============================================================

DATA_DIR = "Rajapet/GeoTiff"

OUTPUT_DIR = "outputs/geotiff_visualization"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FILES
# ============================================================

files = {

    "NDVI 2004":
        "Rajapeta_NDVI_2004.tif",

    "NDVI 2024":
        "Rajapeta_NDVI_2024.tif",

    "NDVI Change 2004-2024":
        "Rajapeta_NDVI_Change_2004_2024.tif",

    "LST 2004":
        "Rajapeta_LST_2004.tif",

    "LST 2024":
        "Rajapeta_LST_2024.tif",

    "LST Change 2004-2024":
        "Rajapeta_LST_Change_2004_2024.tif"
}


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def visualize_raster(title, filename):

    filepath = os.path.join(DATA_DIR, filename)

    with rasterio.open(filepath) as src:

        data = src.read(1).astype(float)

        # Convert NoData if present
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        bounds = src.bounds

    # --------------------------------------------------------
    # SELECT COLOR MAP
    # --------------------------------------------------------

    if "NDVI Change" in title:

        cmap = "RdYlGn"

    elif "NDVI" in title:

        cmap = "YlGn"

    elif "LST Change" in title:

        cmap = "RdBu_r"

    else:

        cmap = "inferno"


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))


    # --------------------------------------------------------
    # DISPLAY RASTER
    # --------------------------------------------------------

    image = plt.imshow(
        data,
        cmap=cmap,
        extent=[
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top
        ]
    )


    # --------------------------------------------------------
    # COLOR BAR
    # --------------------------------------------------------

    colorbar = plt.colorbar(image)

    # Labels

    if "NDVI" in title:

        colorbar.set_label("NDVI")

    else:

        colorbar.set_label("Temperature (°C)")


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    plt.title(
        "Rajapeta - " + title,
        fontsize=14,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # AXIS
    # --------------------------------------------------------

    plt.xlabel("Longitude")

    plt.ylabel("Latitude")


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_name = (
        title
        .replace(" ", "_")
        .replace("-", "")
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        safe_name + ".png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    plt.show()

    plt.close()

    print(
        "Saved:",
        output_file
    )


# ============================================================
# PROCESS ALL SIX FILES
# ============================================================

for title, filename in files.items():

    print("\nProcessing:", title)

    visualize_raster(
        title,
        filename
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 60)

print(
    "All GeoTIFF visualizations completed successfully."
)

print("=" * 60)