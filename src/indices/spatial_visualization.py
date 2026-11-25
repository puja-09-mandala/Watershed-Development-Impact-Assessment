import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/maps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(
    f"{OUTPUT_DIR}/ndvi",
    exist_ok=True
)

os.makedirs(
    f"{OUTPUT_DIR}/lst",
    exist_ok=True
)

os.makedirs(
    f"{OUTPUT_DIR}/lulc",
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("========================================")
print("LOADING PROCESSED DATA")
print("========================================")

ndvi_2004 = np.load(
    f"{DATA_DIR}/ndvi_2004.npy"
)

ndvi_2024 = np.load(
    f"{DATA_DIR}/ndvi_2024.npy"
)

lst_2004 = np.load(
    f"{DATA_DIR}/lst_2004.npy"
)

lst_2024 = np.load(
    f"{DATA_DIR}/lst_2024.npy"
)

lulc_2004 = np.load(
    f"{DATA_DIR}/lulc_2004.npy"
)

lulc_2024 = np.load(
    f"{DATA_DIR}/lulc_2024.npy"
)


print("NDVI 2004:", ndvi_2004.shape)
print("NDVI 2024:", ndvi_2024.shape)

print("LST 2004:", lst_2004.shape)
print("LST 2024:", lst_2024.shape)

print("LULC 2004:", lulc_2004.shape)
print("LULC 2024:", lulc_2024.shape)


# ============================================================
# 3. CALCULATE CHANGES
# ============================================================

print("\n========================================")
print("CALCULATING CHANGES")
print("========================================")


ndvi_change = ndvi_2024 - ndvi_2004

lst_change = lst_2024 - lst_2004

lulc_changed = (
    lulc_2004 != lulc_2024
)


# ============================================================
# 4. COMMON MAP FUNCTION
# ============================================================

def save_continuous_map(
    data,
    title,
    filename,
    cmap,
    vmin=None,
    vmax=None,
    label=""
):

    plt.figure(
        figsize=(10, 7)
    )

    image = plt.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Pixel"
    )

    plt.ylabel(
        "Pixel"
    )

    colorbar = plt.colorbar(
        image
    )

    colorbar.set_label(
        label
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        filename
    )


# ============================================================
# 5. NDVI 2004
# ============================================================

print("\n========================================")
print("CREATING NDVI MAPS")
print("========================================")

save_continuous_map(
    ndvi_2004,
    "Mahbubnagar NDVI - 2004",
    f"{OUTPUT_DIR}/ndvi/NDVI_2004.png",
    "RdYlGn",
    -1,
    1,
    "NDVI"
)


# ============================================================
# 6. NDVI 2024
# ============================================================

save_continuous_map(
    ndvi_2024,
    "Mahbubnagar NDVI - 2024",
    f"{OUTPUT_DIR}/ndvi/NDVI_2024.png",
    "RdYlGn",
    -1,
    1,
    "NDVI"
)


# ============================================================
# 7. NDVI CHANGE
# ============================================================

ndvi_limit = np.nanmax(
    np.abs(ndvi_change)
)

save_continuous_map(
    ndvi_change,
    "Mahbubnagar NDVI Change (2024 - 2004)",
    f"{OUTPUT_DIR}/ndvi/NDVI_CHANGE.png",
    "RdYlGn",
    -ndvi_limit,
    ndvi_limit,
    "NDVI Change"
)


# ============================================================
# 8. LST 2004
# ============================================================

print("\n========================================")
print("CREATING LST MAPS")
print("========================================")

lst_min = min(
    np.nanpercentile(lst_2004, 2),
    np.nanpercentile(lst_2024, 2)
)

lst_max = max(
    np.nanpercentile(lst_2004, 98),
    np.nanpercentile(lst_2024, 98)
)


save_continuous_map(
    lst_2004,
    "Mahbubnagar LST - 2004",
    f"{OUTPUT_DIR}/lst/LST_2004.png",
    "turbo",
    lst_min,
    lst_max,
    "LST"
)


# ============================================================
# 9. LST 2024
# ============================================================

save_continuous_map(
    lst_2024,
    "Mahbubnagar LST - 2024",
    f"{OUTPUT_DIR}/lst/LST_2024.png",
    "turbo",
    lst_min,
    lst_max,
    "LST"
)


# ============================================================
# 10. LST CHANGE
# ============================================================

lst_limit = np.nanmax(
    np.abs(lst_change)
)

save_continuous_map(
    lst_change,
    "Mahbubnagar LST Change (2024 - 2004)",
    f"{OUTPUT_DIR}/lst/LST_CHANGE.png",
    "RdBu_r",
    -lst_limit,
    lst_limit,
    "LST Change"
)


# ============================================================
# 11. LULC COLORS
# ============================================================

print("\n========================================")
print("CREATING LULC MAPS")
print("========================================")


lulc_colors = [
    "#e9ecef",   # Background
    "#2b83ba",   # Water
    "#33a02c",   # Vegetation
    "#e31a1c",   # Built-up
    "#fdae61"    # Barren
]


lulc_labels = [
    "Background",
    "Water",
    "Vegetation",
    "Built-up",
    "Barren"
]


from matplotlib.colors import ListedColormap

lulc_cmap = ListedColormap(
    lulc_colors
)


# ============================================================
# 12. LULC MAP FUNCTION
# ============================================================

def save_lulc_map(
    data,
    title,
    filename
):

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        data,
        cmap=lulc_cmap,
        vmin=0,
        vmax=4
    )

    plt.title(
        title,
        fontsize=16
    )

    plt.xlabel(
        "Pixel"
    )

    plt.ylabel(
        "Pixel"
    )


    # Custom legend

    handles = []

    for i in range(5):

        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=lulc_colors[i],
                label=lulc_labels[i]
            )
        )


    plt.legend(
        handles=handles,
        loc="upper right"
    )


    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        filename
    )


# ============================================================
# 13. LULC 2004
# ============================================================

save_lulc_map(
    lulc_2004,
    "Mahbubnagar LULC - 2004",
    f"{OUTPUT_DIR}/lulc/LULC_2004.png"
)


# ============================================================
# 14. LULC 2024
# ============================================================

save_lulc_map(
    lulc_2024,
    "Mahbubnagar LULC - 2024",
    f"{OUTPUT_DIR}/lulc/LULC_2024.png"
)


# ============================================================
# 15. LULC CHANGE MAP
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    lulc_changed,
    cmap="gray"
)

plt.title(
    "Mahbubnagar LULC Change (2004 - 2024)",
    fontsize=16
)

plt.xlabel(
    "Pixel"
)

plt.ylabel(
    "Pixel"
)

plt.colorbar(
    label="Change: 1 = Changed, 0 = Unchanged"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lulc/LULC_CHANGE.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    f"{OUTPUT_DIR}/lulc/LULC_CHANGE.png"
)


# ============================================================
# 16. CHANGE STATISTICS
# ============================================================

total_pixels = lulc_changed.size

changed_pixels = np.sum(
    lulc_changed
)

unchanged_pixels = (
    total_pixels - changed_pixels
)

change_percentage = (
    changed_pixels /
    total_pixels
) * 100


print("\n========================================")
print("CHANGE STATISTICS")
print("========================================")

print(
    "Total pixels:",
    total_pixels
)

print(
    "Changed pixels:",
    changed_pixels
)

print(
    "Unchanged pixels:",
    unchanged_pixels
)

print(
    "LULC Change Percentage:",
    change_percentage,
    "%"
)


# ============================================================
# 17. FINAL
# ============================================================

print("\n========================================")
print("SPATIAL VISUALIZATION COMPLETED")
print("========================================")

print(
    "Maps saved inside:"
)

print(
    OUTPUT_DIR
)
print("\nGenerated:")
print("NDVI_2004.png")
print("NDVI_2024.png")
print("NDVI_CHANGE.png")
print("LST_2004.png")
print("LST_2024.png")
print("LST_CHANGE.png")
print("LULC_2004.png")
print("LULC_2024.png")
print("LULC_CHANGE.png")