import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. PATHS
# ============================================================

DATA_DIR = "data/processed"

OUTPUT_DIR = "outputs/change_detection"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD 2004 AND 2024 DATA
# ============================================================

print("========================================")
print("LOADING 2004 AND 2024 DATA")
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
# 3. NDVI CHANGE
# ============================================================

print("\nCalculating NDVI change...")

ndvi_change = ndvi_2024 - ndvi_2004

print("Mean NDVI 2004:", np.nanmean(ndvi_2004))
print("Mean NDVI 2024:", np.nanmean(ndvi_2024))
print("Mean NDVI change:", np.nanmean(ndvi_change))


# ============================================================
# 4. LST CHANGE
# ============================================================

print("\nCalculating LST change...")

lst_change = lst_2024 - lst_2004

print("Mean LST 2004:", np.nanmean(lst_2004))
print("Mean LST 2024:", np.nanmean(lst_2024))
print("Mean LST change:", np.nanmean(lst_change))


# ============================================================
# 5. LULC CHANGE
# ============================================================

print("\nCalculating LULC change...")

lulc_change = lulc_2024 != lulc_2004

changed_pixels = np.sum(lulc_change)

total_pixels = lulc_change.size

change_percentage = (
    changed_pixels / total_pixels
) * 100

print("Changed pixels:", changed_pixels)

print(
    "LULC change percentage:",
    change_percentage
)


# ============================================================
# 6. LULC CLASS AREA
# ============================================================

class_names = {
    0: "Background",
    1: "Water",
    2: "Vegetation",
    3: "Built-up",
    4: "Barren"
}


print("\n========================================")
print("LULC AREA COMPARISON")
print("========================================")

for class_id, name in class_names.items():

    count_2004 = np.sum(
        lulc_2004 == class_id
    )

    count_2024 = np.sum(
        lulc_2024 == class_id
    )

    print(
        name,
        "2004:",
        count_2004,
        "| 2024:",
        count_2024,
        "| Change:",
        count_2024 - count_2004
    )


# ============================================================
# 7. SAVE NUMPY RESULTS
# ============================================================

np.save(
    f"{OUTPUT_DIR}/ndvi_change.npy",
    ndvi_change
)

np.save(
    f"{OUTPUT_DIR}/lst_change.npy",
    lst_change
)

np.save(
    f"{OUTPUT_DIR}/lulc_change.npy",
    lulc_change
)


# ============================================================
# 8. NDVI CHANGE MAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    ndvi_change,
    cmap="RdYlGn"
)

plt.colorbar(
    label="NDVI Change"
)

plt.title(
    "NDVI Change: 2004 → 2024"
)

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ndvi_change.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. LST CHANGE MAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    lst_change,
    cmap="coolwarm"
)

plt.colorbar(
    label="Temperature Change (°C)"
)

plt.title(
    "LST Change: 2004 → 2024"
)

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lst_change.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. LULC CHANGE MAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    lulc_change
)

plt.colorbar(
    label="Changed Pixel"
)

plt.title(
    "LULC Change: 2004 → 2024"
)

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lulc_change.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. FINAL
# ============================================================

print("\n========================================")
print("CHANGE DETECTION COMPLETED")
print("========================================")

print(
    "Results saved in:",
    OUTPUT_DIR
)

print("\nGenerated files:")

print("ndvi_change.npy")
print("lst_change.npy")
print("lulc_change.npy")

print("ndvi_change.png")
print("lst_change.png")
print("lulc_change.png")