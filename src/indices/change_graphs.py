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
# 2. LOAD DATA
# ============================================================

print("========================================")
print("CREATING CHANGE DETECTION GRAPHS")
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


# ============================================================
# 3. CALCULATE MEAN VALUES
# ============================================================

mean_ndvi_2004 = np.nanmean(ndvi_2004)
mean_ndvi_2024 = np.nanmean(ndvi_2024)

mean_lst_2004 = np.nanmean(lst_2004)
mean_lst_2024 = np.nanmean(lst_2024)


# ============================================================
# 4. NDVI COMPARISON GRAPH
# ============================================================

print("\nCreating NDVI comparison graph...")

years = ["2004", "2024"]

ndvi_values = [
    mean_ndvi_2004,
    mean_ndvi_2024
]

plt.figure(figsize=(8, 5))

plt.bar(
    years,
    ndvi_values
)

plt.xlabel("Year")
plt.ylabel("Mean NDVI")

plt.title(
    "Mean NDVI Comparison: 2004 vs 2024"
)

for i, value in enumerate(ndvi_values):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ndvi_comparison.png",
    dpi=300
)

plt.close()

print("NDVI graph saved.")


# ============================================================
# 5. LST COMPARISON GRAPH
# ============================================================

print("Creating LST comparison graph...")

lst_values = [
    mean_lst_2004,
    mean_lst_2024
]

plt.figure(figsize=(8, 5))

plt.bar(
    years,
    lst_values
)

plt.xlabel("Year")
plt.ylabel("Mean LST")

plt.title(
    "Mean LST Comparison: 2004 vs 2024"
)

for i, value in enumerate(lst_values):

    plt.text(
        i,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lst_comparison.png",
    dpi=300
)

plt.close()

print("LST graph saved.")


# ============================================================
# 6. LULC CLASS COMPARISON
# ============================================================

class_names = [
    "Background",
    "Water",
    "Vegetation",
    "Built-up",
    "Barren"
]

class_ids = [0, 1, 2, 3, 4]

counts_2004 = []
counts_2024 = []

for class_id in class_ids:

    count_2004 = np.sum(
        lulc_2004 == class_id
    )

    count_2024 = np.sum(
        lulc_2024 == class_id
    )

    counts_2004.append(count_2004)
    counts_2024.append(count_2024)


# ============================================================
# 7. LULC COMPARISON BAR GRAPH
# ============================================================

print("Creating LULC comparison graph...")

x = np.arange(
    len(class_names)
)

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    counts_2004,
    width,
    label="2004"
)

plt.bar(
    x + width / 2,
    counts_2024,
    width,
    label="2024"
)

plt.xlabel("LULC Class")
plt.ylabel("Number of Pixels")

plt.title(
    "LULC Class Comparison: 2004 vs 2024"
)

plt.xticks(
    x,
    class_names,
    rotation=30
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lulc_comparison.png",
    dpi=300
)

plt.close()

print("LULC graph saved.")


# ============================================================
# 8. LULC CHANGE GRAPH
# ============================================================

print("Creating LULC change graph...")

changes = []

for i in range(len(class_names)):

    change = (
        counts_2024[i]
        - counts_2004[i]
    )

    changes.append(change)


plt.figure(figsize=(10, 6))

plt.bar(
    class_names,
    changes
)

plt.xlabel("LULC Class")

plt.ylabel(
    "Change in Number of Pixels"
)

plt.title(
    "LULC Change: 2004 → 2024"
)

plt.xticks(
    rotation=30
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/lulc_change_comparison.png",
    dpi=300
)

plt.close()

print("LULC change graph saved.")


# ============================================================
# 9. OVERALL CHANGE PERCENTAGE GRAPH
# ============================================================

changed_pixels = np.sum(
    lulc_2004 != lulc_2024
)

total_pixels = lulc_2004.size

change_percentage = (
    changed_pixels /
    total_pixels
) * 100

unchanged_percentage = (
    100 - change_percentage
)


plt.figure(figsize=(7, 5))

values = [
    unchanged_percentage,
    change_percentage
]

labels = [
    "Unchanged",
    "Changed"
]

plt.bar(
    labels,
    values
)

plt.ylabel("Percentage (%)")

plt.title(
    "Overall LULC Change: 2004 → 2024"
)

for i, value in enumerate(values):

    plt.text(
        i,
        value,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/overall_lulc_change.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n========================================")
print("CHANGE GRAPH SUMMARY")
print("========================================")

print(
    f"Mean NDVI 2004 : {mean_ndvi_2004:.4f}"
)

print(
    f"Mean NDVI 2024 : {mean_ndvi_2024:.4f}"
)

print(
    f"NDVI Change    : "
    f"{mean_ndvi_2024 - mean_ndvi_2004:.4f}"
)

print()

print(
    f"Mean LST 2004  : {mean_lst_2004:.4f}"
)

print(
    f"Mean LST 2024  : {mean_lst_2024:.4f}"
)

print(
    f"LST Change     : "
    f"{mean_lst_2024 - mean_lst_2004:.4f}"
)

print()

print(
    f"LULC Changed   : "
    f"{change_percentage:.2f}%"
)

print(
    f"LULC Unchanged : "
    f"{unchanged_percentage:.2f}%"
)


# ============================================================
# 11. FINAL
# ============================================================

print("\n========================================")
print("ALL GRAPHS CREATED SUCCESSFULLY")
print("========================================")

print("\nCheck this folder:")

print(
    "outputs/change_detection/"
)

print("\nGenerated graphs:")

print("1. ndvi_comparison.png")
print("2. lst_comparison.png")
print("3. lulc_comparison.png")
print("4. lulc_change_comparison.png")
print("5. overall_lulc_change.png")