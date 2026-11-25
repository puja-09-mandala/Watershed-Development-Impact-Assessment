"""Map figures, charts, and architecture diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import LULC_CLASSES, PATHS, STUDY_AREA

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
})


def _save(fig, name: str) -> Path:
    PATHS["figures"].mkdir(parents=True, exist_ok=True)
    PATHS["charts"].mkdir(parents=True, exist_ok=True)
    PATHS["maps"].mkdir(parents=True, exist_ok=True)
    path = PATHS["figures"] / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_raster(array, title, cmap, vmin, vmax, cbar_label, filename, mask=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    data = np.array(array, dtype=float)
    if mask is not None:
        data = np.where(mask, data, np.nan)
    finite = data[np.isfinite(data)]
    if vmin is None and finite.size:
        vmin = float(np.percentile(finite, 2))
    if vmax is None and finite.size:
        vmax = float(np.percentile(finite, 98))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    ax.set_aspect("equal")
    return _save(fig, filename)


def plot_lulc(array, title, filename):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#d62728", "#c4a35a"])
    display = np.where(np.isfinite(array), array, np.nan)
    im = ax.imshow(display, cmap=cmap, vmin=1, vmax=4)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4], fraction=0.046, pad=0.04)
    cb.ax.set_yticklabels([LULC_CLASSES.get(i, str(i)) for i in range(1, 5)])
    return _save(fig, filename)


def comparison_bars(stats_rows, keys, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    labels = []
    means = []
    for key in keys:
        row = next((r for r in stats_rows if r.get("name") == key), None)
        if row and row.get("n", 0) > 0:
            labels.append(key.replace("_", " "))
            means.append(row["mean"])
    if not labels:
        ax.text(0.5, 0.5, "No statistics available", ha="center", va="center")
    else:
        ax.bar(labels, means, color=["#2ca02c", "#1f77b4"][: len(labels)])
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = PATHS["charts"] / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def histogram(array, title, xlabel, filename, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No valid pixels", ha="center", va="center")
    else:
        ax.hist(finite, bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Pixel count")
    ax.set_title(title)
    fig.tight_layout()
    path = PATHS["charts"] / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def error_scatter(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true.ravel()[::20], y_pred.ravel()[::20], s=6, alpha=0.35)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, color="black", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    path = PATHS["charts"] / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def architecture_boxes(ax, items, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(items) + 1)
    ax.axis("off")
    ax.set_title(title)
    for i, text in enumerate(reversed(items)):
        y = i + 0.4
        box = FancyBboxPatch((1.5, y), 7, 0.7, boxstyle="round,pad=0.08,rounding_size=0.15",
                             facecolor="#e8f1fb", edgecolor="#1f4e79", lw=1.2)
        ax.add_patch(box)
        ax.text(5, y + 0.35, text, ha="center", va="center", fontsize=9)
        if i < len(items) - 1:
            ax.annotate("", xy=(5, y + 0.78), xytext=(5, y + 1.0),
                        arrowprops=dict(arrowstyle="->", color="#1f4e79"))


def figure_study_area(boundary_coords=None):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_title("Figure 1. Rajapeta study area, Wanaparthy, Telangana")
    if boundary_coords is not None:
        xs, ys = zip(*boundary_coords)
        ax.fill(xs, ys, color="#cfe8c6", alpha=0.7, edgecolor="#2c5f2d", lw=2, label="Rajapeta village")
        ax.plot(xs, ys, color="#2c5f2d")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
    else:
        ax.text(
            0.5, 0.55,
            f"{STUDY_AREA['village']}\n{STUDY_AREA['mandal']}\n"
            f"{STUDY_AREA['district']} District, {STUDY_AREA['state']}\n\n"
            "Replace data/boundary/rajapeta_boundary.geojson\n"
            "with the official village polygon.",
            ha="center", va="center", fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, "Figure_01_Study_Area.png")


def figure_system_architecture():
    items = [
        "Satellite data (Landsat 5 2004 / Landsat 8 2024)",
        "Preprocessing, cloud mask, median composite, clip",
        "NDVI + LST + LULC GeoTIFFs",
        "128×128 patches, temporal stack, normalize, split",
        "CNN spatial features → LSTM temporal learning",
        "Reconstruction, change detection, statistics",
        "Mistral multimodal LLM interpretation",
        "Automated report, paper, 5-page Web GIS",
    ]
    fig, ax = plt.subplots(figsize=(8, 9))
    architecture_boxes(ax, items, "Figure 2. Overall system architecture")
    return _save(fig, "Figure_02_System_Architecture.png")


def figure_preprocessing():
    items = [
        "Load GeoTIFF",
        "Verify CRS, resolution, dimensions, NoData",
        "Invalid-pixel handling and alignment",
        "Clip to Rajapeta village extent",
        "128×128 patches and temporal stacking",
        "Normalization and train/test split",
    ]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    architecture_boxes(ax, items, "Figure 3. Remote-sensing preprocessing workflow")
    return _save(fig, "Figure_03_Preprocessing_Workflow.png")


def figure_cnn_lstm():
    items = [
        "Input (samples, 2, 128, 128, 2) — NDVI & LST, 2004→2024",
        "TimeDistributed Conv2D + ReLU + MaxPool",
        "TimeDistributed Conv2D + ReLU + MaxPool",
        "LSTM temporal representation",
        "Dense spatial restore + upsampling decoder",
        "Reconstructed 128×128 NDVI/LST (optional LULC softmax head)",
    ]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    architecture_boxes(ax, items, "Figure 4. CNN-LSTM architecture")
    return _save(fig, "Figure_04_CNN_LSTM_Architecture.png")


def figure_llm_workflow():
    items = [
        "NDVI / LST / LULC maps and change maps",
        "Statistical summaries and CNN-LSTM metrics",
        "Mistral multimodal encoder (maps + numbers)",
        "Interpretation: vegetation, temperature, LULC, model error",
        "Scientific discussion, limitations, conclusion",
        "PDF / DOCX / Markdown report",
    ]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    architecture_boxes(ax, items, "Figure 5. Mistral multimodal LLM workflow")
    return _save(fig, "Figure_05_LLM_Workflow.png")


def lulc_area_chart(areas_2004, areas_2024, filename="lulc_area_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    if not areas_2004 and not areas_2024:
        ax.text(0.5, 0.5, "LULC GeoTIFF data not available yet.", ha="center", va="center")
    else:
        keys = sorted(set(list(areas_2004) + list(areas_2024)))
        names = [ (areas_2004.get(k) or areas_2024.get(k))["name"] for k in keys ]
        a = [areas_2004.get(k, {}).get("area_ha", 0) for k in keys]
        b = [areas_2024.get(k, {}).get("area_ha", 0) for k in keys]
        x = np.arange(len(keys))
        ax.bar(x - 0.18, a, 0.36, label="2004")
        ax.bar(x + 0.18, b, 0.36, label="2024")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylabel("Area (ha)")
        ax.legend()
    ax.set_title("LULC area comparison, 2004 vs 2024")
    fig.tight_layout()
    path = PATHS["charts"] / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
