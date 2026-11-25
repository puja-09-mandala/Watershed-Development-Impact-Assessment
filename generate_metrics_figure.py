"""
IEEE-quality evaluation metrics figure
Rajapeta Watershed CNN-LSTM + LULC Assessment
Run: python generate_metrics_figure.py
Output: outputs/figures/Figure_17_Evaluation_Metrics.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Real values from classification_report.txt & stats.json ──────────────────

# Panel A — CNN-LSTM classification (from classification_report.txt)
CLASSES   = ["Background", "Water", "Vegetation", "Built-up", "Barren"]
PRECISION = [0.95, 0.54, 0.77, 0.13, 0.41]
RECALL    = [0.73, 0.09, 0.97, 0.05, 0.65]
F1        = [0.82, 0.16, 0.85, 0.07, 0.51]
SUPPORT   = [1491130, 5797, 1310320, 3006, 24179]

OVERALL_ACC = 0.8352
WEIGHTED_P  = 0.86
WEIGHTED_R  = 0.84
WEIGHTED_F1 = 0.83
IOU         = 0.7167

# Panel B — CNN-LSTM reconstruction (from stats.json cnn_lstm)
MAE  = 0.1710
RMSE = 0.2412
MSE  = 0.0582
MAE_NDVI  = 0.1823;  RMSE_NDVI  = 0.2573
MAE_LST   = 0.1597;  RMSE_LST   = 0.2240

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#1a5276"
TEAL   = "#148f77"
ORANGE = "#d35400"
GREY   = "#7f8c8d"
RED    = "#922b21"
GREEN  = "#1e8449"
LIGHT  = "#eaf2ff"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e5e5e5",
    "grid.linewidth":     0.5,
    "figure.dpi":         300,
})

fig = plt.figure(figsize=(14, 7), facecolor="white")
fig.suptitle(
    "Evaluation Metrics of the Proposed Watershed Assessment Framework\n"
    "Rajapeta Village Watershed, Wanaparthy, Telangana, India  |  2004–2024",
    fontsize=10, fontweight="bold", y=0.98, color="#1a1a1a"
)

gs = gridspec.GridSpec(
    2, 4,
    figure=fig,
    left=0.06, right=0.97,
    top=0.88,  bottom=0.10,
    wspace=0.45, hspace=0.55
)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A  (left 3 columns, top row) — Per-class bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[0, :3])

x   = np.arange(len(CLASSES))
w   = 0.26
b1  = ax_bar.bar(x - w, PRECISION, w, label="Precision", color=BLUE,   alpha=0.88, zorder=3)
b2  = ax_bar.bar(x,     RECALL,    w, label="Recall",    color=TEAL,   alpha=0.88, zorder=3)
b3  = ax_bar.bar(x + w, F1,        w, label="F1-Score",  color=ORANGE, alpha=0.88, zorder=3)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6, color="#333")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(CLASSES, fontsize=7.5)
ax_bar.set_ylim(0, 1.12)
ax_bar.set_ylabel("Score")
ax_bar.set_title("Panel A — CNN-LSTM Per-Class Classification Performance",
                 fontweight="bold", loc="left", pad=6)
ax_bar.legend(loc="upper right", fontsize=7, framealpha=0.9)
ax_bar.axhline(0.5, color="#bbb", lw=0.8, ls="--", zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A  (right column, top row) — Overall metric cards
# ─────────────────────────────────────────────────────────────────────────────
ax_cards = fig.add_subplot(gs[0, 3])
ax_cards.axis("off")
ax_cards.set_title("Overall Metrics", fontweight="bold", fontsize=8, pad=6)

cards = [
    ("Overall Accuracy", f"{OVERALL_ACC*100:.1f}%", GREEN),
    ("Weighted Precision", f"{WEIGHTED_P:.2f}",     BLUE),
    ("Weighted Recall",    f"{WEIGHTED_R:.2f}",     TEAL),
    ("Weighted F1-Score",  f"{WEIGHTED_F1:.2f}",    ORANGE),
    ("IoU (mean)",         f"{IOU:.4f}",            GREY),
]

for i, (label, val, col) in enumerate(cards):
    y0 = 0.85 - i * 0.19
    ax_cards.add_patch(mpatches.FancyBboxPatch(
        (0.02, y0 - 0.07), 0.96, 0.16,
        boxstyle="round,pad=0.02",
        facecolor=LIGHT, edgecolor=col, linewidth=1.2,
        transform=ax_cards.transAxes, clip_on=False
    ))
    ax_cards.text(0.50, y0 + 0.02, val,
                  ha="center", va="center",
                  fontsize=11, fontweight="bold", color=col,
                  transform=ax_cards.transAxes)
    ax_cards.text(0.50, y0 - 0.04, label,
                  ha="center", va="center",
                  fontsize=6.5, color="#444",
                  transform=ax_cards.transAxes)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B  (bottom row, col 0-1) — Reconstruction metric grouped bars
# ─────────────────────────────────────────────────────────────────────────────
ax_rec = fig.add_subplot(gs[1, :2])

groups  = ["MAE", "RMSE", "MSE"]
overall = [MAE,  RMSE,  MSE]
ndvi_v  = [MAE_NDVI,  RMSE_NDVI,  0.0662]
lst_v   = [MAE_LST,   RMSE_LST,   0.0502]

xr = np.arange(len(groups))
wr = 0.25
ax_rec.bar(xr - wr, overall, wr, label="Overall", color=BLUE,   alpha=0.88, zorder=3)
ax_rec.bar(xr,      ndvi_v,  wr, label="NDVI",    color=GREEN,  alpha=0.88, zorder=3)
ax_rec.bar(xr + wr, lst_v,   wr, label="LST",     color=RED,    alpha=0.88, zorder=3)

for bars in (
    ax_rec.containers[0],
    ax_rec.containers[1],
    ax_rec.containers[2],
):
    for bar in bars:
        h = bar.get_height()
        ax_rec.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=5.8, color="#333")

ax_rec.set_xticks(xr)
ax_rec.set_xticklabels(groups, fontsize=8)
ax_rec.set_ylabel("Error Value")
ax_rec.set_ylim(0, 0.32)
ax_rec.set_title("Panel B — CNN-LSTM Reconstruction Error Metrics\n"
                 "(n_train=5 patches, n_test=2 patches, 128×128 px)",
                 fontweight="bold", loc="left", pad=6, fontsize=8)
ax_rec.legend(loc="upper right", fontsize=7, framealpha=0.9)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B  (bottom row, col 2) — Radar / spider for NDVI vs LST channels
# ─────────────────────────────────────────────────────────────────────────────
ax_rad = fig.add_subplot(gs[1, 2], polar=True)

metrics_r = ["MAE", "MSE", "RMSE"]
ndvi_norm = [MAE_NDVI/0.30, 0.0662/0.10, RMSE_NDVI/0.35]
lst_norm  = [MAE_LST/0.30,  0.0502/0.10, RMSE_LST/0.35]

angles = np.linspace(0, 2*np.pi, len(metrics_r), endpoint=False).tolist()
angles += angles[:1]
ndvi_norm += ndvi_norm[:1]
lst_norm  += lst_norm[:1]

ax_rad.plot(angles, ndvi_norm, "o-", color=GREEN, lw=1.5, label="NDVI")
ax_rad.fill(angles, ndvi_norm, color=GREEN, alpha=0.15)
ax_rad.plot(angles, lst_norm,  "s-", color=RED,   lw=1.5, label="LST")
ax_rad.fill(angles, lst_norm,  color=RED,   alpha=0.15)

ax_rad.set_xticks(angles[:-1])
ax_rad.set_xticklabels(metrics_r, fontsize=7)
ax_rad.set_yticks([0.25, 0.50, 0.75, 1.0])
ax_rad.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=5.5)
ax_rad.set_ylim(0, 1.1)
ax_rad.set_title("Channel\nComparison\n(normalised)",
                 fontsize=7, fontweight="bold", pad=10)
ax_rad.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=6.5, framealpha=0.9)
ax_rad.grid(color="#ddd", linewidth=0.6)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B  (bottom row, col 3) — Reconstruction metric cards
# ─────────────────────────────────────────────────────────────────────────────
ax_rc2 = fig.add_subplot(gs[1, 3])
ax_rc2.axis("off")
ax_rc2.set_title("Reconstruction\nSummary", fontweight="bold", fontsize=8, pad=6)

rec_cards = [
    ("MAE  (Overall)",  f"{MAE:.4f}",  BLUE),
    ("RMSE (Overall)",  f"{RMSE:.4f}", RED),
    ("MSE  (Overall)",  f"{MSE:.4f}",  GREY),
    ("MAE  (NDVI)",     f"{MAE_NDVI:.4f}", GREEN),
    ("MAE  (LST)",      f"{MAE_LST:.4f}",  ORANGE),
]

for i, (label, val, col) in enumerate(rec_cards):
    y0 = 0.85 - i * 0.19
    ax_rc2.add_patch(mpatches.FancyBboxPatch(
        (0.02, y0 - 0.07), 0.96, 0.16,
        boxstyle="round,pad=0.02",
        facecolor=LIGHT, edgecolor=col, linewidth=1.2,
        transform=ax_rc2.transAxes, clip_on=False
    ))
    ax_rc2.text(0.50, y0 + 0.02, val,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=col,
                transform=ax_rc2.transAxes)
    ax_rc2.text(0.50, y0 - 0.04, label,
                ha="center", va="center",
                fontsize=6.5, color="#444",
                transform=ax_rc2.transAxes)

# ─────────────────────────────────────────────────────────────────────────────
# Panel labels A / B
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.01, 0.91, "(A)", fontsize=10, fontweight="bold", color=BLUE)
fig.text(0.01, 0.46, "(B)", fontsize=10, fontweight="bold", color=RED)

# ─────────────────────────────────────────────────────────────────────────────
# Footer note
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    "Panel A: CNN-LSTM pixel-wise LULC classification metrics (weighted avg).  "
    "Panel B: CNN-LSTM spatio-temporal reconstruction errors (prototype, n_test=2 patches).  "
    "Sensor: Landsat 5 (2004) & Landsat 8 (2024), 30 m.",
    ha="center", fontsize=6, color="#666", style="italic"
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_path = OUT / "Figure_17_Evaluation_Metrics.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
