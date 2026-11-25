"""End-to-end Rajapeta watershed pipeline."""
from __future__ import annotations
import json
import traceback
from pathlib import Path
import numpy as np
from config import EXPECTED_RASTERS, LULC_CLASSES, PATHS, PATCH, STUDY_AREA, ensure_directories
from src.change_detection import lst_change_summary, lulc_change_summary, ndvi_change_summary
from src.cnn_lstm import mosaic_predictions, train_and_evaluate
from src.io_utils import discover_inputs, load_raster, write_raster
from src.llm_interpreter import interpret
from src.lst import ensure_celsius, lst_change, write_lst_products
from src.lulc import class_areas, load_training_samples, transition_matrix, write_lulc_change
from src.ndvi import ndvi_change, write_ndvi_products
from src.trends import compute_trends
from src.paper_generator import write_paper
from src.preprocessing import (
    align_group,
    extract_patches,
    normalize_pair,
    verify_and_report,
)
from src.report_generator import generate_reports
from src.statistics import (
    derived_change_stats,
    histogram_data,
    paired_histogram,
    raster_stats,
    save_stats,
)
from src.visualization import (
    comparison_bars,
    error_scatter,
    figure_cnn_lstm,
    figure_llm_workflow,
    figure_preprocessing,
    figure_study_area,
    figure_system_architecture,
    histogram,
    lulc_area_chart,
    plot_lulc,
    plot_raster,
)
from src.web_export import raster_to_png, write_web_manifest


def _load_boundary_coords():
    geojson_path = PATHS["boundary"] / "rajapeta_boundary.geojson"
    if not geojson_path.exists():
        return None, None
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    geom = data["features"][0]["geometry"] if data.get("features") else data.get("geometry")
    if not geom:
        return data, None
    coords = geom["coordinates"]
    ring = coords[0][0] if geom["type"] == "MultiPolygon" else coords[0]
    return data, [(c[0], c[1]) for c in ring]
def _unavailable_layer(name: str, kind: str) -> dict:
    return {"name": name, "available": False, "kind": kind, "png": None}
def _scatter_sample(
    actual: np.ndarray, predicted: np.ndarray, label: str, max_points: int = 1500
) -> dict:
    """Down-sampled (actual, predicted) pairs from the real CNN-LSTM output."""
    valid = np.isfinite(actual) & np.isfinite(predicted)
    a = actual[valid].ravel()
    p = predicted[valid].ravel()
    if a.size == 0:
        return {"label": label, "actual": [], "predicted": [], "n": 0}
    step = max(1, a.size // max_points)
    return {
        "label": label,
        "actual": [round(float(v), 4) for v in a[::step]],
        "predicted": [round(float(v), 4) for v in p[::step]],
        "n": int(a.size),
    }


def run() -> dict:
    ensure_directories()
    log = []
    found = discover_inputs()
    log.append({"discovered": {k: (str(v) if v else None) for k, v in found.items()}})

    metadata = {}
    rasters = {}
    for key, path in found.items():
        if path is None:
            continue
        bundle = load_raster(path)
        if key.startswith("lst_") and "change" not in key:
            bundle.array = ensure_celsius(bundle.array)
        rasters[key] = bundle
        metadata[key] = verify_and_report(bundle)

    keys_needed = [k for k in ("ndvi_2004", "ndvi_2024", "lst_2004", "lst_2024") if k in rasters]
    if len(keys_needed) >= 2:
        aligned = align_group([rasters[k] for k in keys_needed])
        for k, b in zip(keys_needed, aligned):
            rasters[k] = b

    stats_rows = []
    extras = {
        "lulc_note": "LULC GeoTIFF data not available yet.",
        "cnn_note": (
            "Spatio-temporal CNN-LSTM prototype for representation learning, "
            "reconstruction, and change-analysis support. Not trained in this run."
        ),
        "study_area": STUDY_AREA,
        "discovered_files": {k: (str(v) if v else None) for k, v in found.items()},
    }
    change_summaries = {}
    lulc_areas = {"2004": {}, "2024": {}}
    cnn_metrics = {"trained": False}

    # Supervised LULC generation (Random Forest; runs only when labeled
    # reference samples exist — never fabricated otherwise).
    from src.lulc import run_lulc_pipeline

    lulc_result = run_lulc_pipeline(rasters)
    if lulc_result.get("produced"):
        from src.io_utils import load_raster as _load_lulc

        rasters["lulc_2004"] = _load_lulc(PATHS["processed"] / EXPECTED_RASTERS["lulc_2004"])
        rasters["lulc_2024"] = _load_lulc(PATHS["processed"] / EXPECTED_RASTERS["lulc_2024"])
    if lulc_result["note"]:
        extras["lulc_note"] = lulc_result["note"]

    if "ndvi_2004" in rasters and "ndvi_2024" in rasters:
        write_ndvi_products(rasters["ndvi_2004"], rasters["ndvi_2024"])
        if "ndvi_change" not in rasters:
            arr = ndvi_change(rasters["ndvi_2004"].array, rasters["ndvi_2024"].array)
            rasters["ndvi_change"] = rasters["ndvi_2004"].with_array(
                arr, PATHS["processed"] / EXPECTED_RASTERS["ndvi_change"]
            )
        change_summaries["ndvi"] = ndvi_change_summary(rasters["ndvi_change"].array)

    if "lst_2004" in rasters and "lst_2024" in rasters:
        write_lst_products(rasters["lst_2004"], rasters["lst_2024"])
        if "lst_change" not in rasters:
            arr = lst_change(rasters["lst_2004"].array, rasters["lst_2024"].array)
            rasters["lst_change"] = rasters["lst_2004"].with_array(
                arr, PATHS["processed"] / EXPECTED_RASTERS["lst_change"]
            )
        change_summaries["lst"] = lst_change_summary(rasters["lst_change"].array)

    if "lulc_2004" in rasters and "lulc_2024" in rasters:
        change_path = write_lulc_change(rasters["lulc_2004"], rasters["lulc_2024"])
        rasters["lulc_change"] = load_raster(change_path)
        lulc_areas["2004"] = class_areas(rasters["lulc_2004"].array)
        lulc_areas["2024"] = class_areas(rasters["lulc_2024"].array)
        trans = transition_matrix(rasters["lulc_2004"].array, rasters["lulc_2024"].array)
        trans.to_csv(PATHS["stats"] / "lulc_transition_matrix.csv")
        change_summaries["lulc"] = lulc_change_summary(
            rasters["lulc_2004"].array, rasters["lulc_2024"].array
        )
        if lulc_result.get("produced"):
            extras["lulc_note"] = (
                lulc_result["note"]
                + " Class areas and the transition matrix were computed from pixel labels."
            )
        else:
            extras["lulc_note"] = (
                "LULC rasters were ingested. Class areas and the transition matrix were computed "
                "from pixel labels. Accuracy metrics are omitted unless independent reference samples exist."
            )
            if load_training_samples() is None:
                extras["lulc_note"] += (
                    " No training-sample CSV was found, so kappa and overall accuracy were not computed."
                )

        # Class-wise area change 2004 -> 2024 (all five classes).
        area_change = {}
        for cid, cname in LULC_CLASSES.items():
            a04 = float(lulc_areas["2004"].get(cid, {}).get("area_ha", 0.0))
            a24 = float(lulc_areas["2024"].get(cid, {}).get("area_ha", 0.0))
            delta = a24 - a04
            area_change[str(cid)] = {
                "name": cname,
                "area_2004_ha": a04,
                "area_2024_ha": a24,
                "delta_ha": delta,
                "delta_percent": (100.0 * delta / a04) if a04 else None,
            }
        extras["lulc_area_change"] = area_change

    for name, bundle in rasters.items():
        stats_rows.append(raster_stats(bundle.array, name))

    # Architecture figures always
    boundary_gj, coords = _load_boundary_coords()
    figure_study_area(coords)
    figure_system_architecture()
    figure_preprocessing()
    figure_cnn_lstm()
    figure_llm_workflow()

    # Thematic figures
    if "ndvi_2004" in rasters:
        plot_raster(rasters["ndvi_2004"].array, "Figure 6. NDVI 2004 — Rajapeta", "YlGn", -0.2, 0.8, "NDVI", "Figure_06_NDVI_2004.png")
    if "ndvi_2024" in rasters:
        plot_raster(rasters["ndvi_2024"].array, "Figure 7. NDVI 2024 — Rajapeta", "YlGn", -0.2, 0.8, "NDVI", "Figure_07_NDVI_2024.png")
    if "ndvi_change" in rasters:
        plot_raster(rasters["ndvi_change"].array, "Figure 8. NDVI change 2004–2024", "RdBu", -0.4, 0.4, "ΔNDVI", "Figure_08_NDVI_Change.png")
    if "lst_2004" in rasters:
        plot_raster(rasters["lst_2004"].array, "Figure 9. LST 2004 — Rajapeta", "inferno", None, None, "°C", "Figure_09_LST_2004.png")
    if "lst_2024" in rasters:
        plot_raster(rasters["lst_2024"].array, "Figure 10. LST 2024 — Rajapeta", "inferno", None, None, "°C", "Figure_10_LST_2024.png")
    if "lst_change" in rasters:
        plot_raster(rasters["lst_change"].array, "Figure 11. LST change 2004–2024", "RdBu_r", None, None, "Δ°C", "Figure_11_LST_Change.png")
    if "lulc_2004" in rasters:
        plot_lulc(rasters["lulc_2004"].array, "Figure 12. LULC 2004 — Rajapeta", "Figure_12_LULC_2004.png")
    if "lulc_2024" in rasters:
        plot_lulc(rasters["lulc_2024"].array, "Figure 13. LULC 2024 — Rajapeta", "Figure_13_LULC_2024.png")
    if "lulc_2004" in rasters and "lulc_2024" in rasters:
        from src.lulc import lulc_change_map

        plot_raster(
            lulc_change_map(rasters["lulc_2004"].array, rasters["lulc_2024"].array),
            "Figure 14. LULC change (1 = class change)",
            "gray",
            0,
            1,
            "Change",
            "Figure_14_LULC_Change.png",
        )

    comparison_bars(stats_rows, ["ndvi_2004", "ndvi_2024"], "NDVI 2004 vs 2024 (mean)", "NDVI", "ndvi_comparison.png")
    comparison_bars(stats_rows, ["lst_2004", "lst_2024"], "LST 2004 vs 2024 (mean)", "°C", "lst_comparison.png")
    if "ndvi_change" in rasters:
        histogram(rasters["ndvi_change"].array, "NDVI change distribution", "ΔNDVI", "ndvi_change_hist.png", "#2ca02c")
    if "lst_change" in rasters:
        histogram(rasters["lst_change"].array, "LST change distribution", "ΔLST (°C)", "lst_change_hist.png", "#d62728")
    lulc_area_chart(lulc_areas["2004"], lulc_areas["2024"])

    # CNN-LSTM
    pred_ndvi = pred_lst = None
    if all(k in rasters for k in ("ndvi_2004", "ndvi_2024", "lst_2004", "lst_2024")):
        n04, n24, n_scale = normalize_pair(rasters["ndvi_2004"].array, rasters["ndvi_2024"].array)
        t04, t24, t_scale = normalize_pair(rasters["lst_2004"].array, rasters["lst_2024"].array)
        stack = np.stack(
            [
                np.stack([n04, t04], axis=-1),
                np.stack([n24, t24], axis=-1),
            ],
            axis=0,
        )
        patches, origins = extract_patches(stack)
        np.savez_compressed(PATHS["patches"] / "patches.npz", patches=patches, origins=np.array(origins))
        try:
            cnn_metrics = train_and_evaluate(patches)
        except Exception as exc:
            cnn_metrics = {"trained": False, "reason": str(exc), "traceback": traceback.format_exc()}
        if cnn_metrics.get("trained"):
            extras["cnn_note"] = (
                f"{cnn_metrics['role']} Held-out MAE={cnn_metrics['mae']:.4f}, "
                f"RMSE={cnn_metrics['rmse']:.4f}, MSE={cnn_metrics['mse']:.4f} "
                f"(n_test={cnn_metrics['n_test']}). "
                "Limitation: patches are overlapping sliding-window 128x128 crops of a "
                "single small scene, so training and test patches share spatial context; "
                "these metrics are indicative reconstruction errors, not "
                "independent-validation scores."
            )
            data = np.load(PATHS["models"] / "test_predictions.npz")
            error_scatter(data["y_true"][..., 0], data["y_pred"][..., 0], "Actual vs predicted NDVI (normalized patches)", "cnn_ndvi_scatter.png")
            error_scatter(data["y_true"][..., 1], data["y_pred"][..., 1], "Actual vs predicted LST (normalized patches)", "cnn_lst_scatter.png")
            from tensorflow.keras.models import load_model

            model = load_model(cnn_metrics["model_path"])
            full_pred = model.predict(patches, verbose=0)
            mosaic = mosaic_predictions(
                full_pred,
                origins,
                rasters["ndvi_2024"].array.shape[0],
                rasters["ndvi_2024"].array.shape[1],
            )
            pred_ndvi = mosaic[..., 0] * (n_scale["vmax"] - n_scale["vmin"]) + n_scale["vmin"]
            pred_lst = mosaic[..., 1] * (t_scale["vmax"] - t_scale["vmin"]) + t_scale["vmin"]
            write_raster(PATHS["processed"] / "Predicted_NDVI_2024.tif", pred_ndvi, rasters["ndvi_2024"].profile)
            write_raster(PATHS["processed"] / "Predicted_LST_2024.tif", pred_lst, rasters["lst_2024"].profile)
            ndvi_err = pred_ndvi - rasters["ndvi_2024"].array
            lst_err = pred_lst - rasters["lst_2024"].array
            write_raster(PATHS["processed"] / "NDVI_prediction_error.tif", ndvi_err, rasters["ndvi_2024"].profile)
            write_raster(PATHS["processed"] / "LST_prediction_error.tif", lst_err, rasters["lst_2024"].profile)
            plot_raster(pred_ndvi, "Predicted NDVI 2024", "YlGn", -0.2, 0.8, "NDVI", "Predicted_NDVI_2024.png")
            plot_raster(rasters["ndvi_2024"].array, "Actual NDVI 2024", "YlGn", -0.2, 0.8, "NDVI", "Actual_NDVI_2024.png")
            plot_raster(pred_lst, "Predicted LST 2024", "inferno", None, None, "°C", "Predicted_LST_2024.png")
            plot_raster(rasters["lst_2024"].array, "Actual LST 2024", "inferno", None, None, "°C", "Actual_LST_2024.png")
            # Combined actual vs predicted figures
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
            axes[0].imshow(rasters["ndvi_2024"].array, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[0].set_title("Actual NDVI 2024")
            axes[1].imshow(pred_ndvi, cmap="YlGn", vmin=-0.2, vmax=0.8)
            axes[1].set_title("Predicted NDVI 2024")
            fig.suptitle("Figure 15. Actual vs predicted NDVI")
            fig.tight_layout()
            fig.savefig(PATHS["figures"] / "Figure_15_Actual_vs_Predicted_NDVI.png", bbox_inches="tight")
            plt.close(fig)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
            axes[0].imshow(rasters["lst_2024"].array, cmap="inferno")
            axes[0].set_title("Actual LST 2024")
            axes[1].imshow(pred_lst, cmap="inferno")
            axes[1].set_title("Predicted LST 2024")
            fig.suptitle("Figure 16. Actual vs predicted LST")
            fig.tight_layout()
            fig.savefig(PATHS["figures"] / "Figure_16_Actual_vs_Predicted_LST.png", bbox_inches="tight")
            plt.close(fig)
            pred_ndvi_change = pred_ndvi - rasters["ndvi_2004"].array
            pred_lst_change = pred_lst - rasters["lst_2004"].array
            write_raster(PATHS["processed"] / "Predicted_NDVI_Change.tif", pred_ndvi_change, rasters["ndvi_2004"].profile)
            write_raster(PATHS["processed"] / "Predicted_LST_Change.tif", pred_lst_change, rasters["lst_2004"].profile)
        else:
            extras["cnn_note"] = (
                "CNN-LSTM was not trained. "
                + str(cnn_metrics.get("reason", "Insufficient patches or runtime error."))
                + " The architecture remains a prototype for representation learning, not a validated forecast."
            )

    extras["cnn_lstm"] = cnn_metrics
    extras["change_summaries"] = change_summaries
    extras["lulc_areas"] = lulc_areas
    extras["raster_metadata"] = metadata
    save_stats(stats_rows, extras)

    # Web layers
    layers = {}
    mapping = [
        ("ndvi_2004", "ndvi"),
        ("ndvi_2024", "ndvi"),
        ("ndvi_change", "ndvi_change"),
        ("lst_2004", "lst"),
        ("lst_2024", "lst"),
        ("lst_change", "lst_change"),
        ("lulc_2004", "lulc"),
        ("lulc_2024", "lulc"),
        ("lulc_change", "lulc"),
    ]
    # raster_to_png computes the stretch from the actual raster statistics
    # and masks the render to the Rajapeta village polygon (transparent outside).
    for key, kind in mapping:
        if key in rasters:
            layers[key] = raster_to_png(rasters[key], key, kind)
        else:
            layers[key] = _unavailable_layer(key, kind)

    # Histograms and derived change statistics feed the interactive charts page.
    hist_keys = (
        "ndvi_2004",
        "ndvi_2024",
        "ndvi_change",
        "lst_2004",
        "lst_2024",
        "lst_change",
    )
    histograms = {
        k: histogram_data(rasters[k].array) for k in hist_keys if k in rasters
    }

    # Shared-bin 2004-vs-2024 distributions (comparable bars in one chart).
    paired_hists: dict = {}
    for label, ka, kb in (
        ("ndvi", "ndvi_2004", "ndvi_2024"),
        ("lst", "lst_2004", "lst_2024"),
    ):
        if ka in rasters and kb in rasters:
            year_a = ka.rsplit("_", 1)[-1]
            year_b = kb.rsplit("_", 1)[-1]
            unit = label.upper() if label == "ndvi" else "LST (°C)"
            paired_hists[label] = paired_histogram(
                rasters[ka].array,
                rasters[kb].array,
                label_a=f"{unit} {year_a}",
                label_b=f"{unit} {year_b}",
            )

    # Real CNN-LSTM actual-vs-predicted samples (down-sampled, never invented).
    cnn_scatter: dict = {}
    if pred_ndvi is not None and pred_lst is not None:
        cnn_scatter["ndvi"] = _scatter_sample(
            rasters["ndvi_2024"].array, pred_ndvi, "NDVI 2024"
        )
        cnn_scatter["lst"] = _scatter_sample(
            rasters["lst_2024"].array, pred_lst, "LST 2024 (°C)"
        )

    stats_for_web = {
        "rasters": stats_rows,
        "change_summaries": change_summaries,
        "lulc_areas": lulc_areas,
        "cnn_lstm": {k: v for k, v in cnn_metrics.items() if k != "history" and k != "traceback"},
        "lulc_note": extras["lulc_note"],
        "cnn_note": extras["cnn_note"],
        "derived": derived_change_stats(stats_rows),
        "histograms": histograms,
        "paired_histograms": paired_hists,
        "cnn_scatter": cnn_scatter,
        "lulc_available": bool(
            rasters.get("lulc_2004") is not None and rasters.get("lulc_2024") is not None
        ),
        "study_area": STUDY_AREA,
        "lulc_area_change": extras.get("lulc_area_change", {}),
        "trends": compute_trends(rasters),
    }
    write_web_manifest(layers, boundary_gj, stats_for_web)
    if boundary_gj:
        (PATHS["web_data"] / "rajapeta_boundary.geojson").write_text(
            json.dumps(boundary_gj), encoding="utf-8"
        )
    (PATHS["web_data"] / "stats.json").write_text(json.dumps(stats_for_web, indent=2), encoding="utf-8")
    import shutil

    for folder in (PATHS["charts"], PATHS["figures"]):
        for png in folder.glob("*.png"):
            shutil.copy2(png, PATHS["web_data"] / png.name)

    llm_payload = {
        "rasters": stats_rows,
        "cnn_lstm": {k: v for k, v in cnn_metrics.items() if k != "history"},
        "lulc_note": extras["lulc_note"],
        "change_summaries": change_summaries,
        "lulc_areas": lulc_areas,
        "study_area": STUDY_AREA,
    }
    llm_text = interpret(llm_payload)
    (PATHS["reports"] / "llm_raw.md").write_text(llm_text, encoding="utf-8")
    (PATHS["web_data"] / "llm_interpretation.md").write_text(llm_text, encoding="utf-8")

    stats_text = "\n".join(
        f"- {r['name']}: n={r.get('n', 0)}"
        + (
            f", mean={r['mean']:.4f}, median={r['median']:.4f}, min={r['min']:.4f}, "
            f"max={r['max']:.4f}, std={r['std']:.4f}"
            if r.get("n", 0)
            else " (no valid pixels)"
        )
        for r in stats_rows
    ) or "No GeoTIFF inputs were found. Place files in data/raw/ and re-run."

    md = __import__("src.report_generator", fromlist=["build_markdown"]).build_markdown(
        stats_text, llm_text, extras
    )
    reports = generate_reports(md)
    paper = write_paper(stats_text, llm_text, extras)

    summary = {
        "study_area": STUDY_AREA,
        "inputs": extras["discovered_files"],
        "reports": reports,
        "paper": str(paper),
        "cnn_lstm": {k: cnn_metrics.get(k) for k in ("trained", "mae", "rmse", "mse", "reason")},
        "web": str(PATHS["web"] / "ndvi.html"),
    }
    (PATHS["outputs"] / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
