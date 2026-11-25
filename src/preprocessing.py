"""Remote-sensing preprocessing: CRS, alignment, nodata, patches, split."""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import PATCH, PATHS
from src.io_utils import RasterBundle, align_to_reference, raster_metadata, write_raster


def verify_and_report(bundle: RasterBundle, expected_res: float = 30.0) -> dict:
    meta = raster_metadata(bundle)
    meta["resolution_ok"] = (
        abs(abs(meta["resolution_x"]) - expected_res) < 5
        or abs(meta["resolution_x"]) < 0.001
    )
    meta["has_crs"] = bundle.crs is not None
    return meta


def common_valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def normalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    stacked = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    vmin = float(np.nanpercentile(stacked, 2))
    vmax = float(np.nanpercentile(stacked, 98))
    scale = vmax - vmin if vmax != vmin else 1.0

    def _n(x):
        out = (x - vmin) / scale
        return np.clip(out, 0.0, 1.0)

    return _n(a), _n(b), {"vmin": vmin, "vmax": vmax}


def _scan_patches(
    work: np.ndarray,
    patch_size: int,
    stride: int,
    min_valid_fraction: float = 0.6,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    t, h, w, c = work.shape
    patches, origins = [], []
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            patch = work[:, y : y + patch_size, x : x + patch_size, :]
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                continue
            if np.isfinite(patch).mean() < min_valid_fraction:
                continue
            patches.append(np.nan_to_num(patch, nan=0.0))
            origins.append((y, x))
    return patches, origins


def extract_patches(
    stack: np.ndarray,
    patch_size: int = PATCH["size"],
    stride: int = PATCH["stride"],
    min_patches: int = 4,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract 128x128 patches from a (time, height, width, channels) stack.

    Small scenes (e.g. 173x117 for Rajapeta) are smaller than a single
    128x128 patch in one dimension, so the stack is NaN-padded at the
    bottom/right edges. Padding only extends beyond the real data; padded
    rows are excluded when predictions are mosaicked back (slices are
    truncated at the original raster bounds).

    If the configured stride yields fewer than `min_patches` patches, the
    stride is reduced (down to 8 px) so overlapping sliding-window crops
    make CNN-LSTM training possible. Overlapping patches share spatial
    context — this limitation must be reported alongside any metrics.
    """
    t, h, w, c = stack.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    work = (
        np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), constant_values=np.nan)
        if pad_h or pad_w
        else stack
    )

    patches: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    current_stride = stride
    while True:
        patches, origins = _scan_patches(work, patch_size, current_stride)
        if len(patches) >= min_patches or current_stride <= 8:
            break
        current_stride = max(8, current_stride // 2)

    # Also take one edge-aligned patch at the bottom/right so mosaicked
    # predictions cover the full raster extent (no uncovered nodata strip).
    final = (work.shape[1] - patch_size, work.shape[2] - patch_size)
    if final not in origins and all(v >= 0 for v in final):
        edge = work[:, final[0] : final[0] + patch_size, final[1] : final[1] + patch_size, :]
        if np.isfinite(edge).mean() >= 0.6:
            patches.append(np.nan_to_num(edge, nan=0.0))
            origins.append(final)

    print(
        f"[preprocessing] raster {h}x{w} -> work {work.shape[1]}x{work.shape[2]} "
        f"(pad {pad_h}x{pad_w}), stride {stride} -> {current_stride}, "
        f"patches={len(patches)} (min required {min_patches})"
    )

    if not patches:
        return np.empty((0, t, patch_size, patch_size, c), dtype=np.float32), []
    return np.stack(patches).astype(np.float32), origins


def train_test_split_patches(x: np.ndarray, seed: int = 42, test_frac: float = 0.2):
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_frac)) if n > 1 else 0
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], train_idx, test_idx


def align_group(bundles: list[RasterBundle]) -> list[RasterBundle]:
    reference = bundles[0]
    aligned = [reference]
    for extra in bundles[1:]:
        if (
            extra.profile["width"] == reference.profile["width"]
            and extra.profile["height"] == reference.profile["height"]
            and extra.crs == reference.crs
        ):
            aligned.append(extra)
        else:
            aligned.append(align_to_reference(extra, reference))
    return aligned


def save_processed(name: str, bundle: RasterBundle) -> None:
    write_raster(PATHS["processed"] / name, bundle.array, bundle.profile)
