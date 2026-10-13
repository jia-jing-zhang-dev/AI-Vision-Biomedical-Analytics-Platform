"""
segmentation.py
================
Classical computer-vision image segmentation module. Implements and
benchmarks unsupervised segmentation strategies against ground truth:

    1. Global (Otsu) thresholding
    2. Adaptive (local mean) thresholding
    3. K-Means intensity clustering (K=2)
    4. Canny edge detection + hole filling
    5. Top-hat background correction + Otsu
    6. Marker-based Watershed (distance-transform peaks as seeds), which
       also splits touching cells into separate instances

Two synthetic benchmarks are provided so the whole pipeline runs
end-to-end with no external data files:

    make_synthetic_image - three large shapes on a mild gradient (easy)
    make_cell_image      - dozens of touching, textured nuclei under uneven
                           illumination, blur and noise, with per-cell
                           instance labels (realistic)

Methods are scored with pixel-level Dice / IoU and with instance-level F1
(a predicted object counts as found when it overlaps a true cell at
IoU > 0.5), which rewards separating touching cells rather than just
finding foreground.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def dice(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Dice / F1 overlap coefficient between two binary masks."""
    gt, pred = gt.astype(bool), pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    return (2.0 * inter) / (gt.sum() + pred.sum() + eps)


def iou(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """Intersection-over-Union between two binary masks."""
    gt, pred = gt.astype(bool), pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / (union + eps)


def _to_uint8(mask_bool: np.ndarray) -> np.ndarray:
    return (mask_bool.astype(np.uint8) * 255)


def label_objects(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Turns a segmentation output into an instance label image.

    Label images (int32, as returned by the watershed) are kept as they are;
    binary masks are split into connected components. Objects smaller than
    ``min_size`` pixels are treated as noise and dropped.
    """
    if mask.dtype == np.int32:
        labels = mask.copy()
    else:
        _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel())
    small = sizes < min_size
    small[0] = False
    labels[small[labels]] = 0
    return labels


def instance_f1(gt_labels: np.ndarray, pred_labels: np.ndarray,
                iou_thresh: float = 0.5) -> Tuple[float, int, int]:
    """Object-level F1. A true cell and a predicted object match when their
    IoU exceeds ``iou_thresh`` (> 0.5 makes matches one-to-one).
    Returns (f1, n_true, n_pred).
    """
    n_gt = len(np.unique(gt_labels[gt_labels > 0]))
    n_pred = len(np.unique(pred_labels[pred_labels > 0]))
    if n_gt == 0 and n_pred == 0:
        return 1.0, 0, 0

    # Pixel counts of every (true id, predicted id) overlap, in one pass.
    n_p = int(pred_labels.max()) + 1
    pairs = np.bincount(gt_labels.ravel().astype(np.int64) * n_p + pred_labels.ravel(),
                        minlength=(int(gt_labels.max()) + 1) * n_p).reshape(-1, n_p)
    gt_area, pred_area = pairs.sum(1), pairs.sum(0)
    inter = pairs[1:, 1:]
    union = gt_area[1:, None] + pred_area[None, 1:] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.where(union > 0, inter / union, 0.0)
    tp = int((iou_mat > iou_thresh).sum())
    return 2 * tp / (n_gt + n_pred), n_gt, n_pred


# ----------------------------------------------------------------------
# Synthetic benchmark image
# ----------------------------------------------------------------------
def make_synthetic_image(h: int = 256, w: int = 256, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic grayscale image (bright shapes on a dark,
    gradient + noise background) with its ground-truth binary mask, to
    give the segmentation methods something non-trivial to disagree on.
    """
    rng = np.random.default_rng(seed)
    gt_mask = np.zeros((h, w), np.uint8)
    cv2.circle(gt_mask, (80, 90), 40, 1, -1)
    cv2.rectangle(gt_mask, (150, 60), (220, 130), 1, -1)
    cv2.circle(gt_mask, (170, 170), 30, 1, -1)

    img = (gt_mask * 200).astype(np.uint8)
    grad = np.tile(np.linspace(0, 40, w, dtype=np.float32), (h, 1))
    noisy = img.astype(np.float32) + grad + rng.normal(0, 5, size=img.shape).astype(np.float32)
    img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img, gt_mask


def make_cell_image(h: int = 256, w: int = 256, n_cells: int = 45,
                    seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fluorescence-microscopy-like image of nuclei and its
    instance label image (0 = background, 1..N = cells).

    Realistic difficulties: cells of different size and brightness that
    touch or overlap, internal texture, a dim corner from uneven
    illumination, optical blur, and sensor noise.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros((h, w), np.int32)
    signal = np.zeros((h, w), np.float32)

    for cell_id in range(1, n_cells + 1):
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        a = int(rng.integers(7, 14))
        b = int(a * rng.uniform(0.65, 1.0))
        cell = np.zeros((h, w), np.uint8)
        cv2.ellipse(cell, (cx, cy), (a, b), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        cell = cell.astype(bool)
        labels[cell] = cell_id          # later cells sit on top of earlier ones
        signal[cell] = rng.uniform(0.45, 1.0)

    # Re-number so ids stay consecutive after occlusion.
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(h, w).astype(np.int32)

    texture = gaussian_filter(rng.normal(0, 1, (h, w)), 2).astype(np.float32)
    signal = signal * (1 + 0.25 * texture / (np.abs(texture).max() + 1e-8))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    illumination = 1.0 - 0.8 * ((xx / w) ** 2 + (yy / h) ** 2) / 2    # dim bottom-right corner
    img = (signal + 0.08) * illumination
    img = gaussian_filter(img, 1.2)
    img = img + rng.normal(0, 0.035, (h, w)).astype(np.float32)
    img = np.clip(img / img.max(), 0, 1)
    return (img * 255).astype(np.uint8), labels


# ----------------------------------------------------------------------
# Segmentation methods
# ----------------------------------------------------------------------
def segment_otsu(img: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_adaptive(img: np.ndarray, block_size: int = 31, c: int = 5) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )


def segment_kmeans(img: np.ndarray, k: int = 2) -> np.ndarray:
    Z = img.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(img.shape)
    fg_val = np.max(centers)  # brighter cluster = foreground
    return (clustered == fg_val).astype(np.uint8) * 255


def segment_canny_fill(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    edges = cv2.Canny(img, low, high)
    edges_dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    filled = binary_fill_holes(edges_dil.astype(bool))
    return _to_uint8(filled)


def segment_tophat_otsu(img: np.ndarray, radius: int | None = None) -> np.ndarray:
    """Removes uneven illumination with a white top-hat (image minus its
    morphological opening by a disk larger than any object), then applies Otsu.

    By default the disk radius is 1.5x the equivalent radius of the largest
    object in a plain Otsu mask, so large structures are not erased.
    """
    if radius is None:
        _, _, stats, _ = cv2.connectedComponentsWithStats((segment_otsu(img) > 0).astype(np.uint8))
        largest = stats[1:, cv2.CC_STAT_AREA].max() if len(stats) > 1 else 0
        radius = int(np.clip(1.5 * np.sqrt(largest / np.pi), 15, 100))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    corrected = cv2.morphologyEx(cv2.GaussianBlur(img, (3, 3), 0), cv2.MORPH_TOPHAT, disk)
    _, mask = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def segment_watershed(img: np.ndarray, min_distance: int = 6) -> np.ndarray:
    """Splits touching cells. The foreground comes from top-hat + Otsu; each
    peak of the distance transform seeds one cell, and the watershed grows
    the seeds until they meet. Returns an int32 instance label image.
    """
    fg = binary_fill_holes(segment_tophat_otsu(img) > 0)
    dist = distance_transform_edt(fg)
    _, components = cv2.connectedComponents(fg.astype(np.uint8))
    peaks = peak_local_max(gaussian_filter(dist, 1), min_distance=min_distance,
                           labels=components, exclude_border=False)
    markers = np.zeros(img.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=fg).astype(np.int32)


SEGMENTERS = {
    "Otsu": segment_otsu,
    "Adaptive": segment_adaptive,
    "KMeans-2": segment_kmeans,
    "Canny+Fill": segment_canny_fill,
    "TopHat+Otsu": segment_tophat_otsu,
    "Watershed": segment_watershed,
}

CELLPOSE_NAME = "Cellpose"
_cellpose_model = None


def cellpose_available() -> bool:
    """Cellpose is an optional dependency (``pip install "cellpose<4"``)."""
    import importlib.util
    return importlib.util.find_spec("cellpose") is not None


def segment_cellpose(img: np.ndarray) -> np.ndarray:
    """Deep-learning reference method: Cellpose 3's pretrained ``nuclei``
    model (Stringer et al., Nature Methods 2021). The cell diameter is
    estimated automatically, so nothing is tuned to these images. Runs on
    CPU so results match between machines. Returns an int32 label image.
    """
    global _cellpose_model
    if _cellpose_model is None:
        from cellpose import models
        _cellpose_model = models.Cellpose(gpu=False, model_type="nuclei")
    masks = _cellpose_model.eval(img, diameter=None, channels=[0, 0])[0]
    return masks.astype(np.int32)


# ----------------------------------------------------------------------
# Benchmark harness
# ----------------------------------------------------------------------
def has_bright_background(img: np.ndarray) -> bool:
    """True for brightfield-style images (dark cells on a light background,
    e.g. stained blood smears). The image border is mostly background, so the
    background is bright when most border pixels lie above the Otsu threshold.
    """
    thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return bool((border > thresh).mean() > 0.5)


def run_all(img: np.ndarray, auto_polarity: bool = True,
            include_cellpose: bool = False) -> Dict[str, np.ndarray]:
    """Run every registered segmenter on ``img`` and return a dict of masks.

    All methods assume bright objects on a dark background (fluorescence).
    With ``auto_polarity``, brightfield images are inverted first. With
    ``include_cellpose`` (and Cellpose installed), Cellpose is run as well.
    """
    if auto_polarity and has_bright_background(img):
        img = 255 - img
    results = {name: fn(img) for name, fn in SEGMENTERS.items()}
    if include_cellpose and cellpose_available():
        results[CELLPOSE_NAME] = segment_cellpose(img)
    return results


def evaluate(gt_mask: np.ndarray, results: Dict[str, np.ndarray]) -> "list[dict]":
    """Score every method's mask against the ground truth. Returns a list
    of dicts (ready for ``pd.DataFrame``), sorted by Dice descending.

    ``gt_mask`` may be a binary mask or an instance label image; with
    instance labels, instance F1 and the object counts are added.
    """
    gt_bin = (gt_mask > 0)
    gt_labels = gt_mask.astype(np.int32) if gt_mask.max() > 1 else None
    rows = []
    for name, mask in results.items():
        pred_bin = (mask > 0)
        row = {"method": name, "dice": dice(gt_bin, pred_bin), "iou": iou(gt_bin, pred_bin)}
        if gt_labels is not None:
            f1, n_true, n_pred = instance_f1(gt_labels, label_objects(mask))
            row.update({"instance_f1": f1, "cells_true": n_true, "cells_found": n_pred})
        rows.append(row)
    return sorted(rows, key=lambda r: r["dice"], reverse=True)


def benchmark(n_images: int = 10, start_seed: int = 0, include_cellpose: bool = False,
              **image_kwargs) -> "list[dict]":
    """Average every method's scores over ``n_images`` realistic cell images
    (seeds start_seed..start_seed+n_images-1). Returns rows sorted by
    instance F1, each with the mean and standard deviation of every metric.
    """
    per_method: Dict[str, list] = {}
    for seed in range(start_seed, start_seed + n_images):
        img, labels = make_cell_image(seed=seed, **image_kwargs)
        for row in evaluate(labels, run_all(img, include_cellpose=include_cellpose)):
            per_method.setdefault(row["method"], []).append(row)

    summary = []
    for name, rows in per_method.items():
        entry = {"method": name}
        for key in ("dice", "iou", "instance_f1"):
            vals = np.array([r[key] for r in rows])
            entry[key], entry[f"{key}_std"] = float(vals.mean()), float(vals.std())
        summary.append(entry)
    return sorted(summary, key=lambda r: r["instance_f1"], reverse=True)


def plot_results(img: np.ndarray, gt_mask: np.ndarray, results: Dict[str, np.ndarray]):
    """Grid-plot the original image, ground truth, and every method's mask."""
    n = len(results) + 2
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    for i, (name, mask) in enumerate(results.items(), start=2):
        axes[i].imshow(mask, cmap="gray")
        axes[i].set_title(name)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig
