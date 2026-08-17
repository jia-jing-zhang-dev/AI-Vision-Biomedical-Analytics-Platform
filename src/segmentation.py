"""
segmentation.py
================
Classical computer-vision image segmentation module. Implements and
benchmarks four unsupervised segmentation strategies against a
ground-truth mask using Dice and IoU:

    1. Global (Otsu) thresholding
    2. Adaptive (local mean) thresholding
    3. K-Means intensity clustering (K=2)
    4. Canny edge detection + hole filling
    5. Marker-based Watershed (distance-transform seeded)

A synthetic test image (with an intensity gradient + noise) is provided
so the whole pipeline runs end-to-end with no external data files.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, distance_transform_edt
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


def segment_watershed(img: np.ndarray) -> np.ndarray:
    _, bin_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_bool = bin_.astype(bool)
    dist = distance_transform_edt(bin_bool)

    fg_mark = (dist > (0.5 * dist.max())).astype(np.int32)
    bg_mark = (~bin_bool).astype(np.int32)
    markers = np.zeros_like(fg_mark, dtype=np.int32)
    markers[bg_mark == 1] = 1
    markers[fg_mark == 1] = 2

    markers_ws = watershed(-dist, markers, mask=(bin_bool | bg_mark.astype(bool)))
    return (markers_ws == 2).astype(np.uint8) * 255


SEGMENTERS = {
    "Otsu": segment_otsu,
    "Adaptive": segment_adaptive,
    "KMeans-2": segment_kmeans,
    "Canny+Fill": segment_canny_fill,
    "Watershed(DT)": segment_watershed,
}


# ----------------------------------------------------------------------
# Benchmark harness
# ----------------------------------------------------------------------
def run_all(img: np.ndarray) -> Dict[str, np.ndarray]:
    """Run every registered segmenter on ``img`` and return a dict of masks."""
    return {name: fn(img) for name, fn in SEGMENTERS.items()}


def evaluate(gt_mask: np.ndarray, results: Dict[str, np.ndarray]) -> "list[dict]":
    """Score every method's mask against the ground truth. Returns a list
    of dicts (ready for ``pd.DataFrame``), sorted by Dice descending.
    """
    gt_bin = (gt_mask > 0)
    rows = []
    for name, mask in results.items():
        pred_bin = (mask > 0)
        rows.append({"method": name, "dice": dice(gt_bin, pred_bin), "iou": iou(gt_bin, pred_bin)})
    return sorted(rows, key=lambda r: r["dice"], reverse=True)


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
