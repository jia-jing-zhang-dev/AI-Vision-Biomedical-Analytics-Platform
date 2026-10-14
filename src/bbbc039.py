"""
bbbc039.py
==========
Loader for BBBC039 (Broad Bioimage Benchmark Collection): 200 fluorescence
images of U2OS cell nuclei (Hoechst stain, 520x696, 12-bit) with
expert-annotated instance masks and an official training / validation /
test split (100 / 50 / 50). License: CC0.

Download images.zip, masks.zip and metadata.zip from
https://bbbc.broadinstitute.org/BBBC039 and unzip them into one folder
(default ./data/bbbc039) so it contains images/, masks/ and metadata/.

Masks are a 4-colouring: the red channel holds 0 (background) or 1-3, and
touching nuclei always get different colours, so every connected region of
one colour is one nucleus.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

SPLITS = ("training", "validation", "test")


def _path(root: str, *parts: str) -> str:
    return os.path.join(root, *parts)


def split_names(root: str, split: str) -> List[str]:
    """Image base names (without extension) of an official split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = _path(root, "metadata", f"{split}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; download BBBC039 (see the module docstring)")
    with open(path) as f:
        return [os.path.splitext(line.strip())[0] for line in f if line.strip()]


def decode_mask(mask_rgba: np.ndarray) -> np.ndarray:
    """Turns a BBBC039 colour-coded mask into an int32 instance label image."""
    colours = mask_rgba[..., 0] if mask_rgba.ndim == 3 else mask_rgba
    labels = np.zeros(colours.shape, np.int32)
    next_id = 1
    for colour in np.unique(colours):
        if colour == 0:
            continue
        n, comp = cv2.connectedComponents((colours == colour).astype(np.uint8), connectivity=4)
        sel = comp > 0
        labels[sel] = comp[sel] + (next_id - 1)
        next_id += n - 1
    return labels


def normalize_to_uint8(img: np.ndarray, low: float = 0.5, high: float = 99.8) -> np.ndarray:
    """Percentile normalization to uint8, the usual choice for microscopy:
    a few saturated pixels would otherwise compress everything else.
    """
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    scaled = (img.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).round().astype(np.uint8)


def load_split(root: str, split: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Returns (name, uint8 image, int32 labels) for every image of a split."""
    items = []
    for name in split_names(root, split):
        img = np.asarray(Image.open(_path(root, "images", name + ".tif")))
        mask = np.asarray(Image.open(_path(root, "masks", name + ".png")))
        items.append((name, normalize_to_uint8(img), decode_mask(mask)))
    return items
