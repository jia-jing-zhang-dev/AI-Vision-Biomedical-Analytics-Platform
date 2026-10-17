"""
blood_cells.py
==============
Finds the stained (white) blood cells in a blood-smear photo and crops each
one the way BloodMNIST images are framed: one cell, centred.

The BloodMNIST models only ever saw single-cell crops, so a whole smear with
several cells gives a meaningless answer. Leukocyte nuclei and platelets are
stained purple (Wright-Giemsa / May-Grünwald-Giemsa), red blood cells and the
background are not: in RGB, blue minus green separates them.

Calibrated on the BloodMNIST validation split: the stained region of a cell
spans a median 0.60 of the crop's side (lymphocytes 0.50, monocytes 0.75).
Cell size is a clue for the classifier, so one crop size is used for every
cell of an image (from the median stained size) instead of resizing each
cell to fill its crop.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

STAIN_MIN = 25          # blue - green; background and red cells are ~ -15, nuclei 70-95
STAINED_TO_CROP = 0.60  # median stained-region size / crop side (validation split)
KEEP_FRACTION = 0.15    # ignore stained objects under 15% of the largest one's area


def stain_map(rgb: np.ndarray) -> np.ndarray:
    """Blue minus green: high for purple-stained nuclei and platelets."""
    return rgb[..., 2].astype(np.float32) - rgb[..., 1].astype(np.float32)


def find_cells(rgb: np.ndarray) -> List[Tuple[int, int, int]]:
    """Returns (centre x, centre y, size in px) of each stained cell, largest first.
    Lobes of one nucleus (neutrophils) are merged with a closing scaled to the
    typical object size."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 image")
    f = stain_map(rgb)
    t, _ = cv2.threshold(np.clip(f, 0, 255).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (f > max(t, STAIN_MIN)).astype(np.uint8)
    if not mask.any():
        return []
    # First pass: typical object size, then close gaps of about a quarter of it
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    typical = np.sqrt(np.median(areas[areas >= KEEP_FRACTION * areas.max()]))
    k = max(3, int(round(0.25 * typical)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    cells = []
    for i in np.argsort(-areas):
        if areas[i] < max(20, KEEP_FRACTION * areas.max()):
            break
        w, h = stats[i + 1, cv2.CC_STAT_WIDTH], stats[i + 1, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i + 1]
        cells.append((int(round(cx)), int(round(cy)), int(max(w, h))))
    return cells


def crop_side(cells) -> int:
    """One crop size for the whole image, from the median stained size."""
    return int(round(np.median([s for _, _, s in cells]) / STAINED_TO_CROP))


def crop_cells(image: Image.Image, cells, side: int | None = None) -> List[Image.Image]:
    """Square crops centred on each cell; the image is mirrored at its edges
    so cells near the border still get a full crop."""
    rgb = np.asarray(image.convert("RGB"))
    side = side or crop_side(cells)
    half = side // 2
    padded = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_REFLECT_101)
    return [Image.fromarray(padded[cy:cy + side, cx:cx + side]) for cx, cy, _ in cells]


# DinoBloom-S confidence below which a cell is left for a person to review.
# Chosen on the KU-Optofil validation patients (another lab and scanner) as
# the lowest threshold whose accepted cells are >= 98% correct (98.1%). Checked
# once on the KU-Optofil test patients: 98.1% of cells accepted, 96.8% of them
# correct (below the target), only 55 of 209 errors flagged; on smear mosaics
# of BloodMNIST test cells: 99.6% accepted, 99.1% correct.
REVIEW_CONFIDENCE = 0.50


def accepted_label(probs) -> int | None:
    """Predicted class if the model is at least REVIEW_CONFIDENCE sure,
    else None (the cell needs review by a person)."""
    return int(np.argmax(probs)) if float(np.max(probs)) >= REVIEW_CONFIDENCE else None


def differential_count(labels, class_names) -> List[Tuple[str, int, float]]:
    """(class, count, share of the accepted cells) for every class, like a
    lab report; cells needing review (None) are left out of the shares."""
    accepted = [lab for lab in labels if lab is not None]
    return [(name, accepted.count(i), accepted.count(i) / len(accepted) if accepted else 0.0)
            for i, name in enumerate(class_names)]


# Single-cell uploads from other scanners can be zoomed out: on KU-Optofil PBC
# (Sysmex DI-60) the stained cell spans a median 0.36 of the frame, against
# 0.66 on BloodMNIST at 224 px. Such images are re-framed to BloodMNIST's
# scale; images already framed like BloodMNIST are left alone (re-framing
# them cost the ResNet up to 12 points). Both values were chosen on the
# validation splits of BloodMNIST-224 and KU-Optofil.
SINGLE_CELL_RATIO = 0.661       # stained size / frame, BloodMNIST-224 validation median
REFRAME_BELOW = 0.45            # re-frame when the stained cell is smaller than this share


def reframe_single_cell(image: Image.Image) -> Image.Image:
    """Crop a zoomed-out single-cell image around its central stained cell at
    BloodMNIST's framing; any other image is returned unchanged."""
    rgb = np.asarray(image.convert("RGB"))
    cells = find_cells(rgb)
    if not cells:
        return image
    h, w = rgb.shape[:2]
    cx, cy, size = min(cells, key=lambda c: (c[0] - w / 2) ** 2 + (c[1] - h / 2) ** 2)
    if size / min(h, w) >= REFRAME_BELOW:
        return image
    return crop_cells(image, [(cx, cy, size)], int(round(size / SINGLE_CELL_RATIO)))[0]


def is_smear(cells) -> bool:
    """True when the upload shows several stained cells. A single stained
    object is classified as a whole image: a lone platelet or erythroblast
    in a single-cell crop is small, so its size says nothing about the
    field of view (on validation crops a size rule flagged 18% of them)."""
    return len(cells) >= 2
