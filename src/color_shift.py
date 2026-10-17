"""
color_shift.py
==============
Colour robustness for the blood-cell models.

- ``perturb``: a fixed, seeded suite of acquisition-like colour changes used
  only to *evaluate* robustness: a colour cast of the whole slide (e.g. the
  blue-green backgrounds of some photos), stain strength (saturation),
  exposure (gamma) and JPEG compression. It deliberately differs from the
  training-time augmentation (torchvision ColorJitter), so a model is not
  tested on exactly the changes it was trained on.
- ``reinhard``: colour normalization (Reinhard et al., IEEE CG&A 2001):
  match the per-channel mean and standard deviation in CIELAB to reference
  statistics, here those of BloodMNIST.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

SEVERITIES = {          # (max cast per channel, saturation range, gamma range, JPEG quality range)
    "mild": (0.08, (0.8, 1.2), (0.85, 1.15), (70, 95)),
    "strong": (0.18, (0.6, 1.4), (0.7, 1.4), (40, 80)),
}


def perturb(image: Image.Image, severity: str, seed: int) -> Image.Image:
    """One random acquisition-like colour change of the given severity."""
    cast, sat, gamma, quality = SEVERITIES[severity]
    rng = np.random.default_rng(seed)
    x = np.asarray(image.convert("RGB"), np.float32) / 255
    x = x * (1 + rng.uniform(-cast, cast, 3))                          # colour cast (white balance)
    grey = x.mean(2, keepdims=True)
    x = grey + (x - grey) * rng.uniform(*sat)                          # stain strength
    x = np.clip(x, 0, 1) ** rng.uniform(*gamma)                        # exposure
    buf = io.BytesIO()
    Image.fromarray((x * 255).round().astype(np.uint8)).save(buf, "JPEG", quality=int(rng.integers(*quality)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def lab_stats(images) -> tuple:
    """Mean and std of each CIELAB channel over a set of RGB images."""
    labs = [cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            for im in images]
    allpx = np.concatenate(labs)
    return allpx.mean(0), allpx.std(0)


def reinhard(image: Image.Image, target_mean, target_std) -> Image.Image:
    """Shift the image's CIELAB statistics to the target's."""
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean, std = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - mean) / std * np.asarray(target_std) + np.asarray(target_mean)
    return Image.fromarray(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
