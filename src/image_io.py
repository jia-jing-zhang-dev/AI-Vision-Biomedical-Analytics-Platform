"""
image_io.py
===========
Normalizes user-supplied images before any model sees them.

PIL's ``convert("L")`` / ``convert("RGB")`` clip 16-bit and float images
instead of rescaling them (a 16-bit microscopy PNG becomes almost pure
white), and drop the alpha channel so transparent areas turn black.
``to_8bit`` fixes both.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def to_8bit(image: Image.Image, mode: str = "RGB") -> Image.Image:
    """Returns ``image`` as an 8-bit ``mode`` ("RGB" or "L") image.

    - 16-bit / 32-bit / float images are min-max rescaled to 0-255.
    - Transparent pixels are composited onto white.
    """
    if image.mode in HIGH_BIT_DEPTH_MODES:
        arr = np.asarray(image, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        scaled = np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo) * 255.0
        image = Image.fromarray(np.round(scaled).astype(np.uint8), mode="L")
    elif image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert(mode)
