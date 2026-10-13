"""
image_io.py
===========
Normalizes user-supplied images before any model or preview sees them.

PIL's ``convert("L")`` / ``convert("RGB")`` clip 16-bit and float images
instead of rescaling them (a 16-bit microscopy PNG becomes almost pure
white), and drop the alpha channel so transparent areas turn black.
``to_8bit`` fixes both. ``load_upload`` is the single entry point for
uploaded files: it rejects unreadable or oversized files with a readable
message, applies the camera's EXIF orientation, and downsizes large images
before any expensive conversion so the app stays within ~1 GB of memory.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}
MAX_UPLOAD_PIXELS = 25_000_000      # 5000 x 5000; keeps worst-case decoding near 200 MB


class UploadError(ValueError):
    """An uploaded file that cannot be used, with a message fit for the UI."""


def to_8bit(image: Image.Image, mode: str = "RGB") -> Image.Image:
    """Returns ``image`` as an 8-bit ``mode`` ("RGB" or "L") image.

    - 16-bit / 32-bit / float images are min-max rescaled to 0-255 using
      their finite pixels; NaN/inf pixels become 0.
    - Transparent pixels are composited onto white.
    """
    if image.width == 0 or image.height == 0:
        raise ValueError("the image is empty (0 pixels)")
    if image.mode in HIGH_BIT_DEPTH_MODES:
        arr = np.asarray(image, dtype=np.float32)
        finite = np.isfinite(arr)
        lo, hi = (float(arr[finite].min()), float(arr[finite].max())) if finite.any() else (0.0, 0.0)
        arr = np.where(finite, arr, lo)
        scaled = np.zeros_like(arr) if hi == lo else (arr - lo) * (255.0 / (hi - lo))
        image = Image.fromarray(np.round(scaled).astype(np.uint8), mode="L")
    elif image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert(mode)


def load_upload(source, mode: str = "RGB", max_side: int = 1024) -> Tuple[Image.Image, Tuple[int, int]]:
    """Opens an uploaded file (path, file object or Streamlit UploadedFile).

    Returns ``(image, original_size)``: an 8-bit ``mode`` image, upright
    according to its EXIF orientation, whose longer side is at most
    ``max_side``. Raises ``UploadError`` for unreadable, truncated or
    oversized files.
    """
    if max_side < 1:
        raise ValueError(f"max_side must be at least 1, got {max_side}")
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        import os
        if not os.path.exists(source):
            raise FileNotFoundError(source)
    try:
        # Only the formats the UI accepts: Streamlit checks just the file name,
        # so a renamed TIFF/EPS/... would otherwise reach other decoders.
        image = Image.open(source, formats=["PNG", "JPEG"])
        width, height = image.size
        if width * height > MAX_UPLOAD_PIXELS:
            raise UploadError(f"The image is {width}×{height} pixels; please upload one under "
                              f"{MAX_UPLOAD_PIXELS // 1_000_000} megapixels.")
        image.draft("RGB" if mode == "RGB" else "L", (max_side, max_side))  # JPEG: decode at reduced size
        image.load()
    except UploadError:
        raise
    except Image.DecompressionBombError:
        raise UploadError("The image is too large; please upload one under "
                          f"{MAX_UPLOAD_PIXELS // 1_000_000} megapixels.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError("Could not read this file as an image. Please upload a valid JPG or PNG.") from None

    # Size as the user sees it: before JPEG draft decoding, after EXIF rotation.
    rotated = image.getexif().get(0x0112) in (5, 6, 7, 8)
    original_size = (height, width) if rotated else (width, height)
    image = ImageOps.exif_transpose(image)
    if max(image.size) > max_side:
        if image.mode in HIGH_BIT_DEPTH_MODES:
            image = image.convert("I") if image.mode != "F" else image   # resize-capable modes
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return to_8bit(image, mode), original_size
