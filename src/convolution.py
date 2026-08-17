"""
convolution.py
===============
From-scratch 2D convolution (implemented with NumPy, no deep-learning
framework) used to demonstrate the mechanics behind CNN convolutional
layers, plus a small library of classic edge/feature-detection kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

KERNELS = {
    "edge_detect": np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]]),
    "sharpen": np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]),
    "sobel_x": np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]),
    "sobel_y": np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]]),
    "box_blur": np.ones((3, 3)) / 9.0,
}


def load_grayscale(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and convert to single-channel grayscale."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    """A from-scratch, dependency-free 2D convolution (true convolution,
    i.e. the kernel is flipped, matching the mathematical definition).

    Parameters
    ----------
    image : 2D array (H, W)
    kernel : 2D array (kh, kw)
    padding : zero-padding applied symmetrically to all sides
    stride : step size of the sliding window

    Returns
    -------
    2D array with the convolved output.
    """
    kernel = np.flipud(np.fliplr(kernel))
    kh, kw = kernel.shape
    ih, iw = image.shape

    out_h = (ih - kh + 2 * padding) // stride + 1
    out_w = (iw - kw + 2 * padding) // stride + 1
    output = np.zeros((out_h, out_w))

    if padding != 0:
        padded = np.zeros((ih + 2 * padding, iw + 2 * padding))
        padded[padding:-padding, padding:-padding] = image
    else:
        padded = image

    for y in range(out_h):
        y0 = y * stride
        for x in range(out_w):
            x0 = x * stride
            region = padded[x0:x0 + kh, y0:y0 + kw]
            output[x, y] = float((kernel * region).sum())

    return output


def apply_kernel(image: np.ndarray, kernel_name: str, padding: int = 1) -> np.ndarray:
    """Convenience wrapper: apply a named kernel from ``KERNELS``."""
    if kernel_name not in KERNELS:
        raise KeyError(f"Unknown kernel '{kernel_name}'. Options: {list(KERNELS)}")
    return convolve2d(image, KERNELS[kernel_name], padding=padding)
