"""
unet.py
=======
A small U-Net for nucleus instance segmentation, trained on BBBC039 by
train_unet.py.

Touching nuclei are separated with the three-class formulation used for
BBBC039 by Caicedo et al. (Cytometry A, 2019): every pixel is background,
nucleus interior, or nucleus boundary. At inference each connected
interior region seeds one nucleus, and a watershed grows the seeds over
the predicted foreground (interior + boundary).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2
DEPTH = 4                       # input sides are padded to a multiple of 2**DEPTH


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) with batch normalization: DEPTH
    down/up levels, ``base`` channels at full resolution, 3 output classes.
    """

    def __init__(self, base: int = 16, in_channels: int = 1, n_classes: int = 3):
        super().__init__()
        chans = [base * 2 ** i for i in range(DEPTH + 1)]
        self.down = nn.ModuleList([_block(in_channels, chans[0])] +
                                  [_block(chans[i], chans[i + 1]) for i in range(DEPTH)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2)
                                 for i in reversed(range(DEPTH))])
        self.dec = nn.ModuleList([_block(chans[i] * 2, chans[i]) for i in reversed(range(DEPTH))])
        self.head = nn.Conv2d(chans[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < DEPTH:
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def labels_to_classes(labels: np.ndarray) -> np.ndarray:
    """Instance labels -> per-pixel classes (background / interior / boundary),
    following Caicedo et al.: each nucleus's one-pixel contour is expanded by
    one pixel inside and one outside, giving a ~3-pixel boundary band.
    """
    from scipy.ndimage import binary_dilation, grey_dilation, grey_erosion
    fg = labels > 0
    edge = (grey_dilation(labels, size=3) != labels) | (grey_erosion(labels, size=3) != labels)
    contour = fg & edge                                   # innermost ring of every nucleus
    band = binary_dilation(contour, structure=np.ones((3, 3), bool))
    classes = np.zeros(labels.shape, np.int64)
    classes[fg] = INTERIOR
    classes[band] = BOUNDARY
    return classes


def classes_to_labels(probs: np.ndarray, min_seed: int = 10) -> np.ndarray:
    """(3, H, W) class probabilities -> int32 instance labels. Each interior
    region of at least ``min_seed`` pixels seeds one nucleus; a watershed on
    the interior probability grows it over the foreground. The boundary band
    extends one pixel outside each nucleus (labels_to_classes), so the
    predicted foreground is eroded by one pixel first; without this every
    nucleus came out one pixel too large (validation Dice 0.882 vs 0.948).
    """
    import cv2
    from scipy.ndimage import binary_erosion
    from skimage.segmentation import watershed

    pred = probs.argmax(0)
    foreground = binary_erosion(pred != BACKGROUND, structure=np.ones((3, 3), bool))
    _, seeds = cv2.connectedComponents((pred == INTERIOR).astype(np.uint8), connectivity=4)
    sizes = np.bincount(seeds.ravel())
    small = sizes < min_seed
    small[0] = False
    seeds[small[seeds]] = 0
    labels = watershed(-probs[INTERIOR], seeds, mask=foreground)
    return labels.astype(np.int32)


def load_unet(path: str, device: str = "cpu") -> UNet:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UNet(base=checkpoint["base"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def predict_probs(model: UNet, img: np.ndarray) -> np.ndarray:
    """uint8 grayscale image (any size) -> (3, H, W) class probabilities.
    The image is reflect-padded to a multiple of 2**DEPTH and cropped back."""
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 image, got {img.shape} {img.dtype}")
    h, w = img.shape
    m = 2 ** DEPTH
    ph, pw = (-h) % m, (-w) % m
    padded = np.pad(img, ((0, ph), (0, pw)), mode="reflect" if min(h, w) > 1 else "edge")
    device = next(model.parameters()).device
    x = torch.from_numpy(padded.astype(np.float32) / 255.0)[None, None].to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0, :, :h, :w]
    return probs.cpu().numpy()


def segment_unet(model: UNet, img: np.ndarray) -> np.ndarray:
    """Instance label image (int32) for a uint8 grayscale nuclei image."""
    return classes_to_labels(predict_probs(model, img))
