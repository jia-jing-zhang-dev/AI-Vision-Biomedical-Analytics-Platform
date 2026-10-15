"""
vision_cnn.py
=============
Minimal PyTorch neural-network module for image classification
(FashionMNIST-style 28x28 grayscale input). Kept intentionally small so
it doubles as a teaching example of the train/eval loop, while still
being wired up as a reusable component of the framework.

torch/torchvision come from requirements.txt (CPU wheels, pinned versions).
"""

from __future__ import annotations

from typing import Tuple


def get_device() -> str:
    """Pick the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model():
    """Build a 3-layer fully-connected classifier for 28x28 grayscale
    images (e.g. MNIST / FashionMNIST), 10-class output.
    """
    import torch
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.linear_relu_stack(x)

    device = get_device()
    return NeuralNetwork().to(device), device


def train_one_epoch(model, dataloader, loss_fn, optimizer, device: str) -> float:
    """Runs a single training epoch, returns the mean training loss."""
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device: str) -> Tuple[float, float]:
    """Returns (average_loss, accuracy) on the given dataloader."""
    import torch
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            n += y.size(0)
    return total_loss / len(dataloader), correct / n


def get_fashion_mnist_loaders(batch_size: int = 64, data_dir: str = "./data"):
    """Downloads (if needed) and returns FashionMNIST train/test DataLoaders."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def prepare_upload(image):
    """Turns any PIL image into the (1, 1, 28, 28) tensor the FashionMNIST
    model expects: 8-bit grayscale, 28x28, light item on a dark background.

    FashionMNIST items are bright on black, while most photos are dark items
    on a light background, so the image is inverted only when its border is
    bright. Transparent areas count as white background.

    The rest follows how FashionMNIST was made (Xiao et al. 2017): the
    background is removed, the item is cropped and centred, and its longer
    side is scaled to fill the 28 pixels. Squeezing a whole product photo
    to 28x28 instead leaves the item small, distorted and on a grey
    background, which the model never saw.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from src.image_io import to_8bit
    from src.segmentation import has_bright_background

    arr = np.asarray(to_8bit(image, "L"), dtype=np.uint8)
    if has_bright_background(arr):
        arr = 255 - arr
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    item = np.clip(arr.astype(np.float32) - np.median(border), 0, None)   # background -> 0
    if item.max() > 0:
        # Bounding box of the item: components at least 5% the size of the
        # largest, so specks of dust or noise do not stretch the crop
        mask = (item > 0.15 * item.max()).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = stats[1:][areas >= 0.05 * areas.max()]
        x0, y0 = keep[:, cv2.CC_STAT_LEFT].min(), keep[:, cv2.CC_STAT_TOP].min()
        x1 = (keep[:, cv2.CC_STAT_LEFT] + keep[:, cv2.CC_STAT_WIDTH]).max()
        y1 = (keep[:, cv2.CC_STAT_TOP] + keep[:, cv2.CC_STAT_HEIGHT]).max()
        item = item[y0:y1, x0:x1] * (255.0 / item.max())
    side = max(item.shape)
    square = np.zeros((side, side), np.float32)                          # centre, keep aspect
    top, left = (side - item.shape[0]) // 2, (side - item.shape[1]) // 2
    square[top:top + item.shape[0], left:left + item.shape[1]] = item
    small = Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0)[None, None]
