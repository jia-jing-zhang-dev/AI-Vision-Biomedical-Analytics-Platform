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


def build_cnn(dropout: float = 0.3):
    """Small convolutional network for 28x28 FashionMNIST: 5 convolutions with
    batch normalization, two 2x2 poolings (28 -> 14 -> 7), global average
    pooling and a linear head. Unlike the fully connected network, its
    filters respond to an edge or a heel wherever it appears in the image.
    Returns (model, device) like ``build_model``.
    """
    from torch import nn

    def block(c_in, c_out):
        return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]

    model = nn.Sequential(
        *block(1, 32), *block(32, 32), nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
        *block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(dropout), nn.Linear(128, 10),
    )
    device = get_device()
    return model.to(device), device


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


def get_fashion_mnist_splits(batch_size: int = 128, data_dir: str = "./data",
                             val_size: int = 5000, seed: int = 0):
    """(train, validation, test) loaders for the CNN. The validation images
    are a fixed random part of the official training set (FashionMNIST has no
    validation split), so epochs are chosen without looking at the test set.
    Training images are augmented with flips, shifts, scaling and small
    rotations, so uploads need not be framed exactly like the dataset.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.1)),
        transforms.ToTensor(),
    ])
    plain = transforms.ToTensor()
    train_aug = datasets.FashionMNIST(data_dir, train=True, download=True, transform=augment)
    train_plain = datasets.FashionMNIST(data_dir, train=True, download=True, transform=plain)
    test_ds = datasets.FashionMNIST(data_dir, train=False, download=True, transform=plain)
    order = torch.randperm(len(train_aug), generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx, train_idx = order[:val_size], order[val_size:]
    return (
        DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(train_plain, val_idx), batch_size=512),
        DataLoader(test_ds, batch_size=512),
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
    small = np.asarray(Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS),
                       dtype=np.float32)
    # Every FashionMNIST shoe faces the same way (heavier on the right half),
    # and a mirrored ankle boot is never recognized (0% on the test set). Mirror
    # left-heavy items; clothes and bags are near-symmetric, so this changes
    # little for them (test set: 86.9% -> 86.5%, all images mirrored: 63.5% -> 86.5%).
    if small[:, :14].sum() > small[:, 14:].sum():
        small = np.ascontiguousarray(small[:, ::-1])
    return torch.from_numpy(small / 255.0)[None, None]
