"""
vision_cnn.py
=============
Minimal PyTorch neural-network module for image classification
(FashionMNIST-style 28x28 grayscale input). Kept intentionally small so
it doubles as a teaching example of the train/eval loop, while still
being wired up as a reusable component of the framework.

Requires the optional `torch` / `torchvision` packages
(``pip install torch torchvision``).
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
