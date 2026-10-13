"""
medical_cnn.py
==============
Blood-cell classifiers for BloodMNIST (28x28 RGB images, 8 classes): a
compact custom CNN, the fully connected baseline it replaced, and an
ImageNet-pretrained ResNet-18 reference; plus preprocessing, data loading,
training, evaluation and checkpoint helpers.

Requires the optional `torch`, `torchvision` and `medmnist` packages
(``pip install torch torchvision medmnist``).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

N_CLASSES = 8
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocyte",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]


def get_device() -> str:
    """Pick the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BloodCNN(nn.Module):
    """Three convolutional stages (28 -> 14 -> 7 -> global pool) followed
    by a dropout + linear classifier head.
    """

    def __init__(self, base_channels: int = 32, dropout: float = 0.3,
                 n_classes: int = N_CLASSES):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            _conv_block(3, c), _conv_block(c, c), nn.MaxPool2d(2),            # 28 -> 14
            _conv_block(c, 2 * c), _conv_block(2 * c, 2 * c), nn.MaxPool2d(2),  # 14 -> 7
            _conv_block(2 * c, 4 * c), nn.AdaptiveAvgPool2d(1),               # 7 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mlp_baseline(n_classes: int = N_CLASSES) -> nn.Module:
    """The original fully connected baseline (flattened pixels -> 128 -> 8),
    kept so the CNN improvement can be reproduced.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    )


RESNET_INPUT_SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_resnet18(pretrained: bool = True, n_classes: int = N_CLASSES) -> nn.Module:
    """torchvision's ResNet-18 with ImageNet weights and a new 8-class head.
    It is the reference architecture of the MedMNIST v2 benchmark.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def input_transform(arch: str = "cnn", from_upload: bool = False):
    """Preprocessing for one architecture. The CNN reads 28x28 images in
    [0, 1]; ResNet-18 reads 28x28 images upsampled to 224x224 and normalized
    with ImageNet statistics, as its pretrained weights expect.

    ``from_upload`` first reduces an arbitrary image to 28x28 with area
    averaging (BOX); bicubic downsampling shifts the colour/texture
    statistics enough to flip predictions.
    """
    from torchvision import transforms

    steps = []
    if from_upload:
        steps.append(transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BOX))
    if arch == "resnet18":
        steps.append(transforms.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
                                       interpolation=transforms.InterpolationMode.BILINEAR))
    steps.append(transforms.ToTensor())
    if arch == "resnet18":
        steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(steps)


def load_model(path: str = "biomedical_cnn.pth", device: str = "cpu") -> nn.Module:
    """Loads a checkpoint written by train_medical.py (weights + the
    hyperparameters needed to rebuild the architecture). The returned model
    carries its architecture name in ``model.arch``.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    params = checkpoint["params"]
    arch = params.get("arch", "cnn")
    if arch == "resnet18":
        model = build_resnet18(pretrained=False)
    else:
        model = BloodCNN(params["base_channels"], params["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.arch = arch
    return model.to(device).eval()


def predict_proba(model, image, device: str = "cpu"):
    """Returns class probabilities for one PIL image of any size, using the
    preprocessing that matches ``model.arch``.
    """
    from src.image_io import to_8bit

    tfm = input_transform(getattr(model, "arch", "cnn"), from_upload=True)
    x = tfm(to_8bit(image, "RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()


def get_bloodmnist_loaders(batch_size: int = 128, data_dir: str = "./data",
                           augment: bool = True, arch: str = "cnn"):
    """Returns (train, val, test) DataLoaders for BloodMNIST, preprocessed
    for ``arch`` ("cnn"/"mlp" or "resnet18").

    Blood-cell images have no canonical orientation, so random flips and
    90-degree rotations are used as training-time augmentation.
    """
    import medmnist
    from medmnist import INFO
    from torchvision import transforms

    DataClass = getattr(medmnist, INFO["bloodmnist"]["python_class"])
    eval_tfm = input_transform(arch)
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        eval_tfm,
    ]) if augment else eval_tfm

    def load(split, tfm):
        return DataClass(split=split, transform=tfm, download=True, root=data_dir)

    return (
        DataLoader(load("train", train_tfm), batch_size=batch_size, shuffle=True),
        DataLoader(load("val", eval_tfm), batch_size=batch_size),
        DataLoader(load("test", eval_tfm), batch_size=batch_size),
    )


def train_one_epoch(model, dataloader, loss_fn, optimizer, device: str) -> float:
    """Runs a single training epoch, returns the mean training loss."""
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.squeeze(1).long().to(device)
        loss = loss_fn(model(X), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device: str) -> Tuple[float, float]:
    """Returns (average_loss, accuracy) on the given dataloader."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.squeeze(1).long().to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / len(dataloader), correct / n


def fit(model, train_loader, val_loader, device: str, epochs: int = 15,
        lr: float = 1e-3, weight_decay: float = 1e-4, verbose: bool = True):
    """Trains with AdamW + cosine LR schedule and keeps the weights from the
    epoch with the best validation accuracy. Returns that accuracy.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] train loss {train_loss:.4f} | val acc {val_acc:.4f}")

    if best_state is None:
        raise RuntimeError("Validation accuracy never rose above 0; check epochs and data.")
    model.load_state_dict(best_state)
    return best_acc
