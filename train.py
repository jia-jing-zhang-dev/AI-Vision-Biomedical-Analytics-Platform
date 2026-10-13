"""
Train the FashionMNIST classifier (fully connected network, 5 epochs) and
save its weights to fashion_model.pth.

    python train.py
    python train.py --data-dir /path/to/data --out my_model.pth

Training is deterministic for a given --seed on the same hardware. The test
accuracy is printed after every epoch for monitoring only; the final epoch
is saved regardless (no model selection on the test set).
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from src.medical_cnn import set_deterministic
from src.vision_cnn import build_model, evaluate, get_device, get_fashion_mnist_loaders, train_one_epoch

EPOCHS = 5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="./data", help="where FashionMNIST is (or will be) downloaded")
    parser.add_argument("--out", default="fashion_model.pth", help="weights file to write")
    args = parser.parse_args(argv)
    if not Path(args.out).resolve().parent.is_dir():
        parser.error(f"output folder for --out {args.out} does not exist")
    return args


def main(argv=None):
    args = parse_args(argv)
    set_deterministic(args.seed)  # bit-identical reruns on the same hardware
    print("1. Detecting hardware and downloading the FashionMNIST dataset...")
    device = get_device()
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64, data_dir=args.data_dir)

    print(f"2. Building the neural network on {device}...")
    model, _ = build_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"3. Starting training. Total epochs: {EPOCHS}...")
    for t in range(EPOCHS):
        print(f"Epoch {t + 1} / {EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"   Train Loss: {train_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"4. Done! Model weights have been saved to: {args.out}")


if __name__ == "__main__":
    main()
