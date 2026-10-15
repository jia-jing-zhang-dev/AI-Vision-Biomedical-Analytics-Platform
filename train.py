"""
Train a FashionMNIST classifier and save its weights.

    python train.py                     # CNN (the app's model) -> fashion_cnn.pth
    python train.py --model mlp         # original fully connected baseline -> fashion_model.pth
    python train.py --data-dir /path/to/data --out my_model.pth

The CNN trains on 55,000 augmented training images; the epoch with the best
accuracy on the other 5,000 (a fixed validation part of the training set) is
kept, and the test set is evaluated once at the end. The MLP baseline keeps
its original recipe (5 epochs, no augmentation; the test accuracy is printed
after every epoch for monitoring only and the final epoch is saved).
Training is deterministic for a given --seed on the same hardware.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from src.medical_cnn import set_deterministic
from src.vision_cnn import (
    build_cnn, build_model, evaluate, get_device, get_fashion_mnist_loaders,
    get_fashion_mnist_splits, train_one_epoch,
)

EPOCHS = 5            # MLP baseline
CNN_EPOCHS = 25
DEFAULT_OUT = {"cnn": "fashion_cnn.pth", "mlp": "fashion_model.pth"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--epochs", type=int, default=None, help=f"CNN only (default {CNN_EPOCHS})")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="./data", help="where FashionMNIST is (or will be) downloaded")
    parser.add_argument("--out", default=None, help="weights file to write (default: fashion_cnn.pth / fashion_model.pth)")
    args = parser.parse_args(argv)
    if args.model == "mlp" and args.epochs is not None:
        parser.error("--epochs only applies to --model cnn (the MLP baseline always trains 5 epochs)")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    args.epochs = args.epochs or CNN_EPOCHS
    args.out = args.out or DEFAULT_OUT[args.model]
    if not Path(args.out).resolve().parent.is_dir():
        parser.error(f"output folder for --out {args.out} does not exist")
    return args


def train_mlp(args, device):
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64, data_dir=args.data_dir)
    model, _ = build_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for t in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {t + 1}/{EPOCHS}  train loss {train_loss:.4f} | test accuracy {test_acc:.2%}")
    torch.save(model.state_dict(), args.out)


def train_cnn(args, device):
    train_loader, val_loader, test_loader = get_fashion_mnist_splits(data_dir=args.data_dir, seed=args.seed)
    torch.manual_seed(args.seed)
    model, _ = build_cnn()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc, best_state, best_epoch = -1.0, None, 0
    for t in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        _, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {t + 1}/{args.epochs}  train loss {train_loss:.4f} | validation accuracy {val_acc:.2%}")
        if val_acc > best_acc:
            best_acc, best_state, best_epoch = val_acc, copy.deepcopy(model.state_dict()), t + 1
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, loss_fn, device)      # once, at the end
    print(f"Best epoch {best_epoch}: validation {best_acc:.2%}, test {test_acc:.2%}")
    torch.save(model.state_dict(), args.out)
    results = Path(args.out).with_suffix("").as_posix() + "_results.json"
    with open(results, "w") as f:
        json.dump({"model": "cnn", "seed": args.seed, "epochs": args.epochs, "best_epoch": best_epoch,
                   "val_accuracy": best_acc, "test_accuracy": test_acc}, f, indent=2)


def main(argv=None):
    args = parse_args(argv)
    set_deterministic(args.seed)  # bit-identical reruns on the same hardware
    device = get_device()
    print(f"Training the {args.model.upper()} on {device}...")
    (train_cnn if args.model == "cnn" else train_mlp)(args, device)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    main()
