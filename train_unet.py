"""
Train the nucleus-segmentation U-Net on BBBC039 and save unet_bbbc039.pth.

    python train_unet.py --data-dir data/bbbc039

Uses the official split: training images for training, validation images
for model selection (instance F1 every few epochs, best checkpoint kept).
The test split is never touched here; benchmark_bbbc039.py --unet
unet_bbbc039.pth evaluates it once. Deterministic for a given --seed on the
same hardware.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.bbbc039 import load_split
from src.medical_cnn import get_device, set_deterministic
from src.segmentation import evaluate
from src.unet import UNet, labels_to_classes, segment_unet


def random_batches(images, classes, batch_size, crop, rng):
    """Random crops with flips, 90° rotations and brightness/contrast jitter."""
    xs, ys = [], []
    for _ in range(batch_size):
        i = rng.integers(len(images))
        img, cls = images[i], classes[i]
        top = rng.integers(img.shape[0] - crop + 1)
        left = rng.integers(img.shape[1] - crop + 1)
        x = img[top:top + crop, left:left + crop].astype(np.float32) / 255.0
        y = cls[top:top + crop, left:left + crop]
        k = rng.integers(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = np.clip(x * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
        xs.append(x.copy()); ys.append(y.copy())
    return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))


def weighted_cross_entropy(logits, target, class_weights):
    """Same value as nn.CrossEntropyLoss(weight=...) but built from
    deterministic ops: PyTorch's weighted 2-D NLL kernel has no deterministic
    CUDA implementation, which set_deterministic() forbids."""
    log_probs = torch.log_softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(log_probs.dtype)
    nll = -(log_probs * onehot).sum(dim=1)
    weights = class_weights[target]
    return (weights * nll).sum() / weights.sum()


def validation_f1(model, items) -> float:
    model.eval()
    scores = [evaluate(labels, {"U-Net": segment_unet(model, img)})[0]["instance_f1"]
              for _, img, labels in items]
    model.train()
    return float(np.mean(scores))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the BBBC039 nucleus-segmentation U-Net.")
    parser.add_argument("--data-dir", default="data/bbbc039")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--base", type=int, default=16, help="channels at full resolution")
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="unet_bbbc039.pth")
    args = parser.parse_args(argv)
    for name in ("epochs", "steps_per_epoch", "batch_size", "eval_every", "base"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.crop % 16 or not 16 <= args.crop <= 512:      # U-Net halves 4 times; images are 520x696
        parser.error("--crop must be a multiple of 16 between 16 and 512")
    if args.boundary_weight <= 0:
        parser.error("--boundary-weight must be positive")
    if not os.path.isdir(os.path.join(args.data_dir, "metadata")):
        parser.error(f"{args.data_dir} does not contain BBBC039 (images/, masks/, metadata/)")
    if not Path(args.out).resolve().parent.is_dir():
        parser.error(f"output folder for --out {args.out} does not exist")

    set_deterministic(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    train = load_split(args.data_dir, "training")
    val = load_split(args.data_dir, "validation")
    images = [img for _, img, _ in train]
    classes = [labels_to_classes(labels) for _, _, labels in train]

    model = UNet(base=args.base).to(device)
    print(f"U-Net with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters on {device}")
    class_weights = torch.tensor([1.0, 1.0, args.boundary_weight], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            x, y = random_batches(images, classes, args.batch_size, args.crop, rng)
            loss = weighted_cross_entropy(model(x.to(device)), y.to(device), class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        line = f"Epoch {epoch}/{args.epochs} loss {total / args.steps_per_epoch:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            f1 = validation_f1(model, val)
            line += f" | validation instance F1 {f1:.4f}"
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(line)

    params = {k: getattr(args, k) for k in ("epochs", "steps_per_epoch", "batch_size", "crop",
                                            "base", "boundary_weight", "seed")}
    torch.save({"base": args.base, "state_dict": best_state, "params": params}, args.out)
    stem = Path(args.out).with_suffix("").as_posix()
    with open(stem + "_results.json", "w") as f:
        json.dump({"params": params, "best_epoch": best_epoch, "val_instance_f1": best_f1}, f, indent=2)
    print(f"Best validation instance F1 {best_f1:.4f} at epoch {best_epoch}; saved {args.out}")


if __name__ == "__main__":
    main()
