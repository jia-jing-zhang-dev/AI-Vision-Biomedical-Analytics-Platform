"""
Train a classifier on BloodMNIST (8 blood-cell classes) and report test accuracy.

    python train_medical.py                  # CNN with default hyperparameters
    python train_medical.py --optuna 20      # tune the CNN on the validation set first
    python train_medical.py --model mlp      # reproduce the original fully connected baseline
    python train_medical.py --model resnet18 # fine-tune ImageNet-pretrained ResNet-18 for comparison

Hyperparameters are chosen on the validation split only; the test split is
evaluated once, at the end, so the reported accuracy is not over-fit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.medical_cnn import (
    CLASS_NAMES, BloodCNN, build_mlp_baseline, build_resnet18, evaluate, fit,
    get_bloodmnist_loaders, get_device, set_deterministic, train_one_epoch,
)

DEFAULTS = {"lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.3, "base_channels": 32}
# Fine-tuning pretrained weights needs a smaller learning rate than training from scratch.
RESNET_DEFAULTS = {"arch": "resnet18", "lr": 3e-4, "weight_decay": 1e-4}


def tune(n_trials: int, epochs: int, train_loader, val_loader, device: str) -> dict:
    """Optuna search over learning rate, weight decay, dropout and width,
    maximizing validation accuracy.
    """
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
        }
        model = BloodCNN(params["base_channels"], params["dropout"]).to(device)
        return fit(model, train_loader, val_loader, device, epochs=epochs,
                   lr=params["lr"], weight_decay=params["weight_decay"], verbose=False)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best validation accuracy {study.best_value:.4f} with {study.best_params}")
    return study.best_params


def train_mlp_baseline(device: str, data_dir: str) -> float:
    """Original setup: no augmentation, Adam(lr=1e-3), 5 epochs. Returns test accuracy."""
    train_loader, _, test_loader = get_bloodmnist_loaders(batch_size=64, data_dir=data_dir,
                                                          augment=False)
    model = build_mlp_baseline().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch [{epoch + 1}/5] train loss {loss:.4f}")
    return evaluate(model, test_loader, loss_fn, device)[1]


def save_confusion_matrix(model, test_loader, device: str, path: str, title: str):
    """Plots a row-normalized confusion matrix of the test set."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            y_pred.append(model(X.to(device)).argmax(1).cpu().numpy())
            y_true.append(y.squeeze(1).numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, normalize="true",
        values_format=".2f", cmap="Blues", xticks_rotation=45, ax=ax, colorbar=False,
    )
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_title(f"{title}: BloodMNIST test set (row-normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")


def parse_args(argv=None) -> argparse.Namespace:
    """Parses and validates the command line; fills in per-model defaults."""
    parser = argparse.ArgumentParser(description="Train a BloodMNIST classifier and report its test accuracy.")
    parser.add_argument("--model", choices=["cnn", "mlp", "resnet18"], default="cnn")
    parser.add_argument("--epochs", type=int, default=None,
                        help="training epochs (default: 15 for cnn, 10 for resnet18)")
    parser.add_argument("--optuna", type=int, default=0, metavar="N_TRIALS",
                        help="run an Optuna search with this many trials first")
    parser.add_argument("--tune-epochs", type=int, default=None,
                        help="epochs per Optuna trial (default 6; only with --optuna)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default=None,
                        help="weights file (default: biomedical_cnn.pth / biomedical_resnet18.pth)")
    args = parser.parse_args(argv)
    # Options that only apply to some models are rejected rather than silently ignored.
    if args.optuna and args.model != "cnn":
        parser.error("--optuna only applies to --model cnn")
    if args.model == "mlp" and (args.out or args.epochs is not None):
        parser.error("--model mlp reproduces the fixed original baseline (5 epochs, no saved "
                     "weights); --epochs and --out do not apply")
    if args.tune_epochs is not None and not args.optuna:
        parser.error("--tune-epochs only applies together with --optuna")
    for name in ("epochs", "tune_epochs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.optuna < 0:
        parser.error("--optuna must be 0 (off) or a positive number of trials")
    # Fail before hours of training, not when saving at the end
    if args.model != "mlp":
        for kind, path in output_paths(args.model, args.out).items():
            folder = Path(path).resolve().parent
            if not folder.is_dir():
                parser.error(f"folder for the {kind} file ({folder}) does not exist "
                             "(run from the repository root, or create it first)")
    if args.epochs is None:     # the MLP baseline always trains its original 5 epochs
        args.epochs = {"resnet18": 10, "mlp": 5}.get(args.model, 15)
    if args.tune_epochs is None:
        args.tune_epochs = 6
    return args


def output_paths(model: str, out=None) -> dict:
    """Where a run writes its weights, results JSON and confusion matrix.
    The README/app figure is only replaced when the default weights file is
    written; runs with --out keep everything next to their own weights.
    """
    weights = out or ("biomedical_resnet18.pth" if model == "resnet18" else "biomedical_cnn.pth")
    stem = Path(weights).with_suffix("").as_posix()   # same on Windows and Linux
    suffix = "-resnet18" if model == "resnet18" else ""
    figure = f"public/bloodmnist-confusion{suffix}.png" if out is None else stem + "_confusion.png"
    return {"weights": weights, "results": stem + "_results.json", "figure": figure}


def main(argv=None):
    args = parse_args(argv)
    set_deterministic(args.seed)   # bit-identical reruns on the same hardware
    device = get_device()

    if args.model == "mlp":
        print(f"MLP baseline test accuracy: {train_mlp_baseline(device, args.data_dir):.4f}")
        return

    paths = output_paths(args.model, args.out)
    train_loader, val_loader, test_loader = get_bloodmnist_loaders(
        batch_size=64 if args.model == "resnet18" else 128, data_dir=args.data_dir, arch=args.model)

    if args.model == "resnet18":
        params = dict(RESNET_DEFAULTS)
        torch.manual_seed(args.seed)
        model = build_resnet18(pretrained=True).to(device)
    else:
        params = dict(DEFAULTS)
        if args.optuna:
            params.update(tune(args.optuna, args.tune_epochs, train_loader, val_loader, device))
        torch.manual_seed(args.seed)
        model = BloodCNN(params["base_channels"], params["dropout"]).to(device)
    val_acc = fit(model, train_loader, val_loader, device, epochs=args.epochs,
                  lr=params["lr"], weight_decay=params["weight_decay"])
    _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")

    torch.save({"params": params, "state_dict": model.state_dict()}, paths["weights"])
    with open(paths["results"], "w") as f:
        json.dump({"params": params, "seed": args.seed, "epochs": args.epochs,
                   "optuna_trials": args.optuna, "val_accuracy": val_acc,
                   "test_accuracy": test_acc}, f, indent=2)
    print(f"Saved weights to {paths['weights']}")
    title = "ResNet-18 (pretrained)" if args.model == "resnet18" else "Custom CNN"
    save_confusion_matrix(model, test_loader, device, paths["figure"], title)


if __name__ == "__main__":
    main()
