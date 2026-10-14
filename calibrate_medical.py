"""
Calibrate the BloodMNIST classifiers with temperature scaling.

    python calibrate_medical.py --data-dir data

For each shipped model: fits the temperature on the validation split, then
reports accuracy, NLL and ECE on the test split before and after. Writes
<weights>_calibration.json (load_model picks the temperature up, so the app
shows calibrated confidences) and public/bloodmnist-calibration.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.calibration import expected_calibration_error, fit_temperature, reliability_curve
from src.medical_cnn import get_bloodmnist_loaders, get_device, load_model, set_deterministic

MODELS = {"Custom CNN": "biomedical_cnn.pth", "ResNet-18 (pretrained)": "biomedical_resnet18.pth"}
FIGURE = "public/bloodmnist-calibration.png"


def logits_and_labels(model, loader, device):
    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            outs.append(model(x.to(device)).cpu())
            ys.append(y.squeeze(1).long())
    return torch.cat(outs), torch.cat(ys)


def metrics(logits, labels, temperature=1.0):
    probs = torch.softmax(logits / temperature, 1)
    return {"accuracy": float((probs.argmax(1) == labels).float().mean()),
            "nll": float(nn.functional.cross_entropy(logits / temperature, labels)),
            "ece": expected_calibration_error(probs, labels)}


def bootstrap_ece_change(logits, labels, temperature, n=1000, seed=0):
    """Paired bootstrap over test images of ECE(after) - ECE(before):
    (mean, 2.5th percentile, 97.5th percentile)."""
    rng = np.random.default_rng(seed)
    p0, p1 = torch.softmax(logits, 1), torch.softmax(logits / temperature, 1)
    diffs = []
    for _ in range(n):
        idx = torch.from_numpy(rng.integers(0, len(labels), len(labels)))
        diffs.append(expected_calibration_error(p1[idx], labels[idx])
                     - expected_calibration_error(p0[idx], labels[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Temperature-scale the BloodMNIST classifiers.")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args(argv)
    if not Path(FIGURE).resolve().parent.is_dir():
        parser.error("run from the repository root (public/ not found)")
    for weights in MODELS.values():
        if not Path(weights).exists():
            parser.error(f"{weights} not found; train it with train_medical.py first")

    set_deterministic(0)
    device = get_device()
    curves = {}
    for name, weights in MODELS.items():
        model = load_model(weights, device)
        model.temperature = 1.0                       # calibrate from the raw logits
        _, val_loader, test_loader = get_bloodmnist_loaders(data_dir=args.data_dir, arch=model.arch,
                                                            augment=False)
        val_logits, val_y = logits_and_labels(model, val_loader, device)
        test_logits, test_y = logits_and_labels(model, test_loader, device)
        t = fit_temperature(val_logits, val_y)
        before, after = metrics(test_logits, test_y), metrics(test_logits, test_y, t)
        change = bootstrap_ece_change(test_logits, test_y, t)
        print(f"{name}: T = {t:.3f}; ECE change {change[0]:+.4f} (95% CI {change[1]:+.4f} to {change[2]:+.4f})")
        for key in ("accuracy", "nll", "ece"):
            print(f"  test {key:<8} {before[key]:.4f} -> {after[key]:.4f}")
        out = str(Path(weights).with_suffix("")) + "_calibration.json"
        with open(out, "w") as f:
            json.dump({"temperature": t, "fitted_on": "validation", "test_before": before,
                       "test_after": after, "ece_change_bootstrap": {"mean": change[0],
                       "ci95": [change[1], change[2]], "resamples": 1000}}, f, indent=2)
        curves[name] = [(label, reliability_curve(torch.softmax(test_logits / temp, 1).numpy(), test_y.numpy()))
                        for label, temp in (("before", 1.0), (f"after (T={t:.2f})", t))]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(curves), figsize=(5.5 * len(curves), 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (name, pairs) in enumerate(curves.items()):
        ax, hist = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        for label, (centres, acc, conf, counts) in pairs:
            ok = counts > 0
            ax.plot(conf[ok], acc[ok], "o-", label=label)
        ax.set(title=f"{name}: test set", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=9)
        centres, _, _, counts = pairs[1][1]
        hist.bar(centres, counts, width=1 / len(centres), color="tab:orange", edgecolor="white")
        hist.set(yscale="log", xlabel="confidence", ylabel="images")
        hist.set_title("predictions per bin (log scale): low-confidence bins hold few images",
                       fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=110)
    plt.close(fig)
    print(f"Saved reliability diagrams to {FIGURE}")


if __name__ == "__main__":
    main()
