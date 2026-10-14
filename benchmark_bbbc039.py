"""
Benchmark every segmentation method on real microscopy images: the 50
test images of BBBC039 (U2OS nuclei, expert annotations, official split).

    python benchmark_bbbc039.py --data-dir data/bbbc039
    python benchmark_bbbc039.py --split validation --no-cellpose

The classical methods run with the parameters developed on the synthetic
benchmark, without tuning on BBBC039. Prints mean ± std per method, writes
the numbers to bbbc039_results.json and an example figure to public/.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.bbbc039 import SPLITS, load_split
from src.segmentation import (
    cellpose_available, color_labels, evaluate, label_objects, run_all, segment_otsu,
    segment_watershed,
)

FIGURE = "public/bbbc039-segmentation.png"
RESULTS = "bbbc039_results.json"


def summarize(per_method: dict) -> list:
    rows = []
    for name, scores in per_method.items():
        entry = {"method": name, "images": len(scores)}
        for key in ("dice", "iou", "instance_f1", "average_f1"):
            vals = np.array([s[key] for s in scores])
            entry[key], entry[key + "_std"] = float(vals.mean()), float(vals.std())
        rows.append(entry)
    return sorted(rows, key=lambda r: r["instance_f1"], reverse=True)


def save_figure(path: str, name: str, img: np.ndarray, labels: np.ndarray, results: dict):
    import matplotlib.pyplot as plt
    panels = [("BBBC039 test image", img, "gray"),
              (f"Expert annotation ({len(np.unique(label_objects(labels))) - 1} nuclei)",
               color_labels(label_objects(labels)), None)]
    for method in ("Otsu", "Watershed", "Cellpose", "U-Net"):
        if method in results:
            panels.append((method, color_labels(label_objects(results[method])), None))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (title, im, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved example figure to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segmentation benchmark on BBBC039 (real nuclei images).")
    parser.add_argument("--data-dir", default="data/bbbc039", help="folder with images/, masks/, metadata/")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--no-cellpose", action="store_true", help="skip Cellpose even if it is installed")
    parser.add_argument("--unet", default=None, help="U-Net weights from train_unet.py to include")
    parser.add_argument("--no-save", action="store_true", help="print only; do not write results or figure")
    args = parser.parse_args(argv)
    if not os.path.isdir(os.path.join(args.data_dir, "metadata")):
        parser.error(f"{args.data_dir} does not contain BBBC039 (images/, masks/, metadata/)")
    if args.unet and not Path(args.unet).exists():
        parser.error(f"--unet weights {args.unet} not found")
    if not args.no_save and not Path(FIGURE).resolve().parent.is_dir():
        parser.error("run from the repository root (public/ not found)")

    use_cellpose = cellpose_available() and not args.no_cellpose
    unet = None
    if args.unet:
        from src.unet import load_unet
        unet = load_unet(args.unet)

    items = load_split(args.data_dir, args.split)
    per_method, example = {}, None
    for i, (name, img, labels) in enumerate(items):
        results = run_all(img, auto_polarity=False, include_cellpose=use_cellpose)
        if unet is not None:
            from src.unet import segment_unet
            results["U-Net"] = segment_unet(unet, img)
        for row in evaluate(labels, results):
            per_method.setdefault(row["method"], []).append(row)
        if example is None and labels.max() > 50:
            example = (name, img, labels, results)
        print(f"  {i + 1}/{len(items)} {name[:30]}", end="\r")
    print()

    summary = summarize(per_method)
    print(f"BBBC039 {args.split} split, {len(items)} images (mean ± std over images)")
    print(f"{'Method':<14}{'Dice':>16}{'Instance F1@0.5':>20}{'Average F1 .50-.90':>22}")
    for r in summary:
        print(f"{r['method']:<14}{r['dice']:>10.3f} ± {r['dice_std']:.3f}"
              f"{r['instance_f1']:>14.3f} ± {r['instance_f1_std']:.3f}"
              f"{r['average_f1']:>16.3f} ± {r['average_f1_std']:.3f}")
    if not args.no_save:
        with open(RESULTS, "w") as f:
            json.dump({"split": args.split, "images": len(items), "results": summary}, f, indent=2)
        save_figure(FIGURE, *example)


if __name__ == "__main__":
    main()
