"""
Benchmark every segmentation method on synthetic cell images.

    python benchmark_segmentation.py                       # 10 images
    python benchmark_segmentation.py --images 50
    python benchmark_segmentation.py --start-seed 100      # images never used for tuning
    python benchmark_segmentation.py --no-cellpose         # classical methods only

Prints mean ± std of Dice, IoU and instance F1 per method, and saves an
example figure (input, ground truth, Otsu, watershed, Cellpose) to --figure
(default public/cell-segmentation-benchmark.png).
Cellpose is included when it is installed (pip install "cellpose<4").
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.segmentation import (
    benchmark, cellpose_available, color_labels, label_objects, make_cell_image, segment_cellpose,
    segment_otsu, segment_watershed,
)


def save_example(path: str, seed: int = 0, include_cellpose: bool = True):
    img, gt = make_cell_image(seed=seed)
    panels = [
        ("Input (uneven illumination, noise)", img, "gray"),
        (f"Ground truth ({len(np.unique(label_objects(gt))) - 1} cells)", color_labels(gt), None),
        ("Otsu + connected components", color_labels(label_objects(segment_otsu(img))), None),
        ("Top-hat + watershed", color_labels(label_objects(segment_watershed(img))), None),
    ]
    if include_cellpose:
        panels.append(("Cellpose (pretrained nuclei model)",
                       color_labels(label_objects(segment_cellpose(img))), None))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    for ax, (title, im, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved example figure to {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark every segmentation method on synthetic cell images.")
    parser.add_argument("--images", type=int, default=10, help="number of images to average over")
    parser.add_argument("--start-seed", type=int, default=0, help="seed of the first image (100+ = never used for tuning)")
    parser.add_argument("--no-cellpose", action="store_true", help="skip Cellpose even if it is installed")
    parser.add_argument("--figure", default="public/cell-segmentation-benchmark.png", help="where to save the example figure")
    args = parser.parse_args(argv)
    if args.images < 1:
        parser.error("--images must be at least 1")
    if args.start_seed < 0:
        parser.error("--start-seed must be non-negative")
    if not Path(args.figure).resolve().parent.is_dir():   # fail before the benchmark, not after
        parser.error(f"folder for --figure {args.figure} does not exist (run from the repository root?)")
    use_cellpose = cellpose_available() and not args.no_cellpose
    if not use_cellpose and not args.no_cellpose:
        print('Cellpose is not installed; run `pip install "cellpose<4"` to include it.')

    print(f"{'Method':<14}{'Dice':>16}{'IoU':>16}{'Instance F1':>18}")
    for r in benchmark(args.images, start_seed=args.start_seed, include_cellpose=use_cellpose):
        print(f"{r['method']:<14}"
              f"{r['dice']:>10.3f} ± {r['dice_std']:.3f}"
              f"{r['iou']:>10.3f} ± {r['iou_std']:.3f}"
              f"{r['instance_f1']:>12.3f} ± {r['instance_f1_std']:.3f}")
    save_example(args.figure, include_cellpose=use_cellpose)


if __name__ == "__main__":
    main()
