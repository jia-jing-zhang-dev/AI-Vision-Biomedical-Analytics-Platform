"""
Benchmark every segmentation method on synthetic cell images.

    python benchmark_segmentation.py                       # 10 images
    python benchmark_segmentation.py --images 50
    python benchmark_segmentation.py --start-seed 100      # images never used for tuning
    python benchmark_segmentation.py --no-cellpose         # classical methods only

Prints mean ± std of Dice, IoU and instance F1 per method, and saves an
example figure (input, ground truth, Otsu, watershed, Cellpose) to public/.
Cellpose is included when it is installed (pip install "cellpose<4").
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.segmentation import (
    benchmark, cellpose_available, label_objects, make_cell_image, segment_cellpose,
    segment_otsu, segment_watershed,
)


def color_labels(labels: np.ndarray) -> np.ndarray:
    """Random colour per instance, black background."""
    palette = np.random.default_rng(1).uniform(0.25, 1.0, (labels.max() + 1, 3))
    palette[0] = 0
    return palette[labels]


def save_example(path: str, seed: int = 0, include_cellpose: bool = True):
    img, gt = make_cell_image(seed=seed)
    panels = [
        ("Input (uneven illumination, noise)", img, "gray"),
        (f"Ground truth ({gt.max()} cells)", color_labels(gt), None),
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
    print(f"Saved example figure to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--no-cellpose", action="store_true")
    args = parser.parse_args()
    use_cellpose = cellpose_available() and not args.no_cellpose
    if not use_cellpose and not args.no_cellpose:
        print('Cellpose is not installed; run `pip install "cellpose<4"` to include it.')

    print(f"{'Method':<14}{'Dice':>16}{'IoU':>16}{'Instance F1':>18}")
    for r in benchmark(args.images, start_seed=args.start_seed, include_cellpose=use_cellpose):
        print(f"{r['method']:<14}"
              f"{r['dice']:>10.3f} ± {r['dice_std']:.3f}"
              f"{r['iou']:>10.3f} ± {r['iou_std']:.3f}"
              f"{r['instance_f1']:>12.3f} ± {r['instance_f1_std']:.3f}")
    save_example("public/cell-segmentation-benchmark.png", include_cellpose=use_cellpose)


if __name__ == "__main__":
    main()
