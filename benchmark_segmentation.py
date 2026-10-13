"""
Benchmark every segmentation method on synthetic cell images.

    python benchmark_segmentation.py              # 10 images
    python benchmark_segmentation.py --images 50

Prints mean ± std of Dice, IoU and instance F1 per method, and saves an
example figure (input, ground truth, Otsu, watershed) to public/.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.segmentation import benchmark, label_objects, make_cell_image, segment_otsu, segment_watershed


def color_labels(labels: np.ndarray) -> np.ndarray:
    """Random colour per instance, black background."""
    palette = np.random.default_rng(1).uniform(0.25, 1.0, (labels.max() + 1, 3))
    palette[0] = 0
    return palette[labels]


def save_example(path: str, seed: int = 0):
    img, gt = make_cell_image(seed=seed)
    panels = [
        ("Input (uneven illumination, noise)", img, "gray"),
        (f"Ground truth ({gt.max()} cells)", color_labels(gt), None),
        ("Otsu + connected components", color_labels(label_objects(segment_otsu(img))), None),
        ("Top-hat + distance-transform watershed", color_labels(label_objects(segment_watershed(img))), None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
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
    args = parser.parse_args()

    print(f"{'Method':<14}{'Dice':>16}{'IoU':>16}{'Instance F1':>18}")
    for r in benchmark(args.images):
        print(f"{r['method']:<14}"
              f"{r['dice']:>10.3f} ± {r['dice_std']:.3f}"
              f"{r['iou']:>10.3f} ± {r['iou_std']:.3f}"
              f"{r['instance_f1']:>12.3f} ± {r['instance_f1_std']:.3f}")
    save_example("public/cell-segmentation-benchmark.png")


if __name__ == "__main__":
    main()
