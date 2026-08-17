"""Smoke tests for every module in the framework. Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split

from src.clustering import KMeansExplorer, bench_k_means
from src.classification import compare_classifiers
from src.segmentation import make_synthetic_image, run_all, evaluate, dice, iou
from src.convolution import convolve2d, KERNELS


def test_kmeans_explorer_selects_best_k():
    X, _ = make_blobs(n_samples=200, centers=4, random_state=7)
    explorer = KMeansExplorer(k_range=range(2, 7)).fit(X)
    assert explorer.best_k_ in range(2, 7)
    assert 0 <= explorer.silhouette_scores_[explorer.best_k_] <= 1


def test_bench_k_means_returns_expected_keys():
    from sklearn.cluster import KMeans
    X, y = load_iris(return_X_y=True)
    result = bench_k_means(KMeans(n_clusters=3, n_init=4, random_state=0), "k=3", X, y)
    for key in ("homogeneity", "completeness", "v_measure", "adjusted_rand", "silhouette"):
        assert key in result
        assert -1.0 <= result[key] <= 1.0 + 1e-9


def test_classification_pipeline_runs():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    results = compare_classifiers(X_train, y_train, X_test, y_test)
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
    # sorted best-first
    assert results[0].accuracy >= results[-1].accuracy


def test_segmentation_metrics_are_bounded():
    img, gt_mask = make_synthetic_image()
    results = run_all(img)
    scores = evaluate(gt_mask, results)
    assert len(scores) == len(results)
    for row in scores:
        assert 0.0 <= row["dice"] <= 1.0
        assert 0.0 <= row["iou"] <= 1.0


def test_dice_and_iou_perfect_match():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    assert abs(dice(mask, mask) - 1.0) < 1e-6
    assert abs(iou(mask, mask) - 1.0) < 1e-6


def test_convolve2d_output_shape():
    image = np.random.rand(20, 20)
    out = convolve2d(image, KERNELS["box_blur"], padding=1)
    assert out.shape == (20, 20)
