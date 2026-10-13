"""Smoke tests for every module in the framework. Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split

from src.clustering import KMeansExplorer, bench_k_means
from src.classification import compare_classifiers
from src.segmentation import (
    make_synthetic_image, make_cell_image, run_all, evaluate, benchmark, dice, iou, instance_f1,
    has_bright_background,
)
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


def test_optuna_tuning_reports_held_out_test_accuracy():
    pytest.importorskip("optuna")
    from src.classification import tune_with_optuna
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    study = tune_with_optuna(X_train, y_train, X_test, y_test, n_trials=5)
    assert 0.0 <= study.best_value <= 1.0
    assert 0.0 <= study.user_attrs["test_accuracy"] <= 1.0


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


def test_instance_f1_counts_merged_cells_as_misses():
    gt = np.zeros((20, 40), dtype=np.int32)
    gt[5:15, 5:15] = 1
    gt[5:15, 15:25] = 2
    assert instance_f1(gt, gt)[0] == 1.0
    merged = (gt > 0).astype(np.int32)          # both cells predicted as one object
    f1, n_true, n_pred = instance_f1(gt, merged)
    assert (n_true, n_pred) == (2, 1)
    assert f1 == 0.0


def test_cell_image_has_consecutive_instance_labels():
    img, labels = make_cell_image(seed=3)
    assert img.dtype == np.uint8 and img.shape == labels.shape
    ids = np.unique(labels)
    assert ids[0] == 0 and np.array_equal(ids, np.arange(len(ids)))
    assert len(ids) - 1 > 20


def test_watershed_separates_touching_cells_better_than_otsu():
    scores = {row["method"]: row for row in benchmark(n_images=5)}
    assert scores["Watershed"]["instance_f1"] > scores["Otsu"]["instance_f1"] + 0.1


def test_brightfield_images_are_inverted_before_segmentation():
    img, _ = make_cell_image(seed=1)
    brightfield = 255 - img                     # dark cells on a light background
    assert not has_bright_background(img)
    assert has_bright_background(brightfield)
    assert np.array_equal(run_all(img)["Otsu"], run_all(brightfield)["Otsu"])


def test_convolve2d_matches_scipy_on_non_square_images():
    from scipy.signal import convolve2d as scipy_convolve2d
    rng = np.random.default_rng(0)
    for shape in [(20, 30), (30, 20)]:
        image = rng.random(shape)
        for name in ("sobel_x", "sharpen"):
            out = convolve2d(image, KERNELS[name], padding=1)
            assert out.shape == shape
            assert np.allclose(out, scipy_convolve2d(image, KERNELS[name], mode="same"))
    strided = convolve2d(rng.random((21, 31)), KERNELS["box_blur"], padding=1, stride=2)
    assert strided.shape == (11, 16)


def test_convolve2d_output_shape():
    image = np.random.rand(20, 20)
    out = convolve2d(image, KERNELS["box_blur"], padding=1)
    assert out.shape == (20, 20)
