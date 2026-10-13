"""Regression tests for review-loop findings: degenerate inputs, image
formats and reproducibility. Run with:
    pytest tests/
"""
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from PIL import Image

from src.image_io import to_8bit
from src.segmentation import (
    dice, iou, instance_f1, make_cell_image, run_all, segment_adaptive, segment_kmeans,
)
from src.convolution import KERNELS, convolve2d


def png_roundtrip(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Image.open(io.BytesIO(buf.getvalue()))


# ------------------------------------------------------------- segmentation
def test_adaptive_threshold_keeps_flat_background_dark():
    assert not segment_adaptive(np.full((64, 64), 100, np.uint8)).any()
    img, labels = make_cell_image(seed=0)
    background = labels == 0
    false_fg = (segment_adaptive(img) > 0)[background].mean()
    # Sensor noise (sigma ~ 9 grey levels) still crosses a 5-level offset in places,
    # so some background is expected; with the old sign error it was ~35%.
    assert false_fg < 0.20, f"{false_fg:.1%} of background marked as foreground"


@pytest.mark.parametrize("value", [0, 128, 255])
def test_kmeans_on_flat_image_finds_nothing(value):
    assert not segment_kmeans(np.full((32, 32), value, np.uint8)).any()


def test_empty_masks_score_one_on_every_metric():
    empty = np.zeros((10, 10), bool)
    assert dice(empty, empty) == 1.0 and iou(empty, empty) == 1.0
    assert instance_f1(empty.astype(np.int32), empty.astype(np.int32))[0] == 1.0


def test_cell_image_rejects_tiny_sizes_clearly():
    with pytest.raises(ValueError, match="at least 32x32"):
        make_cell_image(20, 20)


# ------------------------------------------------------------- image input
def test_16bit_image_is_rescaled_not_saturated():
    img, labels = make_cell_image(seed=0)
    sixteen = png_roundtrip(Image.fromarray(img.astype(np.uint16) * 257))
    assert sixteen.mode.startswith("I")
    restored = np.asarray(to_8bit(sixteen, "L"))
    assert np.abs(restored.astype(int) - img.astype(int)).max() <= 1
    otsu = run_all(restored)["Otsu"] > 0
    assert dice(labels > 0, otsu) > 0.9


def test_transparent_pixels_become_white():
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    rgba.putpixel((0, 0), (10, 20, 30, 255))
    out = np.asarray(to_8bit(rgba, "RGB"))
    assert tuple(out[0, 0]) == (10, 20, 30) and tuple(out[4, 4]) == (255, 255, 255)


def test_fashion_preprocessing_handles_both_polarities():
    torch = pytest.importorskip("torch")
    from src.vision_cnn import prepare_upload
    item = Image.new("L", (56, 56), 0)
    item.paste(220, (16, 8, 40, 50))                      # light item on black (FashionMNIST style)
    photo = Image.eval(item, lambda v: 255 - v)           # dark item on white (typical photo)
    a, b = prepare_upload(item), prepare_upload(photo)
    assert a.shape == (1, 1, 28, 28)
    assert torch.allclose(a, b, atol=1 / 255 + 1e-6)
    assert a[0, 0, 0, 0] < 0.1 and a[0, 0, 14, 14] > 0.8  # background dark, item light


def test_blood_classifier_accepts_16bit_and_transparent_images():
    pytest.importorskip("torch")
    from src.medical_cnn import BloodCNN, predict_proba
    model = BloodCNN(base_channels=8).eval()
    for img in (Image.fromarray((np.random.default_rng(0).random((40, 40)) * 60000).astype(np.uint16)),
                Image.new("RGBA", (30, 30), (200, 100, 150, 0))):
        assert abs(predict_proba(model, img).sum() - 1) < 1e-5


# ------------------------------------------------------------- utilities
def test_convolve2d_rejects_invalid_arguments():
    img = np.zeros((10, 10))
    for kwargs in ({"stride": 0}, {"padding": -1}):
        with pytest.raises(ValueError):
            convolve2d(img, KERNELS["sharpen"], **kwargs)
    with pytest.raises(ValueError):
        convolve2d(np.zeros((2, 2)), KERNELS["sharpen"])
    with pytest.raises(ValueError):
        convolve2d(np.zeros((10, 10, 3)), KERNELS["sharpen"])


def test_kmeans_explorer_rejects_k_below_two():
    from src.clustering import KMeansExplorer
    for k_range in ([], [1, 2, 3]):
        with pytest.raises(ValueError, match="k >= 2"):
            KMeansExplorer(k_range=k_range)


def test_bench_k_means_silhouette_uses_the_clustered_space():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from src.clustering import bench_k_means
    X, y = make_blobs(200, centers=3, random_state=0)
    X[:, 0] *= 100                                         # very different feature scales
    result = bench_k_means(KMeans(3, n_init=4, random_state=0), "k=3", X, y)
    km = KMeans(3, n_init=4, random_state=0).fit(StandardScaler().fit_transform(X))
    expected = silhouette_score(StandardScaler().fit_transform(X), km.labels_)
    assert result["silhouette"] == pytest.approx(expected, abs=0.02)


def test_decision_tree_is_reproducible():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from src.classification import train_decision_tree
    X, y = load_breast_cancer(return_X_y=True)
    split = train_test_split(X, y, random_state=0)
    assert len({train_decision_tree(*split[::2], *split[1::2]).accuracy for _ in range(5)}) == 1


def test_eda_report_requires_a_numeric_column():
    import pandas as pd
    from src.visualization import eda_report
    with pytest.raises(ValueError, match="numeric"):
        eda_report(pd.DataFrame({"a": ["x", "y"]}))


# ------------------------------------------------------------- upload loading
from src.image_io import MAX_UPLOAD_PIXELS, UploadError, load_upload  # noqa: E402
from src.segmentation import has_bright_background, segment_otsu  # noqa: E402


def png_bytes(img: Image.Image, **save_kwargs) -> io.BytesIO:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **save_kwargs)
    buf.seek(0)
    return buf


@pytest.mark.parametrize("payload", [b"not an image at all", b"", b"\x89PNG\r\n\x1a\n" + b"\0" * 40])
def test_unreadable_uploads_raise_a_friendly_error(payload):
    with pytest.raises(UploadError, match="Could not read"):
        load_upload(io.BytesIO(payload))


def test_truncated_upload_raises_a_friendly_error():
    data = png_bytes(Image.new("RGB", (200, 200), (10, 120, 200))).getvalue()
    with pytest.raises(UploadError):
        load_upload(io.BytesIO(data[: len(data) // 2]))


def test_oversized_upload_is_rejected_before_decoding():
    side = int(MAX_UPLOAD_PIXELS ** 0.5) + 10
    with pytest.raises(UploadError, match="megapixels"):
        load_upload(png_bytes(Image.new("1", (side, side))))


def test_large_16bit_upload_is_downscaled_and_rescaled():
    arr = (np.linspace(0, 60000, 3000 * 2000).reshape(2000, 3000)).astype(np.uint16)
    image, original = load_upload(png_bytes(Image.fromarray(arr)), mode="L", max_side=1024)
    assert original == (3000, 2000)
    assert image.mode == "L" and max(image.size) == 1024
    values = np.asarray(image)
    assert values.min() <= 1 and values.max() >= 254          # full range kept, not saturated


def test_exif_orientation_is_applied():
    upright = Image.new("RGB", (40, 80), "white")              # portrait
    stored = upright.rotate(90, expand=True)                   # how a camera stores it (landscape pixels)
    exif = Image.Exif()
    exif[0x0112] = 6                                           # "rotate 90° clockwise to display"
    buf = io.BytesIO()
    stored.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    image, original = load_upload(buf)
    assert image.size == (40, 80) and original == (40, 80)


@pytest.mark.parametrize("value", [3, 50, 128, 250])
def test_flat_images_are_neither_brightfield_nor_foreground(value):
    flat = np.full((64, 64), value, np.uint8)
    assert not has_bright_background(flat)
    assert not segment_otsu(flat).any()
