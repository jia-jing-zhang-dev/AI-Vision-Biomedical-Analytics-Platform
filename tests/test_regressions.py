"""Tests that pin behaviour precisely enough to catch subtle regressions:
hand-computed metric values, pinned benchmark numbers, shipped-model
accuracy on bundled samples, and the guarantees behind earlier fixes
(no test-set tuning, deterministic training, CLI validation). Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.convolution import convolve2d
from src.segmentation import (
    benchmark, dice, evaluate, has_bright_background, instance_f1, label_objects,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # CLI checks resolve output folders (public/, ...) from the working directory
    monkeypatch.chdir(ROOT)


# ------------------------------------------------------------------ metrics
def test_dice_and_iou_on_a_hand_computed_partial_overlap():
    a = np.zeros(10, bool); a[0:4] = True                 # 4 pixels
    b = np.zeros(10, bool); b[2:8] = True                 # 6 pixels, 2 shared
    assert dice(a, b) == pytest.approx(2 * 2 / (4 + 6))   # 0.4
    from src.segmentation import iou
    assert iou(a, b) == pytest.approx(2 / 8)              # 0.25


def test_instance_f1_on_a_hand_computed_case():
    gt = np.zeros((10, 30), np.int32)
    gt[2:8, 1:7], gt[2:8, 11:17], gt[2:8, 21:27] = 1, 2, 3     # 3 true cells
    pred = np.zeros_like(gt)
    pred[2:8, 1:7] = 1                                         # exact match
    pred[2:4, 11:17] = 2                                       # IoU 1/3 < 0.5: a miss
    f1, n_true, n_pred = instance_f1(gt, pred)
    assert (n_true, n_pred) == (3, 2)
    assert f1 == pytest.approx(2 * 1 / (3 + 2))                # 0.4, not recall 1/3


def test_label_objects_filters_small_objects_without_touching_its_input():
    labels = np.zeros((20, 20), np.int32)
    labels[0:4, 0:5] = 1             # 20 px: kept
    labels[10:13, 10:16] = 2         # 18 px: dropped
    before = labels.copy()
    out = label_objects(labels, min_size=20)
    assert np.array_equal(labels, before)                      # caller's array unchanged
    assert set(np.unique(out)) == {0, 1}

    mask = np.zeros((20, 20), np.uint8)
    mask[0:5, 0:5] = 255
    mask[5:10, 5:10] = 255                                     # touches only diagonally
    assert label_objects(mask, min_size=1).max() == 1          # 8-connectivity: one object


def test_evaluate_rows_are_sorted_by_dice():
    gt = np.zeros((10, 10), np.uint8); gt[2:8, 2:8] = 1
    worse = np.zeros_like(gt); worse[2:5, 2:5] = 1
    rows = evaluate(gt, {"worse": worse, "perfect": gt.copy()})
    assert [r["method"] for r in rows] == ["perfect", "worse"]


# ------------------------------------------------------------ segmentation
def test_benchmark_numbers_are_pinned():
    # Guards the README's segmentation results against silent method changes.
    scores = {r["method"]: r for r in benchmark(n_images=5)}
    assert scores["Watershed"]["instance_f1"] == pytest.approx(0.8111, abs=0.01)
    assert scores["Otsu"]["instance_f1"] == pytest.approx(0.5820, abs=0.01)
    assert scores["Watershed"]["dice"] == pytest.approx(0.9435, abs=0.01)
    assert scores["Adaptive"]["dice"] == pytest.approx(0.7957, abs=0.01)


def test_bright_object_on_dark_background_is_not_brightfield():
    img = np.full((60, 60), 20, np.uint8)
    img[5:55, 5:55] = 220                       # bright object covering ~70% of the image
    assert not has_bright_background(img)


def test_brightfield_is_judged_by_the_whole_border():
    # Tall image: the light side columns are 75% of the border, the top and
    # bottom rows are dark. Checking only some edges, or demanding 90%, fails.
    img = np.full((60, 20), 30, np.uint8)
    img[:, :2] = img[:, -2:] = 230
    img[25:35, 8:12] = 230
    assert has_bright_background(img)


# ------------------------------------------------------------- convolution
def test_convolve2d_matches_scipy_for_asymmetric_kernels_and_stride():
    from scipy.signal import convolve2d as scipy_convolve2d
    rng = np.random.default_rng(1)
    image, kernel = rng.random((17, 23)), rng.random((3, 4))   # no symmetry at all
    assert np.allclose(convolve2d(image, kernel, padding=0), scipy_convolve2d(image, kernel, mode="valid"))
    kernel3 = rng.random((3, 3))
    same = scipy_convolve2d(image, kernel3, mode="same")
    assert np.allclose(convolve2d(image, kernel3, padding=1, stride=2), same[::2, ::2])


# ------------------------------------------------------------ image input
def test_palette_and_la_transparency_become_white():
    from src.image_io import to_8bit
    p = Image.new("P", (4, 4), 0)
    p.putpalette([0, 0, 0] * 256)
    p.info["transparency"] = 0
    la = Image.new("LA", (4, 4), (0, 0))
    for img in (p, la):
        assert tuple(np.asarray(to_8bit(img, "RGB"))[0, 0]) == (255, 255, 255)


def test_flat_16bit_image_maps_to_black():
    from src.image_io import to_8bit
    flat = Image.fromarray(np.full((8, 8), 40000, np.uint16))
    assert np.asarray(to_8bit(flat, "L")).max() == 0


def test_predict_proba_uses_the_normalized_image():
    pytest.importorskip("torch")
    from src.image_io import to_8bit
    from src.medical_cnn import load_model, predict_proba
    model = load_model(str(ROOT / "biomedical_cnn.pth"))
    sample = np.asarray(Image.open(ROOT / "public/blood_samples/3_immature_granulocyte.png").convert("L"))
    img16 = Image.fromarray(sample.astype(np.uint16) * 200)
    assert np.allclose(predict_proba(model, img16), predict_proba(model, to_8bit(img16, "RGB")))


# --------------------------------------------------------- shipped models
def test_fashion_model_classifies_bundled_samples_in_both_polarities():
    torch = pytest.importorskip("torch")
    from PIL import ImageOps
    from src.vision_cnn import FASHION_MNIST_CLASSES, build_model, prepare_upload
    model, device = build_model()
    model.load_state_dict(torch.load(ROOT / "fashion_model.pth", map_location=device, weights_only=True))
    model.eval()
    samples = sorted((ROOT / "public" / "fashion_samples").glob("*.png"))
    assert len(samples) == len(FASHION_MNIST_CLASSES)
    for transform in (lambda im: im, ImageOps.invert):          # FashionMNIST style and photo style
        correct = 0
        with torch.no_grad():
            for path in samples:
                x = prepare_upload(transform(Image.open(path).convert("L"))).to(device)
                correct += model(x).argmax(1).item() == int(path.stem.split("_", 1)[0])
        assert correct >= 8, correct                            # model is 86.9% accurate


# ------------------------------------------------- tuning never sees test
def test_optuna_tunes_on_training_data_only_and_scales_svc(monkeypatch):
    pytest.importorskip("optuna")
    import sklearn.model_selection
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src import classification

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    calls, real = [], sklearn.model_selection.cross_val_score

    def spy(estimator, X_, y_, **kw):
        calls.append((estimator, X_))
        return real(estimator, X_, y_, **kw)

    monkeypatch.setattr(sklearn.model_selection, "cross_val_score", spy)
    study = classification.tune_with_optuna(X_train, y_train, X_test, y_test, n_trials=8)
    assert calls and all(X_ is X_train for _, X_ in calls)
    svcs = [e for e, _ in calls if "SVC" in type(e).__name__ or isinstance(e, Pipeline)]
    assert svcs and all(isinstance(e, Pipeline) and isinstance(e.steps[0][1], StandardScaler) for e in svcs)
    best = study.user_attrs["test_accuracy"]
    assert best != pytest.approx(study.best_value, abs=1e-12)   # reported on test, not the CV score


def test_compare_classifiers_ranks_by_training_cross_validation(monkeypatch):
    import sklearn.model_selection
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from src.classification import compare_classifiers
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    seen, real = [], sklearn.model_selection.cross_val_score
    monkeypatch.setattr(sklearn.model_selection, "cross_val_score",
                        lambda est, X_, y_, **kw: seen.append(X_) or real(est, X_, y_, **kw))
    compare_classifiers(X_train, y_train, X_test, y_test)
    assert len(seen) == 3 and all(X_ is X_train for X_ in seen)


# ------------------------------------------------------------- training
def test_fit_restores_the_best_validation_epoch(monkeypatch):
    torch = pytest.importorskip("torch")
    from src import medical_cnn
    model = torch.nn.Linear(1, 1)
    scripted = iter([0.5, 0.9, 0.7])

    def fake_epoch(m, *args):
        with torch.no_grad():
            m.weight.fill_(m.weight.item() + 1)               # weights differ every epoch
        return 0.0

    monkeypatch.setattr(medical_cnn, "train_one_epoch", fake_epoch)
    monkeypatch.setattr(medical_cnn, "evaluate", lambda *a: (0.0, next(scripted)))
    with torch.no_grad():
        model.weight.fill_(0)
    best = medical_cnn.fit(model, None, None, "cpu", epochs=3, verbose=False)
    assert best == 0.9 and model.weight.item() == 2              # weights after epoch 2


def test_fit_raises_when_validation_never_improves(monkeypatch):
    torch = pytest.importorskip("torch")
    from src import medical_cnn
    monkeypatch.setattr(medical_cnn, "train_one_epoch", lambda *a: 0.0)
    monkeypatch.setattr(medical_cnn, "evaluate", lambda *a: (0.0, 0.0))
    with pytest.raises(RuntimeError, match="never rose"):
        medical_cnn.fit(torch.nn.Linear(1, 1), None, None, "cpu", epochs=2, verbose=False)


def test_set_deterministic_enables_reproducible_training():
    torch = pytest.importorskip("torch")
    from src.medical_cnn import BloodCNN, set_deterministic
    try:
        outputs = []
        for _ in range(2):
            set_deterministic(123)
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            model = BloodCNN(base_channels=4).train()             # dropout active
            outputs.append(model(torch.rand(2, 3, 28, 28)))
        assert torch.equal(outputs[0], outputs[1])
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


# ------------------------------------------------------------------ CLIs
@pytest.mark.parametrize("argv", [
    ["--model", "resnet18", "--optuna", "5"],
    ["--model", "mlp", "--optuna", "5"],
    ["--model", "mlp", "--epochs", "3"],
    ["--model", "mlp", "--out", "x.pth"],
])
def test_train_medical_rejects_inapplicable_options(argv):
    import train_medical
    with pytest.raises(SystemExit) as exc:
        train_medical.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv, epochs", [([], 15), (["--model", "resnet18"], 10),
                                          (["--model", "cnn", "--optuna", "3"], 15),
                                          (["--epochs", "4"], 4)])
def test_train_medical_accepts_valid_options(argv, epochs):
    import train_medical
    assert train_medical.parse_args(argv).epochs == epochs


def test_experiment_outputs_never_overwrite_shipped_files():
    import train_medical
    assert train_medical.output_paths("cnn") == {
        "weights": "biomedical_cnn.pth", "results": "biomedical_cnn_results.json",
        "figure": "public/bloodmnist-confusion.png"}
    assert train_medical.output_paths("resnet18")["figure"] == "public/bloodmnist-confusion-resnet18.png"
    exp = train_medical.output_paths("cnn", "runs/model")             # no .pth suffix
    assert exp == {"weights": "runs/model", "results": "runs/model_results.json",
                   "figure": "runs/model_confusion.png"}


def test_benchmark_script_runs_and_writes_its_figure(tmp_path, capsys):
    import benchmark_segmentation
    figure = tmp_path / "fig.png"
    benchmark_segmentation.main(["--images", "1", "--no-cellpose", "--figure", str(figure)])
    assert figure.exists() and "Watershed" in capsys.readouterr().out


# -------------------------------------------------------------- Cellpose
@pytest.mark.cellpose
def test_cellpose_falls_back_when_auto_diameter_finds_nothing():
    pytest.importorskip("cellpose")
    from src.segmentation import segment_cellpose
    img = np.full((512, 512), 15, np.uint8)
    for cy, cx in [(130, 130), (130, 380), (380, 255)]:          # three cells ~140 px across
        yy, xx = np.ogrid[:512, :512]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < 70 ** 2] = 200
    assert segment_cellpose(img).max() >= 2


# ------------------------------------------------------- round 10 additions
def test_label_ids_are_compacted_so_memory_follows_object_count():
    from src.segmentation import color_labels
    labels = np.zeros((64, 64), np.int64)
    labels[5:25, 5:25], labels[40:60, 40:60] = 10**8, 65535      # two objects, huge ids
    out = label_objects(labels, min_size=1)
    assert out.dtype == np.int32 and set(np.unique(out)) == {0, 1, 2}
    assert instance_f1(labels, labels) == (1.0, 2, 2)
    import tracemalloc
    tracemalloc.start()
    try:
        colors = color_labels(labels)
        label_objects(labels, min_size=1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert colors.dtype == np.uint8 and colors.shape == (64, 64, 3)
    assert peak < 20 * 2**20, f"peak {peak / 2**20:.0f} MiB: per-id arrays sized by the largest id"


def test_negative_label_ids_are_rejected_not_counted_as_cells():
    labels = np.zeros((10, 10), np.int32)
    labels[2:6, 2:6] = 1
    labels[0, 0] = -1
    for fn in (lambda: instance_f1(labels, labels), lambda: label_objects(labels),
               lambda: evaluate(labels, {"p": labels})):
        with pytest.raises(ValueError, match="negative ids"):
            fn()


def test_mismatched_checkpoints_raise_value_error(tmp_path):
    torch = pytest.importorskip("torch")
    from src.medical_cnn import BloodCNN, load_model
    good = BloodCNN(base_channels=8).state_dict()
    cases = {
        "wrong_sizes": {"params": {"base_channels": 16, "dropout": 0.1}, "state_dict": good},
        "missing_param": {"params": {"dropout": 0.1}, "state_dict": good},
        "params_list": {"params": [1, 2], "state_dict": good},
    }
    for name, checkpoint in cases.items():
        path = tmp_path / f"{name}.pth"
        torch.save(checkpoint, path)
        with pytest.raises(ValueError, match="does not match the expected model"):
            load_model(str(path))


def test_very_thin_uploads_are_rejected():
    import io
    from src.image_io import UploadError, load_upload
    for size in ((5000, 1), (1, 2000), (600, 10)):
        buf = io.BytesIO(); Image.new("L", size, 128).save(buf, format="PNG"); buf.seek(0)
        with pytest.raises(UploadError, match="wider than tall"):
            load_upload(buf)
    buf = io.BytesIO(); Image.new("L", (1, 1), 128).save(buf, format="PNG"); buf.seek(0)
    assert load_upload(buf)[0].size == (1, 1)


def test_train_medical_checks_the_figure_folder_too(tmp_path, monkeypatch):
    import train_medical
    monkeypatch.chdir(tmp_path)                       # no public/ folder here
    with pytest.raises(SystemExit) as exc:
        train_medical.parse_args([])
    assert exc.value.code == 2


def test_benchmark_never_replaces_the_shipped_figure_without_cellpose(capsys):
    import benchmark_segmentation
    shipped = ROOT / "public" / "cell-segmentation-benchmark.png"
    before = shipped.read_bytes()
    benchmark_segmentation.main(["--images", "1", "--no-cellpose"])
    assert shipped.read_bytes() == before
    assert "Not updating" in capsys.readouterr().out


def test_grad_cam_matches_pytorch_grad_cam_library():
    # The app's hook-free Grad-CAM must equal the widely used reference library.
    pytest.importorskip("pytorch_grad_cam")
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from src.explain import grad_cam
    from src.image_io import to_8bit
    from src.medical_cnn import input_transform, load_model
    img = Image.open(ROOT / "public" / "blood_samples" / "5_monocyte.png")
    for weights in ("biomedical_cnn.pth", "biomedical_resnet18.pth"):
        model = load_model(str(ROOT / weights))
        layer = model.layer4[-1] if model.arch == "resnet18" else model.features[-2]
        mine, target = grad_cam(model, img)
        x = input_transform(model.arch, from_upload=True)(to_8bit(img, "RGB")).unsqueeze(0)
        with GradCAM(model=model, target_layers=[layer]) as cam:
            ref = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
        assert np.corrcoef(mine.ravel(), ref.ravel())[0, 1] > 0.999, weights


def test_grad_cam_leaves_the_shared_model_untouched():
    torch = pytest.importorskip("torch")
    from src.explain import grad_cam
    from src.medical_cnn import load_model
    model = load_model(str(ROOT / "biomedical_cnn.pth"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    heatmap, _ = grad_cam(model, Image.open(ROOT / "public" / "blood_samples" / "0_basophil.png"))
    assert heatmap.shape == (28, 28) and 0 <= heatmap.min() and heatmap.max() <= 1
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and not model.training


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_fashion_product_photos_are_cropped_like_fashionmnist(arch):
    """A user's product photo (dark item, small and off-centre on an off-white,
    non-square background) was squeezed whole to 28x28: an ankle boot came out
    as "Sandal". Every bundled sample, turned into such a photo and also
    mirrored, must now be classified correctly."""
    import numpy as np
    import torch
    from PIL import Image
    from src.vision_cnn import build_cnn, build_model, prepare_upload

    builder, weights = (build_cnn, "fashion_cnn.pth") if arch == "cnn" else (build_model, "fashion_model.pth")
    model, device = builder()
    model.load_state_dict(torch.load(ROOT / weights, map_location=device, weights_only=True))
    model.eval()
    rng = np.random.default_rng(0)
    for path in sorted((ROOT / "public" / "fashion_samples").glob("*.png")):
        item = Image.open(path).convert("L")
        up = np.asarray(item.resize((150, 150), Image.Resampling.BICUBIC), np.float32)
        canvas = np.full((300, 420), 240.0) + rng.normal(0, 2, (300, 420))
        canvas[40:190, 230:380] = 240 - up * 220 / 255
        photo = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        expected = int(path.stem.split("_", 1)[0])                      # true class
        with torch.no_grad():
            got = model(prepare_upload(photo).to(device)).argmax(1).item()
            # the owner's boot faced the other way; mirrored shoes were never recognized
            mirrored = model(prepare_upload(photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).to(device)).argmax(1).item()
        assert got == expected, path.name
        assert mirrored == expected, f"{path.name} (mirrored)"


def test_fashion_cnn_is_robust_where_the_baseline_fails():
    """The CNN (flip/shift/scale augmentation) must recognize every bundled
    sample mirrored and shifted, without the preprocessing's mirror rule, and
    its recorded test accuracy must beat the fully connected baseline."""
    import json
    import numpy as np
    import torch
    from PIL import Image
    from src.vision_cnn import build_cnn

    model, device = build_cnn()
    model.load_state_dict(torch.load(ROOT / "fashion_cnn.pth", map_location=device, weights_only=True))
    model.eval()
    for path in sorted((ROOT / "public" / "fashion_samples").glob("*.png")):
        x = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
        label = int(path.stem.split("_", 1)[0])
        for variant in (x, x[:, ::-1], np.roll(x, 2, axis=1)):
            t = torch.from_numpy(np.ascontiguousarray(variant))[None, None].to(device)
            with torch.no_grad():
                assert model(t).argmax(1).item() == label, path.name
    results = json.loads((ROOT / "fashion_cnn_results.json").read_text())
    assert results["test_accuracy"] > 0.90 and results["best_epoch"] <= results["epochs"]
