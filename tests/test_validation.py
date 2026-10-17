"""Every public entry point fails early and clearly on bad input instead of
returning a silently wrong result (review round 6). Run with:
    pytest tests/
"""
import io
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import segmentation as seg
from src.image_io import load_upload, to_8bit

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # CLI checks resolve output folders (public/, ...) from the working directory
    monkeypatch.chdir(ROOT)


# ------------------------------------------------- silent wrong results
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint16])
def test_label_images_of_any_wide_integer_dtype_keep_their_cells(dtype):
    _, gt = seg.make_cell_image(seed=0)
    expected = len(np.unique(seg.label_objects(gt))) - 1
    labels = gt.astype(dtype)
    assert len(np.unique(seg.label_objects(labels))) - 1 == expected
    row = seg.evaluate(labels, {"same": labels})[0]
    assert row["instance_f1"] == 1.0 and row["cells_true"] == expected


def test_uint8_and_bool_are_binary_masks():
    mask = np.zeros((10, 10), np.uint8); mask[2:8, 2:8] = 255
    assert "instance_f1" not in seg.evaluate(mask, {"m": mask})[0]
    assert "instance_f1" not in seg.evaluate(mask > 0, {"m": mask})[0]


@pytest.mark.parametrize("thresh", [0.0, 0.3, 0.49, 1.0])
def test_instance_f1_rejects_thresholds_that_break_one_to_one_matching(thresh):
    labels = np.zeros((5, 5), np.int32); labels[1:4, 1:4] = 1
    with pytest.raises(ValueError, match="iou_thresh"):
        seg.instance_f1(labels, labels, iou_thresh=thresh)


def test_instance_f1_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        seg.instance_f1(np.zeros((4, 6), np.int32), np.zeros((6, 4), np.int32))


def test_nan_pixels_do_not_blank_a_float_image():
    arr = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    arr[0, 0], arr[0, 1] = np.nan, np.inf
    out = np.asarray(to_8bit(Image.fromarray(arr, mode="F"), "L"))
    assert out.max() == 255 and out[0, 0] == 0 and out[0, 1] == 0


def test_unknown_architecture_is_rejected():
    pytest.importorskip("torch")
    from src.medical_cnn import input_transform
    for arch in ("resnet", "ResNet18", ""):
        with pytest.raises(ValueError, match="unknown architecture"):
            input_transform(arch)


# ------------------------------------------------- clear errors
@pytest.mark.parametrize("img", [
    np.zeros((20, 20, 3), np.uint8),            # RGB
    np.zeros((20, 20), np.float32),             # float
    np.zeros((20, 20), np.uint16),              # 16-bit
    np.zeros((20, 20), bool),
])
def test_segmenters_reject_non_uint8_grayscale(img):
    for fn in list(seg.SEGMENTERS.values()) + [seg.run_all, seg.has_bright_background]:
        with pytest.raises(ValueError, match="2-D uint8"):
            fn(img)


def test_load_model_explains_bad_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    from src.medical_cnn import load_model
    truncated = tmp_path / "truncated.pth"
    truncated.write_bytes((ROOT / "biomedical_cnn.pth").read_bytes()[:1000])
    with pytest.raises(ValueError, match="not a readable PyTorch checkpoint"):
        load_model(str(truncated))
    with pytest.raises(ValueError, match="not written by train_medical.py"):
        load_model(str(ROOT / "fashion_model.pth"))
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.pth"))


def test_predict_proba_accepts_arrays_and_follows_the_model_device():
    torch = pytest.importorskip("torch")
    from src.medical_cnn import BloodCNN, predict_proba
    model = BloodCNN(base_channels=4).eval()
    if torch.cuda.is_available():
        model = model.cuda()                     # no device argument needed
    probs = predict_proba(model, np.zeros((28, 28, 3), np.uint8))
    assert abs(probs.sum() - 1) < 1e-5
    with pytest.raises(TypeError, match="PIL image or numpy array"):
        predict_proba(model, "cell.png")
    with pytest.raises(ValueError, match="empty"):
        predict_proba(model, Image.new("RGB", (0, 0)))


def test_load_upload_argument_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_upload(str(tmp_path / "nope.png"))
    buf = io.BytesIO(); Image.new("L", (8, 8)).save(buf, format="PNG"); buf.seek(0)
    with pytest.raises(ValueError, match="max_side"):
        load_upload(buf, max_side=0)


# ------------------------------------------------------------------ CLIs
@pytest.mark.parametrize("argv", [
    ["--tune-epochs", "3"],                                  # without --optuna
    ["--model", "resnet18", "--tune-epochs", "3"],
    ["--epochs", "0"], ["--epochs", "-2"], ["--optuna", "-1"],
    ["--optuna", "2", "--tune-epochs", "0"],
    ["--out", "no_such_folder/model.pth"],
])
def test_train_medical_rejects_invalid_arguments(argv):
    import train_medical
    with pytest.raises(SystemExit) as exc:
        train_medical.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--bogus"], ["--model", "mlp", "--epochs", "1"], ["--epochs", "0"],
                                  ["--model", "svm"], ["--out", "no_such_folder/m.pth"]])
def test_train_py_rejects_unknown_or_invalid_arguments(argv):
    import train
    with pytest.raises(SystemExit) as exc:
        train.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--images", "0"], ["--start-seed", "-1"],
                                  ["--figure", "no_such_folder/fig.png"]])
def test_benchmark_script_validates_before_running(argv):
    import benchmark_segmentation
    with pytest.raises(SystemExit) as exc:
        benchmark_segmentation.main(argv)
    assert exc.value.code == 2


# -------------------------------------------------------------------- app
def test_pages_work_from_another_working_directory(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from streamlit.testing.v1 import AppTest
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)                  # nothing of the project here
    at = AppTest.from_file(str(ROOT / "pages" / "3_Blood_Cell_Classifier.py"), default_timeout=120).run()
    assert not at.exception and not at.error
    assert {m.label for m in at.metric} >= {"DinoBloom-S (224 px)", "Custom CNN (28 px)"}
    home = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not home.exception and len(home.image) == 4


# ---------------------------------------------------------------- security
def test_uploads_accept_only_png_and_jpeg_content():
    from src.image_io import UploadError
    for fmt in ("TIFF", "GIF", "BMP", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 50, 50)).save(buf, format=fmt)
        buf.seek(0)
        with pytest.raises(UploadError):                 # e.g. a .tif renamed to .png
            load_upload(buf)


@pytest.mark.cellpose
def test_cellpose_files_are_hash_checked_and_tls_checks_restored(monkeypatch):
    pytest.importorskip("cellpose")
    import ssl
    monkeypatch.setattr(seg, "_cellpose_model", None)
    monkeypatch.setitem(seg.CELLPOSE_FILE_SHA256, "size_nucleitorch_0.npy", "0" * 64)
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        seg.segment_cellpose(np.zeros((64, 64), np.uint8))
    assert ssl._create_default_https_context is ssl.create_default_context


def test_streamlit_config_limits_uploads_and_error_details():
    tomllib = pytest.importorskip("tomllib")              # Python 3.11+
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] <= 50
    assert config["client"]["showErrorDetails"] in ("type", "none")
