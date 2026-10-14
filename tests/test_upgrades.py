"""Tests for the BBBC039 loader, U-Net, Grad-CAM UI and calibration. Run with:
    pytest tests/
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
torch = pytest.importorskip("torch")


# ----------------------------------------------------------------- BBBC039
def test_decode_mask_separates_touching_nuclei_and_reused_colours():
    from src.bbbc039 import decode_mask
    colours = np.zeros((10, 20, 4), np.uint8)
    colours[1:5, 1:5, 0] = 1
    colours[1:5, 5:9, 0] = 2          # touches nucleus 1, different colour
    colours[6:9, 12:16, 0] = 1        # same colour as nucleus 1, far away
    labels = decode_mask(colours)
    assert labels.dtype == np.int32 and labels.max() == 3


def test_percentile_normalization_ignores_a_few_saturated_pixels():
    from src.bbbc039 import normalize_to_uint8
    img = np.full((100, 100), 150, np.uint16)
    img[40:60, 40:60] = 900                       # nuclei
    img[0, :5] = 4095                             # saturated hot pixels
    out = normalize_to_uint8(img)
    assert out.dtype == np.uint8 and out[50, 50] == 255 and out[10, 10] == 0


# --------------------------------------------------------------------- U-Net
def test_unet_boundary_band_follows_caicedo_et_al():
    from src.unet import BOUNDARY, INTERIOR, labels_to_classes
    labels = np.zeros((14, 14), np.int32)
    labels[3:11, 3:11] = 1
    cls = labels_to_classes(labels)
    assert cls[2, 6] == BOUNDARY            # one pixel outside the nucleus
    assert cls[3, 6] == BOUNDARY and cls[4, 6] == BOUNDARY   # contour + one pixel inside
    assert cls[6, 6] == INTERIOR and cls[0, 0] == 0


def test_unet_postprocessing_splits_touching_nuclei():
    from src.unet import classes_to_labels, labels_to_classes
    labels = np.zeros((20, 30), np.int32)
    labels[3:17, 2:14], labels[3:17, 14:26] = 1, 2
    cls = labels_to_classes(labels)
    probs = np.stack([(cls == k).astype(np.float32) for k in range(3)])
    assert classes_to_labels(probs).max() == 2


def test_unet_handles_any_image_size():
    from src.unet import UNet, predict_probs
    model = UNet(base=4).eval()
    for shape in ((37, 53), (1, 1), (520, 696)):
        probs = predict_probs(model, np.zeros(shape, np.uint8))
        assert probs.shape == (3,) + shape and np.allclose(probs.sum(0), 1, atol=1e-5)


@pytest.mark.skipif(not (ROOT / "unet_bbbc039.pth").exists(), reason="U-Net weights not present")
def test_shipped_unet_segments_synthetic_nuclei():
    from src.segmentation import evaluate, make_cell_image
    from src.unet import load_unet, segment_unet
    img, labels = make_cell_image(seed=3)
    f1 = evaluate(labels, {"U-Net": segment_unet(load_unet(str(ROOT / "unet_bbbc039.pth")), img)})[0]["instance_f1"]
    assert f1 > 0.5                                   # trained on real data, still finds these nuclei


def test_average_f1_is_mean_over_iou_thresholds():
    from src.segmentation import AF1_THRESHOLDS, average_f1, instance_f1
    gt = np.zeros((20, 20), np.int32); gt[2:12, 2:12] = 1
    pred = np.zeros_like(gt); pred[2:12, 2:10] = 1             # IoU exactly 0.8
    assert AF1_THRESHOLDS[0] == 0.5 and AF1_THRESHOLDS[-1] == 0.9
    expected = np.mean([instance_f1(gt, pred, t)[0] for t in AF1_THRESHOLDS])
    # a match needs IoU > threshold: 0.50 ... 0.75 match, 0.80 ... 0.90 do not
    assert average_f1(gt, pred) == pytest.approx(expected) == pytest.approx(6 / 9)


def test_bbbc039_benchmark_validates_arguments(tmp_path):
    import benchmark_bbbc039
    for argv in (["--data-dir", str(tmp_path)], ["--unet", "missing.pth", "--data-dir", str(tmp_path)]):
        with pytest.raises(SystemExit) as exc:
            benchmark_bbbc039.main(argv)
        assert exc.value.code == 2


# --------------------------------------------------------------- calibration
def test_fit_temperature_recovers_a_known_temperature():
    pytest.importorskip("torchmetrics")
    from src.calibration import fit_temperature
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 8) * 2
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    assert fit_temperature(true_logits * 3.0, labels) == pytest.approx(3.0, rel=0.03)


def test_ece_matches_a_hand_computation():
    pytest.importorskip("torchmetrics")
    from src.calibration import expected_calibration_error
    probs = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    labels = torch.tensor([0, 1, 0, 0])
    # bins: conf 0.9 -> acc 0.5 (gap 0.4, weight 1/2); conf 0.6 -> acc 1.0 (gap 0.4, weight 1/2)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.4, abs=1e-6)


def test_temperature_is_applied_and_keeps_predictions():
    from PIL import Image
    from src.medical_cnn import load_model, predict_proba
    model = load_model(str(ROOT / "biomedical_cnn.pth"))
    img = Image.open(ROOT / "public" / "blood_samples" / "2_erythroblast.png")
    calibrated = predict_proba(model, img)
    model.temperature = 1.0
    raw = predict_proba(model, img)
    assert calibrated.argmax() == raw.argmax()
    if (ROOT / "biomedical_cnn_calibration.json").exists():
        assert not np.allclose(calibrated, raw)


def test_unet_classes_round_trip_reproduces_nucleus_outlines():
    # Encoding labels as classes and decoding them must give back the same
    # nuclei (the decoder once returned every nucleus one pixel too large).
    from src.segmentation import iou
    from src.unet import classes_to_labels, labels_to_classes
    labels = np.zeros((40, 60), np.int32)
    yy, xx = np.mgrid[:40, :60]
    labels[(yy - 20) ** 2 + (xx - 15) ** 2 < 100] = 1
    labels[(yy - 20) ** 2 + (xx - 42) ** 2 < 130] = 2
    cls = labels_to_classes(labels)
    decoded = classes_to_labels(np.stack([(cls == k).astype(np.float32) for k in range(3)]))
    assert decoded.max() == 2
    for k in (1, 2):
        best = max(iou(labels == k, decoded == j) for j in (1, 2))
        assert best > 0.97, (k, best)
