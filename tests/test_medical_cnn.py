"""Tests for the BloodMNIST CNN module. Run with:
    pytest tests/
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from src.medical_cnn import BloodCNN, CLASS_NAMES, N_CLASSES, build_mlp_baseline, load_model, predict_proba

ROOT = Path(__file__).resolve().parent.parent


def test_cnn_output_shape():
    model = BloodCNN(base_channels=8)
    out = model(torch.rand(4, 3, 28, 28))
    assert out.shape == (4, N_CLASSES)


def test_mlp_baseline_output_shape():
    out = build_mlp_baseline()(torch.rand(2, 3, 28, 28))
    assert out.shape == (2, N_CLASSES)


def test_predict_proba_sums_to_one_for_any_image_size():
    model = BloodCNN(base_channels=8).eval()
    probs = predict_proba(model, Image.new("L", (100, 60)))
    assert probs.shape == (N_CLASSES,)
    assert abs(probs.sum() - 1.0) < 1e-5


@pytest.mark.skipif(not (ROOT / "biomedical_cnn.pth").exists(), reason="trained weights not present")
def test_trained_models_classify_bundled_samples():
    # Each shipped model is ~97% accurate, so at most one miss among the 8
    # samples (one per class) is expected; more indicates broken preprocessing.
    for weights in ("biomedical_cnn.pth", "biomedical_resnet18.pth"):
        model = load_model(str(ROOT / weights))
        misses = []
        for path in sorted((ROOT / "public" / "blood_samples").glob("*.png")):
            probs = predict_proba(model, Image.open(path))
            if probs.argmax() != int(path.stem.split("_", 1)[0]):
                misses.append(f"{path.name} -> {CLASS_NAMES[probs.argmax()]}")
        assert len(misses) <= 1, (weights, misses)
