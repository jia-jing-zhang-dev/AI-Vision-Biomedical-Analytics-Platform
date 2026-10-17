"""Colour robustness helpers (src/color_shift.py). Run with:
    pytest tests/
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
from PIL import Image

from src.color_shift import lab_stats, perturb, reinhard

SAMPLE = Path(__file__).resolve().parent.parent / "public" / "blood_samples" / "1_eosinophil.png"


def test_perturb_is_seeded_and_scales_with_severity():
    img = Image.open(SAMPLE).convert("RGB")
    a, b = perturb(img, "strong", 3), perturb(img, "strong", 3)
    assert np.array_equal(np.asarray(a), np.asarray(b))                  # reproducible
    assert a.size == img.size
    base = np.asarray(img, np.float32)
    change = {s: np.mean([np.abs(np.asarray(perturb(img, s, i), np.float32) - base).mean() for i in range(20)])
              for s in ("mild", "strong")}
    assert 0 < change["mild"] < change["strong"]


def test_reinhard_matches_target_lab_statistics():
    img = Image.open(SAMPLE).convert("RGB")
    target_mean, target_std = np.array([150.0, 140.0, 120.0]), np.array([40.0, 20.0, 15.0])
    out = reinhard(img, target_mean, target_std)
    mean, std = lab_stats([out])
    np.testing.assert_allclose(mean, target_mean, atol=3)
    np.testing.assert_allclose(std, target_std, rtol=0.1)
