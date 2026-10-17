"""Finding and cropping stained cells in blood smears (src/blood_cells.py).
Run with:
    pytest tests/
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.blood_cells import crop_cells, crop_side, find_cells, is_smear

SAMPLES = Path(__file__).resolve().parent.parent / "public" / "blood_samples"


def sample(name: str) -> np.ndarray:
    return np.asarray(Image.open(SAMPLES / f"{name}.png").convert("RGB"))


def test_mosaic_of_leukocytes_gives_one_centred_crop_per_cell():
    names = ["0_basophil", "1_eosinophil", "4_lymphocyte", "6_neutrophil"]
    tiles = [sample(n) for n in names]
    s = tiles[0].shape[0]
    smear = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cells = find_cells(smear)
    assert is_smear(cells) and len(cells) == 4
    tiles_hit = sorted((cy // s) * 2 + cx // s for cx, cy, _ in cells)
    assert tiles_hit == [0, 1, 2, 3]                                   # one per tile
    side = crop_side(cells)
    assert 0.8 * s <= side <= 1.25 * s                                 # BloodMNIST framing
    crops = crop_cells(Image.fromarray(smear), cells)
    assert all(c.size == (side, side) for c in crops)


def test_single_cell_images_are_not_smears():
    for name in ["0_basophil", "2_erythroblast", "4_lymphocyte", "7_platelet"]:
        assert not is_smear(find_cells(sample(name))), name


def test_unstained_image_has_no_cells_and_crops_survive_the_border():
    pink = np.zeros((100, 100, 3), np.uint8)
    pink[:] = (230, 180, 190)                                          # red cells / background only
    assert find_cells(pink) == []
    img = sample("6_neutrophil")
    edge = [(2, 2, 40)]                                                # cell at the corner
    crop = crop_cells(Image.fromarray(img), edge, side=60)[0]
    assert crop.size == (60, 60)


def test_find_cells_rejects_non_rgb():
    with pytest.raises(ValueError):
        find_cells(np.zeros((10, 10), np.uint8))


def test_consensus_needs_agreement_and_confidence():
    from src.blood_cells import REVIEW_CONFIDENCE, consensus
    sure = np.eye(8)[6] * 0.95 + 0.05 / 8                       # confident "neutrophil"
    unsure = np.full(8, 0.02); unsure[6] = REVIEW_CONFIDENCE - 0.05
    other = np.eye(8)[3] * 0.95 + 0.05 / 8                      # confident "immature granulocyte"
    assert consensus([sure, sure]) == 6
    assert consensus([sure, unsure]) is None                   # one model not confident enough
    assert consensus([sure, other]) is None                    # models disagree


def test_differential_count_leaves_review_cells_out_of_the_shares():
    from src.blood_cells import differential_count
    rows = dict((name, (n, share)) for name, n, share in
                differential_count([6, 6, 6, 4, None, None], ["a", "b", "c", "d", "e", "f", "neut", "h"]))
    assert rows["neut"] == (3, 0.75) and rows["e"] == (1, 0.25) and rows["a"] == (0, 0.0)
    assert all(share == 0.0 for _, _, share in differential_count([None, None], ["x", "y"]))
