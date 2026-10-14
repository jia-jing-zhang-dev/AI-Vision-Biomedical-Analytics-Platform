"""End-to-end tests of the Streamlit app: every page, button, selector and
upload path, run headlessly with Streamlit's AppTest. Run with:
    pytest tests/
"""
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
N_METHODS_WITH_CELLPOSE = 6 + int(importlib.util.find_spec("cellpose") is not None)
N_METHODS_WITH_CELLPOSE += int((ROOT / "unet_bbbc039.pth").exists())   # + U-Net


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch, request):
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(ROOT))
    # Cellpose costs 3-14 s per page run on CPU. Only tests marked
    # @pytest.mark.cellpose run it; the rest see the deployment's setup.
    if request.node.get_closest_marker("cellpose") is None:
        import src.segmentation
        monkeypatch.setattr(src.segmentation, "cellpose_available", lambda: False)


def run_page(name: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / name), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def click(at: AppTest, label_start: str) -> AppTest:
    next(b for b in at.button if b.label.startswith(label_start)).click().run()
    assert not at.exception, at.exception
    return at


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def cells_image(size: int, light_background: bool) -> Image.Image:
    """Circles, two pairs of them touching, on a plain background."""
    bg, fg = (230, 110) if light_background else (20, 200)
    img = Image.new("L", (size, size), bg)
    draw = ImageDraw.Draw(img)
    r = size // 18
    for x, y in [(0.2, 0.2), (0.2 + 1.9 * r / size, 0.2), (0.5, 0.5), (0.8, 0.3),
                 (0.3, 0.75), (0.3 + 1.9 * r / size, 0.75), (0.75, 0.8)]:
        cx, cy = x * size, y * size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    return img


# ---------------------------------------------------------------- home page
def test_home_page_shows_three_modules_and_results():
    at = run_page("app.py")
    assert len(at.subheader) == 3
    text = " ".join(m.value for m in at.markdown)
    for number in ("96.7%", "72.9%", "97.9%", "0.80", "0.90", "86.9%"):
        assert number in text
    assert len(at.image) == 4


# ------------------------------------------------------- FashionMNIST page
def test_fashion_page_classifies_an_uploaded_image():
    at = run_page("pages/1_Deep_Learning.py")
    assert not at.button                                  # no button before upload
    img = Image.new("RGB", (120, 160), "white")
    ImageDraw.Draw(img).rectangle([40, 20, 80, 150], fill="black")
    at.get("file_uploader")[0].upload("item.png", png_bytes(img), "image/png").run()
    click(at, "Run the PyTorch Model")
    assert at.success[0].value.startswith("**Prediction Result:**")


# --------------------------------------------------------- segmentation page
SEG = "pages/2_Biomedical_Segmentation.py"


@pytest.mark.cellpose
def test_segmentation_realistic_benchmark_reports_instance_f1():
    at = click(run_page(SEG), "Run Biomedical Image Segmentation Pipeline")
    table = at.dataframe[0].value
    assert {"Dice", "IoU", "Instance F1", "Cells (true)", "Objects found"} <= set(table.columns)
    assert len(table) == N_METHODS_WITH_CELLPOSE
    order = list(table["Method"])                         # sorted by instance F1
    assert order.index("Watershed") < order.index("Otsu")
    deep = [m for m in ("Cellpose", "U-Net") if m in order]
    if deep:                                              # deep-learning methods rank above the classical ones
        assert order[0] in deep
    assert any("Ground Truth (" in s.value for s in at.subheader)


def test_segmentation_seed_changes_the_image():
    at = click(run_page(SEG), "Run Biomedical Image Segmentation Pipeline")
    first = at.dataframe[0].value.set_index("Method")
    at.number_input[0].set_value(7).run()
    click(at, "Run Biomedical Image Segmentation Pipeline")
    second = at.dataframe[0].value.set_index("Method")
    assert not first["Dice"].equals(second.loc[first.index, "Dice"])


def test_segmentation_simple_shapes_benchmark():
    at = run_page(SEG)
    at.selectbox[0].set_value(at.selectbox[0].options[1]).run()
    assert not at.number_input                             # no seed for the fixed image
    click(at, "Run Biomedical Image Segmentation Pipeline")
    table = at.dataframe[0].value.set_index("Method")
    assert "Instance F1" not in table.columns
    assert table.loc["Otsu", "Dice"] == pytest.approx(1.0, abs=1e-3)


def test_segmentation_ten_image_benchmark():
    at = click(run_page(SEG), "Run 10-Image Benchmark")
    table = at.dataframe[0].value
    n_unet = int((ROOT / "unet_bbbc039.pth").exists())
    assert len(table) == 6 + n_unet
    order = list(table["Method"])
    assert order[0] == ("U-Net" if n_unet else "Watershed")   # U-Net 0.86 vs watershed 0.81 on these images
    assert order.index("Watershed") < order.index("Otsu")
    assert all("±" in v for v in table["Instance F1"])


def test_segmentation_upload_brightfield_is_inverted_and_large_image_resized():
    at = run_page(SEG)
    at.radio[0].set_value(at.radio[0].options[1]).run()
    img = cells_image(1600, light_background=True)
    at.get("file_uploader")[0].upload("smear.png", png_bytes(img), "image/png").run()
    assert any("Resized from 1600×1600 to 1024×1024" in c.value for c in at.caption)
    click(at, "Run Multi-Algorithm Segmentation")
    assert any("brightfield" in i.value for i in at.info)


def test_segmentation_upload_fluorescence_is_not_inverted():
    at = run_page(SEG)
    at.radio[0].set_value(at.radio[0].options[1]).run()
    img = cells_image(400, light_background=False)
    at.get("file_uploader")[0].upload("nuclei.png", png_bytes(img), "image/png").run()
    assert not any("Resized" in c.value for c in at.caption)
    click(at, "Run Multi-Algorithm Segmentation")
    assert not at.info


# ------------------------------------------------------ blood-cell page
BLOOD = "pages/3_Blood_Cell_Classifier.py"


def test_blood_page_shows_metrics():
    at = run_page(BLOOD)
    values = {m.label: m.value for m in at.metric}
    assert values["Custom CNN"] == "96.7%"
    assert values["ResNet-18 (pretrained)"] == "97.9%"
    assert values["Fully connected baseline"] == "72.9%"


def test_blood_page_classifies_every_bundled_sample():
    at = run_page(BLOOD)
    box = at.selectbox[0]
    assert len(box.options) == 8
    correct = [0, 0]
    for option in box.options:                            # option text = true label
        at.selectbox[0].set_value(option).run()
        assert not at.exception
        cnn, resnet = at.success[0].value, at.success[1].value
        correct[0] += cnn.startswith(f"**Prediction:** {option} (")
        correct[1] += resnet.startswith(f"**Prediction:** {option} (")
    # Both models are ~97% accurate, so an occasional miss on a bundled sample is expected.
    assert min(correct) >= 7, correct


def test_blood_page_classifies_an_uploaded_image():
    at = run_page(BLOOD)
    at.radio[0].set_value(at.radio[0].options[1]).run()
    assert not at.success                                  # nothing until a file is uploaded
    sample = ROOT / "public" / "blood_samples" / "7_platelet.png"
    big = Image.open(sample).resize((300, 300), Image.NEAREST)  # an upload of a different size
    at.get("file_uploader")[0].upload("cell.png", png_bytes(big), "image/png").run()
    assert not at.exception
    assert at.success[0].value.startswith("**Prediction:** platelet (")


# ------------------------------------------------- unusual uploads, every page
def sixteen_bit_png() -> bytes:
    arr = (np.random.default_rng(0).random((64, 64)) * 60000).astype(np.uint16)
    return png_bytes(Image.fromarray(arr))


@pytest.mark.parametrize("page", ["pages/1_Deep_Learning.py", "pages/3_Blood_Cell_Classifier.py", SEG])
@pytest.mark.parametrize("kind", ["16-bit", "corrupt"])
def test_every_page_handles_16bit_and_corrupt_uploads(page, kind):
    at = run_page(page)
    if page != "pages/1_Deep_Learning.py":
        at.radio[0].set_value(at.radio[0].options[1]).run()
    content = sixteen_bit_png() if kind == "16-bit" else b"definitely not a png"
    at.get("file_uploader")[0].upload("upload.png", content, "image/png").run()
    assert not at.exception, at.exception
    if kind == "corrupt":
        assert any("Could not read" in e.value for e in at.error)
    else:
        assert not at.error and len(at.image) >= 1


def test_ground_truth_count_matches_table():
    at = run_page(SEG)
    at.number_input[0].set_value(5).run()                 # seed 5 has a cell under 20 px
    click(at, "Run Biomedical Image Segmentation Pipeline")
    n = at.dataframe[0].value["Cells (true)"].iloc[0]
    assert any(s.value == f"Ground Truth ({n} cells)" for s in at.subheader)


def test_blood_page_grad_cam_checkbox_adds_one_heatmap_per_model():
    at = run_page(BLOOD)
    before = len(at.image)
    at.checkbox[0].check().run()
    assert not at.exception
    assert len(at.image) == before + 2
