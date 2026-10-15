import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.blood_cells import crop_cells, crop_side, find_cells, is_smear
from src.explain import grad_cam, overlay
from src.image_io import UploadError, load_upload
from src.medical_cnn import CLASS_NAMES, load_model, predict_proba
from src.runtime import heavy_work, limit_torch_threads

limit_torch_threads()

ROOT = Path(__file__).resolve().parent.parent   # works from any working directory
SAMPLE_DIR = ROOT / "public" / "blood_samples"

# Display name -> weights file written by train_medical.py
MODELS = {
    "Custom CNN": str(ROOT / "biomedical_cnn.pth"),
    "ResNet-18 (pretrained)": str(ROOT / "biomedical_resnet18.pth"),
}

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("🩸 Blood Cell Classification (BloodMNIST)")
st.write(
    "Classifies blood-cell microscopy images into 8 cell types with two PyTorch models "
    "trained on BloodMNIST: a compact custom CNN, and torchvision's ImageNet-pretrained "
    "ResNet-18 fine-tuned as a reference."
)


@st.cache_resource
def get_models(name_to_path: tuple):
    # Takes the model list as an argument so the cache is keyed on it and
    # never returns models under outdated names.
    loaded, errors = {}, []
    for name, path in name_to_path:
        if Path(path).exists():
            try:
                loaded[name] = load_model(path)
            except ValueError as err:          # corrupt or foreign checkpoint
                errors.append(f"{name}: {err}")
    return loaded, errors


models, load_errors = get_models(tuple(MODELS.items()))
for message in load_errors:
    st.warning(message)
if not models:
    st.error("No trained model weights found. Run `python train_medical.py` first.")
    st.stop()


def probability_chart(probs, height: int = 240):
    """Horizontal bars on a fixed 0-1 axis (st.bar_chart auto-scaled it, e.g. to -2..1)."""
    import altair as alt
    data = pd.DataFrame({"class": CLASS_NAMES, "probability": [float(p) for p in probs]})
    return alt.Chart(data, height=height).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("class:N", sort=None, title=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")],
    )


MAX_SMEAR_CELLS = 12


def show_smear(image, cells, n_found: int):
    """A smear: number the stained cells, crop each like a BloodMNIST image
    and classify the crops (largest cells first)."""
    from PIL import ImageDraw

    with heavy_work():
        side = crop_side(cells)
        crops = crop_cells(image, cells, side)
        preds = [{name: predict_proba(model, crop) for name, model in models.items()} for crop in crops]

    overview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    width = max(2, side // 40)
    for k, (cx, cy, _) in enumerate(cells, 1):
        draw.rectangle([cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2],
                       outline=(255, 215, 0), width=width)
        at = (cx - side // 2 + 2 * width, cy - side // 2 + width)
        try:
            draw.text(at, str(k), fill=(255, 215, 0), font_size=max(16, side // 2))
        except TypeError:                      # Pillow < 10.1: default font only
            draw.text(at, str(k), fill=(255, 215, 0))
    more = f" (showing the {len(cells)} largest)" if n_found > len(cells) else ""
    st.info(f"This looks like a blood smear: **{n_found} stained cells** found{more}. "
            "Each one is cropped and classified on its own.")
    st.image(overview, caption="Stained cells found (numbered)", width="stretch")
    for row in range(0, len(crops), 4):
        for col, k in zip(st.columns(4), range(row, min(row + 4, len(crops)))):
            col.image(crops[k], caption=f"Cell {k + 1}", width=120)
            for name, probs in preds[k].items():
                short = "CNN" if name.startswith("Custom") else "ResNet"
                col.caption(f"{short}: **{CLASS_NAMES[int(probs.argmax())]}** "
                            f"({probs.max():.0%})")
    st.caption(
        "Cells are found by their purple stain (blue minus green), so platelets next to "
        "larger cells and unstained red cells are skipped. On smear-like test mosaics built "
        "from BloodMNIST, 99% of leukocytes/erythroblasts were found and the crops were "
        "classified with 86% (CNN) / 90% (ResNet-18) accuracy, below the 96–97% on the "
        "original single-cell crops. Grad-CAM is shown for single-cell images only."
    )


def test_accuracy(weights: str):
    path = Path(str(Path(weights).with_suffix("")) + "_results.json")   # as train_medical.output_paths
    return json.loads(path.read_text())["test_accuracy"] if path.exists() else None


st.caption("Test accuracy on the 3,421 held-out BloodMNIST test images")
cols = st.columns(len(models) + 1)
for col, name in zip(cols, models):
    acc = test_accuracy(MODELS[name])
    if acc is not None:
        col.metric(name, f"{acc:.1%}")
cols[-1].metric("Fully connected baseline", "72.9%")

source = st.radio("Image source", ["Pick a sample from the test set", "Upload your own image"],
                  horizontal=True)
# Drawn on every run (not only when an image exists) so the choice survives
# switching between samples and uploads.
show_cam = st.checkbox(
    "Show Grad-CAM: where each model looked",
    help="Grad-CAM highlights the image regions that pushed the model towards its "
         "prediction (red = most important).",
)

image = None
if source == "Pick a sample from the test set":
    samples = sorted(SAMPLE_DIR.glob("*.png"))
    choice = st.selectbox(
        "Sample image (true label shown)",
        samples,
        format_func=lambda p: p.stem.split("_", 1)[1].replace("_", " "),
    )
    image = Image.open(choice)
else:
    uploaded_file = st.file_uploader("Upload a blood-cell image (JPG/PNG)",
                                     type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            with heavy_work():
                image, _ = load_upload(uploaded_file, mode="RGB")   # 8-bit, upright, at most 1024 px
        except UploadError as err:
            st.error(str(err))
    st.caption(
        "The models expect a single, centered blood cell. A smear with several stained "
        "cells is split into one crop per cell. Images are reduced to 28×28 RGB, so photos "
        "from a different microscope or stain may be classified poorly."
    )
    if image is not None:
        cells = find_cells(np.asarray(image))
        if is_smear(cells):
            show_smear(image, cells[:MAX_SMEAR_CELLS], len(cells))
            image = None                       # handled; skip the single-cell view

if image is not None:
    st.image(image, caption="Input image", width=200)

    # Compute under the process-wide lock, render afterwards
    outputs = {}
    with heavy_work():
        for name, model in models.items():
            probs = predict_proba(model, image)
            top = int(probs.argmax())
            cam = overlay(image, grad_cam(model, image, target=top)[0]) if show_cam else None
            outputs[name] = (probs, top, cam)

    for col, (name, (probs, top, cam)) in zip(st.columns(len(outputs)), outputs.items()):
        with col:
            st.markdown(f"**{name}**")
            st.success(f"**Prediction:** {CLASS_NAMES[top]} ({probs[top]:.1%} confidence)")
            st.altair_chart(probability_chart(probs), width="stretch")
            if cam is not None:
                st.image(cam, caption=f"Grad-CAM for '{CLASS_NAMES[top]}'", width="stretch")

with st.expander("Model details and comparison with published results"):
    st.markdown(
        "**Custom CNN** — 5 convolution layers with batch normalization, 2 max-pooling "
        "stages, global average pooling, dropout + linear head (28×28 input). AdamW with a "
        "cosine schedule, flip/rotation augmentation, Optuna search on the validation split.\n\n"
        "**ResNet-18** — torchvision's ImageNet-pretrained ResNet-18 with a new 8-class head, "
        "fine-tuned end to end on images upsampled to 224×224 with the same augmentation "
        "and validation-based model selection.\n\n"
        "Both are evaluated once on the held-out test split (3,421 images)."
    )
    rows = [{"Model": f"{name} (this project)", "Test accuracy": f"{test_accuracy(path):.1%}"}
            for name, path in MODELS.items() if test_accuracy(path) is not None]
    rows += [
        {"Model": "ResNet-18 (28), MedMNIST v2 paper", "Test accuracy": "95.8%"},
        {"Model": "ResNet-18 (224), MedMNIST v2 paper", "Test accuracy": "96.3%"},
        {"Model": "Google AutoML Vision, MedMNIST v2 paper", "Test accuracy": "96.6%"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Published results: Yang et al., MedMNIST v2, Scientific Data 2023, Table 3.")
    cm_cnn, cm_resnet = st.columns(2)
    cm_cnn.image(str(ROOT / "public" / "bloodmnist-confusion.png"), caption="Custom CNN: test-set confusion matrix")
    cm_resnet.image(str(ROOT / "public" / "bloodmnist-confusion-resnet18.png"),
                    caption="ResNet-18: test-set confusion matrix")

st.caption(
    "Educational project, not a medical device. Data: MedMNIST v2 BloodMNIST (CC BY 4.0)."
)
