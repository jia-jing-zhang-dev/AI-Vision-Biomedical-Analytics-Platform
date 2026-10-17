import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.blood_cells import (
    REVIEW_CONFIDENCE, accepted_label, crop_cells, crop_side, differential_count, find_cells,
    is_smear, reframe_single_cell,
)
from src.explain import grad_cam, overlay
from src.image_io import UploadError, load_upload
from src.medical_cnn import CLASS_NAMES, load_model, predict_proba
from src.runtime import heavy_work, limit_torch_threads

limit_torch_threads()

ROOT = Path(__file__).resolve().parent.parent   # works from any working directory
SAMPLE_DIR = ROOT / "public" / "blood_samples"

# Display name -> weights file. The first model decides (smear counts, review
# flags); the custom CNN is shown alongside as the small 28x28 baseline.
MODELS = {
    "DinoBloom-S (224 px)": str(ROOT / "biomedical_dinobloom_s.pth"),
    "Custom CNN (28 px)": str(ROOT / "biomedical_cnn.pth"),
}
# Cross-source results (KU-Optofil PBC, another scanner and stain): eval_external.py
EXTERNAL = ROOT / "external_results.json"
EXTERNAL_NAMES = {"DinoBloom-S (224 px)": "DinoBloom-S probe", "Custom CNN (28 px)": "Custom CNN (28 px)"}

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("🩸 Blood Cell Classification (BloodMNIST)")
st.write(
    "Classifies blood-cell microscopy images into 8 cell types. The main model is "
    "**DinoBloom-S**, a vision transformer pretrained on 380,000 blood-cell images from 13 "
    "sources, with a linear classifier trained here on BloodMNIST (224×224). A compact CNN "
    "trained from scratch on 28×28 images is shown alongside for comparison."
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
main_name = next(iter(models))


def probability_chart(probs, height: int = 240):
    """Horizontal bars on a fixed 0-1 axis (st.bar_chart auto-scaled it, e.g. to -2..1)."""
    import altair as alt
    data = pd.DataFrame({"class": CLASS_NAMES, "probability": [float(p) for p in probs]})
    return alt.Chart(data, height=height).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("class:N", sort=None, title=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".1%")],
    )


def short(name: str) -> str:
    return name.split(" (")[0]


MAX_SMEAR_CELLS = 12
MIN_COUNTED = 10        # fewer confident cells than this: no percentages


def show_smear(image, cells, n_found: int):
    """A smear: number the stained cells, crop each like a BloodMNIST image
    and classify the crops (largest cells first)."""
    from PIL import ImageDraw

    with heavy_work():
        side = crop_side(cells)
        crops = crop_cells(image, cells, side)
        preds = [{name: predict_proba(model, crop) for name, model in models.items()} for crop in crops]
    # The main model decides; a cell counts only when it is confident enough
    labels = [accepted_label(p[main_name]) for p in preds]

    overview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overview)
    width = max(2, side // 40)
    for k, ((cx, cy, _), label) in enumerate(zip(cells, labels), 1):
        colour = (255, 215, 0) if label is not None else (255, 80, 80)   # red: needs review
        draw.rectangle([cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2],
                       outline=colour, width=width)
        at = (cx - side // 2 + 2 * width, cy - side // 2 + width)
        try:
            draw.text(at, str(k), fill=colour, font_size=max(16, side // 2))
        except TypeError:                      # Pillow < 10.1: default font only
            draw.text(at, str(k), fill=colour)
    more = f" (showing the {len(cells)} largest)" if n_found > len(cells) else ""
    st.info(f"This looks like a blood smear: **{n_found} stained cells** found{more}. "
            "Each one is cropped and classified on its own.")

    n_review = labels.count(None)
    n_counted = len(labels) - n_review
    st.subheader("Differential count")
    counts = [r for r in differential_count(labels, CLASS_NAMES) if r[1] > 0]
    if counts:
        table = [{"Cell type": name, "Count": n} for name, n, _ in counts]
        if n_counted >= MIN_COUNTED:
            for row, (_, _, share) in zip(table, counts):
                row["Share"] = f"{share:.0%}"
        st.dataframe(pd.DataFrame(table), hide_index=True)
    if n_counted < MIN_COUNTED:
        st.caption(f"Only {n_counted} confident cells: too few for percentages "
                   f"(shown from {MIN_COUNTED}; a real differential counts 100+ leukocytes).")
    st.caption(
        f"{n_counted} of {len(labels)} cells counted; **{n_review} need review** "
        f"({short(main_name)} below {REVIEW_CONFIDENCE:.0%} confidence; red boxes). On another "
        "scanner and stain (KU-Optofil PBC test patients) this rule counted 93% of cells at 97% "
        "accuracy."
    )
    st.image(overview, caption="Stained cells found (numbered; red = needs review)", width="stretch")
    for row in range(0, len(crops), 4):
        for col, k in zip(st.columns(4), range(row, min(row + 4, len(crops)))):
            col.image(crops[k], caption=f"Cell {k + 1}", width=120)
            verdict = (f"✅ **{CLASS_NAMES[labels[k]]}**" if labels[k] is not None
                       else "⚠️ **Needs review**")
            col.caption(verdict)
            for name, probs in preds[k].items():
                col.caption(f"{short(name)}: {CLASS_NAMES[int(probs.argmax())]} ({probs.max():.0%})")
    st.caption(
        "Cells are found by their purple stain (blue minus green), so platelets next to "
        "larger cells and unstained red cells are skipped. On smear-like mosaics of BloodMNIST "
        "test cells, 98% of leukocytes/erythroblasts were found and DinoBloom-S classified "
        "99% of the crops correctly (the 28×28 CNN: 86%). Grad-CAM is shown for single-cell "
        "images only."
    )


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def test_accuracy(weights: str):
    results = read_json(Path(str(Path(weights).with_suffix("")) + "_results.json"))
    return results["test_accuracy"] if results else None


def external_accuracy(name: str):
    results = read_json(EXTERNAL)
    try:
        return results["test_reframed"][EXTERNAL_NAMES[name]]["accuracy"]
    except (TypeError, KeyError):
        return None


st.caption("Test accuracy: BloodMNIST (3,421 held-out images) / another scanner and stain "
           "(KU-Optofil PBC, 4,949 cells from 55 patients, never used for training)")
cols = st.columns(len(models))
for col, name in zip(cols, models):
    acc, ext = test_accuracy(MODELS[name]), external_accuracy(name)
    if acc is not None:
        col.metric(name, f"{acc:.1%}" + (f" / {ext:.1%}" if ext is not None else ""))

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
        "Upload one centred blood cell, or a smear: several stained cells are split into one "
        "crop per cell. A zoomed-out single cell is re-framed to BloodMNIST's scale first. "
        "Photos from other microscopes or stains can still be misclassified."
    )
    if image is not None:
        cells = find_cells(np.asarray(image))
        if is_smear(cells):
            show_smear(image, cells[:MAX_SMEAR_CELLS], len(cells))
            image = None                       # handled; skip the single-cell view
        else:
            framed = reframe_single_cell(image)
            if framed is not image:
                st.caption("The cell was small in the frame, so the image was re-framed around it "
                           "to BloodMNIST's scale.")
            image = framed

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
            if name == main_name and accepted_label(probs) is None:
                st.warning(f"Below {REVIEW_CONFIDENCE:.0%} confidence: needs review by a person.")
            st.altair_chart(probability_chart(probs), width="stretch")
            if cam is not None:
                st.image(cam, caption=f"Grad-CAM for '{CLASS_NAMES[top]}'", width="stretch")

with st.expander("Model details and comparison with published results"):
    st.markdown(
        "**DinoBloom-S** — a DINOv2 ViT-S/14 pretrained on blood and bone-marrow cells "
        "(Koch et al., MICCAI 2024; BloodMNIST's source images were not in its training data). "
        "The backbone is frozen; a logistic-regression head is trained on BloodMNIST-224.\n\n"
        "**Custom CNN** — 5 convolution layers with batch normalization, 2 max-pooling "
        "stages, global average pooling, dropout + linear head (28×28 input). AdamW with a "
        "cosine schedule, flip/rotation augmentation, Optuna search on the validation split.\n\n"
        "Each model was evaluated once on each test set. The other scanner (KU-Optofil PBC) "
        "frames cells smaller; zoomed-out cells are re-framed before classification."
    )
    ext = read_json(EXTERNAL) or {}
    rows = []
    for label, key in (("Same source (BloodMNIST)", None), ("Other scanner (KU-Optofil)", "test_reframed")):
        if key is None:
            rows += [{"Test set": label, "Model": name, "Accuracy": f"{test_accuracy(path):.1%}"}
                     for name, path in MODELS.items() if test_accuracy(path) is not None]
        else:
            rows += [{"Test set": label, "Model": name, "Accuracy": f"{m['accuracy']:.1%}",
                      "Macro-F1": f"{m['macro_f1']:.2f}"} for name, m in ext.get(key, {}).items()]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Published BloodMNIST results (MedMNIST v2, Yang et al. 2023): ResNet-18 95.8–96.3%, "
               "Google AutoML Vision 96.6%. MedMNIST+ (Doerrich et al. 2025): ResNet-18 at 224 px 98.9%.")
    st.image(str(ROOT / "public" / "bloodmnist-confusion.png"), caption="Custom CNN: test-set confusion matrix")

st.caption(
    "Educational project, not a medical device. Data: MedMNIST v2 / MedMNIST+ BloodMNIST and "
    "KU-Optofil PBC (both CC BY 4.0); DinoBloom weights Apache-2.0."
)
