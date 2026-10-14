import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.explain import grad_cam, overlay
from src.image_io import UploadError, load_upload
from src.medical_cnn import CLASS_NAMES, load_model, predict_proba

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
            image, _ = load_upload(uploaded_file, mode="RGB")   # 8-bit, upright, at most 1024 px
        except UploadError as err:
            st.error(str(err))
    st.caption(
        "The models expect a single, centered blood cell. Images are reduced to 28×28 RGB, "
        "so photos from a different microscope or stain may be classified poorly."
    )

if image is not None:
    st.image(image, caption="Input image", width=200)
    show_cam = st.checkbox(
        "Show Grad-CAM: where each model looked",
        help="Grad-CAM highlights the image regions that pushed the model towards its "
             "prediction (red = most important).",
    )

    for col, (name, model) in zip(st.columns(len(models)), models.items()):
        with col:
            st.markdown(f"**{name}**")
            probs = predict_proba(model, image)
            top = int(probs.argmax())
            st.success(f"**Prediction:** {CLASS_NAMES[top]} ({probs[top]:.1%} confidence)")
            st.bar_chart(
                pd.DataFrame({"probability": probs}, index=CLASS_NAMES),
                horizontal=True,
                height=260,
            )
            if show_cam:
                heatmap, _ = grad_cam(model, image, target=top)
                st.image(overlay(image, heatmap), caption=f"Grad-CAM for '{CLASS_NAMES[top]}'",
                         width="stretch")

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
