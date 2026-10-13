import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.medical_cnn import CLASS_NAMES, load_model, predict_proba

SAMPLE_DIR = Path("public/blood_samples")

st.set_page_config(page_title="Vision Analytics Framework", layout="centered")

st.title("🩸 Blood Cell Classification (BloodMNIST)")
st.write(
    "A PyTorch convolutional neural network that classifies blood-cell microscopy "
    "images into 8 cell types, trained on the BloodMNIST dataset."
)


@st.cache_resource
def get_model():
    return load_model("biomedical_cnn.pth")


model = get_model()

results_path = Path("biomedical_cnn_results.json")
if results_path.exists():
    results = json.loads(results_path.read_text())
    col1, col2 = st.columns(2)
    col1.metric("Test accuracy (CNN)", f"{results['test_accuracy']:.1%}")
    col2.metric("Fully connected baseline", "72.9%")

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
        image = Image.open(uploaded_file)
    st.caption(
        "The model expects a single, centered blood cell. Images are resized to 28×28 RGB, "
        "so photos from a different microscope or stain may be classified poorly."
    )

if image is not None:
    st.image(image, caption="Input image", width=200)

    probs = predict_proba(model, image)
    top = int(probs.argmax())
    st.success(f"**Prediction:** {CLASS_NAMES[top]} ({probs[top]:.1%} confidence)")

    st.bar_chart(
        pd.DataFrame({"probability": probs}, index=CLASS_NAMES),
        horizontal=True,
    )

with st.expander("Model details"):
    st.markdown(
        "- **Architecture:** 5 convolution layers with batch normalization, "
        "2 max-pooling stages, global average pooling, dropout + linear head\n"
        "- **Training:** AdamW with a cosine learning-rate schedule, random flips and "
        "90° rotations as augmentation; the epoch with the best validation accuracy is kept\n"
        "- **Tuning:** Optuna search over learning rate, weight decay, dropout and width, "
        "scored on the validation split only\n"
        "- **Evaluation:** the held-out test split (3,421 images) is evaluated once"
    )
    st.image("public/bloodmnist-confusion.png", caption="Test-set confusion matrix")

st.caption(
    "Educational project, not a medical device. Data: MedMNIST v2 BloodMNIST (CC BY 4.0)."
)
