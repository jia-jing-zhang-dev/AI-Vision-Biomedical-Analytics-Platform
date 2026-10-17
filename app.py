import importlib.util
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent   # works from any working directory
# Cellpose is installed on the Hugging Face Space, not on Streamlit Cloud
CELLPOSE_ONLINE = importlib.util.find_spec("cellpose") is not None

st.set_page_config(
    page_title="Vision Analytics Framework",
    layout="wide"
)

st.title("🏥 AI Vision & Biomedical Analytics Platform")

st.write(
    "Welcome to the end-to-end computer vision and biomedical "
    "image analysis framework!"
)

st.caption(
    "The clothing classifier works on 28×28 images (FashionMNIST); the blood-cell model on "
    "224×224 images (BloodMNIST, MedMNIST+)."
)

st.info(
    "👈 Please use the **Navigation** panel on the left "
    "to switch between different modules."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🖼️ Module 1: PyTorch Deep Learning Classification")
    st.write("- A convolutional neural network (CNN) built with PyTorch.")
    st.write("- Real-time image uploads for 10-class classification: **92.8% test accuracy**, "
             "up from 86.9% for a fully connected baseline, and robust to mirrored or shifted items.")

    st.image(
        str(ROOT / "public" / "fashion-demo.png"),
        caption="FashionMNIST Classification Demo",
        width="stretch"
    )

with col2:
    st.subheader("🔬 Module 2: Biomedical Image Segmentation")
    st.write(
        "- Six classical computer vision methods, including Otsu thresholding, "
        "top-hat background correction and a watershed that splits touching cells."
    )
    st.write(
        "- Benchmarked with **Dice / IoU** and **instance F1**: the watershed raises "
        "instance F1 from 0.60 (Otsu) to 0.80 on images of touching cells; "
        "Cellpose, the standard deep-learning tool, scores 0.90 as a reference"
        + (" (also run on the segmentation page)." if CELLPOSE_ONLINE
           else " (benchmarked offline; not run in this online demo).")
    )
    st.write(
        "- On **real microscopy images** (BBBC039, expert annotations) a U-Net trained here "
        "reaches F1 0.95 at IoU 0.5 (watershed 0.90), on par with the pretrained Cellpose model (0.94)."
    )

    st.image(
        str(ROOT / "public" / "cell-segmentation-benchmark.png"),
        caption="Synthetic Cell Benchmark: Input, Ground Truth, Otsu, Watershed, Cellpose",
        width="stretch"
    )

    st.image(
        str(ROOT / "public" / "brightfield-demo.png"),
        caption="Real Blood Smear: Automatic Inversion + Watershed",
        width="stretch"
    )

with col3:
    st.subheader("🩸 Module 3: Blood Cell Classification")
    st.write("- Classifies 8 blood-cell types with **DinoBloom-S**, a hematology foundation "
             "model, plus a classifier head trained here (98.9% on BloodMNIST).")
    st.write(
        "- Tested on **another lab's scanner** it was never trained on: **95.8%**, where "
        "models scoring 96–99% on BloodMNIST fall to 65–76%; robust to colour changes."
    )
    st.write(
        "- Blood smears are split into single cells for a differential count; uncertain cells "
        "are flagged for review. Grad-CAM heatmaps show where each model looked."
    )

    st.image(
        str(ROOT / "public" / "bloodmnist-confusion.png"),
        caption="BloodMNIST Test-Set Confusion Matrix",
        width="stretch"
    )